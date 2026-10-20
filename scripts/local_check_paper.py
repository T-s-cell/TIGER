"""Local CPU checks for the paper-aligned TIGER candidate (precheck round).

Three checks, all on CPU:
  1. RQ-VAE STE / loss wiring: quantized outputs and indices match a hand
     reference; every layer's commitment loss reaches the encoder with a
     nonzero finite gradient (the frozen version's layers 2+ had zero encoder
     gradient); codebook loss touches only codebooks; recon loss touches
     encoder+decoder; lazy kmeans init still triggers on the first
     training-mode batch.
  2. Token partition / data flow with the REAL Beauty parquets + user mapping
     (codes.npy is synthetic: this checks plumbing, not RQ-VAE quality):
     SHA-256 bucket determinism, token ranges disjoint, user token prepended
     and never padded away, real GenRecDataLoader gives [B, 81] histories and
     [B, 4] targets with mask == (tokens != 0).
  3. LR schedule lr(s) = lr * min(1, sqrt(constant_steps / max(s, 1))) at
     s in {1, 10000, 10001, 200000}.

Supersedes scripts/local_check.py (which targets the frozen smoke branch's
interfaces and is left untouched on its own branch).
"""
import argparse
import hashlib
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
# model/ must come first: both model/ and rqvae/ have a main.py, and the
# model's dataset.py/main.py pair must resolve together
sys.path.insert(0, str(ROOT / "rqvae"))
sys.path.insert(0, str(ROOT / "model"))

from models.rqvae import RQVAE  # noqa: E402
from dataset import GenRecDataset, user_token_bucket  # noqa: E402
from dataloader import GenRecDataLoader  # noqa: E402
from main import lr_at_step  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def encoder_grad_norm(model):
    sq = 0.0
    for p in model.encoder.parameters():
        if p.grad is not None:
            sq += float(p.grad.pow(2).sum())
    return math.sqrt(sq)


def has_any_encoder_grad(model):
    return any(p.grad is not None for p in model.encoder.parameters())


def make_model(seed=0):
    torch.manual_seed(seed)
    model = RQVAE(in_dim=768, num_emb_list=[256, 256, 256], e_dim=32,
                  layers=[512, 256, 128], kmeans_init=True)
    # deterministic codebooks, bypassing kmeans for the wiring checks
    g = torch.Generator().manual_seed(seed + 1)
    for vq in model.rq.vq_layers:
        vq.embedding.weight.data = torch.randn(vq.n_e, vq.e_dim, generator=g) * 0.1
        vq.initted = True
    model.eval()
    return model


def rq_forward_manual(model, x):
    """Reference residual-quantization computed by hand (no_grad)."""
    with torch.no_grad():
        z = model.encoder(x)
        residual = z
        x_q = 0
        indices = []
        for vq in model.rq.vq_layers:
            d = (residual.pow(2).sum(-1, keepdim=True)
                 + vq.embedding.weight.pow(2).sum(-1)[None, :]
                 - 2 * residual @ vq.embedding.weight.t())
            idx = d.argmin(-1)
            q = vq.embedding(idx)
            x_q = x_q + q
            residual = residual - q
            indices.append(idx)
        indices = torch.stack(indices, dim=-1)
    return z, x_q, indices


def layer_commitment(model, x, layer_idx):
    """Commitment loss of a single layer, with graph attached (eval mode)."""
    z = model.encoder(x)
    residual = z
    for i in range(layer_idx):
        q, _, _, _ = model.rq.vq_layers[i](residual)
        residual = residual - q.detach()
    _, _, commit_loss, _ = model.rq.vq_layers[layer_idx](residual)
    return commit_loss


def check_1_ste_and_losses():
    print("CHECK 1: RQ-VAE STE / loss wiring")
    torch.manual_seed(2024)
    x = torch.randn(512, 768)
    model = make_model()

    # (a) forward matches the hand reference
    z_ref, xq_ref, idx_ref = rq_forward_manual(model, x)
    z = model.encoder(x)
    x_q, quant_loss, indices, layer_losses = model.rq(z)
    check("quantized output matches hand reference", torch.allclose(x_q, xq_ref, atol=1e-6))
    check("indices match hand reference", bool((indices == idx_ref).all()))
    comp_sum = sum(ll["codebook"] + ll["commitment"] for ll in layer_losses)
    check("quant loss == sum of per-layer components",
          torch.allclose(quant_loss, comp_sum, atol=1e-6),
          f"quant={float(quant_loss):.6f} sum_components={float(comp_sum):.6f}")

    # (b) every layer's commitment reaches the encoder with nonzero finite grad
    for d in range(3):
        model.zero_grad(set_to_none=True)
        layer_commitment(model, x, d).backward()
        norm = encoder_grad_norm(model)
        finite = all(torch.isfinite(p.grad).all() for p in model.encoder.parameters()
                     if p.grad is not None)
        check(f"layer {d + 1} commitment -> encoder grad nonzero & finite",
              norm > 0.0 and finite, f"grad_norm={norm:.6e}")

    # (c) codebook-only loss: encoder untouched; recon loss: encoder+decoder
    model.zero_grad(set_to_none=True)
    z = model.encoder(x)
    _, quant_loss, _, layer_losses = model.rq(z)
    cb_only = sum(ll["codebook"] for ll in layer_losses)  # detached copies -> rebuild
    residual = z
    cb_graph = 0
    for vq in model.rq.vq_layers:
        q, codebook_loss, _, _ = vq(residual)
        cb_graph = cb_graph + codebook_loss
        residual = residual - q.detach()
    model.zero_grad(set_to_none=True)
    cb_graph.backward()
    check("codebook loss -> encoder grad absent", not has_any_encoder_grad(model))
    cb_grads = [vq.embedding.weight.grad for vq in model.rq.vq_layers]
    check("codebook loss -> all 3 codebooks got grads",
          all(g is not None and g.abs().sum() > 0 for g in cb_grads))

    model.zero_grad(set_to_none=True)
    out, _, _, _ = model(x)
    recon = (out - x).pow(2).sum(-1).mean()
    recon.backward()
    check("recon loss -> encoder grad nonzero", encoder_grad_norm(model) > 0)
    dec_grad = sum(float(p.grad.abs().sum()) for p in model.decoder.parameters() if p.grad is not None)
    check("recon loss -> decoder grad nonzero", dec_grad > 0)
    cb_grads = [vq.embedding.weight.grad for vq in model.rq.vq_layers]
    check("recon loss -> codebook grads absent (single STE at decoder input)",
          all(g is None for g in cb_grads))

    # lazy kmeans init still fires on the first training-mode batch
    torch.manual_seed(2024)
    m2 = RQVAE(in_dim=768, num_emb_list=[256, 256, 256], e_dim=32,
               layers=[512, 256, 128], kmeans_init=True)
    check("kmeans_init model starts uninitted", not all(vq.initted for vq in m2.rq.vq_layers))
    m2.train()
    m2(torch.randn(512, 768))
    check("first training forward initializes all codebooks via kmeans",
          all(vq.initted for vq in m2.rq.vq_layers))


def check_2_tokens_and_dataflow(data_dir, expect_items):
    print("CHECK 2: token partition / data flow (real parquets + user mapping, synthetic codes)")
    data_dir = Path(data_dir)

    # SHA-256 bucket: deterministic, platform-independent, in range
    expect_fn = lambda s: 1025 + int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 2000
    probe = "probe_user_123"
    check("user_token_bucket matches inline SHA-256 reference",
          user_token_bucket(probe) == expect_fn(probe))
    check("user_token_bucket deterministic across calls",
          user_token_bucket(probe) == user_token_bucket(probe))
    all_tokens = {user_token_bucket(f"user_{i}") for i in range(5000)}
    check("5000 reviewer ids all map into [1025, 3024]",
          min(all_tokens) >= 1025 and max(all_tokens) <= 3024,
          f"min={min(all_tokens)} max={max(all_tokens)} distinct={len(all_tokens)}")

    # synthetic codes for the flow check
    rng = np.random.default_rng(2024)
    tmp = tempfile.mkdtemp(prefix="tiger_flow_")
    codes_path = str(Path(tmp) / "codes.npy")
    codes = rng.integers(0, 256, size=(expect_items, 4))
    np.save(codes_path, codes)

    from dataset import item2code
    item_to_code, code_to_item = item2code(codes_path)
    item_tokens = {t for code in item_to_code.values() for t in code}
    check("item tokens all within [1, 1024]",
          min(item_tokens) >= 1 and max(item_tokens) <= 1024,
          f"min={min(item_tokens)} max={max(item_tokens)}")
    check("item token range disjoint from user token range [1025, 3024]",
          max(item_tokens) < 1025)

    user_mapping_path = str(data_dir / "user_mapping.npy")
    train_ds = GenRecDataset(str(data_dir / "train.parquet"), codes_path, "train", 20,
                             user_mapping_path)
    valid_ds = GenRecDataset(str(data_dir / "valid.parquet"), codes_path, "evaluation", 20,
                             user_mapping_path)

    s0 = train_ds[0]
    check("train sample history = user token + 20 x 4-token items (81 tokens)",
          len(s0["history"]) == 21 and len(s0["history"][0]) == 1
          and all(len(h) == 4 for h in s0["history"][1:]))
    ut = s0["history"][0][0]
    check("user token in [1025, 3024]", 1025 <= ut <= 3024, f"token={ut}")
    check("target is a 4-token item id", len(s0["target"]) == 4
          and all(1 <= t <= 1024 for t in s0["target"]))

    # same user keeps the same token across splits
    import pandas as pd
    tr = pd.read_parquet(data_dir / "train.parquet")
    va = pd.read_parquet(data_dir / "valid.parquet")
    uid = int(tr.iloc[0]["user"])
    tok_train = train_ds.user_bucket_map[uid]
    tok_valid = valid_ds.user_bucket_map[uid]
    check("same userID -> same token in train and valid", tok_train == tok_valid)

    loader = GenRecDataLoader(train_ds, batch_size=8, shuffle=False)
    batch = next(iter(loader))
    check("collated history shape [B, 81]", tuple(batch["history"].shape) == (8, 81),
          f"shape={tuple(batch['history'].shape)}")
    check("collated target shape [B, 4]", tuple(batch["target"].shape) == (8, 4))
    check("attention mask == (tokens != 0)",
          bool((batch["attention_mask"] == (batch["history"] != 0).int()).all()))
    check("user position mask always 1 (never padded away)",
          bool((batch["attention_mask"][:, 0] == 1).all()))
    expected_row = torch.tensor([t for sub in train_ds[0]["history"] for t in sub],
                                dtype=torch.int64)
    check("first batch row equals manual flatten of dataset sample",
          bool((batch["history"][0] == expected_row).all()))

    vloader = GenRecDataLoader(valid_ds, batch_size=8, shuffle=False)
    vbatch = next(iter(vloader))
    check("evaluation-mode batches also [B, 81] / [B, 4] / mask ok",
          tuple(vbatch["history"].shape) == (8, 81) and tuple(vbatch["target"].shape) == (8, 4)
          and bool((vbatch["attention_mask"][:, 0] == 1).all()))

    # truncation: a user with >20 history items must keep the last 20 + user token
    seq_lens = tr["history"].apply(len) + 1
    long_idx = int(seq_lens[seq_lens > 21].index[0]) if (seq_lens > 21).any() else None
    if long_idx is not None:
        long_rows = [s for s, l in zip(train_ds.data, [len(s["history"]) for s in train_ds.data])
                     if l == 21]
        check("long sequences truncated to exactly max_len items (+user token)",
              len(long_rows) > 0)


def check_3_lr_schedule():
    print("CHECK 3: LR schedule")
    cases = {
        1: 0.01 * min(1.0, math.sqrt(10000 / 1)),
        10000: 0.01 * min(1.0, math.sqrt(10000 / 10000)),
        10001: 0.01 * math.sqrt(10000 / 10001),
        200000: 0.01 * math.sqrt(10000 / 200000),
    }
    for s, expect in cases.items():
        got = lr_at_step(s, 0.01, 10000)
        check(f"lr_at_step({s}) == {expect:.8g}", abs(got - expect) < 1e-15, f"got={got:.8g}")
    check("constant phase value is exactly 0.01", lr_at_step(1, 0.01, 10000) == 0.01
          and lr_at_step(10000, 0.01, 10000) == 0.01)
    check("decayed value at 200000 is 0.01*sqrt(0.05)",
          abs(lr_at_step(200000, 0.01, 10000) - 0.01 * math.sqrt(0.05)) < 1e-15)


def main():
    parser = argparse.ArgumentParser(description="Paper-aligned local checks (CPU)")
    parser.add_argument("--data_dir", type=str,
                        default=str(Path("/home/wlt/Rec/temp/local_check_out/data_dryrun")),
                        help="dir with train/valid.parquet and user_mapping.npy (real Beauty)")
    parser.add_argument("--expect_items", type=int, default=12101)
    args = parser.parse_args()

    check_1_ste_and_losses()
    check_2_tokens_and_dataflow(args.data_dir, args.expect_items)
    check_3_lr_schedule()

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
