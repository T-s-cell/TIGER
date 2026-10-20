"""Local (CPU) module-level checks for original_smoke_beauty.

Scope is deliberately narrow — syntax, imports, argparse wiring, and module-level
forward/backward passes. No end-to-end run happens locally; real end-to-end
verification runs on iota. RQ-VAE forward uses >=256 synthetic samples because
the original k-means codebook init (sklearn KMeans, n_clusters=256) is invoked
on the first training batch and would fail with fewer samples.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = "/home/wlt/Rec/local_check_out"

sys.path.insert(0, os.path.join(REPO, "rqvae"))
sys.path.insert(0, os.path.join(REPO, "model"))

passed = []
failed = []


def check(name, fn):
    try:
        fn()
        passed.append(name)
        print(f"PASS {name}")
    except Exception as e:  # noqa: BLE001
        failed.append((name, repr(e)))
        print(f"FAIL {name}: {e!r}")


def check_argparse():
    scripts = [
        "data/preprocess.py",
        "data/encode_text.py",
        "rqvae/generate_code.py",
        "rqvae/main.py",
        "model/evaluate.py",
    ]
    for rel in scripts:
        r = subprocess.run([sys.executable, os.path.join(REPO, rel), "--help"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{rel} --help failed:\n{r.stderr[-500:]}")


def check_rqvae_forward_backward():
    import torch
    import numpy as np
    from models.rqvae import RQVAE

    torch.manual_seed(0)
    n, in_dim = 512, 64  # >= 256 samples so kmeans init (256 clusters) works
    x = torch.randn(n, in_dim)
    model = RQVAE(in_dim=in_dim, num_emb_list=[256, 256, 256], e_dim=8,
                  layers=[32, 16], dropout_prob=0.0, bn=False, loss_type="mse",
                  quant_loss_weight=1.0, beta=0.25, kmeans_init=True,
                  kmeans_iters=10, sk_epsilons=[0.0, 0.0, 0.003], sk_iters=20)
    model.train()
    out, rq_loss, indices = model(x, use_sk=True)
    assert out.shape == x.shape
    assert indices.shape == (n, 3)
    loss, recon = model.compute_loss(out, rq_loss, xs=x)
    assert torch.isfinite(loss), "loss not finite"
    loss.backward()
    enc_grad = model.encoder.mlp_layers[1].weight.grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    for i, vq in enumerate(model.rq.vq_layers):
        assert vq.embedding.weight.grad is not None, f"codebook {i} has no grad"
    # eval-path indices
    model.eval()
    with torch.no_grad():
        idx = model.get_indices(x, use_sk=False)
    assert idx.shape == (n, 3)


def check_tiger_forward_backward():
    import torch
    from main import TIGER

    config = dict(num_layers=2, num_decoder_layers=2, d_model=32, d_ff=64,
                  num_heads=2, d_kv=16, dropout_rate=0.1, vocab_size=1025,
                  pad_token_id=0, eos_token_id=0, feed_forward_proj="relu")
    torch.manual_seed(0)
    model = TIGER(config)
    model.train()
    B, L = 4, 80
    input_ids = torch.randint(1, 1025, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.int64)
    labels = torch.randint(1, 1025, (B, 4))
    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    assert torch.isfinite(loss), "loss not finite"
    assert logits.shape[-1] == 1025
    loss.backward()
    w = model.model.get_input_embeddings().weight.grad
    assert w is not None and torch.isfinite(w).all()


def check_dataset_offsets():
    import numpy as np
    from dataset import item2code, pad_or_truncate

    codes = np.array([[0, 0, 0, 0], [255, 255, 255, 255]])
    i2c, c2i = item2code_from(codes)
    assert i2c[1] == [1, 257, 513, 769], i2c[1]
    assert i2c[2] == [256, 512, 768, 1024], i2c[2]
    assert c2i[(1, 257, 513, 769)] == 1
    assert pad_or_truncate([1, 2], 5) == [0, 0, 0, 1, 2]
    assert pad_or_truncate(list(range(30)), 20) == list(range(10, 30))


def item2code_from(data, codebook_size=256):
    item_to_code, code_to_item = {}, {}
    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i, c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1
    return item_to_code, code_to_item


def check_collate():
    import numpy as np
    import torch
    from dataloader import GenRecDataLoader

    # in the real pipeline process_data() already left-pads every history to
    # max_len items; the pad "item" is the all-zero 4-token code
    pad_item = np.array([0, 0, 0, 0])
    items = [
        {"history": [np.array([1, 257, 513, 769]), np.array([2, 258, 514, 770])],
         "target": np.array([3, 259, 515, 771])},
        {"history": [pad_item, np.array([4, 260, 516, 772])],
         "target": np.array([5, 261, 517, 773])},
    ]
    batch = GenRecDataLoader(items, batch_size=2).collate_fn(items)
    assert batch["history"].shape == (2, 8)
    assert batch["attention_mask"][0].tolist() == [1] * 8
    assert batch["attention_mask"][1].tolist() == [0] * 4 + [1] * 4


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    check("argparse_wiring", check_argparse)
    check("rqvae_forward_backward_kmeans_init", check_rqvae_forward_backward)
    check("tiger_forward_backward", check_tiger_forward_backward)
    check("item2code_offsets_and_padding", check_dataset_offsets)
    check("collate_masks", check_collate)
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    if failed:
        sys.exit(1)
