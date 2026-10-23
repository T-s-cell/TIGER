import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def load_state_dict(ckpt_path):
    obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in obj and hasattr(obj[key], "values"):
                return obj[key]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", required=True, help="path whose rqvae package is imported")
    ap.add_argument("--ckpt", required=True, help="baseline best_collision_model.pth")
    ap.add_argument("--data", required=True, help="item_emb.parquet")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    sys.path.insert(0, args.worktree)
    from rqvae.models.rqvae import RQVAE

    torch.manual_seed(2024)

    model = RQVAE(
        in_dim=768,
        num_emb_list=[256, 256, 256],
        e_dim=32,
        layers=[512, 256, 128, 64],
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=True,
        kmeans_iters=100,
        sk_epsilons=[0.0, 0.0, 0.003],
        sk_iters=50,
    )
    model.load_state_dict(load_state_dict(args.ckpt), strict=True)
    model.initted = True
    model.eval()

    df = pd.read_parquet(args.data)
    x = torch.from_numpy(
        np.vstack(df["embedding"].to_numpy()).astype(np.float32)[: args.batch]
    )

    layer_losses = []

    def hook(_m, _i, output):
        layer_losses.append(output[1])

    handles = [q.register_forward_hook(hook) for q in model.rq.vq_layers]

    out, rq_loss, indices = model(x, use_sk=True)
    loss_full, loss_recon = model.compute_loss(out, rq_loss, xs=x)

    enc = [p for _, p in model.encoder.named_parameters()]

    def g(loss, retain=True):
        grads = torch.autograd.grad(loss, enc, retain_graph=retain)
        return torch.cat([t.reshape(-1) for t in grads]).to(torch.float64).numpy()

    g_l = [g(l) for l in layer_losses]
    g_rq_all = g(torch.stack(layer_losses).sum())
    g_recon = g(F.mse_loss(out, x))
    g_full = g(loss_full, retain=False)

    for h in handles:
        h.remove()

    np.savez_compressed(
        args.out,
        indices=indices.numpy(),
        out=out.detach().numpy().astype(np.float64),
        x=x.numpy(),
        g_l1=g_l[0],
        g_l2=g_l[1],
        g_l3=g_l[2],
        g_rq_all=g_rq_all,
        g_recon=g_recon,
        g_full=g_full,
        meta=np.array(
            json.dumps(
                {
                    "rq_loss": float(rq_loss.detach()),
                    "layer_losses": [float(l.detach()) for l in layer_losses],
                    "loss_recon": float(loss_recon.detach()),
                    "loss_full": float(loss_full.detach()),
                    "enc_numel": int(sum(p.numel() for p in enc)),
                }
            )
        ),
    )
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
