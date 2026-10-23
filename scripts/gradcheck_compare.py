import argparse
import json

import numpy as np

ATOL, RTOL = 1e-6, 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    A = np.load(args.orig)
    B = np.load(args.fixed)

    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    maxabs = lambda v: float(np.abs(v).max())

    idx_eq = np.array_equal(A["indices"], B["indices"])
    add("forward_indices_exact", idx_eq, {"shape": list(A["indices"].shape)})

    out_diff = maxabs(A["out"] - B["out"])
    ma = json.loads(str(A["meta"]))
    mb = json.loads(str(B["meta"]))
    rq_diff = abs(ma["rq_loss"] - mb["rq_loss"])
    ll_diff = max(abs(a - b) for a, b in zip(ma["layer_losses"], mb["layer_losses"]))
    add("forward_out_tol", out_diff <= ATOL, {"max_abs_diff": out_diff})
    add(
        "forward_losses_tol",
        rq_diff <= ATOL and ll_diff <= ATOL,
        {"rq_loss_diff": rq_diff, "max_layer_loss_diff": ll_diff},
    )

    add(
        "orig_commit_l2_l3_zero_and_l1_nonzero",
        maxabs(A["g_l2"]) == 0.0 and maxabs(A["g_l3"]) == 0.0 and maxabs(A["g_l1"]) > 0.0,
        {k: maxabs(A[k]) for k in ("g_l1", "g_l2", "g_l3")},
    )
    add(
        "fixed_commit_all_nonzero",
        all(maxabs(B[k]) > 0.0 for k in ("g_l1", "g_l2", "g_l3")),
        {k: maxabs(B[k]) for k in ("g_l1", "g_l2", "g_l3")},
    )

    rdiff = maxabs(A["g_recon"] - B["g_recon"])
    ratio = float(np.linalg.norm(B["g_recon"]) / np.linalg.norm(A["g_recon"]))
    add(
        "recon_grad_identical_not_3x",
        bool(np.allclose(A["g_recon"], B["g_recon"], atol=ATOL, rtol=RTOL))
        and 0.99 <= ratio <= 1.01,
        {"max_abs_diff": rdiff, "norm_ratio_fixed_over_orig": ratio},
    )

    for tag, D in (("orig", A), ("fixed", B)):
        s = D["g_l1"] + D["g_l2"] + D["g_l3"]
        add(
            f"linearity_rq_sum_{tag}",
            bool(np.allclose(D["g_rq_all"], s, atol=ATOL, rtol=RTOL)),
            {"max_abs_diff": maxabs(D["g_rq_all"] - s)},
        )

    for tag, D in (("orig", A), ("fixed", B)):
        s = D["g_recon"] + (D["g_l1"] + D["g_l2"] + D["g_l3"]) / 3.0
        add(
            f"full_loss_decomp_{tag}",
            bool(np.allclose(D["g_full"], s, atol=ATOL, rtol=RTOL)),
            {"max_abs_diff": maxabs(D["g_full"] - s)},
        )

    ok_all = all(c["pass"] for c in checks)
    with open(args.out, "w") as f:
        json.dump({"pass": ok_all, "atol": ATOL, "rtol": RTOL, "checks": checks}, f, indent=1)
    for c in checks:
        print(("PASS" if c["pass"] else "FAIL"), c["check"], c["detail"])
    print("GRADCHECK", "PASS" if ok_all else "FAIL")
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
