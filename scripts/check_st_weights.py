"""Value-level verification that SentenceTransformer loaded pretrained weights.

The encode step logs "T5EncoderModel ... newly initialized: ['shared.weight']".
Key-name-based warnings cannot rule out a silent random-embedding failure, so
this script loads the SentenceTransformer, then compares every loaded parameter
value-by-value against the raw checkpoint files on disk (model.safetensors and
pytorch_model.bin), plus the 2_Dense module against its own pytorch_model.bin.
PASS requires every loaded tensor to match some reference tensor exactly
(max|diff| == 0) and no loaded key to be left unmatched.
"""
import argparse
import hashlib
import json
import os
from time import time

import torch


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compare_one(loaded_k, loaded_t, refs):
    best = None
    for fname, sd in refs.items():
        if loaded_k in sd:
            ref_t = sd[loaded_k]
            if ref_t.shape != loaded_t.shape:
                return {'ref_key': loaded_k, 'file': fname, 'exact': False,
                        'max_abs_diff': None, 'shape_mismatch': [list(loaded_t.shape), list(ref_t.shape)]}
            diff = float((ref_t.to(loaded_t.dtype) - loaded_t).abs().max())
            cand = {'ref_key': loaded_k, 'file': fname, 'exact': bool(torch.equal(ref_t.to(loaded_t.dtype), loaded_t)),
                    'max_abs_diff': diff, 'shape': list(loaded_t.shape)}
            if best is None or (cand['exact'] and not best['exact']):
                best = cand
    return best if best is not None else {'ref_key': None, 'exact': False, 'max_abs_diff': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', required=True)
    ap.add_argument('--out_file', required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--revision', default='unknown')
    args = ap.parse_args()

    t0 = time()
    import transformers
    import sentence_transformers
    from safetensors.torch import load_file
    from sentence_transformers import SentenceTransformer

    refs = {}
    for fname, loader in [('model.safetensors', load_file), ('pytorch_model.bin', lambda p: torch.load(p, map_location='cpu'))]:
        path = os.path.join(args.model_dir, fname)
        if os.path.exists(path):
            refs[fname] = {k: v for k, v in loader(path).items()}
    dense_refs = {}
    for fname, loader in [('2_Dense/model.safetensors', load_file), ('2_Dense/pytorch_model.bin', lambda p: torch.load(p, map_location='cpu'))]:
        path = os.path.join(args.model_dir, fname)
        if os.path.exists(path):
            dense_refs[fname] = {k: v for k, v in loader(path).items()}

    import sentence_transformers.models as st_models
    st = SentenceTransformer(args.model_dir, device=args.device)
    auto = st._first_module().auto_model
    enc_sd = {k: v.detach().cpu() for k, v in auto.state_dict().items()}
    dense_sd = {}
    for mkey, mod in st._modules.items():
        if isinstance(mod, st_models.Dense):
            dense_sd.update({k: v.detach().cpu() for k, v in mod.state_dict().items()})

    emb_checks = {}
    for k in ('shared.weight', 'encoder.embed_tokens.weight'):
        if k in enc_sd:
            emb_checks[k] = compare_one(k, enc_sd[k], refs)
    tie_ok = bool(torch.equal(enc_sd['shared.weight'], enc_sd['encoder.embed_tokens.weight'])) \
        if 'shared.weight' in enc_sd and 'encoder.embed_tokens.weight' in enc_sd else None

    enc_results = {k: compare_one(k, v, refs) for k, v in enc_sd.items()}
    dense_results = {k: compare_one(k, v, dense_refs) for k, v in dense_sd.items()}

    enc_ok = all(r['exact'] for r in enc_results.values()) and len(enc_results) > 0
    dense_ok = all(r['exact'] for r in dense_results.values()) and len(dense_results) > 0
    emb_ok = all(r.get('exact') for r in emb_checks.values()) and len(emb_checks) > 0
    passed = enc_ok and dense_ok and emb_ok and (tie_ok is not False)

    report = {
        'model_dir': args.model_dir,
        'revision': args.revision,
        'device': args.device,
        'versions': {'transformers': transformers.__version__,
                     'sentence_transformers': sentence_transformers.__version__,
                     'torch': torch.__version__},
        'files_sha256': {os.path.relpath(os.path.join(d, f), args.model_dir): sha256_file(os.path.join(d, f))
                         for d, _, fs in os.walk(args.model_dir) for f in fs if f.endswith(('.safetensors', '.bin'))},
        'n_encoder_params_compared': len(enc_results),
        'n_dense_params_compared': len(dense_results),
        'embedding_checks': emb_checks,
        'shared_vs_embed_tokens_tied_exact': tie_ok,
        'encoder_all_exact': enc_ok,
        'dense_all_exact': dense_ok,
        'encoder_non_exact': {k: r for k, r in enc_results.items() if not r['exact']},
        'dense_non_exact': {k: r for k, r in dense_results.items() if not r['exact']},
        'ref_keys_not_matched_by_loaded': sorted(set().union(*[set(sd) for sd in refs.values()])
                                                 - set(enc_results) - {r['ref_key'] for r in enc_results.values() if r['ref_key']}),
        'pass': bool(passed),
        'wall_sec': round(time() - t0, 3),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    with open(args.out_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"st weight check: encoder {len(enc_results)} params all_exact={enc_ok}, "
          f"dense {len(dense_results)} all_exact={dense_ok}, embedding exact={emb_ok}, tied={tie_ok} -> "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"report: {args.out_file}")
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
