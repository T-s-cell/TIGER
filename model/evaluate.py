"""Standalone evaluation and reload-verification entry for the TIGER rec model.

model/main.py's --mode argument was never wired to an independent evaluation
branch at the base commit; this script is the real entry point. It rebuilds the
model from the checkpoint's own config, verifies the checkpoint is bound to
the same codes file (sha256), evaluates valid/test with both batch-mean and
sample-weighted metrics, stores per-sample results, and (with --reload_check)
compares against the save-time reference outputs in <ckpt>_ref.npz: the fixed
inputs (history, attention mask, labels) must rebuild identically from the
current data, predictions must match exactly and logits max|diff| <= 1e-6.
"""
import argparse
import hashlib
import json
import os
from time import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dataset import GenRecDataset
from dataloader import GenRecDataLoader
from main import TIGER, evaluate


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_model_and_load(save_path, code_path, device):
    ckpt = torch.load(save_path, map_location='cpu')
    config = ckpt['config']
    ckpt_codes_sha = ckpt.get('codes_sha256')
    if ckpt_codes_sha is None:
        raise SystemExit("FATAL: checkpoint has no codes_sha256 (wrong format for this pipeline)")
    actual_codes_sha = sha256_file(code_path)
    if ckpt_codes_sha != actual_codes_sha:
        raise SystemExit(
            f"FATAL: codes mismatch: checkpoint bound to {ckpt_codes_sha} "
            f"but --code_path is {actual_codes_sha}")
    model = TIGER(config)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.to(device)
    model.eval()
    print(f"loaded {save_path} (epoch {ckpt.get('epoch')}, codes {ckpt_codes_sha[:12]}...)")
    return model, config, ckpt


def full_eval(model, config, dataset_path, code_path, split, batch_size, beam_size, device, out_dir):
    dataset = GenRecDataset(
        dataset_path=os.path.join(dataset_path, f'{split}.parquet'),
        code_path=code_path,
        mode='evaluation',
        max_len=config['max_len'],
    )
    loader = GenRecDataLoader(dataset, batch_size=batch_size, shuffle=False)
    avg_recalls, avg_ndcgs, sw_recalls, sw_ndcgs, wall, n, pos_index = evaluate(
        model, loader, config['topk_list'], beam_size, device, return_per_sample=True)
    print(f"[{split}] n={n} wall={wall:.1f}s")
    print(f"[{split}] batch-mean:      {json.dumps({**avg_recalls, **avg_ndcgs})}")
    print(f"[{split}] sample-weighted: {json.dumps({**sw_recalls, **sw_ndcgs})}")
    os.makedirs(out_dir, exist_ok=True)
    pos = pos_index.numpy().astype(bool)
    np.savez_compressed(os.path.join(out_dir, f'{split}_per_sample.npz'), pos_index=pos)
    # a few review samples: first 10 rows of the boolean hit/rank matrix
    pd.DataFrame(pos[:10]).to_parquet(os.path.join(out_dir, f'{split}_per_sample_head10.parquet'))
    result = {
        'split': split, 'n_samples': n, 'wall_sec': round(wall, 3),
        'batch_mean': {**avg_recalls, **avg_ndcgs},
        'sample_weighted': {**sw_recalls, **sw_ndcgs},
    }
    return result


def reload_check(model, config, save_path, code_path, dataset_path, device, ref_size, beam_size, tol=1e-6):
    ref_path = save_path.replace('.pth', '_ref.npz')
    if not os.path.exists(ref_path):
        raise SystemExit(f"FATAL: reference file not found: {ref_path}")
    ref = np.load(ref_path)
    # cross-check that the fixed inputs still rebuild identically from the
    # current dataset + codes (guards against wrong data being attached)
    dataset = GenRecDataset(
        dataset_path=os.path.join(dataset_path, 'valid.parquet'),
        code_path=code_path,
        mode='evaluation',
        max_len=config['max_len'],
    )
    loader = GenRecDataLoader(dataset, batch_size=ref_size, shuffle=False)
    rebuilt = loader.collate_fn([dataset[i] for i in range(ref_size)])
    input_ok = (rebuilt['history'].numpy() == ref['input_ids']).all()
    mask_ok = (rebuilt['attention_mask'].numpy() == ref['attention_mask']).all()
    labels_ok = (rebuilt['target'].numpy() == ref['labels']).all()
    fixed_ok = bool(input_ok and mask_ok and labels_ok)
    print(f"fixed inputs rebuilt from current data match reference: history={bool(input_ok)} "
          f"mask={bool(mask_ok)} labels={bool(labels_ok)}")

    input_ids = torch.from_numpy(ref['input_ids']).to(device)
    attention_mask = torch.from_numpy(ref['attention_mask']).to(device)
    labels = torch.from_numpy(ref['labels']).to(device)
    # the reference beam outputs were produced with a fixed beam width; derive
    # it so the comparison cannot be skewed by a mismatched --beam_size
    ref_beam = ref['preds'].shape[0] // ref['input_ids'].shape[0]
    if ref_beam != beam_size:
        print(f"note: reference was saved with beam_size={ref_beam} (arg {beam_size}); using {ref_beam}")
        beam_size = ref_beam
    with torch.no_grad():
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
    logits_diff = float(np.abs(logits.detach().cpu().numpy() - ref['logits']).max())
    preds_match = bool((preds.detach().cpu().numpy() == ref['preds']).all())
    ok = preds_match and logits_diff <= tol and fixed_ok
    result = {
        'logits_max_abs_diff': logits_diff,
        'tolerance': tol,
        'predictions_match': preds_match,
        'fixed_inputs_match_current_data': fixed_ok,
        'fixed_inputs_detail': {'history': bool(input_ok), 'attention_mask': bool(mask_ok), 'labels': bool(labels_ok)},
        'pass': bool(ok),
    }
    print(f"reload check: logits max|diff|={logits_diff:.3e} (tol {tol:.0e}), preds match={preds_match}, "
          f"fixed inputs match={fixed_ok} -> {'PASS' if ok else 'FAIL'}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Independent evaluation / reload check for TIGER")
    parser.add_argument('--save_path', type=str, required=True, help='checkpoint (.pth) from this run')
    parser.add_argument('--code_path', type=str, required=True, help='codes .npy from this run')
    parser.add_argument('--dataset_path', type=str, required=True, help='dir with train/valid/test.parquet')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--infer_size', type=int, default=96)
    parser.add_argument('--beam_size', type=int, default=30)
    parser.add_argument('--split', type=str, default='test', choices=['valid', 'test', 'both'])
    parser.add_argument('--out_dir', type=str, default='./eval_out')
    parser.add_argument('--reload_check', action='store_true',
                        help='compare against save-time reference outputs instead of full evaluation')
    parser.add_argument('--ref_size', type=int, default=32)
    args = parser.parse_args()

    requested = torch.device(args.device)
    if requested.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit(f"FATAL: --device {args.device} requested but torch.cuda.is_available() is False")
    device = requested

    os.makedirs(args.out_dir, exist_ok=True)
    model, config, ckpt = build_model_and_load(args.save_path, args.code_path, device)
    config['topk_list'] = [5, 10, 20]  # never via CLI (argparse type=list is broken upstream)

    results = {'save_path': args.save_path, 'epoch': ckpt.get('epoch'),
               'device': str(device), 'cuda_available': torch.cuda.is_available()}
    if args.reload_check:
        results['reload_check'] = reload_check(
            model, config, args.save_path, args.code_path, args.dataset_path,
            device, args.ref_size, args.beam_size)
        if not results['reload_check']['pass']:
            raise SystemExit(1)
    else:
        splits = ['valid', 'test'] if args.split == 'both' else [args.split]
        for split in splits:
            results[split] = full_eval(model, config, args.dataset_path, args.code_path,
                                       split, args.infer_size, args.beam_size, device, args.out_dir)

    with open(os.path.join(args.out_dir, 'eval_report.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
