"""Generate 4-digit semantic IDs with a trained RQ-VAE checkpoint.

Paper-aligned pipeline (§4.1): pure nearest-neighbor assignment (no Sinkhorn
reallocation), then a 4th disambiguation digit — unique triplets get 0,
collision groups are numbered 0..n-1 in ItemID ascending order. A collision
group larger than the codebook size (256) is a hard error.
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE


def parse_args():
    parser = argparse.ArgumentParser(description="Generate semantic IDs from an RQ-VAE checkpoint")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="best_collision_model.pth from this run")
    parser.add_argument("--data_path", type=str, required=True,
                        help="item_emb.parquet for THIS run (overrides ckpt-internal path)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="output .npy path for the codes array")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ckpt_path = args.ckpt_path
    output_file = args.output_file
    device = torch.device(args.device)

    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    # Explicitly use this run's embedding table, not the one frozen in the ckpt
    data = EmbDataset(args.data_path)

    model = RQVAE(in_dim=data.dim,
                  num_emb_list=ckpt_args.num_emb_list,
                  e_dim=ckpt_args.e_dim,
                  layers=ckpt_args.layers,
                  dropout_prob=ckpt_args.dropout_prob,
                  bn=ckpt_args.bn,
                  loss_type=ckpt_args.loss_type,
                  quant_loss_weight=ckpt_args.quant_loss_weight,
                  kmeans_init=ckpt_args.kmeans_init,
                  kmeans_iters=ckpt_args.kmeans_iters,
                  )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(model)

    data_loader = DataLoader(data, num_workers=ckpt_args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    for d in tqdm(data_loader):
        d = d.to(device)
        indices = model.get_indices(d)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = []
            for i, ind in enumerate(index):
                code.append(prefix[i].format(int(ind)))

            all_indices.append(code)
            all_indices_str.append(str(code))

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    print("All indices number: ", len(all_indices))

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    print("Collision Rate (before 4th digit)", (tot_item - tot_indice) / tot_item)

    all_indices_dict = {}
    for item, indices in enumerate(all_indices.tolist()):
        all_indices_dict[item] = list(indices)

    codes = []
    for key, value in all_indices_dict.items():
        code = [int(item.split('_')[1].strip('>')) for item in value]
        codes.append(code)

    codes_array = np.array(codes)

    # Add an extra dimension to all codes
    codes_array = np.hstack((codes_array, np.zeros((codes_array.shape[0], 1), dtype=int)))

    # Resolve duplicates by incrementing the last dimension
    unique_codes, counts = np.unique(codes_array, axis=0, return_counts=True)
    max_group = int(counts.max())
    if max_group > 256:
        print(f"FATAL: largest collision group has {max_group} items > codebook size 256; "
              "the 4th digit cannot disambiguate. Not truncating IDs or enlarging the vocab.")
        sys.exit(1)
    duplicates = unique_codes[counts > 1]

    if len(duplicates) > 0:
        print("Resolving duplicates in codes...")
        for duplicate in duplicates:
            duplicate_indices = np.where((codes_array == duplicate).all(axis=1))[0]
            for i, idx in enumerate(duplicate_indices):
                codes_array[idx, -1] = i  # Increment the last digit for resolving duplicates

    new_unique_codes, new_counts = np.unique(codes_array, axis=0, return_counts=True)
    duplicates = new_unique_codes[new_counts > 1]

    if len(duplicates) > 0:
        print("There still have duplicates:", duplicates)
    else:
        print("There are no duplicates in the codes after resolution.")

    # ---- smoke-run validation: uniqueness and digit ranges ----
    n_items = codes_array.shape[0]
    n_unique = len(set(map(tuple, codes_array.tolist())))
    if n_unique != n_items:
        print(f"FATAL: codes not unique: {n_items - n_unique} collisions remain")
        sys.exit(1)
    first3 = codes_array[:, :3]
    last = codes_array[:, -1]
    if first3.min() < 0 or first3.max() > 255:
        print(f"FATAL: first-3 digits out of [0,255]: min={first3.min()}, max={first3.max()}")
        sys.exit(1)
    if last.min() < 0 or last.max() > 255:
        n_out = int(((last < 0) | (last > 255)).sum())
        print(f"FATAL: 4th digit out of [0,255] for {n_out} items "
              f"(min={last.min()}, max={last.max()}); collision groups too large. "
              "Not truncating IDs or enlarging the vocab.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    print(f"Saving codes to {output_file}")
    print(f"the first 5 codes: {codes_array[:5]}")
    n_items = codes_array.shape[0]
    for li in range(codes_array.shape[1]):
        n_unique = len(set(codes_array[:, li].tolist()))
        print(f"layer {li} usage: {n_unique}/{codes_array[:, li].max() + 1} distinct "
              f"(vs codebook 256: {n_unique}/256 = {n_unique / 256:.4f})")
    np.save(output_file, codes_array)
    print(f"codes shape: {codes_array.shape} ({n_items} items), all digits in [0,255], unique: OK")
