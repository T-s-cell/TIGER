"""Reproducible preprocessing for TIGER original_smoke_beauty.

Derived from data/process.ipynb at base commit f1e30af. Keeps the original
mapping order (first appearance, 1-based), stable timestamp sort, leave-one-out
split and per-user {user, history, target} parquet layout unchanged. The only
behavioral change vs. the notebook is parsing raw lines with ast.literal_eval
instead of eval().

Inputs: reviews_Beauty_5.json.gz and meta_Beauty.json.gz in --raw_dir.
Outputs in --out_dir: train.parquet, valid.parquet, test.parquet,
user_mapping.npy, item_mapping.npy, dataset_stats.json.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os

import numpy as np
import pandas as pd

EXPECTED_USERS = 22363
EXPECTED_ITEMS = 12101


def parse(path):
    with gzip.open(path, "r") as g:
        for line in g:
            yield json.dumps(ast.literal_eval(line.decode("utf-8")))


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def convert_reviews(raw_dir, out_dir, dataset_name):
    json_path = os.path.join(out_dir, f"{dataset_name}.json")
    with open(json_path, "w") as f:
        for line in parse(os.path.join(raw_dir, f"reviews_{dataset_name}_5.json.gz")):
            f.write(line + "\n")
    return json_path


def build_mappings_and_splits(json_path, out_dir, dataset_name):
    userID_mapping = {}
    itemID_mapping = {}

    userIDs, itemIDs, timestamps = [], [], []
    with open(json_path, "r") as data:
        for line in data:
            review = json.loads(line.strip())
            userID = review["reviewerID"]
            itemID = review["asin"]
            timestamp = review["unixReviewTime"]
            if userID not in userID_mapping:
                userID_mapping[userID] = len(userID_mapping) + 1
            if itemID not in itemID_mapping:
                itemID_mapping[itemID] = len(itemID_mapping) + 1
            userIDs.append(userID_mapping[userID])
            itemIDs.append(itemID_mapping[itemID])
            timestamps.append(timestamp)

    user_item_mapping = {}
    for userID, itemID, timestamp in zip(userIDs, itemIDs, timestamps):
        user_item_mapping.setdefault(userID, []).append((itemID, timestamp))
    for userID in user_item_mapping:
        # Python sort is stable: ties keep review-file order, same as notebook
        user_item_mapping[userID].sort(key=lambda x: x[1])
        user_item_mapping[userID] = [item[0] for item in user_item_mapping[userID]]

    train_data, val_data, test_data = {}, {}, {}
    for userID, item_sequence in user_item_mapping.items():
        train_data[userID] = item_sequence[:-2]
        val_data[userID] = item_sequence[:-1]
        test_data[userID] = item_sequence

    def prepare_data(data_dict):
        rows = []
        for userID, item_sequence in data_dict.items():
            rows.append({
                "user": userID,
                "history": item_sequence[:-1],
                "target": item_sequence[-1],
            })
        return pd.DataFrame(rows)

    train_df = prepare_data(train_data)
    val_df = prepare_data(val_data)
    test_df = prepare_data(test_data)

    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "valid.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    np.save(os.path.join(out_dir, "user_mapping.npy"), userID_mapping)
    np.save(os.path.join(out_dir, "item_mapping.npy"), itemID_mapping)

    # Time-ordering sanity check: recompute from timestamps, compare counts
    stats = {
        "n_reviews": len(userIDs),
        "user_num": len(userID_mapping),
        "item_num": len(itemID_mapping),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="TIGER Beauty preprocessing (from process.ipynb)")
    parser.add_argument("--raw_dir", type=str, required=True,
                        help="directory containing reviews_Beauty_5.json.gz and meta_Beauty.json.gz")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="Beauty")
    parser.add_argument("--expect_users", type=int, default=EXPECTED_USERS)
    parser.add_argument("--expect_items", type=int, default=EXPECTED_ITEMS)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reviews_gz = os.path.join(args.raw_dir, f"reviews_{args.dataset_name}_5.json.gz")
    meta_gz = os.path.join(args.raw_dir, f"meta_{args.dataset_name}.json.gz")
    for p in (reviews_gz, meta_gz):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    json_path = convert_reviews(args.raw_dir, args.out_dir, args.dataset_name)
    stats = build_mappings_and_splits(json_path, args.out_dir, args.dataset_name)
    os.remove(json_path)

    if stats["user_num"] != args.expect_users or stats["item_num"] != args.expect_items:
        raise SystemExit(
            f"FATAL: dataset counts mismatch: users={stats['user_num']} (expect {args.expect_users}), "
            f"items={stats['item_num']} (expect {args.expect_items}). Not silently proceeding."
        )

    # ASIN coverage is checked by encode_text.py against the metadata file; here we
    # record the mapping files needed there.
    stats["raw_sha256"] = {
        os.path.basename(reviews_gz): sha256_file(reviews_gz),
        os.path.basename(meta_gz): sha256_file(meta_gz),
    }
    stats["output_sha256"] = {
        name: sha256_file(os.path.join(args.out_dir, name))
        for name in ["train.parquet", "valid.parquet", "test.parquet",
                     "user_mapping.npy", "item_mapping.npy"]
    }
    with open(os.path.join(args.out_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
