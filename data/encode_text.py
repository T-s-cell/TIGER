"""Text encoding for TIGER original_smoke_beauty.

Replicates data/process.ipynb cell 6-7 exactly: metadata field extraction
(title/price/salesRank/brand/categories with the original falsy->None rule),
the exact f-string text template (missing keys become '', None values become
the literal string 'None'), and SentenceTransformer.encode with default
settings (no normalization, model's own truncation).

Differences from the notebook (recorded for the audit):
- raw metadata lines are parsed with ast.literal_eval instead of eval();
- embedding rows are written in ascending itemID order (notebook relied on
  metadata scan order, which downstream code silently treats as item_id-1);
- embeddings are stored as a float32 arrow list column.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer


def parse_metadata_lines(path):
    """Yield parsed dicts from a metadata file (.json.gz of python literals, or .json)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                yield ast.literal_eval(line.strip())
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line.strip())


def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_text(info):
    # exact template from process.ipynb (note the literal spaces after \n and
    # the single quotes around keys); info.get(key, '') keeps notebook behavior
    return (
        f"'title':{info.get('title', '')}\n 'price':{info.get('price', '')}\n"
        f" 'salesRank':{info.get('salesRank', '')}\n 'brand':{info.get('brand', '')}\n"
        f" 'categories':{info.get('categories', '')}"
    )


def main():
    parser = argparse.ArgumentParser(description="SentT5 text encoding for Beauty items")
    parser.add_argument("--meta", type=str, required=True,
                        help="meta_Beauty.json.gz (python literals) or converted .json")
    parser.add_argument("--item_mapping", type=str, required=True,
                        help="item_mapping.npy from preprocess.py (asin -> itemID)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="local SentenceTransformer snapshot dir of sentence-t5-base")
    parser.add_argument("--out_file", type=str, required=True,
                        help="output item_emb.parquet path")
    parser.add_argument("--mapping_out", type=str, default="",
                        help="where to write asins_by_itemid.json (default: next to out_file)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="1 replicates the notebook's per-item encode()")
    parser.add_argument("--revision", type=str, default="unknown",
                        help="model revision actually downloaded (for audit)")
    parser.add_argument("--expect_items", type=int, default=12101)
    args = parser.parse_args()

    item_mapping = np.load(args.item_mapping, allow_pickle=True).item()
    asin_to_id = dict(item_mapping)

    # notebook cell 6: falsy metadata values become None, then cell 7's
    # info.get(key, '') turns a missing key into '' and a None value into 'None'
    item_info = {}
    n_meta = 0
    for metadata in parse_metadata_lines(args.meta):
        n_meta += 1
        asin = metadata.get("asin")
        if asin in asin_to_id:
            itemID = asin_to_id[asin]
            item_info[itemID] = {
                "title": metadata.get("title") if metadata.get("title") else None,
                "price": metadata.get("price") if metadata.get("price") else None,
                "salesRank": metadata.get("salesRank") if metadata.get("salesRank") else None,
                "brand": metadata.get("brand") if metadata.get("brand") else None,
                "categories": metadata.get("categories") if metadata.get("categories") else None,
            }

    n_items = len(asin_to_id)
    missing = sorted(set(range(1, n_items + 1)) - set(item_info.keys()))
    if missing:
        raise SystemExit(
            f"FATAL: {len(missing)} items have no metadata (e.g. item_ids {missing[:10]}). "
            "Refusing to write a misaligned or partial embedding table."
        )
    if n_items != args.expect_items:
        raise SystemExit(f"FATAL: item_num={n_items}, expect {args.expect_items}")

    item_ids = list(range(1, n_items + 1))
    texts = [build_text(item_info[i]) for i in item_ids]

    print(f"metadata lines: {n_meta}, items to encode: {n_items}")
    model = SentenceTransformer(args.model_dir, device=args.device)
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                              convert_to_numpy=True)
    encode_time = time.time() - t0
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape != (n_items, 768):
        raise SystemExit(f"FATAL: embedding shape {embeddings.shape}, expect ({n_items}, 768)")
    if not np.isfinite(embeddings).all():
        raise SystemExit("FATAL: non-finite values in embeddings")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
    emb_col = pa.array([embeddings[i] for i in range(n_items)], type=pa.list_(pa.float32()))
    table = pa.table({"ItemID": pa.array(item_ids, type=pa.int64()), "embedding": emb_col})
    pq.write_table(table, args.out_file)

    mapping_out = args.mapping_out or os.path.join(
        os.path.dirname(os.path.abspath(args.out_file)), "asins_by_itemid.json")
    id_to_asin = {v: k for k, v in item_mapping.items()}
    with open(mapping_out, "w") as f:
        # row i of item_emb.parquet corresponds to item_id i (1-based) = ItemID
        json.dump({str(i): id_to_asin[i] for i in item_ids}, f, indent=0)

    report = {
        "n_items": n_items,
        "embedding_shape": list(embeddings.shape),
        "dtype": "float32",
        "encode_wall_time_sec": round(encode_time, 2),
        "batch_size": args.batch_size,
        "device": args.device,
        "model_dir": args.model_dir,
        "model_revision": args.revision,
        "st_model_max_seq_length": getattr(model, "max_seq_length", None),
        "out_sha256": sha256_file(args.out_file),
        "mapping_out": mapping_out,
    }
    with open(os.path.join(os.path.dirname(os.path.abspath(args.out_file)),
                           "encode_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
