import hashlib
import json

import pandas as pd
import numpy as np
from torch.utils.data import Dataset

NUM_USER_BUCKETS = 2000
USER_TOKEN_BASE = 1025  # user tokens occupy 1025..3024 (items are 1..1024, PAD/BOS is 0)


def user_token_bucket(reviewer_id):
    """Hashing trick (paper §4.1): SHA-256 of the raw reviewerID string, full
    digest as a big-endian integer mod NUM_USER_BUCKETS, offset into the user
    token range. Deterministic across platforms/processes, unlike Python's
    built-in salted hash() which is forbidden here."""
    digest = int(hashlib.sha256(str(reviewer_id).encode("utf-8")).hexdigest(), 16)
    return USER_TOKEN_BASE + digest % NUM_USER_BUCKETS


def build_user_bucket_map(user_mapping_path):
    """int userID -> user token. Accepts preprocess.py's user_mapping.npy
    ({raw reviewerID string: 1-based int userID}) or the user_bucket_map.json
    artifact saved by training ({str(int userID): token}); both produce the
    identical mapping, so reload checks can rebuild inputs directly from the
    artifact whose sha256 is bound inside the checkpoint."""
    path = str(user_mapping_path)
    if path.endswith(".json"):
        with open(path) as f:
            return {int(uid): int(tok) for uid, tok in json.load(f).items()}
    mapping = np.load(path, allow_pickle=True).item()
    return {int_uid: user_token_bucket(reviewer_id) for reviewer_id, int_uid in mapping.items()}


def save_user_bucket_map(bucket_map, out_path):
    with open(out_path, "w") as f:
        json.dump({str(uid): tok for uid, tok in sorted(bucket_map.items())}, f, indent=0)


def process_data(file_path, mode, max_len, PAD_TOKEN=0):
    """
    Process parquet data based on mode ('train' or 'evaluation').

    Args:
        file_path (str): Path to the parquet file (columns: user, history, target).
        mode (str): Mode of operation ('train' or 'evaluation').
        max_len (int): Maximum length for padding or truncation.

    Returns:
        list: Processed data with the user id carried on every window.
    """
    # Load parquet data
    data = pd.read_parquet(file_path)

    # Combine "history" and "target" columns into a single sequence
    # Important: Ensure 'history' is a list and 'target' is appended correctly
    data['sequence'] = data['history'].apply(lambda x: list(x)) + data['target'].apply(lambda x: [x])

    if mode == 'train':
        # Sliding window processing
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            for i in range(1, len(sequence)):
                processed_data.append({
                    'user': row.user,
                    'history': sequence[:i],
                    'target': sequence[i]
                })
    elif mode == 'evaluation':
        # Use the last item as target and the rest as history
        processed_data = []
        for row in data.itertuples(index=False):
            sequence = row.sequence
            processed_data.append({
                'user': row.user,
                'history': sequence[:-1],
                'target': sequence[-1]
            })
    else:
        raise ValueError("Mode must be 'train' or 'evaluation'.")

    # Apply padding or truncation
    for item in processed_data:
        item['history'] = pad_or_truncate(item['history'], max_len)

    return processed_data

def pad_or_truncate(sequence, max_len, PAD_TOKEN=0):
    """
    Pad or truncate a sequence to a specified maximum length.

    Args:
        sequence (list): Input sequence.
        max_len (int): Maximum length for the sequence.

    Returns:
        list: Padded or truncated sequence.
    """
    if len(sequence) > max_len:
        # Truncate sequence
        return sequence[-max_len:]
    else:
        # Left pad sequence with PAD_TOKEN
        return [PAD_TOKEN] * (max_len - len(sequence)) + sequence

def item2code(code_path, codebook_size=256):
    """
    Convert itemID to code
    :param code_path: npy file path to store rqvae codes
    :return: dict item_to_code, code_to_item
    """
    data = np.load(code_path, allow_pickle=True)
    item_to_code = {}
    code_to_item = {}

    for index, code in enumerate(data):
        offsets = [c + i * codebook_size + 1 for i,c in enumerate(code)]
        item_to_code[index + 1] = offsets
        code_to_item[tuple(offsets)] = index + 1

    return item_to_code, code_to_item

class GenRecDataset(Dataset):
    def __init__(self, dataset_path, code_path, mode, max_len, user_mapping_path, PAD_TOKEN=0):
        """
        Initialize the GenRecDataset.
        Args:
            dataset_path (str): Path to the dataset file.
            code_path (str): Path to the item-to-code mapping file.
            mode (str): Mode of operation ('train' or 'evaluation').
            max_len (int): Maximum length for padding or truncation.
            user_mapping_path (str): Path to user_mapping.npy from preprocess.py.
            PAD_TOKEN (int, optional): Token used for padding. Defaults to 0.
        """
        self.dataset_path = dataset_path
        self.code_path = code_path
        self.mode = mode
        self.max_len = max_len
        self.PAD_TOKEN = PAD_TOKEN
        # Load item-to-code mapping
        self.item_to_code, self.code_to_item = item2code(code_path)
        # int userID -> user token (hashing trick)
        self.user_bucket_map = build_user_bucket_map(user_mapping_path)
        # Process the dataset
        self.data = self._prepare_data()

    def _prepare_data(self):
        """
        Process the dataset and convert items to codes.
        Each sample's 'history' is [[user_token]] + [4-token item IDs] * max_len:
        the user token rides in front and is never truncated or padded away.
        Returns:
            list: Processed data with items converted to codes.
        """
        # Process the data using the process_data function
        processed_data = process_data(
            self.dataset_path, self.mode, self.max_len, self.PAD_TOKEN
        )
        # Convert items to codes
        for item in processed_data:
            history_codes = []
            for x in item['history']:
                if x == self.PAD_TOKEN:
                    history_codes.append([self.PAD_TOKEN] * 4)
                    continue
                code = self.item_to_code.get(x)
                if code is None:
                    raise SystemExit(
                        f"FATAL: history itemID {x} not found in code mapping "
                        f"({self.code_path}); data/encoding mismatch")
                history_codes.append(code)
            target_code = self.item_to_code.get(item['target'])
            if target_code is None:
                raise SystemExit(
                    f"FATAL: target itemID {item['target']} not found in code mapping "
                    f"({self.code_path}); data/encoding mismatch — refusing to "
                    "evaluate with a PAD target")
            item['target'] = target_code
            item['history'] = [[self.user_bucket_map[item['user']]]] + history_codes
        return processed_data

    def __getitem__(self, index):
        """
        Get a single data item by index.
        Args:
            index (int): Index of the data item.
        Returns:
            dict: A dictionary containing 'history' and 'target'.
        """
        return self.data[index]

    def __len__(self):
        """
        Get the total number of data.
        Returns:
            int: Total number of data.
        """
        return len(self.data)
