import torch
from transformers import T5ForConditionalGeneration, T5Config
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import json
import numpy as np
import os
from time import time
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse
import random
import pandas as pd
from tqdm import tqdm
import logging
from dataset import GenRecDataset
from dataloader import GenRecDataLoader

class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
        num_layers=config['num_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_heads=config['num_heads'],
        d_kv=config['d_kv'],
        dropout_rate=config['dropout_rate'],
        vocab_size=config['vocab_size'],
        pad_token_id=config['pad_token_id'],
        eos_token_id=config['eos_token_id'],
        decoder_start_token_id=config['pad_token_id'],
        feed_forward_proj=config['feed_forward_proj'],
    )
        # Initialize T5 model with the specified configuration
        self.model = T5ForConditionalGeneration(t5config)

    @property
    def n_parameters(self):
      """Calculates the number of parameters in the model."""
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
      """Forward pass of the model. Returns the output logits and the loss value."""
      outputs = self.model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          labels=labels
      )
      return outputs.loss, outputs.logits

    def generate(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,  num_beams: int = 20, **kwargs):
        """Generate recommendations using the model."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs
        )

def calculate_pos_index(preds, labels, maxk=20):
    """Calculate the position index of the ground truth items."""
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    assert (
        preds.shape[1] == maxk
    ), f'preds.shape[1] = {preds.shape[1]} != {maxk}'

    pos_index = torch.zeros((preds.shape[0], maxk), dtype=torch.bool)
    for i in range(preds.shape[0]):
      cur_label = labels[i].tolist()
      for j in range(maxk):
        cur_pred = preds[i, j].tolist()
        if cur_pred == cur_label:
          pos_index[i, j] = True
          break
    return pos_index

def recall_at_k(pos_index, k):
  return pos_index[:, :k].sum(dim=1).cpu().float()

def ndcg_at_k(pos_index, k):
  # Assume only one ground truth item per example
  ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
  dcg = 1.0 / torch.log2(ranks + 1)
  dcg = torch.where(pos_index, dcg, torch.tensor(0.0, dtype=torch.float, device=dcg.device))
  return dcg[:, :k].sum(dim=1).cpu().float()

def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    n_steps = 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time()
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['history'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['target'].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall = time() - t_start
    return total_loss / len(train_loader), n_steps, wall

def evaluate(model, eval_loader, topk_list, beam_size, device, return_per_sample=False):
    """Original batch-mean metrics plus sample-weighted metrics (distinct names).

    The original code averaged per-batch means, which mis-weights the final
    short batch; the SampleWeighted_* metrics are computed per sample and are
    the trustworthy aggregate. Batch-mean values are kept for comparability.
    """
    model.eval()
    recalls = {'Recall@' + str(k): [] for k in topk_list}
    ndcgs = {'NDCG@' + str(k): [] for k in topk_list}
    sw_hit = {'SampleWeighted_Recall@' + str(k): 0.0 for k in topk_list}
    sw_dcg = {'SampleWeighted_NDCG@' + str(k): 0.0 for k in topk_list}
    n_samples = 0
    per_sample = []

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['history'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
            preds = preds[:, 1:]  # Exclude the start token
            preds = preds.reshape(input_ids.shape[0], beam_size, -1)  # (batch_size, beam_size, seq_len)
            pos_index = calculate_pos_index(preds, labels, maxk=beam_size)
            n_samples += pos_index.shape[0]
            for k in topk_list:
                per_sample_recall = recall_at_k(pos_index, k)
                per_sample_ndcg = ndcg_at_k(pos_index, k)
                recall = per_sample_recall.mean().item()
                ndcg = per_sample_ndcg.mean().item()
                recalls['Recall@' + str(k)].append(recall)
                ndcgs['NDCG@' + str(k)].append(ndcg)
                sw_hit['SampleWeighted_Recall@' + str(k)] += per_sample_recall.sum().item()
                sw_dcg['SampleWeighted_NDCG@' + str(k)] += per_sample_ndcg.sum().item()
            if return_per_sample:
                per_sample.append(pos_index.clone())
    if device.type == 'cuda':
        torch.cuda.synchronize()
    wall = time() - t_start
    # Calculate average recalls and ndcgs (original batch-mean)
    avg_recalls = {k: sum(v) / len(v) for k, v in recalls.items()}
    avg_ndcgs = {k: sum(v) / len(v) for k, v in ndcgs.items()}
    # Sample-weighted over all samples
    sw_recalls = {k: v / n_samples for k, v in sw_hit.items()}
    sw_ndcgs = {k: v / n_samples for k, v in sw_dcg.items()}
    if return_per_sample:
        return avg_recalls, avg_ndcgs, sw_recalls, sw_ndcgs, wall, n_samples, torch.cat(per_sample, dim=0)
    return avg_recalls, avg_ndcgs, sw_recalls, sw_ndcgs, wall, n_samples

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def sha256_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def save_checkpoint(path, model, config, epoch, codes_sha256):
    torch.save({
        'state_dict': model.state_dict(),
        'config': {k: v for k, v in config.items()},
        'epoch': epoch,
        'codes_sha256': codes_sha256,
    }, path)

def save_reference_outputs(path, model, fixed_batch, beam_size, device):
    """Reference outputs recorded from the in-memory model at save time.

    A later process must reproduce these from the saved checkpoint alone; this
    is what makes the reload check meaningful (re-forwarding a loaded model
    without a save-time reference cannot detect a bad save).
    """
    was_training = model.training
    model.eval()
    input_ids = fixed_batch['history'].to(device)
    attention_mask = fixed_batch['attention_mask'].to(device)
    labels = fixed_batch['target'].to(device)
    with torch.no_grad():
        _, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        preds = model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=beam_size)
    np.savez_compressed(
        path,
        input_ids=input_ids.cpu().numpy().astype(np.int64),
        attention_mask=attention_mask.cpu().numpy().astype(np.int64),
        labels=labels.cpu().numpy().astype(np.int64),
        logits=logits.detach().cpu().numpy().astype(np.float32),
        preds=preds.detach().cpu().numpy().astype(np.int64),
    )
    if was_training:
        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIGER configuration")
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--infer_size', type=int, default=96, help='Inference size for generating recommendations')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers in the model')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--d_ff', type=int, default=1024, help='Dimension of the feed-forward layer')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--d_kv', type=int, default=64, help='Dimension of key and value vectors')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--vocab_size', type=int, default=1025, help='Vocabulary size')
    parser.add_argument('--pad_token_id', type=int, default=0, help='Padding token ID')
    parser.add_argument('--eos_token_id', type=int, default=0, help='End of sequence token ID')
    parser.add_argument('--feed_forward_proj', type=str, default='relu', help='Feed forward projection type')
    parser.add_argument('--max_len', type=int, default=20, help='Maximum length for padding or truncation')
    parser.add_argument('--dataset_path', type=str, default='../data/Beauty', help='Path to the dataset')
    parser.add_argument('--code_path', type=str, default='../data/Beauty/Beauty_t5_rqvae.npy', help='Path to the item-to-code mapping file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluation'], help='Mode of operation (kept for CLI compatibility; independent evaluation lives in evaluate.py)')
    parser.add_argument('--log_path', type=str, default='./logs/tiger.log', help='Path to the log file')
    parser.add_argument('--seed', type=int, default=2025, help='Random seed for reproducibility')
    parser.add_argument('--save_path', type=str, default='./ckpt/tiger.pth', help='Path to save the trained model')
    parser.add_argument('--early_stop', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--topk_list', type=list, default=[5,10,20], help='List of top-k values for evaluation metrics (argparse type=list is broken; do NOT pass this on the CLI)')
    parser.add_argument('--beam_size', type=int, default=30, help='Beam size for generation')
    parser.add_argument('--ref_size', type=int, default=32, help='Number of fixed validation samples for reference outputs')
    config = vars(parser.parse_args())
    # Set up logging
    log_dir = os.path.dirname(os.path.abspath(config['log_path']))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=config['log_path'],
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logging.info(f"Configuration: {config}")

    # Set random seed BEFORE model creation (original code set it after)
    set_seed(config['seed'])

    # Explicit device check: never silently fall back to CPU
    requested = torch.device(config['device'])
    if requested.type == 'cuda' and not torch.cuda.is_available():
        raise SystemExit(f"FATAL: --device {config['device']} requested but torch.cuda.is_available() is False")
    device = requested
    logging.info(f"Using device: {device}")

    # Initialize model (after seed)
    model = TIGER(config)
    print(model.n_parameters)
    logging.info(model.n_parameters)

    codes_sha256 = sha256_file(config['code_path'])
    logging.info(f"code_path sha256: {codes_sha256}")

    train_dataset = GenRecDataset(
        dataset_path=config['dataset_path']+ '/train.parquet',
        code_path=config['code_path'],
        mode='train',
        max_len=config['max_len']
    )
    validation_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/valid.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len']
    )
    test_dataset = GenRecDataset(
        dataset_path=config['dataset_path'] + '/test.parquet',
        code_path=config['code_path'],
        mode='evaluation',
        max_len=config['max_len']
    )

    train_dataloader = GenRecDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = GenRecDataLoader(validation_dataset, batch_size=config['infer_size'], shuffle=False)
    test_dataloader = GenRecDataLoader(test_dataset, batch_size=config['infer_size'], shuffle=False)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Train the model
    model.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    best_ndcg = 0.0
    early_stop_counter = 0
    history = []
    run_start = time()

    # fixed reference batch: first ref_size validation samples in dataset order
    fixed_batch = validation_dataloader.collate_fn([validation_dataset[i] for i in range(config['ref_size'])])

    save_dir = os.path.dirname(os.path.abspath(config['save_path']))
    os.makedirs(save_dir, exist_ok=True)
    last_smoke_path = os.path.splitext(config['save_path'])[0] + '_last_smoke.pth'

    for epoch in range(config['num_epochs']):
        logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        train_loss, n_steps, train_wall = train(model, train_dataloader, optimizer, device)
        logging.info(f"Training loss: {train_loss} steps: {n_steps} wall_sec: {train_wall:.3f}")
        # Evaluate the model
        avg_recalls, avg_ndcgs, sw_recalls, sw_ndcgs, eval_wall, n_valid = evaluate(
            model, validation_dataloader, config['topk_list'], config['beam_size'], device)
        logging.info(f"Validation Dataset (batch-mean): {avg_recalls} {avg_ndcgs} wall_sec: {eval_wall:.3f}")
        logging.info(f"Validation Dataset (sample-weighted): {sw_recalls} {sw_ndcgs}")
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss, 'train_steps': n_steps, 'train_wall_sec': round(train_wall, 3),
            'valid_batch_mean': {**avg_recalls, **avg_ndcgs},
            'valid_sample_weighted': {**sw_recalls, **sw_ndcgs},
            'valid_wall_sec': round(eval_wall, 3), 'valid_n_samples': n_valid,
        })
        # save last_smoke every epoch so a short/failed run is still reloadable
        save_checkpoint(last_smoke_path, model, config, epoch + 1, codes_sha256)
        save_reference_outputs(last_smoke_path.replace('.pth', '_ref.npz'), model, fixed_batch, config['beam_size'], device)
        if avg_ndcgs['NDCG@20'] > best_ndcg:
            best_ndcg = avg_ndcgs['NDCG@20']
            early_stop_counter = 0  # Reset early stop counter
            test_avg_recalls, test_avg_ndcgs, test_sw_recalls, test_sw_ndcgs, test_wall, n_test, test_pos = evaluate(
                model, test_dataloader, config['topk_list'], config['beam_size'], device, return_per_sample=True)
            logging.info(f"Best NDCG@20: {best_ndcg}")
            logging.info(f"Test Dataset (batch-mean): {test_avg_recalls} {test_avg_ndcgs} wall_sec: {test_wall:.3f}")
            logging.info(f"Test Dataset (sample-weighted): {test_sw_recalls} {test_sw_ndcgs}")
            np.savez_compressed(os.path.join(save_dir, 'test_per_sample.npz'),
                                pos_index=test_pos.numpy().astype(bool),
                                epoch=epoch + 1)
            # Save the best model (unified format, tied to this run's codes)
            save_checkpoint(config['save_path'], model, config, epoch + 1, codes_sha256)
            save_reference_outputs(config['save_path'].replace('.pth', '_ref.npz'), model, fixed_batch, config['beam_size'], device)
            logging.info(f"Best model saved to {config['save_path']}")
        else:
            early_stop_counter += 1
            logging.info(f"No improvement in NDCG@20. Early stop counter: {early_stop_counter}")
            if early_stop_counter >= config['early_stop']:
                logging.info("Early stopping triggered.")
                break

    total_wall = time() - run_start
    peak_mem = None
    if device.type == 'cuda':
        peak_mem = {
            'allocated_bytes': torch.cuda.max_memory_allocated(device),
            'reserved_bytes': torch.cuda.max_memory_reserved(device),
        }
    summary = {
        'config': config,
        'codes_sha256': codes_sha256,
        'total_wall_sec': round(total_wall, 3),
        'peak_mem': peak_mem,
        'best_ndcg20_batch_mean': best_ndcg,
        'save_path': config['save_path'],
        'last_smoke_path': last_smoke_path,
        'history': history,
    }
    summary_path = os.path.join(save_dir, 'rec_train_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Summary written to {summary_path}; total wall {total_wall:.1f}s")
    print(f"Summary written to {summary_path}")
