import collections
import hashlib
import json
import logging
import os

import numpy as np
import torch
from time import time
from torch import optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from utils import ensure_dir,set_color,get_local_time,delete_file

import heapq
class Trainer(object):

    def __init__(self, args, model, data_num):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.lr_scheduler_type = args.lr_scheduler_type

        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.warmup_steps = args.warmup_epochs * data_num
        self.max_steps = args.epochs * data_num

        self.save_limit = args.save_limit
        self.best_save_heap = []
        self.newest_save_queue = []
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)

        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.scheduler = self._get_scheduler()
        self.model = self.model.to(self.device)

        # ---- bookkeeping added for the smoke run (no change to training math) ----
        self.total_optimizer_steps = 0
        self.records = {"epochs": [], "param_checks": [], "first_batch_wall": None,
                        "kmeans_init": None, "peak_mem_allocated": None,
                        "peak_mem_reserved": None, "steps_per_epoch": data_num}

        # ---- record-only diagnostics (--diagnose): training math unchanged ----
        self.diagnose = getattr(args, "diagnose", False)
        if self.diagnose:
            self.records["diagnostics"] = {
                "per_epoch": [], "first_steps": [],
                "note": "record-only instrumentation; usage/latent/grad-norm "
                        "measured on the training batches themselves",
            }
            self.records["data_stats"] = getattr(args, "data_stats", None)
        # spot-check param updates on a few epoch starts after warmup (lr>0),
        # not every step, so timing is not distorted
        warmup = args.warmup_epochs
        candidates = [warmup + 10, max(warmup + 10, self.epochs - 20), self.epochs - 1]
        self.param_check_epochs = sorted({c for c in candidates if warmup < c < self.epochs})

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _peak_mem(self):
        if self.device.type == "cuda":
            alloc = torch.cuda.max_memory_allocated(self.device)
            resv = torch.cuda.max_memory_reserved(self.device)
            self.records["peak_mem_allocated"] = max(self.records["peak_mem_allocated"] or 0, alloc)
            self.records["peak_mem_reserved"] = max(self.records["peak_mem_reserved"] or 0, resv)

    def _component_hashes(self):
        m = self.model
        blobs = {"encoder": m.encoder.state_dict(),
                 "decoder": m.decoder.state_dict()}
        out = {}
        for name, sd in blobs.items():
            h = hashlib.sha256()
            for k in sorted(sd):
                h.update(k.encode()); h.update(sd[k].detach().cpu().numpy().tobytes())
            out[name] = h.hexdigest()
        for li, vq in enumerate(m.rq.vq_layers):
            out[f"codebook_l{li}"] = hashlib.sha256(
                vq.embedding.weight.detach().cpu().numpy().tobytes()).hexdigest()
        return out

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            # assumption params (paper gives only lr): initial_accumulator_value
            # and eps follow the common PyTorch recipe, lr_decay disabled
            optimizer = optim.Adagrad(
                params, lr=learning_rate, lr_decay=0.0,
                weight_decay=weight_decay,
                initial_accumulator_value=0.1, eps=1e-7,
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def _get_scheduler(self):
        if self.lr_scheduler_type.lower() == "constant":
            return None
        if self.lr_scheduler_type.lower() == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                           num_warmup_steps=self.warmup_steps,
                                                           num_training_steps=self.max_steps)
        else:
            lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer,
                                                             num_warmup_steps=self.warmup_steps)

        return lr_scheduler
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")


    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_codebook_loss = 0
        total_commit_loss = 0
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        do_param_check = epoch_idx in self.param_check_epochs
        pre_hashes = self._component_hashes() if do_param_check else None
        check_done = False

        epoch_indices = [] if self.diagnose else None
        epoch_grad_norms = [] if self.diagnose else None

        self._sync()
        epoch_start = time()
        lr_this_epoch = None
        for batch_idx, data in enumerate(iter_data):
            if batch_idx == 0 and self.records["first_batch_wall"] is None:
                self._sync()
                t_fwd0 = time()
                init_flags_before = [vq.initted for vq in self.model.rq.vq_layers]
            data = data.to(self.device)
            if self.diagnose and batch_idx == 0:
                with torch.no_grad():
                    z0 = self.model.encoder(data)
                epoch_latent_stats = {
                    "latent_per_dim_std_mean": float(z0.std(dim=0).mean()),
                    "latent_norm_mean": float(z0.norm(dim=1).mean()),
                    "latent_norm_std": float(z0.norm(dim=1).std()),
                }
            self.optimizer.zero_grad()
            out, rq_loss, indices, layer_losses = self.model(data)
            loss, loss_recon = self.model.compute_loss(out, rq_loss, xs=data)
            self._check_nan(loss)
            loss.backward()
            total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if not torch.isfinite(total_grad_norm):
                raise ValueError("Gradient norm is not finite (nan/inf)")
            self.optimizer.step()
            self.total_optimizer_steps += 1
            if self.diagnose:
                epoch_indices.append(indices.detach().cpu().numpy())
                gn = float(total_grad_norm)
                epoch_grad_norms.append(gn)
                if self.total_optimizer_steps <= 5:
                    self.records["diagnostics"]["first_steps"].append({
                        "global_step": self.total_optimizer_steps,
                        "epoch": epoch_idx, "batch": batch_idx,
                        "loss": float(loss), "recon": float(loss_recon),
                        "grad_norm_pre_clip": gn,
                    })
            if self.scheduler is not None:
                self.scheduler.step()
                lr_this_epoch = self.scheduler.get_last_lr()[0]
            else:
                lr_this_epoch = self.optimizer.param_groups[0]["lr"]
            if batch_idx == 0 and self.records["first_batch_wall"] is None:
                self._sync()
                self.records["first_batch_wall"] = time() - t_fwd0
                init_flags_after = [vq.initted for vq in self.model.rq.vq_layers]
                if not all(init_flags_before) and all(init_flags_after):
                    self.records["kmeans_init"] = {
                        "first_batch_includes_kmeans_init": True,
                        "wall_sec": round(self.records["first_batch_wall"], 3),
                        "note": "first-batch forward+backward+step wall time; sklearn kmeans runs inside init_emb on CPU",
                    }
            # spot-check: did each component actually update on a real (lr>0) step?
            if do_param_check and not check_done and lr_this_epoch and lr_this_epoch > 0:
                post_hashes = self._component_hashes()
                self.records["param_checks"].append({
                    "epoch": epoch_idx, "batch": batch_idx,
                    "lr": lr_this_epoch,
                    "changed": {k: post_hashes[k] != pre_hashes[k] for k in post_hashes},
                })
                check_done = True
            # print(self.scheduler.get_last_lr())
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_codebook_loss += sum(ll["codebook"].item() for ll in layer_losses)
            total_commit_loss += sum(ll["commitment"].item() for ll in layer_losses)

        self._sync()
        epoch_time = time() - epoch_start
        self._peak_mem()
        if self.diagnose:
            arr = np.concatenate(epoch_indices, axis=0)
            groups = collections.Counter(map(tuple, arr.tolist()))
            diag = {
                "epoch": epoch_idx,
                "n_assigned": int(arr.shape[0]),
                "layer_unique": [int(len(set(arr[:, l].tolist()))) for l in range(arr.shape[1])],
                "triplet_unique": len(groups),
                "max_group": int(max(groups.values())),
                "top1_share": float(max(groups.values()) / arr.shape[0]),
                "grad_norm_preclip_mean": float(np.mean(epoch_grad_norms)),
                "grad_norm_preclip_max": float(np.max(epoch_grad_norms)),
                **epoch_latent_stats,
            }
            self.records["diagnostics"]["per_epoch"].append(diag)
            self.logger.info(
                "diag epoch %d: layer_unique %s triplet_unique %d max_group %d "
                "top1_share %.4f grad_norm %.4f latent_std %.6f" % (
                    epoch_idx, diag["layer_unique"], diag["triplet_unique"],
                    diag["max_group"], diag["top1_share"],
                    diag["grad_norm_preclip_mean"],
                    diag["latent_per_dim_std_mean"]))
        self.records["epochs"].append({
            "epoch": epoch_idx,
            "wall_sec": round(epoch_time, 3),
            "steps": len(iter_data) if hasattr(iter_data, "__len__") else batch_idx + 1,
            "lr_last": lr_this_epoch,
            "loss": total_loss,
            "recon_loss": total_recon_loss,
            "codebook_loss": total_codebook_loss,
            "commit_loss": total_commit_loss,
        })
        self.logger.info(
            f"epoch {epoch_idx} wall {epoch_time:.3f}s steps {self.total_optimizer_steps} "
            f"lr {lr_this_epoch:.6g} loss {total_loss:.4f} "
            f"recon {total_recon_loss:.4f} codebook {total_codebook_loss:.4f} "
            f"commit(incl beta) {total_commit_loss:.4f}"
        )

        return total_loss, total_recon_loss

    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )

        indices_set = set()
        num_sample = 0
        layer_unique = [set() for _ in range(self.model.rq.num_quantizers)]
        for batch_idx, data in enumerate(iter_data):
            num_sample += len(data)
            data = data.to(self.device)
            indices = self.model.get_indices(data)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for li in range(indices.shape[-1]):
                layer_unique[li].update(indices[:, li].tolist())
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(list(indices_set)))/num_sample
        n_e = self.model.rq.vq_layers[0].n_e
        layer_usage = [round(len(s) / n_e, 4) for s in layer_unique]

        return collision_rate, layer_usage

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

        return ckpt_path

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        return train_loss_output + "]"


    def fit(self, data):

        cur_eval_step = 0

        for epoch_idx in range(self.epochs):
            # train
            training_start_time = time()
            train_loss, train_recon_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)


            # eval
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate, layer_usage = self._valid_epoch(data)

                if train_loss < self.best_loss:
                    self.best_loss = train_loss
                    self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)

                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    cur_eval_step = 0
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                else:
                    cur_eval_step += 1


                valid_end_time = time()
                valid_score_output = (
                    set_color("epoch %d evaluating", "green")
                    + " ["
                    + set_color("time", "blue")
                    + ": %.2fs, "
                    + set_color("collision_rate", "blue")
                    + ": %f]"
                ) % (epoch_idx, valid_end_time - valid_start_time, collision_rate)

                self.logger.info(valid_score_output)
                self.records["final_layer_usage"] = layer_usage
                self.records.setdefault("validations", []).append({
                    "epoch": epoch_idx,
                    "collision_rate": collision_rate,
                    "layer_usage": layer_usage,
                    "wall_sec": round(valid_end_time - valid_start_time, 3),
                })
                ckpt_path = self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
                now_save = (-collision_rate, ckpt_path)
                if len(self.newest_save_queue) < self.save_limit:
                    self.newest_save_queue.append(now_save)
                    heapq.heappush(self.best_save_heap, now_save)
                else:
                    old_save = self.newest_save_queue.pop(0)
                    self.newest_save_queue.append(now_save)
                    if collision_rate < -self.best_save_heap[0][0]:
                        bad_save = heapq.heappop(self.best_save_heap)
                        heapq.heappush(self.best_save_heap, now_save)

                        if bad_save not in self.newest_save_queue:
                            delete_file(bad_save[1])

                    if old_save not in self.best_save_heap:
                        delete_file(old_save[1])


        self.records["total_optimizer_steps"] = self.total_optimizer_steps
        self.records["best_loss"] = float(self.best_loss) if np.isfinite(self.best_loss) else None
        self.records["best_collision_rate"] = float(self.best_collision_rate) if np.isfinite(self.best_collision_rate) else None
        self.records["loss_reduction"] = {
            "recon": "per-sample feature-dim sum, then batch mean",
            "quant": "sum over layers of (codebook + beta*commitment), same reduction",
        }
        final_path = self._save_checkpoint(epoch=self.epochs - 1, ckpt_file="final_model.pth")
        self.records["final_model_path"] = final_path
        stats_path = os.path.join(self.ckpt_dir, "train_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"train stats written to {stats_path}")

        return self.best_loss, self.best_collision_rate
