"""Training / evaluation loops shared by run_experiments.py and the notebook's Step 9.

The loops follow exactly the same protocol as the notebook baseline
(Steps 4-7): AdamW, cross-entropy, one pass over the shuffled training set per
epoch, model selection on validation macro-AUROC + macro-F1, and a single
evaluation of the selected checkpoint on the held-out test set.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import classification_metrics


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
class CachedFeatureDataset(Dataset):
    """Loads every slide of an HDF5FeatureDataset once and keeps it in RAM.

    Reading a multi-GB HDF5 file from disk every epoch dominates the run time
    of these small models; caching once per dataset makes a 30-epoch run take
    about a minute on a GPU. Features are stored as ``dtype`` (float16 by
    default to halve memory) and returned as float32, so the model sees the
    same values in every run and for every method.
    """

    def __init__(self, base, dtype=torch.float16, device="cpu", verbose=True):
        self.data_names = list(base.data_names)
        self.device = torch.device(device)
        self.items = []
        t0 = time.time()
        for i in range(len(base)):
            item = base[i]
            self.items.append({
                "input": item["input"].to(device=self.device, dtype=dtype).contiguous(),
                "coords": None,
                "label": int(item["label"]),
                "label_t": torch.tensor([int(item["label"])], dtype=torch.long, device=self.device),
                "slide_id": item["slide_id"],
            })
        if verbose:
            n = sum(int(it["input"].shape[0]) for it in self.items)
            print(f"  cached {len(self.items)} slides / {n:,} patches in {time.time() - t0:.1f}s")

    def __len__(self):
        return len(self.items)

    @property
    def feature_dim(self):
        return int(self.items[0]["input"].shape[-1])

    def __getitem__(self, index):
        it = self.items[index]
        return {"input": it["input"].float(), "coords": None, "label": it["label"], "slide_id": it["slide_id"]}


class GPUBagLoader:
    """Batch-size-1 loader over a CachedFeatureDataset whose bags already live on the GPU.

    Yields the same dictionary as ``mil_collate_fn`` applied to one bag
    (input ``[1, N, D]`` float32, label ``[1]``, mask ``None`` because a single
    bag has no padding, slide_id ``[id]``) without any CPU-side copy. Shuffling
    draws ``torch.randperm`` from a seeded generator every epoch, the same
    sampling scheme as ``DataLoader(shuffle=True)``. On a 2-vCPU Colab VM this
    makes a run several times faster than the CPU DataLoader path.
    """

    def __init__(self, dataset, shuffle=False, generator=None):
        self.dataset, self.shuffle, self.generator = dataset, shuffle, generator

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        n = len(self.dataset)
        order = torch.randperm(n, generator=self.generator).tolist() if self.shuffle else range(n)
        for i in order:
            it = self.dataset.items[i]
            yield {"input": it["input"].float().unsqueeze(0), "label": it["label_t"], "mask": None,
                   "slide_id": [it["slide_id"]], "coords": [None]}


def move_batch(data, device):
    bag = data["input"].to(device, dtype=torch.float32, non_blocking=True)
    labels = data["label"].to(device, dtype=torch.long, non_blocking=True)
    mask = data.get("mask")
    if mask is not None:
        mask = mask.to(device, non_blocking=True)
    return bag, labels, mask


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_one_epoch(model, criterion, data_loader, optimizer, device):
    """One optimisation pass. Uses ``model.training_loss`` so that models with
    auxiliary objectives (ACMIL) and plain cross-entropy models share the loop.
    The reported loss is the full training objective."""
    model.train()
    total_loss, total_correct, total_examples = 0.0, 0, 0
    for data in data_loader:
        bag, labels, mask = move_batch(data, device)
        optimizer.zero_grad(set_to_none=True)
        loss, logits = model.training_loss(bag, labels, criterion, mask=mask)
        loss.backward()
        optimizer.step()

        bs = labels.shape[0]
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += bs
    return {"loss": total_loss / total_examples, "acc": 100.0 * total_correct / total_examples}


def attention_statistics(attention_1d):
    """Concentration statistics of one slide's attention distribution.

    entropy_norm  H(alpha) / log N in [0, 1]; 1 = uniform, 0 = one patch.
    eff_frac      exp(H) / N: effective fraction of patches that are attended.
    top1pct_mass  attention mass on the top 1% of patches (at least 1 patch).
    max_weight    largest single weight.
    """
    a = attention_1d.double().clamp_min(0)
    a = a / a.sum()
    n = a.numel()
    h = -(a[a > 0] * a[a > 0].log()).sum().item()
    k = max(1, int(math.ceil(0.01 * n)))
    return {
        "n_patches": int(n),
        "entropy_norm": h / math.log(n) if n > 1 else 1.0,
        "eff_frac": math.exp(h) / n,
        "top1pct_mass": a.topk(k).values.sum().item(),
        "max_weight": a.max().item(),
    }


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, n_class, return_details=False):
    """Loss, macro-AUROC, accuracy and macro-F1 on one split.

    With ``return_details=True`` also returns per-slide predictions and
    attention-concentration statistics (used for the extension analysis).
    """
    model.eval()
    total_loss, total_examples = 0.0, 0
    probs_all, labels_all, rows = [], [], []
    for data in data_loader:
        bag, labels, mask = move_batch(data, device)
        logits, attention = model(bag, mask=mask, return_attention=True)
        loss = criterion(logits, labels)
        probabilities = F.softmax(logits, dim=1)

        total_loss += loss.item() * labels.shape[0]
        total_examples += labels.shape[0]
        probs_all.append(probabilities)
        labels_all.append(labels)

        if return_details:
            for b in range(labels.shape[0]):
                valid = mask[b] if mask is not None else slice(None)
                row = {
                    "slide_id": data["slide_id"][b],
                    "label": int(labels[b]),
                    "pred": int(probabilities[b].argmax()),
                    **{f"p{c}": float(probabilities[b, c]) for c in range(probabilities.shape[1])},
                }
                if attention is not None:
                    row.update(attention_statistics(attention[b, 0][valid].cpu()))
                rows.append(row)

    probs_all = torch.cat(probs_all)
    labels_all = torch.cat(labels_all)
    metrics = classification_metrics(probs_all, labels_all, n_class)
    metrics["loss"] = total_loss / total_examples
    if return_details:
        return metrics, rows
    return metrics


def fit(model, criterion, optimizer, train_loader, val_loader, device, n_class, n_epochs, verbose=True):
    """Train for ``n_epochs`` and keep the epoch with the best validation
    macro-AUROC + macro-F1 (same rule as the notebook baseline)."""
    best_score, best_epoch, best_state = float("-inf"), -1, None
    history = []
    for epoch in range(n_epochs):
        t0 = time.time()
        tr = train_one_epoch(model, criterion, train_loader, optimizer, device)
        va = evaluate(model, criterion, val_loader, device, n_class)
        score = va["auc"] + va["f1"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        row = {
            "epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["acc"],
            "val_loss": va["loss"], "val_acc": va["acc"], "val_auc": va["auc"], "val_f1": va["f1"],
        }
        history.append(row)
        if verbose:
            print(f"    ep {epoch + 1:02d}/{n_epochs} | train loss={tr['loss']:.4f} acc={tr['acc']:.1f} | "
                  f"val loss={va['loss']:.4f} auc={va['auc']:.3f} f1={va['f1']:.3f} | "
                  f"best={best_epoch + 1} | {time.time() - t0:.1f}s")
    return {"best_state": best_state, "best_epoch": best_epoch, "best_score": best_score, "history": history}


def summarize_attention(rows):
    keys = ["entropy_norm", "eff_frac", "top1pct_mass", "max_weight"]
    if not rows or keys[0] not in rows[0]:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
