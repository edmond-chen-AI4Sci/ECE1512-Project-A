"""MIL models for ECE1512 Project A.

All models share one calling convention so the same training/evaluation code,
the attention statistics and ``wsi_visualization.visualize_wsi_attention`` work
for every model:

    logits = model(x, mask=None)                                  # [B, C]
    logits, attention = model(x, mask=None, return_attention=True)  # attention: [B, 1, N]
    loss, logits = model.training_loss(x, labels, criterion, mask=None)

``x`` is a padded bag of patch features ``[B, N, D_feat]`` and ``mask`` is the
boolean ``[B, N]`` tensor from ``mil_collate_fn`` (True = real patch). With the
default batch size B=1 there is no padding, but every model honours the mask so
that larger batches remain correct.

Models
------
ABMIL      Gated-attention MIL baseline (Ilse et al., 2018); identical to the
           model completed in the notebook.
MeanPoolMIL / MaxPoolMIL
           Same encoder and classifier as ABMIL, but a parameter-free aggregator.
           Used to measure what the learned attention aggregator contributes.
ACMIL      Research extension: multi-branch gated attention with an attention
           diversity loss and stochastic top-k instance masking, re-implemented
           for this code base following Zhang et al. (ECCV 2024),
           "Attention-Challenging Multiple Instance Learning for Whole Slide
           Image Classification".
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks (same as the notebook)
# ---------------------------------------------------------------------------
class Classifier1FC(nn.Module):
    """Dropout followed by one linear layer: slide feature -> class logits."""

    def __init__(self, n_channels, n_classes, droprate=0.0):
        super().__init__()
        self.dropout = nn.Dropout(droprate) if droprate > 0 else nn.Identity()
        self.fc = nn.Linear(n_channels, n_classes)

    def forward(self, x):
        return self.fc(self.dropout(x))


class DimReduction(nn.Module):
    """Trainable instance encoder on top of the frozen patch features."""

    def __init__(self, n_channels, m_dim):
        super().__init__()
        self.fc = nn.Linear(n_channels, m_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.fc(x))


class AttentionGated(nn.Module):
    """Gated attention a_i = w^T [tanh(V h_i) * sigmoid(U h_i)] with K heads."""

    def __init__(self, L=256, D=128, K=1):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(L, D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(L, D), nn.Sigmoid())
        self.attention_weights_w = nn.Linear(D, K)

    def forward(self, x):
        # x: [B, N, L] -> attention logits [B, K, N]
        a_v = self.attention_V(x)
        a_u = self.attention_U(x)
        a = self.attention_weights_w(a_v * a_u)
        return a.transpose(1, 2)


def mask_logits(logits, mask):
    """Set padded instances to -inf so they receive zero attention."""
    if mask is None:
        return logits
    return logits.masked_fill(~mask[:, None, :].bool(), float("-inf"))


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
class ABMIL(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=1)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def forward(self, x, mask=None, return_attention=False):
        med_feat = self.dimreduction(x)                      # [B, N, D_inner]
        a_logits = mask_logits(self.attention(med_feat), mask)  # [B, 1, N]
        attention = F.softmax(a_logits, dim=-1)              # sums to 1 over N
        bag_feat = torch.bmm(attention, med_feat).squeeze(1)  # [B, D_inner]
        logits = self.classifier(bag_feat)                   # [B, n_class]
        if return_attention:
            return logits, attention
        return logits

    def training_loss(self, x, labels, criterion, mask=None):
        logits = self(x, mask=mask)
        return criterion(logits, labels), logits


# ---------------------------------------------------------------------------
# Aggregator ablations (encoder and classifier unchanged)
# ---------------------------------------------------------------------------
class MeanPoolMIL(nn.Module):
    """Every instance gets weight 1/N: a non-learned, permutation-invariant pooling."""

    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def forward(self, x, mask=None, return_attention=False):
        h = self.dimreduction(x)
        if mask is None:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        w = mask.float() / mask.float().sum(dim=1, keepdim=True)   # [B, N]
        attention = w.unsqueeze(1)                                  # [B, 1, N]
        logits = self.classifier(torch.bmm(attention, h).squeeze(1))
        if return_attention:
            return logits, attention
        return logits

    def training_loss(self, x, labels, criterion, mask=None):
        logits = self(x, mask=mask)
        return criterion(logits, labels), logits


class MaxPoolMIL(nn.Module):
    """Feature-wise max over instances.

    The returned "attention" is the share of feature dimensions for which each
    instance is the arg-max, so it is a valid distribution over instances that
    shows which patches drive the pooled representation.
    """

    def __init__(self, conf):
        super().__init__()
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    def forward(self, x, mask=None, return_attention=False):
        h = self.dimreduction(x)                                # [B, N, L], >= 0 after ReLU
        if mask is not None:
            h = h.masked_fill(~mask[:, :, None].bool(), float("-inf"))
        pooled, idx = h.max(dim=1)                              # [B, L]
        logits = self.classifier(pooled)
        if return_attention:
            counts = torch.zeros(h.shape[0], h.shape[1], device=h.device)
            counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))
            attention = (counts / counts.sum(dim=1, keepdim=True)).unsqueeze(1)
            return logits, attention
        return logits

    def training_loss(self, x, labels, criterion, mask=None):
        logits = self(x, mask=mask)
        return criterion(logits, labels), logits


# ---------------------------------------------------------------------------
# Research extension: ACMIL-style multi-branch attention + stochastic top-k masking
# ---------------------------------------------------------------------------
class ACMIL(nn.Module):
    """Multi-branch gated attention with stochastic top-k instance masking.

    Motivation: ABMIL trained on a few hundred slides tends to put almost all of
    its attention mass on a handful of very salient patches, which lets it fit
    the training set from a few patterns and generalise poorly (attention
    over-concentration). Two mechanisms push the model to use more evidence:

    * Multi-branch attention (MBA): ``n_branch`` attention heads share the
      encoder. Each head pools its own slide feature and has its own classifier
      trained with cross-entropy, and a diversity loss (mean pairwise cosine
      similarity between the heads' attention distributions) discourages the
      heads from attending to the same patches.
    * Stochastic top-k instance masking (STKIM): during training only, for each
      head the ``n_masked_patch`` highest-attention patches are found and a random
      ``mask_drop`` fraction of them is removed before the softmax, so the model
      cannot rely exclusively on its current top patches.

    The final prediction comes from a separate slide classifier applied to the
    feature pooled with the head-averaged attention. Inference is deterministic
    (masking is disabled in ``eval()``) and costs the same as ABMIL up to the K
    attention heads.

    Setting ``n_branch=1`` gives STKIM only; ``n_masked_patch=0`` or
    ``mask_drop=0`` gives MBA only; both together recover ABMIL plus an extra
    (redundant) classifier head.
    """

    def __init__(self, conf, n_branch=5, n_masked_patch=10, mask_drop=0.6, div_weight=1.0):
        super().__init__()
        self.n_branch = int(n_branch)
        self.n_masked_patch = int(n_masked_patch)
        self.mask_drop = float(mask_drop)
        self.div_weight = float(div_weight)
        self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)
        self.attention = AttentionGated(conf.D_inner, conf.D_attn, K=self.n_branch)
        self.branch_classifiers = nn.ModuleList(
            [Classifier1FC(conf.D_inner, conf.n_class, conf.droprate) for _ in range(self.n_branch)]
        )
        self.slide_classifier = Classifier1FC(conf.D_inner, conf.n_class, conf.droprate)

    # -- STKIM -------------------------------------------------------------
    def _stochastic_topk_mask(self, a_logits):
        """Randomly hide part of each head's top-k patches (training only)."""
        B, K, N = a_logits.shape
        k = min(self.n_masked_patch, N)
        n_drop = int(k * self.mask_drop)
        if n_drop <= 0:
            return a_logits
        topk_idx = torch.topk(a_logits.detach(), k, dim=-1).indices          # [B, K, k]
        pick = torch.rand(B, K, k, device=a_logits.device).argsort(dim=-1)[..., :n_drop]
        drop_idx = torch.gather(topk_idx, -1, pick)                          # [B, K, n_drop]
        drop = torch.zeros_like(a_logits, dtype=torch.bool).scatter_(
            -1, drop_idx, torch.ones_like(drop_idx, dtype=torch.bool))
        masked = a_logits.masked_fill(drop, float("-inf"))
        # Never mask every valid patch of a head (possible only with padding).
        all_hidden = torch.isinf(masked).all(dim=-1, keepdim=True)
        return torch.where(all_hidden, a_logits, masked)

    # -- forward -----------------------------------------------------------
    def forward_all(self, x, mask=None):
        h = self.dimreduction(x)                                  # [B, N, L]
        a_logits = mask_logits(self.attention(h), mask)           # [B, K, N]
        if self.training and self.n_masked_patch > 0 and self.mask_drop > 0:
            a_logits = self._stochastic_topk_mask(a_logits)
        branch_attention = F.softmax(a_logits, dim=-1)            # [B, K, N]
        branch_feat = torch.bmm(branch_attention, h)              # [B, K, L]
        branch_logits = torch.stack(
            [clf(branch_feat[:, k]) for k, clf in enumerate(self.branch_classifiers)], dim=1
        )                                                         # [B, K, C]
        attention = branch_attention.mean(dim=1, keepdim=True)    # [B, 1, N]
        slide_feat = torch.bmm(attention, h).squeeze(1)           # [B, L]
        logits = self.slide_classifier(slide_feat)                # [B, C]
        return {
            "logits": logits,
            "attention": attention,
            "branch_logits": branch_logits,
            "branch_attention": branch_attention,
        }

    def forward(self, x, mask=None, return_attention=False):
        out = self.forward_all(x, mask=mask)
        if return_attention:
            return out["logits"], out["attention"]
        return out["logits"]

    # -- objective -----------------------------------------------------------
    @staticmethod
    def attention_diversity(branch_attention):
        """Mean pairwise cosine similarity between the heads' attention maps."""
        K = branch_attention.shape[1]
        if K < 2:
            return branch_attention.new_zeros(())
        a = F.normalize(branch_attention, dim=-1)
        sim = torch.bmm(a, a.transpose(1, 2))                     # [B, K, K]
        iu = torch.triu_indices(K, K, offset=1, device=a.device)
        return sim[:, iu[0], iu[1]].mean()

    def training_loss(self, x, labels, criterion, mask=None, return_parts=False):
        out = self.forward_all(x, mask=mask)
        B, K, C = out["branch_logits"].shape
        loss_slide = criterion(out["logits"], labels)
        loss_branch = criterion(out["branch_logits"].reshape(B * K, C), labels.repeat_interleave(K))
        loss_div = self.attention_diversity(out["branch_attention"])
        loss = loss_slide + loss_branch + self.div_weight * loss_div
        if return_parts:
            parts = {"slide": loss_slide.item(), "branch": loss_branch.item(), "div": loss_div.item()}
            return loss, out["logits"], parts
        return loss, out["logits"]


# ---------------------------------------------------------------------------
# Registry used by run_experiments.py
# ---------------------------------------------------------------------------
MODEL_VARIANTS = {
    # name: (class, kwargs, description)
    "abmil": (ABMIL, {}, "ABMIL baseline (gated attention, K=1)"),
    "meanpool": (MeanPoolMIL, {}, "Encoder + mean pooling + classifier"),
    "maxpool": (MaxPoolMIL, {}, "Encoder + max pooling + classifier"),
    "acmil": (ACMIL, dict(n_branch=5, n_masked_patch=10, mask_drop=0.6, div_weight=1.0),
              "Extension: multi-branch attention + diversity loss + STKIM"),
    "acmil_mba": (ACMIL, dict(n_branch=5, n_masked_patch=0, mask_drop=0.0, div_weight=1.0),
                  "Ablation: multi-branch attention + diversity loss only"),
    "acmil_stkim": (ACMIL, dict(n_branch=1, n_masked_patch=10, mask_drop=0.6, div_weight=0.0),
                    "Ablation: stochastic top-k masking only (single branch)"),
}


def build_model(name, conf):
    if name not in MODEL_VARIANTS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODEL_VARIANTS)}")
    cls, kwargs, _ = MODEL_VARIANTS[name]
    return cls(conf, **kwargs)
