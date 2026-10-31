"""Two-minute end-to-end check of the whole pipeline on synthetic data (CPU is fine).

    python smoke_test.py

1. Unit checks of every model: output shapes, attention normalisation, padding
   invariance (a padded batch of two bags gives the same logits as the two bags
   alone), stochastic top-k masking never hides every patch, and evaluation is
   deterministic.
2. Writes small synthetic HDF5 files with the real slide IDs / splits of
   CAMELYON16, CAMELYON17 and BRACS (random features with a weak class signal)
   into a temporary folder, runs run_experiments.py for 2 epochs with every
   model, and runs summarize_results.py on the output.

Nothing in the project folder is modified; everything is written to a temp dir.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import h5py
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from models import ABMIL, ACMIL, MeanPoolMIL, MaxPoolMIL, MODEL_VARIANTS, build_model  # noqa: E402
from dataloader import mil_collate_fn  # noqa: E402
from utils import Struct  # noqa: E402


def unit_checks():
    torch.manual_seed(0)
    for n_class in (2, 4, 7):
        conf = Struct(D_feat=384, D_inner=256, D_attn=128, n_class=n_class, droprate=0.0)
        for name in MODEL_VARIANTS:
            m = build_model(name, conf).eval()
            x = torch.randn(2, 9, 384)
            logits, attn = m(x, return_attention=True)
            assert logits.shape == (2, n_class), (name, logits.shape)
            assert attn.shape == (2, 1, 9), (name, attn.shape)
            torch.testing.assert_close(attn.sum(-1), torch.ones(2, 1))

            # Padding invariance: bag A (9 patches) and bag B (5 patches) in one padded batch.
            a, b = torch.randn(9, 384), torch.randn(5, 384)
            batch = mil_collate_fn([{"input": a, "label": 0}, {"input": b, "label": 1}])
            with torch.no_grad():
                joint = m(batch["input"], mask=batch["mask"])
                alone = torch.cat([m(a[None]), m(b[None])])
            torch.testing.assert_close(joint, alone, rtol=1e-5, atol=1e-5)

            # Loss is finite and backpropagates.
            m.train()
            loss, lg = m.training_loss(batch["input"], batch["label"], torch.nn.CrossEntropyLoss(), mask=batch["mask"])
            assert torch.isfinite(loss), name
            loss.backward()
            assert all(p.grad is not None for p in m.parameters() if p.requires_grad), name

    conf = Struct(D_feat=384, D_inner=256, D_attn=128, n_class=2, droprate=0.0)
    acmil = ACMIL(conf, n_branch=5, n_masked_patch=10, mask_drop=0.6)
    acmil.train()
    for n in (1, 2, 3, 10, 50):
        a = acmil.attention(acmil.dimreduction(torch.randn(1, n, 384)))
        masked = acmil._stochastic_topk_mask(a)
        hidden = torch.isinf(masked).sum(-1)
        assert (hidden == int(min(10, n) * 0.6)).all(), (n, hidden)
        assert (~torch.isinf(masked)).any(-1).all()
    acmil.eval()
    x = torch.randn(1, 40, 384)
    with torch.no_grad():
        torch.testing.assert_close(acmil(x), acmil(x))
    print("unit checks passed")


def write_synthetic(root):
    rng = np.random.default_rng(0)

    def bag(label, n_class):
        n = int(rng.integers(20, 80))
        feat = rng.normal(size=(n, 384)).astype(np.float32)
        if label > 0:  # a few "lesion" patches carry the class signal
            k = max(1, n // 10)
            feat[:k, label % 384] += 3.0
            feat[:k, (label * 7) % 384] += 2.0
        coords = rng.integers(0, 10_000, size=(n, 2))
        return feat, coords

    def write(path, items, n_class):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with h5py.File(path, "w") as h5:
            for sid, label in items:
                feat, coords = bag(label, n_class)
                g = h5.create_group(sid)
                g.create_dataset("feat", data=feat)
                g.create_dataset("coords", data=coords)
                g.attrs["label"] = int(label)

    name = "patch_feats_pretrain_medical_ssl.h5"
    c16 = [(f"normal_{i:03d}", 0) for i in range(80)] + [(f"tumor_{i:03d}", 1) for i in range(60)] + \
          [(f"test_{i:03d}", int(i % 3 == 0)) for i in range(50)]
    write(os.path.join(root, "C16", name), c16, 2)
    c17 = pd.read_csv(os.path.join(HERE, "dataset_csv", "camelyon17.csv"))
    write(os.path.join(root, "C17", name), list(zip(c17.slide_id, c17.label)), 4)
    bracs = pd.read_csv(os.path.join(HERE, "dataset_csv", "bracs.csv"))
    write(os.path.join(root, "BRACS", name), list(zip(bracs.slide_id, bracs.label)), 7)


def main():
    unit_checks()
    with tempfile.TemporaryDirectory() as tmp:
        feat_root = os.path.join(tmp, "feature")
        write_synthetic(feat_root)
        print("synthetic HDF5 written")
        run = [sys.executable, os.path.join(HERE, "run_experiments.py"), "--feature_root", feat_root,
               "--out_dir", os.path.join(tmp, "results"), "--epochs", "2", "--seeds", "0", "1",
               "--models", *MODEL_VARIANTS, "--device", "cpu"]
        subprocess.run(run, check=True)
        subprocess.run([sys.executable, os.path.join(HERE, "summarize_results.py"),
                        "--results_dir", os.path.join(tmp, "results"),
                        "--report_dir", os.path.join(tmp, "report")], check=True)
        made = sorted(os.listdir(os.path.join(tmp, "report", "tables"))) + \
               sorted(os.listdir(os.path.join(tmp, "report", "figures")))
        print("generated:", made)
    print("\nSMOKE TEST PASSED - the pipeline runs end to end. Now run the real experiments (see README).")


if __name__ == "__main__":
    main()
