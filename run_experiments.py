"""Run the Project A experiments: baseline, aggregator ablations and the extension.

Every (dataset, model, seed) run uses the same features, the same
train/validation/test split (fixed by ``seed`` in the YAML config, 42), the
same optimiser, epoch budget and model-selection rule as the notebook. Only
the training seed (initialisation + shuffling + STKIM randomness) changes
between seeds.

Examples
--------
# Everything reported in the paper (about 1-2 h on a Colab T4 GPU):
python run_experiments.py --datasets C16 C17 BRACS \
    --models abmil meanpool maxpool acmil acmil_mba acmil_stkim --seeds 0 1 2 3 4

# Quick smoke test:
python run_experiments.py --datasets C16 --models abmil acmil --seeds 0 --epochs 2

# Features stored outside the project folder (e.g. Colab local disk):
python run_experiments.py --feature_root /content/feature ...

Finished runs are skipped (resume after a disconnect); use --overwrite to redo.
Then run:  python summarize_results.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from dataloader import build_HDF5_feat_dataset, mil_collate_fn, _labels_for_names  # noqa: E402
from utils import load_config, seed_everything, infer_feature_dim  # noqa: E402
from models import MODEL_VARIANTS, build_model  # noqa: E402
from engine import CachedFeatureDataset, GPUBagLoader, evaluate, fit, summarize_attention  # noqa: E402

CONFIG_FILES = {
    "C16": "camelyon16_medical_ssl_config.yml",
    "C17": "camelyon17_medical_ssl_config.yml",
    "BRACS": "bracs_medical_ssl_config.yml",
}
H5_NAME = "patch_feats_pretrain_medical_ssl.h5"


def resolve(path, base):
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def load_dataset_config(ds, args):
    conf = load_config(os.path.join(args.project_dir, "config", CONFIG_FILES[ds]))
    if args.feature_root:
        conf.data_dir = os.path.join(args.feature_root, ds, H5_NAME)
    else:
        conf.data_dir = resolve(conf.data_dir, args.project_dir)
    conf.dataset_csv_dir = resolve(getattr(conf, "dataset_csv_dir", "./dataset_csv"), args.project_dir)
    if args.epochs is not None:
        conf.train_epoch = args.epochs
    return conf


def make_loaders(train_data, val_data, test_data, conf, seed, device):
    if getattr(train_data, "device", torch.device("cpu")).type == "cuda":
        g = torch.Generator()
        g.manual_seed(seed)
        return (GPUBagLoader(train_data, shuffle=True, generator=g),
                GPUBagLoader(val_data), GPUBagLoader(test_data))
    pin = bool(conf.pin_memory) and device.type == "cuda"
    common = dict(batch_size=int(conf.B), num_workers=int(conf.n_worker),
                  pin_memory=pin, collate_fn=mil_collate_fn)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_data, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_data, shuffle=False, **common)
    test_loader = DataLoader(test_data, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def class_counts(splits, conf):
    out = {}
    for name, d in zip(("train", "val", "test"), splits):
        if hasattr(d, "items"):
            labels = [it["label"] for it in d.items]
        else:
            labels = _labels_for_names(conf.data_dir, d.data_names, conf)
        out[name] = [labels.count(c) for c in range(int(conf.n_class))]
    return out


def run_one(ds, model_name, seed, conf, splits, device, args):
    out_dir = os.path.join(args.out_dir, ds, model_name, f"seed{seed}")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path) and not args.overwrite:
        print(f"  [skip] {ds}/{model_name}/seed{seed} already finished")
        return
    os.makedirs(out_dir, exist_ok=True)
    print(f"  [run ] {ds} | {model_name} | seed {seed}")

    seed_everything(seed)
    train_loader, val_loader, test_loader = make_loaders(*splits, conf, seed, device)
    model = build_model(model_name, conf).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.lr), weight_decay=float(conf.wd))

    t0 = time.time()
    result = fit(model, criterion, optimizer, train_loader, val_loader, device,
                 int(conf.n_class), int(conf.train_epoch), verbose=not args.quiet)
    train_time = time.time() - t0

    model.load_state_dict(result["best_state"])
    test_metrics, rows = evaluate(model, criterion, test_loader, device, int(conf.n_class), return_details=True)
    best = result["history"][result["best_epoch"]]

    payload = {
        "dataset": ds,
        "model": model_name,
        "model_description": MODEL_VARIANTS[model_name][2],
        "model_kwargs": MODEL_VARIANTS[model_name][1],
        "seed": seed,
        "split_seed": int(conf.seed),
        "n_train": len(splits[0]), "n_val": len(splits[1]), "n_test": len(splits[2]),
        "class_counts": class_counts(splits, conf),
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "epochs": int(conf.train_epoch),
        "best_epoch": result["best_epoch"] + 1,
        "val": {"auc": best["val_auc"], "f1": best["val_f1"], "acc": best["val_acc"], "loss": best["val_loss"]},
        "test": test_metrics,
        "test_attention": summarize_attention(rows),
        "train_time_s": train_time,
        "device": str(device),
        "data_pipeline": "gpu_cache" if isinstance(train_loader, GPUBagLoader) else
                         ("ram_cache" if hasattr(splits[0], "items") else "hdf5"),
        "config": {k: v for k, v in vars(conf).items() if isinstance(v, (int, float, str, bool))},
    }
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(result["history"], f, indent=1)
    with open(os.path.join(out_dir, "test_predictions.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if args.save_ckpt:
        torch.save(result["best_state"], os.path.join(out_dir, "best_model.pt"))
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"         test auc={test_metrics['auc']:.4f} acc={test_metrics['acc']:.2f} "
          f"f1={test_metrics['f1']:.4f} | best epoch {result['best_epoch'] + 1} | {train_time:.0f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", default=["C16", "C17", "BRACS"], choices=list(CONFIG_FILES))
    p.add_argument("--models", nargs="+", default=["abmil", "acmil"], choices=list(MODEL_VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=None, help="override train_epoch from the YAML")
    p.add_argument("--project_dir", default=HERE)
    p.add_argument("--feature_root", default=None,
                   help="folder holding C16/, C17/, BRACS/ with the HDF5 files (overrides data_dir)")
    p.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no_cache", action="store_true", help="read HDF5 from disk every epoch")
    p.add_argument("--cache_dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--no_gpu_cache", action="store_true",
                   help="keep the feature cache in CPU RAM instead of GPU memory")
    p.add_argument("--no_ckpt", dest="save_ckpt", action="store_false")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true", help="do not print per-epoch logs")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device} | torch {torch.__version__}")
    for ds in args.datasets:
        conf = load_dataset_config(ds, args)
        print(f"\n=== {ds}: {conf.data_dir}")
        if not os.path.exists(conf.data_dir):
            print(f"  feature file not found, skipping {ds}. Download from https://zenodo.org/records/23130543")
            continue
        train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)
        conf.D_feat = infer_feature_dim(train_data)
        print(f"  split sizes: train {len(train_data)}, val {len(val_data)}, test {len(test_data)} | D_feat {conf.D_feat}")
        if not args.no_cache:
            dtype = getattr(torch, args.cache_dtype)
            cache_device = device if (device.type == "cuda" and not args.no_gpu_cache) else torch.device("cpu")
            print(f"  caching features as {args.cache_dtype} on {cache_device}")
            train_data, val_data, test_data = (CachedFeatureDataset(d, dtype, cache_device)
                                               for d in (train_data, val_data, test_data))
        for model_name in args.models:
            for seed in args.seeds:
                run_one(ds, model_name, seed, conf, (train_data, val_data, test_data), device, args)
        del train_data, val_data, test_data


if __name__ == "__main__":
    main()
