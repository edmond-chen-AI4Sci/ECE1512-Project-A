"""Aggregate results/ into tables (CSV, Markdown, LaTeX) and figures for the report.

python summarize_results.py                      # reads ./results, writes ./results and ./report
python summarize_results.py --results_dir results --report_dir report

Outputs
  results/all_runs.csv              one row per (dataset, model, seed)
  results/summary.md                mean +- std tables, readable on GitHub
  report/tables/*.tex               tables \\input by report/main.tex
  report/figures/*.pdf (+ .png)     loss curves, validation AUC curves, attention concentration
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["C16", "C17", "BRACS"]
DS_LABEL = {"C16": "CAMELYON16", "C17": "CAMELYON17", "BRACS": "BRACS"}
MODEL_LABEL = {
    "abmil": "ABMIL (baseline)",
    "meanpool": "Mean pooling",
    "maxpool": "Max pooling",
    "acmil": "ACMIL (MBA + STKIM)",
    "acmil_mba": "MBA only",
    "acmil_stkim": "STKIM only",
}
# Categorical slots 1-3 of the validated reference palette, then neutral grays.
COLORS = {"abmil": "#2a78d6", "acmil": "#eb6834", "acmil_mba": "#1baf7a",
          "acmil_stkim": "#6f6e69", "meanpool": "#6f6e69", "maxpool": "#a3a29b"}
METRICS = ["auc", "acc", "f1"]


# ---------------------------------------------------------------------------
def load_runs(results_dir):
    rows, histories, splits = [], {}, {}
    for path in sorted(glob.glob(os.path.join(results_dir, "*", "*", "seed*", "metrics.json"))):
        with open(path) as f:
            m = json.load(f)
        row = {"dataset": m["dataset"], "model": m["model"], "seed": m["seed"],
               "best_epoch": m["best_epoch"], "n_params": m["n_params"], "train_time_s": m["train_time_s"]}
        row.update({f"test_{k}": m["test"][k] for k in METRICS + ["loss"]})
        row.update({f"val_{k}": m["val"][k] for k in ["auc", "f1", "acc", "loss"]})
        row.update({f"attn_{k}": v for k, v in m.get("test_attention", {}).items()})
        rows.append(row)
        splits.setdefault(m["dataset"], {k: m.get(k) for k in ("n_train", "n_val", "n_test", "class_counts")})
        hpath = os.path.join(os.path.dirname(path), "history.json")
        if os.path.exists(hpath):
            with open(hpath) as f:
                histories[(m["dataset"], m["model"], m["seed"])] = pd.DataFrame(json.load(f))
    return pd.DataFrame(rows), histories, splits


def fmt(mean, std, scale=1.0, digits=3):
    return f"{mean * scale:.{digits}f} $\\pm$ {std * scale:.{digits}f}"


def fmt_md(mean, std, scale=1.0, digits=3):
    return f"{mean * scale:.{digits}f} ± {std * scale:.{digits}f}"


def agg(df):
    cols = [c for c in df.columns if c.startswith(("test_", "val_", "attn_")) or c in ("best_epoch",)]
    g = df.groupby(["dataset", "model"])[cols]
    out = g.mean().add_suffix("_mean").join(g.std(ddof=1).fillna(0).add_suffix("_std"))
    out["n_seeds"] = df.groupby(["dataset", "model"]).size()
    out["n_params"] = df.groupby(["dataset", "model"])["n_params"].first()
    return out.reset_index()


def paired_delta(df, model, ref="abmil", metric="test_auc"):
    """Per-seed difference model - ref on the same seed; returns mean, std, wins, n."""
    a = df[df.model == model].set_index(["dataset", "seed"])[metric]
    b = df[df.model == ref].set_index(["dataset", "seed"])[metric]
    d = (a - b).dropna()
    res = {}
    for ds in DATASETS:
        if ds in d.index.get_level_values(0):
            x = d.loc[ds].values
            res[ds] = (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0, int((x > 0).sum()), len(x))
    return res


# ---------------------------------------------------------------------------
def latex_table(summary, models, caption, label, metrics=("auc", "acc", "f1"), delta=None):
    present = [m for m in models if m in set(summary.model)]
    if not present:
        return None
    head = {"auc": "AUC", "acc": "ACC (\\%)", "f1": "Macro-F1"}
    lines = [
        "\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{4pt}", "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{l" + "c" * (len(metrics) * len(DATASETS)) + "}", "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{{len(metrics)}}}{{c}}{{{DS_LABEL[d]}}}" for d in DATASETS) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + i * len(metrics)}-{1 + (i + 1) * len(metrics)}}}" for i in range(len(DATASETS))),
        "Method & " + " & ".join(head[m] for _ in DATASETS for m in metrics) + " \\\\", "\\midrule",
    ]
    for model in present:
        cells = []
        for ds in DATASETS:
            sub = summary[(summary.dataset == ds)]
            r = sub[sub.model == model]
            for met in metrics:
                if r.empty:
                    cells.append("--")
                    continue
                mean, std = float(r[f"test_{met}_mean"].iloc[0]), float(r[f"test_{met}_std"].iloc[0])
                scale, dig = (1.0, 2) if met == "acc" else (1.0, 3)
                s = fmt(mean, std, scale, dig)
                best = sub[sub.model.isin(present)][f"test_{met}_mean"].max()
                if np.isclose(mean, best) and len(present) > 1:
                    s = "\\textbf{" + s + "}"
                cells.append(s)
        lines.append(MODEL_LABEL.get(model, model) + " & " + " & ".join(cells) + " \\\\")
    if delta:
        lines.append("\\midrule")
        for model, d in delta.items():
            if model not in present:
                continue
            cells = []
            for ds in DATASETS:
                if ds in d:
                    mean, std, wins, n = d[ds]
                    cells.append(f"\\multicolumn{{{len(metrics)}}}{{c}}{{$\\Delta$AUC {mean:+.3f} $\\pm$ {std:.3f} ({wins}/{n} seeds better)}}")
                else:
                    cells.append(f"\\multicolumn{{{len(metrics)}}}{{c}}{{--}}")
            lines.append(f"{MODEL_LABEL.get(model, model)} vs.\\ ABMIL & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines)


def latex_attention_table(summary, models):
    present = [m for m in models if m in set(summary.model) and "attn_entropy_norm_mean" in summary]
    if not present:
        return None
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Attention concentration on the test slides (mean over slides, then mean $\\pm$ std over seeds). "
             "$\\tilde H$: entropy normalized by $\\log N$; Top-1\\%: attention mass on the top 1\\% of patches.}",
             "\\label{tab:attention}", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{l" + "cc" * len(DATASETS) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DS_LABEL[d]}}}" for d in DATASETS) + " \\\\",
             "Method & " + " & ".join(["$\\tilde H$ & Top-1\\%"] * len(DATASETS)) + " \\\\", "\\midrule"]
    for model in present:
        cells = []
        for ds in DATASETS:
            r = summary[(summary.dataset == ds) & (summary.model == model)]
            if r.empty or pd.isna(r["attn_entropy_norm_mean"].iloc[0]):
                cells += ["--", "--"]
                continue
            cells.append(fmt(r["attn_entropy_norm_mean"].iloc[0], r["attn_entropy_norm_std"].iloc[0], digits=2))
            cells.append(fmt(r["attn_top1pct_mass_mean"].iloc[0], r["attn_top1pct_mass_std"].iloc[0], digits=2))
        lines.append(MODEL_LABEL.get(model, model) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def latex_splits_table(splits, summary):
    if not splits:
        return None
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Slide-level splits shared by every experiment (split seed 42). Parentheses: slides per class, in label "
             "order CAMELYON16 normal/tumor; CAMELYON17 negative/ITC/micro/macro; BRACS N/PB/UDH/FEA/ADH/DCIS/IC.}",
             "\\label{tab:splits}", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{lccc}", "\\toprule", "Dataset & Train & Val & Test \\\\", "\\midrule"]
    for ds in DATASETS:
        if ds not in splits:
            continue
        sp = splits[ds]
        cc = sp.get("class_counts") or {}
        cell = lambda k: f"{sp[f'n_{k}']} ({'/'.join(map(str, cc[k]))})" if k in cc else str(sp[f"n_{k}"])
        lines.append(f"{DS_LABEL[ds]} & {cell('train')} & {cell('val')} & {cell('test')} \\\\")
    rows = []
    for model in ["abmil", "acmil"]:
        r = summary[summary.model == model]
        if not r.empty:
            vals = [f"{int(r[r.dataset == d].n_params.iloc[0]):,}" if not r[r.dataset == d].empty else "--" for d in DATASETS]
            rows.append(f"{MODEL_LABEL[model].split(' (')[0]} parameters & " + " & ".join(vals) + " \\\\")
    if rows:
        lines += ["\\midrule", " & " + " & ".join(DS_LABEL[d] for d in DATASETS) + " \\\\"] + rows
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def latex_baseline_table(summary):
    r = summary[summary.model == "abmil"]
    if r.empty:
        return None
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{ABMIL baseline: validation metrics at the selected epoch and held-out test metrics "
             "(mean $\\pm$ std over seeds).}",
             "\\label{tab:baseline}", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Metric & " + " & ".join(DS_LABEL[d] for d in DATASETS) + " \\\\", "\\midrule"]
    spec = [("Val AUC", "val_auc", 3), ("Val Macro-F1", "val_f1", 3), ("Test AUC", "test_auc", 3),
            ("Test ACC (\\%)", "test_acc", 2), ("Test Macro-F1", "test_f1", 3),
            ("Test loss", "test_loss", 3), ("Selected epoch", "best_epoch", 1)]
    for name, key, dig in spec:
        cells = []
        for ds in DATASETS:
            x = r[r.dataset == ds]
            cells.append("--" if x.empty else fmt(x[f"{key}_mean"].iloc[0], x[f"{key}_std"].iloc[0], digits=dig))
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    seeds = ", ".join(f"{DS_LABEL[d]}: {int(r[r.dataset == d].n_seeds.iloc[0])}" for d in DATASETS if not r[r.dataset == d].empty)
    lines += ["\\midrule", f"\\multicolumn{{4}}{{l}}{{\\footnotesize Seeds -- {seeds}}} \\\\",
              "\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def markdown_summary(summary, df):
    out = ["# Project A results", "",
           "Test metrics, mean ± std over training seeds (same split, features and protocol for every method).", ""]
    for ds in DATASETS:
        sub = summary[summary.dataset == ds]
        if sub.empty:
            continue
        out += [f"## {DS_LABEL[ds]}", "",
                "| Method | seeds | Test AUC | Test ACC (%) | Test Macro-F1 | Val AUC | best epoch | attn entropy | top-1% mass |",
                "|---|---|---|---|---|---|---|---|---|"]
        for model in MODEL_LABEL:
            r = sub[sub.model == model]
            if r.empty:
                continue
            r = r.iloc[0]
            ent = fmt_md(r.get("attn_entropy_norm_mean", np.nan), r.get("attn_entropy_norm_std", np.nan), digits=2)
            top = fmt_md(r.get("attn_top1pct_mass_mean", np.nan), r.get("attn_top1pct_mass_std", np.nan), digits=2)
            out.append(f"| {MODEL_LABEL[model]} | {int(r.n_seeds)} | {fmt_md(r.test_auc_mean, r.test_auc_std)} | "
                       f"{fmt_md(r.test_acc_mean, r.test_acc_std, digits=2)} | {fmt_md(r.test_f1_mean, r.test_f1_std)} | "
                       f"{fmt_md(r.val_auc_mean, r.val_auc_std)} | {r.best_epoch_mean:.1f} | {ent} | {top} |")
        out.append("")
    for model in ["acmil", "acmil_mba", "acmil_stkim", "meanpool", "maxpool"]:
        d = paired_delta(df, model)
        if d:
            out.append(f"**{MODEL_LABEL[model]} − ABMIL (paired by seed), test AUC:** " + "; ".join(
                f"{DS_LABEL[ds]} {m:+.3f} ± {s:.3f} ({w}/{n} seeds better)" for ds, (m, s, w, n) in d.items()))
            out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
def plot_curves(histories, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b29", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "legend.frameon": False, "pdf.fonttype": 42})

    def band(ax, key, model, color, label, ls="-"):
        hs = [h for (d, m, s), h in histories.items() if d == ds and m == model]
        if not hs:
            return False
        n = min(len(h) for h in hs)
        arr = np.stack([h[key].values[:n] for h in hs])
        x = np.arange(1, n + 1)
        mu, sd = arr.mean(0), arr.std(0)
        ax.plot(x, mu, color=color, lw=1.5, ls=ls, label=label)
        if len(hs) > 1:
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
        return True

    made = []
    # (a) ABMIL baseline loss curves; (b) validation loss ABMIL vs ACMIL.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), constrained_layout=True)
    any_curve = False
    for ax, ds in zip(axes, DATASETS):
        ok1 = band(ax, "train_loss", "abmil", COLORS["abmil"], "ABMIL train")
        ok2 = band(ax, "val_loss", "abmil", COLORS["abmil"], "ABMIL val", ls="--")
        ok3 = band(ax, "val_loss", "acmil", COLORS["acmil"], "ACMIL val", ls="--")
        any_curve |= ok1 or ok2 or ok3
        ax.set_title(DS_LABEL[ds], fontsize=8.5)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=3)
    if any_curve:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(fig_dir, f"loss_curves.{ext}"), dpi=200)
        made.append("loss_curves")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), constrained_layout=True)
    any_curve = False
    for ax, ds in zip(axes, DATASETS):
        for model in ["abmil", "acmil", "acmil_mba", "acmil_stkim"]:
            ls = "-" if model in ("abmil", "acmil") else ":"
            any_curve |= band(ax, "val_auc", model, COLORS[model], MODEL_LABEL[model], ls=ls)
        ax.set_title(DS_LABEL[ds], fontsize=8.5)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Validation macro-AUC")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="outside upper center", ncol=4)
    if any_curve:
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(fig_dir, f"val_auc_curves.{ext}"), dpi=200)
        made.append("val_auc_curves")
    plt.close(fig)
    return made


def plot_attention(df, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if "attn_entropy_norm" not in df:
        return []
    models = [m for m in ["abmil", "acmil_stkim", "acmil_mba", "acmil"] if m in set(df.model)]
    if not models:
        return []
    fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        means, sds = [], []
        for ds in DATASETS:
            x = df[(df.dataset == ds) & (df.model == model)]["attn_entropy_norm"]
            means.append(x.mean() if len(x) else np.nan)
            sds.append(x.std(ddof=1) if len(x) > 1 else 0)
        pos = np.arange(len(DATASETS)) + (i - (len(models) - 1) / 2) * width
        ax.bar(pos, means, width * 0.92, yerr=sds, color=COLORS[model], label=MODEL_LABEL[model],
               error_kw=dict(lw=0.8, capsize=2, ecolor="#52514e"))
    ax.set_xticks(np.arange(len(DATASETS)), [DS_LABEL[d] for d in DATASETS])
    ax.set_ylabel("Normalized attention entropy")
    ax.set_ylim(0, 1)
    fig.legend(*ax.get_legend_handles_labels(), fontsize=6.5, loc="outside upper center", ncol=2)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"attention_entropy.{ext}"), dpi=200)
    plt.close(fig)
    return ["attention_entropy"]


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    p.add_argument("--report_dir", default=os.path.join(HERE, "report"))
    args = p.parse_args()

    df, histories, splits = load_runs(args.results_dir)
    if df.empty:
        print(f"No finished runs under {args.results_dir}. Run run_experiments.py first.")
        return
    tab_dir = os.path.join(args.report_dir, "tables")
    fig_dir = os.path.join(args.report_dir, "figures")
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    df.sort_values(["dataset", "model", "seed"]).to_csv(os.path.join(args.results_dir, "all_runs.csv"), index=False)
    summary = agg(df)
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"), index=False)
    md = markdown_summary(summary, df)
    with open(os.path.join(args.results_dir, "summary.md"), "w") as f:
        f.write(md)

    tables = {
        "baseline.tex": latex_baseline_table(summary),
        "aggregator.tex": latex_table(
            summary, ["abmil", "meanpool", "maxpool"],
            "Aggregator ablation: the encoder and classifier are fixed and only the pooling operator changes "
            "(test metrics, mean $\\pm$ std over seeds).", "tab:aggregator"),
        "extension.tex": latex_table(
            summary, ["abmil", "acmil", "acmil_mba", "acmil_stkim"],
            "Research extension versus the ABMIL baseline and ablations of its two components "
            "(test metrics, mean $\\pm$ std over seeds). $\\Delta$AUC is paired by seed.", "tab:extension",
            delta={m: paired_delta(df, m) for m in ["acmil", "acmil_mba", "acmil_stkim"]}),
        "splits.tex": latex_splits_table(splits, summary),
        "attention.tex": latex_attention_table(summary, ["abmil", "acmil_stkim", "acmil_mba", "acmil"]),
    }
    for name, tex in tables.items():
        if tex:
            with open(os.path.join(tab_dir, name), "w") as f:
                f.write(tex + "\n")
    figs = plot_curves(histories, fig_dir) + plot_attention(df, fig_dir)
    print(md)
    print(f"\nWrote tables {[k for k, v in tables.items() if v]} to {tab_dir}")
    print(f"Wrote figures {figs} to {fig_dir}")


if __name__ == "__main__":
    main()
