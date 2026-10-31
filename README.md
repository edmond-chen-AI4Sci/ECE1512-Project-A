# ECE1512 Project A: Attention-based MIL for whole-slide image classification

PyTorch implementation of ABMIL (Ilse et al., 2018) on pre-extracted WSI patch features for
**CAMELYON16**, **CAMELYON17** and **BRACS**, plus a research extension: attention-challenging MIL
(multi-branch gated attention, attention-diversity loss and stochastic top-k instance masking,
re-implemented after Zhang et al., ECCV 2024). The written report is in [`report/`](report/).

## Repository layout

| Path | What it is |
|---|---|
| `ECE1512_ProjectA_Student.ipynb` | Completed course notebook: DataLoaders, ABMIL, loss/optimiser, training/validation loops, test evaluation, attention visualisation, answers to the notebook questions, and Step 9 (the extension, run on one dataset). |
| `models.py` | `ABMIL` (identical to the notebook), aggregator ablations `MeanPoolMIL` / `MaxPoolMIL`, and the extension `ACMIL` with its ablations (`MODEL_VARIANTS`). All models honour the padding mask. |
| `engine.py` | Shared training loop, evaluation (macro-AUC, accuracy, macro-F1), attention-concentration statistics, in-RAM feature cache. |
| `run_experiments.py` | Runs every (dataset, model, seed) with the baseline protocol and writes `results/<dataset>/<model>/seed<k>/`. Resumable. |
| `summarize_results.py` | Aggregates runs into `results/summary.md`, `results/all_runs.csv`, `report/tables/*.tex` and `report/figures/*.pdf`. |
| `smoke_test.py` | Two-minute end-to-end test on synthetic data (unit checks + all scripts). |
| `dataloader.py`, `utils.py`, `wsi_visualization.py`, `config/`, `dataset_csv/` | Provided by the course (unchanged). |
| `report/` | IEEE-style LaTeX report (`main.tex`, `refs.bib`); tables and figures are generated. |

## Setup

```bash
pip install -r requirements.txt          # torch, torchmetrics, h5py, pandas, scikit-learn, pyyaml, matplotlib, openslide
python smoke_test.py                      # should end with "SMOKE TEST PASSED"
```

Download the features from Zenodo (https://zenodo.org/records/23130543, 14.3 GB in total) and place them as

```text
feature/C16/patch_feats_pretrain_medical_ssl.h5
feature/C17/patch_feats_pretrain_medical_ssl.h5
feature/BRACS/patch_feats_pretrain_medical_ssl.h5
```

or pass `--feature_root /path/to/feature` to `run_experiments.py`. For the WSI attention figure, put
`test_067.tif`, `test_068.tif` and `test_068.xml` from https://huggingface.co/datasets/LeonYe/ECE1512-WSI-Feature in `slide/`.

### Google Colab (recommended: free T4 GPU)

```bash
!git clone https://github.com/edmond-chen-AI4Sci/ECE1512-Project-A.git
%cd ECE1512-Project-A
!pip install -q -r requirements.txt
# Download to the local Colab disk (much faster than reading from Google Drive), one dataset at a time.
!for D in C16 C17 BRACS; do mkdir -p /content/feature/$D && \
   wget -q -O /content/$D.zip "https://zenodo.org/records/23130543/files/$D.zip?download=1" && \
   unzip -q -o /content/$D.zip -d /content/feature/$D && rm /content/$D.zip && \
   find /content/feature/$D -name "*.h5" -exec mv {} /content/feature/$D/patch_feats_pretrain_medical_ssl.h5 \; ; done
!ls -lh /content/feature/*
```

## Reproducing the results

```bash
# 3 datasets x 6 methods x 5 seeds = 90 runs of 30 epochs (about 1-2 h on a T4; finished runs are skipped on rerun)
python run_experiments.py --feature_root /content/feature --datasets C16 C17 BRACS \
    --models abmil meanpool maxpool acmil acmil_mba acmil_stkim --seeds 0 1 2 3 4 --quiet
python summarize_results.py
```

If RAM is short (e.g. 12 GB Colab), run one dataset per command (`--datasets C16`, then `C17`, then `BRACS`)
or add `--no_cache` to read the HDF5 file every epoch (slower).

| Method key | Description |
|---|---|
| `abmil` | Baseline: gated attention ABMIL, K = 1 |
| `meanpool`, `maxpool` | Same encoder and classifier, attention replaced by mean / max pooling (component analysis) |
| `acmil` | Extension: 5 attention branches + diversity loss + stochastic top-10 masking (60%) |
| `acmil_mba` | Ablation: branches + diversity loss only |
| `acmil_stkim` | Ablation: masking only (single branch) |

**Protocol (identical for every method):** fixed split (`seed: 42` in the YAML), AdamW (lr 1e-4, weight decay 1e-5),
cross-entropy, batch size 1, 30 epochs, checkpoint selected by validation macro-AUC + macro-F1, single test
evaluation. Seeds 0-4 vary initialisation, shuffling and masking randomness only.

Each run directory contains `metrics.json` (validation metrics at the selected epoch, test metrics, attention statistics,
split sizes, class counts, run time), `history.json` (per-epoch curves), `test_predictions.csv` (per-slide probabilities and
attention statistics) and `best_model.pt` (selected checkpoint).

## Results

Full tables: [`results/summary.md`](results/summary.md); per-run numbers: `results/all_runs.csv`.
Test macro-AUC / macro-F1, mean ± std over 5 seeds (identical splits, features and protocol; one Tesla T4 on Colab,
PyTorch 2.11, 64 GPU-minutes for all 90 runs):

| Method | CAMELYON16 AUC | F1 | CAMELYON17 AUC | F1 | BRACS AUC | F1 |
|---|---|---|---|---|---|---|
| ABMIL (baseline) | 0.914 ± 0.034 | 0.869 | 0.855 ± 0.012 | 0.536 | 0.807 ± 0.010 | 0.340 |
| Mean pooling | 0.637 ± 0.042 | 0.607 | 0.798 ± 0.016 | 0.460 | 0.661 ± 0.007 | 0.249 |
| Max pooling | 0.944 ± 0.009 | 0.867 | 0.874 ± 0.006 | 0.556 | 0.765 ± 0.006 | 0.342 |
| **ACMIL (extension)** | 0.918 ± 0.032 | 0.849 | 0.872 ± 0.006 | 0.612 | 0.811 ± 0.014 | 0.382 |
| MBA only (ablation) | 0.915 ± 0.049 | 0.851 | 0.884 ± 0.013 | 0.639 | 0.809 ± 0.016 | 0.398 |
| STKIM only (ablation) | 0.897 ± 0.029 | 0.852 | 0.848 ± 0.007 | 0.551 | 0.804 ± 0.013 | 0.359 |

Paired by seed, the extension improves CAMELYON17 AUC on 5/5 seeds (+0.017) and macro-F1 on CAMELYON17 (+0.077)
and BRACS (+0.042); its AUC changes on CAMELYON16 (+0.004) and BRACS (+0.005) are within seed noise. The gain comes
from the multi-branch attention; stochastic top-k masking alone does not help. See the report for the analysis.

`ECE1512_ProjectA_Student_executed.ipynb` is the course notebook executed end to end on CAMELYON16 (Colab T4), including
the WSI attention figures saved to `report/figures/`.

## Code reproducibility checklist

This repository and the report are separate artefacts; the report has its own checklist (report appendix A).
Following the ML Code Completeness Checklist / NeurIPS reproducibility guidelines:

- [x] **Dependencies:** `requirements.txt`; tested with Python 3.10+ and PyTorch 2.x (record the exact versions of your runs in the report).
- [x] **Training code:** `run_experiments.py` (all experiments), notebook Steps 1-9 (guided single runs).
- [x] **Evaluation code:** `engine.evaluate`, notebook Steps 5 and 7; metrics from the course-provided `utils.classification_metrics`.
- [x] **Pre-trained models:** `results/<dataset>/<model>/seed<k>/best_model.pt` (load with `models.build_model(name, conf)`).
- [x] **Results with exact commands:** the commands above regenerate every number, table and figure in the report.
- [x] **Data:** public features (Zenodo link above); splits are deterministic from `config/*.yml` and `dataset_csv/*.csv`.
- [x] **Randomness:** `utils.seed_everything(seed)`; cuDNN deterministic mode; per-seed outputs are saved.
- [x] **Tests:** `smoke_test.py` (shapes, attention normalisation, padding invariance, masking, end-to-end scripts).

## References

- M. Ilse, J. Tomczak, M. Welling. Attention-based deep multiple instance learning. ICML 2018.
- Y. Zhang et al. Attention-challenging multiple instance learning for whole slide image classification. ECCV 2024. (The extension is our own re-implementation of its multi-branch attention and stochastic top-k masking for this code base.)
- Course starter code (`dataloader.py`, `utils.py`, `wsi_visualization.py`, configs, notebook skeleton) by the ECE1512 teaching team.
