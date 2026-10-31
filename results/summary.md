# Project A results

Test metrics, mean ± std over training seeds (same split, features and protocol for every method).

## CAMELYON16

| Method | seeds | Test AUC | Test ACC (%) | Test Macro-F1 | Val AUC | best epoch | attn entropy | top-1% mass |
|---|---|---|---|---|---|---|---|---|
| ABMIL (baseline) | 5 | 0.914 ± 0.034 | 88.37 ± 3.59 | 0.869 ± 0.044 | 1.000 ± 0.000 | 2.6 | 0.84 ± 0.04 | 0.26 ± 0.05 |
| Mean pooling | 5 | 0.637 ± 0.042 | 67.75 ± 1.95 | 0.607 ± 0.031 | 0.923 ± 0.008 | 23.2 | 1.00 ± 0.00 | 0.01 ± 0.00 |
| Max pooling | 5 | 0.944 ± 0.009 | 88.22 ± 2.01 | 0.867 ± 0.025 | 1.000 ± 0.000 | 6.4 | 0.59 ± 0.00 | 0.58 ± 0.01 |
| ACMIL (MBA + STKIM) | 5 | 0.918 ± 0.032 | 85.74 ± 3.82 | 0.849 ± 0.038 | 1.000 ± 0.000 | 4.8 | 0.89 ± 0.02 | 0.23 ± 0.04 |
| MBA only | 5 | 0.915 ± 0.049 | 86.51 ± 3.22 | 0.851 ± 0.039 | 1.000 ± 0.000 | 4.4 | 0.89 ± 0.02 | 0.21 ± 0.05 |
| STKIM only | 5 | 0.897 ± 0.029 | 86.51 ± 4.70 | 0.852 ± 0.050 | 1.000 ± 0.000 | 2.6 | 0.83 ± 0.03 | 0.28 ± 0.04 |

## CAMELYON17

| Method | seeds | Test AUC | Test ACC (%) | Test Macro-F1 | Val AUC | best epoch | attn entropy | top-1% mass |
|---|---|---|---|---|---|---|---|---|
| ABMIL (baseline) | 5 | 0.855 ± 0.012 | 77.20 ± 3.51 | 0.536 ± 0.027 | 0.879 ± 0.014 | 14.2 | 0.77 ± 0.02 | 0.33 ± 0.02 |
| Mean pooling | 5 | 0.798 ± 0.016 | 71.30 ± 2.66 | 0.460 ± 0.035 | 0.541 ± 0.014 | 18.6 | 1.00 ± 0.00 | 0.01 ± 0.00 |
| Max pooling | 5 | 0.874 ± 0.006 | 84.20 ± 0.91 | 0.556 ± 0.040 | 0.888 ± 0.025 | 23.6 | 0.59 ± 0.00 | 0.53 ± 0.00 |
| ACMIL (MBA + STKIM) | 5 | 0.872 ± 0.006 | 75.70 ± 7.72 | 0.612 ± 0.030 | 0.894 ± 0.021 | 24.6 | 0.81 ± 0.03 | 0.34 ± 0.05 |
| MBA only | 5 | 0.884 ± 0.013 | 83.20 ± 2.41 | 0.639 ± 0.061 | 0.898 ± 0.042 | 19.2 | 0.84 ± 0.04 | 0.30 ± 0.08 |
| STKIM only | 5 | 0.848 ± 0.007 | 79.90 ± 2.84 | 0.551 ± 0.041 | 0.880 ± 0.012 | 14.6 | 0.79 ± 0.03 | 0.30 ± 0.05 |

## BRACS

| Method | seeds | Test AUC | Test ACC (%) | Test Macro-F1 | Val AUC | best epoch | attn entropy | top-1% mass |
|---|---|---|---|---|---|---|---|---|
| ABMIL (baseline) | 5 | 0.807 ± 0.010 | 43.26 ± 2.52 | 0.340 ± 0.018 | 0.831 ± 0.016 | 6.8 | 0.73 ± 0.03 | 0.46 ± 0.04 |
| Mean pooling | 5 | 0.661 ± 0.007 | 33.49 ± 1.91 | 0.249 ± 0.019 | 0.738 ± 0.003 | 23.0 | 1.00 ± 0.00 | 0.01 ± 0.00 |
| Max pooling | 5 | 0.765 ± 0.006 | 42.56 ± 2.55 | 0.342 ± 0.027 | 0.786 ± 0.013 | 22.0 | 0.60 ± 0.00 | 0.51 ± 0.01 |
| ACMIL (MBA + STKIM) | 5 | 0.811 ± 0.014 | 46.28 ± 2.24 | 0.382 ± 0.032 | 0.825 ± 0.008 | 14.2 | 0.75 ± 0.04 | 0.45 ± 0.05 |
| MBA only | 5 | 0.809 ± 0.016 | 46.28 ± 1.91 | 0.398 ± 0.012 | 0.836 ± 0.007 | 13.0 | 0.76 ± 0.04 | 0.44 ± 0.06 |
| STKIM only | 5 | 0.804 ± 0.013 | 42.09 ± 1.91 | 0.359 ± 0.015 | 0.827 ± 0.016 | 11.8 | 0.65 ± 0.04 | 0.57 ± 0.06 |

**ACMIL (MBA + STKIM) − ABMIL (paired by seed), test AUC:** CAMELYON16 +0.004 ± 0.056 (3/5 seeds better); CAMELYON17 +0.017 ± 0.015 (5/5 seeds better); BRACS +0.005 ± 0.023 (3/5 seeds better)

**MBA only − ABMIL (paired by seed), test AUC:** CAMELYON16 +0.001 ± 0.077 (3/5 seeds better); CAMELYON17 +0.029 ± 0.005 (5/5 seeds better); BRACS +0.002 ± 0.025 (3/5 seeds better)

**STKIM only − ABMIL (paired by seed), test AUC:** CAMELYON16 -0.016 ± 0.026 (2/5 seeds better); CAMELYON17 -0.007 ± 0.017 (2/5 seeds better); BRACS -0.003 ± 0.019 (3/5 seeds better)

**Mean pooling − ABMIL (paired by seed), test AUC:** CAMELYON16 -0.277 ± 0.066 (0/5 seeds better); CAMELYON17 -0.057 ± 0.026 (0/5 seeds better); BRACS -0.146 ± 0.006 (0/5 seeds better)

**Max pooling − ABMIL (paired by seed), test AUC:** CAMELYON16 +0.030 ± 0.033 (4/5 seeds better); CAMELYON17 +0.019 ± 0.014 (5/5 seeds better); BRACS -0.042 ± 0.014 (0/5 seeds better)
