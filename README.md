# cv-homography-degradation

Degradation-aware lightweight matching for robust homography estimation on HPatches.

## What this repo is for
This project compares classical and lightweight image matching pipelines under degradations such as blur, noise, JPEG compression, brightness shifts, and contrast changes. The goal is planar homography estimation with robust fitting.

The codebase is intentionally modular so different teammates can work on data loading, matchers, geometry, and analysis in parallel.

## Current scope
- Dataset: HPatches full image sequences with ground-truth homographies
- Baseline: ORB first
- Other lightweight matchers (XFeat for now ) : XFeat, JamMa, optional RoMa fallback (heavy matcher)
- Robust fitting: RANSAC ( MAGSAC++ is better but time doesn't allow)
- Evaluation: corner transfer error, thresholded success, runtime

## Repo layout
```text
cv-homography-degradation/
  configs/
    default.yaml
  notebooks/
    main_colab.ipynb
  scripts/
    run_experiment.py
    summarize_results.py
  src/
    data/
      hpatches.py
      corruptions.py
    matchers/
      orb.py
      xfeat.py
      jamma.py
      roma.py
      proposed.py
    geom/
      homography.py
      metrics.py
    utils/
      io.py
      timing.py
      seeding.py
  results/
  reports/figures/
```

## What is already supported
- Corruption helpers
- Metric helpers
- Experiment runner 
- Summary script for CSV outputs

## What is left as TODO
- XFeat integration details (optional)
- JamMa integration details  (optional)
- RoMa integration details  (optional)
- Proposed degradation-aware fallback logic

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Example run
```bash
python scripts/run_experiment.py --config configs/default.yaml --limit-pairs 10
python scripts/summarize_results.py --input results/dev_run/metrics.csv
```
