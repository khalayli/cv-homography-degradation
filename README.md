# cv-homography-degradation

Degradation-aware lightweight matching for robust homography estimation on HPatches.

## What this repo is for
This project compares classical and lightweight image matching pipelines under degradations such as blur, noise, JPEG compression, brightness shifts, and contrast changes. The goal is planar homography estimation with robust fitting.

## Current scope
- Dataset: HPatches full image sequences with ground-truth homographies
- Baseline: ORB first
- Other lightweight matchers (XFeat for now ) : XFeat, JamMa, optional RoMa fallback (heavy matcher)
- Robust fitting: RANSAC ( MAGSAC++ is better but time doesn't allow)
- Evaluation: corner transfer error, thresholded success, runtime

## What is left as TODO
- JamMa integration details  (optional)
- RoMa integration details  (optional)

## Example run
Use the notebook provided.
