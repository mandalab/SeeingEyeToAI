# Eye-Tracking Gaze Similarity Analysis

This repository contains all relevant code, data artifacts, and experiments from our eye-tracking based analysis of model-human gaze similarity across video memorability datasets.

---

## 🗂 Directory Overview

```
VIDEOMEM_EYETRACKING_CODE/
├── config/
├── dataset/
├── experiments/
├── pickles/
├── src/
```

---

## 📁 `config/`

- **default.yaml** – Central configuration file specifying dataset paths, model prediction files, screen resolutions, and runtime parameters for all modules. Feel free to change according to your requirements.

---

## 📁 `dataset/`

This directory is a placeholder our **small-scale eye-tracking dataset**, collected from 10 participants on subsets of videos from the Memento10k and VideoMem datasets. We have not been able to push this to the github repository because of size limitations.  

Each subdirectory corresponds to a participant session:
- `memento_fixations/` – Fixation data for Memento videos.
- `videomem_fixations/` – Fixation data for VideoMem videos.

Each session folder (e.g. `001/`) contains:
- `.evs` and `.res` – Eyelink session files.
- `Output/` – Contains:
  - `*_fixation.xls` – Fixation events.
  - `*_saccade.xls` – Saccade events.
  - `*_sample.xls` – Sample points.

---

## 📁 `experiments/`

Contains **Jupyter notebooks** for exploratory analysis and figure generation:

- `01_visualise_eyetracking_dataset.ipynb`  
  Browse and preview raw fixation files from the `dataset/` directory.

- `02_visualise_heatmaps.ipynb`  
  Load preprocessed heatmaps from pickles and visualize fixation density across frames.

- `03_model_human_gaze_comparision.ipynb`  
  Compute and tabulate model-human similarity metrics (AUC-Judd, NSS, CC, KLD) as shown in the paper.

- `04_gaze_similarity_figures.ipynb`  
  Plot metric scores across memorability bins and frame-wise gaze similarity comparisons using shuffled and off-center controls.

---

## 📁 `pickles/`

Structured as follows:

### `eyetracking_data/`
- Precomputed Gaussian-smoothed heatmaps:
  - `memento_fixation_gBr.pkl`
  - `videomem_fixation_gBr.pkl`

### `model_human_metrics/`
- Frame-level evaluation metrics across different splits:
  - `eyetracking_split_all_*.pkl` – Metrics for all center-cropped videos.
  - `eyetracking_split_nonCenter35_*.pkl` – Metrics for a curated off-center video set.
  - `eyetracking_split_shuffled_*.pkl` – Metrics using shuffled fixation maps.

- Summary metric pickles:
  - `FINAL_*_metrics_gaussianBeforeRescale.pkl` – Model vs. Human similarity metrics.
  - `FINAL_*_percentiles_all.pkl` – Percentile-based evaluation of model predictions.

### `model_predictions/`
- `name2id.pkl` – Mapping from video filenames to integer video IDs for Memento10k.
- `memento.csv` – Contains `video_id`, predicted memorability score (`preds`), and ground-truth memorability (`true`) for Memento10k subset.
- `videomem.csv` – Same structure as above, for VideoMem subset.

> ⚠️ Note: Some files in `model_predictions/` and `eyetracking_data/` have been added to `.gitignore` as they contain true memorability scores from Memento10k and VideoMem, which cannot be shared publicly due to dataset restrictions.

---

## 📁 `src/eyetracking/`

Core processing modules:

- `preprocessing.py` – Loads raw fixation files, aligns gaze coordinates to video dimensions, applies Gaussian blur, and outputs per-frame fixation heatmaps.
- `metrics.py` – Implements gaze similarity metrics (AUC-Judd, NSS, CC, KLD), computes average and percentile-based scores.
- `__init__.py` – Package init file for modular imports.

---