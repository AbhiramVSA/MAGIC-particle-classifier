<p align="center">
  <img src="assets/magic-mark.png" alt="Gamma-ray and hadron particle-shower classifier mark" width="200" />
</p>

<h1 align="center">MAGIC Particle Classifier</h1>

<p align="center">A Gradio prototype that classifies gamma and hadron events from CSV rows with saved neural-network and SVM models.</p>

## What it does

- Reads event features from a CSV file.
- Applies a scaler and the selected model. The scaler behavior depends on the entry point.
- Returns one gamma or hadron label for each row.
- Provides an SVM debug tab in `interface.py`.

The input file must include these columns:

```text
fLength, fWidth, fSize, fConc, fConc1, fAsym, fM3Long, fM3Trans, fAlpha, fDist
```

## Repository contents

| Path | Purpose |
|---|---|
| `Learning_L_Regression/interface.py` | Two-tab prediction and SVM debug interface |
| `Learning_L_Regression/interface2.py` | Single-tab prediction interface |
| `Learning_L_Regression/model.h5` | Saved neural-network model |
| `Learning_L_Regression/svm_model.pkl` | Saved SVM model |
| `Learning_L_Regression/scaler.pkl` | Saved feature scaler |
| `Learning_L_Regression/magic04.data` | Data file |
| `requirements.txt` | Recorded package versions |

## Run locally

The scripts load the model and scaler from fixed paths under `/home/abhirama/`. Edit both scripts to use the files in this checkout.

The current `requirements.txt` includes OS-specific packages, so it does not define a portable `pip install` step. Install compatible versions of Gradio, pandas, NumPy, joblib, scikit-learn, TensorFlow, and imbalanced-learn in your Python environment.

Run the single-tab interface after you fix the paths:

```sh
python Learning_L_Regression/interface2.py
```

`interface.py` uses the saved scaler. It starts with `share=True`, which requests a public Gradio share link. Review that flag before you run it.

`interface2.py` fits a new `StandardScaler` to each uploaded file. It does not use the saved scaler. Each entry point therefore scales its inputs differently. `interface2.py` does not request a share link.

## Scope and limits

This repository contains inference interfaces and saved model files. No script trains the models, and no portable deployment setup exists. The source does not report validation metrics, so this README makes no accuracy claim.
