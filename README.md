# CRC Radiomics Classification Pipeline

Complete workflow for CRC (Colorectal Cancer) Radiomics analysis: **Classification → ROC Curve → Calibration Curve → DCA Curve**

## Quick Start

```batch
build_venv.bat
python classification_pipeline.py
```

## Project Structure

```
CRC-Radiomics/
├── classification_pipeline.py  # Main pipeline (classification + visualization)
├── classification_for_delong_test.py  # Original DeLong test classification
├── codes/                       # Standalone utility modules
│   ├── classification_10fold.py
│   ├── feature_extraction.py    # PyRadiomics feature extraction
│   ├── dicom_to_nii.py          # DICOM to NIfTI conversion
│   ├── roc_curve.py
│   ├── Calibration_curve.py
│   └── DCA_curve.py
├── requirements.txt
├── build_venv.bat
└── README.md
```

## Main Pipeline (`classification_pipeline.py`)

Integrated workflow with complete visualization output:

| Step | Output |
|------|--------|
| Classification | AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1 |
| ROC Curve | `*_ROC.png` |
| Calibration Curve | `*_Calibration.png` (calibration plot + histogram) |
| DCA Curve | `*_DCA.png` |

## Configuration

Edit parameters at the top of `classification_pipeline.py`:

```python
# Feature Selection
feature_selection = 'lasso'  # Pearson, lasso, ANOVA

# Classifier
classifier = 'CatBoost'      # CatBoost, SVM, RF, MLP

# Image Types
img_type1 = 'DWI_b800'       # ADC, DWI_b100, DWI_b800, T2, T2_FS
img_type2 = 'T2'
img_type_binding = '2'       # 1, 2, or 3

# Lasso Alpha
alpha_count = 0.07

# ROI
roi = '_roi'                 # _roi, _bbox
```

## Features

- **Feature Selection**: Lasso, Pearson correlation, ANOVA
- **Classifiers**: CatBoost, SVM, Random Forest, MLP
- **Cross-Validation**: 10-fold stratified
- **Metrics**: AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1 Score, 95% CI
- **Visualization**: ROC, Calibration, DCA curves

## Data Format

- Input CSV: `case`, `label`, feature columns
- Output: `{model_name}_pred_results.csv`

## Requirements

See `requirements.txt` for dependencies.
