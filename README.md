# CRC Radiomics Classification (DeLong Test)

Classification pipeline for CRC (Colorectal Cancer) Radiomics analysis with DeLong statistical testing.

## Features

- **Feature Selection**: Lasso, Pearson correlation, ANOVA
- **Classifiers**: CatBoost, SVM, Random Forest, MLP
- **Cross-Validation**: 10-fold stratified cross-validation
- **Metrics**: AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1 Score
- **Data Augmentation**: SMOTE, BorderlineSMOTE, ADASYN

## Configuration

Edit the parameters at the top of `classification_for_delong_test.py`:

```python
feature_selection = 'lasso'  # Pearson, lasso, ANOVA
classifier = 'CatBoost'       # CatBoost, SVM, RF, MLP
img_type1 = 'DWI_b800'       # ADC, DWI_b100, DWI_b800, T2, T2_FS
img_type2 = 'T2'
normalization_type = "_min-max_nii_10pixel_sing"
alpha_count = 0.07           # For Lasso feature selection
roi = '_roi'                 # _roi, _bbox
```

## Data Format

- Input CSV files should contain columns: `case`, `label`, and feature columns
- Training/Testing CSV files should contain: `index`, `label` columns

## Usage

### 1. Setup Environment

```batch
build_venv.bat
```

### 2. Run Classification

```bash
python classification_for_delong_test.py
```

## Output

- Prediction results saved to: `datapath/model_name+pred_results.csv`
- Trained models saved to: `D:\chen_radiomics\chen_radiomics\model\`
- Console outputs: AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1 with 95% CI

## Requirements

See `requirements.txt` for dependencies.
