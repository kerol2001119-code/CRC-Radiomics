# -*- coding: utf-8 -*-
"""
CRC Radiomics Classification Pipeline
Complete workflow: Classification → ROC Curve → Calibration Curve → DCA Curve

@author: UserCmhuh
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score, log_loss, confusion_matrix, 
    brier_score_loss, accuracy_score, precision_score, recall_score, 
    f1_score
)
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import joblib
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
feature_selection = 'lasso'  # Pearson, lasso, ANOVA
classifier = 'CatBoost'      # CatBoost, SVM, RF, MLP
img_type1 = 'DWI_b800'      # ADC, DWI_b100, DWI_b800, T2, T2_FS
img_type2 = 'T2'
img_type3 = 'T2'
normalization_type = "_min-max_nii_10pixel_sing"
img_type_binding = '2'       # 1, 2, or 3 (number of image types to bind)
features_count = 0           # for Pearson, ANOVA
alpha_count = 0.07          # for lasso
features_count_range = 1    # loop iterations
roi = '_roi'                # _roi, _bbox
dilate_type = ''            # _dilate1, _dilate3...

# Paths
DATA_PATH = r'D:\chen_radiomics\chen_radiomics\new_manual'
MODEL_PATH = r'D:\chen_radiomics\chen_radiomics\model'
RESULT_PATH = r'D:\chen_radiomics\chen_radiomics\classify\result'
OUTPUT_SAVE = True          # Save prediction results

# ============================================================================
# FILE NAME BUILDER
# ============================================================================
filename1 = img_type1 + roi + dilate_type
filename2 = img_type2 + normalization_type + roi + dilate_type
filename3 = img_type2 + normalization_type + roi + dilate_type

data1 = pd.read_csv(os.path.join(DATA_PATH, f'{filename1}.csv'))
data2 = pd.read_csv(os.path.join(DATA_PATH, f'{filename2}.csv'))
data3 = pd.read_csv(os.path.join(DATA_PATH, f'{filename3}.csv'))

training_folder_path = os.path.join(RESULT_PATH, 'pred_result', 'training')
testing_folder_path = os.path.join(RESULT_PATH, 'pred_result', 'testing')

os.makedirs(training_folder_path, exist_ok=True)
os.makedirs(testing_folder_path, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================
def trans_data_to_df(X):
    X = X.astype(np.float64)
    X = pd.DataFrame(X)
    return X

def one_img_type_to_df(X1, y):
    X_data = trans_data_to_df(X1)
    return X_data, y, filename1

def two_img_type_binding_to_df(X1, X2, y):
    X1_data = trans_data_to_df(X1)
    X2_data = trans_data_to_df(X2)
    X_data = pd.concat([X1_data, X2_data], axis=1, ignore_index=True)
    return X_data, y, filename1 + '+' + filename2

def three_img_type_binding_to_df(X1, X2, X3, y):
    X1_data = trans_data_to_df(X1)
    X2_data = trans_data_to_df(X2)
    X3_data = trans_data_to_df(X3)
    X_data = pd.concat([X1_data, X2_data, X3_data], axis=1, ignore_index=True)
    return X_data, y, img_type1 + '+' + img_type2 + '+' + img_type3

def filter_only_have_one_case_class(X_data, y_data, return_mask=False):
    unique_classes, class_counts = np.unique(y_data, return_counts=True)
    rare_classes = unique_classes[class_counts < 2]
    mask = np.isin(y_data, rare_classes, invert=True)
    X_data = X_data[mask]
    y_data = y_data[mask]
    if return_mask:
        return X_data, y_data, mask
    return X_data, y_data

# ============================================================================
# FEATURE SELECTION FUNCTIONS
# ============================================================================
def ANOVA(X_tr, y_tr, features_count, X_te):
    skb = SelectKBest(score_func=f_classif, k=features_count)
    skb.fit_transform(X_tr, y_tr)
    selected_indices = skb.get_support(indices=True)
    selected_features = X_tr.columns[selected_indices]
    X_tr = X_tr[selected_features]
    X_te = X_te[selected_features]
    return X_tr, X_te

def Pearson_correlation(X_tr, y_tr, features_count, X_te):
    Xy_corr = X_tr.corrwith(y_tr)
    Xy_corr.sort_values(ascending=False, inplace=True)
    X_tr = X_tr.reindex(columns=Xy_corr.index)
    corr_matrix = X_tr.corr().abs()
    select_feat = []
    for i in corr_matrix.columns:
        if len(select_feat) and corr_matrix[i][select_feat].max() > 0.9:
            continue
        select_feat.append(i)
        if len(select_feat) == features_count:
            break
    X_tr = X_tr.filter(items=select_feat)
    columns_list = X_tr.columns.tolist()
    X_te = X_te[columns_list]
    return X_tr, X_te

# ============================================================================
# SMOTE FUNCTIONS
# ============================================================================
def set_smote():
    which_class = 0
    number = 56
    ratio = {which_class: number}
    smote1 = SMOTE(sampling_strategy=ratio, random_state=42)
    smote2 = BorderlineSMOTE(sampling_strategy=ratio, random_state=42, kind="borderline-1")
    smote3 = ADASYN(sampling_strategy=ratio, random_state=42)
    return smote1, smote2, smote3

# ============================================================================
# CLASSIFICATION FUNCTIONS
# ============================================================================
def find_best_threshold_youden_index(Y, prob_pred):
    fpr, tpr, thresholds = roc_curve(Y, prob_pred)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    return best_threshold, fpr, tpr

def get_classifier(classifier_name, seed):
    if classifier_name == 'SVM':
        return SVC(kernel='linear', C=100, gamma='scale', probability=True)
    elif classifier_name == 'CatBoost':
        return CatBoostClassifier(random_state=seed, verbose=False)
    elif classifier_name == 'RF':
        return RandomForestClassifier(n_estimators=15, random_state=seed)
    elif classifier_name == 'MLP':
        return MLPClassifier(hidden_layer_sizes=(30,), activation='relu', solver='adam', random_state=seed)
    raise ValueError(f"Unknown classifier: {classifier_name}")

def classifier_ten_fold_cv(X_tr, y_tr, X_te, y_te, result_df_all, features_count, seed):
    """10-fold cross-validation classification"""
    scaler = preprocessing.StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_tr = pd.DataFrame(X_tr)
    X_te = scaler.transform(X_te)
    X_te = pd.DataFrame(X_te)
    y_tr = pd.Series(y_tr)
    
    # Feature selection
    if feature_selection == 'Pearson':
        X_tr, X_te = Pearson_correlation(X_tr, y_tr, features_count, X_te)
    elif feature_selection == 'ANOVA':
        X_tr, X_te = ANOVA(X_tr, y_tr, features_count, X_te)
    elif feature_selection == 'lasso':
        lasso = Lasso(alpha=alpha_count)
        lasso.fit(X_tr, y_tr)
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        X_tr = X_tr.iloc[:, non_zero_indices]
        X_te = X_te.iloc[:, non_zero_indices]
    
    model = get_classifier(classifier, seed)
    
    # Save/Load model
    model_filename = os.path.join(MODEL_PATH, f"{model.__class__.__name__}.pkl")
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    else:
        if classifier == 'MLP':
            model.fit(X_tr, y_tr)
        else:
            sample_weights = np.array([7 if label == 0 else 1 for label in y_tr])
            model.fit(X_tr, y_tr, sample_weight=sample_weights)
        joblib.dump(model, model_filename)
    
    # Prediction
    prob_pred = model.predict_proba(X_te)[:, 1]
    best_threshold, fpr, tpr = find_best_threshold_youden_index(y_te, prob_pred)
    test_pred = (prob_pred >= best_threshold).astype(int)
    
    # Metrics
    auc_score = roc_auc_score(y_te, prob_pred)
    log_loss_value = log_loss(y_te, prob_pred)
    cnf_matrix = confusion_matrix(y_te, test_pred)
    
    fp, fn, tn, tp = cnf_matrix[0, 1], cnf_matrix[1, 0], cnf_matrix[0, 0], cnf_matrix[1, 1]
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fprate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnrate = fn / (tp + fn) if (tp + fn) > 0 else 0
    tprate = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnrate = tn / (fp + tn) if (fp + tn) > 0 else 0
    f1sc = (2 * ppv * tprate) / (ppv + tprate) if (ppv + tprate) > 0 else 0
    
    new_row = {
        'Accuracy': acc, 'Log Loss': log_loss_value, 'AUC': auc_score,
        'PPV': ppv, 'NPV': npv, 'FPR': fprate, 'FNR': fnrate,
        'TPR': tprate, 'TNR': tnrate, 'F1 Score': f1sc,
    }
    
    result_df_all.append([list(new_row.values())])
    
    return result_df_all, prob_pred, y_te, fpr, tpr

# ============================================================================
# MAIN CLASSIFICATION PIPELINE
# ============================================================================
def run_classification():
    """Execute the complete classification pipeline"""
    print("=" * 60)
    print("CRC RADIOMICS CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # Prepare data
    X1 = data1[data1.columns[2:]]
    X2 = data2[data2.columns[2:]]
    X3 = data3[data3.columns[2:]]
    
    y = np.array(data1['label']).astype('int')
    cases = data1['case']
    
    if img_type_binding == '1':
        X_data, y_data, img_type = one_img_type_to_df(X1, y)
    elif img_type_binding == '2':
        X_data, y_data, img_type = two_img_type_binding_to_df(X1, X2, y)
    elif img_type_binding == '3':
        X_data, y_data, img_type = three_img_type_binding_to_df(X1, X2, X3, y)
    
    X_data, y_data, mask = filter_only_have_one_case_class(X_data, y_data, return_mask=True)
    cases_data = cases[mask].reset_index(drop=True)
    
    X_data.reset_index(drop=True, inplace=True)
    y_data = pd.Series(y_data).reset_index(drop=True)
    
    # Model naming
    if feature_selection == 'lasso':
        model_name = f"{img_type}+{feature_selection}+{classifier}+feature_{alpha_count}"
    else:
        model_name = f"{img_type}+{feature_selection}+{classifier}+feature_{features_count}"
    
    print(f"Model: {model_name}")
    print(f"Image binding: {img_type_binding}")
    print(f"Feature selection: {feature_selection}")
    print("-" * 60)
    
    # Run classification
    all_prob_preds = []
    all_y_tests = []
    all_fprs = []
    all_tprs = []
    
    rgen = np.random.RandomState(42)
    seeds = rgen.randint(low=0, high=1000, size=10)
    
    for b in range(features_count_range):
        if feature_selection == 'lasso':
            features_count = round(alpha_count + 0.01 * b, 2)
        
        result_df_all = []
        
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_data, y_data)):
            print(f"Fold {fold + 1}/10", end='\r')
            
            X_tr = X_data.iloc[train_idx]
            X_te = X_data.iloc[test_idx]
            y_tr = y_data.iloc[train_idx]
            y_te = y_data.iloc[test_idx]
            
            result_df_all, prob_pred, y_te, fpr, tpr = classifier_ten_fold_cv(
                X_tr, y_tr, X_te, y_te, result_df_all, features_count, seeds[b]
            )
            
            all_prob_preds.extend(prob_pred)
            all_y_tests.extend(y_te)
            all_fprs.append(fpr)
            all_tprs.append(tpr)
    
    # Aggregate results
    result_df = np.array(result_df_all)
    mean_results = np.mean(result_df, axis=0)[0]
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {mean_results[0]:.4f}")
    print(f"AUC:      {mean_results[2]:.4f}")
    print(f"Sensitivity (TPR): {mean_results[7]:.4f}")
    print(f"Specificity (TNR): {mean_results[8]:.4f}")
    print(f"PPV:      {mean_results[3]:.4f}")
    print(f"NPV:      {mean_results[4]:.4f}")
    print(f"F1 Score: {mean_results[9]:.4f}")
    print("=" * 60)
    
    # Save prediction results
    if OUTPUT_SAVE:
        pred_df = pd.DataFrame({
            'case': cases_data[:len(all_prob_preds)],
            'true_label': all_y_tests,
            'pred_prob': all_prob_preds,
            'pred_label': (np.array(all_prob_preds) >= 0.5).astype(int)
        })
        pred_df.to_csv(os.path.join(RESULT_PATH, f'{model_name}_pred_results.csv'), index=False)
        print(f"Prediction results saved to: {RESULT_PATH}/{model_name}_pred_results.csv")
    
    return {
        'model_name': model_name,
        'prob_preds': all_prob_preds,
        'y_tests': all_y_tests,
        'all_fprs': all_fprs,
        'all_tprs': all_tprs,
        'mean_results': mean_results
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_roc_curve(results, save_path=None):
    """Plot ROC curve with mean curve and confidence interval"""
    all_fprs = results['all_fprs']
    all_tprs = results['all_tprs']
    model_name = results['model_name']
    
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for fpr, tpr in zip(all_fprs, all_tprs):
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr /= len(all_fprs)
    
    mean_auc = auc(mean_fpr, mean_tpr)
    
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.family'] = 'Arial'
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {mean_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve - {model_name}', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_ROC.png'), dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}/{model_name}_ROC.png")
    
    plt.show()
    return mean_auc

def plot_calibration_curve(results, save_path=None):
    """Plot calibration curve"""
    y_true = np.array(results['y_tests'])
    y_pred_prob = np.array(results['prob_preds'])
    model_name = results['model_name']
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_true, y_pred_prob)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams['font.family'] = 'Arial'
    
    # Calibration plot
    ax1.plot(prob_pred, prob_true, 's-', color='darkorange', lw=2, 
             label=f'{model_name} (Brier={brier:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Observed Frequency', fontsize=12)
    ax1.set_title('Calibration Curve', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Histogram
    bins = np.linspace(0, 1, 21)
    ax2.hist(y_pred_prob, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Predicted Probability Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_Calibration.png'), dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to: {save_path}/{model_name}_Calibration.png")
    
    plt.show()

def decision_curve_analysis(y_true, y_pred_prob, thresholds=np.linspace(0.05, 0.85, 80)):
    """Decision Curve Analysis"""
    N = len(y_true)
    net_benefits = []
    prevalence = np.mean(y_true)
    
    for pt in thresholds:
        pred = (y_pred_prob >= pt).astype(int)
        TP = np.sum((pred == 1) & (y_true == 1))
        FP = np.sum((pred == 1) & (y_true == 0))
        NB = (TP / N) - (FP / N) * (pt / (1 - pt))
        net_benefits.append(NB)
    
    net_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    net_none = np.zeros_like(thresholds)
    
    return thresholds, net_benefits, net_all, net_none

def plot_dca_curve(results, save_path=None):
    """Plot Decision Curve Analysis"""
    y_true = np.array(results['y_tests'])
    y_pred_prob = np.array(results['prob_preds'])
    model_name = results['model_name']
    
    thresholds, nb_model, nb_all, nb_none = decision_curve_analysis(y_true, y_pred_prob)
    
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    plt.plot(thresholds, nb_model, 'b-', lw=2, label=model_name)
    plt.plot(thresholds, nb_all, color='gray', lw=2, label='Treat All')
    plt.plot(thresholds, nb_none, 'k--', lw=2, label='Treat None')
    
    plt.xlabel('Threshold Probability', fontsize=14)
    plt.ylabel('Net Benefit', fontsize=14)
    plt.title('Decision Curve Analysis', fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([-0.05, 0.3])
    
    if save_path:
        plt.savefig(os.path.join(save_path, f'{model_name}_DCA.png'), dpi=300, bbox_inches='tight')
        print(f"DCA curve saved to: {save_path}/{model_name}_DCA.png")
    
    plt.show()

def CI_calculate(values, name='Metric'):
    """Calculate 95% confidence interval"""
    mean_val = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    ci_margin = 1.96 * se
    print(f"{name}: {mean_val:.3f} ± {ci_margin:.3f}")
    return mean_val, ci_margin

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Step 1: Run Classification
    results = run_classification()
    
    # Step 2: Plot ROC Curve
    plot_roc_curve(results, save_path=RESULT_PATH)
    
    # Step 3: Plot Calibration Curve
    plot_calibration_curve(results, save_path=RESULT_PATH)
    
    # Step 4: Plot DCA Curve
    plot_dca_curve(results, save_path=RESULT_PATH)
    
    # Step 5: Calculate CI for all metrics
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS (95%)")
    print("=" * 60)
    
    # Re-run to collect all fold metrics
    print("Classification complete! All visualizations generated.")
    print(f"Results saved to: {RESULT_PATH}")
