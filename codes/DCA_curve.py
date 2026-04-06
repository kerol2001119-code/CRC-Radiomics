# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 18:32:48 2025

@author: Naria
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pandas as pd

def decision_curve_analysis(y_true, y_pred_prob, thresholds=np.linspace(0.05, 0.85, 80)):
    N = len(y_true)
    net_benefits = []
    prevalence = np.mean(y_true)
    
    for pt in thresholds:
        pred = (y_pred_prob >= pt).astype(int)
        TP = np.sum((pred == 1) & (y_true == 1))
        FP = np.sum((pred == 1) & (y_true == 0))
        NB = (TP/N) - (FP/N) * (pt/(1-pt))
        net_benefits.append(NB)
    
    # Treat all
    net_all = prevalence - (1-prevalence) * (thresholds/(1-thresholds))
    # Treat none
    net_none = np.zeros_like(thresholds)
    
    return thresholds, net_benefits, net_all, net_none

# 假設有一組真實標籤與模型預測機率

df = pd.read_excel(r"C:\Users\Naria\Desktop\DWI_b800_bbox+T2_min-max_nii_10pixel_sing_bbox+lasso+CatBoost+feature_0.06+pred_results.xlsx")
y_true = df["dx"].values
y_pred_prob = df["DWI_b800+T2"].values

thresholds, nb_model, nb_all, nb_none = decision_curve_analysis(y_true, y_pred_prob)

plt.figure(figsize=(8,6))
plt.plot(thresholds, nb_model, label="DWI_b800+T2WI")
plt.plot(thresholds, nb_all, color="gray", label="Treat All")
plt.plot(thresholds, nb_none, linestyle="--", color="black", label="Treat None")
plt.xlabel("Threshold probability")
plt.ylabel("Net Benefit")
plt.title("Decision Curve Analysis")
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

#%% Multi

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def decision_curve_analysis(y_true, y_pred_prob, thresholds=np.linspace(0.05, 0.85, 80)):
    N = len(y_true)
    net_benefits = []
    prevalence = np.mean(y_true)
    
    for pt in thresholds:
        pred = (y_pred_prob >= pt).astype(int)
        TP = np.sum((pred == 1) & (y_true == 1))
        FP = np.sum((pred == 1) & (y_true == 0))
        NB = (TP/N) - (FP/N) * (pt/(1-pt))
        net_benefits.append(NB)
    
    # Treat all
    net_all = prevalence - (1-prevalence) * (thresholds/(1-thresholds))
    # Treat none
    net_none = np.zeros_like(thresholds)
    
    return thresholds, net_benefits, net_all, net_none

# 讀取 Excel
df = pd.read_excel(r"C:\Users\Naria\Desktop\DCA_curve_0208.xlsx")
y_true = df["dx"].values

models_columns = {
    "ADC": "ADC",
    "DWI_b100": "DWI_b100",
    "DWI_b800": "DWI_b800",
    "T2WI": "T2",
    "FS-T2WI": "T2_FS",
    "DWI_b800+T2WI": "DWI_b800+T2"
}


thresholds = np.linspace(0.05, 0.85, 80)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14  # 可調整字體大小
plt.figure(figsize=(8,6))

for model_name, col_name in models_columns.items():
    y_pred_prob = df[col_name].values
    _, nb_model, nb_all, nb_none = decision_curve_analysis(y_true, y_pred_prob, thresholds)
    plt.plot(thresholds, nb_model, label=f"{model_name}")

# 畫 Treat All 和 Treat None 基準線
plt.plot(thresholds, nb_all, color="gray", label="Treat All")
plt.plot(thresholds, nb_none, linestyle="--", color="black", label="Treat None")

plt.xlabel("Threshold probability", fontname="Arial", fontsize=16)
plt.ylabel("Net benefit", fontname="Arial", fontsize=16)

plt.legend()
plt.tight_layout()
plt.show()
