# -*- coding: utf-8 -*-
"""
Created on Sun Sep 14 21:39:41 2025

@author: Naria
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import pandas as pd
from sklearn.metrics import brier_score_loss

df = pd.read_excel(r"C:\Users\Naria\Desktop\DWI_b800_bbox+T2_min-max_nii_10pixel_sing_bbox+lasso+CatBoost+feature_0.06+pred_results.xlsx")
y_true = df["dx"].values

y_preds = {
    "DWI_b800 model": df["DWI_b800"].values,
    "T2WI model": df["T2"].values,
    "DWI_b800+T2WI model": df["DWI_b800+T2"].values
}
plt.figure(figsize=(6,6))

for model_name, y_pred_prob in y_preds.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_true, y_pred_prob)  # 計算 Brier score
    plt.plot(prob_pred, prob_true, "", label=f"{model_name}")

# 畫完美校準線
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Plot")
plt.legend()
plt.grid(True)
plt.show()

# --- 第二張圖：Histogram (樣本分布) ---
# 設定 bin
bins = np.linspace(0, 1, 11)  # 10 個 bin
bin_width = bins[1] - bins[0]
n_models = len(y_preds)

plt.figure(figsize=(7,5))

# 並列 bar 的偏移量
for i, (model_name, y_pred_prob) in enumerate(y_preds.items()):
    counts, _ = np.histogram(y_pred_prob, bins=bins)
    # 計算每個 bar 的 x 位置，並作偏移
    x = bins[:-1] + i * (bin_width / n_models)
    plt.bar(x, counts, width=bin_width / n_models, label=model_name, align='edge')

plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Predicted Probability Distribution")
plt.legend(loc="best")
plt.show()
#%% Multi curve

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd

df = pd.read_excel(r"C:\Users\Naria\Desktop\clinical_pred.xlsx")
y_true = df["dx"].values

y_preds = {
    "clinical model": df["clinical model"].values,
    "Clinical and radiomic binding model": df["binding model"].values,
}

plt.figure(figsize=(6,6))

for model_name, y_pred_prob in y_preds.items():
    prob_true, prob_pred = calibration_curve(y_true, y_pred_prob, n_bins=10, strategy='quantile')
    brier = brier_score_loss(y_true, y_pred_prob)  # 計算 Brier score
    plt.plot(prob_pred, prob_true, "", label=f"{model_name}")

# 畫完美校準線
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Calibration Plot")
plt.legend()
plt.grid(True)
plt.show()

# --- 第二張圖：Histogram (樣本分布) ---
# 設定 bin
bins = np.linspace(0, 1, 11)  # 10 個 bin
bin_width = bins[1] - bins[0]
n_models = len(y_preds)

plt.figure(figsize=(7,5))

# 並列 bar 的偏移量
for i, (model_name, y_pred_prob) in enumerate(y_preds.items()):
    counts, _ = np.histogram(y_pred_prob, bins=bins)
    # 計算每個 bar 的 x 位置，並作偏移
    x = bins[:-1] + i * (bin_width / n_models)
    plt.bar(x, counts, width=bin_width / n_models, label=model_name, align='edge')

plt.xlabel("Predicted probability")
plt.ylabel("Count")
plt.title("Predicted Probability Distribution")
plt.legend(loc="best")
plt.show()
