# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:55:28 2024

@author: UserCmhuh
"""

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import nibabel as nib


csv_path =r'C:/Users/UserCmhuh/Desktop/chen_radiomics/classify/result/best'
file_name = '新增 XLS Worksheet.csv'
file_path = os.path.join(csv_path,file_name)
df = pd.read_csv(file_path)
#%%
# 提取 FPR 和 TPR 數據
fpr_data = df.iloc[1:101, 0].tolist()  # 假設 FPR 數據在 A 列
tpr_data = df.iloc[1:101, 1:7].values  # 假設 TPR 數據在 B 到 G 列

# 提取 ROC 曲線的名稱
roc_curve_names = df.iloc[0, 1:7].tolist()

# 設置字體和圖形尺寸
plt.rcParams['font.family'] = 'Arial'
plt.figure(figsize=(6, 6))

# 繪製每條 ROC 曲線
for i, tpr in enumerate(tpr_data.T):
    label = roc_curve_names[i]
    plt.plot(fpr_data, tpr, lw=3, label=label)

# 繪製對角線
plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')

# 設置坐標軸和標籤
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')
plt.title('ROC Curves for Different Feature Selection \nand Classification Models', fontsize=16, fontweight='bold')

# 設置圖例
plt.legend(loc='lower right', prop={'weight': 'bold', 'size': 11})

# 設置坐標軸樣式
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(axis='both', which='major', labelsize=12, width=2)
ax.grid(False)

# 保存圖像並顯示
# plt.savefig('ROC_SVM.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
CatBoostClassifier=[]
CatBoostClassifier_tr=[]
SVC=[]
SVC_tr=[]

def append_plt_value(i,all_results_df):
    if 'catboost (te)' in all_results_df.index[i]:
        CatBoostClassifier.append(all_results_df.iloc[i])
    elif 'catboost (tr)' in all_results_df.index[i] :
        CatBoostClassifier_tr.append(all_results_df.iloc[i])
    elif 'svm (te)'in  all_results_df.index[i]:
        SVC.append(all_results_df.iloc[i])
    elif 'svm (tr)' in all_results_df.index[i]:
        SVC_tr.append(all_results_df.iloc[i])

for i in range(len(all_results_df)):
    append_plt_value(i,all_results_df)
    
CatBoostClassifier_df = pd.concat(CatBoostClassifier, axis=1).T
CatBoostClassifier_tr_df = pd.concat(CatBoostClassifier_tr, axis=1).T
SVC_df = pd.concat(SVC, axis=1).T
SVC_tr_df = pd.concat(SVC_tr, axis=1).T
def set_index(*dfs):
    for df in dfs:
        df.index = count_all

# # 將所有 DataFrame 運用函數更改索引
set_index(CatBoostClassifier_df,CatBoostClassifier_tr_df,SVC_df,SVC_tr_df)

def create_folder_if_not_exist(fp):
    if fp is not None and not os.path.exists(fp):
        os.makedirs(fp)
save_folder_path = 'C:\\Users\\UserCmhuh\\Desktop\\feature quantity\\'+'\\'+img_type
create_folder_if_not_exist(save_folder_path)

def plt_df(df):
    # plt.xticks(count_all[0:1] + count_all[9:])
    plt.xticks(count_all)
    plt.yticks([i/10 for i in range(11)])

    plt.title(title)
    plt.xlabel('features')
    plt.ylabel('value')
    plt.legend()
    plt.savefig(os.path.join(save_folder_path, f"{title}.png"))
    plt.show()
       
for df, title in zip([CatBoostClassifier_df,CatBoostClassifier_tr_df, SVC_df,SVC_tr_df], [features_count+'features_CatBoostClassifier_test_'+img_type,features_count+'features_CatBoostClassifier_training_'+img_type,features_count+'features_SVC_test_'+img_type,features_count+'features_SVC_training_'+img_type]):
    # df.reset_index(drop=True, inplace=True)
    # df.index += 1

    df[['Accuracy', 'TPR', 'TNR']].plot()
    plt_df(df)

plt.plot(CatBoostClassifier_df.index, CatBoostClassifier_df['Accuracy'],label='Cat_test')
plt.plot(CatBoostClassifier_tr_df.index, CatBoostClassifier_tr_df['Accuracy'], label='Cat_training')
plt.plot(SVC_df.index, SVC_df['Accuracy'],label='SVC_test')
plt.plot(SVC_tr_df.index, SVC_tr_df['Accuracy'], label='SVC_training')

title= features_count+'features_Comparison of Accuracy_'+img_type
plt_df(df)
