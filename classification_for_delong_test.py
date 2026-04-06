# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:11:14 2024

@author: UserCmhuh
"""

import os
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

import csv
from sklearn.metrics import roc_curve


#ex:'DWI_b100''DWI_b800''T2''T2_FS''ADC'
feature_selection='lasso' #Pearson,lasso,ANOVA
classifier='CatBoost' #CatBoost,SVM,RF,MLP
img_type1 ='DWI_b800' #ADC,DWI_b100,DWI_b800,T2,T2_FS
img_type2='T2'
img_type3='T2'
normalization_type="_min-max_nii_10pixel_sing"
img_type_binding='2' #2=filename1+filename2
features_count=0 #for Pearson,ANOVA
alpha_count=0.07 #for lasso
features_count_range=1 #增加feature_number的次數
roi='_roi' #_roi,_bbox
dilate_type = '' #_dilate1,_dilate3...

filename1=img_type1+roi+dilate_type
filename2=img_type2+normalization_type+roi+dilate_type
filename3=img_type2+normalization_type+roi+dilate_type

data1 = pd.read_csv(rf'D:\chen_radiomics\chen_radiomics\new_manual\{filename1}.csv')#讀檔路徑
data2=pd.read_csv(rf'D:\chen_radiomics\chen_radiomics\new_manual\{filename2}.csv')
data3=pd.read_csv(rf'D:\chen_radiomics\chen_radiomics\new_manual\{filename3}.csv')
datapath="D:\\chen_radiomics\\chen_radiomics\\classify\\result\\"#選擇nii檔的路徑
training_folder_path=datapath+'pred_result\\training\\'
testing_folder_path=datapath+'pred_result\\testing\\'


def load_setted_training_and_testing_set(training_folder_path,testing_folder_path):
    tr_file_names = [f for f in os.listdir(training_folder_path) if f.endswith('.csv')]
    te_file_names = [f for f in os.listdir(testing_folder_path) if f.endswith('.csv')]
    
    tr_dfs_list = []
    for tr_file_name in tr_file_names:
        tr_file_path = os.path.join(training_folder_path, tr_file_name)
        tr_df = pd.read_csv(tr_file_path)
        tr_dfs_list.append(tr_df)
        
    te_dfs_list=[]    
    for te_file_name in te_file_names:
        te_file_path = os.path.join(testing_folder_path, te_file_name)
        te_df = pd.read_csv(te_file_path)
        te_dfs_list.append(te_df)
        
    return tr_dfs_list,te_dfs_list,te_file_names,tr_file_names


#smote增加case數量
def set_smote():
    which_class=0
    number=56
    ratio={which_class:number}
    smote1 = SMOTE(sampling_strategy=ratio, random_state=42)
    smote2 = BorderlineSMOTE(sampling_strategy=ratio,random_state=42,kind="borderline-1")
    smote3 = ADASYN(sampling_strategy=ratio,random_state=42)
    return smote1,smote2,smote3

def drop_redundant_cases_in_different_roi(data1,data2):    
    data1['case_id'] = data1['case'].apply(lambda k: '_'.join(k.split('_')[:2]))
    data2['case_id'] = data2['case'].apply(lambda k: '_'.join(k.split('_')[:2]))
    data1 = data1[data1['case_id'].isin(data2['case_id'])]
    data1=data1.reset_index(drop=True) 
    return data1

# data1=drop_redundant_cases_in_different_roi(data1,data2)

def trans_data_to_df(X):
    X = X.astype(np.float64)
    X = pd.DataFrame(X)
    X_data = X
    return X_data

def one_img_type_to_df(X1,y):
    X_data=trans_data_to_df(X1)
    y_data = y
    img_type=filename1
    return X_data,y_data,img_type

def two_img_type_binding_to_df(X1,X2,y):    
    X1_data=trans_data_to_df(X1)
    X2_data=trans_data_to_df(X2)
    
    X_data = pd.concat([X1_data,X2_data], axis=1, ignore_index=True)
    y_data = y
    img_type = filename1+'+'+filename2
    return X_data,y_data,img_type

# --- 【MODIFIED 1】: 修改函數以回傳 mask ---
#過濾掉只有一個的class
def filter_only_have_one_case_class(X_data, y_data, return_mask=False):
    unique_classes, class_counts = np.unique(y_data, return_counts=True)  
    rare_classes = unique_classes[class_counts < 2]
    
    mask = np.isin(y_data, rare_classes, invert=True)
    
    X_data = X_data[mask]
    y_data = y_data[mask]
    
    if return_mask:
        return X_data, y_data, mask
    else:
        return X_data,y_data

def ANOVA(X_tr,y_tr,features_count,X_te):
    skb = SelectKBest(score_func=f_classif, k=features_count)
    skb.fit_transform(X_tr, y_tr)
    
    # 獲取被選擇的特徵索引
    selected_indices = skb.get_support(indices=True)
    selected_features = X_tr.columns[selected_indices]
    
    # 根據選擇的特徵索引重新排列特徵矩陣2; 
    X_tr = X_tr[selected_features]
    X_te = X_te[selected_features]
    
    return X_tr,X_te

def Pearson_correlation(X_tr,y_tr,features_count,X_te):
    Xy_corr = X_tr.corrwith(y_tr)
    Xy_corr.sort_values(ascending=False, inplace=True)
    X_tr = X_tr.reindex(columns=Xy_corr.index)
    corr_matrix = X_tr.corr().abs()

    select_feat= []
    delete_feature=[]
    for i in corr_matrix.columns:
        if len(select_feat) and corr_matrix[i][select_feat].max() > 0.9:
            delete_feature.append(i)
            continue 
        select_feat.append(i)
        
        if len(select_feat) == features_count:
            break          
    X_tr = X_tr.filter(items=select_feat)  
    columns_list = X_tr.columns.tolist()
    features = columns_list
    X_te = X_te[features]
    print("Expected number of features Pearson3:", X_tr.shape[1]) 
    print("delete feature:",len(delete_feature))
     
    del Xy_corr,corr_matrix,columns_list,features
    return X_tr,X_te

all_threshold=[]
def find_best_threshold_Youden_index(Y, prob_pred):
    # 計算 ROC 曲線
    fpr, tpr, thresholds = roc_curve(Y, prob_pred)
    all_threshold.append(thresholds)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    
    return best_threshold,fpr,tpr

feature_importance_all=[]
prob_pred_all=[]
tnr_forsta=[]
tpr_forsta=[]
acc_all=[]
f1sc_all=[]
ppv_all=[]
npv_all=[]

# --- 【MODIFIED 2】: 函數簽名加入 test_cases ---
def classifier_ten_fold_cross_validation(X_tr, y_tr, X_te, y_te, test_cases, result_df_all, features_count):
    global every_fold_traininf_and_testing
    seeds = seeds_in[a]
    # 選擇當前迴圈使用的種子
    # normalization
    scaler = preprocessing.StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_tr= pd.DataFrame(X_tr)
    # X_tr,y_tr=smote1.fit_resample(X_tr,y_tr)
    
    X_te = scaler.transform(X_te)
    X_te = pd.DataFrame(X_te)
    y_tr = pd.Series(y_tr)
    
    
    if feature_selection=='Pearson':
        X_tr,X_te=Pearson_correlation(X_tr,y_tr,features_count,X_te)
    elif feature_selection=='ANOVA':
        X_tr,X_te=ANOVA(X_tr,y_tr,features_count,X_te)
    elif feature_selection=='lasso':
        lasso = Lasso(alpha=alpha_count)
        lasso.fit(X_tr, y_tr)
        
        # 獲取非零係數的特徵
        non_zero_indices = np.where(lasso.coef_ != 0)[0]
        X_tr = X_tr.iloc[:, non_zero_indices]
        X_te = X_te.iloc[:, non_zero_indices]
        
        feature_importance = pd.DataFrame({'Feature': X_data.columns[non_zero_indices], 'Coefficient': lasso.coef_[non_zero_indices]})
        feature_importance_all.append(feature_importance)

    if classifier=='SVM':
        models = [SVC(kernel='linear', C=100, gamma= 'scale', probability=True)]
    elif classifier=='CatBoost':
        models = [CatBoostClassifier(random_state=seeds, verbose=False)]
    elif classifier=='RF':
        models = [RandomForestClassifier(n_estimators=15, random_state=seeds)]
    elif classifier=='MLP':
        models = [MLPClassifier(hidden_layer_sizes=(30), activation='relu', solver='adam', random_state=seeds)]

    results_df = []

    all_test_pred_results = [] 
    all_prob_pred_results = []  
    print(X_tr.shape)
    print(X_te.shape)
    for model in models:
        # 設定絕對路徑
        absolute_path = r'D:\chen_radiomics\chen_radiomics\model'
        model_filename = os.path.join(absolute_path, f"{model.__class__.__name__}.pkl")

        if False:#os.path.exists(model_filename):
            model = joblib.load(model_filename)
        else:
            # 如果模型檔案不存在，保存模型
            if classifier=='MLP':
                model.fit(X_tr, y_tr)
            else:
                weight_0 = 7
                weight_1 = 1 
                sample_weights = np.array([weight_0 if label == 0 else weight_1 for label in y_tr])
                model.fit(X_tr, y_tr, sample_weight=sample_weights)
                
            joblib.dump(model, model_filename)
            
        X_all=[X_te]
        Y_all=[y_te]
        Cases_all = [test_cases] # --- 【MODIFIED 3】: 將 test_cases 加入列表
        
        for x, Y, cases in zip(X_all, Y_all, Cases_all): # --- 【MODIFIED 4】: 在 zip 中加入 cases

            # Predict the test set
            prob_pred = model.predict_proba(x)[:, 1]  # For log loss and AUC
            prob_pred_all.append(prob_pred)
            best_threshold,fpr,tpr = find_best_threshold_Youden_index(Y, prob_pred)
            print(f"最佳閾值 (Best Threshold): {best_threshold}")
            
            test_pred = (prob_pred >= best_threshold).astype(int)
            all_test_pred_results.append(test_pred)
            all_prob_pred_results.append(prob_pred)
            
   
            # Calculate various metrics
            auc = roc_auc_score(Y, prob_pred)
            log_loss_value = log_loss(Y, prob_pred)
            cnf_matrix = confusion_matrix(Y, test_pred)
            
            df = pd.DataFrame({
                'True_Label': Y,
                'Predicted_Label': test_pred
            })
            
            df['loop'] = a + 1
   
            # 保存當前結果
            all_results.append(df)

            fp = cnf_matrix[0][1]
            fn = cnf_matrix[1][0]
            tn = cnf_matrix[0][0]
            tp = cnf_matrix[1][1]
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0  # 準確率
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0                             # 精確率
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0                             # 陰性預測值
            fprate = fp / (fp + tn) if (fp + tn) > 0 else 0                          # 假陽性率
            fnrate = fn / (tp + fn) if (tp + fn) > 0 else 0                          # 假陰性率
            tprate = tp / (tp + fn) if (tp + fn) > 0 else 0                          # 靈敏度
            tnrate = tn / (fp + tn) if (fp + tn) > 0 else 0                          # 特異度
            f1sc = (2 * ppv * tprate) / (ppv + tprate) if (ppv + tprate) > 0 else 0
            
   
            # Append results to the DataFrame
            # Append results to the DataFrame
            new_row={
                'Accuracy': acc,
                'Log Loss': log_loss_value,
                'AUC': auc,
                'PPV': ppv,
                'NPV': npv,
                'FPR': fprate,
                'FNR': fnrate,
                'TPR': tprate,
                'TNR': tnrate,
                'F1 Score': f1sc,
            }
            
            auc_scores.append(auc)
            acc_all.append(acc)
            tpr_forsta.append(tprate)
            tnr_forsta.append(tnrate)
            ppv_all.append(ppv)
            npv_all.append(npv)
            f1sc_all.append(f1sc)
            
            pred_rows = []

            # --- 【MODIFIED 5】: 在 zip 中加入 case_name 並存入 pred_rows ---
            for case_name, true_label, pred_prob, test_pred in zip(cases, Y, prob_pred, test_pred):
                pred_rows.append({
                    'case_name': case_name,
                    'dx': true_label,
                    "pred_prob"+model_name: pred_prob,
                    "test_pred"+model_name: test_pred
                })
            
            # 创建 DataFrame
            pred_results_df = pd.DataFrame(pred_rows)
            pred_results_df_all.append(pred_results_df)
            
            
    results_df.append(list(new_row.values()))
    result_df_all.append(results_df)
    
    del X_tr, X_te, y_tr, y_te, results_df
    return result_df_all

alpha_all=[]    
auc_scores=[]
def mean_value_of_ten_fold_cross_validation(result_df_all,count_all,seperate_mean_value,features_count):
    result_df_all = np.array(result_df_all)
    result_df_all = np.mean(result_df_all, axis=0)
    cols = ['Accuracy', 'log loss', 'AUC', 'PPV', 'NPV', 'FPR', 'FNR', 'TPR', 'TNR', 'F1']
    if feature_selection=='lasso':
        index = [classifier +str(alpha_count)]
    else:
        index = [classifier +str(features_count)]
    result_df_all = pd.DataFrame(result_df_all, columns=cols, index=index)
    seperate_mean_value.append(result_df_all)
    alpha_all.append(alpha_count)
    count_all.append(features_count)
    del result_df_all
    return seperate_mean_value


def print_roc_curve(all_fpr,all_tpr):
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    
    for i in range(len(all_fpr)):
        mean_tpr += np.interp(mean_fpr, all_fpr[i], all_tpr[i])
        
    mean_tpr /= len(all_fpr)
    mean_auc = auc(mean_fpr, mean_tpr)
    # Plot average ROC curve
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=f'Average ROC curve (AUC = {mean_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.show()
    
    return mean_tpr

def CI_calculate(auc_scores):
    mean_auc = np.mean(auc_scores)
    se_auc = np.std(auc_scores, ddof=1) / np.sqrt(len(auc_scores))
    ci_margin = 1.96 * se_auc
    print(f"{mean_auc:.3f} ± {ci_margin:.3f}")
    
    return
    
# --- 【MODIFIED 6】: 函數簽名加入 cases_data，並回傳 test_cases ---
def use_setted_training_and_testing_case(tr_df, te_df, X_data, cases_data):    
    fold_train_index = tr_df['index']
    fold_train_label = tr_df['label']
    print(fold_train_index)
    
    fold_test_index = te_df['index']
    fold_test_label = te_df['label']
    
    # 选择对应的训练数据和标签
    X_tr = X_data.loc[fold_train_index]
    y_tr = fold_train_label
    
    # 选择对应的测试数据和标签
    X_te = X_data.loc[fold_test_index]
    y_te = fold_test_label
    
    # 根據相同的 test_index 獲取 case names
    test_cases = cases_data.loc[fold_test_index]
    
    
    return X_tr, y_tr, X_te, y_te, test_cases

def save_fpr_tpr_for_roc_curve(filename, data, datapath):
    with open(f'{datapath}{filename}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)
    print(f"CSV 文件已成功保存至: {datapath}{filename}.csv")
            
print("---------開始大型迴圈 ---------")
X1=data1[data1.columns[2:]]
X2=data2[data2.columns[2:]]

rgen = np.random.RandomState(42)
seeds_in = rgen.randint(low=0, high=1000, size=10)

# --- 【MODIFIED 7】: 同時提取 y (label) 和 cases (case_name) ---
y=np.array(data1['label'])
y=y.astype('int')
cases = data1['case'] # 假設您的 case 欄位名稱為 'case'

if img_type_binding=='1':
    X_data,y_data,img_type= one_img_type_to_df(X1,y)
elif img_type_binding=='2':
    X_data,y_data,img_type= two_img_type_binding_to_df(X1,X2,y)

# --- 【MODIFIED 8】: 使用 mask 同時過濾 X, y 和 cases ---
X_data, y_data, mask = filter_only_have_one_case_class(X_data, y_data, return_mask=True)
cases_data = cases[mask]

# --- 【MODIFIED 9】: 重設所有資料的索引以確保對齊 ---
X_data.reset_index(drop=True, inplace=True)
y_data = pd.Series(y_data).reset_index(drop=True)
cases_data.reset_index(drop=True, inplace=True)

tr_dfs_list,te_dfs_list,te_file_names,tr_file_names=load_setted_training_and_testing_set(training_folder_path,testing_folder_path)
seperate_mean_value = []
count_all=[]
pred_results_df_each_feature=[]
pred_results_df_all_feature=[]
for b in range(features_count_range):
    if feature_selection=='lasso':
        model_name=img_type+'+'+feature_selection+'+'+classifier+'+feature_'+str(alpha_count)
    else :
        model_name=img_type+'+'+feature_selection+'+'+classifier+'+feature_'+str(features_count)
    print(f"feature數量為{features_count}")
    new_rows=[]
    result_df_all=[]
    all_fpr=[]
    all_tpr=[]
    all_results = []
    pred_results_df_all=[]
    for a, (tr_df, te_df) in enumerate(zip(tr_dfs_list, te_dfs_list)):
        # --- 【MODIFIED 10】: 傳入 cases_data 並接收 test_cases ---
        X_tr, y_tr, X_te, y_te, test_cases = use_setted_training_and_testing_case(tr_df, te_df, X_data, cases_data)
        
        # --- 【MODIFIED 11】: 將 test_cases 傳入分類器 ---
        result_df_all = classifier_ten_fold_cross_validation(X_tr, y_tr, X_te, y_te, test_cases, result_df_all, features_count)
        final_df = pd.concat(all_results, ignore_index=True)
        pred_results_df_each_feature = pd.concat(pred_results_df_all,ignore_index=True)
    seperate_mean_value=mean_value_of_ten_fold_cross_validation(result_df_all,count_all,seperate_mean_value,features_count) 
    # mean_tpr=print_roc_curve(all_fpr,all_tpr)
    pred_results_df_all_feature.append(pred_results_df_each_feature)
    # save_fpr_tpr_for_roc_curve('all_fpr3',all_fpr,datapath)
    # save_fpr_tpr_for_roc_curve('all_tpr3',all_tpr,datapath)
    
CI_calculate(auc_scores)
CI_calculate(acc_all)
CI_calculate(tpr_forsta)
CI_calculate(tnr_forsta)
CI_calculate(ppv_all)
CI_calculate(npv_all)
CI_calculate(f1sc_all)
print('"AUC":',auc_scores,",")
print('"ACC":',acc_all,",")
print('"TPR":',tpr_forsta,",")
print('"TNR":',tnr_forsta,",")
print('"PPV":',ppv_all,",")
print('"NPV":',npv_all,",")
print('"F1":',f1sc_all,",")
pred_results_df_all_feature=pd.concat(pred_results_df_all_feature,axis=1)
#%%
pred_results_df_all_feature.to_csv(datapath+model_name+'+pred_results.csv',index=False)
features_count=str(features_count)
# all_results_df = pd.concat(seperate_mean_value, ignore_index=False)
# all_results_df.to_csv(datapath+model_name+'.csv', index=True)