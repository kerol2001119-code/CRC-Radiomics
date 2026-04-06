import os
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
# import xgboost as xgb
# import lightgbm as lgb
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
from sklearn.linear_model import LassoCV
feature_selection='lasso' #Pearson,lasso,ANOVA
classifier='CatBoost' #CatBoost,SVM,RF,MLP
img_type1 ='DWI_b800' #ADC,DWI_b100,DWI_b800,T2,T2_FS
img_type2='T2'
img_type3='T2'
normalization_type="_min-max_nii_10pixel_sing_manual_new"
img_type_binding='2' #2=filename1+filename2
features_count=0 #for Pearson,ANOVA
alpha_count=0 #for lasso
features_count_range=18 #增加feature_number的次數
roi='new_0122_manual_bbox' #_roi,_bbox
dilate_type = '' #_dilate1,_dilate3...

filename1=img_type1+roi+dilate_type
filename2=img_type2+normalization_type+roi+dilate_type
filename3=img_type2+normalization_type+roi+dilate_type

data1 = pd.read_csv(rf'C:\Users\Naria\Desktop\{filename1}.csv')#讀檔路徑
data2=pd.read_csv(rf'C:\Users\Naria\Desktop\{filename2}.csv')
data3=pd.read_csv(rf'C:\Users\Naria\Desktop\{filename3}.csv')
datapath="D:\\chen_radiomics\\chen_radiomics\\classify\\result\\"#選擇nii檔的路徑
training_folder_path=datapath+'pred_result\\training\\'
testing_folder_path=datapath+'pred_result\\testing\\'

os.makedirs(training_folder_path,exist_ok=True)
os.makedirs(testing_folder_path,exist_ok=True)
def feature_category_filter(data1,img_type1):
    columns = ['firstorder','glcm','gldm','glrlm','glszm','ngtdm','shape','shape 2D'] 
    rows = ['Original','Exponential','Gradient','LBP-2','LBP-3','LoG','logarithm','Square','Square Root','Wavelet'] 
    df = pd.DataFrame(0,columns=columns, index=rows)
    for column_name in data1.columns:
        column_name_lower = column_name.lower()
        if 'firstorder' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'firstorder'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'firstorder'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'firstorder'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'firstorder'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'firstorder'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'firstorder'] += 1
            elif 'log' in column_name_lower:
                df.loc['LoG', 'firstorder'] += 1
            elif 'wavelet' in column_name_lower:
                 df.loc['Wavelet', 'firstorder'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'firstorder'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'firstorder'] += 1
           
            data1 = data1.drop(columns=[column_name])
        elif 'glcm' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'glcm'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'glcm'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'glcm'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'glcm'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'glcm'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'glcm'] += 1
            elif 'log-sigma' in column_name_lower:
                df.loc['LoG', 'glcm'] += 1
            elif 'wavelet' in column_name_lower:
                df.loc['Wavelet', 'glcm'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'glcm'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'glcm'] += 1
           
            data1 = data1.drop(columns=[column_name])
        elif 'gldm' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'gldm'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'gldm'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'gldm'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'gldm'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'gldm'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'gldm'] += 1
            elif 'log' in column_name_lower:
                df.loc['LoG', 'gldm'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'gldm'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'gldm'] += 1
            elif 'wavelet' in column_name_lower:
                df.loc['Wavelet', 'gldm'] += 1
            data1 = data1.drop(columns=[column_name])
        elif 'glrlm' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'glrlm'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'glrlm'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'glrlm'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'glrlm'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'glrlm'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'glrlm'] += 1
            elif 'log' in column_name_lower:
                df.loc['LoG', 'glrlm'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'glrlm'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'glrlm'] += 1
            elif 'wavelet' in column_name_lower:
                df.loc['Wavelet', 'glrlm'] += 1
            data1 = data1.drop(columns=[column_name])
        elif 'glszm' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'glszm'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'glszm'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'glszm'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'glszm'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'glszm'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'glszm'] += 1
            elif 'log' in column_name_lower:
                df.loc['LoG', 'glszm'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'glszm'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'glszm'] += 1
            elif 'wavelet' in column_name_lower:
                df.loc['Wavelet', 'glszm'] += 1
            data1 = data1.drop(columns=[column_name])
        elif 'ngtdm' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'ngtdm'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'ngtdm'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'ngtdm'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'ngtdm'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'ngtdm'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'ngtdm'] += 1
            elif 'log' in column_name_lower:
                df.loc['LoG', 'ngtdm'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'ngtdm'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'ngtdm'] += 1
            elif 'wavelet' in column_name_lower:
                df.loc['Wavelet', 'ngtdm'] += 1
            data1 = data1.drop(columns=[column_name])
        elif 'shape' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'shape'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'shape'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'shape'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'shape'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'shape'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'shape'] += 1
            elif 'log' in column_name_lower:
                df.loc['LoG', 'shape'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'shape'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'shape'] += 1
            elif 'wavelet' in column_name_lower:
                df.loc['Wavelet', 'shape'] += 1
            data1 = data1.drop(columns=[column_name])
        elif 'shape 2d' in column_name_lower:
            if 'original' in column_name_lower:
                df.loc['Original', 'shape 2D'] += 1
            elif 'exponential' in column_name_lower:
                df.loc['Exponential', 'shape 2D'] += 1
            elif 'gradient' in column_name_lower:
                df.loc['Gradient', 'shape 2D'] += 1
            elif 'lbp-2' in column_name_lower:
                df.loc['LBP-2', 'shape 2D'] += 1
            elif 'lbp-3' in column_name_lower:
                df.loc['LBP-3', 'shape 2D'] += 1
            elif 'logarithm' in column_name_lower:
                df.loc['logarithm', 'shape 2D'] += 1
            elif 'log' in column_name_lower:
                df.loc['LoG', 'shape 2D'] += 1
            elif 'squareroot' in column_name_lower:
                df.loc['Square Root', 'shape 2D'] += 1
            elif 'square' in column_name_lower:
                df.loc['Square', 'shape 2D'] += 1
            elif 'wavelet' in column_name_lower:
                df.loc['Wavelet', 'shape 2D'] += 1
            data1 = data1.drop(columns=[column_name])
    df.select_dtypes(include='number').sum().sum()
    df.to_csv(rf'C:\\Users\\Naria\\Desktop\\{img_type1}_feature_category.csv', index=True)
feature_category_filter(data1,img_type1)   
#%%
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

def three_img_type_binding_to_df(X1,X2,X3,y):   
    X1_data=trans_data_to_df(X1)
    X2_data=trans_data_to_df(X2)
    X3_data=trans_data_to_df(X3)
    
    X_data = pd.concat([X1_data,X2_data,X3_data], axis=1, ignore_index=True)
    y_data = y
    img_type = img_type1+'+'+img_type2+'+'+img_type3
    return X_data,y_data,img_type

#過濾掉只有一個的class
def filter_only_have_one_case_class(X_data,y_data):
    unique_classes, class_counts = np.unique(y_data, return_counts=True)  
    rare_classes = unique_classes[class_counts < 2]
    
    mask = np.isin(y_data, rare_classes, invert=True)
    
    X_data = X_data[mask]
    y_data = y_data[mask]
    return X_data,y_data

def set_different_features_quantity(features_count):
    if feature_selection=='lasso':
        global alpha_count
        alpha_count = round(alpha_count + 0.01, 2)
    elif features_count < 10:
        features_count += 1
    elif features_count < 100:
        features_count += 10
    else:
        features_count += 50
    return features_count

def ANOVA(X_tr,y_tr,features_count,X_te):
    skb = SelectKBest(score_func=f_classif, k=features_count)
    skb.fit_transform(X_tr, y_tr)
    
    # 獲取被選擇的特徵索引
    selected_indices = skb.get_support(indices=True)
    selected_features = X_tr.columns[selected_indices]
    
    # 根據選擇的特徵索引重新排列特徵矩陣2; 
    X_tr = X_tr[selected_features]
    X_te = X_te[selected_features]
    print("Expected number of features:", X_tr.shape[1])
    
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
            # delete_feature.append(i)
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


def find_best_threshold_Youden_index(Y, prob_pred):
    # 計算 ROC 曲線
    fpr, tpr, thresholds = roc_curve(Y, prob_pred)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    
    return best_threshold,fpr,tpr

feature_importance_all=[]
prob_pred_all=[]
all_test_pred_results = [] 
all_prob_pred_results = []  
def classifier_ten_fold_cross_validation(X_tr,y_tr,X_te,y_te,result_df_all,features_count,alpha_count):
    global every_fold_traininf_and_testing
    feature_importance = None
    seeds = seeds_in[fold]
    # 選擇當前迴圈使用的種子
    def get_trainning_and_test_sets(X_tr, X_te, y_tr, y_te):
        df_y_te = pd.DataFrame({'case':X_data_major.loc[X_te.index]['case'].values,
                                'index': X_te.index, 
                                'label': y_te})    
        df_y_tr=pd.DataFrame({'case':X_data_major.loc[X_tr.index]['case'].values,
                                'index': X_tr.index, 
                                'label': y_tr})
        # 添加到列表中
        every_fold_testing.append(df_y_te)
        every_fold_training.append(df_y_tr)
        
        
    get_trainning_and_test_sets(X_tr, X_te, y_tr, y_te)    
        
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
        # alphas = np.logspace(-3,1,50)
        # model_lassoCV = LassoCV(alphas=alphas,cv = 10,max_iter = 100000).fit(X_tr,y_tr)
        # coef = pd.Series(model_lassoCV.coef_, index = X_tr.columns)
        # index = coef[coef!=0].index
        # X_tr = X_tr[index]
        # X_te = X_te[index]
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
    if classifier=='CatBoost':
        models = [CatBoostClassifier(random_state=seeds, verbose=False)]
    elif classifier=='RF':
        models = [RandomForestClassifier(n_estimators=15, random_state=seeds)]
    elif classifier=='MLP':
        models = [MLPClassifier(hidden_layer_sizes=(30), activation='relu', solver='adam', random_state=seeds)]

  
    results_df = []


    print(X_tr.shape)
    print(X_te.shape)
    for model in models:
        # 設定儲存模型之絕對路徑
        absolute_path = r"D:\chen_radiomics\chen_radiomics\model"
        model_filename = os.path.join(absolute_path, f"{model.__class__.__name__}_loop{fold + 1}.pkl")

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
        
        for x,Y in zip(X_all,Y_all):

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
            f1sc = (2 * ppv * tprate) / (ppv + tprate)
            
    
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
            
            all_fpr.append(fpr)
            all_tpr.append(tpr)
            tnr_forsta.append(tnrate)
            tpr_forsta.append(tprate)
            
            
            pred_results.append({
                'True Labels': Y,
                'Predicted Probabilities': prob_pred
            })
            
    results_df.append(list(new_row.values()))
    result_df_all.append(results_df)
    
    # del X_tr, X_te, y_tr, y_te, results_df
    return result_df_all
    
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
    plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=f'Average ROC curve (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model_name)
    plt.legend(loc="lower right")
    plt.show()
    del all_fpr,all_tpr
    
print("---------開始大型迴圈 ---------")
def save_every_fold_training_and_testing(i,df,training_folder_path):
    file_name = f'df_{i}.csv'  
    file_path = training_folder_path+file_name
    df.to_csv(file_path, index=False) 
    
    
    
#-----------------主程式-----------------------    
seperate_mean_value = []
test_result_df_all=[]
alpha_all=[]
count_all=[]
X1=data1[data1.columns[2:]]
X2=data2[data2.columns[2:]]
X3=data3[data3.columns[2:]]

rgen = np.random.RandomState(42)
seeds_in = rgen.randint(low=0, high=1000, size=10)
y=np.array(data1['label'])
y=y.astype('int')


if img_type_binding=='1':
    X_data,y_data,img_type= one_img_type_to_df(X1,y)
elif img_type_binding=='2':
    X_data,y_data,img_type= two_img_type_binding_to_df(X1,X2,y)
elif img_type_binding=='3':
    X_data,y_data,img_type= three_img_type_binding_to_df(X1,X2,X3,y)
    
X_data_major,y_data_major = filter_only_have_one_case_class(data1,y_data)
X_data_major.reset_index(drop=True, inplace=True)
X_data,y_data=filter_only_have_one_case_class(X_data,y_data)
X_data.reset_index(drop=True, inplace=True)
y_data = pd.Series(y_data)
print('X_data:', X_data.index)
for b in range(features_count_range):
    features_count=set_different_features_quantity(features_count)
    model_name=img_type+'+'+feature_selection+'+'+classifier
    result_df_all=[]
    all_fpr=[]
    all_tpr=[]
    tnr_forsta=[]
    tpr_forsta=[]
    pred_results=[]
    every_fold_testing=[]
    every_fold_training=[]
    seeds=42
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seeds)
    # X_data 是特征数据，y_data 是标签数据
    for fold, (train_index, test_index) in enumerate(skf.split(X_data, y_data)):
        print(f"Fold {fold + 1}")
        
        # 获取训练和测试集的索引
        X_tr, X_te = X_data.iloc[train_index], X_data.iloc[test_index]
        y_tr, y_te = y_data.iloc[train_index], y_data.iloc[test_index]
  
        result_df_all=classifier_ten_fold_cross_validation(X_tr,y_tr,X_te,y_te,result_df_all,features_count,alpha_count)
    for i, df in enumerate(every_fold_training):
        save_every_fold_training_and_testing(i,df,training_folder_path)
    for i, df in enumerate(every_fold_testing):
        save_every_fold_training_and_testing(i,df,testing_folder_path)
    seperate_mean_value=mean_value_of_ten_fold_cross_validation(result_df_all,count_all,seperate_mean_value,features_count)  
    # print_roc_curve(all_fpr,all_tpr)

features_count=str(features_count)
#%%
all_results_df = pd.concat(seperate_mean_value, ignore_index=False)
all_results_df.to_csv(rf"C:\Users\Naria\Desktop\{model_name}.csv", index=True)
