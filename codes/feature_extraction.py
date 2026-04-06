import SimpleITK as sitk
from radiomics import featureextractor
import pandas as pd
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import logging
import cv2

datapath='D:\\chen_radiomics\\chen_radiomics\\classify\\'
label_excel_datapath=r"D:\chen_radiomics\chen_radiomics\1008\rectal_cancer\updated_caselist.xlsx"
doesnt_match_datapath="D:\\chen_radiomics\\chen_radiomics\\classify\\need_Modify\\"
modified_img_datapath="D:\\chen_radiomics\\chen_radiomics\\classify\\Modified\\"

casetype='T2_FS'#選擇提取特徵的影像類型
normalize_type = ''
roi_name = "new_0122_manual_"
roi_type='bbox'#選擇提取特徵的roi類型
dilate_type='' #normalize的value須根據不同影像調整

def bulid_labelpath(datapath,casetype): 
    labelpath=os.listdir(datapath+casetype)
    labelpath=sorted([[casetype,x[0:-4]] for x in labelpath])
    return labelpath

def tablefilter(table):
    for i in table.columns[2:]:
        if(str(pd.to_numeric(table[i],'coerce').values[0])=='nan'):
            table=table.drop(i,axis=1)
    return table
def makefeaturetable(feature,featurename):
    df_tmp=pd.DataFrame.from_dict(feature).T
    df_tmp.columns=featurename
    return df_tmp

def open_extractor(normalize_type):
    extractor = featureextractor.RadiomicsFeatureExtractor()#建立提取器
    extractor.enableAllImageTypes()#開啟全部的濾波器
    extractor.enableAllFeatures()#開啟全部特徵
    extractor.enabledImagetypes['LoG']={'sigma':[1.0,3.0]}
    extractor.enabledImagetypes['Wavelet']={'binWidth':15}
    extractor.settings['binWidth']=3
    extractor.Normalize = False
    return extractor

def set_logger():
    nib.imageglobals.logger.level = 40
    # 無視radiomics的提醒
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)
    logger = logging.getLogger("radiomics.glcm")
    logger.setLevel(logging.ERROR)
    return logger


def feature_extraction(i,labelpath,extractor):
    casetype = labelpath[i][0]
    case_i = labelpath[i][1]
    print(case_i)
    case_id = '_'.join(case_i.split('_')[:2])
    if case_i.endswith(casetype):
        try:
            # 讀取nii
            nii = nib.load(datapath + casetype +normalize_type+ '\\' + case_i + '.nii') 

            # 讀取ROI
            roi_nii = nib.load(datapath + roi_name + roi_type + dilate_type+'\\' + case_id +'_'+roi_type+'.nii')
            
            
            def bbox(nii,roi_nii):
                nii = nii.get_fdata()
                roi_nii = roi_nii.get_fdata()
                roi_nii = np.where(roi_nii != 0, 1, 0).astype(np.uint8)
                # plt.imshow(roi_nii[:,:,12], cmap='gray')
                if 'T2' in casetype:
                    new_width = 512
                    new_height = 512
                    roi_nii= cv2.resize(roi_nii, (new_width, new_height),interpolation=cv2.INTER_NEAREST)
                    
                    
                    # plt.imshow(roi_nii[:,:,12], cmap='gray')
            
                
            
                nii = sitk.GetImageFromArray((nii.transpose(2, 0, 1))) 
                roi_nii = sitk.GetImageFromArray((roi_nii.transpose(2, 0, 1)))
              
                IMG = nii
                ROI = roi_nii
                return IMG,ROI
            
            
            IMG,ROI =bbox(nii,roi_nii)
            
            pixel_count_1 = np.sum(ROI == 1)

            print(f"值為 1 的像素數量：{pixel_count_1}")
  
            # 特徵提取
            IMG1d = extractor.execute(IMG, ROI, label=1)  # 寫入IMG1d的特徵
            IMG1d_feature = list(IMG1d.values())
            feature1d_name = list(IMG1d.keys())

            r = ['label', 'case']
            r2 = ['ADC1d-' + x for x in feature1d_name]

            # read label
            labeltype_1 = pd.read_excel(label_excel_datapath)
            
            # 创建条件掩码以选择与 case_1 相匹配的行
            mask = labeltype_1['資料夾名'].str.strip().str[:-1] == case_id
            labeltype_2 = labeltype_1.loc[mask, 'feature_label'].values
            labeltype = labeltype_2[0]
            dfMRI_temp = makefeaturetable([labeltype, case_i] + IMG1d_feature, r + r2)
            
            return dfMRI_temp
    
        except FileNotFoundError:
            NO_file.append(case_id)
            return None
labelpath=bulid_labelpath(datapath,casetype)
extractor=open_extractor(normalize_type)

dfMRI = pd.DataFrame()
doesnt_match=[]
doesnt_fix=[]
NO_file=[]
    
for i in range(len(labelpath)):
    dfMRI_temp = feature_extraction(i, labelpath,extractor)
    if dfMRI_temp is not None:
        dfMRI = pd.concat([dfMRI, dfMRI_temp])
#%%
dfMRI = tablefilter(dfMRI)
output_path = r'C:\Users\Naria\Desktop\\' + casetype + normalize_type + roi_name + roi_type + dilate_type + '.csv'
dfMRI.to_csv(output_path,index=False)
print("final csv",dfMRI)