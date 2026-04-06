# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:49:51 2024

@author: UserCmhuh
"""

import pydicom
import os
import numpy as np
import pandas as pd
import nibabel as nib
      
path=r"C:\Users\UserCmhuh\Desktop\chen_radiomics\image"
csv_path=r"C:\Users\UserCmhuh\Desktop\chen_radiomics\classify\dicom_inf_csv"
nii_save_path = r"C:/Users/UserCmhuh/Desktop/chen_radiomics/classify/image_type_test"

T1_doesnt_exist=[]
all_vol_img_dwi_seriesnumber=[]
all_vol_img_dwi_seriesnumber_strlist=[]
all_file_paths=[]
all_df=[]

def create_folder_if_not_exist(*paths):
    for path in paths:
        if path is not None and not os.path.exists(path):
            os.makedirs(path)
            
def create_subfolders_list(loading_path):
    subfolders = [f for f in os.listdir(loading_path) if os.path.isdir(os.path.join(loading_path, f))]    
    return subfolders

def create_csv_data_list(loading_path):
    csv_datas = [f for f in os.listdir(loading_path) if f.endswith('.csv') and os.path.isfile(os.path.join(loading_path, f))]  
    return csv_datas

def create_data_path(data_path,data):
    file_path = os.path.join(data_path,data)
    all_file_paths.append(file_path)
    del file_path
    return all_file_paths
    
def read_csv_info(csv_path,csv_data):
    file_path = os.path.join(csv_path,csv_data)
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        data['csv_file_name'] = csv_data
        data.sort_values(by=['series_numbers', 'instance_number'], inplace=True)
        all_df.append(data)
    return all_df  
     
def build_dicom_information(subfolder):
    print(f"Processing subfolder: {subfolder}")
    subfolder_path = os.path.join(path, subfolder)

    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    for dirpath, dirnames, files in os.walk(subfolder_path):
        for file in files:
            img_path = os.path.join(dirpath, file)
            img = pydicom.read_file(img_path)
            list1.append(file)
            list2.append(img[0x00200013].value if 0x00200013 in img else None) #避免部分影像沒有slice location造成的error
            list3.append(img[0x0008103E].value if 0x0008103E in img else None)
            list4.append(img[0x00200011].value if 0x00200011 in img else None)
            list5.append(img[0x00201041].value if 0x00201041 in img else None)
                  
    df = pd.DataFrame([list1, list2, list3, list4, list5]).transpose()
    df.columns = ['file_names', 'instance_number', 'series_descriptions', 'series_numbers', 'slice_locations']
    df.sort_values(by=['series_numbers' ,'instance_number'], inplace=True)

    filename = f"df_{subfolder}.csv"
    file_path = os.path.join(csv_path,filename)
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved as {filename}")
        
    return df

def trans_T2WI_and_T2WIFS_to_nii(df,subfolder_path,nii_save_path):
    vol_img_T2=[]
    vol_img_T2_FS=[]
    df = df.reset_index()
    for i in range(len(df)):    
        if 'Ax T2' in df['series_descriptions'][i]:
            if 'FS' in df['series_descriptions'][i]:
                vol_img_T2_FS.append(pydicom.read_file(subfolder_path + "/" + df['file_names'][i]).pixel_array)
            else:
                vol_img_T2.append(pydicom.read_file(subfolder_path + "/" + df['file_names'][i]).pixel_array)
                
    if len(vol_img_T2) == 0:
            print("no T2"+subfolder)
    elif len(vol_img_T2_FS) == 0:
        print("no T2FS"+subfolder)
    if len(vol_img_T2_FS) != 0:
        vol_img_T2_np = np.transpose(np.array(vol_img_T2_FS), (1, 2, 0))
        nib.save(nib.Nifti1Image(vol_img_T2_np, affine=None), os.path.join(nii_save_path, "T2_FS", subfolder_path.split('\\')[-1] + '_T2_FS.nii'))

    if len(vol_img_T2) != 0:
        vol_img_T2_np = np.transpose(np.array(vol_img_T2), (1, 2, 0))
        nib.save(nib.Nifti1Image(vol_img_T2_np, affine=None), os.path.join(nii_save_path, "T2", subfolder_path.split('\\')[-1] + '_T2.nii'))

def trans_DWI_to_nii(df,subfolder_path,nii_save_path):
    vol_img_dwi=[]
    df = df.reset_index()
    
    ax_dwi_df = df[df['series_descriptions'].str.contains('Ax DWI')]
    ax_dwi_df=ax_dwi_df.reset_index()
    
    if len(ax_dwi_df['series_descriptions'].unique()) > 1:
        min_length_value = min(ax_dwi_df['series_numbers'], key=lambda x: len(str(x)))
        min_length_df = ax_dwi_df[ax_dwi_df['series_numbers'] == min_length_value]
        max_value_in_min_length = max(min_length_df['series_numbers'])
        max_value_df = min_length_df[min_length_df['series_numbers'] == max_value_in_min_length]
        for j in range(len(max_value_df)): 
            vol_img_dwi.append(pydicom.read_file(subfolder_path + "/" + max_value_df['file_names'].iloc[j]).pixel_array)
       
        
    else:
        for i in range(len(ax_dwi_df)):
            vol_img_dwi.append(pydicom.read_file(subfolder_path + "/" + ax_dwi_df['file_names'][i]).pixel_array)

        
    mid_index = len(vol_img_dwi) // 2
    vol_img_dwi_b100=np.transpose(np.array(vol_img_dwi[:mid_index]), (1, 2, 0))
    vol_img_dwi_b800=np.transpose(np.array(vol_img_dwi[mid_index:]), (1, 2, 0))
    
    nib.save(nib.Nifti1Image(vol_img_dwi_b100, affine=None), os.path.join(nii_save_path, "DWI_b100", subfolder_path.split('\\')[-1] + '_DWI_b100.nii'))
    nib.save(nib.Nifti1Image(vol_img_dwi_b800, affine=None), os.path.join(nii_save_path, "DWI_b800", subfolder_path.split('\\')[-1] + '_DWI_b800.nii'))

              
#T1_first


def trans_T1first_to_nii(df,subfolder_path,nii_save_path):
    vol_img_T1_first=[]
    df = df.reset_index()    
    ax_T1_df = df[df['series_descriptions'].str.contains('WATER: Ax LAVA Flex Dyn C+')]
    ax_T1_df = ax_T1_df.reset_index()
    if len(ax_T1_df) == 0:
        T1_doesnt_exist.append(subfolder_path)
        return
    
    aprev_location=float('inf')
    for j in range(len(ax_T1_df)):
        current_slice_location = ax_T1_df['slice_locations'][j]
        if current_slice_location>aprev_location:
            break
        vol_img_T1_first.append(pydicom.read_file(subfolder_path + "/" + ax_T1_df['file_names'].iloc[j]).pixel_array)  # 选择之前所有的值
        aprev_location=current_slice_location
    
    vol_img_T1_first_np = np.transpose(np.array(vol_img_T1_first), (1, 2, 0))
    nib.save(nib.Nifti1Image(vol_img_T1_first_np, affine=None), os.path.join(nii_save_path, "T1_first", subfolder_path.split('\\')[-1] + '_T1.nii'))
    
#T1_peak

def  trans_T1peak_to_nii(df,subfolder_path,nii_save_path):
    vol_img_T1_peak=[]
    df = df.reset_index()    
    ax_T1_df = df[df['series_descriptions'].str.contains('WATER: Ax LAVA Flex Dyn C+')]
    ax_T1_df = ax_T1_df.reset_index()
    if len(ax_T1_df) == 0:
        return
    
    aprev_location=float('inf')
    count=0
    index_20th = None
    for j in range(len(ax_T1_df)):
        current_slice_location = ax_T1_df['slice_locations'][j]
        if current_slice_location>aprev_location:
            count+=1
        aprev_location=current_slice_location

        if count==20:
            index_20th = j
            break
    aprev_location = float('inf')
    for k in range(index_20th,len(ax_T1_df)):
        current_slice_location = ax_T1_df['slice_locations'][k]
        if current_slice_location>aprev_location:
            break
        vol_img_T1_peak.append(pydicom.read_file(subfolder_path + "/" + ax_T1_df['file_names'].iloc[k]).pixel_array)  # 选择之前所有的值
        aprev_location=current_slice_location
    vol_img_T1_peak_np = np.transpose(np.array(vol_img_T1_peak), (1, 2, 0))
    nib.save(nib.Nifti1Image(vol_img_T1_peak_np, affine=None), os.path.join(nii_save_path, "T1_peak", subfolder_path.split('\\')[-1] + '_T1_peak.nii'))                                 
            
                     
#adc

  
def adding_all_dwi_seriesnumber(df,subfolder_path):
    vol_img_dwi_seriesnumber=[]
    for i in range(len(df)):
        if 'Ax DWI' in df['series_descriptions'][i]:
            vol_img_dwi_seriesnumber.append(df['series_numbers'][i]) #儲存dwi的series number為list' 
    all_vol_img_dwi_seriesnumber.append(vol_img_dwi_seriesnumber)

def trans_all_dwi_seriesnumber_to_str(vol_img_dwi_seriesnumber):    
    vol_img_dwiseriesnumber_strlist = [str(x) for x in vol_img_dwi_seriesnumber]
    all_vol_img_dwi_seriesnumber_strlist.append(vol_img_dwiseriesnumber_strlist)
        
def trans_adc_to_nii(subfolder_path, df, vol_img_dwiseriesnumber_strlist):           
    vol_img_adc = []
    df = df.reset_index()
    
    for i in range(len(df)):
        if any((string[:-2] + '50' if string[-2:]=='00' else string+'50') in str(df['series_numbers'][i]) for string in vol_img_dwiseriesnumber_strlist):
            vol_img_adc.append(pydicom.read_file(subfolder_path + "/" + df['file_names'][i]).pixel_array)
    if len(vol_img_adc) == 0: 
        print("no ADC:"+subfolder_path)
        return
    vol_img_adc_np = np.transpose(np.array(vol_img_adc), (1, 2, 0))
    nib.save(nib.Nifti1Image(vol_img_adc_np, affine=None),  os.path.join(nii_save_path, "ADC", subfolder_path.split('\\')[-1] + '_ADC.nii'))


create_folder_if_not_exist(path,csv_path,nii_save_path)  
csv_datas = create_csv_data_list(csv_path)
subfolders=create_subfolders_list(path)


img_types = ['T2','T2_FS', 'T1_first', 'T1_peak', 'DWI_b100','DWI_b800','ADC']
for img_type in img_types:
    dir_path = os.path.join(nii_save_path, img_type)
    create_folder_if_not_exist(dir_path)

for idx, subfolder in enumerate(subfolders):
    # build_dicom_information(subfolder)
    subfolder_paths=create_data_path(path,subfolder)
    
for csv_data in csv_datas:
    all_df=read_csv_info(csv_path,csv_data)

for idx, (subfolder_path, df) in enumerate(zip(subfolder_paths, all_df)):
   folder_name = os.path.basename(subfolder_path)
    
   if df['csv_file_name'].str.contains(folder_name).any():
        print(f"Processing {idx}: {subfolder_path} - {df['csv_file_name'].values[0]}")
    
        trans_T2WI_and_T2WIFS_to_nii(df, subfolder_path, nii_save_path)
        trans_DWI_to_nii(df, subfolder_path, nii_save_path)
        trans_T1first_to_nii(df, subfolder_path, nii_save_path)
        trans_T1peak_to_nii(df, subfolder_path, nii_save_path)
        adding_all_dwi_seriesnumber(df, subfolder_path)
   else:
        print(f'Unmatched folder: {subfolder_path}')
          
for idx, vol_img_dwi_seriesnumber in enumerate(all_vol_img_dwi_seriesnumber):
    trans_all_dwi_seriesnumber_to_str(vol_img_dwi_seriesnumber)
    
for idx, (subfolder_path, df, vol_img_dwiseriesnumber_strlist) in enumerate(zip(subfolder_paths, all_df, all_vol_img_dwi_seriesnumber_strlist)):
    print(f"Processing {idx}: {subfolder_path}")
    trans_adc_to_nii(subfolder_path, df, vol_img_dwiseriesnumber_strlist)