
from pathlib import Path
import pandas as pd
import os
import numpy as np

    
    
def list_files(directory):
    """返回指定目錄下的所有文件和目錄名稱列表。"""
    try:
        # 獲取目錄下的所有條目
        files = os.listdir(directory)
        return files
    except FileNotFoundError:
        return "指定的目錄不存在"
    except PermissionError:
        return "沒有權限訪問這個目錄"    
    
    
df = pd.read_csv('./csv_data/mimic_all.csv')
path_list_new1 = pd.read_csv('./csv_data_new/Pneumonia.csv')
path_list_new2 = pd.read_csv('./csv_data_new/No_Finding.csv')
path_list_new1['StudyID'] = path_list_new1['study_id']
path_list_new2['StudyID'] = path_list_new2['study_id']

path_list_new1 = pd.concat([path_list_new1, path_list_new2], ignore_index=True)

df =df[df['StudyID'].isin(path_list_new1['StudyID'])]

path_list = []
path_list_new = list_files('/Volumes/G-DRIVE ArmorATD/MIT/mimic_all/p')


for value in df['path']:
    path_list.append(value)



print(path_list[0])
print(path_list_new[0])

result = [x for x in path_list if x.split('/')[-1] not in path_list_new]



# 你的用户名和密码
username = "jerryshu"
password = "w45**67dFJbAjVx"

# 下载文件的基础 URL
base_url = "https://physionet.org/files/mimic-cxr/2.0.0/"
base_url = "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
# 指定下载目录
now = 0

c = 1
#14927
#gsutil -u mitexposure cp gs://mimic-cxr-2.0.0.physionet.org/files/p10/p10000032/s53911762/68b5c4b1-227d0485-9cc38c3f-7b84ab51-4b472714.dcm ./
print(len(result))
for file_path in result:
    print("=================================================================")
    print(c)
    print("=================================================================")
    if(c%5000==0):
        now = now+1
    # 构造完整的下载命令，包括下载目录
    download_dir = " ./mimic_jpg/p"+str(now)+'/'+file_path.split('/')[-1]
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    command = f"wget --user {username} --password {password} -P {download_dir} {base_url}{file_path}"
    command = 'gsutil -u mitexposure cp gs://mimic-cxr-2.0.0.physionet.org/'+file_path+ download_dir
    print(command)
    if c>=-1:
        os.system(command)
    c+=1
