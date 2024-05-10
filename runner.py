import pickle
from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2
import numpy as np
from collections import Counter
import tensorflow as tf
import os
from sklearn.metrics import roc_auc_score
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
import random
from tensorflow.keras.models import load_model
def balance_classes(images, labels):
    # 找出每個類別的索引
    class0_indices = np.where(labels == 0)[0]
    class1_indices = np.where(labels == 1)[0]

    # 平衡類別
    if len(class0_indices) > len(class1_indices):
        extra_indices = np.random.choice(class0_indices, size=len(class1_indices), replace=False)
        balanced_indices = np.concatenate([class1_indices, extra_indices])
    else:
        extra_indices = np.random.choice(class1_indices, size=len(class0_indices), replace=False)
        balanced_indices = np.concatenate([class0_indices, extra_indices])

    return images[balanced_indices], labels[balanced_indices]

def augment_images(image_list):
    # 定義增強變換
    transform = A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.HorizontalFlip(p=0.5),
        #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Resize(height=256, width=256)  # 確保影像大小一致
    ])

    # 應用增強
    augmented_images = np.zeros_like(image_list)
    for idx in range(len(image_list)):
        augmented_images[idx] = transform(image=image_list[idx])['image']

    return augmented_images
def resize_image_to_256x256(image_path):
    dcm = pydicom.dcmread(image_path)
    image_data = dcm.pixel_array
    resized_image = cv2.resize(image_data, (256,256),interpolation=cv2.INTER_CUBIC) 
    normalized_image = resized_image / 255.0
    
    return np.array(normalized_image)
def list_files(directory):
    try:
        # 獲取目錄下的所有條目
        files = os.listdir(directory)
        return files
    except FileNotFoundError:
        return "指定的目錄不存在"
    except PermissionError:
        return "沒有權限訪問這個目錄"   
    
def get_data(df, available_list):
    data_point_list = []
    root_path = '/Volumes/G-DRIVE ArmorATD/MIT/mimic_all/p/'
    for row in df.itertuples():
        if row.path.split('/')[-1] in available_list:
            data_now = {}
            data_now['dicom'] = row.path.split('/')[-1]
            data_now['race'] = row.race
            data_now['gender'] = row.gender
            data_now['diagnose_label'] = row.diagnose_label 
            data_now['parameter_label'] = row.parameter_label
            data_now['parameters'] = [int(row.Exposure),int(row.ExposureInuAs),int(row.XRayTubeCurrent),int(row.ExposureTime)]
            data_now['image'] = resize_image_to_256x256(root_path+data_now['dicom'])
            data_point_list.append(data_now)
    return data_point_list
        

with open('./image_dic_list/diagnose_dic_list_Pneumothorax_98.pkl', 'rb') as f:
    diagnose_dic_list = pickle.load(f)
with open('./image_dic_list/no_finding_dic_Pneumothorax.pkl_98_all.pkl', 'rb') as f:
    no_finding_dic_list = pickle.load(f)
random.seed(42)
no_finding_dic_list = random.sample(no_finding_dic_list, len(diagnose_dic_list))
image_list = []
exp_label = []
cli_label = []
imb_label = []

for i in range(len(diagnose_dic_list)):
    image_list.append(diagnose_dic_list[i]['image'])
for i in range(len(no_finding_dic_list)):
    image_list.append(no_finding_dic_list[i]['image'])
    
for i in range(len(diagnose_dic_list)):
    exp_label.append(diagnose_dic_list[i]['parameter_label'])
for i in range(len(no_finding_dic_list)):
    exp_label.append(no_finding_dic_list[i]['parameter_label'])
    
for i in range(len(diagnose_dic_list)):
    cli_label.append(diagnose_dic_list[i]['diagnose_label'])
for i in range(len(no_finding_dic_list)):
    cli_label.append(no_finding_dic_list[i]['diagnose_label'])

for i in range(len(diagnose_dic_list)):
    if(diagnose_dic_list[i]['parameters'][2]<=320):
        if(exp_label[i]!=1):
            print("!!!!")
     
for i in range(len(no_finding_dic_list)):
    if(no_finding_dic_list[i]['parameters'][2]<=320):
        if(exp_label[len(diagnose_dic_list)+i]!=1):
            print("!!!!")

        
X = np.array(image_list) 
X = X.reshape((-1, 256, 256, 1)) 
y = np.array(cli_label) 



X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def build_resnet50_v2_model(input_shape=(256, 256, 1), num_classes=3):
    model = models.Sequential()
    
    model.add(layers.experimental.preprocessing.Resizing(224, 224, interpolation='bilinear'))
    
    model.add(layers.Conv2D(3, (3, 3), padding='same'))
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    model.add(base_model)
    
    model.add(layers.GlobalAveragePooling2D())
    
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model
model_resnet50_v2 = build_resnet50_v2_model()
model_resnet50_v2.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auroc', curve='ROC')  # AUROC 指標
    ]
)

exp_y = np.array(exp_label) 
X_train, X_temp, y_train, y_temp = train_test_split(X, exp_y, test_size=0.3, random_state=42)
X_val, X_test, exp_y_val, exp_y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

group_a_image_list =[]
group_a_cli_label_list = []
group_b_image_list =[]
group_b_cli_label_list = []


for i in range(len(exp_y_val)):
    if(exp_y_val[i]==1):
        group_a_image_list.append(X_val[i])
        group_a_cli_label_list.append(y_val[i])
    else:
        group_b_image_list.append(X_val[i])
        group_b_cli_label_list.append(y_val[i])
        
for i in range(len(exp_y_val)):
    if(exp_y_val[i]==1):
        group_a_image_list.append(X_test[i])
        group_a_cli_label_list.append(y_test[i])
    else:
        group_b_image_list.append(X_test[i])
        group_b_cli_label_list.append(y_test[i])
        
X = np.array(group_a_image_list) 
group_a_image = X.reshape((-1, 256, 256, 1)) 
group_a_cli_label= np.array(group_a_cli_label_list) 


X = np.array(group_b_image_list) 
group_b_image = X.reshape((-1, 256, 256, 1)) 
group_b_cli_label= np.array(group_b_cli_label_list) 
Counter(cli_label)



# 加載模型
#model_resnet50_v2 = load_model('./Models/Pneumothorax/Clinical_model_01_simple_aug/model_5')
base = 0
for i in range(10):
    #X_train_new = augment_images(X_train)
    X_train_new = (X_train)
    model_resnet50_v2.fit(X_train_new, y_train, epochs=1, validation_data=(X_val, y_val))
    print("test:")
    model_resnet50_v2.evaluate(X_test, y_test)
    print("group a:")
    model_resnet50_v2.evaluate(group_a_image, group_a_cli_label)
    print("group b:")
    model_resnet50_v2.evaluate(group_b_image, group_b_cli_label)
    model_resnet50_v2.save('./Models/Pneumothorax/Clinical_model_02/model_'+str(base+i)) 
    

