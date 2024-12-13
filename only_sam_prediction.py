import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
import pydicom
import cv2
import nibabel as nib
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random

import automatic_mask_generator
from automatic_mask_generator import SAM2AutomaticMaskGenerator
import sam2_base

from sam2_base import SAM2Base

import memory_attention
import memory_encoder
import image_encoder
import hieradet
import position_encoding
import transformer

from transformer import RoPEAttention
from hieradet import Hiera
from memory_attention import MemoryAttention, MemoryAttentionLayer
from memory_encoder import MemoryEncoder
from image_encoder import ImageEncoder,FpnNeck
from position_encoding import PositionEmbeddingSine
from memory_encoder import MaskDownSampler
from memory_encoder import Fuser
from memory_encoder import CXBlock

class ImageDataset(Dataset):
    def __init__(self, data, labels, clinical_data):
        self.data = data
        self.labels = labels
        self.clinical_data = clinical_data  

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #totensor
        data_tensor = torch.tensor(self.data[idx], dtype=torch.float32).squeeze(0)
        label_tensor = torch.tensor(self.labels[idx].astype(int), dtype=torch.long)
        clinical_tensor = torch.tensor(self.clinical_data[idx].values, dtype=torch.float32)
        #test nan
        if torch.isnan(data_tensor).any() or torch.isnan(clinical_tensor).any():
            print(f"NaN found at index {idx}, skipping this sample.")
        clinical_tensor = F.normalize(clinical_tensor, p=2, dim=0)  

        return data_tensor, label_tensor, clinical_tensor  


class CNNClassifier(nn.Module):
    def __init__(self, in_channel, clinical_input_size):
        super(CNNClassifier, self).__init__()

        self.img_extractor = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.backbone = nn.Sequential(
            nn.Linear(64 + clinical_input_size, 128),
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(64, 32) 
        self.fc2 = nn.Linear(32, 1) 

    def forward(self, img, clinical_data):
        # feature
        img_feat = self.img_extractor(img)
        #print(f"img_feat: {img_feat.shape}")

        img_feat_pooled = self.global_avg_pool(img_feat).squeeze(-1).squeeze(-1)
        #print(f"img_feat_pool: {img_feat_pooled.shape}")
        img_feat_pooled= F.normalize(img_feat_pooled, p=2, dim=0) 
        # combine cl and feature
        combined_feat  = torch.cat([img_feat_pooled, clinical_data], dim=-1)  
        #print(f"Combined input shape: {combined_feat .shape}")

        x = self.backbone(combined_feat )

        x = self.fc1(x)
        x = self.fc2(x)

        return torch.sigmoid(x)  # Sigmoid

def check_for_nans(tensor):
    # check NaN
    return torch.isnan(tensor).any()



def extract_features_and_labels(ct_folder, mask_folder, merged_df, encoder, device):
    save_path = "/onlysam_processed_241_data.pt"
    output_file = "/otrain_folder_processed.txt"
    patient_folders = []

    # Check if the file exists and open it to read
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            # Read all lines and strip the newline characters
            patient_folders = [line.rstrip('\n') for line in f.readlines()]
        # Print the list of patient folders
        print("Patient Folders loaded length:", len(patient_folders))
    else:
        print(f"File {output_file} does not exist.")

    # Load the saved data
    if os.path.exists(save_path):
        loaded_data = torch.load(save_path)

        # Assign the data to the variables
        features = loaded_data["features"]
        labels = loaded_data["labels"]
        clinical_data = loaded_data["clinic"]
    else:
        features = []
        labels = []
        clinical_data = []

    X = merged_df.drop(columns=['imageName', 'Group', 'set', 'ORR'])  # 剔除非特征列
    
    for patient_folder in os.listdir(ct_folder):
        if patient_folder in patient_folders:
            print(f"already processed:{patient_folder}")
            continue
        if patient_folder.startswith('.') or not os.path.isdir(os.path.join(ct_folder, patient_folder)):
            continue

        # label
        if patient_folder not in merged_df['imageName'].values:
            continue


        group_label = merged_df.loc[merged_df['imageName'] == patient_folder, 'Group'].values[0]
        clinical_info = X.loc[merged_df['imageName'] == patient_folder].iloc[0]
        clinical_info= clinical_info.drop('Unnamed: 0')

        ct_folder_path = os.path.join(ct_folder, patient_folder)
        mask_path = os.path.join(mask_folder, patient_folder)
        print(f"image encoder for patient {ct_folder_path}")

        # find mask
        if not os.path.exists(mask_path) or len(os.listdir(mask_path)) != 1:
            continue
        
        mask_file = os.listdir(mask_path)[0]
        mask_data = nib.load(os.path.join(mask_path, mask_file)).get_fdata()
        
        #find the slice
        max_mask=0
        for dicom_file in sorted(os.listdir(ct_folder_path)):
            dicom_file_path = os.path.join(ct_folder_path, dicom_file)
            if not os.path.isfile(dicom_file_path):
                continue

            # getmask
            slice_index = int(dicom_file.split('_')[-1].split('.')[0]) - 1
            if slice_index < 0 or slice_index >= mask_data.shape[-1]:
                continue

            mask_slice = mask_data[:, :, slice_index]
            if np.sum(mask_slice > 0) >max_mask:
                max_mask=np.sum(mask_slice > 0)
                max_dicom_file=dicom_file
            
        if max_mask==0:
            print(f"no mask file{patient_folder}")
            continue

        dicom_file=max_dicom_file
        dicom_file_path = os.path.join(ct_folder_path, dicom_file)
        slice_index = int(dicom_file.split('_')[-1].split('.')[0]) - 1
        mask_slice = mask_data[:, :, slice_index]
        #mask_resized = Image.fromarray(mask_slice.astype(np.float32)).resize((64, 64), Image.BILINEAR)
        #mask_tensor = torch.as_tensor(np.array(mask_resized).astype('float')).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 64, 64]
        #  mask -- vision_features 
        #mask_expanded = mask_tensor.expand(-1, 256, -1, -1)  

        # read file
        dicom_data = pydicom.dcmread(dicom_file_path)
            
        # convert
        image_array = dicom_data.pixel_array.astype(np.uint8) 
        image = Image.fromarray(image_array).convert("L")  
        image_rgb = Image.merge("RGB", (image, image, image)) 

        image_resized = image_rgb.resize((1024, 1024), Image.BILINEAR)

        # to tensor
        img_tensor = torch.as_tensor(np.array(image_resized).astype('float')).permute(2, 0, 1).unsqueeze(0).to(device)

        # get feature from sam
        img_tensor = img_tensor.float()
        vision_features = encoder(img_tensor)["vision_features"]  # [1, 256, 1024, 1024]

        combined_features = vision_features 
        print(f"combined_features shape: {combined_features.shape}") 
        #combined_features_flat = combined_features.flatten(start_dim=1)    

        features.append(combined_features)
        labels.append(group_label)
        clinical_data.append(clinical_info)

        torch.save({
        "features": features,
        "labels": labels,
        "clinic": clinical_data
        }, save_path)

        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:  
                f.write(f"{patient_folder}\n") 
            print(f"File {output_file} created.")

        else:
            with open(output_file, 'a') as f:
                folder_summary = f"{patient_folder}\n"
                f.write(folder_summary)

        print(f"{patient_folder} processed data saved to {save_path},{output_file}")

    return features, labels,clinical_data
source_ct_folder = '/241_image/'
source_mask_folder = '/241_mask/'
excel_path_ct_first = '/cl_features_cleaned_ct_first_image.xlsx'
excel_path_ct_241 = '/cl_features_cleaned_241.xlsx'
cl_features_CT_first = pd.read_excel(excel_path_ct_first, engine='openpyxl')
cl_features_CT_241 = pd.read_excel(excel_path_ct_241, engine='openpyxl')
merged_df_all = pd.concat([cl_features_CT_first, cl_features_CT_241], axis=0, ignore_index=True)
merged_df_all = merged_df_all[merged_df_all['Group'].isin([0, 1])]

# convert names
categorical_features = ['gender', 'smoking status', 'PDL1_expression', 'Pathological diagnosis', 'total stage']

# dic
category_mappings = {}

for feature in categorical_features:
    if feature in merged_df_all.columns:  
        
        merged_df_all[feature] = merged_df_all[feature].astype('category')

        categories = merged_df_all[feature].cat.categories
        merged_df_all[feature] = merged_df_all[feature].cat.codes

        # save
        category_mappings[feature] = {index: category for index, category in enumerate(categories)}

for feature, mapping in category_mappings.items():
    print(f'Feature: {feature}')
    print('Mapping:', mapping)
    print('---------------------------')

#sam model
trunk = Hiera(embed_dim=96,num_heads=1,stages=[1, 2, 11, 2], global_att_blocks=[7, 10, 13], window_pos_embed_bkg_spatial_size=[7, 7])
position_encoding = PositionEmbeddingSine(num_pos_feats=256,normalize=True, scale=None, temperature=10000)
neck = FpnNeck(
    position_encoding=position_encoding,
    d_model=256,
    backbone_channel_list=[768, 384, 192, 96],
    fpn_top_down_levels=[2, 3],
    fpn_interp_model= 'nearest'
)
image_encoder = ImageEncoder(trunk=trunk, neck=neck, scalp=1)

#load pretrained model
try:
    checkpoint_path = "/sam2_hiera_small.pt"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    image_encoder.load_state_dict(checkpoint['model'], strict=False)
    print("Checkpoint loaded successfully with strict=False.")
except Exception as e:
    print(f"An error occurred while loading checkpoint: {e}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_encoder = image_encoder.to(device)

#generate data for train
features, labels,clinic = extract_features_and_labels(
    source_ct_folder, source_mask_folder, merged_df_all, image_encoder, device
)

save_path = "/onlysam_processed_241_data.pt"
torch.save({
    "features": features,
    "labels": labels,
    "clinic": clinic
}, save_path)

print(f"Processed data saved to {save_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "/onlysam_processed_241_data.pt"

# get data
try:
    loaded_data = torch.load(data_path)
    print("Data loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading data: {e}")
    loaded_data = None

if loaded_data:
    features = loaded_data['features']
    labels = loaded_data['labels']
    clinic = loaded_data['clinic']


for i in range(len(clinic)):
    try:
        clinic[i] = clinic[i].drop('PDL1_number')
    except:
        print(i,'no nan')

# split data
x_train, x_test, y_train, y_test, cli_train, cli_test = train_test_split(features, labels, clinic, test_size=0.2, random_state=42)
print("x_train size:", len(x_train))
print("x_test size:", len(x_test))
train_dataset = ImageDataset(x_train, y_train, cli_train)
test_dataset = ImageDataset(x_test, y_test, cli_test)

#  DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#init model
model = CNNClassifier(in_channel=256, clinical_input_size=8).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# train
all_train_accuracies = []
all_test_accuracies = []
all_train_losses = []
all_test_losses = []
all_fpr = []
all_tpr = []
all_roc_auc = []

epochs = 40
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for img, labels, clinical_data in train_loader:
        img, labels, clinical_data = img.to(device), labels.to(device).float(), clinical_data.to(device)

        optimizer.zero_grad()
        outputs = model(img, clinical_data)
        loss = criterion(outputs.squeeze(), labels)  # 使用squeeze()去掉多余维度
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    all_outputs = []
    with torch.no_grad():
        for img, labels, clinical_data in test_loader:
            img, labels, clinical_data = img.to(device), labels.to(device).float(), clinical_data.to(device)

            outputs = model(img, clinical_data)
            outputs = outputs.squeeze() 
            predicted = (outputs > 0.5).long() 
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)

    # auc 
    fpr, tpr, _ = roc_curve(all_labels, all_outputs)
    roc_auc = auc(fpr, tpr)

    all_train_losses.append(train_loss / len(train_loader))  # 平均训练损失
    all_test_losses.append(test_loss / len(test_loader))    # 平均测试损失
    all_train_accuracies.append(accuracy)
    all_test_accuracies.append(correct / len(test_dataset))
    all_fpr.append(fpr)
    all_tpr.append(tpr)
    all_roc_auc.append(roc_auc)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    print(f"Validation Loss: {test_loss / len(test_loader):.4f}, Accuracy: {correct / len(test_dataset):.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {roc_auc:.4f}")

plt.figure(figsize=(12, 5))

# accuracy figure
plt.subplot(1, 3, 1)
plt.plot(range(1, epochs + 1), all_train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs + 1), all_test_accuracies, label="Test Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()

# loss figure
plt.subplot(1, 3, 2)
plt.plot(range(1, epochs + 1), all_train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), all_test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss per Epoch")
plt.legend()

# roc
plt.subplot(1, 3, 3)
plt.plot(range(len(all_roc_auc)), all_roc_auc, label='AUC Score')
plt.xlabel("Epoch")
plt.ylabel("AUC Score")
plt.ylim(0)
plt.legend()
plt.grid()
plt.show()