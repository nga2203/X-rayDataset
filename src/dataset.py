import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import json
import ast
import os
from transformers import AutoTokenizer
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = "C:/"

class ASSISTANCEDataset(Dataset):
    def __init__(self, csv_path, class_mapping_path, img_size=256, max_images=3, 
                 tokenizer_name="vinai/phobert-base-v2", max_text_len=128, transform=None):
        
        self.df = pd.read_csv(csv_path)
        with open(class_mapping_path, 'r', encoding='utf-8') as f:
            self.label2id = json.load(f)
        self.num_classes = len(self.label2id)
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_text_len = max_text_len
        self.max_images = max_images
        self.img_size = img_size

        if transform is None:
            # self.transform = A.Compose([
            #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            #     ToTensorV2(),
            # ])
            self.transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            self.transform = transform
            
        self.resize_base = A.Compose([A.Resize(height=img_size, width=img_size)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Images
        paths = ast.literal_eval(row['filepath']) if isinstance(row['filepath'], str) else row['filepath']
        img_tensors = []
        for p in paths[:self.max_images]:
            p = os.path.join(ROOT, p)
            if os.path.exists(p):
                # print(f"Loading image: {p}")
                img = cv2.imread(p)
                if img is not None:
                    # print(f"Image shape before processing: {img.shape}")
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = self.resize_base(image=img)['image']
                    aug = self.transform(image=img)
                    img_tensors.append(aug['image'])
                else:
                    # print(f"Warning: Image at {p} could not be read. Using blank image.")
                    img_tensors.append(torch.zeros((3, self.img_size, self.img_size)))
            else:
                # print(f"Warning: Image path {p} does not exist. Using blank image.")
                img_tensors.append(torch.zeros((3, self.img_size, self.img_size)))
        
        while len(img_tensors) < self.max_images:
            img_tensors.append(torch.zeros((3, self.img_size, self.img_size)))
            
        images = torch.stack(img_tensors)

        # 2. Input Text
        text = str(row['ketqua_Diagnose'])
        if text == "nan": text = ""
        inp_enc = self.tokenizer.encode_plus(
            text, max_length=self.max_text_len, padding='max_length', truncation=True, return_tensors='pt'
        )

        # 3. Target Text (Report)
        target_text = str(row['clean_result'])
        if target_text == "nan": target_text = "Mô tả bình thường"
        tgt_enc = self.tokenizer.encode_plus(
            target_text, max_length=100, padding='max_length', truncation=True, return_tensors='pt'
        )

        # 4. Demo & Label
        age = float(row['Age']) / 100.0
        sex = float(row['Sex'])
        demo = torch.tensor([age, sex], dtype=torch.float32)
        
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        lbl_list = ast.literal_eval(row['label']) if isinstance(row['label'], str) else row['label']
        for lbl in lbl_list:
            if lbl in self.label2id:
                label_vec[self.label2id[lbl]] = 1.0
                
        return {
            'images': images,
            'input_ids': inp_enc['input_ids'].squeeze(0),
            'attention_mask': inp_enc['attention_mask'].squeeze(0),
            'target_ids': tgt_enc['input_ids'].squeeze(0),
            'demographics': demo,
            'labels': label_vec
        }

if __name__ == "__main__":
    try:
        dataset = ASSISTANCEDataset(
            csv_path='data/dataset_train_lab03.csv',
            class_mapping_path='data/class_mapping_lab03.json'
        )
        print(f"Dataset length: {len(dataset)}")
        sample = dataset[10]
        print("Sample keys:", sample.keys())
        print("Images shape:", sample['images'].shape) # [3, 3, 256, 256]
        print("Input IDs shape:", sample['input_ids'].shape)
        print("Target IDs shape:", sample['target_ids'].shape)
        print("Labels:", sample['labels'])
        print("DATASET TEST PASSED!")

        # Save a sample image for visual inspection
        img_tensor = sample['images'][0]  # First image in the set
        img_np = img_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0  # Denormalize
        img_np = img_np.astype(np.uint8) 
        cv2.imwrite('data/sample_image.png', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        print("Sample image saved as 'data/sample_image.png'")
    except Exception as e:
        print(f"DATASET TEST FAILED: {e}")