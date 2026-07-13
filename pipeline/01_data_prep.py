#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 1 - Chuẩn bị & Tiền xử lý (Data Prep)
- Lọc bỏ các ca bị thiếu text kết luận hoặc rác (< 5 ký tự)
- Chuẩn hóa 9 nhãn tự động cũ thành vector nhị phân 0/1
- Chạy batch inference bằng model VN-MultiXR lấy xác suất và nhãn dự đoán
- Hợp nhất thành 1 DataFrame duy nhất phục vụ các giai đoạn tiếp theo
"""

import os
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ast
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import cv2
from transformers import AutoTokenizer

# Import config từ pipeline
from pipeline.config_pipeline import (
    LABELS_9,
    LABEL2ID,
    DEFAULT_THRESHOLDS,
    PROJECT_ROOT,
    TEST_CSV,
    CLEAN_OUTPUT_CSV,
    CHECKPOINT_PATH,
    IMAGE_ROOT,
)
from src.model import ASSISTANCE_Model


class FastInferenceDataset(Dataset):
    """
    Dataset tối ưu cho batch inference Giai đoạn 1 (không augmentation ngẫu nhiên)
    """

    def __init__(self, df, tokenizer, img_size=224, max_images=2, max_text_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.img_size = img_size
        self.max_images = max_images
        self.max_text_len = max_text_len

        self.transform = A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Load Images
        paths = ast.literal_eval(row['filepath']) if isinstance(row['filepath'], str) else row['filepath']
        img_tensors = []
        for p in paths[:self.max_images]:
            full_path = os.path.join(IMAGE_ROOT, p)
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    aug = self.transform(image=img)
                    img_tensors.append(aug['image'])
                else:
                    img_tensors.append(torch.zeros((3, self.img_size, self.img_size)))
            else:
                img_tensors.append(torch.zeros((3, self.img_size, self.img_size)))

        while len(img_tensors) < self.max_images:
            img_tensors.append(torch.zeros((3, self.img_size, self.img_size)))
        images = torch.stack(img_tensors)

        # 2. Input Text (ketqua_Diagnose)
        text = str(row['ketqua_Diagnose'])
        if text == "nan" or text.strip() == "":
            text = "Không có chẩn đoán sơ bộ"
        inp_enc = self.tokenizer.encode_plus(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        # 3. Demographics
        age = float(row.get('Age', 50)) / 100.0
        sex = float(row.get('Sex', 0))
        demo = torch.tensor([age, sex], dtype=torch.float32)

        return {
            'images': images,
            'input_ids': inp_enc['input_ids'].squeeze(0),
            'attention_mask': inp_enc['attention_mask'].squeeze(0),
            'demographics': demo,
            'orig_idx': idx,
        }


def parse_old_labels(label_raw):
    """
    Chuyển đổi chuỗi label từ CSV (VD: "['NORMAL', 'FRACTURE']") thành dictionary nhị phân cho 9 nhãn chuẩn
    """
    try:
        if isinstance(label_raw, str):
            lbl_list = ast.literal_eval(label_raw)
        elif isinstance(label_raw, list):
            lbl_list = label_raw
        else:
            lbl_list = []
    except Exception:
        lbl_list = []

    res = {f"old_{l}": 0 for l in LABELS_9}
    for l in lbl_list:
        if l in LABELS_9:
            res[f"old_{l}"] = 1
    return res


def run_data_prep(input_csv=TEST_CSV, output_csv=CLEAN_OUTPUT_CSV, batch_size=32, num_workers=2):
    print("=" * 75)
    print(">>> GIAI ĐOẠN 1: DATA PREP & PREPROCESSING (VN-MultiXR INFERENCE)")
    print("=" * 75)

    # 1. Đọc dữ liệu gốc
    print(f"[1/5] Đọc dữ liệu từ: {input_csv}")
    df_raw = pd.read_csv(input_csv)
    raw_count = len(df_raw)

    # 2. Lọc bỏ các ca thiếu text kết luận hoặc rác (< 5 ký tự)
    print("[2/5] Lọc bỏ ca thiếu text kết luận hoặc rác...")
    valid_mask = (
        df_raw['clean_result'].notna()
        & (df_raw['clean_result'].astype(str).str.strip().str.len() >= 5)
    )
    df_clean = df_raw[valid_mask].copy().reset_index(drop=True)
    removed_count = raw_count - len(df_clean)
    print(f"      -> Tổng ca ban đầu : {raw_count}")
    print(f"      -> Ca rác/thiếu text bị lọc : {removed_count}")
    print(f"      -> Ca hợp lệ phân tích    : {len(df_clean)}")

    # 3. Chuẩn hóa 9 nhãn tự động cũ
    print("[3/5] Chuẩn hóa 9 nhãn tự động cũ thành vector nhị phân...")
    old_labels_df = df_clean['label'].apply(parse_old_labels).apply(pd.Series)
    df_clean = pd.concat([df_clean, old_labels_df], axis=1)

    # 4. Load Model VN-MultiXR & Chạy Batch Inference
    print(f"[4/5] Khởi tạo mô hình VN-MultiXR từ checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      -> Sử dụng thiết bị: {device}")

    model_config = {
        'architecture': {
            'image_encoder': 'swin_tiny_patch4_window7_224',
            'text_encoder': 'vinai/phobert-base-v2',
            'fusion_dim': 768,
            'dropout': 0.1,
            'num_layers_fusion': 2,
            'num_classes': len(LABELS_9),
        }
    }
    model = ASSISTANCE_Model(model_config)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    dataset = FastInferenceDataset(df_clean, tokenizer=tokenizer, img_size=224, max_images=2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_probs = []
    print("      -> Đang chạy suy luận (Batch Inference)...")
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference VN-MultiXR"):
            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            att_mask = batch['attention_mask'].to(device)
            demo = batch['demographics'].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                cls_logits, _ = model(images, input_ids, att_mask, demo)
                probs = torch.sigmoid(cls_logits).cpu().numpy()

            all_probs.append(probs)

    all_probs = np.vstack(all_probs)
    elapsed = time.time() - t0
    print(f"      -> Suy luận hoàn tất trong {elapsed:.1f}s ({len(df_clean)/elapsed:.1f} samples/s)")

    # 5. Gán kết quả dự đoán (probs + pred labels) vào DataFrame
    print("[5/5] Đóng gói và xuất ra DataFrame duy nhất...")
    for i, lbl in enumerate(LABELS_9):
        prob_col = f"pred_prob_{lbl}"
        pred_col = f"pred_{lbl}"
        thresh = DEFAULT_THRESHOLDS.get(lbl, 0.5)

        df_clean[prob_col] = np.round(all_probs[:, i], 4)
        df_clean[pred_col] = (all_probs[:, i] >= thresh).astype(int)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print("-" * 75)
    print(f"THÀNH CÔNG! Dữ liệu sạch đã được lưu tại: {output_csv}")
    print("Kích thước DataFrame:", df_clean.shape)

    # Thống kê nhanh nhãn cũ vs nhãn dự đoán
    print("\n[THỐNG KÊ NHANH TẦN SUẤT 9 NHÃN - TẬP SẠCH]")
    stats = []
    for lbl in LABELS_9:
        old_cnt = int(df_clean[f"old_{lbl}"].sum())
        pred_cnt = int(df_clean[f"pred_{lbl}"].sum())
        stats.append({
            "Nhãn (Disease)": lbl,
            "Nhãn cũ (Old Label)": old_cnt,
            "VN-MultiXR Đoán (Predicted)": pred_cnt,
            "Chênh lệch": pred_cnt - old_cnt
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df.to_string(index=False))
    print("=" * 75)
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giai đoạn 1: Data Prep & VN-MultiXR Inference")
    parser.add_argument("--input_csv", type=str, default=TEST_CSV, help="Đường dẫn file CSV đầu vào")
    parser.add_argument("--output_csv", type=str, default=CLEAN_OUTPUT_CSV, help="Đường dẫn file CSV đầu ra")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch inference")
    parser.add_argument("--num_workers", type=int, default=2, help="Số worker cho dataloader")
    args = parser.parse_args()

    run_data_prep(args.input_csv, args.output_csv, args.batch_size, args.num_workers)
