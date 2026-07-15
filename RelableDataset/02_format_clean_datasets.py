#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chuẩn hóa lại cột nhãn (`label`) theo định dạng chuỗi list (VD: "['DEGENERATION', 'SPINAL_ALIGNMENT']")
cho 3 tập dữ liệu đã được LLM gán nhãn: Train, Val, và Test.
Giúp dataset.py có thể đọc trực tiếp và huấn luyện/đánh giá mô hình VN-MultiXR v2.
"""

import os
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pipeline.config_pipeline import LABELS_9, DATA_DIR, PROJECT_ROOT

RELABEL_DIR = os.path.join(PROJECT_ROOT, "RelableDataset")

# Các file đầu vào (đã có cột 0/1 từ LLM)
INPUT_FILES = [
    {
        "name": "TRAIN",
        "in_path": os.path.join(RELABEL_DIR, "clean_train.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR, "clean_train.csv"), # Cập nhật trực tiếp
            os.path.join(RELABEL_DIR, "clean_train_formatted.csv")
        ],
        "prefix": "clean_"
    },
    {
        "name": "VAL",
        "in_path": os.path.join(RELABEL_DIR, "clean_val.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR, "clean_val.csv"), # Cập nhật trực tiếp
            os.path.join(RELABEL_DIR, "clean_val_formatted.csv")
        ],
        "prefix": "clean_"
    },
    {
        "name": "TEST",
        "in_path": os.path.join(DATA_DIR, "audited_pipeline_test.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR, "clean_test_formatted.csv"),
            os.path.join(DATA_DIR, "clean_test_formatted.csv")
        ],
        "prefix": "llm_"
    }
]


def format_binary_to_list_string(df, prefix):
    """
    Chuyển đổi các cột nhị phân prefix_LABEL (0 hoặc 1) thành chuỗi list "['LABEL1', 'LABEL2']"
    """
    new_labels = []
    text_col = "clean_result" if "clean_result" in df.columns else "ketqua_Diagnose"

    for idx, row in df.iterrows():
        active_labels = []
        for lbl in LABELS_9:
            col_name = f"{prefix}{lbl}"
            if col_name in df.columns and int(row[col_name]) == 1:
                active_labels.append(lbl)
        
        # Nếu toàn bộ 9 cột đều là 0 (fallback)
        if len(active_labels) == 0:
            text = str(row.get(text_col, "")).lower()
            if "bình thường" in text:
                active_labels.append("NORMAL")
            else:
                active_labels.append("OTHER")
                
        # Sắp xếp để nhất quán
        active_labels.sort()
        new_labels.append(str(active_labels))
        
    return new_labels


def run_formatting():
    print("=" * 80)
    print(">>> CHUẨN HÓA NHÃN (LABEL LIST FORMAT) CHO 3 TẬP TRAIN, VAL, TEST")
    print("=" * 80)

    for item in INPUT_FILES:
        name = item["name"]
        in_path = item["in_path"]
        prefix = item["prefix"]

        if not os.path.exists(in_path):
            print(f"[CẢNH BÁO] Không tìm thấy file {in_path} cho tập {name}. Bỏ qua.")
            continue

        print(f"\n[{name}] Đang xử lý file: {in_path} ...")
        df = pd.read_csv(in_path)

        # Kiểm tra xem có cột nhị phân LLM không
        binary_cols = [f"{prefix}{lbl}" for lbl in LABELS_9 if f"{prefix}{lbl}" in df.columns]
        if not binary_cols:
            # Thử tìm với prefix khác (VD: trong test có thể là clean_ hoặc ngược lại)
            alt_prefix = "llm_" if prefix == "clean_" else "clean_"
            alt_cols = [f"{alt_prefix}{lbl}" for lbl in LABELS_9 if f"{alt_prefix}{lbl}" in df.columns]
            if alt_cols:
                prefix = alt_prefix
                binary_cols = alt_cols
            else:
                print(f"  -> [LỖI] Không tìm thấy các cột nhị phân LLM ({prefix}*) trong {in_path}!")
                continue

        # Lưu lại nhãn cũ sang cột old_label nếu chưa có
        if "old_label" not in df.columns and "label" in df.columns:
            df["old_label"] = df["label"]

        # Tạo cột label mới theo định dạng string list
        new_label_list = format_binary_to_list_string(df, prefix)
        df["label"] = new_label_list

        # Thống kê nhanh vài mẫu
        print(f"  -> Đã chuẩn hóa {len(df):,} ca sang cột `label` (định dạng string list).")
        print(f"  -> 3 mẫu đầu tiên của cột `label` mới:")
        for i in range(min(3, len(df))):
            print(f"     Row {i}: {df['label'].iloc[i]} (từ text: {str(df['clean_result'].iloc[i])[:45]}...)")

        # Xuất ra các file đích
        for out_path in item["out_paths"]:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  -> Đã lưu tại: {out_path}")

    print("\n" + "=" * 80)
    print("THÀNH CÔNG! Cột `label` của cả 3 tập Train, Val, Test đã sẵn sàng cho `src/dataset.py`.")
    print("=" * 80)


if __name__ == "__main__":
    run_formatting()
