#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chuẩn hóa cột nhãn (label format: "['FRACTURE', 'NORMAL']") từ kết quả VLM Qwen2.5-VL
cho 3 tập dữ liệu: Train, Val và Test.
Giúp src/dataset.py có thể đọc trực tiếp để huấn luyện hoặc đánh giá mô hình.
"""

import os
import sys
import io
import argparse
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_2.config_pipeline_2 import (
    LABELS_9,
    DATA_DIR_2,
    RELABEL_DIR_2,
    VLM_AUDITED_TEST_CSV,
)

INPUT_FILES = [
    {
        "name": "TEST",
        "in_path": VLM_AUDITED_TEST_CSV,
        "out_paths": [
            os.path.join(RELABEL_DIR_2, "clean_test_vlm_formatted.csv"),
            os.path.join(DATA_DIR_2, "clean_test_vlm_formatted.csv")
        ]
    },
    {
        "name": "TRAIN",
        "in_path": os.path.join(RELABEL_DIR_2, "clean_train_vlm.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR_2, "clean_train_vlm_formatted.csv"),
            os.path.join(DATA_DIR_2, "clean_train_vlm_formatted.csv")
        ]
    },
    {
        "name": "VAL",
        "in_path": os.path.join(RELABEL_DIR_2, "clean_val_vlm.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR_2, "clean_val_vlm_formatted.csv"),
            os.path.join(DATA_DIR_2, "clean_val_vlm_formatted.csv")
        ]
    }
]


def format_binary_to_list_string(df, prefix="vlm_"):
    """
    Chuyển đổi 9 cột nhị phân prefix_LABEL (0 hoặc 1) thành chuỗi list "['LBL1', 'LBL2']"
    """
    new_labels = []
    for idx, row in df.iterrows():
        active = []
        for lbl in LABELS_9:
            col_name = f"{prefix}{lbl}"
            if col_name in df.columns and int(row[col_name]) == 1:
                active.append(lbl)

        if not active:
            active.append("NORMAL")

        active.sort()
        new_labels.append(str(active))
    return new_labels


def run_formatting_vlm():
    print("=" * 80)
    print(">>> CHUẨN HÓA ĐỊNH DẠNG CỘT LABEL (VLM LIST FORMAT) CHO DATASET.PY")
    print("=" * 80)

    for item in INPUT_FILES:
        name = item["name"]
        in_path = item["in_path"]

        if not os.path.exists(in_path):
            print(f"[-] [Bỏ qua tập {name}]: Chưa có file {in_path}")
            continue

        print(f"\n[{name}] Đang định dạng file: {in_path} ...")
        df = pd.read_csv(in_path)

        # Lưu lại nhãn ban đầu sang old_label nếu chưa có
        if "old_label" not in df.columns and "label" in df.columns:
            df["old_label"] = df["label"]

        df["label"] = format_binary_to_list_string(df, prefix="vlm_")
        print(f"  -> Đã chuẩn hóa {len(df):,} ca sang cột `label` (chuỗi list).")

        for out_path in item["out_paths"]:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  -> Đã lưu tại: {out_path}")

    print("\n" + "=" * 80)
    print("HOÀN TẤT CHUẨN HÓA ĐỊNH DẠNG DỮ LIỆU!")
    print("=" * 80)


if __name__ == "__main__":
    run_formatting_vlm()
