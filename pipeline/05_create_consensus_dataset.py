#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo bộ dữ liệu Đồng thuận (Consensus Dataset) cho 3 tập Train, Val và Test.
Chỉ giữ lại các mẫu (samples) có nhãn HOÀN TOÀN TRÙNG KHỚP giữa:
1. Bộ nhãn ban đầu (old_label / old_<lbl>)
2. Bộ nhãn do LLM Qwen2.5-7B gán (label / clean_<lbl> hoặc llm_<lbl>)

Bộ dữ liệu này đảm bảo độ tin cậy cực cao (High-Confidence Consensus Data) để huấn luyện
và đánh giá mô hình mà không lo bị nhiễu nhãn (label noise).
"""

import os
import sys
import io
import ast
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from pipeline.config_pipeline import LABELS_9, DATA_DIR, PROJECT_ROOT

RELABEL_DIR = os.path.join(PROJECT_ROOT, "RelableDataset")

DATASETS_TO_PROCESS = [
    {
        "name": "TRAIN",
        "in_path": os.path.join(RELABEL_DIR, "clean_train.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR, "consensus_train.csv"),
            os.path.join(DATA_DIR, "consensus_train.csv")
        ],
        "prefix": "clean_"
    },
    {
        "name": "VAL",
        "in_path": os.path.join(RELABEL_DIR, "clean_val.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR, "consensus_val.csv"),
            os.path.join(DATA_DIR, "consensus_val.csv")
        ],
        "prefix": "clean_"
    },
    {
        "name": "TEST",
        "in_path": os.path.join(RELABEL_DIR, "clean_test_formatted.csv"),
        "out_paths": [
            os.path.join(RELABEL_DIR, "consensus_test.csv"),
            os.path.join(DATA_DIR, "consensus_test.csv")
        ],
        "prefix": "llm_"
    }
]


def is_consensus_sample(row, prefix):
    """
    Kiểm tra sự trùng khớp tuyệt đối giữa nhãn ban đầu và nhãn LLM
    cả trên 9 cột nhị phân lẫn trên chuỗi list label.
    """
    # 1. Kiểm tra khớp 100% trên 9 cột nhị phân 0/1
    for lbl in LABELS_9:
        old_col = f"old_{lbl}"
        new_col = f"{prefix}{lbl}"
        if old_col in row and new_col in row:
            try:
                if int(row[old_col]) != int(row[new_col]):
                    return False
            except (ValueError, TypeError):
                return False

    # 2. Kiểm tra khớp chuỗi list label vs old_label
    if "label" in row and "old_label" in row:
        try:
            l1 = sorted(ast.literal_eval(row["label"]) if isinstance(row["label"], str) else row["label"])
            l2 = sorted(ast.literal_eval(row["old_label"]) if isinstance(row["old_label"], str) else row["old_label"])
            if l1 != l2:
                return False
        except Exception:
            return False

    return True


def run_create_consensus_dataset():
    print("=" * 80)
    print(">>> TẠO BỘ DỮ LIỆU ĐỒNG THUẬN (CONSENSUS DATASET: OLD LABELS == LLM LABELS)")
    print("=" * 80)

    summary_rows = []

    for item in DATASETS_TO_PROCESS:
        name = item["name"]
        in_path = item["in_path"]
        prefix = item["prefix"]
        out_paths = item["out_paths"]

        if not os.path.exists(in_path):
            # Thử kiểm tra fallback path trong Data/
            fallback_path = os.path.join(DATA_DIR, os.path.basename(in_path))
            if os.path.exists(fallback_path):
                in_path = fallback_path
            else:
                print(f"[CẢNH BÁO] Không tìm thấy file nguồn cho {name}: {in_path}. Bỏ qua.")
                continue

        print(f"\n[{name}] Đang đọc file nguồn: {in_path} ...")
        df = pd.read_csv(in_path)
        total_orig = len(df)

        # Lọc ra các mẫu đồng thuận
        mask = df.apply(lambda r: is_consensus_sample(r, prefix), axis=1)
        df_consensus = df[mask].copy().reset_index(drop=True)
        total_consensus = len(df_consensus)
        retain_rate = (total_consensus / total_orig * 100.0) if total_orig > 0 else 0.0

        print(f"  -> Tổng số ca ban đầu: {total_orig:,}")
        print(f"  -> Số ca đồng thuận (Trùng khớp 100% Nhãn Cũ & Nhãn LLM): {total_consensus:,} ({retain_rate:.2f}%)")

        # Lưu ra các file đích
        for out_path in out_paths:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            df_consensus.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"  -> Đã lưu file đồng thuận tại: {out_path}")

        # Thống kê phân phối theo từng lớp bệnh lý trong bộ đồng thuận
        class_counts = {}
        for lbl in LABELS_9:
            col = f"{prefix}{lbl}"
            if col in df_consensus.columns:
                class_counts[lbl] = int(df_consensus[col].sum())
            else:
                class_counts[lbl] = 0

        summary_rows.append({
            "Tập Dữ Liệu": name,
            "Số Ca Ban Đầu": total_orig,
            "Số Ca Đồng Thuận": total_consensus,
            "Tỷ Lệ Giữ Lại (%)": f"{retain_rate:.1f}%",
            "CARDIOMEGALY": class_counts.get("CARDIOMEGALY", 0),
            "DEGENERATION": class_counts.get("DEGENERATION", 0),
            "FRACTURE": class_counts.get("FRACTURE", 0),
            "NORMAL": class_counts.get("NORMAL", 0),
            "OTHER": class_counts.get("OTHER", 0),
            "PLEURAL_EFFUSION": class_counts.get("PLEURAL_EFFUSION", 0),
            "PNEUMONIA/INFILTRATION": class_counts.get("PNEUMONIA/INFILTRATION", 0),
            "SPINAL_ALIGNMENT": class_counts.get("SPINAL_ALIGNMENT", 0),
            "TUBERCULOSIS": class_counts.get("TUBERCULOSIS", 0)
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("BẢNG TỔNG HỢP BỘ DỮ LIỆU ĐỒNG THUẬN (CONSENSUS DATASET SUMMARY)")
    print("=" * 80)
    # Hiển thị bảng tóm tắt chung
    print(summary_df[["Tập Dữ Liệu", "Số Ca Ban Đầu", "Số Ca Đồng Thuận", "Tỷ Lệ Giữ Lại (%)"]].to_string(index=False))
    print("\n[PHÂN PHỐI SỐ LƯỢNG CA DƯƠNG TÍNH THEO BỆNH LÝ TRONG BỘ ĐỒNG THUẬN]")
    for _, row in summary_df.iterrows():
        print(f"\n--- {row['Tập Dữ Liệu']} (Tổng {row['Số Ca Đồng Thuận']:,} ca) ---")
        for lbl in LABELS_9:
            print(f"  + {lbl:<25}: {row[lbl]:,} ca")
    print("=" * 80)


if __name__ == "__main__":
    run_create_consensus_dataset()
