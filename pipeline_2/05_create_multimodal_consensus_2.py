#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo Bộ Dữ liệu Đồng thuận Đa tầng (Multimodal High-Confidence Consensus Dataset)
Chỉ giữ lại các mẫu (samples) đạt sự đồng thuận tuyệt đối giữa:
1. Nhãn Thị giác VLM (Qwen2.5-VL đọc ảnh X-quang)
2. Nhãn Văn bản Bác sĩ (Text Ground Truth từ Plan 1)
3. Nhãn Ban đầu (Old Labels từ luật cũ)
Bộ dữ liệu này loại bỏ 100% mọi tranh chấp và hoài nghi về nhãn.
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

CONSENSUS_DATASETS = [
    {
        "name": "TEST",
        "in_path": VLM_AUDITED_TEST_CSV,
        "out_path": os.path.join(DATA_DIR_2, "consensus_test_vlm.csv")
    },
    {
        "name": "TRAIN",
        "in_path": os.path.join(RELABEL_DIR_2, "clean_train_vlm.csv"),
        "out_path": os.path.join(DATA_DIR_2, "consensus_train_vlm.csv")
    },
    {
        "name": "VAL",
        "in_path": os.path.join(RELABEL_DIR_2, "clean_val_vlm.csv"),
        "out_path": os.path.join(DATA_DIR_2, "consensus_val_vlm.csv")
    }
]


def is_multimodal_consensus(row):
    """
    Kiểm tra sự trùng khớp 100% trên cả 9 nhãn giữa VLM (Ảnh) và các nguồn đối chứng
    """
    for lbl in LABELS_9:
        vlm_val = row.get(f"vlm_{lbl}", None)
        old_val = row.get(f"old_{lbl}", None)
        text_val = row.get(f"text_llm_{lbl}", None)

        if vlm_val is None:
            return False

        # Kiểm tra trùng với nhãn cũ
        if old_val is not None:
            try:
                if int(vlm_val) != int(old_val):
                    return False
            except (ValueError, TypeError):
                return False

        # Kiểm tra trùng với nhãn text bác sĩ (nếu có)
        if text_val is not None and not pd.isna(text_val):
            try:
                if int(vlm_val) != int(text_val):
                    return False
            except (ValueError, TypeError):
                return False

    return True


def run_create_multimodal_consensus():
    print("=" * 80)
    print(">>> TẠO BỘ DỮ LIỆU ĐỒNG THUẬN ĐA TẦNG (MULTIMODAL CONSENSUS DATASET)")
    print("=" * 80)

    summary_rows = []

    for item in CONSENSUS_DATASETS:
        name = item["name"]
        in_path = item["in_path"]
        out_path = item["out_path"]

        if not os.path.exists(in_path):
            print(f"[-] [Bỏ qua tập {name}]: Chưa có file nguồn {in_path}")
            continue

        print(f"\n[{name}] Đang lọc mẫu đồng thuận từ: {in_path} ...")
        df = pd.read_csv(in_path)
        total_orig = len(df)

        mask = df.apply(is_multimodal_consensus, axis=1)
        df_consensus = df[mask].copy().reset_index(drop=True)
        total_consensus = len(df_consensus)
        retain_rate = (total_consensus / total_orig * 100.0) if total_orig > 0 else 0.0

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_consensus.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"  -> Tổng số ca ban đầu : {total_orig:,}")
        print(f"  -> Số ca đồng thuận   : {total_consensus:,} ({retain_rate:.2f}%)")
        print(f"  -> Đã lưu tại         : {out_path}")

        class_counts = {}
        for lbl in LABELS_9:
            col = f"vlm_{lbl}"
            class_counts[lbl] = int(df_consensus[col].sum()) if col in df_consensus.columns else 0

        summary_rows.append({
            "Tập Dữ Liệu": name,
            "Số Ca Ban Đầu": total_orig,
            "Số Ca Đồng Thuận": total_consensus,
            "Tỷ Lệ Giữ Lại (%)": f"{retain_rate:.1f}%",
            **class_counts
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n" + "=" * 80)
        print("BẢNG TỔNG HỢP BỘ DỮ LIỆU ĐỒNG THUẬN ĐA TẦNG (MULTIMODAL CONSENSUS)")
        print("=" * 80)
        print(summary_df[["Tập Dữ Liệu", "Số Ca Ban Đầu", "Số Ca Đồng Thuận", "Tỷ Lệ Giữ Lại (%)"]].to_string(index=False))
        print("=" * 80)


if __name__ == "__main__":
    run_create_multimodal_consensus()
