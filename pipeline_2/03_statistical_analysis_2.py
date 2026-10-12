#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 3 (Pipeline 2) - Khai thác Thống kê Dịch tễ học Thị giác & Nhóm tuổi
- Tính toán Ma trận Đồng xuất hiện & Hệ số Cramér's V từ nhãn thị giác VLM
- Phân tích hiệu năng nhận diện tổn thương của VLM theo 3 nhóm tuổi (18-30, 31-50, 50+)
- Lưu kết quả vào Data_2/cramers_v_matrix_vlm.csv và Data_2/age_subgroup_f1_recall_vlm.csv
"""

import os
import sys
import io
import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_recall_fscore_support

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_2.config_pipeline_2 import (
    LABELS_9,
    VLM_AUDITED_TEST_CSV,
    VLM_CRAMERS_V_CSV,
    VLM_AGE_SUBGROUP_CSV,
)


def compute_cramers_v(x, y):
    """
    Tính chỉ số Cramér's V cho bảng tiếp liên 2x2 giữa hai biến nhị phân x và y.
    """
    conf_matrix = pd.crosstab(x, y)
    if conf_matrix.shape != (2, 2):
        return 0.0, 1.0
    chi2, p_val, dof, ex = chi2_contingency(conf_matrix, correction=False)
    n = len(x)
    v = np.sqrt(chi2 / n) if n > 0 else 0.0
    return float(v), float(p_val)


def run_statistical_analysis_vlm(input_csv=VLM_AUDITED_TEST_CSV):
    print("=" * 80)
    print(">>> GIAI ĐOẠN 3 (PIPELINE 2): KHAI THÁC THỐNG KÊ THỊ GIÁC & NHÓM TUỔI")
    print("=" * 80)

    if not os.path.exists(input_csv):
        print(f"[LỖI] Chưa tìm thấy file: {input_csv}. Vui lòng chạy Giai đoạn 2 trước.")
        return

    df = pd.read_csv(input_csv)
    print(f"Đã tải {len(df):,} ca từ {os.path.basename(input_csv)}")

    # 1. Tính toán Ma trận Đồng xuất hiện & Cramér's V
    print("\n[1/2] Tính toán Ma trận Đồng xuất hiện & Cramér's V từ Nhãn Thị giác VLM...")
    cramer_matrix = pd.DataFrame(1.0, index=LABELS_9, columns=LABELS_9)
    pval_matrix = pd.DataFrame(0.0, index=LABELS_9, columns=LABELS_9)
    pairs = []

    for i in range(len(LABELS_9)):
        for j in range(i + 1, len(LABELS_9)):
            l1, l2 = LABELS_9[i], LABELS_9[j]
            col1 = f"vlm_{l1}"
            col2 = f"vlm_{l2}"

            if col1 in df.columns and col2 in df.columns:
                v, p = compute_cramers_v(df[col1], df[col2])
            else:
                v, p = 0.0, 1.0

            cramer_matrix.loc[l1, l2] = v
            cramer_matrix.loc[l2, l1] = v
            pval_matrix.loc[l1, l2] = p
            pval_matrix.loc[l2, l1] = p

            pairs.append({
                "Bệnh 1": l1,
                "Bệnh 2": l2,
                "Cramér's V": round(v, 3),
                "p-value": round(p, 4),
                "Đồng xuất hiện mạnh?": "CÓ (V > 0.25)" if v > 0.25 else "Bình thường"
            })

    os.makedirs(os.path.dirname(VLM_CRAMERS_V_CSV), exist_ok=True)
    cramer_matrix.to_csv(VLM_CRAMERS_V_CSV, encoding='utf-8-sig')
    print(f"      -> Đã lưu ma trận Cramér's V VLM tại: {VLM_CRAMERS_V_CSV}")

    pairs_df = pd.DataFrame(pairs).sort_values(by="Cramér's V", ascending=False)
    print("\n[TOP 5 CẶP BỆNH ĐỒNG XUẤT HIỆN MẠNH NHẤT THEO NHẬN DIỆN CỦA VLM]")
    print(pairs_df.head(5).to_string(index=False))

    # 2. Phân tích Hiệu năng theo Nhóm tuổi
    print("\n[2/2] Phân tích Hiệu năng nhận diện của VLM theo Nhóm tuổi...")
    bins = [0, 30, 50, 120]
    age_labels = ["18-30 (Trẻ)", "31-50 (Trung niên)", "50+ (Cao tuổi)"]
    df['Age_Bin'] = pd.cut(df['Age'], bins=bins, labels=age_labels, right=True)

    has_text_gt = "text_llm_NORMAL" in df.columns
    age_reports = []

    for bin_name in age_labels:
        df_sub = df[df['Age_Bin'] == bin_name]
        n_cases = len(df_sub)
        if n_cases == 0:
            continue

        for lbl in LABELS_9:
            gt_col = f"text_llm_{lbl}" if has_text_gt else f"old_{lbl}"
            vlm_col = f"vlm_{lbl}"

            if gt_col not in df_sub.columns or vlm_col not in df_sub.columns:
                continue

            y_true = pd.to_numeric(df_sub[gt_col], errors='coerce').fillna(0).values.astype(int)
            y_pred = pd.to_numeric(df_sub[vlm_col], errors='coerce').fillna(0).values.astype(int)

            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            age_reports.append({
                "Nhóm Tuổi": bin_name,
                "Số ca": n_cases,
                "Nhãn Bệnh lý": lbl,
                "Precision": round(p, 3),
                "Recall": round(r, 3),
                "F1-Score": round(f1, 3)
            })

    age_df = pd.DataFrame(age_reports)
    os.makedirs(os.path.dirname(VLM_AGE_SUBGROUP_CSV), exist_ok=True)
    age_df.to_csv(VLM_AGE_SUBGROUP_CSV, index=False, encoding='utf-8-sig')
    print(f"      -> Đã lưu phân tích nhóm tuổi tại: {VLM_AGE_SUBGROUP_CSV}")

    pivot_f1 = age_df.pivot(index="Nhãn Bệnh lý", columns="Nhóm Tuổi", values="F1-Score")
    print("\n[BẢNG SO SÁNH F1-SCORE CỦA VLM THEO 3 NHÓM TUỔI]")
    print(pivot_f1.to_string())
    print("=" * 80)

    return cramer_matrix, age_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giai đoạn 3 (Pipeline 2): Thống kê VLM")
    parser.add_argument("--input_csv", type=str, default=VLM_AUDITED_TEST_CSV)
    args = parser.parse_args()

    run_statistical_analysis_vlm(args.input_csv)
