#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 3 - Luồng 2 & 3: Khai thác Thống kê
- Đồng xuất hiện (Co-occurrence): Lập ma trận tương quan giữa 9 bệnh.
  Dùng scipy.stats.chi2_contingency tính p-value & Cramér's V.
- Phân tích nhóm tuổi: Chia tuổi thành các bin (18-30, 31-50, 50+)
  Dùng classification_report / sklearn tính Recall/F1 của model VN-MultiXR trên từng nhóm tuổi.
"""

import os
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import precision_recall_fscore_support, classification_report

from pipeline.config_pipeline import (
    LABELS_9,
    DATA_DIR,
)

AUDIT_INPUT_CSV = os.path.join(DATA_DIR, "audited_pipeline_test.csv")
CRAMERS_V_OUTPUT_CSV = os.path.join(DATA_DIR, "cramers_v_matrix_8.csv")
AGE_SUBGROUP_OUTPUT_CSV = os.path.join(DATA_DIR, "age_subgroup_f1_recall_8.csv")


def compute_cramers_v(x, y):
    """
    Tính chỉ số Cramér's V cho bảng 2x2 giữa 2 biến nhị phân x và y
    """
    conf_matrix = pd.crosstab(x, y)
    if conf_matrix.shape != (2, 2):
        return 0.0, 1.0
    chi2, p_val, dof, ex = chi2_contingency(conf_matrix, correction=False)
    n = len(x)
    v = np.sqrt(chi2 / n)
    return float(v), float(p_val)


def run_statistical_analysis(input_csv=AUDIT_INPUT_CSV):
    print("=" * 75)
    print(">>> GIAI ĐOẠN 3: KHAI THÁC THỐNG KÊ (SCIPY & MATPLOTLIB)")
    print("=" * 75)

    if not os.path.exists(input_csv):
        print(f"[LỖI] Không tìm thấy file: {input_csv}. Vui lòng chạy Giai đoạn 2 trước.")
        return

    df = pd.read_csv(input_csv)
    print(f"Đã tải {len(df)} ca từ {os.path.basename(input_csv)}")

    # -------------------------------------------------------------------------
    # LUỒNG 2: MA TRẬN ĐỒNG XUẤT HIỆN & CRAMÉR'S V
    # -------------------------------------------------------------------------
    print("\n[1/2] Tính toán Ma trận Đồng xuất hiện & Cramér's V giữa 8 bệnh...")
    cramer_matrix = pd.DataFrame(1.0, index=LABELS_9, columns=LABELS_9)
    pval_matrix = pd.DataFrame(0.0, index=LABELS_9, columns=LABELS_9)
    pairs = []

    for i in range(len(LABELS_9)):
        for j in range(i + 1, len(LABELS_9)):
            l1, l2 = LABELS_9[i], LABELS_9[j]
            # Sử dụng nhãn chuẩn từ LLM (llm_<label>) hoặc fallback về old_<label> nếu chưa có
            col1 = f"llm_{l1}" if f"llm_{l1}" in df.columns else f"old_{l1}"
            col2 = f"llm_{l2}" if f"llm_{l2}" in df.columns else f"old_{l2}"

            v, p = compute_cramers_v(df[col1], df[col2])
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

    cramer_matrix.to_csv(CRAMERS_V_OUTPUT_CSV, encoding='utf-8-sig')
    print(f"-> Đã lưu ma trận Cramér's V tại: {CRAMERS_V_OUTPUT_CSV}")

    pairs_df = pd.DataFrame(pairs).sort_values(by="Cramér's V", ascending=False)
    print("\n[TOP 5 CẶP BỆNH ĐỒNG XUẤT HIỆN MẠNH NHẤT TRONG DỮ LIỆU]")
    print(pairs_df.head(5).to_string(index=False))

    # -------------------------------------------------------------------------
    # LUỒNG 3: PHÂN TÍCH THEO NHÓM TUỔI (18-30, 31-50, 50+)
    # -------------------------------------------------------------------------
    print("\n[2/2] Phân tích Hiệu năng model VN-MultiXR theo Nhóm tuổi...")
    bins = [0, 30, 50, 120]
    labels = ["18-30 (Trẻ)", "31-50 (Trung niên)", "50+ (Cao tuổi)"]
    df['Age_Bin'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    age_reports = []
    for bin_name in labels:
        df_sub = df[df['Age_Bin'] == bin_name]
        n_cases = len(df_sub)
        if n_cases == 0:
            continue

        for lbl in LABELS_9:
            y_true_col = f"llm_{lbl}" if f"llm_{lbl}" in df_sub.columns else f"old_{lbl}"
            y_pred_col = f"pred_{lbl}"

            y_true = df_sub[y_true_col].values
            y_pred = df_sub[y_pred_col].values

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
    age_df.to_csv(AGE_SUBGROUP_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"-> Đã lưu kết quả phân tích nhóm tuổi tại: {AGE_SUBGROUP_OUTPUT_CSV}")

    # Hiển thị bảng tóm tắt F1 theo nhóm tuổi cho 9 bệnh
    pivot_f1 = age_df.pivot(index="Nhãn Bệnh lý", columns="Nhóm Tuổi", values="F1-Score")
    print("\n[BẢNG SO SÁNH F1-SCORE CỦA VN-MultiXR THEO 3 NHÓM TUỔI]")
    print(pivot_f1.to_string())
    print("=" * 75)
    return cramer_matrix, age_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giai đoạn 3: Khai thác Thống kê")
    parser.add_argument("--input_csv", type=str, default=AUDIT_INPUT_CSV)
    args = parser.parse_args()

    run_statistical_analysis(args.input_csv)
