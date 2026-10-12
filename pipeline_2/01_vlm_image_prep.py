#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 1 (Pipeline 2) - Tiền xử lý & Chuẩn bị Hình ảnh X-quang cho VLM
- Quét và kiểm tra tính toàn vẹn của file ảnh X-quang trên đĩa (C:/XQUANGRESIZE/...)
- Xử lý multi-image (1 hoặc 2 ảnh/ca)
- Lọc bỏ các ca thiếu ảnh hoặc đường dẫn hỏng
- Chuẩn hóa nhãn ban đầu thành vector nhị phân 0/1 để đối chiếu sau này
- Xuất dữ liệu sẵn sàng cho VLM vào Data_2/clean_test_vlm_ready.csv
"""

import os
import sys
import io
import ast
import argparse
import pandas as pd
from tqdm import tqdm

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_2.config_pipeline_2 import (
    LABELS_9,
    TEST_CSV,
    VLM_TEST_READY_CSV,
    IMAGE_ROOT,
    DATA_DIR_2,
    PLAN1_AUDITED_TEST_CSV,
)


def parse_image_paths(filepath_raw, image_root=IMAGE_ROOT):
    """
    Parse chuỗi filepath và kiểm tra sự tồn tại của từng file ảnh trên đĩa.
    Trả về danh sách các đường dẫn tuyệt đối hợp lệ.
    """
    try:
        if isinstance(filepath_raw, str):
            p_list = ast.literal_eval(filepath_raw)
        elif isinstance(filepath_raw, list):
            p_list = filepath_raw
        else:
            p_list = []
    except Exception:
        p_list = [str(filepath_raw)]

    valid_paths = []
    for p in p_list:
        p_str = str(p).strip().replace('\\', '/')
        full_path = os.path.join(image_root, p_str).replace('\\', '/')
        if os.path.exists(full_path):
            valid_paths.append(full_path)

    return valid_paths


def parse_old_labels(label_raw):
    """
    Chuyển đổi chuỗi label từ CSV thành dictionary nhị phân 0/1 cho 9 nhãn chuẩn
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


def run_image_prep(input_csv=TEST_CSV, output_csv=VLM_TEST_READY_CSV, image_root=IMAGE_ROOT):
    print("=" * 80)
    print(">>> GIAI ĐOẠN 1 (PIPELINE 2): IMAGE PREP & INTEGRITY VERIFICATION")
    print("=" * 80)

    if not os.path.exists(input_csv):
        print(f"[LỖI] Không tìm thấy file nguồn: {input_csv}")
        return None

    print(f"[1/4] Đọc dữ liệu từ: {input_csv}")
    df = pd.read_csv(input_csv)
    total_raw = len(df)
    print(f"      -> Tổng số ca ban đầu: {total_raw:,} ca")

    # 1. Kiểm tra sự tồn tại của file ảnh
    print(f"[2/4] Quét và kiểm tra file ảnh trên đĩa tại: {image_root} ...")
    valid_img_paths_list = []
    num_images_list = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Kiểm tra ảnh"):
        v_paths = parse_image_paths(row.get('filepath', ''), image_root=image_root)
        valid_img_paths_list.append(str(v_paths))
        num_images_list.append(len(v_paths))

    df['valid_images'] = valid_img_paths_list
    df['num_valid_images'] = num_images_list

    # Lọc bỏ các ca không có bất kỳ ảnh nào
    valid_mask = df['num_valid_images'] > 0
    df_clean = df[valid_mask].copy().reset_index(drop=True)
    dropped_count = total_raw - len(df_clean)

    print(f"      -> Số ca có ít nhất 1 ảnh hợp lệ : {len(df_clean):,} ca")
    print(f"      -> Số ca thiếu ảnh / đường dẫn lỗi: {dropped_count:,} ca")

    # Thống kê phân bố số ảnh trên mỗi ca
    count_1_img = int((df_clean['num_valid_images'] == 1).sum())
    count_2_img = int((df_clean['num_valid_images'] == 2).sum())
    count_more_img = int((df_clean['num_valid_images'] > 2).sum())
    print(f"      -> Phân bố số ảnh: 1 ảnh = {count_1_img:,} ca | 2 ảnh = {count_2_img:,} ca | >2 ảnh = {count_more_img:,} ca")

    # 2. Chuẩn hóa 9 nhãn ban đầu (old_labels)
    print("[3/4] Chuẩn hóa nhãn ban đầu sang vector nhị phân 0/1...")
    old_labels_df = df_clean['label'].apply(parse_old_labels).apply(pd.Series)
    df_clean = pd.concat([df_clean, old_labels_df], axis=1)

    # 3. Tích hợp nhãn chuẩn Text LLM từ Plan 1 (nếu có) để đối chiếu
    if os.path.exists(PLAN1_AUDITED_TEST_CSV):
        print(f"[4/4] Tích hợp nhãn Text LLM (Plan 1) từ {os.path.basename(PLAN1_AUDITED_TEST_CSV)} để đối chiếu chéo...")
        try:
            df_plan1 = pd.read_csv(PLAN1_AUDITED_TEST_CSV)
            plan1_cols = [f"llm_{l}" for l in LABELS_9 if f"llm_{l}" in df_plan1.columns]
            if "ketqua_ID" in df_clean.columns and "ketqua_ID" in df_plan1.columns and plan1_cols:
                sub_plan1 = df_plan1[["ketqua_ID"] + plan1_cols].rename(
                    columns={f"llm_{l}": f"text_llm_{l}" for l in LABELS_9}
                )
                df_clean = pd.merge(df_clean, sub_plan1, on="ketqua_ID", how="left")
                print("      -> Đã ghép thành công nhãn Text LLM Ground Truth.")
        except Exception as e:
            print(f"      -> [Bỏ qua ghép Plan 1 do lỗi nhẹ: {e}]")
    else:
        print("[4/4] Bỏ qua ghép Plan 1 (chưa tìm thấy audited_pipeline_test.csv).")

    # Lưu ra đĩa
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("-" * 80)
    print(f"THÀNH CÔNG! Dữ liệu sẵn sàng cho VLM đã lưu tại: {output_csv}")
    print(f"Kích thước DataFrame: {df_clean.shape}")
    print("=" * 80)
    return df_clean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giai đoạn 1 (Pipeline 2): Image Prep")
    parser.add_argument("--input_csv", type=str, default=TEST_CSV, help="File CSV nguồn")
    parser.add_argument("--output_csv", type=str, default=VLM_TEST_READY_CSV, help="File CSV đích")
    parser.add_argument("--image_root", type=str, default=IMAGE_ROOT, help="Thư mục gốc chứa ảnh")
    args = parser.parse_args()

    run_image_prep(args.input_csv, args.output_csv, args.image_root)
