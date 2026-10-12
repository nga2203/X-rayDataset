#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 6 (Pipeline 2): Tạo Bộ Dữ Liệu Đồng Thuận Riêng Giữa Nhãn Gốc và Plan 2 (VLM)
(Dual Consensus: Old Labels == VLM Vision Labels)

Đặc điểm:
1. Tiêu chí chọn lọc 2 bên thuần túy:
   - Chỉ giữ lại các ca có sự trùng khớp 100% giữa Nhãn cũ của bệnh viện (old_*) và Nhãn thị giác VLM (vlm_*)
     trên toàn bộ 9 nhãn bệnh lý.
   - Hoàn toàn độc lập với Plan 1 (không cần tham chiếu nhãn text bác sĩ).
2. Chuẩn hóa định dạng cột `label`:
   - Định dạng list string (ví dụ: "['NORMAL']", "['DEGENERATION']") tương thích trực tiếp với src/dataset.py.
   - Bảo toàn cột `old_label` gốc và toàn bộ các cột nhị phân old_* và vlm_*.
3. Xuất ra bộ dữ liệu riêng biệt:
   - Data_2/consensus_vlm_old/ (consensus_train.csv, consensus_val.csv, consensus_test.csv)
   - RelableDataset_2/consensus_vlm_old/
   - Xuất file log tóm tắt: log_2/06_consensus_vlm_old.log
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
    LOG_DIR_2,
    VLM_AUDITED_TEST_CSV,
)

# Thư mục riêng lưu trữ bộ dữ liệu đồng thuận VLM vs Gốc
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR_2, "consensus_vlm_old")
SYNC_RELABEL_DIR = os.path.join(RELABEL_DIR_2, "consensus_vlm_old")

DATASET_CONFIGS = [
    {
        "name": "TEST",
        "in_path": VLM_AUDITED_TEST_CSV,
        "filename": "consensus_test.csv"
    },
    {
        "name": "TRAIN",
        "in_path": os.path.join(RELABEL_DIR_2, "clean_train_vlm.csv"),
        "filename": "consensus_train.csv"
    },
    {
        "name": "VAL",
        "in_path": os.path.join(RELABEL_DIR_2, "clean_val_vlm.csv"),
        "filename": "consensus_val.csv"
    }
]


def check_old_vlm_consensus(row):
    """
    Kiểm tra sự trùng khớp tuyệt đối 100% giữa Nhãn Cũ (old_*) và Nhãn VLM (vlm_*)
    trên toàn bộ 9 nhãn bệnh lý.
    """
    for lbl in LABELS_9:
        old_val = row.get(f"old_{lbl}", None)
        vlm_val = row.get(f"vlm_{lbl}", None)

        if old_val is None or vlm_val is None:
            return False

        try:
            if int(old_val) != int(vlm_val):
                return False
        except (ValueError, TypeError):
            return False

    return True


def format_labels_to_list_string(row):
    """
    Tạo chuỗi nhãn định dạng list "['NORMAL']" hoặc "['DEGENERATION', 'FRACTURE']"
    từ các nhãn đồng thuận nhị phân.
    """
    active_labels = []
    for lbl in LABELS_9:
        col = f"vlm_{lbl}"
        if col in row and int(row[col]) == 1:
            active_labels.append(lbl)

    if not active_labels:
        active_labels.append("NORMAL")

    active_labels.sort()
    return str(active_labels)


def run_create_vlm_old_consensus(output_dir=DEFAULT_OUTPUT_DIR, sync_dir=SYNC_RELABEL_DIR):
    os.makedirs(output_dir, exist_ok=True)
    if sync_dir:
        os.makedirs(sync_dir, exist_ok=True)
    os.makedirs(LOG_DIR_2, exist_ok=True)

    log_lines = []
    def log_print(msg=""):
        print(msg)
        log_lines.append(msg)

    log_print("=" * 80)
    log_print(">>> TẠO BỘ DỮ LIỆU ĐỒNG THUẬN RIÊNG: NHÃN GỐC == PLAN 2 (VLM)")
    log_print(f"    Thư mục xuất chính : {output_dir}")
    if sync_dir:
        log_print(f"    Thư mục đồng bộ    : {sync_dir}")
    log_print("=" * 80)

    summary_rows = []
    class_breakdown = {}

    for cfg in DATASET_CONFIGS:
        name = cfg["name"]
        in_path = cfg["in_path"]
        fname = cfg["filename"]

        if not os.path.exists(in_path):
            log_print(f"[-] [Bỏ qua tập {name}]: Không tìm thấy file {in_path}")
            continue

        log_print(f"\n[{name}] Đang lọc mẫu đồng thuận từ: {in_path} ...")
        df = pd.read_csv(in_path)
        total_orig = len(df)

        mask = df.apply(check_old_vlm_consensus, axis=1)
        df_consensus = df[mask].copy().reset_index(drop=True)
        total_consensus = len(df_consensus)
        retain_rate = (total_consensus / total_orig * 100.0) if total_orig > 0 else 0.0

        # Lưu lại nhãn gốc sang old_label nếu chưa có
        if "old_label" not in df_consensus.columns and "label" in df_consensus.columns:
            df_consensus["old_label"] = df_consensus["label"]

        # Chuẩn hóa cột `label` thành chuỗi list chuẩn
        df_consensus["label"] = df_consensus.apply(format_labels_to_list_string, axis=1)

        # Lưu vào thư mục đích
        out_main = os.path.join(output_dir, fname)
        df_consensus.to_csv(out_main, index=False, encoding="utf-8-sig")
        log_print(f"  -> Tổng số ca ban đầu : {total_orig:,}")
        log_print(f"  -> Số ca đồng thuận   : {total_consensus:,} ({retain_rate:.2f}%)")
        log_print(f"  -> Đã lưu tại         : {out_main}")

        if sync_dir:
            out_sync = os.path.join(sync_dir, fname)
            df_consensus.to_csv(out_sync, index=False, encoding="utf-8-sig")
            log_print(f"  -> Đồng bộ tại        : {out_sync}")

        # Đếm phân bố các nhãn bệnh lý
        counts = {}
        for lbl in LABELS_9:
            col = f"vlm_{lbl}"
            counts[lbl] = int(df_consensus[col].sum()) if col in df_consensus.columns else 0
        class_breakdown[name] = counts

        summary_rows.append({
            "Tập Dữ Liệu": name,
            "Số Ca Ban Đầu": total_orig,
            "Số Ca Đồng Thuận": total_consensus,
            "Tỷ Lệ Giữ Lại (%)": f"{retain_rate:.1f}%",
            "File": fname
        })

    # In bảng tổng hợp
    log_print("\n" + "=" * 80)
    log_print("BẢNG TỔNG HỢP BỘ DỮ LIỆU ĐỒNG THUẬN (OLD LABELS == VLM LABELS)")
    log_print("=" * 80)
    summary_df = pd.DataFrame(summary_rows)
    log_print(summary_df[["Tập Dữ Liệu", "Số Ca Ban Đầu", "Số Ca Đồng Thuận", "Tỷ Lệ Giữ Lại (%)", "File"]].to_string(index=False))

    # In phân bố chi tiết số ca dương tính
    log_print("\n[PHÂN PHỐI SỐ LƯỢNG CA THEO BỆNH LÝ TRONG BỘ ĐỒNG THUẬN (GỐC == VLM)]")
    for name in ["TRAIN", "VAL", "TEST"]:
        if name in class_breakdown:
            counts = class_breakdown[name]
            total_set = [r["Số Ca Đồng Thuận"] for r in summary_rows if r["Tập Dữ Liệu"] == name][0]
            log_print(f"\n--- {name} (Tổng {total_set:,} ca) ---")
            for lbl in LABELS_9:
                log_print(f"  + {lbl:<25}: {counts.get(lbl, 0):,} ca")

    log_print("=" * 80)

    # Xuất file log
    log_file = os.path.join(LOG_DIR_2, "06_consensus_vlm_old.log")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n-> Đã lưu log thống kê đầy đủ tại: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo bộ dữ liệu đồng thuận riêng giữa Nhãn Gốc và Plan 2 (VLM)")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Thư mục lưu trữ bộ data riêng")
    parser.add_argument("--sync_dir", type=str, default=SYNC_RELABEL_DIR, help="Thư mục phụ đồng bộ dữ liệu")
    parser.add_argument("--no_sync", action="store_true", help="Không đồng bộ sang RelableDataset_2")
    args = parser.parse_args()

    sync_path = None if args.no_sync else args.sync_dir
    run_create_vlm_old_consensus(output_dir=args.output_dir, sync_dir=sync_path)
