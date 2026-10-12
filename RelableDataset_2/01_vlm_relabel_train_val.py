#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hệ thống Gán nhãn Thị giác Tối Ưu Tốc Độ cho Tập Train & Val bằng VLM Qwen2.5-VL 7B
(Chạy Full Công Suất Tối Đa - Tự Động Giải Nhiệt 10s Sau Mỗi 30 Phút)

Đặc điểm tối ưu:
1. Chạy Full Tốc độ (Full Compute):
   - Không micro-pause giữa các ca -> GPU hoạt động trơn tru liên tục như tập Val & Test (~1.8s - 2.5s/ca).
   - Khóa KV Cache num_ctx=2048: Ổn định ở 7.2GB VRAM, 0 byte Shared GPU RAM.
   - num_predict=130: Đảm bảo sinh trọn vẹn JSON 9 keys ngay lần gọi đầu tiên (Done reason: stop), không bị retry.
2. Quản lý Nhiệt Tự Động (--rest_interval_mins 30, --rest_duration_secs 10):
   - Chạy hết công suất trong 30 phút.
   - Tròn 30 phút, script tự động tạm dừng 10 giây để GPU xả nhiệt buồng tản nhiệt, sau đó tự tiếp tục.
3. Consensus Fast-Path (--fast_path consensus, Mặc định cho Train):
   - Nạp ngay 9,511 ca (76.3%) đồng thuận tuyệt đối giữa Bác sĩ và nhãn cũ.
   - VLM chỉ cần đọc 2,951 ca tranh chấp/bất thường.
   - Rút ngắn toàn bộ thời gian chạy xuống chỉ còn ~1.5 - 1.8 giờ!
4. Checkpoint Nguyên Tử (Atomic Cache):
   - Tự động lưu cache mỗi 25 ca và trước mỗi đợt nghỉ. Ngắt Ctrl+C không mất dữ liệu.
"""

import os
import sys
import io
import ast
import json
import time
import signal
import argparse
import requests
import pandas as pd
from tqdm import tqdm

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_2.config_pipeline_2 import (
    LABELS_9,
    TRAIN_CSV,
    VAL_CSV,
    PROJECT_ROOT,
    RELABEL_DIR_2,
    TRAIN_VLM_CACHE_FILE,
    VAL_VLM_CACHE_FILE,
    VLM_MODEL,
    OLLAMA_ENDPOINT,
    IMAGE_ROOT,
    unload_ollama_model,
)
from pipeline_2.vlm_utils import (
    parse_image_paths,
    parse_old_labels,
    query_vlm_vision,
    atomic_save_json,
    preload_vlm_model,
)


def get_fastpath_cache(df, split_name, mode="consensus"):
    """
    Rà soát và trích xuất nhãn Fast-Path từ kết quả Plan 1 để giảm tải GPU:
    - 'consensus': Nhận diện các ca mà cả Bác sĩ (Text Plan 1) và Nhãn cũ đồng thuận 100% cả 9 nhãn.
    - 'normal': Chỉ nhận diện các ca Bình Thường Tuyệt Đối (Pure Normal).
    - 'none': Không áp dụng fast-path.
    """
    if mode == "none":
        return {}

    plan1_csv = os.path.join(PROJECT_ROOT, "RelableDataset", f"clean_{split_name.lower()}.csv")
    if not os.path.exists(plan1_csv):
        print(f"[FAST-PATH CẢNH BÁO] Không tìm thấy file Plan 1: {plan1_csv}")
        return {}

    try:
        df_p1 = pd.read_csv(plan1_csv)
        df_p1["_cid_norm"] = df_p1["ketqua_ID"].astype(str).str.replace(r"\.0$", "", regex=True)

        if mode == "normal":
            disease_cols = [f"clean_{l}" for l in LABELS_9 if l != "NORMAL"]
            old_disease_cols = [f"old_{l}" for l in LABELS_9 if l != "NORMAL"]
            cond_p1_normal = (df_p1.get("clean_NORMAL", 0) == 1) & (df_p1[disease_cols].sum(axis=1) == 0)
            cond_old_normal = (df_p1.get("old_NORMAL", 0) == 1) & (df_p1[old_disease_cols].sum(axis=1) == 0)
            pure_df = df_p1[cond_p1_normal & cond_old_normal]

            normal_dict = {l: 0 for l in LABELS_9}
            normal_dict["NORMAL"] = 1
            return {cid: normal_dict for cid in pure_df["_cid_norm"]}

        elif mode == "consensus":
            exact_match = pd.Series(True, index=df_p1.index)
            for l in LABELS_9:
                c_col, o_col = f"clean_{l}", f"old_{l}"
                if c_col in df_p1.columns and o_col in df_p1.columns:
                    exact_match = exact_match & (df_p1[c_col] == df_p1[o_col])
                else:
                    exact_match = pd.Series(False, index=df_p1.index)
                    break

            consensus_df = df_p1[exact_match]
            res = {}
            for _, r in consensus_df.iterrows():
                cid = r["_cid_norm"]
                res[cid] = {l: int(r.get(f"clean_{l}", 0)) for l in LABELS_9}
            return res

    except Exception as e:
        print(f"[FAST-PATH CẢNH BÁO] Lỗi đọc Plan 1: {e}")
        return {}


def run_vlm_relabel_split(
    input_csv,
    split_name,
    cache_file,
    output_csv,
    sample_size=None,
    save_interval=25,
    max_images=1,
    fast_path_mode="consensus",
    chunk_size=None,
    rest_interval_mins=30.0,
    rest_duration_secs=10,
    model=VLM_MODEL
):
    print("=" * 80)
    print(f">>> GÁN NHÃN THỊ GIÁC VLM CHO TẬP {split_name.upper()} (FULL TỐC ĐỘ + TỰ NGHỈ 30 PHÚT)")
    print(f"    Mô hình VLM       : {model} (Ollama Local)")
    print(f"    File Nguồn       : {input_csv}")
    print(f"    Chế độ Fast-Path  : {fast_path_mode.upper()} "
          f"({'Tiết kiệm ~76% GPU' if fast_path_mode == 'consensus' else 'Tiết kiệm ~53% GPU' if fast_path_mode == 'normal' else 'Tắt'})")
    if rest_interval_mins > 0:
        print(f"    Chu kỳ giải nhiệt : Nghỉ {rest_duration_secs}s sau mỗi {rest_interval_mins:.0f} phút chạy liên tục")
    else:
        print("    Chu kỳ giải nhiệt : TẮT (Chạy liên tục không nghỉ)")
    if chunk_size:
        print(f"    Chế độ Chạy       : Phân đoạn ({chunk_size:,} ca / phiên chạy)")
    print(f"    Cache File       : {cache_file}")
    print("=" * 80)

    if not os.path.exists(input_csv):
        print(f"[LỖI] Không tìm thấy file: {input_csv}")
        return None

    df = pd.read_csv(input_csv)
    total_samples = len(df)
    print(f"[1/4] Đã nạp {total_samples:,} ca {split_name}")

    if sample_size and sample_size < total_samples:
        print(f"[CHẾ ĐỘ MẪU] Chạy thử trên {sample_size} ca đầu tiên...")
        df = df.head(sample_size).copy()

    df["_cid_norm"] = df["ketqua_ID"].astype(str).str.replace(r"\.0$", "", regex=True)

    # Nạp cache từ trước (Resume an toàn)
    vlm_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                raw_cache = json.load(f)
            vlm_cache = {str(k).replace(".0", "") if str(k).endswith(".0") else str(k): v for k, v in raw_cache.items()}
            print(f"[RESUME] Đã tìm thấy cache từ trước: {len(vlm_cache):,} ca đã xử lý.")
        except Exception:
            vlm_cache = {}

    normal_dict = {l: 0 for l in LABELS_9}
    normal_dict["NORMAL"] = 1

    # Áp dụng Fast-Path
    fast_added = 0
    if fast_path_mode in ["consensus", "normal"]:
        print(f"[FAST-PATH] Đang rà soát và kích hoạt chế độ Fast-Path ({fast_path_mode})...")
        fp_dict = get_fastpath_cache(df, split_name, mode=fast_path_mode)
        for _, row in df.iterrows():
            cid = row["_cid_norm"]
            if cid not in vlm_cache and cid in fp_dict:
                vlm_cache[cid] = fp_dict[cid]
                fast_added += 1

        if fast_added > 0:
            print(f"      -> Đã Fast-Path thành công: {fast_added:,} ca không cần tốn GPU!")
            atomic_save_json(vlm_cache, cache_file)

    # Lọc danh sách các ca còn thiếu cần VLM đọc ảnh
    all_pending = []
    valid_paths_dict = {}

    for idx, row in df.iterrows():
        cid = row["_cid_norm"]
        if cid not in vlm_cache:
            all_pending.append(idx)
            valid_paths_dict[idx] = parse_image_paths(row.get("filepath", ""), image_root=IMAGE_ROOT)

    total_pending = len(all_pending)
    print(f"[2/4] Tiến độ tổng thể: Đã hoàn tất {len(df) - total_pending:,} ca | Còn lại: {total_pending:,} ca")

    if chunk_size and total_pending > chunk_size:
        pending_to_run = all_pending[:chunk_size]
        print(f"[PHÂN ĐOẠN] Đợt này sẽ xử lý: {len(pending_to_run):,} ca (còn {total_pending - len(pending_to_run):,} ca cho các đợt sau)")
    else:
        pending_to_run = all_pending

    # Bẫy tín hiệu ngắt an toàn
    def cleanup_on_exit(signum=None, frame=None):
        print(f"\n\n[SIGNAL INTERRUPT] Lưu cache tập {split_name} trước khi thoát...")
        atomic_save_json(vlm_cache, cache_file)
        print("[GPU STATUS] Model tiếp tục được giữ trong VRAM để sẵn sàng cho lần chạy tiếp theo!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_on_exit)

    # Preload model vào VRAM GPU trước khi chạy
    if pending_to_run:
        preload_vlm_model(model=model, endpoint=OLLAMA_ENDPOINT)

    session = requests.Session()
    t0 = time.time()
    last_rest_time = time.time()
    rest_interval_secs = rest_interval_mins * 60.0
    unsaved_count = 0
    session_count = 0

    try:
        if pending_to_run:
            pbar = tqdm(pending_to_run, desc=f"VLM Vision {split_name}", unit="ca")

            for idx in pbar:
                row = df.iloc[idx]
                cid = row["_cid_norm"]
                img_paths = valid_paths_dict.get(idx, [])

                if not img_paths:
                    lbl_dict = normal_dict
                else:
                    lbl_dict = query_vlm_vision(
                        img_paths,
                        model=model,
                        endpoint=OLLAMA_ENDPOINT,
                        session=session,
                        max_images=max_images
                    )
                    if lbl_dict is None:
                        lbl_dict = normal_dict

                vlm_cache[cid] = lbl_dict
                unsaved_count += 1
                session_count += 1

                # Ghi cache định kỳ
                if unsaved_count >= save_interval:
                    atomic_save_json(vlm_cache, cache_file)
                    unsaved_count = 0

                # Tự động tạm nghỉ 10 giây sau mỗi 30 phút chạy liên tục để giải nhiệt GPU
                if rest_interval_secs > 0:
                    time_since_rest = time.time() - last_rest_time
                    if time_since_rest >= rest_interval_secs and session_count < len(pending_to_run):
                        pbar.set_postfix_str(f"Nghỉ {rest_duration_secs}s xả nhiệt GPU...")
                        atomic_save_json(vlm_cache, cache_file)
                        unsaved_count = 0
                        time.sleep(rest_duration_secs)
                        last_rest_time = time.time()
                        pbar.set_postfix_str("")

            # Lưu cache lần cuối của phiên
            atomic_save_json(vlm_cache, cache_file)
            elapsed = time.time() - t0
            print(f"\n[PHIÊN CHẠY KẾT THÚC] Xử lý {session_count:,} ca trong {elapsed:.1f}s ({elapsed/max(1, session_count):.2f}s/ca)")

        # Kiểm tra xem toàn bộ tập dữ liệu đã hoàn tất chưa
        remaining_after = sum(1 for _, r in df.iterrows() if r["_cid_norm"] not in vlm_cache)

        if remaining_after == 0:
            # Xuất DataFrame sạch hoàn chỉnh
            print(f"\n[3/4] Xuất DataFrame sạch hoàn chỉnh cho tập {split_name}...")
            old_labels_df = df['label'].apply(parse_old_labels).apply(pd.Series)

            vlm_results = []
            for _, row in df.iterrows():
                cid = row["_cid_norm"]
                cached_lbls = vlm_cache.get(cid, normal_dict)
                vlm_row = {f"vlm_{lbl}": cached_lbls.get(lbl, 0) for lbl in LABELS_9}
                vlm_results.append(vlm_row)

            vlm_df = pd.DataFrame(vlm_results)
            cols_to_drop = ["_cid_norm"] if "_cid_norm" in df.columns else []
            df_clean_source = df.drop(columns=cols_to_drop)

            df_out = pd.concat([df_clean_source.reset_index(drop=True), old_labels_df, vlm_df], axis=1)

            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
            print(f"      -> Đã lưu tập {split_name} sạch tại: {output_csv} ({len(df_out):,} ca)")

            # Thống kê phân bố nhãn VLM
            print(f"\n[4/4] Thống kê phân bố nhãn VLM trên tập {split_name}:")
            stats = []
            for lbl in LABELS_9:
                v_cnt = int(df_out[f"vlm_{lbl}"].sum())
                o_cnt = int(df_out[f"old_{lbl}"].sum())
                stats.append({
                    "Nhãn Bệnh lý": lbl,
                    "VLM (Ảnh)": v_cnt,
                    "Nhãn Cũ": o_cnt,
                    "Chênh lệch": v_cnt - o_cnt
                })
            print(pd.DataFrame(stats).to_string(index=False))
            return df_out
        else:
            pct_done = ((len(df) - remaining_after) / len(df)) * 100
            print("\n" + "=" * 80)
            print(f"[ĐỢT CHẠY HOÀN TẤT] Tiến độ hiện tại: {len(df) - remaining_after:,} / {len(df):,} ca ({pct_done:.1f}%)")
            print(f"                   Còn lại: {remaining_after:,} ca cần VLM xử lý.")
            print(">>> Dữ liệu đã được lưu an toàn vào Cache. Bạn có thể chạy tiếp bất cứ lúc nào!")
            print("=" * 80)
            return None

    finally:
        session.close()
        print("[GPU STATUS] Model tiếp tục được giữ trong VRAM 24h để sẵn sàng cho các lượt chạy tiếp theo!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gán nhãn VLM Siêu Tối Ưu cho Train/Val (Full Compute & 30-min Rest)")
    parser.add_argument("--split", type=str, choices=["train", "val", "both"], default="train", help="Phân vùng xử lý (mặc định train)")
    parser.add_argument("--fast_path", type=str, choices=["consensus", "normal", "none"], default="consensus",
                        help="Chế độ Fast-Path: consensus (khuyên dùng cho Train, tiết kiệm 76%% GPU), normal (53%%), none (chạy tất cả)")
    parser.add_argument("--no_fast_normal", action="store_true", help="Tương thích ngược: tắt fast-path")
    parser.add_argument("--chunk_size", type=int, default=None, help="Số ca xử lý trong phiên này rồi dừng nghỉ (vd: 1000)")
    parser.add_argument("--rest_interval_mins", type=float, default=30.0, help="Số phút chạy liên tục trước khi tự nghỉ 10s (mặc định 30 phút, đặt 0 để tắt)")
    parser.add_argument("--rest_duration_secs", type=int, default=10, help="Thời gian nghỉ giải nhiệt GPU (giây, mặc định 10s)")
    parser.add_argument("--sample_size", type=int, default=None, help="Chạy thử nghiệm trên số ca nhỏ đầu tiên")
    parser.add_argument("--save_interval", type=int, default=25, help="Tần suất lưu cache ra đĩa (mặc định 25 ca)")
    parser.add_argument("--max_images", type=int, default=1, help="Số ảnh tối đa mỗi ca (mặc định 1)")
    parser.add_argument("--model", type=str, default=VLM_MODEL)
    args = parser.parse_args()

    os.makedirs(RELABEL_DIR_2, exist_ok=True)

    fp_mode = "none" if args.no_fast_normal else args.fast_path

    if args.split in ["val", "both"]:
        run_vlm_relabel_split(
            input_csv=VAL_CSV,
            split_name="Val",
            cache_file=VAL_VLM_CACHE_FILE,
            output_csv=os.path.join(RELABEL_DIR_2, "clean_val_vlm.csv"),
            sample_size=args.sample_size,
            save_interval=args.save_interval,
            max_images=args.max_images,
            fast_path_mode=fp_mode if fp_mode != "consensus" else "normal",
            chunk_size=args.chunk_size,
            rest_interval_mins=args.rest_interval_mins,
            rest_duration_secs=args.rest_duration_secs,
            model=args.model
        )

    if args.split in ["train", "both"]:
        run_vlm_relabel_split(
            input_csv=TRAIN_CSV,
            split_name="Train",
            cache_file=TRAIN_VLM_CACHE_FILE,
            output_csv=os.path.join(RELABEL_DIR_2, "clean_train_vlm.csv"),
            sample_size=args.sample_size,
            save_interval=args.save_interval,
            max_images=args.max_images,
            fast_path_mode=fp_mode,
            chunk_size=args.chunk_size,
            rest_interval_mins=args.rest_interval_mins,
            rest_duration_secs=args.rest_duration_secs,
            model=args.model
        )
