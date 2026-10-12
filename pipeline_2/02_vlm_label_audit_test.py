#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 2 (Pipeline 2) - Gán nhãn Thị giác Tập TEST bằng VLM Qwen2.5-VL 7B
Tối ưu hóa Siêu tốc & Chống OOM:
- Preload / Warm-up nạp sẵn model vào GPU VRAM trước khi vào vòng lặp
- Tái sử dụng HTTP Persistent Session (requests.Session)
- Giới hạn tối đa 2 ảnh/ca (phim thẳng + phim nghiêng)
- Checkpointing an toàn (Atomic JSON save)
- Tự động giải phóng VRAM (keep_alive: 0) khi kết thúc hoặc ngắt (Ctrl+C)
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
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pipeline_2.config_pipeline_2 import (
    LABELS_9,
    VLM_TEST_READY_CSV,
    VLM_AUDITED_TEST_CSV,
    TEST_VLM_CACHE_FILE,
    VLM_MODEL,
    OLLAMA_ENDPOINT,
    unload_ollama_model,
)
from pipeline_2.vlm_utils import (
    query_vlm_vision,
    atomic_save_json,
    preload_vlm_model,
)


def run_vlm_label_audit_test(
    input_csv=VLM_TEST_READY_CSV,
    output_csv=VLM_AUDITED_TEST_CSV,
    cache_file=TEST_VLM_CACHE_FILE,
    sample_size=None,
    save_interval=20,
    max_images=1,
    clear_cache=False,
    unload_at_exit=False,
    model=VLM_MODEL
):
    print("=" * 80)
    print(">>> GIAI ĐOẠN 2 (PIPELINE 2): VLM VISION LABEL AUDIT (TẬP TEST - SIÊU TỐI ƯU)")
    print(f"    Mô hình VLM : {model} (Ollama Local)")
    print(f"    Max Images  : {max_images} ảnh/ca")
    print(f"    Cache File  : {cache_file}")
    print("=" * 80)

    if not os.path.exists(input_csv):
        print(f"[LỖI] Chưa có file: {input_csv}. Vui lòng chạy 01_vlm_image_prep.py trước.")
        return

    df = pd.read_csv(input_csv)
    total_samples = len(df)
    print(f"[1/4] Đã nạp {total_samples:,} ca kiểm thử từ {os.path.basename(input_csv)}")

    # Xóa cache nếu có yêu cầu
    if clear_cache and os.path.exists(cache_file):
        os.remove(cache_file)
        print("[CACHE RESET] Đã xóa cache cũ theo yêu cầu.")

    # Giới hạn số lượng mẫu nếu có cờ sample_size
    if sample_size and sample_size < total_samples:
        print(f"[CHẾ ĐỘ MẪU] Chạy thử nghiệm trên {sample_size} ca đầu tiên...")
        df = df.head(sample_size).copy()
    else:
        print(f"[CHẾ ĐỘ TOÀN BỘ] Sẽ xử lý toàn bộ {len(df):,} ca...")

    # Nạp cache đã lưu từ trước (Resume capability)
    vlm_cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                vlm_cache = json.load(f)
            print(f"[RESUME] Đã tìm thấy cache từ trước: {len(vlm_cache):,} ca đã hoàn thành.")
        except Exception:
            vlm_cache = {}

    # Kiểm tra kết nối Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code != 200:
            print("[LỖI] Không thể kết nối tới Ollama Server tại http://localhost:11434. Vui lòng bật Ollama.")
            return
        print("[OLLAMA STATUS] Kết nối Ollama thành công.")
    except Exception as e:
        print(f"[LỖI] Không kết nối được tới Ollama: {e}")
        return

    # Lọc ra các ca chưa có trong cache
    pending_indices = []
    for idx, row in df.iterrows():
        case_id = str(row.get("ketqua_ID", idx))
        if case_id not in vlm_cache:
            pending_indices.append(idx)

    print(f"[2/4] Tiến độ cache: Đã có {len(df) - len(pending_indices):,} ca | Còn lại cần chạy: {len(pending_indices):,} ca")

    # Thiết lập cơ chế lưu checkpoint khi bị ngắt
    def cleanup_on_exit(signum=None, frame=None):
        print("\n\n[SIGNAL INTERRUPT] Đang lưu lại cache an toàn trước khi thoát...")
        atomic_save_json(vlm_cache, cache_file)
        print(f"[CHECKPOINT] Đã bảo toàn {len(vlm_cache):,} ca vào: {cache_file}")
        if unload_at_exit:
            unload_ollama_model(model=model)
        else:
            print("[GPU STATUS] Model tiếp tục được giữ trong VRAM để sẵn sàng cho lần chạy tiếp theo!")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_on_exit)

    # PRELOAD MODEL VÀO VRAM TRƯỚC KHI VÀO VÒNG LẶP ĐỂ KHÔNG BỊ COLD START TRONG TQDM
    if pending_indices:
        preload_vlm_model(model=model, endpoint=OLLAMA_ENDPOINT)

    session = requests.Session()
    t0 = time.time()
    unsaved_count = 0

    try:
        if pending_indices:
            pbar = tqdm(pending_indices, desc="VLM Vision Auditing", unit="ca")
            for idx in pbar:
                row = df.iloc[idx]
                case_id = str(row.get("ketqua_ID", idx))

                # Lấy danh sách ảnh hợp lệ
                try:
                    v_imgs = ast.literal_eval(row["valid_images"]) if isinstance(row["valid_images"], str) else row["valid_images"]
                except Exception:
                    v_imgs = []

                if not v_imgs:
                    lbl_dict = {l: 0 for l in LABELS_9}
                    lbl_dict["NORMAL"] = 1
                else:
                    lbl_dict = query_vlm_vision(
                        v_imgs,
                        model=model,
                        endpoint=OLLAMA_ENDPOINT,
                        session=session,
                        max_images=max_images
                    )
                    if lbl_dict is None:
                        lbl_dict = {l: 0 for l in LABELS_9}
                        lbl_dict["NORMAL"] = 1

                vlm_cache[case_id] = lbl_dict
                unsaved_count += 1

                # Checkpoint tự động sau mỗi save_interval ca
                if unsaved_count >= save_interval:
                    atomic_save_json(vlm_cache, cache_file)
                    unsaved_count = 0

            # Lưu lần cuối
            atomic_save_json(vlm_cache, cache_file)
            elapsed = time.time() - t0
            print(f"\n[SUY LUẬN HOÀN TẤT] Thời gian chạy: {elapsed:.1f}s ({elapsed/max(1, len(pending_indices)):.2f}s/ca)")

        # 3. Ánh xạ kết quả VLM vào DataFrame
        print("\n[3/4] Đóng gói kết quả VLM vào DataFrame...")
        vlm_results = []
        for idx, row in df.iterrows():
            case_id = str(row.get("ketqua_ID", idx))
            cached_lbls = vlm_cache.get(case_id, {l: 0 for l in LABELS_9})
            vlm_row = {f"vlm_{lbl}": cached_lbls.get(lbl, 0) for lbl in LABELS_9}
            vlm_results.append(vlm_row)

        vlm_df = pd.DataFrame(vlm_results)
        df_out = pd.concat([df.reset_index(drop=True), vlm_df], axis=1)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"      -> Đã xuất kết quả kiểm định VLM tại: {output_csv} ({len(df_out):,} ca)")

        # 4. Thống kê & Báo cáo so sánh sơ bộ
        print("\n" + "=" * 85)
        print("BẢNG ĐỐI CHIẾU SƠ BỘ: NHÃN THỊ GIÁC VLM (ẢNH) VS NHÃN BÁC SĨ (TEXT GROUND TRUTH)")
        print("=" * 85)
        metrics_list = []
        has_text_gt = "text_llm_NORMAL" in df_out.columns

        for lbl in LABELS_9:
            y_vlm = df_out[f"vlm_{lbl}"].values
            y_old = df_out[f"old_{lbl}"].values

            gt_col = f"text_llm_{lbl}" if has_text_gt else f"old_{lbl}"
            y_gt = pd.to_numeric(df_out[gt_col], errors='coerce').fillna(0).values.astype(int)
            y_vlm = pd.to_numeric(df_out[f"vlm_{lbl}"], errors='coerce').fillna(0).values.astype(int)

            vlm_cnt = int(np.sum(y_vlm))
            gt_cnt = int(np.sum(y_gt))

            kappa = cohen_kappa_score(y_gt, y_vlm) if (np.sum(y_gt) + np.sum(y_vlm) > 0) else 1.0
            p = precision_score(y_gt, y_vlm, zero_division=0)
            r = recall_score(y_gt, y_vlm, zero_division=0)
            f1 = f1_score(y_gt, y_vlm, zero_division=0)

            metrics_list.append({
                "Nhãn Bệnh lý": lbl,
                "Số ca VLM (Ảnh)": vlm_cnt,
                "Số ca Bác sĩ (Text)": gt_cnt,
                "Cohen's Kappa": round(kappa, 3),
                "Precision": round(p, 3),
                "Recall": round(r, 3),
                "F1-Score": round(f1, 3)
            })

        metrics_df = pd.DataFrame(metrics_list)
        print(metrics_df.to_string(index=False))
        print("=" * 85)

    finally:
        session.close()
        if unload_at_exit:
            unload_ollama_model(model=model)
        else:
            print("[GPU STATUS] Model tiếp tục được giữ trong VRAM 24h để sẵn sàng cho các lượt chạy tiếp theo!")

    return df_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giai đoạn 2 (Pipeline 2): VLM Label Audit Test (Ultra Fast)")
    parser.add_argument("--input_csv", type=str, default=VLM_TEST_READY_CSV)
    parser.add_argument("--output_csv", type=str, default=VLM_AUDITED_TEST_CSV)
    parser.add_argument("--cache_file", type=str, default=TEST_VLM_CACHE_FILE)
    parser.add_argument("--sample_size", type=int, default=None, help="Chạy thử nghiệm trên N ca mẫu")
    parser.add_argument("--save_interval", type=int, default=20, help="Số ca mỗi lần flush checkpoint ra đĩa")
    parser.add_argument("--max_images", type=int, default=1, help="Số ảnh tối đa mỗi ca (mặc định 1 ảnh thẳng siêu nhanh)")
    parser.add_argument("--clear_cache", action="store_true", help="Xóa cache cũ và chạy lại từ đầu")
    parser.add_argument("--unload_at_exit", action="store_true", help="Giải phóng model khỏi VRAM khi kết thúc")
    parser.add_argument("--model", type=str, default=VLM_MODEL)
    args = parser.parse_args()

    run_vlm_label_audit_test(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        cache_file=args.cache_file,
        sample_size=args.sample_size,
        save_interval=args.save_interval,
        max_images=args.max_images,
        clear_cache=args.clear_cache,
        unload_at_exit=args.unload_at_exit,
        model=args.model
    )
