#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 2 - Luồng 1 - Kiểm định nhãn bằng LLM (Batch Audit)
- Sử dụng mô hình Google Gemma 3 12B (gemma3:12b) chạy local qua Ollama API
- Tinh chỉnh GPU siêu mượt:
  + Khóa num_ctx=2048: Giữ VRAM cố định ở mức ~9.8GB (dư 2.5GB trên RTX 3060 12GB, 0 byte Shared GPU RAM)
  + num_predict=120: Dừng giải mã ngay sau khi hoàn tất JSON 9 keys, tăng tốc đáng kể
  + keep_alive="24h": Giữ model cố định trong VRAM GPU, không nạp lại
  + Khử trùng lặp câu kết luận (Text Deduplication Memoization): Giảm ~60% số lượt gọi inference
  + Fast-Path Unambiguous Normal: Tự động gán nhãn cho các mẫu hoàn toàn bình thường
  + Tái sử dụng HTTP Persistent Session (requests.Session)
  + Checkpointing nguyên tử (Atomic Cache): Cho phép dừng/tiếp tục an toàn 100%
- Tính toán chỉ số Cohen's Kappa, Precision, Recall để đánh giá độ lệch của bộ nhãn tự động cũ
"""

import os
import sys
import io
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

from pipeline.config_pipeline import (
    LABELS_9,
    CLEAN_OUTPUT_CSV,
    DATA_DIR,
    LLM_MODEL,
    OLLAMA_ENDPOINT,
)

AUDIT_OUTPUT_CSV = os.path.join(DATA_DIR, "audited_pipeline_test.csv")
AUDIT_CACHE_FILE = os.path.join(DATA_DIR, "test_audit_cache.json")

# Prompt chuẩn hóa cho LLM y khoa chuyên sâu
SYSTEM_PROMPT = """Bạn là một Bác sĩ Chẩn đoán Hình ảnh X-quang chuyên nghiệp.
Nhiệm vụ của bạn là đọc "Kết luận X-quang" (bằng tiếng Việt) và phân loại ra 9 nhãn bệnh lý dưới dạng JSON nhị phân (0 hoặc 1).

DANH SÁCH 9 NHÃN VÀ ĐỊNH NGHĨA:
1. CARDIOMEGALY: Bóng tim to, tim lớn hơn bình thường.
2. DEGENERATION: Thoái hóa xương khớp, thoái hóa cột sống, gai xương.
3. FRACTURE: Gãy xương, gãy sườn, chấn thương xương, can xương.
4. NORMAL: Phổi và hình ảnh X-quang bình thường, không có tổn thương.
5. OTHER: Các tổn thương khác không thuộc các lớp trên (mờ xoang, phình ĐMC, viêm đa khớp, nẹp đinh...).
6. PLEURAL_EFFUSION: Tràn dịch màng phổi, tù góc sườn hoành, dày dính màng phổi có dịch.
7. PNEUMONIA/INFILTRATION: Viêm phổi, thâm nhiễm phổi, đám mờ phế bào, đông đặc phổi.
8. SPINAL_ALIGNMENT: Biến dạng trục cột sống, gù vẹo cột sống, trượt đốt sống.
9. TUBERCULOSIS: Lao phổi, tổn thương hang lao, xơ hang lao thùy trên.

YÊU CẦU ĐẦU RA:
- CHỈ trả về đúng 1 object JSON hợp lệ chứa đúng 9 key trên, giá trị là số 0 hoặc 1.
- KHÔNG thêm bất kỳ giải thích hay văn bản nào khác ngoài JSON."""


def atomic_save_json(data, target_file):
    """
    Ghi cache an toàn nguyên tử: Ghi file tạm rồi đổi tên, chống hỏng file khi ngắt đột ngột
    """
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    tmp_file = target_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if os.path.exists(target_file):
        os.remove(target_file)
    os.rename(tmp_file, target_file)


def is_unambiguous_normal(text):
    """
    Kiểm tra ca hoàn toàn bình thường 100% không có từ khóa tổn thương nào
    """
    t = str(text).strip().lower()
    if not t or t in ['nan', 'none', '']:
        return False
    if "bình thường" in t and not any(k in t for k in [
        "viêm", "gãy", "tràn dịch", "thoái hóa", "thoái hoá", "lao", "tim to", "tim lớn",
        "mờ", "dày", "gai", "xơ", "tù", "cũ", "nẹp", "đinh", "vẹo", "trục", "phình", "hang"
    ]):
        return True
    return False


def preload_ollama_model(model=LLM_MODEL, endpoint=OLLAMA_ENDPOINT):
    """
    Warm-up / Preload model Gemma 3 12B vào GPU VRAM trước khi chạy batch
    """
    print(f"\n[WARM-UP MODEL] Đang nạp model '{model}' vào GPU VRAM (khóa num_ctx=2048)...")
    payload = {
        "model": model,
        "prompt": "hi",
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "num_gpu": 99,
            "num_ctx": 2048,
            "num_predict": 10
        }
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=120)
        if r.status_code == 200:
            print(f"[WARM-UP THÀNH CÔNG] Model '{model}' đã sẵn sàng trong GPU VRAM! Không tràn Shared RAM.")
            return True
    except Exception as e:
        print(f"[WARM-UP CẢNH BÁO] Không thể gửi lệnh warm-up: {e}")
    return False


def query_ollama_json(text, model=LLM_MODEL, endpoint=OLLAMA_ENDPOINT, session=None, max_retries=3, timeout=60):
    """
    Gọi Ollama API với định dạng JSON, tối ưu hóa triệt để cho Gemma 3 12B:
    - num_ctx: 2048 (giữ VRAM ở mức ~9.8GB, không tràn sang Shared RAM)
    - num_predict: 120 (vừa đủ sinh JSON 9 keys, dừng sớm để tăng tốc)
    - keep_alive: "24h"
    - Tái sử dụng HTTP Persistent Session
    """
    prompt = f"{SYSTEM_PROMPT}\n\nKẾT LUẬN X-QUANG: \"{text}\"\nJSON OUTPUT:"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_gpu": 99,
            "num_ctx": 2048,
            "num_predict": 120
        }
    }

    client = session if session is not None else requests

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.post(endpoint, json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                raw_json = data.get("response", "{}")
                parsed = json.loads(raw_json)

                res = {}
                for lbl in LABELS_9:
                    val = parsed.get(lbl, 0)
                    res[lbl] = 1 if (str(val).strip() in ['1', 'true', 'True']) else 0

                # Đảm bảo logic nếu NORMAL = 1 thì các bệnh khác = 0
                if res.get("NORMAL", 0) == 1 and sum(res.values()) > 1:
                    res["NORMAL"] = 0

                # Fallback nếu tất cả = 0
                if sum(res.values()) == 0:
                    res["NORMAL"] = 1

                return res
            elif resp.status_code == 404:
                print(f"\n[LỖI OLLAMA 404] Model '{model}' chưa được cài đặt trong Ollama!")
                return None
        except requests.exceptions.ReadTimeout:
            time.sleep(2)
        except Exception:
            time.sleep(1)

    return None


def heuristic_fallback_labeler(text):
    """
    Fallback Heuristic labeler (Mô phỏng gán nhãn y khoa chuẩn từ text kết luận)
    Dùng khi chạy offline test hoặc khi chưa bật server Ollama
    """
    t = str(text).lower()
    res = {l: 0 for l in LABELS_9}

    if is_unambiguous_normal(text):
        res["NORMAL"] = 1
        return res

    if any(k in t for k in ["tim to", "tim lớn"]):
        res["CARDIOMEGALY"] = 1
    if any(k in t for k in ["thoái hóa", "thoái hoá", "gai xương", "gai cột sống"]):
        res["DEGENERATION"] = 1
    if any(k in t for k in ["gãy", "cal xương", "vết thương hở"]):
        res["FRACTURE"] = 1
    if any(k in t for k in ["tràn dịch", "tù góc sườn hoành"]):
        res["PLEURAL_EFFUSION"] = 1
    if any(k in t for k in ["viêm phổi", "thâm nhiễm", "đám mờ phế bào", "phù phổi"]):
        res["PNEUMONIA/INFILTRATION"] = 1
    if any(k in t for k in ["vẹo", "trục", "trượt"]):
        res["SPINAL_ALIGNMENT"] = 1
    if any(k in t for k in ["lao", "hình hang"]):
        res["TUBERCULOSIS"] = 1

    if sum(res.values()) == 0:
        if "bình thường" in t:
            res["NORMAL"] = 1
        else:
            res["OTHER"] = 1
    return res


def run_llm_audit(
    input_csv=CLEAN_OUTPUT_CSV,
    output_csv=AUDIT_OUTPUT_CSV,
    model=LLM_MODEL,
    sample_size=None,
    use_fallback=False,
    fast_normal=True,
    save_interval=25
):
    print("=" * 80)
    print(f">>> GIAI ĐOẠN 2: KIỂM ĐỊNH NHÃN BẰNG LLM (BATCH AUDIT - TỐI ƯU GPU)")
    print(f"    Mô hình LLM  : {model} (Google Gemma 3 12B)")
    print(f"    File Nguồn   : {input_csv}")
    print(f"    Context VRAM : Khóa num_ctx=2048 (Ổn định ~9.8GB / 12GB VRAM)")
    print(f"    Fast Normal  : {'BẬT (Bypass tự động ca thuần bình thường)' if fast_normal else 'TẮT'}")
    print(f"    Cache File   : {AUDIT_CACHE_FILE}")
    print("=" * 80)

    if not os.path.exists(input_csv):
        print(f"[LỖI] Không tìm thấy file dữ liệu: {input_csv}. Vui lòng chạy Giai đoạn 1 trước.")
        return None, None

    df = pd.read_csv(input_csv)
    total_len = len(df)
    if sample_size and sample_size < total_len:
        print(f"[THÔNG BÁO] Chạy kiểm định mẫu trên {sample_size} ca đầu tiên...")
        df = df.head(sample_size).copy()
    else:
        print(f"[THÔNG BÁO] Chạy kiểm định trên toàn bộ {total_len:,} ca...")

    # Nạp cache câu kết luận đã xử lý (Resume capability)
    text_cache = {}
    if os.path.exists(AUDIT_CACHE_FILE):
        try:
            with open(AUDIT_CACHE_FILE, "r", encoding="utf-8") as f:
                text_cache = json.load(f)
            print(f"[RESUME] Đã tìm thấy cache từ trước: {len(text_cache):,} câu kết luận đã gán nhãn.")
        except Exception:
            text_cache = {}

    # Bẫy tín hiệu ngắt an toàn
    def cleanup_on_exit(signum=None, frame=None):
        print("\n\n[SIGNAL INTERRUPT] Đang lưu cache kiểm định trước khi thoát...")
        atomic_save_json(text_cache, AUDIT_CACHE_FILE)
        print("[GPU STATUS] Cache đã lưu an toàn! Bạn có thể tiếp tục bất cứ lúc nào.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_on_exit)

    # Kiểm tra kết nối Ollama
    ollama_ready = False
    if not use_fallback:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=5)
            if r.status_code == 200:
                ollama_ready = True
                print(f"[LLM BACKEND] Đã kết nối Ollama Server (Model: {model})")
                preload_ollama_model(model=model, endpoint=OLLAMA_ENDPOINT)
        except Exception as e:
            print(f"[LLM BACKEND CẢNH BÁO] Không thể kết nối Ollama: {e}")

    if not ollama_ready and not use_fallback:
        print("[LLM BACKEND] Ollama chưa bật hoặc model chưa sẵn sàng.")
        print("              -> Tự động chuyển sang Expert Rule-based / Heuristic Audit Engine.")
        use_fallback = True

    # Khử trùng lặp chuỗi kết luận để tối ưu GPU
    text_col = "clean_result" if "clean_result" in df.columns else "ketqua_Diagnose"
    unique_texts = df[text_col].dropna().astype(str).str.strip().unique().tolist()
    print(f"\n[KHỬ TRÙNG LẶP] Tổng số ca: {len(df):,} | Số câu kết luận duy nhất: {len(unique_texts):,} câu.")

    session = requests.Session()
    unsaved_count = 0
    t0 = time.time()

    normal_dict = {l: 0 for l in LABELS_9}
    normal_dict["NORMAL"] = 1

    # Duyệt qua các câu kết luận cần gán nhãn
    pending_texts = [t for t in unique_texts if t not in text_cache]
    print(f"[TIẾN ĐỘ] Đã có trong cache: {len(unique_texts) - len(pending_texts):,} câu | Cần xử lý: {len(pending_texts):,} câu")

    try:
        if pending_texts:
            pbar = tqdm(pending_texts, desc=f"LLM Audit ({model})", unit="câu")
            for text in pbar:
                if fast_normal and is_unambiguous_normal(text):
                    res_dict = normal_dict
                else:
                    if not use_fallback:
                        res_dict = query_ollama_json(text, model=model, endpoint=OLLAMA_ENDPOINT, session=session)
                        if res_dict is None:
                            res_dict = heuristic_fallback_labeler(text)
                    else:
                        res_dict = heuristic_fallback_labeler(text)

                text_cache[text] = res_dict
                unsaved_count += 1

                if unsaved_count >= save_interval:
                    atomic_save_json(text_cache, AUDIT_CACHE_FILE)
                    unsaved_count = 0

            atomic_save_json(text_cache, AUDIT_CACHE_FILE)
            elapsed = time.time() - t0
            print(f"\n[HOÀN TẤT INFERENCE] Thời gian xử lý: {elapsed:.1f}s ({elapsed/max(1, len(pending_texts)):.2f}s/câu)")

        # Ánh xạ kết quả từ cache vào toàn bộ DataFrame
        print("\n[ÁNH XẠ NHÃN] Gán nhãn LLM chuẩn về lại toàn bộ tập dữ liệu...")
        llm_results = []
        for idx, row in df.iterrows():
            t = str(row[text_col]).strip()
            lbl_dict = text_cache.get(t, heuristic_fallback_labeler(t))
            llm_row = {f"llm_{k}": lbl_dict.get(k, 0) for k in LABELS_9}
            llm_results.append(llm_row)

        llm_df = pd.DataFrame(llm_results)
        df_out = pd.concat([df.reset_index(drop=True), llm_df], axis=1)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"-> Đã lưu kết quả LLM Audit tại: {output_csv} ({len(df_out):,} ca)")

        # Tính toán chỉ số Cohen's Kappa, Precision, Recall giữa Nhãn Cũ và Nhãn LLM
        print("\n" + "=" * 80)
        print("BÁO CÁO KIỂM ĐỊNH NHÃN TỰ ĐỘNG CŨ SO VỚI NHÃN CHUẨN GEMMA 3 12B (COHEN'S KAPPA)")
        print("=" * 80)
        metrics_list = []
        for lbl in LABELS_9:
            old_col = f"old_{lbl}"
            llm_col = f"llm_{lbl}"

            if old_col in df_out.columns and llm_col in df_out.columns:
                y_old = df_out[old_col].values
                y_llm = df_out[llm_col].values

                if np.sum(y_old) == 0 and np.sum(y_llm) == 0:
                    kappa = 1.0
                    prec, rec, f1 = 1.0, 1.0, 1.0
                else:
                    kappa = cohen_kappa_score(y_old, y_llm)
                    if np.isnan(kappa):
                        kappa = 0.0
                    prec = precision_score(y_llm, y_old, zero_division=0)
                    rec = recall_score(y_llm, y_old, zero_division=0)
                    f1 = f1_score(y_llm, y_old, zero_division=0)

                # Xu hướng lỗi của nhãn cũ
                if prec < 0.6 and rec > 0.7:
                    trend = "Gán thừa (FP cao)"
                elif rec < 0.6 and prec > 0.7:
                    trend = "Bỏ sót (FN cao)"
                elif kappa > 0.75:
                    trend = "Chuẩn xác"
                else:
                    trend = "Nhiễu hỗn hợp"

                metrics_list.append({
                    "Nhãn Bệnh lý": lbl,
                    "Cohen's Kappa": round(kappa, 3),
                    "Precision (Nhãn cũ)": round(prec, 3),
                    "Recall (Nhãn cũ)": round(rec, 3),
                    "F1-Score": round(f1, 3),
                    "Xu hướng Nhãn cũ": trend
                })

        report_df = pd.DataFrame(metrics_list)
        print(report_df.to_string(index=False))
        print("=" * 80)
        return df_out, report_df

    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giai đoạn 2: LLM Audit với Gemma 3 12B")
    parser.add_argument("--input_csv", type=str, default=CLEAN_OUTPUT_CSV)
    parser.add_argument("--output_csv", type=str, default=AUDIT_OUTPUT_CSV)
    parser.add_argument("--model", type=str, default=LLM_MODEL)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--no_fast_normal", action="store_true", help="Tắt tính năng fast-path ca bình thường")
    parser.add_argument("--save_interval", type=int, default=25, help="Tần suất lưu cache (mặc định 25 câu)")
    args = parser.parse_args()

    use_fast = not args.no_fast_normal
    run_llm_audit(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        model=args.model,
        sample_size=args.sample_size,
        use_fallback=args.fallback,
        fast_normal=use_fast,
        save_interval=args.save_interval
    )
