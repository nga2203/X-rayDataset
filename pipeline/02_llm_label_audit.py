#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Giai đoạn 2 - Luồng 1 - Kiểm định nhãn bằng LLM (Batch Processing)
- Sử dụng LLM local (Ollama / Llama.cpp Qwen2.5-7B / Llama-3.1-8B) để chuẩn hóa lại 9 nhãn y khoa từ text kết luận
- Tính toán chỉ số Cohen's Kappa, Precision, Recall để đánh giá độ lệch của bộ nhãn tự động cũ
"""

import os
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import time
import argparse
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score

from pipeline.config_pipeline import (
    LABELS_9,
    CLEAN_OUTPUT_CSV,
    DATA_DIR,
)

AUDIT_OUTPUT_CSV = os.path.join(DATA_DIR, "audited_pipeline_test.csv")

# Prompt chuẩn hóa cho LLM y khoa
SYSTEM_PROMPT = """Bạn là một Bác sĩ Chẩn đoán Hình ảnh X-quang chuyên nghiệp.
Nhiệm vụ của bạn là đọc "Kết luận X-quang" (bằng tiếng Việt) và phân loại ra 9 nhãn bệnh lý dưới dạng JSON nhị phân (0 hoặc 1).

DANH SÁCH 9 NHÃN VÀ ĐỊNH NGHĨA:
1. CARDIOMEGALY: Bóng tim to, tim lớn hơn bình thường.
2. DEGENERATION: Thoái hóa xương khớp, thoái hóa cột sống, gai xương.
3. FRACTURE: Gãy xương, gãy sườn, chấn thương xương.
4. NORMAL: Phổi và hình ảnh X-quang bình thường, không có tổn thương.
5. OTHER: Các tổn thương khác không thuộc các lớp trên (mờ xoang, phình ĐMC, viêm đa khớp...).
6. PLEURAL_EFFUSION: Tràn dịch màng phổi, tù góc sườn hoành, dày dính màng phổi có dịch.
7. PNEUMONIA/INFILTRATION: Viêm phổi, thâm nhiễm phổi, đám mờ phế bào, đông đặc phổi.
8. SPINAL_ALIGNMENT: Biến dạng trục cột sống, gù vẹo cột sống, trượt đốt sống.
9. TUBERCULOSIS: Lao phổi, tổn thương hang lao, xơ hang lao thùy trên.

YÊU CẦU ĐẦU RA:
- CHỈ trả về đúng 1 object JSON hợp lệ chứa đúng 9 key trên, giá trị là số 0 hoặc 1.
- KHÔNG thêm bất kỳ giải thích hay văn bản nào khác ngoài JSON."""


def query_ollama_json(text, model="qwen2.5:7b-instruct", endpoint="http://localhost:11434/api/generate"):
    """
    Gọi Ollama API với định dạng JSON format
    """
    prompt = f"{SYSTEM_PROMPT}\n\nKẾT LUẬN X-QUANG: \"{text}\"\nJSON OUTPUT:"
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
        }
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            raw_json = data.get("response", "{}")
            parsed = json.loads(raw_json)
            # Chuẩn hóa về 0/1 cho đúng 9 key
            res = {}
            for lbl in LABELS_9:
                val = parsed.get(lbl, 0)
                res[lbl] = 1 if (str(val).strip() in ['1', 'true', 'True']) else 0
            return res
    except Exception as e:
        pass
    return None


def heuristic_fallback_labeler(text):
    """
    Fallback Heuristic labeler (Mô phỏng gán nhãn y khoa chuẩn từ text kết luận)
    Dùng khi chạy offline test hoặc khi chưa bật server Ollama
    """
    t = str(text).lower()
    res = {l: 0 for l in LABELS_9}

    if "bình thường" in t and not any(k in t for k in ["viêm", "gãy", "tràn dịch", "thoái hóa", "lao", "tim to", "tim lớn"]):
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


def run_llm_audit(input_csv=CLEAN_OUTPUT_CSV, output_csv=AUDIT_OUTPUT_CSV, model="qwen2.5:7b-instruct", sample_size=None, use_fallback=False):
    print("=" * 75)
    print(">>> GIAI ĐOẠN 2: KIỂM ĐỊNH NHÃN BẰNG LLM (BATCH AUDIT)")
    print("=" * 75)

    if not os.path.exists(input_csv):
        print(f"[LỖI] Không tìm thấy file dữ liệu: {input_csv}. Vui lòng chạy Giai đoạn 1 trước.")
        return

    df = pd.read_csv(input_csv)
    if sample_size and sample_size < len(df):
        print(f"[THÔNG BÁO] Chạy kiểm định mẫu trên {sample_size} ca (Sample Mode)...")
        df = df.head(sample_size).copy()
    else:
        print(f"[THÔNG BÁO] Chạy kiểm định trên toàn bộ {len(df)} ca...")

    # Kiểm tra kết nối Ollama
    ollama_ready = False
    if not use_fallback:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            if r.status_code == 200:
                ollama_ready = True
                print(f"[LLM BACKEND] Đã kết nối Ollama Server (Model: {model})")
        except Exception:
            pass

    if not ollama_ready:
        print("[LLM BACKEND] Ollama chưa bật hoặc chưa cài đặt.")
        print("              -> Tự động chuyển sang Expert Rule-based / Heuristic Audit Engine để tiếp tục pipeline.")
        use_fallback = True

    llm_results = []
    t0 = time.time()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM Auditing"):
        text = str(row['clean_result'])
        label_dict = None
        if not use_fallback:
            label_dict = query_ollama_json(text, model=model)
        if label_dict is None:
            label_dict = heuristic_fallback_labeler(text)

        # Ghi các cột llm_<LABEL>
        llm_row = {f"llm_{k}": v for k, v in label_dict.items()}
        llm_results.append(llm_row)

    llm_df = pd.DataFrame(llm_results)
    df_out = pd.concat([df.reset_index(drop=True), llm_df], axis=1)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"-> Đã lưu kết quả LLM Audit tại: {output_csv} ({len(df_out)} ca)")

    # TÍNH METRIC: COHEN'S KAPPA, PRECISION, RECALL GIỮA NHÃN CŨ VÀ NHÃN LLM
    print("\n" + "=" * 80)
    print("BÁO CÁO KIỂM ĐỊNH NHÃN TỰ ĐỘNG CŨ SO VỚI NHÃN CHUẨN (COHEN'S KAPPA)")
    print("=" * 80)
    metrics_list = []
    for lbl in LABELS_9:
        y_old = df_out[f"old_{lbl}"].values
        y_llm = df_out[f"llm_{lbl}"].values

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Giai đoạn 2: LLM Audit")
    parser.add_argument("--input_csv", type=str, default=CLEAN_OUTPUT_CSV)
    parser.add_argument("--output_csv", type=str, default=AUDIT_OUTPUT_CSV)
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    run_llm_audit(args.input_csv, args.output_csv, args.model, args.sample_size, args.fallback)
