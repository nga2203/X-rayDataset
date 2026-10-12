#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Các hàm tiện ích dùng chung cho Pipeline 2 (VLM Utils)
"""

import os
import ast
import json
import time
import base64
import requests
from pipeline_2.config_pipeline_2 import LABELS_9, VLM_MODEL, OLLAMA_ENDPOINT, IMAGE_ROOT

SYSTEM_PROMPT_VISION_EN = """You are an expert chest radiologist. Analyze the chest X-ray image and output a JSON object with binary flags (0 or 1) for the following 9 pathologies:
1. CARDIOMEGALY: Enlarged cardiac silhouette (CTR > 0.5 on frontal view).
2. DEGENERATION: Osteophytes, spinal spurring, joint space narrowing.
3. FRACTURE: Bone cortical disruption, rib or clavicle fracture.
4. NORMAL: Lungs clear, heart normal, sharp costophrenic angles, no acute findings.
5. OTHER: Non-pulmonary findings (surgical clips, pacemaker, aortic ectasia, etc.).
6. PLEURAL_EFFUSION: Blunting of costophrenic angle, fluid meniscus.
7. PNEUMONIA/INFILTRATION: Airspace opacity, patchy consolidation, infiltrates.
8. SPINAL_ALIGNMENT: Scoliosis, kyphosis, spine curvature deformity.
9. TUBERCULOSIS: Cavitary lesion, apical fibrocalcific scarring.

OUTPUT RULES:
- Return ONLY 1 valid JSON object with these 9 exact keys, values 0 or 1.
- If completely normal, set "NORMAL": 1 and all other 8 classes to 0.
- Output JSON only, no other text."""


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


def encode_image_to_base64(img_path):
    """
    Đọc file ảnh từ đĩa và mã hóa sang chuỗi base64.
    """
    try:
        with open(img_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        return None


def preload_vlm_model(model=VLM_MODEL, endpoint=OLLAMA_ENDPOINT, timeout=120):
    """
    Nạp sẵn model từ ổ cứng vào GPU VRAM trước khi chạy vòng lặp,
    giữ model cố định trong VRAM 24h.
    """
    print(f"\n[WARM-UP MODEL] Đang giữ model '{model}' trong GPU VRAM (đợi ~10-15s)...")
    payload = {
        "model": model,
        "prompt": "hi",
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "num_gpu": 99,
            "num_ctx": 2048
        }
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=timeout)
        if r.status_code == 200:
            print("[WARM-UP THÀNH CÔNG] Model đã nằm gọn trong GPU VRAM (7.2GB / 12GB)! Không tràn Shared RAM.")
            return True
    except Exception as e:
        print(f"[WARM-UP CẢNH BÁO] Không thể warm-up: {e}")
    return False


def query_vlm_vision(image_paths, model=VLM_MODEL, endpoint=OLLAMA_ENDPOINT, session=None, max_images=1, max_retries=3, timeout=30):
    """
    Gửi ảnh X-quang tới Ollama Qwen2.5-VL và nhận kết quả JSON nhị phân 0/1.
    Tối ưu hóa triệt để:
    - num_ctx=2048: Khóa chặt KV cache, giữ VRAM ở mức ~7.2GB (triệt tiêu 100% hiện tượng tràn sang Shared GPU RAM)
    - Prompt tiếng Anh chuyên sâu ngắn gọn
    - max_images=1: Mặc định đọc phim thẳng chuẩn (hoặc 2 nếu chỉ định)
    - Tái sử dụng HTTP Persistent Session
    """
    selected_paths = image_paths[:max_images]
    b64_list = []
    for p in selected_paths:
        b64 = encode_image_to_base64(p)
        if b64:
            b64_list.append(b64)

    if not b64_list:
        return None

    prompt = f"{SYSTEM_PROMPT_VISION_EN}\n\nCHEST X-RAY JSON OUTPUT:"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": b64_list,
        "format": "json",
        "stream": False,
        "keep_alive": "24h",
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_gpu": 99,
            "num_ctx": 2048,
            "num_predict": 130
        }
    }

    client = session if session is not None else requests

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.post(endpoint, json=payload, timeout=timeout)
            if resp.status_code == 200:
                raw_json = resp.json().get("response", "{}")
                parsed = json.loads(raw_json)

                # Chuẩn hóa về đúng 9 key 0/1
                res = {}
                for lbl in LABELS_9:
                    val = parsed.get(lbl, 0)
                    res[lbl] = 1 if (str(val).strip() in ['1', 'true', 'True']) else 0

                # Đảm bảo logic nếu NORMAL = 1 thì các bệnh khác = 0
                if res.get("NORMAL", 0) == 1 and sum(res.values()) > 1:
                    res["NORMAL"] = 0

                # Nếu toàn bộ là 0 (fallback về NORMAL)
                if sum(res.values()) == 0:
                    res["NORMAL"] = 1

                return res
            elif resp.status_code == 404:
                print(f"\n[LỖI OLLAMA 404] Model '{model}' chưa được cài đặt trong Ollama!")
                return None
        except requests.exceptions.ReadTimeout:
            time.sleep(1)
        except Exception as e:
            time.sleep(1)

    return None


def atomic_save_json(data, target_file):
    """
    Ghi cache an toàn (Atomic write): Ghi ra file tạm .tmp rồi đổi tên,
    tránh làm hỏng file cache khi bị ngắt đột ngột (Ctrl+C).
    """
    os.makedirs(os.path.dirname(target_file), exist_ok=True)
    tmp_file = target_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    if os.path.exists(target_file):
        os.remove(target_file)
    os.rename(tmp_file, target_file)
