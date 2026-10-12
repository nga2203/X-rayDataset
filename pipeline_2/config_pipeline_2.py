#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cấu hình tập trung cho Pipeline 2: VLM-based X-ray Re-labeling & Diagnostics
Sử dụng Vision-Language Model Qwen2.5-VL 7B chạy local qua Ollama API.
Tách biệt hoàn toàn với Pipeline 1.
"""

import os
import requests

# 9 Nhãn bệnh lý chuẩn trong dataset và mô hình VN-MultiXR
LABELS_9 = [
    "CARDIOMEGALY",
    "DEGENERATION",
    "FRACTURE",
    "NORMAL",
    "OTHER",
    "PLEURAL_EFFUSION",
    "PNEUMONIA/INFILTRATION",
    "SPINAL_ALIGNMENT",
    "TUBERCULOSIS"
]

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS_9)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS_9)}

# Thư mục gốc dự án
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Các thư mục dành riêng cho Pipeline 2 (tách biệt hoàn toàn)
DATA_DIR_1 = os.path.join(PROJECT_ROOT, "Data")
RELABEL_DIR_1 = os.path.join(PROJECT_ROOT, "RelableDataset")

DATA_DIR_2 = os.path.join(PROJECT_ROOT, "Data_2")
RELABEL_DIR_2 = os.path.join(PROJECT_ROOT, "RelableDataset_2")
LOG_DIR_2 = os.path.join(PROJECT_ROOT, "log_2")

# Đường dẫn file dữ liệu gốc
TEST_CSV = os.path.join(DATA_DIR_1, "dataset_test.csv")
TRAIN_CSV = os.path.join(DATA_DIR_1, "dataset_train.csv")
VAL_CSV = os.path.join(DATA_DIR_1, "dataset_val.csv")

# File dữ liệu Plan 1 để đối chiếu chéo (Cross-validation)
PLAN1_AUDITED_TEST_CSV = os.path.join(DATA_DIR_1, "audited_pipeline_test.csv")

# File dữ liệu đầu ra cho Pipeline 2
VLM_TEST_READY_CSV = os.path.join(DATA_DIR_2, "clean_test_vlm_ready.csv")
VLM_AUDITED_TEST_CSV = os.path.join(DATA_DIR_2, "audited_vlm_test.csv")
VLM_TRIANGULATION_METRICS_CSV = os.path.join(DATA_DIR_2, "vlm_triangulation_metrics.csv")
VLM_CRAMERS_V_CSV = os.path.join(DATA_DIR_2, "cramers_v_matrix_vlm.csv")
VLM_AGE_SUBGROUP_CSV = os.path.join(DATA_DIR_2, "age_subgroup_f1_recall_vlm.csv")
VLM_DIAGNOSTICS_REPORT_MD = os.path.join(DATA_DIR_2, "vlm_model_diagnostics_report.md")

# Checkpoint cache files
TEST_VLM_CACHE_FILE = os.path.join(RELABEL_DIR_2, "test_vlm_cache.json")
TRAIN_VLM_CACHE_FILE = os.path.join(RELABEL_DIR_2, "train_vlm_cache.json")
VAL_VLM_CACHE_FILE = os.path.join(RELABEL_DIR_2, "val_vlm_cache.json")

# Cấu hình Ollama VLM
VLM_MODEL = "qwen2.5vl:7b"
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
IMAGE_ROOT = "C:/"


def unload_ollama_model(model=VLM_MODEL, endpoint=OLLAMA_ENDPOINT):
    """
    Giải phóng hoàn toàn model khỏi GPU VRAM ngay lập tức (Chống chiếm dụng VRAM).
    Gửi keep_alive: 0 tới Ollama để đưa GPU về trạng thái idle.
    """
    try:
        payload = {"model": model, "keep_alive": 0}
        resp = requests.post(endpoint, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[VRAM CLEANUP] Đã giải phóng hoàn toàn model '{model}' khỏi VRAM GPU thành công!")
            return True
    except Exception as e:
        print(f"[VRAM CLEANUP CẢNH BÁO] Không thể gửi lệnh unload tới Ollama: {e}")
    return False
