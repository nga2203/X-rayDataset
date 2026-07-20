import os
import json

# 9 Nhãn tự động chuẩn trong dataset và model VN-MultiXR
LABELS_9 = [
    "CARDIOMEGALY",
    "DEGENERATION",
    "FRACTURE",
    # "NORMAL",
    "OTHER",
    "PLEURAL_EFFUSION",
    "PNEUMONIA/INFILTRATION",
    "SPINAL_ALIGNMENT",
    "TUBERCULOSIS"
]

LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS_9)}
ID2LABEL = {i: lbl for i, lbl in enumerate(LABELS_9)}

# Ngưỡng tối ưu lấy từ configs/best_thresholds.json cho 9 nhãn
DEFAULT_THRESHOLDS = {
    "CARDIOMEGALY": 0.85,
    "DEGENERATION": 0.65,
    "FRACTURE": 0.70,
    "NORMAL": 0.30,
    "OTHER": 0.35,
    "PLEURAL_EFFUSION": 0.55,
    "PNEUMONIA/INFILTRATION": 0.55,
    "SPINAL_ALIGNMENT": 0.50,
    "TUBERCULOSIS": 0.80
}

# Đường dẫn mặc định
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

TEST_CSV = os.path.join(DATA_DIR, "dataset_test.csv")
CLEAN_OUTPUT_CSV = os.path.join(DATA_DIR, "clean_pipeline_test.csv")

CHECKPOINT_PATH = os.path.join(CHECKPOINTS_DIR, "assistance_best_5_20260514_223619.pth")
IMAGE_ROOT = "C:/"
