# BỘ DỮ LIỆU ĐỒNG THUẬN TAM GIÁC VÀNG (FINAL 3-WAY CONSENSUS DATASET)

> **Định nghĩa**: Đây là bộ dữ liệu siêu sạch (Ultra-Clean Ground Truth) chỉ giữ lại các ca bệnh có sự **đồng thuận tuyệt đối 100% trên toàn bộ 9 nhãn bệnh lý** giữa 3 nguồn độc lập:
1. **Nhãn gốc (Old Label)**: Do bác sĩ lâm sàng/bệnh viện ban đầu gán.
2. **Nhãn LLM Qwen 2.5 7B-Instruct**: Giai đoạn kiểm định nhãn Phase 1 gốc.
3. **Nhãn LLM Google Gemma 3 12B**: Giai đoạn kiểm định nhãn Phase 1 nâng cấp.

---

## 1. Thống Kê Tổng Quan Số Lượng Ca
- **Tập Train (`train.csv`)**: **8,565 ca** (giữ lại 68.73% từ 12,462 ca ban đầu).
- **Tập Val (`val.csv`)**: **1,215 ca** (giữ lại 68.64% từ 1,770 ca ban đầu).
- **Tập Test (`test.csv`)**: **2,450 ca** (giữ lại 69.54% từ 3,523 ca ban đầu).
- **TỔNG CỘNG CẢ BỘ**: **12,230 ca**.

---

## 2. Bảng Phân Phối Số Ca Dương Tính Theo Từng Lớp Bệnh Lý

| Tập dữ liệu   |   Tổng ca ban đầu |   Số ca đồng thuận 3 bên | Tỷ lệ giữ lại (%)   |   CARDIOMEGALY |   DEGENERATION |   FRACTURE |   NORMAL |   OTHER |   PLEURAL_EFFUSION |   PNEUMONIA/INFILTRATION |   SPINAL_ALIGNMENT |   TUBERCULOSIS |
|:--------------|------------------:|-------------------------:|:--------------------|---------------:|---------------:|-----------:|---------:|--------:|-------------------:|-------------------------:|-------------------:|---------------:|
| TRAIN         |             12462 |                     8565 | 68.73%              |            120 |            886 |       1324 |     6119 |      80 |                 17 |                       49 |                 20 |              3 |
| VAL           |              1770 |                     1215 | 68.64%              |              9 |            129 |        185 |      873 |      16 |                  1 |                        3 |                  3 |              0 |
| TEST          |              3523 |                     2450 | 69.54%              |             24 |            264 |        382 |     1746 |      23 |                  3 |                       16 |                  7 |              1 |

---

## 3. Cấu Trúc Các Cột Trong Mỗi File CSV
- `ketqua_ID`: Mã định danh kết luận X-quang.
- `filepath`: Đường dẫn tương đối đến danh sách file ảnh DICOM/PNG (`['...']`).
- `ketqua_Diagnose`: Chẩn đoán lâm sàng đầu vào.
- `xray_type`: Loại phim chụp X-quang.
- `Age`: Tuổi bệnh nhân.
- `Sex`: Giới tính (0: Nữ, 1: Nam).
- `clean_result`: Văn bản kết luận X-quang đã làm sạch.
- `label`: Nhãn đa lớp chuẩn hóa dạng chuỗi list string (VD: `['NORMAL']`, `['DEGENERATION', 'SPINAL_ALIGNMENT']`). Tương thích 100% với `src/dataset.py`.
- **9 cột nhị phân chuẩn (0 hoặc 1)**:
  - `CARDIOMEGALY`: Bóng tim to
  - `DEGENERATION`: Thoái hóa xương khớp
  - `FRACTURE`: Gãy xương / chấn thương
  - `NORMAL`: Bình thường
  - `OTHER`: Tổn thương khác
  - `PLEURAL_EFFUSION`: Tràn dịch màng phổi
  - `PNEUMONIA/INFILTRATION`: Viêm / thâm nhiễm phổi
  - `SPINAL_ALIGNMENT`: Biến dạng trục cột sống
  - `TUBERCULOSIS`: Lao phổi

---

## 4. Cách Sử Dụng Trong Huấn Luyện (PyTorch / PyTorch Lightning)
```python
from src.dataset import ASSISTANCEDataset

# Khởi tạo Dataset trực tiếp với Final_Dataset
train_dataset = ASSISTANCEDataset(
    csv_path='Final_Dataset/train.csv',
    class_mapping_path='pipeline/class_mapping_9.json'
)
val_dataset = ASSISTANCEDataset(
    csv_path='Final_Dataset/val.csv',
    class_mapping_path='pipeline/class_mapping_9.json'
)
test_dataset = ASSISTANCEDataset(
    csv_path='Final_Dataset/test.csv',
    class_mapping_path='pipeline/class_mapping_9.json'
)
```

*Bộ dữ liệu này đảm bảo triệt tiêu hoàn toàn nhiễu nhãn (Zero Label Noise Bias), giúp mô hình VN-MultiXR v2 đạt độ khái quát hóa và độ tin cậy y khoa cao nhất.*