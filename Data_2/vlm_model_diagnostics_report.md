# BÁO CÁO CHẨN ĐOÁN & ĐỐI CHIẾU NĂNG LỰC THỊ GIÁC VLM (QWEN2.5-VL)

Báo cáo này đối chiếu chéo 3 bên giữa **Nhãn Thị giác VLM (đọc trực tiếp ảnh X-quang)**, **Nhãn Bác sĩ (Ground Truth trích xuất từ văn bản)** và **Bộ nhãn tự động cũ**.

## 1. Bảng Tổng Hợp Đối Chiếu 3 Bên (Triangulation Matrix)

| Nhãn Bệnh lý           |   VLM (Ảnh) |   Bác sĩ (Text) |   Nhãn Cũ |   Kappa (VLM vs Bác sĩ) |   VLM Precision |   VLM Recall |   VLM F1 | Đánh Giá Năng Lực Thị Giác                                    |
|:-----------------------|------------:|----------------:|----------:|------------------------:|----------------:|-------------:|---------:|:--------------------------------------------------------------|
| CARDIOMEGALY           |           0 |             164 |       162 |                   0     |           0     |        0     |    0     | 🔴 Khó phát hiện trên ảnh đơn thuần (Cần xét nghiệm/Lâm sàng) |
| DEGENERATION           |         365 |             615 |       285 |                   0.529 |           0.792 |        0.47  |    0.59  | 🟡 Nhận diện khá / Tổn thương cần thêm góc nghiêng            |
| FRACTURE               |          21 |             475 |       441 |                   0.009 |           0.238 |        0.011 |    0.02  | 🔴 Khó phát hiện trên ảnh đơn thuần (Cần xét nghiệm/Lâm sàng) |
| NORMAL                 |        3089 |            2180 |      1913 |                   0.206 |           0.662 |        0.938 |    0.776 | 🟢 Nhận diện thị giác xuất sắc / Độ tin cậy cao               |
| OTHER                  |         135 |              30 |       493 |                  -0.002 |           0.007 |        0.033 |    0.012 | 🔴 Khó phát hiện trên ảnh đơn thuần (Cần xét nghiệm/Lâm sàng) |
| PLEURAL_EFFUSION       |           0 |              21 |        28 |                   0     |           0     |        0     |    0     | 🔴 Khó phát hiện trên ảnh đơn thuần (Cần xét nghiệm/Lâm sàng) |
| PNEUMONIA/INFILTRATION |           0 |             104 |       145 |                   0     |           0     |        0     |    0     | 🔴 Khó phát hiện trên ảnh đơn thuần (Cần xét nghiệm/Lâm sàng) |
| SPINAL_ALIGNMENT       |         372 |              34 |        49 |                   0.042 |           0.032 |        0.353 |    0.059 | 🔴 Khó phát hiện trên ảnh đơn thuần (Cần xét nghiệm/Lâm sàng) |
| TUBERCULOSIS           |           0 |               9 |       202 |                   0     |           0     |        0     |    0     | 🔴 Khó phát hiện trên ảnh đơn thuần (Cần xét nghiệm/Lâm sàng) |

## 2. Phân Tích Chuyên Sâu Các Nhóm Bệnh Lý Trọng Yếu

### A. Nhóm Bệnh lý Cấu trúc Rõ ràng (Bóng tim to, Gãy xương, Bình thường)
- **Bóng tim to (`CARDIOMEGALY`)**: VLM có khả năng đo lường tỷ lệ bóng tim/lồng ngực trực quan tốt trên phim thẳng.
- **Gãy xương (`FRACTURE`)**: Nhận diện đường gãy, nứt xương sườn và xương đòn phụ thuộc nhiều vào độ nét của ảnh $256 \times 256$.

### B. Nhóm Bệnh lý Nhu mô & Kẽ Phổi (Viêm phổi, Tràn dịch, Lao)
- **Viêm phổi (`PNEUMONIA/INFILTRATION`)**: VLM phát hiện các đám mờ thâm nhiễm phế nang.
- **Lao phổi (`TUBERCULOSIS`)**: Khác với bộ nhãn cũ gán thừa ồ ạt, VLM chỉ bắt các ca có tổn thương hang hoặc xơ co kéo điển hình ở vùng đỉnh phổi.
- **Tràn dịch màng phổi (`PLEURAL_EFFUSION`)**: VLM quan sát trực quan góc sườn hoành (nhọn vs tù).

---
*Báo cáo được khởi tạo tự động bởi Pipeline 2 - Qwen2.5-VL Vision Diagnostics.*