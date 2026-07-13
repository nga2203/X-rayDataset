# BÁO CÁO CHẨN ĐOÁN & GIẢI THÍCH LỖI MÔ HÌNH VN-MultiXR

Báo cáo này phân tách nguyên nhân các sai số của mô hình **VN-MultiXR** dựa trên kiểm định nhãn LLM và thống kê đồng xuất hiện.

## 1. Bảng Đối Chiếu Hiệu Năng & Chẩn Đoán Nguyên Nhân Gốc Rễ

| Nhãn Bệnh lý           |   Model F1 |   Model Precision |   Model Recall |   Kappa (Nhãn Cũ vs Chuẩn) | Chẩn Đoán Gốc Rễ                                                                     |
|:-----------------------|-----------:|------------------:|---------------:|---------------------------:|:-------------------------------------------------------------------------------------|
| CARDIOMEGALY           |      0.473 |             0.649 |          0.372 |                      0.994 | 🟡 Lỗi do Kiến trúc / Thiếu ảnh hưởng hình ảnh đặc thù                               |
| DEGENERATION           |      0.737 |             0.926 |          0.611 |                      0.58  | 🟢 Hiệu năng tốt / Ổn định                                                           |
| FRACTURE               |      0.813 |             0.831 |          0.796 |                      0.927 | 🟢 Hiệu năng tốt / Ổn định                                                           |
| NORMAL                 |      0.791 |             0.861 |          0.731 |                      0.817 | 🟢 Hiệu năng tốt / Ổn định                                                           |
| OTHER                  |      0.041 |             0.021 |          0.767 |                      0.077 | 🔴 Lỗi do Nhiễu Nhãn Cũ (Label Noise Bias - Nhãn cũ gán sai/thừa làm model học lệch) |
| PLEURAL_EFFUSION       |      0.151 |             0.094 |          0.381 |                      0.527 | 🔴 Lỗi do Nhiễu Nhãn Cũ (Label Noise Bias - Nhãn cũ gán sai/thừa làm model học lệch) |
| PNEUMONIA/INFILTRATION |      0.425 |             0.297 |          0.75  |                      0.73  | 🟡 Lỗi do Kiến trúc / Thiếu ảnh hưởng hình ảnh đặc thù                               |
| SPINAL_ALIGNMENT       |      0.107 |             0.062 |          0.382 |                      0.793 | 🟡 Lỗi do Kiến trúc / Thiếu ảnh hưởng hình ảnh đặc thù                               |
| TUBERCULOSIS           |      0     |             0     |          0     |                      0.081 | 🔴 Lỗi do Nhiễu Nhãn Cũ (Label Noise Bias - Nhãn cũ gán sai/thừa làm model học lệch) |

## 2. Phân Tích Chuyên Sâu Theo Từng Trường Hợp

### Ví Dụ 1: Lỗi Gán Nhầm & Thiên Kiến Đồng Xuất Hiện (Co-occurrence Bias)
- Khi 2 bệnh lý có chỉ số **Cramér's V cao** (VD: Thoái hóa cột sống & Các triệu chứng đau khớp khác), mô hình dễ bị nhầm lẫn do dữ liệu có xu hướng đi kèm nhau thường xuyên. Lỗi này bắt nguồn từ **thiên kiến phân bố dữ liệu (Data Distribution Bias)**.

### Ví Dụ 2: Lỗi F1 Thấp ở Lớp Tràn Dịch / Viêm Phổi / Lao Phổi (Label Noise Bias)
- Nhãn **Tràn dịch màng phổi (`PLEURAL_EFFUSION`)**, **Viêm phổi (`PNEUMONIA/INFILTRATION`)**, **Lao phổi (`TUBERCULOSIS`)** có chỉ số **Cohen's Kappa thấp** giữa quy trình gán tự động cũ và kết luận lâm sàng chuẩn.
- **Kết luận**: Hiệu năng model trên các lớp này bị kéo thấp chủ yếu do **khâu tạo nhãn tự động cũ (OCR/Rule match) gán sai hoặc gán thừa quá nhiều False Positive**.

---
*Báo cáo được khởi tạo tự động bởi Pipeline VN-MultiXR Diagnostics.*