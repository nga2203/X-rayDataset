# BÁO CÁO CHẨN ĐOÁN & GIẢI THÍCH LỖI MÔ HÌNH VN-MultiXR

Báo cáo này phân tách nguyên nhân các sai số của mô hình **VN-MultiXR** dựa trên kiểm định nhãn LLM và thống kê đồng xuất hiện.

## 1. Bảng Đối Chiếu Hiệu Năng & Chẩn Đoán Nguyên Nhân Gốc Rễ

| Nhãn Bệnh lý           |   Model F1 |   Model Precision |   Model Recall |   Kappa (Nhãn Cũ vs Chuẩn) | Chẩn Đoán Gốc Rễ                                                                     |
|:-----------------------|-----------:|------------------:|---------------:|---------------------------:|:-------------------------------------------------------------------------------------|
| CARDIOMEGALY           |      0.461 |             0.691 |          0.346 |                      0.922 | 🟡 Lỗi do Kiến trúc / Thiếu ảnh hưởng hình ảnh đặc thù                               |
| DEGENERATION           |      0.812 |             0.921 |          0.726 |                      0.679 | 🟢 Hiệu năng tốt / Ổn định                                                           |
| FRACTURE               |      0.794 |             0.824 |          0.767 |                      0.918 | 🟢 Hiệu năng tốt / Ổn định                                                           |
| NORMAL                 |      0.772 |             0.775 |          0.768 |                      0.85  | 🟢 Hiệu năng tốt / Ổn định                                                           |
| OTHER                  |      0.227 |             0.18  |          0.306 |                      0.205 | 🔴 Lỗi do Nhiễu Nhãn Cũ (Label Noise Bias - Nhãn cũ gán sai/thừa làm model học lệch) |
| PLEURAL_EFFUSION       |      0.37  |             0.294 |          0.5   |                      0.715 | 🟡 Lỗi do Kiến trúc / Thiếu ảnh hưởng hình ảnh đặc thù                               |
| PNEUMONIA/INFILTRATION |      0.552 |             0.471 |          0.667 |                      0.813 | 🟢 Hiệu năng tốt / Ổn định                                                           |
| SPINAL_ALIGNMENT       |      0.162 |             0.1   |          0.412 |                      0.858 | 🟡 Lỗi do Kiến trúc / Thiếu ảnh hưởng hình ảnh đặc thù                               |
| TUBERCULOSIS           |      0     |             0     |          0     |                      0.106 | 🔴 Lỗi do Nhiễu Nhãn Cũ (Label Noise Bias - Nhãn cũ gán sai/thừa làm model học lệch) |

## 2. Phân Tích Chuyên Sâu Theo Từng Trường Hợp

### Ví Dụ 1: Lỗi Gán Nhầm & Thiên Kiến Đồng Xuất Hiện (Co-occurrence Bias)
- Khi 2 bệnh lý có chỉ số **Cramér's V cao** (VD: Thoái hóa cột sống & Các triệu chứng đau khớp khác), mô hình dễ bị nhầm lẫn do dữ liệu có xu hướng đi kèm nhau thường xuyên. Lỗi này bắt nguồn từ **thiên kiến phân bố dữ liệu (Data Distribution Bias)**.

### Ví Dụ 2: Lỗi F1 Thấp ở Lớp Tràn Dịch / Viêm Phổi / Lao Phổi (Label Noise Bias)
- Nhãn **Tràn dịch màng phổi (`PLEURAL_EFFUSION`)**, **Viêm phổi (`PNEUMONIA/INFILTRATION`)**, **Lao phổi (`TUBERCULOSIS`)** có chỉ số **Cohen's Kappa thấp** giữa quy trình gán tự động cũ và kết luận lâm sàng chuẩn.
- **Kết luận**: Hiệu năng model trên các lớp này bị kéo thấp chủ yếu do **khâu tạo nhãn tự động cũ (OCR/Rule match) gán sai hoặc gán thừa quá nhiều False Positive**.

---
*Báo cáo được khởi tạo tự động bởi Pipeline VN-MultiXR Diagnostics.*