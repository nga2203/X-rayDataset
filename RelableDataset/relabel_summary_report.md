# BÁO CÁO GÁN NHÃN LẠI TẬP TRAIN & VAL (LLM: gemma3:12b)

## 1. Thống Kê Tổng Quan
- **Tổng số ca**: 12,462 ca Train + 1,770 ca Val = **14,232 ca**.
- **Khử trùng lặp (Deduplication)**: Rút gọn còn **4,597 chuỗi kết luận duy nhất**.
- **Kế thừa Test-Cache**: 0 câu được nạp từ tập Test (0s GPU).
- **Fast Normal Bypass**: 0 câu bình thường thuần túy (0s GPU).
- **Số câu xử lý thực tế qua LLM**: 683 câu với 3 luồng GPU song song.

## 2. Bảng Đối Chiếu Số Lượng Nhãn Trước & Sau Khi Làm Sạch

| Nhãn Bệnh lý           |   Train Cũ |   Train Sạch (LLM) |   Train Chênh lệch |   Val Cũ |   Val Sạch (LLM) |   Val Chênh lệch |
|:-----------------------|-----------:|-------------------:|-------------------:|---------:|-----------------:|-----------------:|
| CARDIOMEGALY           |        539 |                604 |                 65 |       75 |               83 |                8 |
| DEGENERATION           |        977 |               1785 |                808 |      138 |              252 |              114 |
| FRACTURE               |       1560 |               1670 |                110 |      217 |              234 |               17 |
| NORMAL                 |       6677 |               6644 |                -33 |      952 |              942 |              -10 |
| OTHER                  |       1735 |               2375 |                640 |      244 |              353 |              109 |
| PLEURAL_EFFUSION       |        101 |                176 |                 75 |       15 |               25 |               10 |
| PNEUMONIA/INFILTRATION |        521 |                634 |                113 |       71 |               85 |               14 |
| SPINAL_ALIGNMENT       |        177 |                192 |                 15 |       27 |               29 |                2 |
| TUBERCULOSIS           |        756 |                 25 |               -731 |       98 |                7 |              -91 |

---
*Dữ liệu sạch sẵn sàng để huấn luyện mô hình VN-MultiXR v2 với chuẩn gán nhãn gemma3:12b.*