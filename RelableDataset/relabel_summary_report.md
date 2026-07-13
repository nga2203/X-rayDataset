# BÁO CÁO GÁN NHÃN LẠI TẬP TRAIN & VAL (OPTIMAL DEDUPLICATED RELABELING)

## 1. Thống Kê Tổng Quan
- **Tổng số dòng dữ liệu**: 12,462 ca Train + 1,770 ca Val = **14,232 ca**.
- **Khử trùng lặp (Deduplication)**: Chỉ cần gán nhãn cho **4,597 chuỗi kết luận duy nhất** (tiết kiệm **68% thời gian GPU**).

## 2. Bảng Đối Chiếu Số Lượng Nhãn Trước & Sau Khi Làm Sạch

| Nhãn Bệnh lý           |   Train Cũ |   Train Sạch (LLM) |   Train Chênh lệch |   Val Cũ |   Val Sạch (LLM) |   Val Chênh lệch |
|:-----------------------|-----------:|-------------------:|-------------------:|---------:|-----------------:|-----------------:|
| CARDIOMEGALY           |        539 |                539 |                  0 |       75 |               76 |                1 |
| DEGENERATION           |        977 |               1385 |                408 |      138 |              199 |               61 |
| FRACTURE               |       1560 |               1425 |               -135 |      217 |              196 |              -21 |
| NORMAL                 |       6677 |               4552 |              -2125 |      952 |              624 |             -328 |
| OTHER                  |       1735 |               4349 |               2614 |      244 |              646 |              402 |
| PLEURAL_EFFUSION       |        101 |                 75 |                -26 |       15 |                9 |               -6 |
| PNEUMONIA/INFILTRATION |        521 |                133 |               -388 |       71 |                9 |              -62 |
| SPINAL_ALIGNMENT       |        177 |                140 |                -37 |       27 |               21 |               -6 |
| TUBERCULOSIS           |        756 |                 20 |               -736 |       98 |                6 |              -92 |

---
*Dữ liệu sạch sẵn sàng để huấn luyện lại mô hình VN-MultiXR v2.*