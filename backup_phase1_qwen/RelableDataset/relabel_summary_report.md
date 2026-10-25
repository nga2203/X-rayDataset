# BÁO CÁO GÁN NHÃN LẠI TẬP TRAIN & VAL (OPTIMAL DEDUPLICATED RELABELING)

## 1. Thống Kê Tổng Quan
- **Tổng số dòng dữ liệu**: 12,462 ca Train + 1,770 ca Val = **14,232 ca**.
- **Khử trùng lặp (Deduplication)**: Chỉ cần gán nhãn cho **4,597 chuỗi kết luận duy nhất** (tiết kiệm **68% thời gian GPU**).

## 2. Bảng Đối Chiếu Số Lượng Nhãn Trước & Sau Khi Làm Sạch

| Nhãn Bệnh lý           |   Train Cũ |   Train Sạch (LLM) |   Train Chênh lệch |   Val Cũ |   Val Sạch (LLM) |   Val Chênh lệch |
|:-----------------------|-----------:|-------------------:|-------------------:|---------:|-----------------:|-----------------:|
| CARDIOMEGALY           |        539 |                548 |                  9 |       75 |               77 |                2 |
| DEGENERATION           |        977 |               2123 |               1146 |      138 |              304 |              166 |
| FRACTURE               |       1560 |               1658 |                 98 |      217 |              230 |               13 |
| NORMAL                 |       6677 |               7526 |                849 |      952 |             1054 |              102 |
| OTHER                  |       1735 |                198 |              -1537 |      244 |               35 |             -209 |
| PLEURAL_EFFUSION       |        101 |                 67 |                -34 |       15 |                7 |               -8 |
| PNEUMONIA/INFILTRATION |        521 |                377 |               -144 |       71 |               49 |              -22 |
| SPINAL_ALIGNMENT       |        177 |                128 |                -49 |       27 |               20 |               -7 |
| TUBERCULOSIS           |        756 |                 17 |               -739 |       98 |                4 |              -94 |

---
*Dữ liệu sạch sẵn sàng để huấn luyện lại mô hình VN-MultiXR v2.*

================================================================================
BẢNG ĐỐI CHIẾU SỐ LƯỢNG CA TRƯỚC VÀ SAU KHI LÀM SẠCH NHÃN (TRAIN & VAL)
================================================================================
          Nhãn Bệnh lý  Train Cũ  Train Sạch (LLM)  Train Chênh lệch  Val Cũ  Val Sạch (LLM)  Val Chênh lệch
          CARDIOMEGALY       539               548                 9      75              77               2
          DEGENERATION       977              2123              1146     138             304             166
              FRACTURE      1560              1658                98     217             230              13
                NORMAL      6677              7526               849     952            1054             102
                 OTHER      1735               198             -1537     244              35            -209
      PLEURAL_EFFUSION       101                67               -34      15               7              -8
PNEUMONIA/INFILTRATION       521               377              -144      71              49             -22
      SPINAL_ALIGNMENT       177               128               -49      27              20              -7
          TUBERCULOSIS       756                17              -739      98               4             -94
================================================================================