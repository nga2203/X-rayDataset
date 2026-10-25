# BẢN SAO LƯU DỮ LIỆU PHASE 1 (MÔ HÌNH QWEN 2.5 7B-INSTRUCT)

Thư mục này chứa toàn bộ dữ liệu, cache kết quả, báo cáo và log của Phase 1 gốc (khi sử dụng mô hình LLM `qwen2.5:7b-instruct`).
Được trích xuất từ Git commit `HEAD` (`8df6be02d5afac21f66b60340cdfd50f1de7ebda`).

---

## 1. Cấu trúc thư mục sao lưu

### `Data/`
- `audited_pipeline_test.csv`: 3,523 ca kiểm định nhãn tự động trên tập test bằng Qwen 2.5.
- `clean_pipeline_test.csv`: File test đã làm sạch đầu vào của pipeline.
- `consensus_test.csv`, `consensus_train.csv`, `consensus_val.csv`: Bộ dữ liệu đồng thuận giữa nhãn cũ và nhãn Qwen 2.5.
- `cramers_v_matrix.csv`, `cramers_v_matrix_8.csv`: Ma trận tương quan Cramér's V giữa các bệnh lý theo Qwen 2.5.
- `age_subgroup_f1_recall.csv`, `age_subgroup_f1_recall_8.csv`: Phân tích hiệu năng phân lớp theo 3 nhóm tuổi.
- `model_diagnostics_report.md`: Báo cáo chẩn đoán lỗi mô hình VN-MultiXR dựa trên kiểm định nhãn Qwen 2.5.

### `RelableDataset/`
- `label_cache.json`: Cache toàn bộ 4,597 kết luận duy nhất gán bởi Qwen 2.5 7B-Instruct.
- `clean_train.csv`, `clean_val.csv`: Bộ dữ liệu train/val đã được gán nhãn lại bởi Qwen 2.5.
- `clean_train_formatted.csv`, `clean_val_formatted.csv`, `clean_test_formatted.csv`: Dữ liệu đã chuẩn hóa định dạng cột.
- `consensus_train.csv`, `consensus_val.csv`, `consensus_test.csv`: Dữ liệu đồng thuận lưu tại RelableDataset.
- `relabel_summary_report.md`: Báo cáo tổng kết phân phối nhãn của Qwen 2.5.

### `log/`
- `02.log`: Log chạy kiểm định nhãn tập test bằng Qwen 2.5.
- `03.log`, `03x2.log`: Log chạy phân tích tương quan và nhóm tuổi.
- `04.log`: Log chạy chẩn đoán model.
- `05.log`: Log tạo tập dữ liệu đồng thuận.

---
Mục đích: Sử dụng để so sánh đối chiếu hiệu năng, độ chính xác (Accuracy / Kappa / F1) giữa Qwen 2.5 7B và Google Gemma 3 12B sau khi hoàn tất kiểm định trên toàn bộ tập dữ liệu.
