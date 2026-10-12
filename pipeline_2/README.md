# HƯỚNG DẪN THỰC THI PIPELINE 2: VLM-BASED X-RAY AUDIT & RELABELING

Hệ thống Pipeline 2 sử dụng **Vision-Language Model Qwen2.5-VL 7B (`qwen2.5vl:7b`)** chạy local qua Ollama để đọc trực tiếp ảnh X-quang và phân loại 9 bệnh lý mà không phụ thuộc vào văn bản kết luận của bác sĩ.

---

## 🛠️ CƠ CHẾ BẢO VỆ GPU & CHỐNG MẤT DỮ LIỆU ĐÃ TÍCH HỢP

1. **Chống Tràn & Tự động Giải phóng VRAM (Auto-Unload)**:
   - Trong quá trình chạy, model chỉ chiếm ~6.5GB / 12GB VRAM.
   - Khi script chạy xong (hoặc khi bạn bấm `Ctrl+C` ngắt giữa chừng), mã nguồn luôn tự động gửi lệnh `keep_alive: 0` tới Ollama để trả GPU về mức 0–1GB VRAM idle ngay lập tức.
2. **Cơ chế Checkpoint Nguyên tử (Atomic Cache)**:
   - Toàn bộ kết quả từng ca được ghi vào file JSON đệm (`RelableDataset_2/test_vlm_cache.json`).
   - Tự động lưu sau mỗi 20 ca. Nếu gặp sự cố, bạn chỉ cần chạy lại lệnh cũ; chương trình sẽ tự động đọc cache và chỉ chạy tiếp các ca còn thiếu (Resume 100%).

---

## 🚀 THỨ TỰ CÁC BƯỚC THỰC THI (WORKFLOW)

### BƯỚC 1: Tiền xử lý & Kiểm tra tính toàn vẹn hình ảnh
Kiểm tra xem tất cả các ca trong tập Test có file ảnh hợp lệ trên đĩa `C:/XQUANGRESIZE/test/` hay không.
```powershell
python pipeline_2/01_vlm_image_prep.py
```
- **Kết quả**: Tạo file `Data_2/clean_test_vlm_ready.csv`.

---

### BƯỚC 2: Chạy VLM gán nhãn thị giác cho tập TEST

#### Cách A: Chạy thử nghiệm trên 20 ca mẫu (Khuyên dùng trước khi chạy toàn bộ)
```powershell
python pipeline_2/02_vlm_label_audit_test.py --sample_size 20
```

#### Cách B: Chạy trên toàn bộ 3,523 ca tập Test
```powershell
python pipeline_2/02_vlm_label_audit_test.py
```
*(Nếu muốn lưu checkpoint thường xuyên hơn, ví dụ 10 ca/lần: thêm `--save_interval 10`)*
- **Kết quả**: Tạo file kết quả `Data_2/audited_vlm_test.csv` và checkpoint `RelableDataset_2/test_vlm_cache.json`.

---

### BƯỚC 3: Khai thác Thống kê Đồng xuất hiện & Phân tích Nhóm tuổi
Tính toán ma trận Cramér's V giữa các bệnh và F1/Recall theo 3 nhóm tuổi (18-30, 31-50, 50+).
```powershell
python pipeline_2/03_statistical_analysis_2.py
```
- **Kết quả**: Xuất `Data_2/cramers_v_matrix_vlm.csv` và `Data_2/age_subgroup_f1_recall_vlm.csv`.

---

### BƯỚC 4: Xuất Báo cáo Đối chiếu Chéo 3 Bên & Chẩn đoán Năng lực VLM
Lập bảng đối chiếu 3 bên (VLM Ảnh vs Bác sĩ Text vs Nhãn cũ) và phân tích chuyên sâu.
```powershell
python pipeline_2/04_vlm_model_diagnostics_2.py
```
- **Kết quả**: Xuất báo cáo Markdown tại `Data_2/vlm_model_diagnostics_report.md` và bảng chỉ số `Data_2/vlm_triangulation_metrics.csv`.

---

## 🔮 BƯỚC MỞ RỘNG (KHI TEST ĐÃ ĐẠT KẾT QUẢ TỐT)

### Gán nhãn cho tập Train & Val bằng VLM:
```powershell
# Chạy cho tập Val trước (1,770 ca):
python RelableDataset_2/01_vlm_relabel_train_val.py --split val

# Chạy cho tập Train (12,462 ca):
python RelableDataset_2/01_vlm_relabel_train_val.py --split train
```

### Chuẩn hóa định dạng nhãn dạng list string cho DataLoader:
```powershell
python RelableDataset_2/02_format_clean_datasets_2.py
```

### Tạo Bộ dữ liệu Đồng thuận Đa tầng (Multimodal Consensus 3 bên):
```powershell
python pipeline_2/05_create_multimodal_consensus_2.py
```

### Tạo Bộ dữ liệu Đồng thuận Riêng giữa Nhãn Gốc và Plan 2 (VLM 2 bên):
```powershell
python pipeline_2/06_create_vlm_old_consensus.py
```
- **Mục tiêu**: Lọc các mẫu đạt sự trùng khớp 100% chỉ giữa Nhãn Cũ bệnh viện và Nhãn Thị giác VLM (loại bỏ hoàn toàn phụ thuộc vào nhãn text bác sĩ).
- **Kết quả**:
  - Lưu tại thư mục dữ liệu riêng: `Data_2/consensus_vlm_old/` (`consensus_train.csv`, `consensus_val.csv`, `consensus_test.csv`)
  - Đồng bộ sang: `RelableDataset_2/consensus_vlm_old/`
  - Xuất log thống kê tại: `log_2/06_consensus_vlm_old.log`

