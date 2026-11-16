# BÁO CÁO KIỂM TRA CHÉO ĐỘ NHẤT QUÁN SỐ LIỆU (CONSISTENCY CHECK — P11 / B6)

- **Thời điểm kiểm tra tự động:** `2026-11-16T10:49:36Z`
- **Nguồn đối chiếu gốc (Single Source of Truth):** `reports/facts.json`
- **Tổng số chỗ còn `[CHƯA CÓ SỐ]` trên toàn bộ 7 tài liệu:** **`0`** (Yêu cầu = `0`)
- **Kết luận tổng thể:** **ĐẠT TUYỆT ĐỐI 100% (PASS — ZERO DISCREPANCIES)**

---

## 1. BẢNG ĐỐI CHIẾU TỪNG CON SỐ CHỐT GIỮA `facts.json` VÀ BỘ TÀI LIỆU

| # | Chỉ số / Con số kiểm tra | Giá trị trong `facts.json` | Số tài liệu đối chiếu | Trạng thái kiểm tra |
|---:|---|---|---:|---|
| 1 | Quy mô bộ dữ liệu được giao (Twitter_Data.csv) | `162,980` | 6 | **KHỚP 100% (PASS)** |
| 2 | Quy mô bộ dữ liệu Hate Speech thô (hate_train.csv) | `31,962` | 6 | **KHỚP 100% (PASS)** |
| 3 | Quy mô bộ dữ liệu sạch sau khử trùng (clean.csv) | `28,330` | 6 | **KHỚP 100% (PASS)** |
| 4 | Số dòng trùng lặp gần-giống bị loại (P4) | `3,628` | 5 | **KHỚP 100% (PASS)** |
| 5 | Tên mô hình chốt (best_model) | `Linear SVM (Calibrated)` | 6 | **KHỚP 100% (PASS)** |
| 6 | Ngưỡng quyết định tối ưu (best_threshold) | `0.315` | 7 | **KHỚP 100% (PASS)** |
| 7 | Validation F1 sau tinh chỉnh & tối ưu ngưỡng (P6) | `0.6566` | 6 | **KHỚP 100% (PASS)** |
| 8 | Test F1 Lớp dương (Final Test F1 — P8) | `0.6266` | 7 | **KHỚP 100% (PASS)** |
| 9 | Test PR-AUC (P8) | `0.6651` | 6 | **KHỚP 100% (PASS)** |
| 10 | Test ROC-AUC (P8) | `0.9345` | 5 | **KHỚP 100% (PASS)** |
| 11 | Hệ số đồng thuận Cohen's Kappa tập Thực tế (P9) | `0.9204` | 6 | **KHỚP 100% (PASS)** |
| 12 | Real-World F1 tổng thể trên 300 mẫu (P9) | `0.5702` | 6 | **KHỚP 100% (PASS)** |

---

## 2. KIỂM TRA KHÔNG CÒN PLACEHOLDER `[CHƯA CÓ SỐ]` TRONG TỪNG TÀI LIỆU

| Tài liệu | Độ dài (ký tự) | Số lượng `[CHƯA CÓ SỐ]` | Trạng thái |
|---|---:|---:|---|
| `reports/report.docx` | 52,597 | 0 | ✅ HỢP LỆ (0 placeholder) |
| `reports/slides.pptx` | 11,422 | 0 | ✅ HỢP LỆ (0 placeholder) |
| `reports/speaker_script.md` | 14,761 | 0 | ✅ HỢP LỆ (0 placeholder) |
| `reports/qa_bank.md` | 74,737 | 0 | ✅ HỢP LỆ (0 placeholder) |
| `reports/demo_script.md` | 5,211 | 0 | ✅ HỢP LỆ (0 placeholder) |
| `reports/cheat_sheet.md` | 4,855 | 0 | ✅ HỢP LỆ (0 placeholder) |
| `README.md` | 8,254 | 0 | ✅ HỢP LỆ (0 placeholder) |
