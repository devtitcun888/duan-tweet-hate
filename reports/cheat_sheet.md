# BẢNG SỐ LIỆU CHỐT 1 TRANG (CHEAT SHEET — P11 / B5)

> **Dùng để tra nhanh trong lúc bảo vệ đồ án.** Mọi con số được trích xuất tự động từ `reports/facts.json` (Seed = `42`).

---

## 1. SÁU CON SỐ VÀNG CẦN THUỘC LÒNG

| STT | Hạng mục | Con số chính xác | Ghi chú / Ý nghĩa |
|---|---|---|---|
| **1** | **Quy mô dữ liệu huấn luyện** | **31,962 dòng thô → 28,330 dòng sạch** | Đã loại **3,628** dòng trùng/retweet + **2** dòng mâu thuẫn nhãn + **2** dòng rỗng |
| **2** | **Tỉ lệ lớp & Mất cân bằng** | **Lớp 1 = 7.01%** (thô) / **6.52%** (sạch) | Tỉ lệ lệch **13.26:1** (Lớp 0 = 26,484, Lớp 1 = 1,846 sau khử trùng) |
| **3** | **Mô hình chốt & Ngưỡng tối ưu** | **`Linear SVM (Calibrated)` (`Thr = 0.315`)** | `ngram_range=(1,3)`, `min_df=2`, `max_features=50,000`, `C=0.50191`, `loss='squared_hinge'`, `class_weight='balanced'` |
| **4** | **Val F1 Trước → Sau Tuning** | **0.5727 → 0.6566** (**+0.0839**) | Tăng **+8.39 điểm %** nhờ kết hợp RandomizedSearchCV và hạ ngưỡng quyết định từ `0.500` về `0.315` |
| **5** | **Kết quả Tập Test độc lập (P8)** | **Test F1 = 0.6266** (PR-AUC = **0.6651**) | 95% CI Bootstrap: **[0.5764, 0.6716]**; Precision = **0.6324**, Recall = **0.6209**, ROC-AUC = **0.9345** |
| **6** | **Kết quả Tập Thực tế (P9)** | **Real-World F1 = 0.5702** (**Kappa $\kappa = 0.9204$**) | $N = 300$ mẫu (4 nguồn A/B/C/D), đồng thuận 2 người gắn nhãn đạt **97.00%** |

---

## 2. TÓM TẮT CÁC BỘ DỮ LIỆU & CHIA TẬP (70 / 15 / 15)

| Bộ dữ liệu / Tập | Số dòng | Phân bố nhãn | Vai trò trong dự án |
|---|---:|---|---|
| `Twitter_Data.csv` (Bộ được giao) | 162,980 (162,969 hợp lệ) | `+1`: 72,250 \| `0`: 55,213 \| `-1`: 35,510 \| `NaN`: 7 | **Chỉ có nhãn sentiment chính trị (-1/0/+1)**; dùng cho EDA (Câu 1) & kiểm thử lệch miền (Câu 3) |
| `Reddit_Data.csv` (Bộ được giao) | 37,249 (37,149 hợp lệ) | `+1`: 15,830 \| `0`: 13,142 \| `-1`: 8,277 \| `NaN`: 100 | Dùng lấy mẫu thực tế khác nền tảng (Nguồn C ở P9) |
| `hate_train.csv` (Phương án A) | 31,962 thô → 28,330 sạch | Lớp 0: 26,484 (93.48%) \| Lớp 1: 1,846 (6.52%) | Bộ dữ liệu chuẩn có nhãn `0/1` Racist/Sexist dùng để chia tập huấn luyện |
| `data/splits/train.csv` (70%) | 19,830 | Lớp 0: 18,538 \| Lớp 1: 1,292 (6.52%) | Huấn luyện mô hình, 5-fold CV, phân tích SQL (`db/tweets.db`) |
| `data/splits/val.csv` (15%) | 4,250 | Lớp 0: 3,973 \| Lớp 1: 277 (6.52%) | So sánh mô hình (P5), chọn siêu tham số & tối ưu ngưỡng quyết định (P6) |
| `data/splits/test.csv` (15%) | 4,250 | Lớp 0: 3,973 \| Lớp 1: 277 (6.52%) | Giữ kín hoàn toàn, **chỉ chạy đúng 1 lần duy nhất ở P8** (`2026-11-16T10:26:46Z`) |
| `real_world_labeled.csv` (P9) | 300 | Lớp 0: 226 \| Lớp 1: 74 ($\kappa = 0.9204$) | Kiểm thử thực tế 4 nguồn: A (100), B (100), C (50), D (50 câu hiểm hóc) |
| `davidson_binary.csv` (Cross-test) | 5,593 | Lớp 0: 4,163 \| Lớp 1: 1,430 | Kiểm thử chéo ngoài miền (Cross-dataset evaluation) |

---

## 3. BẢNG TÓM TẮT KẾT QUẢ CÁC MÔ HÌNH CHÍNH (TẬP VALIDATION & TEST)

| Mô hình | Đặc trưng | CV F1 (Train) | Val Precision | Val Recall | Val F1 | Val PR-AUC | Val ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| **Linear SVM (Calibrated — Tuned, Thr=0.315)** | Word TF-IDF (1-3 gram) | **0.6112** | **0.6877** | **0.6282** | **0.6566** | **0.6636** | **0.9321** |
| Logistic Regression (Untuned) | Word+Char TF-IDF | 0.6478 | 0.5795 | 0.7365 | 0.6486 | 0.7090 | 0.9466 |
| Logistic Regression (Tuned, Thr=0.565) | Word TF-IDF (1-2 gram) | 0.5976 | 0.6367 | 0.6137 | 0.6250 | 0.6419 | 0.9198 |
| Linear SVM (Calibrated — Untuned) | Word+Char TF-IDF | 0.5569 | 0.8323 | 0.4657 | 0.5972 | 0.7039 | 0.9342 |
| Complement Naive Bayes (Tuned) | Word TF-IDF (1-1 gram) | 0.5858 | 0.4951 | 0.7365 | 0.5922 | 0.6985 | 0.9377 |
| LightGBM (Untuned) | Word TF-IDF | 0.5416 | 0.4562 | 0.6209 | 0.5260 | 0.5713 | 0.9035 |
| Random Forest (Untuned) | Word TF-IDF | 0.5344 | 0.4447 | 0.6245 | 0.5195 | 0.5950 | 0.9072 |
| XGBoost (Untuned) | Word TF-IDF | 0.4990 | 0.4053 | 0.6643 | 0.5034 | 0.5513 | 0.8955 |
| RBF-Kernel SVM (Mẫu 8k) | Word TF-IDF | 0.3741 | 0.8925 | 0.2996 | 0.4486 | 0.5988 | 0.8994 |
| Cosine KNN | Word TF-IDF | 0.3796 | 0.2955 | 0.5740 | 0.3902 | 0.2857 | 0.7981 |
| Decision Tree | Word TF-IDF | 0.3669 | 0.3316 | 0.4549 | 0.3836 | 0.2582 | 0.7296 |
| **BiLSTM (PyTorch Bonus — P7)** | Embedding 128d | 0.5973 | 0.5663 | 0.6318 | **0.5973** | 0.6168 | 0.9133 |
| **DistilBERT (Fine-tuned Bonus — P7)** | Subword WP (len=80) | 0.4434 | 0.3010 | 0.8412 | **0.4434** | 0.5600 | 0.9266 |

---

## 4. LỆNH CHẠY NHANH (LOCAL & GOOGLE COLAB)
- **Chạy ứng dụng Web Streamlit:** `streamlit run app/app.py`
- **Chạy kiểm thử tự động (17 unit tests):** `pytest -q`
- **Tái lập toàn bộ pipeline trên CPU:** `python run_all.py --cpu-only`
- **Notebook Google Colab:** Mở lần lượt `notebooks/00_colab_bootstrap.ipynb` → `08_realworld_test.ipynb` và chọn `Runtime → Run all`.


---

## 5. BẢNG SỐ LIỆU CHỐT PHẦN MỞ RỘNG TIẾNG VIỆT (ViHSD — V0 → V8)

| Hạng mục Tiếng Việt (ViHSD) | Con số chính xác | Ghi chú / Ý nghĩa |
|---|---|---|
| **Quy mô & Chia tập chuẩn ViHSD** | **33,400 dòng** (`24,048` train / `2,672` dev / `6,680` test) | Nhãn: `CLEAN` = 27,624 (82.71%) \| `HATE` = 3,514 (10.52%) \| `OFFENSIVE` = 2,262 (6.77%) |
| **Kiểm tra trùng chéo Split chuẩn** | `train∩dev` = **322** \| `train∩test` = **791** \| `dev∩test` = **92** | Giữ nguyên chia chuẩn của Luu et al. (2021) để so sánh công bằng với bài báo gốc |
| **Đặc trưng ngôn ngữ ViHSD** | Không dấu = **7.78%** \| Có Teencode = **24.18%** | Độ phủ từ điển `vi_profanity.txt`: `CLEAN` = 3.65%, `OFFENSIVE` = 37.80%, `HATE` = 35.92% |
| **Mô hình chốt `bin_A` (HATE vs CLEAN)** | **`Linear SVM (Calibrated) (word+char)`** (`Thr = 0.285`) | Dev F1 = **0.6736** \| **Test F1 = 0.6658** (95% CI: `[0.6379, 0.6912]`, PR-AUC = **0.7198**) |
| **Kết quả Test `bin_B` & `multi3` (3 lớp)** | `bin_B` Test F1 = **0.6855** \| `multi3` **Macro-F1 = 0.6481** | `multi3` Accuracy = **84.31%** (Khớp mốc baseline công bố `0.60 – 0.66`, PhoBERT-CNN `~0.675`) |
| **Transformer 3 Seeds (42, 43, 44 — V4)** | `PhoBERT-v2` Dev F1 = **0.5732 ± 0.0034** | `XLM-RoBERTa` Dev F1 = **0.4861 ± 0.0100** |
| **Kiểm tra Độ bền (Robustness — V5)** | Gốc: **0.6658** \| Teencode: **0.6204** \| **Không dấu: 0.5102** | Bỏ dấu hoàn toàn (`strip_accents`) làm F1 giảm **-0.1556** điểm |
| **Bộ Thực tế Tiếng Việt ($N = 210$ — V6)** | **Kappa 3 lớp $\kappa = 0.8743$** (Weighted $\kappa = 0.9538$) | Đồng thuận **92.86%** \| **Real-World `bin_A` F1 = 0.8034** (PR-AUC = **0.9074**) |
