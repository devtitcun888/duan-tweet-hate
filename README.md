# Gắn Nhãn Các Đoạn Tweet Có Nội Dung Phân Biệt Chủng Tộc, Phân Biệt Giới Tính (Twitter Sentimental Analysis)

> **Loại đề tài:** Phân loại nhị phân & Khai phá văn bản (*Binary Classification / Text Mining*)  
> **Môi trường hỗ trợ:** Google Colab & Python Local (`seed = 42` cố định toàn dự án)  
> **Mô hình chốt (`best_model`):** `Linear SVM (Calibrated)` (`threshold = 0.315`) — **Test F1 = `0.6266`** (95% CI: `[0.5764, 0.6716]`), **PR-AUC = `0.6651`**, **ROC-AUC = `0.9345`**

---

## 1. Giới thiệu & Quyết định Phương pháp luận về Dữ liệu

Đề tài xây dựng hệ thống tự động phát hiện và gắn nhãn nhị phân các đoạn tweet có nội dung **phân biệt chủng tộc (racism)** hoặc **phân biệt giới tính (sexism)**:
- **Nhãn `1` (Phản cảm — Racist/Sexist):** Tweet chứa ngôn từ thù ghét, kỳ thị, miệt thị hoặc hạ thấp nhân phẩm dựa trên chủng tộc, sắc tộc, màu da, tôn giáo hoặc giới tính.
- **Nhãn `0` (Bình thường — Non-Hate):** Giao tiếp đời thường, tin tức, bình luận chính trị hoặc câu lên án nạn phân biệt đối xử (*counter-speech*).

### Phát hiện quan trọng về bộ dữ liệu được giao (`Twitter_Data.csv`)
Qua kiểm định thực tế tại giai đoạn P1 (`src/data_io.py` & `reports/data_report.json`):
- Bộ dữ liệu được giao trong đề (`cosmos98/twitter-and-reddit-sentimental-analysis-dataset`, file `Twitter_Data.csv` gồm **162,980 dòng**) chỉ có 2 cột `clean_text` và `category` (`-1.0, 0.0, +1.0`), phản ánh **cảm xúc chính trị Ấn Độ** (xoay quanh thủ tướng Modi và bầu cử), **hoàn toàn không có nhãn racist/sexist**.
- **Quyết định xử lý (Phương án A):**
  1. Sử dụng bộ dữ liệu chuẩn **Tweets Hate Speech Detection** (**31,962 dòng** có nhãn nhị phân `0/1` racist/sexist chuẩn xác) để làm sạch, khử trùng (**loại 3,628 dòng trùng/retweet** và **2 dòng mâu thuẫn nhãn**, còn **28,330 dòng sạch**) và chia tập huấn luyện/đánh giá.
  2. Sử dụng trọn vẹn bộ **`Twitter_Data.csv` (162,980 dòng)**, **`Reddit_Data.csv` (37,249 dòng)** và bộ **Davidson et al. (24,783 dòng)** cho phân tích thống kê mô tả (Câu 1) và kiểm thử thực tế khác miền (*domain shift* — Câu 3).

---

## 2. Cấu trúc Thư mục Dự án

```text
tweet-hate/
├─ AGENTS.md                      # Đặc tả kỹ thuật chi tiết P0 -> P12
├─ README.md                      # Tài liệu hướng dẫn chạy lại toàn diện
├─ requirements.txt               # Danh sách thư viện tương thích Colab & Local
├─ config.yaml                    # Cấu hình trung tâm (seed=42, đường dẫn, tham số)
├─ run_all.py                     # Script chạy tái lập toàn bộ pipeline (CPU/GPU)
├─ data/
│   ├─ raw/                       # Dữ liệu gốc tải tự động (Twitter_Data, Reddit_Data, hate_train, davidson)
│   ├─ processed/clean.csv        # Dữ liệu sau làm sạch 12 bước & khử trùng (28,330 dòng)
│   ├─ splits/{train,val,test}.csv# Chia cố định 70/15/15 (19,830 / 4,250 / 4,250 dòng)
│   └─ real_world/                # Bộ thực tế 300 mẫu (real_world_labeled.csv, to_label.csv, ANNOTATION_GUIDE.md)
├─ db/tweets.db                   # Cơ sở dữ liệu SQLite phục vụ truy vấn SQL (P3)
├─ sql/                           # 5 kịch bản SQL (00_schema, 10_token_freq, 20_log_odds, 30_cooccur, 40_lexicon)
├─ src/                           # Mã nguồn mô-đun hóa (utils, data_io, clean, features, models, evaluate, threshold, predict, transformer)
├─ notebooks/                     # 9 Jupyter Notebooks (00_colab_bootstrap.ipynb -> 08_realworld_test.ipynb) đã chạy sẵn output
├─ app/app.py                     # Ứng dụng Web Streamlit 3 Tab tiếng Việt
├─ models/                        # Mô hình chốt (best_model.joblib, threshold.json, model_card.json)
├─ reports/                       # Toàn bộ báo cáo (report.docx/pdf, slides.pptx/pdf, qa_bank.md, speaker_script.md, demo_script.md, cheat_sheet.md, facts.json, figures/, tables/)
└─ tests/                         # Bộ 17 unit tests tự động (test_clean.py, test_leakage.py, test_predict.py)
```

---

## 3. Hướng dẫn Kết nối & Chạy trên Google Colab và Máy Local

### Cách 1: Chạy trực tiếp trên Google Colab (Khuyên dùng khi cần GPU T4)
1. Tải thư mục dự án lên Google Drive tại đường dẫn `/content/drive/MyDrive/tweet-hate` (hoặc `/content/drive/MyDrive/DOAN_COLAB`).
2. Mở bất kỳ notebook nào trong thư mục `notebooks/` (từ `00_colab_bootstrap.ipynb` đến `08_realworld_test.ipynb`) bằng **Google Colab**.
3. Cell đầu tiên (`bootstrap`) sẽ tự động nhận diện môi trường Colab, Mount Google Drive, cài đặt `requirements.txt`, tải tài nguyên NLTK và cố định `seed = 42`.
4. Chọn **Runtime → Run all** để chạy lại toàn bộ thực nghiệm từ đầu đến cuối.

### Cách 2: Chạy lại trên Máy Local (Windows / Linux / macOS)
```bash
# 1. Tạo môi trường ảo và cài đặt thư viện
python -m venv .venv
.venv\Scripts\activate          # Trên Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2. Chạy bộ kiểm thử tự động (17 unit tests kiểm tra làm sạch, chống rò rỉ dữ liệu & dự đoán)
pytest -q

# 3. Chạy lại toàn bộ pipeline tái lập kết quả (P1 -> P6, P8, P9)
python run_all.py --cpu-only

# 4. Khởi động ứng dụng Web Streamlit (tại http://localhost:8501)
streamlit run app/app.py
```

### Cách 3: Chạy Ứng dụng Streamlit ngay trong Google Colab qua Cloudflared Tunnel
Trong một ô code trên Google Colab, chạy:
```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!nohup streamlit run app/app.py --server.port 8501 &
!./cloudflared tunnel --url http://localhost:8501
```

---

## 4. Nguồn Dữ liệu & Giấy phép (Licenses)

| Bộ dữ liệu | Nguồn tải | Số dòng | Giấy phép |
|---|---|---:|---|
| `Twitter_Data.csv` & `Reddit_Data.csv` | Kaggle: `cosmos98/twitter-and-reddit-sentimental-analysis-dataset` | 162,980 & 37,249 | CC0: Public Domain |
| `Tweets Hate Speech Detection` | HuggingFace: `tweets-hate-speech-detection/tweets_hate_speech_detection` | 31,962 (train) / 17,197 (test) | MIT / Public Research |
| `Hate Speech and Offensive Language` | Davidson et al. (ICWSM 2017) — `tdavidson/hate_speech_offensive` | 24,783 (5,593 binary) | MIT License |

---

## 5. Tổng hợp Kết quả Thực nghiệm Chính (Đồng bộ từ `reports/facts.json`)

### 5.1. So sánh các Mô hình trên Tập Validation ($N = 4,250$) & Tinh chỉnh (P5 – P7)
- **Chiến lược cân bằng dữ liệu được chọn (P4):** `RandomOverSampler (ros)` đạt 5-fold CV F1 = **0.5876 ± 0.0117** (so với không cân bằng chỉ đạt F1 = **0.2390**).
- **Mô hình chốt sau tinh chỉnh (P6):** **`Linear SVM (Calibrated)`** với `ngram_range=(1,3)`, `min_df=2`, `max_features=50,000`, `C=0.50191`, `loss='squared_hinge'`, `class_weight='balanced'` và ngưỡng tối ưu **`threshold = 0.315`** giúp tăng Validation F1 từ **0.5727** lên **`0.6566`** (**+8.39 điểm %**).
- **Mô hình Học sâu / Transformer điểm cộng (P7):** `BiLSTM (PyTorch)` đạt Val F1 = **`0.5973`**, `DistilBERT (Fine-tuned)` đạt Val F1 = **`0.4434`**.

### 5.2. Kết quả Đánh giá Cuối trên Tập Test Độc lập ($N = 4,250$ — Chạy 1 lần duy nhất ở P8)
- **Test F1 Lớp dương (`f1_pos`):** **`0.6266`** (Khoảng tin cậy 95% Bootstrap: **`[0.5764, 0.6716]`**)
- **Test Precision / Recall Lớp dương:** **`0.6324`** / **`0.6209`**
- **Test PR-AUC / ROC-AUC / MCC:** **`0.6651`** / **`0.9345`** / **`0.6008`**
- **Ma trận nhầm lẫn Test:** `TN = 3873`, `FP = 100`, `FN = 105`, `TP = 172`

### 5.3. Kết quả Kiểm thử Thực tế Đa nguồn ($N = 300$ mẫu — P9)
- **Độ đồng thuận 2 người gắn nhãn độc lập:** Đồng thuận **`97.00%`**, hệ số **Cohen's Kappa $\kappa = 0.9204$**.
- **Hiệu năng tổng thể trên bộ Thực tế:** **Real-World F1 = `0.5702`** (Precision = **`0.4221`**, Recall = **`0.8784`**, PR-AUC = **`0.5742`**).

---

## 6. Hạn chế & Nguyên tắc Đạo đức (Ethics & Limitations)
1. **Ẩn danh & Che từ nhạy cảm:** Mọi tài khoản cá nhân được ẩn thành `@user`; toàn bộ từ tục trong biểu đồ, bảng biểu, notebook và báo cáo đều được che tự động qua `src/utils.py::mask_profanity()`.
2. **Thiên lệch nhóm (Subgroup Bias) & Ngữ cảnh mỉa mai:** Mô hình dựa trên tần suất n-gram có tỷ lệ báo động giả (FPR) cao hơn trên các câu bình thường chứa từ nhận diện chủng tộc/tôn giáo/giới tính và khó bắt các câu mỉa mai sâu không dùng từ khóa đặc trưng.
3. **Khuyến nghị vận hành:** Công cụ chỉ dùng để **hỗ trợ sàng lọc sơ bộ (Human-in-the-loop)**, không dùng để tự động khóa tài khoản khi chưa có con người kiểm tra lại.
