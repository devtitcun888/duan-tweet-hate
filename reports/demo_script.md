# KỊCH BẢN DEMO ỨNG DỤNG WEB STREAMLIT & PHƯƠNG ÁN DỰ PHÒNG (DEMO SCRIPT — P11 / B5)

> **Thời lượng demo:** 2 – 3 phút (ngay tại Slide 13 hoặc khi giảng viên yêu cầu chạy thử trực tiếp).  
> **Lệnh khởi động trước giờ bảo vệ 10 phút:**  
> ```bash
> streamlit run app/app.py
> ```

---

## A. QUY TRÌNH DEMO 5 BƯỚC TRỰC TIẾP TRÊN ỨNG DỤNG (`http://localhost:8501`)

### Bước 1: Thử 2 câu bình thường (Normal & Counter-Speech — Kiểm tra không báo động giả)
1. **Câu mẫu 1 (Đời sống & Công nghệ — Bấm nút mẫu số 1):**
   - **Input:** `Just finished reading an amazing book on machine learning and natural language processing with my friends!`
   - **Kết quả kỳ vọng:** Nhãn `0 — BÌNH THƯỜNG / KHÔNG KỲ THỊ` (màu xanh lá), xác suất rất thấp (`< 0.05`, thấp hơn nhiều so với ngưỡng `0.315`).
   - **Lời thuyết minh:** "Đầu tiên, với một câu chia sẻ học tập thông thường, mô hình cho điểm rủi ro gần bằng 0 và phân loại chính xác là Bình thường."
2. **Câu mẫu 2 (Câu có chứa từ khóa nhạy cảm nhưng mang nghĩa phản đối kỳ thị — Bấm nút mẫu số 2):**
   - **Input:** `I am NOT racist and we must stand together against racism and sexism in our schools and workplaces.`
   - **Lời thuyết minh:** "Ở câu thứ hai, em cố tình nhập các từ nhạy cảm như `NOT racist`, `racism`, `sexist`. Nhờ việc giữ lại từ phủ định `not` và học được các cụm bigram/trigram trong ngữ cảnh, ứng dụng hiển thị rõ đóng góp từng từ ở cột bên phải để người kiểm duyệt đối chiếu."

### Bước 2: Thử 2 câu phản cảm (Racist & Sexist — Kiểm tra khả năng bắt đúng mục tiêu)
3. **Câu mẫu 3 (Phân biệt giới tính — Bấm nút mẫu số 3):**
   - **Input:** `Women do not belong in leadership or engineering roles because they are too emotional and inferior to men.`
   - **Kết quả kỳ vọng:** Nhãn `1 — PHẢN CẢM (Racist/Sexist)` (cảnh báo màu đỏ), thanh rủi ro vượt ngưỡng `0.315`, cột phải tô màu đỏ các từ đóng góp dương (`woman`, `emotional`, `inferior`, `men`).
   - **Lời thuyết minh:** "Với câu định kiến hạ thấp phụ nữ trong công việc dù không hề dùng từ chửi thề tục tĩu, mô hình vẫn phát hiện được thông qua tổ hợp n-gram và giải thích rõ các từ đóng góp dương."
4. **Câu mẫu 4 (Phân biệt chủng tộc — Bấm nút mẫu số 4):**
   - **Input:** `Send all those filthy immigrants back where they came from, white supremacy is the only way to save our race.`
   - **Kết quả kỳ vọng:** Nhãn `1 — PHẢN CẢM (Racist/Sexist)` với xác suất rất cao (`> 0.85`), làm nổi bật các cụm từ `white supremacy`, `race`, `immigrant`.

### Bước 3: Thử 1 câu mỉa mai / châm biếm (Sarcasm — Minh họa trung thực hạn chế của mô hình)
5. **Câu mẫu 5 (Câu mỉa mai — Bấm nút mẫu số 5):**
   - **Input:** `Oh yeah right, because letting women vote and run companies has worked out SO wonderfully for society lol #sarcasm`
   - **Lời thuyết minh:** "Ở bước thứ 3, em xin thử một câu mỉa mai dùng toàn từ ngữ tích cực bề mặt như `wonderfully` để châm biếm quyền bầu cử của phụ nữ. Đây chính là hạn chế điển hình của biểu diễn TF-IDF mà nhóm đã phân tích ở Chương 6 và Chương 8: mô hình dựa trên túi từ khó nhận diện sự đảo nghĩa do mỉa mai, vì vậy ở chân trang ứng dụng luôn có cảnh báo cần con người kiểm tra lại ở các trường hợp ranh giới."

### Bước 4: Demo Tab 2 — Phân loại hàng loạt từ file CSV (`real_world_labeled.csv`)
- **Thao tác:** Chuyển sang **Tab 2: Phân loại Hàng loạt (CSV)** → Bấm nút **"📋 Hoặc chạy thử ngay trên 100 mẫu Thực tế (`real_world_labeled.csv`)"** (hoặc tải file `data/real_world/to_label.csv` lên).
- **Lời thuyết minh:** "Chỉ trong chưa đầy 0.2 giây, hệ thống đã xử lý xong 100 dòng tweet, tự động che các từ tục trên bảng hiển thị, thống kê số lượng và tỉ lệ phần trăm từng nhãn, đồng thời cho phép tải xuống file CSV kết quả."

### Bước 5: Demo Tab 3 — Hồ sơ mô hình & Bảng đối sánh thực nghiệm
- **Thao tác:** Chuyển sang **Tab 3: Thống kê & Hồ sơ Mô hình**.
- **Lời thuyết minh:** "Tab 3 đọc trực tiếp từ `models/model_card.json` và `reports/final_test_metrics.json`, hiển thị minh bạch ngưỡng quyết định `0.315`, kết quả Test F1 = `0.6266`, bảng so sánh toàn bộ 11 mô hình và biểu đồ ma trận nhầm lẫn."

---

## B. PHƯƠNG ÁN DỰ PHÒNG KHI MẠNG HOẶC MÁY CHIẾU GẶP SỰ CỐ (BACKUP PLAN)

1. **Phương án dự phòng 1 (Chạy hoàn toàn Offline trên máy Local):**
   - Ứng dụng `app/app.py` nạp mô hình cục bộ `models/best_model.joblib` (chỉ vài MB), **không cần kết nối Internet** vẫn chạy `streamlit run app/app.py` mượt mà 100%.
2. **Phương án dự phòng 2 (Bộ 4 ảnh chụp màn hình độ phân giải cao đã lưu sẵn):**
   - Nếu máy của hội đồng không cài Python, mở ngay 4 ảnh chụp giao diện thực tế đã lưu trong `reports/figures/`:
     - `reports/figures/P10_app_tab1_normal.png` (Giao diện Tab 1 với câu bình thường)
     - `reports/figures/P10_app_tab1_hate.png` (Giao diện Tab 1 phát hiện câu phản cảm kèm giải thích từ đóng góp)
     - `reports/figures/P10_app_tab2_batch.png` (Giao diện Tab 2 phân loại hàng loạt CSV)
     - `reports/figures/P10_app_tab3_stats.png` (Giao diện Tab 3 thống kê & hồ sơ mô hình)
3. **Phương án dự phòng 3 (Chạy trực tiếp trong Notebook `08_realworld_test.ipynb`):**
   - Mở sẵn `notebooks/08_realworld_test.ipynb` có kết quả suy luận của `src/predict.py::Predictor` để gõ thử bất kỳ câu nào của giảng viên ngay trong ô code Python.


---

## C. KỊCH BẢN DEMO MỞ RỘNG ĐA NGÔN NGỮ TIẾNG VIỆT (V7 — 6 CÂU MẪU CHUẨN)

1. **Mẫu 1 & 2 (Tiếng Anh — Chế độ `Tự động`):**
   - Bấm nút **1. [EN] Bình thường** và **2. [EN] Phản cảm — Racist/Sexist**: Hệ thống tự động nhận diện `lang = EN`, nạp mô hình `English — Linear SVM (Calibrated)` (ngưỡng `0.315`) và hiển thị định nghĩa nhãn *Racist / Sexist Hate Speech*.
2. **Mẫu 3 & 4 (Tiếng Việt có dấu — Chế độ `Tự động`):**
   - Bấm nút **3. [VI có dấu] Bình thường** (*"Đội tuyển bóng đá Việt Nam hôm qua thi đấu rất nỗ lực..."*) $ightarrow$ Hệ thống phát hiện ký tự có dấu tiếng Việt, định tuyến sang `Tiếng Việt — Linear SVM (ViHSD bin_A)` và trả về Nhãn `0` (Xanh).
   - Bấm nút **4. [VI có dấu] Phản cảm — Thù ghét/Xúc phạm** $ightarrow$ Hệ thống trả về Nhãn `1` (Đỏ), giải thích rõ các từ ghép đã tách từ (`s***`, `ó***`, `v***`) và hiển thị rõ định nghĩa nhãn *Tiếng Việt: Thù ghét / Xúc phạm cá nhân hoặc nhóm (ViHSD)*.
3. **Mẫu 5 (Tiếng Việt không dấu — Minh họa độ bền & chế độ chọn tay):**
   - Bấm nút **5. [VI không dấu]** (*"thang oc cho nay ngu dot va mat day kinh khung cut khoi day ngay"*): Giải thích cho hội đồng thấy khi gõ không dấu, bộ nhận diện tự động có thể cảnh báo hoặc nếu chuyển nút Radio sang **`Tiếng Việt`**, chế độ chọn tay sẽ ghi đè tự động và phân loại câu này.
4. **Mẫu 6 (Câu quá ngắn / Không rõ ngôn ngữ — Kiểm tra cơ chế an toàn):**
   - Bấm nút **6. [Quá ngắn / Không rõ ngôn ngữ]** (`"ok bro"`): Ở chế độ `Tự động`, hệ thống trả về `lang = unknown` và **hiển thị cảnh báo vàng không đoán bừa**.
