# KỊCH BẢN THUYẾT TRÌNH CHI TIẾT TỪNG SLIDE (SPEAKER SCRIPT — P11 / B3)

> **Thời lượng chuẩn:** 10 – 12 phút (15 slides, trung bình 40–50 giây/slide).  
> **Quy tắc trình bày:** Không đọc nguyên văn chữ trên slide; nhìn vào giảng viên/hội đồng, chỉ tay vào biểu đồ khi nêu số liệu. Mọi con số dưới đây được đồng bộ 100% từ `reports/facts.json`.

---

## PHẦN 1: KỊCH BẢN ĐẦY ĐỦ 10–12 PHÚT

### Slide 1: Tiêu đề & Giới thiệu đề tài `[00:00 – 00:40 | 40 giây]`
- **Lời nói:** "Em xin kính chào thầy/cô và các bạn. Hôm nay nhóm em xin trình bày báo cáo đồ án môn học với đề tài: *Gắn nhãn các đoạn tweet có nội dung phân biệt chủng tộc, phân biệt giới tính (Twitter Sentimental Analysis)*. Đây là bài toán phân loại nhị phân trong khai phá văn bản thực tế. Kết quả cốt lõi của nhóm là xây dựng được quy trình học máy chống rò rỉ dữ liệu hoàn chỉnh, đạt F1 lớp dương trên tập Test là **0.6266** và PR-AUC đạt **0.6651**, kèm ứng dụng web sàng lọc trực quan."
- **Câu chuyển slide:** "Trước hết, em xin đi vào bối cảnh bài toán và 4 mục tiêu chính của đề tài ở Slide 2."

### Slide 2: Bài toán & Mục tiêu nghiên cứu `[00:40 – 01:20 | 40 giây]`
- **Lời nói:** "Trên mạng xã hội X/Twitter, các nội dung phân biệt chủng tộc và phân biệt giới tính gây tổn hại nghiêm trọng đến cộng đồng, trong khi kiểm duyệt thủ công không thể xử lý xuể hàng trăm triệu bài đăng mỗi ngày. Do đó, nhóm xây dựng mô hình phân loại nhị phân: gán nhãn `1` cho tweet phản cảm (racist/sexist) và nhãn `0` cho tweet bình thường. Nội dung báo cáo bám sát 4 câu trong đề bài: từ thu thập & phân tích SQL (Câu 1), làm sạch & cân bằng dữ liệu (Câu 2), huấn luyện & tối ưu 11 mô hình học máy và chạy thử thực tế (Câu 3), đến triển khai ứng dụng Streamlit (Câu 4)."
- **Câu chuyển slide:** "Khi bắt tay vào kiểm tra dữ liệu được giao ở Câu 1, nhóm đã phát hiện một vấn đề rất quan trọng về nhãn dữ liệu, xin mời thầy/cô xem ở Slide 3."

### Slide 3: Nguồn dữ liệu & Phát hiện về nhãn bộ được giao `[01:20 – 02:15 | 55 giây]`
- **Lời nói:** "Khi tải bộ dữ liệu được giao trong đề (`cosmos98/twitter-and-reddit-sentimental-analysis-dataset`), nhóm kiểm tra thực tế và xác nhận file `Twitter_Data.csv` có **162,980 dòng**, nhưng chỉ gồm 2 cột `clean_text` và `category` nhận giá trị `-1, 0, +1`. Nội dung thực chất là bình luận cảm xúc chính trị xoay quanh bầu cử Ấn Độ và thủ tướng Modi, hoàn toàn **không có nhãn phân biệt chủng tộc hay giới tính**. Nếu lấy đại nhãn `-1` làm nhãn phản cảm thì mô hình sẽ học sai bản chất đề tài. Vì vậy, nhóm áp dụng **Phương án A**: sử dụng bộ dữ liệu chuẩn *Tweets Hate Speech Detection* (**31,962 dòng**, có nhãn `0/1` racist/sexist) để huấn luyện mô hình; đồng thời giữ nguyên bộ `Twitter_Data` (**162,980 dòng**), `Reddit_Data` (**37,249 dòng**) và bộ *Davidson et al.* (**24,783 dòng**) để làm thống kê Câu 1 và kiểm thử độ lệch miền ở Câu 3."
- **Câu chuyển slide:** "Chi tiết các con số thống kê mô tả của hai bộ dữ liệu này được thể hiện ở Slide 4."

### Slide 4: Thống kê mô tả: Phân bố nhãn & Đặc trưng bề mặt `[02:15 – 03:05 | 50 giây]`
- **Lời nói:** "Nhìn vào biểu đồ tròn ở góc phải trên, đặc điểm lớn nhất của bộ dữ liệu Hate Speech là **mất cân bằng lớp rất mạnh**: Lớp 0 (Bình thường) chiếm tới **29,720 mẫu (92.99%)**, trong khi Lớp 1 (Phản cảm) chỉ có **2,242 mẫu (7.01%)**, tức tỉ lệ lệch **13.26:1**. Ngoài ra, dữ liệu thô có **2,435 dòng trùng nguyên bản**. Ở biểu đồ góc dưới, kiểm định thống kê Chi-square và Mann-Whitney U cho thấy sự khác biệt rất có ý nghĩa ($p < 10^{-10}$): **65.03%** tweet phản cảm có gắn thẻ `@user` (so với **46.46%** ở lớp bình thường) và **18.78%** chứa dấu hỏi `?` mang tính chất vấn, mỉa mai."
- **Câu chuyển slide:** "Để trả lời Yêu cầu 1 của đề bài là tìm ra điểm chung của các tweet phản cảm bằng SQL, nhóm đã xây dựng cơ sở dữ liệu SQLite ở Slide 5."

### Slide 5: Phân tích SQL: Điểm chung của Tweet phản cảm `[03:05 – 04:00 | 55 giây]`
- **Lời nói:** "Nhóm thiết kế cơ sở dữ liệu `db/tweets.db` gồm 5 bảng và **chỉ nạp tập Train (19,830 tweet, 165,408 token)** để tuyệt đối không nhìn trộm tập Test. Kết quả chạy 4 truy vấn SQL cho thấy: Thứ nhất, ở truy vấn `40_lexicon.sql`, có **20.59%** tweet Lớp 1 chứa từ trong từ điển xúc phạm công khai, cao gấp **8.6 lần** so với mức **2.40%** của Lớp 0. Tuy nhiên, điều đó cũng có nghĩa là **79.41%** tweet phản cảm không dùng từ tục trực tiếp trong từ điển — nếu chỉ lọc bằng từ khóa cứng sẽ bỏ sót gần 4/5 số mẫu. Thứ hai, truy vấn `20_log_odds.sql` và `30_cooccur.sql` chỉ ra các từ đặc trưng nhất tập trung vào 3 chủ đề: miệt thị chủng tộc/màu da (`racist`, `white`, `black`, `n****r`), miệt thị giới tính (`sexist`, `woman`, `b***h`, `misogyny`), và xung đột tôn giáo/ý thức hệ (`allahsoil`, `islam`, `libtard`)."
- **Câu chuyển slide:** "Từ những phát hiện trên, nhóm thiết kế quy trình làm sạch văn bản 12 bước và cân bằng dữ liệu ở Slide 6."

### Slide 6: Tiền xử lý 12 bước, Khử trùng & Cân bằng dữ liệu `[04:00 – 04:55 | 55 giây]`
- **Lời nói:** "Trong hàm `clean_text()`, điểm kỹ thuật quan trọng nhất của nhóm là **loại bỏ các từ phủ định (`no, not, never`) và các đại từ/nhóm người (`you, she, they, women, men`) khỏi danh sách stopwords của NLTK**, vì trong bài toán hate speech, việc xóa chữ `not` hoặc chữ `women` sẽ làm đảo ngược hoàn toàn nghĩa câu. Đặc biệt, nhóm thực hiện khử trùng dựa trên văn bản đã làm sạch trước khi chia tập: loại bỏ **3,628 mẫu trùng lặp/retweet** và **2 mẫu có nhãn mâu thuẫn**, thu được **28,330 mẫu sạch duy nhất**. Khi so sánh 6 chiến lược cân bằng dữ liệu bằng 5-fold CV trên tập Train, nếu không cân bằng (`none`), Recall chỉ đạt **13.85%** (F1 = **0.2390**). Khi dùng `RandomOverSampler (ros)` hoặc `class_weight='balanced'`, F1 tăng vọt lên **0.5876** và **0.5742**."
- **Câu chuyển slide:** "Thiết kế chia tập chống rò rỉ và lý do chọn thước đo đánh giá được trình bày ở Slide 7."

### Slide 7: Thiết kế thực nghiệm & Tiêu chí đánh giá `[04:55 – 05:40 | 45 giây]`
- **Lời nói:** "Bộ dữ liệu sạch **28,330 mẫu** được chia Stratified một lần duy nhất theo tỉ lệ 70/15/15: Tập Train có **19,830 mẫu**, Tập Validation có **4,250 mẫu**, và Tập Test có **4,250 mẫu**, tất cả đều giữ đúng tỉ lệ lớp dương **6.52%**. Nhóm kiểm soát chống rò rỉ dữ liệu bằng `Pipeline` và kiểm chứng bằng unit test tự động: giao nhau giữa 3 tập bằng rỗng. Về tiêu chí đánh giá, nhóm chọn **F1-score lớp dương (`f1_pos`)** và **PR-AUC** làm thước đo chính thay vì Accuracy, vì dữ liệu lệch 93.5% lớp 0."
- **Câu chuyển slide:** "Trên nền tảng thực nghiệm đó, nhóm tiến hành huấn luyện 11 mô hình ở Slide 8."

### Slide 8: Hệ thống 11 Mô hình thực nghiệm `[05:40 – 06:20 | 40 giây]`
- **Lời nói:** "Nhóm thực nghiệm đầy đủ 9 mô hình cổ điển trải rộng trên 4 họ thuật toán: (1) Tuyến tính & Xác suất gồm Logistic Regression, Calibrated Linear SVM, Complement Naive Bayes; (2) Phi tuyến & Khoảng cách gồm RBF SVM và Cosine KNN; (3) Cây quyết định & Rừng ngẫu nhiên; (4) Gradient Boosting gồm XGBoost và LightGBM. Ngoài ra, để đạt điểm cộng mở rộng, nhóm cài đặt thêm 2 mô hình Học sâu bằng PyTorch và HuggingFace là mạng hồi quy hai chiều **BiLSTM** và mô hình ngôn ngữ **DistilBERT fine-tuned**."
- **Câu chuyển slide:** "Kết quả đọ sức giữa toàn bộ các mô hình này trên tập Validation được tổng hợp ở Slide 9."

### Slide 9: Kết quả so sánh các mô hình trên Tập Validation `[06:20 – 07:15 | 55 giây]`
- **Lời nói:** "Nhìn vào biểu đồ so sánh ở bên phải, chúng ta thấy một quy luật rất rõ: nhóm mô hình tuyến tính (`Logistic Regression`, `Linear SVM`, `Complement NB`) vượt trội hoàn toàn so với nhóm cây quyết định (`Random Forest` F1 = 0.5195, `Decision Tree` F1 = 0.3836) và `KNN` (F1 = 0.3902). Lý do là với ma trận TF-IDF 50,000 chiều cực kỳ thưa, siêu phẳng tuyến tính kết hợp được trọng số của hàng nghìn n-gram cùng lúc, trong khi cây quyết định chỉ rẽ nhánh từng từ đơn lẻ nên bị phân mảnh dữ liệu. Khi kết hợp thêm đặc trưng ký tự (`word+char`), Logistic Regression đạt F1 = **0.6486** ngay trước khi tuning. Với hai mô hình điểm cộng, BiLSTM đạt Val F1 = **0.5973** và DistilBERT đạt **0.4434**."
- **Câu chuyển slide:** "Tiếp theo, nhóm đưa 3 mô hình cổ điển dẫn đầu vào bước tinh chỉnh siêu tham số và tối ưu ngưỡng quyết định ở Slide 10."

### Slide 10: Tinh chỉnh siêu tham số (P6) & Tối ưu ngưỡng quyết định `[07:15 – 08:05 | 50 giây]`
- **Lời nói:** "Nhóm sử dụng `RandomizedSearchCV` với 40 tổ hợp tham số và 5-fold CV trên tập Train để tối ưu đồng thời cả tham số của bộ trích xuất TF-IDF (`ngram_range`, `min_df`, `max_features`) lẫn tham số mô hình. Sau đó, trên tập Validation, nhóm quét ngưỡng xác suất từ 0.05 đến 0.95. Biểu đồ góc trên bên phải minh họa rất rõ: với **Linear SVM (Calibrated)** sử dụng `ngram_range=(1,3)`, `min_df=2`, `C=0.50191`, tại ngưỡng mặc định 0.50 mô hình có Precision cao nhưng Recall chỉ đạt 50.54% (F1 = 0.5996). Khi tối ưu ngưỡng quyết định về **`threshold = 0.315`**, Recall tăng lên **62.82%**, đưa Validation F1 đạt đỉnh **0.6566** — tăng **+8.39 điểm %** so với baseline ban đầu. Do đó, `Linear SVM (Calibrated)` được chọn làm mô hình chốt `best_model`."
- **Câu chuyển slide:** "Sau khi khóa mô hình chốt, nhóm thực hiện đánh giá một lần duy nhất trên tập Test và chạy thử thực tế ở Slide 11."

### Slide 11: Đánh giá cuối trên Tập Test (P8) & Chạy thử thực tế (P9) `[08:05 – 09:00 | 55 giây]`
- **Lời nói:** "Trên tập Test độc lập gồm **4,250 mẫu**, mô hình chốt đạt **F1 = 0.6266** với khoảng tin cậy 95% Bootstrap là **[0.5764, 0.6716]**, **Precision = 0.6324**, **Recall = 0.6209**, **PR-AUC = 0.6651** và **ROC-AUC = 0.9345**. Tiếp theo, để kiểm thử thực tế theo Yêu cầu 4, nhóm xây dựng bộ dữ liệu **300 mẫu** từ 4 nguồn: 100 mẫu từ tập test chưa gán nhãn, 100 mẫu từ `Twitter_Data`, 50 mẫu từ `Reddit_Data`, và 50 câu hiểm hóc tự biên soạn. Hai người gắn nhãn độc lập đạt độ đồng thuận **97.00%**, hệ số **Cohen's Kappa $\kappa = 0.9204$**. Khi chạy mô hình trên 300 mẫu thực tế này, F1 tổng thể đạt **0.5702** (PR-AUC = **0.5742**). Đặc biệt, biểu đồ góc dưới bên phải chứng minh trên 162,969 dòng `Twitter_Data.csv`, chỉ một phần nhỏ các câu `sentiment = -1` bị dự đoán là phản cảm, khẳng định thực nghiệm rằng cảm xúc tiêu cực chính trị hoàn toàn khác với phân biệt chủng tộc/giới tính."
- **Câu chuyển slide:** "Để hiểu sâu hơn mô hình sai ở đâu và có thiên lệch hay không, nhóm thực hiện phân tích lỗi ở Slide 12."

### Slide 12: Phân tích lỗi (FP/FN), Giải thích trọng số & Thiên lệch nhóm `[09:00 – 09:50 | 50 giây]`
- **Lời nói:** "Nhóm đã trích xuất các lỗi Báo động giả (FP = 100) và Bỏ sót (FN = 105) trên tập Test để phân loại nguyên nhân. Biểu đồ góc trên bên phải cho thấy: lỗi Bỏ sót (FN) chủ yếu rơi vào các câu **kỳ thị ngầm, định kiến xã hội hoặc châm biếm mỉa mai** mà không dùng từ tục trực tiếp; trong khi lỗi Báo động giả (FP) thường xảy ra khi câu bình thường hoặc câu **lên án nạn phân biệt chủng tộc (counter-speech)** có chứa các từ nhạy cảm như `women`, `black`, `racist`. Phân tích công bằng mô hình (Subgroup Bias) ở góc dưới cũng chỉ ra tỉ lệ báo động giả trên các câu bình thường có từ chỉ nhóm chủng tộc/giới tính cao hơn mức trung bình toàn tập Test — đây là điểm cần con người phối hợp kiểm duyệt."
- **Câu chuyển slide:** "Để hỗ trợ con người sàng lọc nhanh và minh bạch, nhóm đã đóng gói mô hình thành ứng dụng Web Streamlit ở Slide 13."

### Slide 13: Demo Ứng dụng Web Streamlit `[09:50 – 10:35 | 45 giây]`
- **Lời nói:** "Ứng dụng Streamlit của nhóm gồm 3 tab bằng tiếng Việt: Tab 1 cho phép kiểm tra từng đoạn tweet, hiển thị nhãn cảnh báo đỏ/xanh, thanh mức độ rủi ro so với ngưỡng `0.315` và đặc biệt là giải thích trực tiếp Top-8 từ đóng góp (`TF-IDF × Hệ số`) giúp người kiểm duyệt biết ngay vì sao mô hình đưa ra phán đoán. Tab 2 cho phép tải lên file CSV tới 20,000 dòng để gắn nhãn hàng loạt chỉ trong vài giây và xuất file kết quả. Tab 3 cung cấp toàn bộ hồ sơ mô hình `model_card.json` và bảng đối sánh thực nghiệm."
- **Câu chuyển slide:** "Từ toàn bộ kết quả trên, em xin rút ra kết luận và hướng phát triển ở Slide 14."

### Slide 14: Kết luận, Hạn chế & Hướng phát triển `[10:35 – 11:10 | 35 giây]`
- **Lời nói:** "Tóm lại, đồ án đã giải quyết trọn vẹn bài toán phân loại tweet phân biệt chủng tộc/giới tính: từ việc phát hiện đúng bản chất nhãn của bộ dữ liệu được giao, phân tích SQL tìm quy luật từ vựng, chống rò rỉ dữ liệu nghiêm ngặt, so sánh 11 mô hình để đạt Test F1 = **0.6266**, đến kiểm thử thực tế 300 mẫu đạt Kappa = **0.9204** và triển khai ứng dụng Streamlit. Trong tương lai, nhóm đề xuất mở rộng lên bài toán phân loại đa nhãn (tách riêng Racist, Sexist, Offensive) và kết hợp kiến trúc RoBERTa/HateBERT cùng dữ liệu *counter-speech* để giảm thiểu thiên lệch từ khóa."
- **Câu chuyển slide:** "Em xin kết thúc phần trình bày tại Slide 15."

### Slide 15: Cảm ơn & Hỏi đáp (Q&A) `[11:10 – 11:30 | 20 giây]`
- **Lời nói:** "Em xin chân thành cảm ơn thầy/cô và các bạn đã lắng nghe. Trên màn hình là bảng tóm tắt các con số cốt lõi của dự án, nhóm em rất mong nhận được các câu hỏi và góp ý từ thầy/cô ạ!"

---

## PHẦN 2: PHIÊN BẢN RÚT GỌN 5 PHÚT (DÙNG KHI GIẢNG VIÊN YÊU CẦU TRÌNH BÀY NHANH)

1. **Phút 1 (Slide 1–3 — Bài toán & Dữ liệu):** Nêu mục tiêu phân loại nhị phân tweet phân biệt chủng tộc/giới tính. Chỉ ra phát hiện quan trọng: `Twitter_Data.csv` (162,980 dòng) chỉ có nhãn sentiment chính trị `-1/0/+1`, nên nhóm dùng bộ `hate_train.csv` (31,962 dòng, tỉ lệ lớp 1 = **7.01%**, lệch **13.26:1**) để huấn luyện và dùng `Twitter_Data` để phân tích + kiểm thử lệch miền.
2. **Phút 2 (Slide 4–6 — SQL & Tiền xử lý chống rò rỉ):** Nêu kết quả SQL trên tập Train: **20.59%** tweet phản cảm chứa từ trong Lexicon (gấp **8.6 lần** lớp 0), nhưng **79.41%** không dùng từ tục trực tiếp. Pipeline làm sạch 12 bước giữ lại từ phủ định/đại từ, khử trùng **3,628 mẫu retweet** trước khi chia tập 70/15/15 (28,330 mẫu sạch), và chọn `RandomOverSampler` qua 5-fold CV (F1 = **0.5876**).
3. **Phút 3 (Slide 8–10 — So sánh 11 Mô hình & Tinh chỉnh):** So sánh 9 mô hình cổ điển + BiLSTM & DistilBERT bằng F1 lớp dương và PR-AUC. Các mô hình tuyến tính vượt trội so với cây quyết định trên TF-IDF thưa. Sau `RandomizedSearchCV` và tối ưu ngưỡng trên Val (`threshold = 0.315`), mô hình chốt **Linear SVM (Calibrated)** tăng Val F1 từ **0.5727** lên **0.6566** (+8.39 điểm %).
4. **Phút 4 (Slide 11–12 — Kết quả Test, Thực tế & Phân tích lỗi):** Chạy Test độc lập (4,250 mẫu) đúng 1 lần đạt **F1 = 0.6266** (PR-AUC = **0.6651**). Chạy thử bộ thực tế 300 mẫu (4 nguồn A/B/C/D, Cohen's Kappa $\kappa = 0.9204$) đạt **F1 = 0.5702**. Chỉ rõ nguyên nhân lỗi FP (từ nhạy cảm trong câu phản đối kỳ thị) và FN (kỳ thị ngầm/mỉa mai).
5. **Phút 5 (Slide 13–15 — Demo App & Kết luận):** Demo nhanh 3 tab ứng dụng Streamlit (phân loại 1 câu kèm giải thích từ đóng góp, phân loại file CSV hàng loạt, bảng thống kê) và kết luận.
