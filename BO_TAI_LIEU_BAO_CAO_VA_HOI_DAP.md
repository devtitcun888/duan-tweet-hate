# BỘ TÀI LIỆU BÁO CÁO & HỎI ĐÁP (bổ sung cho PLAN_Tweet_Hate_Classification.md)

> Dùng sau khi xong GĐ 5-8 (đã có số liệu thật). Giao cho agent làm từng bước ở Phần B. **Quy tắc vàng:** mọi con số trong tài liệu phải lấy từ file kết quả thật (`reports/model_comparison.csv`, `reports/final_test_metrics.json`...). Agent không được tự bịa số; chỗ chưa có ghi `[CHƯA CÓ SỐ]`.

---

## A. DANH SÁCH TÀI LIỆU CẦN CÓ

| # | Tài liệu | File | Mục đích |
|---|---|---|---|
| 1 | Báo cáo đầy đủ | `reports/report.docx` (+ PDF) | Nộp giáo viên |
| 2 | Slide thuyết trình 12-15 trang | `reports/slides.pptx` | Trình bày 10-12 phút |
| 3 | Kịch bản nói (script) từng slide | `reports/speaker_script.md` | Học thuộc ý, tránh đọc slide |
| 4 | Ngân hàng câu hỏi - trả lời (Q&A) | `reports/qa_bank.md` | Chuẩn bị giáo viên hỏi |
| 5 | Kịch bản demo app + phương án dự phòng | `reports/demo_script.md` | Demo không bị "đứng hình" |
| 6 | Bảng "số liệu chốt" 1 trang (cheat sheet) | `reports/cheat_sheet.md` | Tra nhanh số liệu khi bị hỏi |
| 7 | README + hướng dẫn chạy lại | `README.md` | Giáo viên tự chạy được |

---

## B. PROMPT CHO AGENT (làm theo thứ tự)

**B0 – Gom số liệu (bắt buộc làm trước)**
> Đọc toàn bộ notebook và file kết quả trong dự án. Tạo `reports/facts.json` gồm: kích thước từng bộ dữ liệu, phân bố nhãn, tỉ lệ lớp, các bước làm sạch, kỹ thuật cân bằng đã chọn và lý do, danh sách mô hình + metric (Precision, Recall, F1, ROC-AUC, PR-AUC, thời gian), tham số tốt nhất sau tuning, kết quả test cuối, kết quả bộ thực tế, kappa, top từ đặc trưng, 20 ví dụ dự đoán sai. Nếu thiếu mục nào, liệt kê rõ là thiếu, không tự điền.

**B1 – Báo cáo**
> Dựa **chỉ** trên `facts.json` và các figure trong `reports/figures/`, viết báo cáo tiếng Việt theo cấu trúc ở PLAN mục 9 (Giới thiệu → Dữ liệu → Tiền xử lý → Mô hình → Kết quả → Thử thực tế → Ứng dụng → Kết luận). Mỗi chương có: mục tiêu, phương pháp, kết quả, nhận xét. Mỗi bảng/biểu đồ có số thứ tự, chú thích, và được nhắc đến trong văn bản. Có mục "Hạn chế & đạo đức" (nhãn bộ được giao, mất cân bằng, domain shift, thiên lệch, che từ tục). Xuất docx dùng skill docx, mục lục tự động, phông Times New Roman 13, cách dòng 1.5, 25-35 trang.

**B2 – Slide**
> Tạo `slides.pptx` 12-15 trang theo bố cục ở mục C. Mỗi slide tối đa 5 dòng chữ, ưu tiên biểu đồ; con số lấy từ `facts.json`. Che từ tục. Ghi chú nói (speaker notes) bằng ý chính ở mỗi slide.

**B3 – Kịch bản nói**
> Viết `speaker_script.md`: lời nói tự nhiên, mỗi slide 30-60 giây, tổng 10-12 phút, kèm câu chuyển slide và dấu thời gian cộng dồn. Cuối file có phiên bản rút gọn 5 phút.

**B4 – Ngân hàng Q&A**
> Dựa trên mục D (khung câu hỏi) + `facts.json`, viết `qa_bank.md`: mỗi câu hỏi có (1) câu trả lời ngắn 2-3 câu để nói, (2) ý mở rộng nếu bị hỏi tiếp, (3) số liệu/hình minh chứng chỉ rõ ở slide/trang nào, (4) "bẫy" thường gặp. Thêm ≥15 câu riêng cho dự án này dựa trên điểm yếu thật thấy trong kết quả (vd. mô hình sai ở đâu, recall còn thấp...). Câu nào cần số liệu mà thiếu thì ghi `[CHƯA CÓ SỐ]`.

**B5 – Demo script & cheat sheet**
> Viết `demo_script.md` (5 bước demo, câu nhập mẫu: 2 câu bình thường, 2 câu phản cảm đã che, 1 câu mỉa mai để giải thích hạn chế) và phương án dự phòng (video quay màn hình + ảnh chụp nếu mạng/Colab lỗi). Viết `cheat_sheet.md` 1 trang: bảng số liệu chốt, tên mô hình, tham số tốt nhất, link Colab.

**B6 – Kiểm tra chéo**
> Đối chiếu từng con số trong report, slides, script, qa_bank với `facts.json`; liệt kê mọi chỗ không khớp hoặc không có nguồn, sửa lại. Xuất `reports/consistency_check.md`.

---

## C. BỐ CỤC SLIDE (12-15 trang)

1. Tiêu đề + nhóm/tên + giảng viên
2. Bài toán & mục tiêu (vì sao cần phát hiện tweet phản cảm)
3. Dữ liệu: nguồn, kích thước, vấn đề nhãn của bộ được giao, hướng xử lý
4. Thống kê mô tả (phân bố nhãn, độ dài)
5. Điểm chung của tweet phản cảm (từ khóa SQL, WordCloud đã che)
6. Tiền xử lý & cân bằng dữ liệu
7. Thiết kế thực nghiệm (chia tập, metric, vì sao F1/PR-AUC)
8. Các mô hình đã thử
9. Bảng/biểu đồ so sánh kết quả
10. Tinh chỉnh tham số & mô hình được chọn
11. Chạy thử thực tế (bộ tự gắn nhãn, kappa, domain shift)
12. Phân tích lỗi (ví dụ đúng/sai)
13. Demo ứng dụng
14. Kết luận, hạn chế, hướng phát triển
15. Cảm ơn + Q&A

---

## D. KHUNG NGÂN HÀNG CÂU HỎI (agent điền số, bạn học ý)

### D1. Về đề tài & dữ liệu
1. Vì sao chọn bài toán nhị phân? Nhãn 0/1 nghĩa là gì?
2. Bộ dữ liệu được giao có những cột nào? Có nhãn phân biệt chủng tộc/giới tính không? *(Trả lời trung thực: bộ được giao là sentiment -1/0/1; em dùng thêm bộ hate-speech có nhãn để huấn luyện, và dùng bộ được giao cho phân tích/kiểm tra chéo.)*
3. Dữ liệu có bao nhiêu dòng, tỉ lệ lớp bao nhiêu? Có NaN/trùng không, xử lý ra sao?
4. Nhãn tiêu cực (sentiment) khác phân biệt chủng tộc/giới tính thế nào?
5. Dữ liệu có thiên lệch (bias) không? (thời gian, chủ đề, nhóm đối tượng)
6. Tweet phản cảm có điểm chung gì? Bằng chứng ở đâu? (SQL/tần suất từ)
7. Vì sao dùng SQL? Query nào quan trọng nhất?

### D2. Tiền xử lý & đặc trưng
8. Em làm sạch những gì? Vì sao giữ lại/bỏ stopword?
9. Vì sao giữ từ phủ định, đại từ?
10. TF-IDF là gì, khác Bag-of-Words ở đâu? Vì sao dùng bigram?
11. Vì sao chọn `max_features`, `min_df` như vậy?
12. Lemmatization khác stemming thế nào?
13. Emoji, hashtag, mention xử lý ra sao?

### D3. Mất cân bằng & đánh giá
14. Vì sao accuracy không đủ? Cho ví dụ (đoán toàn lớp 0 vẫn ~93%).
15. Precision/Recall/F1 là gì? Trong bài này ưu tiên cái nào, vì sao?
16. ROC-AUC và PR-AUC khác nhau? Vì sao PR-AUC hợp dữ liệu lệch?
17. SMOTE/oversampling hoạt động thế nào? Vì sao chỉ áp dụng lên tập train?
18. Data leakage là gì, em tránh bằng cách nào?
19. Em chia tập thế nào? Vì sao `stratify`? Cross-validation để làm gì?
20. Ngưỡng 0.5 có phải tối ưu? Em chỉnh ngưỡng ra sao?

### D4. Mô hình
21. Vì sao thử các mô hình này? Ưu/nhược từng mô hình (LR, NB, SVM, RF, KNN, XGBoost)?
22. Naive Bayes giả định gì? Vì sao vẫn hoạt động tốt với văn bản?
23. SVM tuyến tính vs kernel với văn bản?
24. Random Forest vì sao thường kém hơn LR/SVM trên TF-IDF thưa?
25. Mô hình nào tốt nhất? Chênh lệch có ý nghĩa thống kê không? (nhắc CV, độ lệch chuẩn)
26. Vì sao chọn mô hình này làm mô hình cuối (hiệu năng, tốc độ, khả năng giải thích)?
27. Mô hình điểm cộng (BERT/LSTM): cấu hình, kết quả, đáng đánh đổi thời gian không?
28. Overfitting là gì? Em kiểm tra thế nào (train vs val)?
29. Tham số nào ảnh hưởng nhiều nhất? Dùng Grid hay Random search, vì sao?

### D5. Thực tế & hạn chế
30. Bộ tweet thực tế lấy ở đâu, gắn nhãn thế nào? Kappa bao nhiêu?
31. Vì sao hiệu năng thực tế thấp hơn trên test? (domain shift)
32. Mô hình sai ở loại tweet nào? (mỉa mai, từ lóng, ngữ cảnh, từ ngoài từ điển)
33. Có tweet bình thường chứa từ tục bị đoán nhầm không? Vì sao?
34. Mô hình có phân biệt được racist với sexist không? (hiện chỉ nhị phân) Hướng mở rộng đa lớp?
35. Có thiên vị với phương ngữ/nhóm người (như AAE) không? (rủi ro bias của mô hình hate-speech)
36. Dùng thật có nguy cơ gì (chặn nhầm, kiểm duyệt quá mức)? Vai trò con người?

### D6. Ứng dụng & triển khai
37. App dùng công nghệ gì, luồng xử lý thế nào? Mô hình được nạp ra sao?
38. Làm sao cải thiện nếu có thêm thời gian? (thêm dữ liệu, BERT, active learning, đa ngôn ngữ/tiếng Việt)
39. Đạo đức dữ liệu: cách bảo vệ thông tin người dùng, che từ tục?
40. Em đóng góp gì riêng so với hướng dẫn chung? (nêu các phần: SQL, so sánh cân bằng, tuning ngưỡng, bộ thực tế)

### D7. Câu "gài" cần chuẩn bị
- "Nếu bộ dữ liệu giao không có nhãn thì kết quả của em có đáng tin không?"
- "Con số F1 này có bị rò rỉ không? Chứng minh."
- "Em có chạy test set nhiều lần để chọn mô hình không?"
- "Tại sao không dùng ChatGPT/LLM để phân loại luôn?" *(so sánh chi phí, độ trễ, kiểm soát, tái lập; có thể nhắc đây là hướng so sánh mở rộng)*
- "Từ này ở nhãn 1 nhiều nghĩa là tweet chứa nó là phản cảm?" *(không: tương quan ≠ nhân quả, cần ngữ cảnh)*

---

## E. MẸO TRẢ LỜI

- Cấu trúc 3 bước: **Kết luận 1 câu → Lý do/ số liệu → Hạn chế/ hướng cải thiện**.
- Không biết thì nói: "Phần này em chưa thử, theo em dự đoán là… vì…, và em sẽ kiểm chứng bằng…".
- Luôn nhớ 6 con số chốt: số dòng, tỉ lệ lớp, F1 của mô hình chọn, F1 trước/sau tuning, F1 trên bộ thực tế, kappa.
- Không đọc slide; nhìn người hỏi; chỉ vào biểu đồ khi giải thích.
- Chuẩn bị sẵn: notebook mở trên tab, app đã chạy thử, video demo dự phòng, link Colab mở quyền xem.
- Tập nói 3 lần, bấm giờ; 1 lần phải có người khác hỏi thử bằng chính ngân hàng Q&A.

---

## F. CHECKLIST TRƯỚC NGÀY BÁO CÁO
- [ ] `consistency_check.md` không còn mục sai lệch
- [ ] Mọi `[CHƯA CÓ SỐ]` đã được thay bằng số thật
- [ ] Báo cáo đã xuất PDF, kiểm tra mục lục, số trang, hình không vỡ
- [ ] Slide mở tốt trên máy trình chiếu (font nhúng/ xuất PDF dự phòng)
- [ ] Colab "Run all" thành công trong phiên mới
- [ ] Demo app chạy offline được hoặc có video dự phòng
- [ ] Đã tập nói + tập trả lời ≥ 20 câu trong `qa_bank.md`
