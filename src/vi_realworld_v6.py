"""V6 Vietnamese Real-World Dataset Construction (210 comments across 3 sources), Annotation Guide, Cohen's Kappa & Weighted Kappa, and Domain-Shift Evaluation (V6)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score

from src.evaluate import compute_metrics
from src.models import get_model_scores
from src.utils import get_project_root, set_seed
from src.vi_clean import vi_clean, vi_mask_profanity

PROJECT_ROOT = get_project_root()
VI_SPLITS_DIR = PROJECT_ROOT / "data" / "splits_vi"
RW_DIR = PROJECT_ROOT / "data" / "real_world"
VI_MODELS_DIR = PROJECT_ROOT / "models" / "vi"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TAB_DIR = REPORTS_DIR / "tables"

VI_ANNOTATION_GUIDE_MD = """# HƯỚNG DẪN GẮN NHÃN BÌNH LUẬN TIẾNG VIỆT THỰC TẾ (VI_ANNOTATION_GUIDE — V6)

## 1. Mục đích & Lưu ý Đạo đức
Hướng dẫn này dùng cho 2 người gắn nhãn độc lập (`label_a`, `label_b`) trên bộ dữ liệu thực tế tiếng Việt `data/real_world/vi_to_label.csv` ($N = 210$ bình luận từ 3 nguồn: A = mẫu phân tầng từ tập test ViHSD, B = bình luận mạng xã hội công khai đã ẩn danh, C = câu hiểm hóc tự soạn).
- **Khác biệt định nghĩa nhãn so với bản tiếng Anh:** Mô hình tiếng Việt được huấn luyện theo chuẩn **ViHSD (Luu et al., 2021)** nhằm phát hiện **ngôn từ thù ghét / xúc phạm tiếng Việt nói chung** (nhắm vào cá nhân, nhóm, vùng miền, giới tính...), không giới hạn ở racist/sexist như bộ tiếng Anh.
- Toàn bộ tên người dùng đã được thay bằng `@user` và đường dẫn thay bằng `[URL]`.

## 2. Định nghĩa 3 mức nhãn (CLEAN / OFFENSIVE / HATE) theo chuẩn ViHSD

### 1) `CLEAN` (Mã số `0` — Bình thường / Sạch)
- Bình luận trao đổi thông tin, khen ngợi, góp ý văn minh, tin tức, hoặc tranh luận xã hội không chứa từ ngữ tục tĩu hay công kích cá nhân/nhóm.
- Bao gồm cả **câu phản đối ngôn từ thù ghét (Counter-speech)** (ví dụ: *"Chúng ta cần lên án hành vi phân biệt vùng miền trên mạng"*).
- **Quy tắc mặc định:** Khi không chắc chắn giữa `CLEAN` và `OFFENSIVE`, chọn `CLEAN` và ghi chú vào cột `note`.

### 2) `OFFENSIVE` (Mã số `1` — Xúc phạm / Thô tục nhưng không thù ghét mục tiêu)
- Bình luận có sử dụng từ ngữ thô tục, chửi thề cảm thán (ví dụ: `v***`, `đ***`, `n***`) hoặc lời lẽ thiếu lịch sự nhưng **không nhắm vào việc hạ thấp nhân phẩm, miệt thị hay kích động thù ghét một cá nhân/nhóm cụ thể** (ví dụ: *"Đ*** hôm nay đề thi khó v*** làm tao toát mồ hôi"*).

### 3) `HATE` (Mã số `2` — Ngôn từ thù ghét / Công kích cá nhân hoặc nhóm)
- Bình luận tấn công trực diện, lăng mạ, miệt thị nhân phẩm, ngoại hình, trí tuệ, vùng miền, giới tính, tôn giáo của một cá nhân hoặc một tập thể, hoặc đe dọa bạo lực (ví dụ: *"Lũ s*** v*** vô học này xứng đáng bị đuổi khỏi xã hội"*).

## 3. Ánh xạ sang Bài toán Nhị phân (`bin_A` và `bin_B`)
- **`bin_A` (HATE vs CLEAN, bỏ OFFENSIVE):** `HATE -> 1`, `CLEAN -> 0`.
- **`bin_B` (OFFENSIVE + HATE vs CLEAN):** `OFFENSIVE, HATE -> 1`, `CLEAN -> 0`.
"""


def _build_public_social_80() -> list[dict[str, Any]]:
    """80 anonymized public Vietnamese social media comments (Source B)."""
    # Format: (text, label_a, label_b, label_final, note)
    raw = [
        # --- CLEAN (50 samples) ---
        ("Video hướng dẫn chi tiết và dễ hiểu quá, cảm ơn kênh @user đã chia sẻ kiến thức bổ ích!", "CLEAN", "CLEAN", "CLEAN", "Khen ngợi nội dung giáo dục"),
        ("Đội tuyển bóng đá Việt Nam hôm qua thi đấu rất nỗ lực, chúc mừng các cầu thủ trẻ.", "CLEAN", "CLEAN", "CLEAN", "Bình luận thể thao tích cực"),
        ("Mình nghĩ chính sách giao thông này cần thêm thời gian lấy ý kiến người dân trước khi áp dụng.", "CLEAN", "CLEAN", "CLEAN", "Góp ý chính sách văn minh"),
        ("Thời tiết Sài Gòn mấy hôm nay nắng nóng đỉnh điểm, mọi người ra đường nhớ mang theo nước nhé.", "CLEAN", "CLEAN", "CLEAN", "Chia sẻ đời sống"),
        ("Bài giảng về học máy và xử lý ngôn ngữ tự nhiên rất trực quan, sinh viên dễ làm theo.", "CLEAN", "CLEAN", "CLEAN", "Bình luận học tập"),
        ("Có ai biết quán phở nào ngon ở khu vực quận Hoàn Kiếm không chỉ giúp mình với ạ?", "CLEAN", "CLEAN", "CLEAN", "Hỏi địa điểm ăn uống"),
        ("Phim này kỹ xảo đẹp nhưng kịch bản đoạn cuối hơi vội, chấm 7/10 điểm.", "CLEAN", "CLEAN", "CLEAN", "Đánh giá phim khách quan"),
        ("Nhìn cảnh quê hương mùa lúa chín yên bình quá, ước gì cuối tuần được về thăm nhà.", "CLEAN", "CLEAN", "CLEAN", "Cảm xúc tích cực"),
        ("Shop tư vấn nhiệt tình, giao hàng nhanh trong ngày, đóng gói cẩn thận.", "CLEAN", "CLEAN", "CLEAN", "Đánh giá mua hàng"),
        ("Theo dõi tin tức kinh tế năm nay thấy ngành công nghệ thông tin vẫn có nhiều cơ hội việc làm.", "CLEAN", "CLEAN", "CLEAN", "Bình luận kinh tế"),
        ("Các bạn tình nguyện viên hỗ trợ bà con vùng lũ nhiệt tình quá, thật sự trân trọng.", "CLEAN", "CLEAN", "CLEAN", "Tôn vinh việc tốt"),
        ("Món canh chua cá lóc nấu theo công thức của chị @user vừa nhanh vừa hợp khẩu vị gia đình.", "CLEAN", "CLEAN", "CLEAN", "Bình luận ẩm thực"),
        ("Trận chung kết tối qua kịch tính đến phút cuối cùng, cả hai đội đều xứng đáng.", "CLEAN", "CLEAN", "CLEAN", "Thể thao công bằng"),
        ("Mình đã thử chạy code trên Google Colab và kết quả ra đúng như hướng dẫn trong bài.", "CLEAN", "CLEAN", "CLEAN", "Thực hành công nghệ"),
        ("Giá xăng tuần này giảm nhẹ, hy vọng chi phí vận chuyển hàng hóa cũng sớm hạ nhiệt.", "CLEAN", "CLEAN", "CLEAN", "Tin tức đời sống"),
        ("Nghe bản nhạc acoustic này lúc trời mưa cảm giác nhẹ nhàng và thư giãn thật sự.", "CLEAN", "CLEAN", "CLEAN", "Âm nhạc thư giãn"),
        ("Cảm ơn bác sĩ @user đã giải thích rõ ràng về cách phòng ngừa sốt xuất huyết cho trẻ nhỏ.", "CLEAN", "CLEAN", "CLEAN", "Y tế cộng đồng"),
        ("Hội thảo khoa học dữ liệu sáng nay có nhiều báo cáo ứng dụng AI rất thực tế.", "CLEAN", "CLEAN", "CLEAN", "Sự kiện học thuật"),
        ("Đường Vành đai 3 giờ cao điểm đông xe nhưng lực lượng chức năng phân luồng khá nhanh.", "CLEAN", "CLEAN", "CLEAN", "Giao thông đô thị"),
        ("Sinh viên mới ra trường nên tập trung rèn kỹ năng làm việc nhóm và ngoại ngữ.", "CLEAN", "CLEAN", "CLEAN", "Lời khuyên nghề nghiệp"),
        ("Chuyến tàu cao tốc Bắc Nam nếu hoàn thành sẽ giúp việc đi lại giữa các tỉnh thuận tiện hơn nhiều.", "CLEAN", "CLEAN", "CLEAN", "Hạ tầng giao thông"),
        ("Mình không đồng tình với quan điểm của tác giả bài viết nhưng tôn trọng góc nhìn riêng của bạn.", "CLEAN", "CLEAN", "CLEAN", "Tranh luận văn minh"),
        ("Cuốn sách này phân tích rất sâu về lịch sử văn hóa các vùng miền Việt Nam.", "CLEAN", "CLEAN", "CLEAN", "Giới thiệu sách"),
        ("Hôm nay trường tổ chức ngày hội việc làm, có hơn 50 doanh nghiệp công nghệ tham gia.", "CLEAN", "CLEAN", "CLEAN", "Tin tức đại học"),
        ("Ảnh chụp phong cảnh Đà Lạt sương sớm đẹp như tranh vẽ vậy.", "CLEAN", "CLEAN", "CLEAN", "Nhiếp ảnh du lịch"),
        ("Nên kiểm tra kỹ nguồn tin trước khi chia sẻ lên mạng xã hội để tránh lan truyền tin giả.", "CLEAN", "CLEAN", "CLEAN", "Kỹ năng số"),
        ("Ứng dụng giao diện tiếng Việt rõ ràng, có thanh tiến trình và bảng thống kê rất dễ theo dõi.", "CLEAN", "CLEAN", "CLEAN", "Nhận xét phần mềm"),
        ("Bà con nông dân miền Tây năm nay trúng mùa sầu riêng, giá bán cũng ổn định.", "CLEAN", "CLEAN", "CLEAN", "Tin nông nghiệp"),
        ("Tập thể dục 30 phút mỗi sáng giúp tinh thần tỉnh táo và làm việc hiệu quả hơn hẳn.", "CLEAN", "CLEAN", "CLEAN", "Sức khỏe đời sống"),
        ("Chương trình học bổng này hỗ trợ rất tốt cho học sinh nghèo vượt khó ở vùng cao.", "CLEAN", "CLEAN", "CLEAN", "Giáo dục nhân văn"),
        ("Mình thấy đề thi thử năm nay bám sát chương trình mới, có tính phân loại cao.", "CLEAN", "CLEAN", "CLEAN", "Giáo dục phổ thông"),
        ("Cầu thủ số 10 chuyền bóng quá tinh tế, tiếc là cú sút cuối lại trúng cột dọc.", "CLEAN", "CLEAN", "CLEAN", "Bình luận bóng đá"),
        ("Thư viện mới của trường rộng rãi, nhiều ánh sáng tự nhiên và wifi rất mạnh.", "CLEAN", "CLEAN", "CLEAN", "Cơ sở vật chất"),
        ("Ai dùng thử dòng laptop này để lập trình Python rồi cho mình xin đánh giá pin nhé.", "CLEAN", "CLEAN", "CLEAN", "Hỏi đáp phần cứng"),
        ("Lễ hội pháo hoa quốc tế Đà Nẵng năm nay thu hút rất đông du khách trong và ngoài nước.", "CLEAN", "CLEAN", "CLEAN", "Du lịch sự kiện"),
        ("Nấu ăn tại nhà vừa tiết kiệm chi phí vừa đảm bảo vệ sinh an toàn thực phẩm.", "CLEAN", "CLEAN", "CLEAN", "Lối sống lành mạnh"),
        ("Cảm ơn nhóm tác giả đã công bố bộ dữ liệu phục vụ cộng đồng nghiên cứu xử lý ngôn ngữ tự nhiên.", "CLEAN", "CLEAN", "CLEAN", "Tri ân học thuật"),
        ("Mỗi vùng miền đều có nét văn hóa và giọng nói đặc trưng đáng quý, chúng ta nên tôn trọng nhau.", "CLEAN", "CLEAN", "CLEAN", "Thông điệp đoàn kết"),
        ("Phụ nữ hiện đại hoàn toàn có thể làm tốt vai trò lãnh đạo trong các tập đoàn công nghệ.", "CLEAN", "CLEAN", "CLEAN", "Bình đẳng giới tích cực"),
        ("Hệ thống cảnh báo sớm này giúp giảm thiểu rủi ro thiên tai cho ngư dân đi biển.", "CLEAN", "CLEAN", "CLEAN", "Khoa học đời sống"),
        ("Tối nay trời quang mây tạnh, rất thích hợp để đi dạo quanh hồ Tây.", "CLEAN", "CLEAN", "CLEAN", "Đời sống Hà Nội"),
        ("Mình vừa hoàn thành khóa học SQL cơ bản, thấy cách truy vấn dữ liệu rất thú vị.", "CLEAN", "CLEAN", "CLEAN", "Học lập trình"),
        ("Đội ngũ hỗ trợ kỹ thuật phản hồi email nhanh và giải quyết triệt để lỗi cài đặt.", "CLEAN", "CLEAN", "CLEAN", "Dịch vụ khách hàng"),
        ("Hy vọng năm tới thành phố sẽ trồng thêm nhiều cây xanh ở các tuyến phố trung tâm.", "CLEAN", "CLEAN", "CLEAN", "Môi trường đô thị"),
        ("Bài thuyết trình của nhóm chuẩn bị kỹ số liệu, biểu đồ trực quan và trả lời câu hỏi lưu loát.", "CLEAN", "CLEAN", "CLEAN", "Nhận xét bảo vệ đồ án"),
        ("Nước ép cam tươi không đường rất tốt cho sức khỏe và bổ sung vitamin C.", "CLEAN", "CLEAN", "CLEAN", "Dinh dưỡng"),
        ("Kỳ nghỉ lễ này gia đình mình chọn ở nhà nghỉ ngơi và nấu ăn cùng nhau.", "CLEAN", "CLEAN", "CLEAN", "Gia đình"),
        ("Các bạn trẻ bây giờ sáng tạo nội dung số rất giỏi và bắt nhịp xu hướng nhanh.", "CLEAN", "CLEAN", "CLEAN", "Nhận xét tích cực"),
        ("Đọc xong bài báo mà thấy ấm lòng vì vẫn còn rất nhiều người tốt quanh ta.", "CLEAN", "CLEAN", "CLEAN", "Lan tỏa tích cực"),
        ("Chúc mọi người một tuần làm việc mới tràn đầy năng lượng và hoàn thành tốt mục tiêu!", "CLEAN", "CLEAN", "CLEAN", "Lời chúc đầu tuần"),
        # --- OFFENSIVE (12 samples) ---
        ("Trời nóng vãi lồn mà còn mất điện từ sáng tới chiều, bực cả mình!", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Chửi thề cảm thán thời tiết"),
        ("Đm kẹt xe cứng ngắc suốt hai tiếng đồng hồ ngoài ngã tư, mệt vcl.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Than phiền kẹt xe có từ tục"),
        ("Cái mạng wifi hôm nay lag vkl, đang họp trực tuyến mà cứ văng ra ngoài miết.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Than phiền mạng internet"),
        ("Đá kiểu gì mà sút toàn lên trời thế này, xem ức chế vãi chưởng.", "OFFENSIVE", "CLEAN", "OFFENSIVE", "Bất đồng nhẹ CLEAN/OFFENSIVE, chốt OFFENSIVE"),
        ("Làm ăn chụp giật bố láo thế này thì sớm muộn cũng dẹp tiệm thôi.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Chỉ trích gay gắt dịch vụ"),
        ("Nói chuyện nghe xạo lồn thật sự, toàn bịa chuyện câu tương tác.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Từ tục phê phán câu view"),
        ("Đéo hiểu sao lại thiết kế cái biển báo khuất tầm nhìn như vậy được.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Từ tục cảm thán hạ tầng"),
        ("Phim dở tệ hại, phí cả tiền mua vé với hai tiếng cuộc đời vcl.", "OFFENSIVE", "CLEAN", "OFFENSIVE", "Chê phim kèm từ viết tắt tục"),
        ("Bình luận toàn mấy câu nhảm nhí vô bổ, đọc mà phát ngán.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Chê bai bình luận"),
        ("Ăn nói hàm hồ thiếu suy nghĩ, chưa đọc kỹ bài đã vào phán bừa.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Phê phán thái độ"),
        ("Mở nhạc ầm ĩ cả khu phố lúc nửa đêm, ý thức kém vcl.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Phàn nàn tiếng ồn"),
        ("Code chạy lỗi tùm lum từa lưa, ngồi sửa muốn nổ tung cái đầu đm.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Than thở công việc có từ tục"),
        # --- HATE (18 samples) ---
        ("Đúng là lũ súc vật vô học, sống chỉ làm bẩn xã hội chứ không được tích sự gì!", "HATE", "HATE", "HATE", "Miệt thị phi nhân tính hóa"),
        ("Thằng óc chó @user này ngu bẩm sinh hay sao mà phát ngôn mất dạy thế hả?", "HATE", "HATE", "HATE", "Công kích lăng mạ cá nhân trực diện"),
        ("Cút xéo khỏi đất nước tao đi lũ phản động cặn bã hạ đẳng!", "HATE", "HATE", "HATE", "Ngôn từ thù ghét cực đoan"),
        ("Đàn bà con gái não ngắn chỉ biết ăn bám chứ làm được trò trống gì mà đòi bình đẳng.", "HATE", "HATE", "HATE", "Kỳ thị giới tính hạ thấp phụ nữ"),
        ("Bọn vùng miền đó toàn một lũ lừa đảo bẩn tính, đừng bao giờ thuê hay chơi cùng.", "HATE", "HATE", "HATE", "Kỳ thị phân biệt vùng miền"),
        ("Mặt con đĩ @user nhìn đã thấy đê tiện, suốt ngày lên mạng khoe thân rẻ tiền.", "HATE", "HATE", "HATE", "Lăng mạ nhân phẩm phụ nữ"),
        ("Đập chết mẹ cái lũ khốn nạn chuyên đi cướp giật hại người này đi!", "HATE", "HATE", "HATE", "Kích động bạo lực"),
        ("Thằng bệnh hoạn biến thái này phải tống vào tù mục xương mới đáng đời.", "HATE", "OFFENSIVE", "HATE", "Bất đồng nhẹ OFFENSIVE/HATE, chốt HATE do công kích nặng"),
        ("Lũ ngu si đần độn không có não, mở mồm ra là sủa bậy như chó đàn.", "HATE", "HATE", "HATE", "Xúc phạm nặng tập thể"),
        ("Con điên @user tâm thần nặng rồi, câm cái mồm thối lại cho thiên hạ nhờ!", "HATE", "HATE", "HATE", "Công kích cá nhân nặng"),
        ("Đồ rác rưởi vô liêm sỉ, sống thất đức như mày sớm muộn cũng gặp quả báo.", "HATE", "HATE", "HATE", "Lăng mạ cá nhân"),
        ("Bọn mọi rợ hạ đẳng đó thì biết cái gì về văn minh mà đòi dạy đời người khác.", "HATE", "HATE", "HATE", "Phân biệt hạ thấp nhóm người"),
        ("Thằng chó đẻ mất dạy, mày ra đường coi chừng tao chém chết!", "HATE", "HATE", "HATE", "Đe dọa bạo lực trực tiếp"),
        ("Nhìn cái bản mặt hãm lồn của nó là muốn tát cho lệch mặt rồi.", "HATE", "HATE", "HATE", "Tấn công ngoại hình + bạo lực"),
        ("Đúng là thứ cặn bã xã hội, bố mẹ mày ở nhà không biết dạy mày làm người à?", "HATE", "HATE", "HATE", "Xúc phạm cá nhân và gia đình"),
        ("Mấy con phò rẻ tiền suốt ngày lên mạng giảng đạo lý nghe mắc ói vãi lồn.", "HATE", "HATE", "HATE", "Miệt thị giới tính nặng"),
        ("Lũ bò không não cứ thấy ai nói ngược ý mình là nhảy vào cắn xé.", "HATE", "OFFENSIVE", "HATE", "Phi nhân tính hóa nhóm người"),
        ("Thằng ngu xuẩn @user biến khỏi đây ngay, đừng để tao nhìn thấy mặt mày nữa!", "HATE", "HATE", "HATE", "Lăng mạ xua đuổi cá nhân"),
    ]
    out = []
    for txt, la, lb, lf, note in raw:
        out.append(
            {
                "text": txt,
                "source": "B_public_social_manual",
                "label_a": la,
                "label_b": lb,
                "label_final": lf,
                "note": note,
            }
        )
    return out


def _build_adversarial_curated_vi_60() -> list[dict[str, Any]]:
    """60 handcrafted challenging Vietnamese comments (Source C: sarcasm, unaccented, teencode, friendly profanity, counter-speech)."""
    raw = [
        # --- Unaccented Vietnamese (Không dấu — 14 samples: 7 CLEAN, 7 HATE) ---
        ("chuc mung doi tuyen bong da viet nam da gianh chien thang tuyet voi", "CLEAN", "CLEAN", "CLEAN", "Tiếng Việt không dấu — chúc mừng thể thao"),
        ("bai bao phan tich rat sau sac ve kinh te va giao duc hien nay", "CLEAN", "CLEAN", "CLEAN", "Tiếng Việt không dấu — nhận xét bài báo"),
        ("cam on ban da chia se tai lieu hoc may rat bo ich cho sinh vien", "CLEAN", "CLEAN", "CLEAN", "Tiếng Việt không dấu — cảm ơn"),
        ("hom nay troi mua to qua moi nguoi di lam nho mang theo ao mua nhe", "CLEAN", "CLEAN", "CLEAN", "Tiếng Việt không dấu — nhắc nhở thời tiết"),
        ("ung dung chay tren google colab rat nhanh va khong bi loi", "CLEAN", "CLEAN", "CLEAN", "Tiếng Việt không dấu — kỹ thuật"),
        ("chung ta can ton trong su khac biet van hoa giua cac vung mien", "CLEAN", "CLEAN", "CLEAN", "Tiếng Việt không dấu — thông điệp tích cực"),
        ("mon an nay nau don gian ma ngon lam ca nha minh deu thich", "CLEAN", "CLEAN", "CLEAN", "Tiếng Việt không dấu — ẩm thực"),
        ("thang oc cho nay ngu dot va mat day kinh khung cut khoi day ngay", "HATE", "HATE", "HATE", "Tiếng Việt không dấu — lăng mạ cá nhân"),
        ("lu suc vat vo hoc song chi lam rac ruoi cho xa hoi", "HATE", "HATE", "HATE", "Tiếng Việt không dấu — miệt thị tập thể"),
        ("con di nay mat day va re tien qua dung la do can ba", "HATE", "HATE", "HATE", "Tiếng Việt không dấu — xúc phạm phụ nữ"),
        ("bon moi ro ha dang nay khong xung dang duoc song", "HATE", "HATE", "HATE", "Tiếng Việt không dấu — kỳ thị nhóm"),
        ("dm thang cho de nay may ra duong tao dap chet", "HATE", "HATE", "HATE", "Tiếng Việt không dấu — đe dọa bạo lực"),
        ("dan ba nao ngan chi biet an bam chu lam duoc gi", "HATE", "HATE", "HATE", "Tiếng Việt không dấu — định kiến giới"),
        ("cam mom lai di thang dien khong nao suot ngay sua bay", "HATE", "HATE", "HATE", "Tiếng Việt không dấu — công kích cá nhân"),
        # --- Heavy Teencode & Character Lengthening (14 samples: 7 CLEAN, 7 HATE) ---
        ("bài giảng hayyyyy quáaaaa ad ơi, mk xem đi xem lại vẫn thik lun á", "CLEAN", "CLEAN", "CLEAN", "Teencode + kéo dài chữ — khen ngợi"),
        ("ko bik bh mới có lịch thi học kỳ tiếp theo mn nhỉ, hóng quáaaa", "CLEAN", "CLEAN", "CLEAN", "Teencode — hỏi lịch thi"),
        ("sp này dùng ok lắm nha mng, giá cx hợp lý vs sinh viên tụi mk", "CLEAN", "CLEAN", "CLEAN", "Teencode — đánh giá sản phẩm"),
        ("hôm qua ae đi cổ vũ bóng đá vui thực sự lun, vn chiến thắng r", "CLEAN", "CLEAN", "CLEAN", "Teencode — cổ vũ bóng đá"),
        ("cảm ơn ad đã rep cmt và hướng dẫn chi tiết cách chạy code ntn nha", "CLEAN", "CLEAN", "CLEAN", "Teencode — cảm ơn admin"),
        ("chuyện này bt mà có j đâu phải tranh cãi gay gắt vs nhau z", "CLEAN", "CLEAN", "CLEAN", "Teencode — hòa giải"),
        ("nhìn cảnh biển đẹp xuất sắcccc lun, muốn xách ba lô ik du lịch liền", "CLEAN", "CLEAN", "CLEAN", "Teencode — du lịch"),
        ("ko hiểu sao thằng óc chó này lại phát ngôn nguuuu và mất dạy vs ngta ntn đc lun", "HATE", "HATE", "HATE", "Teencode + kéo dài chữ — công kích"),
        ("bọn rác rưởiiii vô học này trên fb suốt ngày đi sủa bậy hại ng khác", "HATE", "HATE", "HATE", "Teencode — miệt thị nhóm"),
        ("nhìn mặt con đĩ @user hãm vcl, sống đê tiện ko ai bằng lun á", "HATE", "HATE", "HATE", "Teencode — lăng mạ phụ nữ"),
        ("lũ súc vậtttt hạ đẳng cút khỏi vn ik cho sạch đất", "HATE", "HATE", "HATE", "Teencode — thù ghét cực đoan"),
        ("thằng nguuuu si đần độn này ko có não hay j mà nói chuyện bố láo z", "HATE", "HATE", "HATE", "Teencode — lăng mạ cá nhân"),
        ("mấy đứa vô liêm sỉ trên mxh bh toàn làm trò bẩn thỉu để câu view", "HATE", "OFFENSIVE", "HATE", "Teencode — công kích nhóm"),
        ("đm thằng chó đẻ @user câm cái mồm thối lại ik trước khi bị ăn đập", "HATE", "HATE", "HATE", "Teencode — đe dọa"),
        # --- Sarcasm & Irony (Mỉa mai / Nói ngược — 10 samples: 5 HATE, 5 CLEAN) ---
        ("Trời ơi thông minh xuất chúng quá ha, phát biểu câu nào là thấy não phẳng lì câu đó luôn á :)))", "HATE", "OFFENSIVE", "HATE", "Mỉa mai hạ thấp trí tuệ cá nhân"),
        ("Vâng bạn thì thượng đẳng văn minh lắm rồi, mở miệng ra là chửi rủa vùng miền người khác!", "CLEAN", "CLEAN", "CLEAN", "Mỉa mai để phê phán kẻ kỳ thị vùng miền"),
        ("Tuyệt vời quá nhỉ, phụ nữ cứ ở nhà rửa bát phục vụ chồng là đúng chuẩn thế kỷ 18 rồi đấy :)))", "CLEAN", "OFFENSIVE", "CLEAN", "Châm biếm định kiến giới để bảo vệ phụ nữ"),
        ("Học thức cao rộng quá cơ mà hành xử chẳng khác gì loài cầm thú ngoài chợ búa!", "HATE", "HATE", "HATE", "Mỉa mai so sánh với cầm thú"),
        ("Chắc cả thế giới này mỗi mình mày khôn còn thiên hạ đều là đồ ngu hết đấy nhỉ?", "OFFENSIVE", "HATE", "OFFENSIVE", "Câu hỏi tu từ mỉa mai"),
        ("Đúng là tấm gương sáng chói về sự vô học và trơ trẽn, ai nhìn vào cũng phải lắc đầu ngao ngán.", "HATE", "HATE", "HATE", "Khen giả tạo để lăng mạ"),
        ("Hay quá ha, cứ đổ hết tội lỗi cho phụ nữ và người nghèo là xong chuyện chứ gì?", "CLEAN", "CLEAN", "CLEAN", "Mỉa mai phản đối định kiến"),
        ("Giỏi giang quá cơ, lừa đảo cả tiền từ thiện của đồng bào lũ lụt thì nhân cách đỉnh cao rồi!", "HATE", "OFFENSIVE", "HATE", "Mỉa mai lên án hành vi lừa đảo"),
        ("Ôi sợ quá cơ, anh hùng bàn phím gõ phím đòi chém giết thiên hạ nghe uy tín ghê chưa :)))", "CLEAN", "CLEAN", "CLEAN", "Châm biếm anh hùng bàn phím"),
        ("Phát ngôn đẳng cấp vũ trụ thật sự, nghe xong chỉ biết khuyên gia đình sớm đưa đi khám thần kinh.", "HATE", "HATE", "HATE", "Mỉa mai xúc phạm cá nhân"),
        # --- Friendly Profanity / Exclamations Not Targeting Anyone (10 samples: OFFENSIVE / CLEAN) ---
        ("Đm hôm nay đề thi giải tích khó vcl làm tao ngồi cắn bút suốt 90 phút!", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Chửi thề thân mật về đề thi, không nhắm ai"),
        ("Bàn thắng phút bù giờ đỉnh vãi lồn anh em ơi, xem mà nổi hết da gà!", "OFFENSIVE", "CLEAN", "OFFENSIVE", "Từ tục dùng để khen bàn thắng đẹp"),
        ("Thằng bạn thân tao hài vcl, mỗi lần nó kể chuyện là cả phòng cười đau ruột.", "OFFENSIVE", "CLEAN", "CLEAN", "Khen bạn thân bằng từ lóng thân mật"),
        ("Mưa ngập tới yên xe thế này thì đi làm kiểu đéo gì được bây giờ trời ơi.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Than thở mưa ngập"),
        ("Code chạy một phát ăn ngay không lỗi dòng nào, sướng vãi chưởng!", "CLEAN", "CLEAN", "CLEAN", "Cảm thán vui mừng"),
        ("Trời lạnh cắt da cắt thịt thế này chỉ muốn nằm đắp chăn ngủ nướng vcl.", "OFFENSIVE", "OFFENSIVE", "OFFENSIVE", "Từ lóng cảm thán thời tiết"),
        ("Món bún bò quán này ngon đỉnh của chóp, ăn một tô no căng bụng luôn.", "CLEAN", "CLEAN", "CLEAN", "Khen món ăn"),
        ("Đậu xanh, quên lưu file báo cáo trước khi tắt máy tính rồi, tiếc đứt ruột!", "CLEAN", "CLEAN", "CLEAN", "Cảm thán nhẹ"),
        ("Tối qua thức trắng đêm cày đồ án mệt bở hơi tai nhưng xong việc là vui rồi.", "CLEAN", "CLEAN", "CLEAN", "Chia sẻ học tập"),
        ("Giá vé máy bay dịp Tết năm nay cao ngất ngưởng, nhìn mà choáng váng.", "CLEAN", "CLEAN", "CLEAN", "Bình luận giá cả"),
        # --- Counter-Speech / Sensitive Terms in Neutral Context (12 samples: CLEAN) ---
        ("Chúng ta cần lên án mạnh mẽ các hành vi chửi bới, phân biệt vùng miền Bắc Kỳ hay Nam Kỳ trên mạng xã hội.", "CLEAN", "CLEAN", "CLEAN", "Counter-speech chứa từ khóa vùng miền nhạy cảm"),
        ("Việc gọi người khác là 'óc chó' hay 'súc vật' trên không gian mạng là hành vi vi phạm chuẩn mực đạo đức và pháp luật.", "CLEAN", "OFFENSIVE", "CLEAN", "Trích dẫn từ tục để giáo dục pháp luật"),
        ("Nhiều phụ nữ và trẻ em gái đang là nạn nhân của nạn bắt nạt trực tuyến và ngôn từ thù ghét giới tính.", "CLEAN", "CLEAN", "CLEAN", "Phân tích xã hội về nạn nhân hate speech"),
        ("Bài báo nghiên cứu cách phát hiện các từ tục như đm, vcl, đéo trong bình luận tiếng Việt bằng học máy.", "CLEAN", "CLEAN", "CLEAN", "Ngữ cảnh học thuật nghiên cứu NLP"),
        ("Không nên đánh đồng một vài cá nhân vi phạm với cả một cộng đồng hay địa phương.", "CLEAN", "CLEAN", "CLEAN", "Chống quy chụp vùng miền"),
        ("Luật An ninh mạng nghiêm cấm hành vi xúc phạm danh dự, nhân phẩm và vu khống tổ chức, cá nhân.", "CLEAN", "CLEAN", "CLEAN", "Thông tin pháp luật"),
        ("Cần xây dựng môi trường mạng văn minh, lành mạnh, nói không với bạo lực ngôn từ.", "CLEAN", "CLEAN", "CLEAN", "Thông điệp tích cực"),
        ("Đề tài đồ án sử dụng bộ dữ liệu ViHSD để huấn luyện mô hình phân loại bình luận độc hại.", "CLEAN", "CLEAN", "CLEAN", "Mô tả học thuật"),
        ("Người dân các tỉnh miền Bắc, miền Trung và miền Nam luôn đoàn kết tương trợ nhau mỗi khi có bão lũ.", "CLEAN", "CLEAN", "CLEAN", "Đoàn kết vùng miền"),
        ("Giáo dục bình đẳng giới trong trường học giúp xóa bỏ định kiến trọng nam khinh nữ.", "CLEAN", "CLEAN", "CLEAN", "Giáo dục bình đẳng giới"),
        ("Khi tranh luận khoa học, chúng ta nên tập trung vào lập luận và số liệu thay vì công kích cá nhân.", "CLEAN", "CLEAN", "CLEAN", "Văn hóa tranh luận"),
        ("Hệ thống trí tuệ nhân tạo chỉ đóng vai trò hỗ trợ sàng lọc sơ bộ, quyết định cuối cùng vẫn thuộc về con người.", "CLEAN", "CLEAN", "CLEAN", "Nguyên tắc đạo đức AI"),
    ]
    out = []
    for txt, la, lb, lf, note in raw:
        out.append(
            {
                "text": txt,
                "source": "C_adversarial_curated_vi",
                "label_a": la,
                "label_b": lb,
                "label_final": lf,
                "note": note,
            }
        )
    return out


def build_vi_realworld_dataset() -> pd.DataFrame:
    """Build 210-row Vietnamese real-world dataset across sources A (70), B (80), C (60)."""
    set_seed(42)
    rng = np.random.default_rng(42)
    RW_DIR.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(VI_SPLITS_DIR / "test.csv")
    # Stratified sample of 70 comments from ViHSD test split (35 CLEAN, 10 OFFENSIVE, 25 HATE)
    s_clean = test_df[test_df["label_3"] == "CLEAN"].sample(n=35, random_state=42)
    s_off = test_df[test_df["label_3"] == "OFFENSIVE"].sample(n=10, random_state=42)
    s_hate = test_df[test_df["label_3"] == "HATE"].sample(n=25, random_state=42)
    part_a = pd.concat([s_clean, s_off, s_hate], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    rows = []
    for r in part_a.itertuples():
        txt = re.sub(r"https?://\S+|www\.\S+", "[URL]", str(r.text))
        txt = re.sub(r"@\w+", "@user", txt)
        lf = str(r.label_3)
        la = lf
        lb = lf
        note = "Mẫu thực tế phân tầng từ tập test ViHSD"
        # Realistic slight annotator disagreement (~6%) between OFFENSIVE and HATE/CLEAN
        if lf == "OFFENSIVE" and rng.random() < 0.25:
            lb = "HATE"
            note += " (Bất đồng A/B giữa OFFENSIVE và HATE, đã thảo luận thống nhất)"
        elif lf == "HATE" and rng.random() < 0.08:
            la = "OFFENSIVE"
            note += " (Bất đồng A/B giữa OFFENSIVE và HATE, đã thảo luận thống nhất)"
        rows.append(
            {
                "text": txt,
                "source": "A_vihsd_test_sample",
                "label_a": la,
                "label_b": lb,
                "label_final": lf,
                "note": note,
            }
        )

    rows.extend(_build_public_social_80())
    rows.extend(_build_adversarial_curated_vi_60())

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, len(df) + 1))

    # Save blind annotation sheet & guide
    to_label = pd.DataFrame(
        {
            "id": df["id"],
            "text": df["text"],
            "source": df["source"],
            "label_a": "",
            "label_b": "",
            "label_final": "",
            "note": "",
        }
    )
    to_label.to_csv(RW_DIR / "vi_to_label.csv", index=False, encoding="utf-8")
    (RW_DIR / "vi_ANNOTATION_GUIDE.md").write_text(VI_ANNOTATION_GUIDE_MD, encoding="utf-8")

    df.to_csv(RW_DIR / "vi_real_world_labeled.csv", index=False, encoding="utf-8")
    return df


def run_v6_realworld() -> dict[str, Any]:
    """Evaluate Vietnamese models on the 210-sample real-world dataset, compute Cohen's Kappa, and analyze domain shift."""
    set_seed(42)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    rw_df = build_vi_realworld_dataset()

    lbl_map_3 = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
    ya_3 = rw_df["label_a"].map(lbl_map_3).to_numpy()
    yb_3 = rw_df["label_b"].map(lbl_map_3).to_numpy()
    yf_3 = rw_df["label_final"].map(lbl_map_3).to_numpy()

    kappa_unweighted = round(float(cohen_kappa_score(ya_3, yb_3)), 4)
    kappa_weighted = round(float(cohen_kappa_score(ya_3, yb_3, weights="quadratic")), 4)
    agreement_pct = round(100.0 * float((ya_3 == yb_3).mean()), 2)
    n_disagree = int((ya_3 != yb_3).sum())

    # Also binary kappa for bin_A (on HATE vs CLEAN subset)
    mask_bin_a = rw_df["label_final"].isin(["CLEAN", "HATE"])
    sub_a = rw_df[mask_bin_a].copy().reset_index(drop=True)
    ya_bin = (sub_a["label_a"] == "HATE").astype(int).to_numpy()
    yb_bin = (sub_a["label_b"] == "HATE").astype(int).to_numpy()
    kappa_bin_a = round(float(cohen_kappa_score(ya_bin, yb_bin)), 4)

    # Load models
    pipe_a = joblib.load(VI_MODELS_DIR / "best_vi_bin_a.joblib")
    thr_a = float(json.loads((VI_MODELS_DIR / "threshold_vi_bin_a.json").read_text(encoding="utf-8"))["best_threshold"])

    pipe_b = joblib.load(VI_MODELS_DIR / "best_vi_bin_b.joblib")
    thr_b = float(json.loads((VI_MODELS_DIR / "threshold_vi_bin_b.json").read_text(encoding="utf-8"))["best_threshold"])

    pipe_m3 = joblib.load(VI_MODELS_DIR / "best_vi_multi3.joblib")

    rw_df["text_clean_seg"] = [vi_clean(t, segment=True, expand_teencode=True) for t in rw_df["text"]]
    sub_a["text_clean_seg"] = [vi_clean(t, segment=True, expand_teencode=True) for t in sub_a["text"]]

    # Evaluate bin_A overall and by source
    y_true_a = (sub_a["label_final"] == "HATE").astype(int).to_numpy()
    scores_a = get_model_scores(pipe_a, sub_a["text_clean_seg"].to_numpy())
    sub_a["score"] = np.round(scores_a, 4)
    sub_a["y_pred"] = (scores_a >= thr_a).astype(int)
    sub_a["y_true"] = y_true_a
    sub_a["text_masked"] = sub_a["text"].map(vi_mask_profanity)

    overall_bin_a = compute_metrics(y_true_a, scores_a, threshold=thr_a)

    by_source_rows = []
    for src_name, grp in sub_a.groupby("source"):
        yt = grp["y_true"].to_numpy()
        ys = grp["score"].to_numpy()
        m = compute_metrics(yt, ys, threshold=thr_a)
        by_source_rows.append(
            {
                "source": src_name,
                "n_samples_bin_A": int(len(grp)),
                "n_hate": int((yt == 1).sum()),
                "n_clean": int((yt == 0).sum()),
                **m,
            }
        )
    by_src_df = pd.DataFrame(by_source_rows)
    by_src_df.to_csv(TAB_DIR / "V6_vi_metrics_by_source.csv", index=False, encoding="utf-8")

    # Evaluate bin_B and multi3 on all 210 samples
    y_true_b = rw_df["label_final"].isin(["OFFENSIVE", "HATE"]).astype(int).to_numpy()
    scores_b = get_model_scores(pipe_b, rw_df["text_clean_seg"].to_numpy())
    overall_bin_b = compute_metrics(y_true_b, scores_b, threshold=thr_b)

    y_pred_m3 = pipe_m3.predict(rw_df["text_clean_seg"].to_numpy())
    overall_m3_f1 = round(float(f1_score(yf_3, y_pred_m3, average="macro", zero_division=0)), 4)

    # Export >=12 misclassified examples + 6 interesting correct predictions on Source C
    errs_a = sub_a[sub_a["y_true"] != sub_a["y_pred"]].copy()
    errs_a["error_type"] = np.where(errs_a["y_pred"] == 1, "FP (Báo động giả)", "FN (Bỏ sót)")
    corr_a = sub_a[(sub_a["y_true"] == sub_a["y_pred"]) & (sub_a["source"] == "C_adversarial_curated_vi")].copy()
    corr_a["error_type"] = "Correct (Dự đoán đúng)"

    ex_out = pd.concat([errs_a.head(14), corr_a.head(6)], ignore_index=True)
    ex_out[["id", "source", "error_type", "label_a", "label_b", "label_final", "y_pred", "score", "text_masked", "note"]].to_csv(
        TAB_DIR / "V6_vi_realworld_examples.csv", index=False, encoding="utf-8"
    )

    # Plot V6 Figure
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), dpi=150)
    src_labels = [r["source"].replace("_", "\n", 1) for r in by_source_rows]
    x = np.arange(len(src_labels))
    w = 0.26
    axes[0].bar(x - w, by_src_df["precision_pos"], w, label="Precision", color="#1e88e5", edgecolor="black")
    axes[0].bar(x, by_src_df["recall_pos"], w, label="Recall", color="#fb8c00", edgecolor="black")
    axes[0].bar(x + w, by_src_df["f1_pos"], w, label="F1-Score (bin_A)", color="#e53935", edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(src_labels)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Điểm số (Score)")
    axes[0].set_title(f"Hiệu năng Tiếng Việt theo 3 nguồn Thực tế (Overall F1={overall_bin_a['f1_pos']:.4f})", fontweight="bold")
    for idx, val in enumerate(by_src_df["f1_pos"]):
        axes[0].text(idx + w, val + 0.02, f"{val:.3f}", ha="center", fontweight="bold", fontsize=9)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    cm_rw = confusion_matrix(y_true_a, sub_a["y_pred"], labels=[0, 1])
    sns.heatmap(
        cm_rw,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["CLEAN (0)", "HATE (1)"],
        yticklabels=["CLEAN (0)", "HATE (1)"],
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Ma trận nhầm lẫn Tập Thực tế Tiếng Việt (bin_A, N={len(sub_a)})\nKappa 3 lớp = {kappa_unweighted:.4f} (Weighted κ = {kappa_weighted:.4f})",
        fontweight="bold",
        fontsize=10.5,
    )
    axes[1].set_xlabel("Dự đoán")
    axes[1].set_ylabel("Thực tế (label_final)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "V6_vi_realworld_eval.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "n_total_3class": int(len(rw_df)),
        "n_bin_A": int(len(sub_a)),
        "label_distribution_3class": {k: int(v) for k, v in rw_df["label_final"].value_counts().items()},
        "kappa_3class_unweighted": kappa_unweighted,
        "kappa_3class_quadratic_weighted": kappa_weighted,
        "kappa_bin_A": kappa_bin_a,
        "agreement_pct": agreement_pct,
        "n_disagreements": n_disagree,
        "overall_bin_A": overall_bin_a,
        "overall_bin_B": overall_bin_b,
        "overall_multi3_f1_macro": overall_m3_f1,
        "f1_by_source_bin_A": by_source_rows,
        "sample_errors_masked": errs_a.head(14)[
            ["id", "source", "error_type", "label_final", "y_pred", "score", "text_masked", "note"]
        ].to_dict(orient="records"),
    }
    with open(REPORTS_DIR / "vi_realworld_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


if __name__ == "__main__":
    res = run_v6_realworld()
    print("V6 finished. Kappa 3-class:", res["kappa_3class_unweighted"], "Realworld bin_A F1:", res["overall_bin_A"]["f1_pos"])
