"""Streamlit Web Application for Racist/Sexist Tweet Binary Classification (P10)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.predict import Predictor, load_model
from src.utils import mask_profanity

st.set_page_config(
    page_title="Hệ thống Phát hiện Tweet Phản cảm (Racist/Sexist)",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def get_cached_predictor() -> Predictor:
    """Load and cache the trained best_model.joblib and threshold.json."""
    return load_model(
        path=PROJECT_ROOT / "models" / "best_model.joblib",
        thr_path=PROJECT_ROOT / "models" / "threshold.json",
    )


EXAMPLE_TWEETS = [
    (
        "1. Bình thường (Đời sống & Công nghệ)",
        "Just finished reading an amazing book on machine learning and natural language processing with my friends!",
    ),
    (
        "2. Bình thường (Có từ nhạy cảm nhưng phản đối kỳ thị)",
        "I am NOT racist and we must stand together against racism and sexism in our schools and workplaces.",
    ),
    (
        "3. Phản cảm — Phân biệt giới tính (Đã che từ tục)",
        "Women do not belong in leadership or engineering roles because they are too emotional and inferior to men.",
    ),
    (
        "4. Phản cảm — Phân biệt chủng tộc (Đã che từ tục)",
        "Send all those filthy immigrants back where they came from, white supremacy is the only way to save our race.",
    ),
    (
        "5. Câu mỉa mai / Ẩn ý (Thử thách hạn chế mô hình)",
        "Oh yeah right, because letting women vote and run companies has worked out SO wonderfully for society lol #sarcasm",
    ),
]


def main() -> None:
    st.title("🛡️ Hệ thống Phân loại Tweet Phân biệt Chủng tộc & Giới tính")
    st.markdown(
        "**Đề tài:** Gắn nhãn các đoạn tweet có nội dung phân biệt chủng tộc, phân biệt giới tính "
        "(*Twitter Sentiment / Hate Speech Binary Classification*)"
    )

    try:
        predictor = get_cached_predictor()
    except Exception as e:
        st.error(f"Không thể nạp mô hình từ `models/best_model.joblib`: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Tab 1: Phân loại 1 Tweet",
            "📂 Tab 2: Phân loại Hàng loạt (CSV)",
            "📊 Tab 3: Thống kê & Hồ sơ Mô hình",
        ]
    )

    # =========================================================================
    # TAB 1: PHÂN LOẠI 1 TWEET
    # =========================================================================
    with tab1:
        st.subheader("Phân tích & Giải thích từng đoạn Tweet")
        st.caption("Chọn nhanh một trong 5 câu mẫu dưới đây hoặc nhập trực tiếp nội dung tweet (tối đa 1,000 ký tự):")

        if "input_tweet" not in st.session_state:
            st.session_state["input_tweet"] = EXAMPLE_TWEETS[0][1]

        cols_ex = st.columns(5)
        for idx, (btn_label, sample_txt) in enumerate(EXAMPLE_TWEETS):
            if cols_ex[idx].button(btn_label, use_container_width=True):
                st.session_state["input_tweet"] = sample_txt

        raw_input = st.text_area(
            "Nội dung Tweet cần kiểm tra:",
            value=st.session_state["input_tweet"],
            height=110,
            max_chars=1000,
        )

        analyze_clicked = st.button("🚀 Phân tích Tweet", type="primary")
        if analyze_clicked or raw_input.strip():
            if not raw_input or not raw_input.strip():
                st.warning("Vui lòng nhập nội dung tweet không rỗng để phân tích.")
            else:
                res = predictor.predict_one(raw_input)
                score = float(res["score"])
                thr = float(res["threshold"])
                label = int(res["label"])

                col_res1, col_res2 = st.columns([1.15, 1.0])
                with col_res1:
                    if label == 1:
                        st.error(
                            f"🚨 **KẾT QUẢ: PHẢN CẢM — PHÂN BIỆT CHỦNG TỘC / GIỚI TÍNH (Nhãn = 1)**\n\n"
                            f"- **Điểm xác suất mô hình:** `{score:.4f}` (Ngưỡng quyết định: `{thr:.4f}`)\n"
                            f"- **Đánh giá:** Điểm vượt ngưỡng `{thr:.4f}` (+`{score - thr:.4f}`)."
                        )
                    else:
                        st.success(
                            f"✅ **KẾT QUẢ: BÌNH THƯỜNG / KHÔNG KỲ THỊ (Nhãn = 0)**\n\n"
                            f"- **Điểm xác suất mô hình:** `{score:.4f}` (Ngưỡng quyết định: `{thr:.4f}`)\n"
                            f"- **Đánh giá:** Điểm thấp hơn ngưỡng `{thr:.4f}` (`{score - thr:.4f}`)."
                        )

                    st.markdown(f"**Thanh mức độ rủi ro so với ngưỡng quyết định (`{thr:.3f}`):**")
                    st.progress(min(max(score, 0.0), 1.0))
                    st.markdown(f"**Văn bản sau làm sạch (`clean_text`):** `{mask_profanity(res['clean_text'])}`")

                with col_res2:
                    st.markdown("#### 🔎 Giải thích đóng góp đặc trưng (Top Terms: `TF-IDF × Hệ số`)")
                    top_terms = res.get("top_terms", [])
                    if not top_terms:
                        st.info("Không có từ khóa đặc trưng nào nằm trong từ điển hoặc mô hình không hỗ trợ.")
                    else:
                        badges_md = []
                        for term, weight in top_terms:
                            if weight > 0:
                                badges_md.append(
                                    f":red[**+{term}** (`+{weight:.3f}` ↑ Phản cảm)]"
                                )
                            else:
                                badges_md.append(
                                    f":green[**{term}** (`{weight:.3f}` ↓ Bình thường)]"
                                )
                        for b_line in badges_md:
                            st.markdown(f"- {b_line}")

    # =========================================================================
    # TAB 2: PHÂN LOẠI HÀNG LOẠT (CSV)
    # =========================================================================
    with tab2:
        st.subheader("Phân loại Hàng loạt từ File CSV (Tối đa 20,000 dòng)")
        st.markdown(
            "Tải lên file `.csv` có chứa cột `text` (hoặc `tweet` / `clean_text`). "
            "Hệ thống sẽ làm sạch, dự đoán nhãn nhị phân và cho phép tải xuống kết quả."
        )

        uploaded_file = st.file_uploader("Chọn file CSV:", type=["csv"])
        use_sample_rw = st.button("📋 Hoặc chạy thử ngay trên 100 mẫu Thực tế (`real_world_labeled.csv`)")

        df_in = None
        if uploaded_file is not None:
            df_in = pd.read_csv(uploaded_file)
        elif use_sample_rw:
            rw_path = PROJECT_ROOT / "data" / "real_world" / "real_world_labeled.csv"
            if rw_path.exists():
                df_in = pd.read_csv(rw_path).head(100)

        if df_in is not None:
            text_col = None
            for cand in ["text", "tweet", "clean_text", "clean_comment"]:
                if cand in df_in.columns:
                    text_col = cand
                    break

            if text_col is None:
                st.error(f"Không tìm thấy cột văn bản (`text`, `tweet`, `clean_text`) trong file! Các cột hiện có: {list(df_in.columns)}")
            else:
                if len(df_in) > 20000:
                    st.warning("File vượt quá 20,000 dòng. Hệ thống tự động cắt 20,000 dòng đầu tiên.")
                    df_in = df_in.head(20000).copy()

                prog = st.progress(0, text="Đang phân loại hàng loạt...")
                texts = df_in[text_col].fillna("").astype(str).tolist()
                batch_out = predictor.predict_batch(texts)
                prog.progress(100, text="Hoàn tất phân loại!")

                batch_display = batch_out.copy()
                batch_display["text"] = batch_display["text"].map(mask_profanity)
                batch_display["text_clean"] = batch_display["text_clean"].map(mask_profanity)

                n_total = len(batch_display)
                n_pos = int((batch_display["label"] == 1).sum())
                n_neg = n_total - n_pos

                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng số dòng đã xử lý", f"{n_total:,}")
                m2.metric("Số tweet Bình thường (Nhãn 0)", f"{n_neg:,} ({100*n_neg/max(n_total,1):.1f}%)")
                m3.metric("Số tweet Phản cảm (Nhãn 1)", f"{n_pos:,} ({100*n_pos/max(n_total,1):.1f}%)")

                st.dataframe(batch_display, use_container_width=True, height=340)

                csv_bytes = batch_out.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="⬇️ Tải xuống kết quả phân loại (.csv)",
                    data=csv_bytes,
                    file_name="ket_qua_phan_loai_tweet.csv",
                    mime="text/csv",
                )

    # =========================================================================
    # TAB 3: THỐNG KÊ & HỒ SƠ MÔ HÌNH
    # =========================================================================
    with tab3:
        st.subheader("Hồ sơ Mô hình được chọn (`model_card.json`) & So sánh Thực nghiệm")

        mc_path = PROJECT_ROOT / "models" / "model_card.json"
        ft_path = PROJECT_ROOT / "reports" / "final_test_metrics.json"
        comp_path = PROJECT_ROOT / "reports" / "model_comparison.csv"

        col_info1, col_info2 = st.columns(2)
        if mc_path.exists():
            mc = json.loads(mc_path.read_text(encoding="utf-8"))
            with col_info1:
                st.markdown("#### 📌 Thông tin Mô hình Chốt (`best_model`)")
                st.write(f"- **Tên mô hình:** `{mc.get('model_name')}` (`{mc.get('model_key')}`)")
                st.write(f"- **Ngưỡng quyết định tối ưu (Val):** `{mc.get('best_threshold')}`")
                st.write(f"- **Số mẫu huấn luyện (Train + Val):** `{mc.get('refit_on_train_plus_val_rows', 0):,}` dòng")
                st.write(f"- **Lý do lựa chọn:** {mc.get('selection_reason')}")
                st.json(mc.get("best_params", {}))

        if ft_path.exists():
            ft = json.loads(ft_path.read_text(encoding="utf-8"))
            tm = ft.get("metrics_at_tuned_threshold", {})
            with col_info2:
                st.markdown("#### 🏆 Kết quả Đánh giá Cuối trên Tập Test (`test.csv`)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Test F1 (Lớp 1)", f"{tm.get('f1_pos', 0):.4f}")
                k2.metric("Test PR-AUC", f"{tm.get('pr_auc', 0):.4f}")
                k3.metric("Test Precision", f"{tm.get('precision_pos', 0):.4f}")
                k4.metric("Test Recall", f"{tm.get('recall_pos', 0):.4f}")
                st.write(f"- **ROC-AUC:** `{tm.get('roc_auc', 0):.4f}` | **MCC:** `{tm.get('mcc', 0):.4f}`")
                st.write(f"- **Ma trận nhầm lẫn Test:** TN=`{tm.get('tn')}`, FP=`{tm.get('fp')}`, FN=`{tm.get('fn')}`, TP=`{tm.get('tp')}`")
                st.write(f"- **Thời điểm chạy Test (1 lần duy nhất):** `{ft.get('evaluated_once_at')}`")

        if comp_path.exists():
            st.markdown("#### 📈 Bảng So sánh Hiệu năng các Mô hình trên Tập Validation (`model_comparison.csv`)")
            comp_df = pd.read_csv(comp_path)
            st.dataframe(comp_df, use_container_width=True)

        fig_col1, fig_col2 = st.columns(2)
        fig_test = PROJECT_ROOT / "reports" / "figures" / "P8_test_evaluation.png"
        fig_feat = PROJECT_ROOT / "reports" / "figures" / "P8_feature_importance.png"
        if fig_test.exists():
            fig_col1.image(str(fig_test), caption="Ma trận nhầm lẫn, ROC và PR trên tập Test (P8)")
        if fig_feat.exists():
            fig_col2.image(str(fig_feat), caption="Top từ đặc trưng đóng góp cho Lớp 1 và Lớp 0 (Đã che từ tục)")

    st.divider()
    st.caption(
        "⚠️ **Cảnh báo đạo đức & vận hành:** *Mô hình có thể sai (đặc biệt với câu mỉa mai, ẩn ý hoặc từ nhạy cảm "
        "dùng trong ngữ cảnh trung tính/phản đối kỳ thị); công cụ chỉ hỗ trợ sàng lọc sơ bộ, cần con người kiểm tra lại.*"
    )


if __name__ == "__main__":
    main()
