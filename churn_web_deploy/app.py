"""Telco churn demonstration app based on [DS_6]_XG_BOOST_(1).ipynb.
Run locally: streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Telco Churn | ML Capstone",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
RANDOM_STATE = 42
GAIN_TP = 150
COST_FP = 50

st.markdown(
    """<style>
    .block-container {padding-top: 2.0rem; max-width: 1330px;}
    div[data-testid="stMetric"] {padding: 12px 16px; border: 1px solid #e5eaf0; border-radius: 12px;}
    div[data-testid="stMetricLabel"] {font-size: 0.85rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Đang lấy dữ liệu công khai từ IBM...")
def get_dataset() -> pd.DataFrame:
    raw = pd.read_csv(DATA_URL)
    necessary = {"customerID", "Churn", *NUM_COLS}
    if not necessary.issubset(raw.columns):
        raise ValueError("Nguồn dữ liệu chưa có đủ các cột Telco Churn.")
    return raw


@st.cache_resource(show_spinner="Đang chuẩn bị 2 mô hình dự báo...")
def get_models() -> dict:
    raw = get_dataset().copy()
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce")
    raw["Churn"] = raw["Churn"].eq("Yes").astype(int)
    features = raw.drop(columns=["customerID", "Churn"])
    target = raw["Churn"]

    # Same train/test split and preprocessing logic as the supplied notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE,
        stratify=target,
    )
    cat_cols = [c for c in features.columns if c not in NUM_COLS]

    def make_pipeline(classifier):
        prep = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ])
        return Pipeline([("prep", prep), ("model", classifier)])

    # Notebook's final chosen model: best LogisticRegression C=3.
    logistic = make_pipeline(LogisticRegression(C=3, max_iter=2000))
    logistic.fit(X_train, y_train)

    # Notebook's requirement 5: same XGBoost hyperparameters.
    xgb = make_pipeline(XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ))
    xgb.fit(X_train, y_train)
    return {
        "models": {"Logistic Regression": logistic, "XGBoost": xgb},
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "raw": raw,
    }


st.title("📊 Telco Customer Churn")
st.caption("ML Capstone · Phân tích nguy cơ rời bỏ khách hàng · Dựa trên Lab 6")

try:
    store = get_models()
except Exception as exc:
    st.error("Không thể tải hoặc huấn luyện mô hình từ bộ dữ liệu IBM công khai.")
    st.info("Kiểm tra kết nối tới GitHub Raw hoặc tải lại trang. Chi tiết lỗi phía dưới:")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_name = st.radio(
        "Mô hình dự báo",
        ["Logistic Regression", "XGBoost"],
        help="Logistic là mô hình cuối từ notebook; XGBoost là mô hình bổ sung tại Yêu cầu 5.",
    )
    default_threshold = 0.28 if model_name == "Logistic Regression" else 0.50
    threshold = st.slider(
        "Ngưỡng gửi ưu đãi",
        min_value=0.05, max_value=0.95, value=default_threshold,
        step=0.01, key="threshold_" + model_name,
    )
    st.caption(
        "0,28 là ngưỡng tối ưu từ OOF train của notebook cho Logistic. "
        "Ngưỡng 0,50 của XGBoost là mốc mặc định, không phải tối ưu OOF."
    )
    st.divider()
    st.markdown("**Nguồn:** IBM Telco Customer Churn (7.043 bản ghi)")
    st.caption("Ứng dụng minh họa học thuật, không dùng làm quyết định kinh doanh thực tế.")

model = store["models"][model_name]
X_test, y_test = store["X_test"], store["y_test"]
p_test = model.predict_proba(X_test)[:, 1]
pred_test = (p_test >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, pred_test, labels=[0, 1]).ravel()
profit_test = GAIN_TP * tp - COST_FP * fp

tab_overview, tab_single, tab_batch, tab_benchmark = st.tabs([
    "📈 Tổng quan", "👤 Dự báo 1 khách hàng", "📁 Dự báo từ CSV", "🧪 Benchmark Lab 6"
])

with tab_overview:
    st.subheader("Đánh giá trên tập test độc lập 20%")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("ROC–AUC", f"{roc_auc_score(y_test, p_test):.3f}")
    k2.metric("PR–AUC", f"{average_precision_score(y_test, p_test):.3f}")
    k3.metric("Ngưỡng hành động", f"{threshold:.2f}")
    k4.metric("Lợi nhuận mô phỏng", f"${profit_test:,.0f}")
    st.caption("Lợi nhuận minh họa = 150 USD × TP − 50 USD × FP; giả định từ notebook.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Ma trận nhầm lẫn theo ngưỡng đang chọn**")
        confusion = pd.DataFrame(
            [[int(tn), int(fp)], [int(fn), int(tp)]],
            columns=["Dự báo ở lại", "Dự báo rời bỏ"],
            index=["Thực tế ở lại", "Thực tế rời bỏ"],
        )
        st.dataframe(confusion, use_container_width=True)
        st.markdown("**Tần suất rời bỏ theo hợp đồng**")
        by_contract = (
            store["raw"].groupby("Contract")["Churn"]
            .mean().mul(100).rename("Tỷ lệ churn (%)")
        )
        st.bar_chart(by_contract)
    with right:
        st.markdown("**Phân bố xác suất dự báo**")
        hist, edges = np.histogram(p_test, bins=np.linspace(0, 1, 21))
        hist_df = pd.DataFrame({
            "Khoảng xác suất": [f"{a:.2f}–{b:.2f}" for a, b in zip(edges[:-1], edges[1:])],
            "Số khách hàng": hist,
        }).set_index("Khoảng xác suất")
        st.bar_chart(hist_df)
        st.info(
            "Ngưỡng hành động của Logistic được chọn trên dự báo out-of-fold "
            "của tập train, tránh chọn ngưỡng trên tập test. "
            "XGBoost đang dùng ngưỡng tùy chỉnh để minh họa."
        )

with tab_single:
    st.subheader("Nhập thông tin để dự báo")
    st.caption("Nhập đầy đủ 19 biến như trong bộ dữ liệu gốc. Tổng tiền có thể được ước tính từ số tháng × phí tháng.")
    X = store["features"]
    inputs = {}
    category_cols = [c for c in X.columns if c not in NUM_COLS]
    seed = X.iloc[0]
    with st.form("customer_prediction"):
        cols = st.columns(3)
        for idx, field in enumerate(X.columns):
            with cols[idx % 3]:
                if field == "SeniorCitizen":
                    inputs[field] = st.selectbox(
                        "SeniorCitizen (0/1)", [0, 1], index=int(seed[field])
                    )
                elif field == "tenure":
                    inputs[field] = st.number_input(
                        "tenure (tháng)", min_value=0, max_value=120,
                        value=int(seed[field]), step=1,
                    )
                elif field == "MonthlyCharges":
                    inputs[field] = st.number_input(
                        "MonthlyCharges (USD)", min_value=0.0, max_value=10000.0,
                        value=float(seed[field]), step=1.0,
                    )
                elif field == "TotalCharges":
                    inputs[field] = st.number_input(
                        "TotalCharges (USD)", min_value=0.0, max_value=1000000.0,
                        value=float(seed[field]) if pd.notna(seed[field]) else 0.0, step=10.0,
                    )
                else:
                    options = sorted(X[field].dropna().astype(str).unique().tolist())
                    current = str(seed[field])
                    inputs[field] = st.selectbox(
                        field, options, index=options.index(current) if current in options else 0
                    )
        submitted = st.form_submit_button("🔍 Dự báo nguy cơ Churn", type="primary")
    if submitted:
        new_customer = pd.DataFrame([inputs], columns=X.columns)
        probability = float(model.predict_proba(new_customer)[0, 1])
        action = probability >= threshold
        a, b = st.columns(2)
        a.metric("Xác suất khách hàng rời bỏ", f"{probability:.1%}")
        b.metric("Gợi ý chiến dịch", "Ưu tiên tiếp cận" if action else "Chưa cần ưu tiên")
        st.progress(max(0.0, min(1.0, probability)))
        st.caption(
            "Đây là xác suất dự báo của mô hình thống kê trên dữ liệu mẫu; "
            "không phải tác động nhân quả của việc gửi ưu đãi."
        )

with tab_batch:
    st.subheader("Dự báo hàng loạt từ file CSV")
    st.write("CSV cần có 19 biến đầu vào như dữ liệu IBM. Có thể giữ cột customerID để đối chiếu.")
    example = store["features"].head(5).copy()
    st.download_button(
        "⬇️ Tải CSV mẫu (5 khách hàng)",
        example.to_csv(index=False).encode("utf-8-sig"),
        file_name="telco_churn_input_sample.csv", mime="text/csv",
    )
    upload = st.file_uploader("Chọn CSV dữ liệu khách hàng", type="csv")
    if upload is not None:
        try:
            data = pd.read_csv(upload)
        except Exception as exc:
            st.error(f"Không đọc được CSV: {exc}")
            st.stop()
        required = list(store["features"].columns)
        missing = [c for c in required if c not in data.columns]
        if missing:
            st.error("Thiếu cột bắt buộc: " + ", ".join(missing))
        elif data.empty:
            st.warning("CSV không có bản ghi.")
        else:
            prepared = data[required].copy()
            for col in NUM_COLS:
                prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
            probabilities = model.predict_proba(prepared)[:, 1]
            result = data.copy()
            result["churn_probability"] = np.round(probabilities, 4)
            result["recommend_offer"] = np.where(probabilities >= threshold, "Yes", "No")
            st.success(f"Đã dự báo {len(result):,} khách hàng.")
            st.dataframe(result.head(100), use_container_width=True)
            st.download_button(
                "⬇️ Tải kết quả dự báo CSV",
                result.to_csv(index=False).encode("utf-8-sig"),
                file_name="telco_churn_predictions.csv", mime="text/csv", type="primary",
            )

with tab_benchmark:
    st.subheader("So sánh mô hình từ notebook đã chạy")
    st.caption(
        "Các điểm số dưới đây được ghi nhận từ kết quả 5-fold cross-validation "
        "đã lưu trong notebook gốc, KHÔNG phải kết quả chạy lại trên web."
    )
    benchmark = pd.DataFrame([
        ("LogisticRegression", 0.8460, 0.6610, 0.5920),
        ("RandomForest", 0.8450, 0.6560, 0.5720),
        ("XGBoost", 0.8447, 0.6638, 0.5855),
        ("MLP", 0.8420, 0.6490, 0.5730),
        ("HistGradBoosting", 0.8330, 0.6410, 0.5680),
        ("kNN (k=25)", 0.8320, 0.6110, 0.6020),
        ("GaussianNB", 0.8210, 0.6150, 0.5970),
        ("SVC (RBF)", 0.8000, 0.6370, 0.5730),
    ], columns=["Mô hình", "ROC-AUC", "PR-AUC", "F1"])
    st.dataframe(
        benchmark.style.format({"ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}", "F1": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )
    st.bar_chart(benchmark.set_index("Mô hình")[["ROC-AUC", "PR-AUC", "F1"]])
    st.markdown("**Kết quả học thuật nổi bật từ notebook**")
    st.markdown(
        "- Mô hình cuối được chọn: **Logistic Regression (C=3)**; "
        "test ROC-AUC khoảng **0,842**, PR-AUC khoảng **0,630**.\n"
        "- Ngưỡng tối ưu OOF của mô hình cuối: **0,28**.\n"
        "- Fairness: độ chênh recall giữa nhóm SeniorCitizen ≈ **17,3 điểm %**.\n"
        "- XGBoost đạt **ROC-AUC CV = 0,8447** trong bài tập bổ sung."
    )
    st.warning(
        "Chú ý: Mô hình được fit lại 1 lần trên tập train khi khởi động web "
        "(không chạy lại đầy đủ GridSearchCV / 6 bài tập). "
        "Để tái lập toàn bộ báo cáo, hãy dùng notebook đính kèm trong dự án."
    )

st.divider()
st.caption("Nguồn dữ liệu: IBM Telco Customer Churn · Minh họa học thuật, không sử dụng dữ liệu thật của khách hàng.")
