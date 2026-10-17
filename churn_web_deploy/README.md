# Telco Churn — Streamlit web app

Ứng dụng web dự báo khách hàng rời bỏ từ notebook Lab 6 `[DS_6]_XG_BOOST_(1).ipynb`.

## Deploy miễn phí lên Streamlit Community Cloud

1. Truy cập https://github.com/new và tạo repository GitHub mới, ví dụ `telco-churn-web` (Public).
2. Giải nén file ZIP, chọn **Add file → Upload files** tại repo GitHub và tải lên **nội dung bên trong thư mục**, đảm bảo có `app.py` và `requirements.txt` ở thư mục gốc. Commit changes.
3. Truy cập https://share.streamlit.io/ và đăng nhập bằng GitHub, bấm **Create app** → chọn repository vừa tạo → branch `main` → main file path `app.py` → **Deploy**.
4. Mở URL `*.streamlit.app` được cung cấp và sử dụng trang dự báo.

## Chạy thử trên máy cá nhân

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Chức năng

- Dashboard trực tiếp từ IBM Telco Churn (test ROC-AUC, PR-AUC, confusion matrix, lợi nhuận mô phỏng).
- Chọn **Logistic Regression** (mô hình cuối notebook, `C=3`) hoặc **XGBoost** (mô hình bổ sung Yêu cầu 5).
- Dự báo thủ công một khách hàng với 19 biến gốc.
- Upload nhiều khách hàng bằng CSV và tải bảng kết quả.
- Xem bảng benchmark từ outputs notebook gốc.

## Giữ logic notebook gốc

- Dữ liệu IBM được tải từ đường dẫn GitHub Raw gốc, mỗi lần khởi tạo môi trường Streamlit.
- `TotalCharges` chuyển số; `customerID` không dùng làm feature; train/test stratified 80/20, `random_state=42`.
- Tiền xử lý gồm median imputation, scaling và OneHotEncoder (`handle_unknown='ignore'`).
- Logistic `C=3` là tham số tốt nhất được notebook ghi nhận, ngưỡng kinh doanh mặc định `0.28` từ OOF train.
- XGBoost có siêu tham số đúng với notebook, nhưng mặc định threshold `0.50` (không phải threshold đã tối ưu). 
- Website fit **chỉ 2 mô hình** cho mục đích dự báo; **không** chạy lại toàn bộ 5-fold benchmark và GridSearchCV vốn tốn tài nguyên.
- Kết quả benchmark hiển thị là số liệu lưu sẵn của notebook; không tự nhận là những số vừa tính trên web.

## Những lưu ý quan trọng

- Mô hình dùng dữ liệu công khai, không phải mô hình được thẩm định cho doanh nghiệp thật.
- Mỗi khi app khởi động sau một thời gian ngủ, mô hình có thể phải được fit lại; Streamlit cache giữ mô hình cho các tương tác sau đó.
- Nếu Github Raw bị chặn thì tải dữ liệu từ https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv vào cùng repo tên `Telco-Customer-Churn.csv`, rồi đổi hàm `get_dataset()` đọc local CSV.
- Nếu muốn đưa đúng file `churn_model.joblib` từ Colab lên web, cần sửa mã để `joblib.load(...)` thay phần fit; với dự án minh họa hiện tại không bắt buộc.
