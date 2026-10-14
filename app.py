import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Dự Báo Cổ Phiếu AI", layout="wide")
st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")

# Khung nhập mã cổ phiếu
ticker = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", "VNM.VN")

# Tạo 3 cột để đặt ngày bắt đầu, ngày kết thúc và số ngày dự báo
col1, col2, col3 = st.columns(3)

with col1:
    start_date = st.date_input("🗓️ Chọn ngày bắt đầu:", value=date(2018, 1, 1))

with col2:
    end_date = st.date_input("🗓️ Chọn ngày kết thúc:", value=date.today())

with col3:
    future_days = st.number_input("🔮 Số ngày muốn dự báo:", min_value=1, max_value=60, value=10, step=1)

if start_date >= end_date:
    st.error("Lỗi: Ngày bắt đầu phải diễn ra trước ngày kết thúc!")
else:
    if st.button("Tải dữ liệu & Chạy dự báo", type="primary"):
        with st.spinner("Hệ thống đang tải dữ liệu và huấn luyện mô hình (sẽ mất khoảng 1-2 phút)..."):
            
            # Cố định seed
            seed = 42
            np.random.seed(seed)
            tf.random.set_seed(seed)
            random.seed(seed)
            
            # Tải dữ liệu
            df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            
            if df.empty:
                st.error(f"Không tìm thấy dữ liệu cho mã {ticker}. Vui lòng kiểm tra lại!")
            else:
                data = df[['Close']].values
                
                if len(data) <= 60:
                    st.error("Khoảng thời gian quá ngắn. Vui lòng chọn khoảng thời gian dài hơn (ít nhất 3 tháng)!")
                else:
                    train_size = int(0.8 * len(data))
                    train_data = data[:train_size]
                    test_data = data[train_size:]
                    
                    scaler = MinMaxScaler()
                    train_scaled = scaler.fit_transform(train_data)
                    test_scaled = scaler.transform(test_data)
                    
                    def create_sequences(data, window_size=60):
                        X, y = [], []
                        for i in range(window_size, len(data)):
                            X.append(data[i-window_size:i])
                            y.append(data[i])
                        return np.array(X), np.array(y)
                        
                    X_train, y_train = create_sequences(train_scaled)
                    test_combined = np.vstack((train_scaled[-60:], test_scaled))
                    X_test, y_test = create_sequences(test_combined)
                    
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    val_size = int(0.1 * len(X_train))
                    X_val = X_train[-val_size:]
                    y_val = y_train[-val_size:]
                    X_train_final = X_train[:-val_size]
                    y_train_final = y_train[:-val_size]
                    
                    # Xây dựng và train mô hình
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
                    model.add(LSTM(50))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    
                    history = model.fit(
                        X_train_final, y_train_final,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=32,
                        shuffle=False,   
                        callbacks=[early_stop],
                        verbose=0 
                    )
                    
                    test_pred = model.predict(X_test, verbose=0)
                    test_pred = scaler.inverse_transform(test_pred)
                    y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
                    
                    # --- DỰ BÁO TƯƠNG LAI ---
                    last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
                    last_60_days = last_60_days.reshape(1, 60, 1)
                    future_predictions = []
                    
                    for i in range(future_days):
                        next_pred = model.predict(last_60_days, verbose=0)
                        future_predictions.append(next_pred[0,0])
                        last_60_days = np.append(
                            last_60_days[:,1:,:],
                            next_pred.reshape(1,1,1),
                            axis=1
                        )
                    
                    future_predictions = np.array(future_predictions).reshape(-1,1)
                    future_predictions = scaler.inverse_transform(future_predictions)
                    
                    # --- KẾT LUẬN & KHUYẾN NGHỊ ---
                    st.subheader("💡 Kết luận & Khuyến nghị (Tham khảo)")
                    
                    current_price = y_test_real[-1][0]
                    final_future_price = future_predictions[-1][0]
                    
                    price_diff = final_future_price - current_price
                    percent_diff = (price_diff / current_price) * 100
                    
                    st.write(f"- Giá trị thực tế gần nhất: **{current_price:,.2f}**")
                    st.write(f"- Giá trị dự báo sau {future_days} ngày: **{final_future_price:,.2f}**")
                    
                    if price_diff > 0:
                        st.success(f"📈 **Xu hướng:** TĂNG. Khuyến nghị: Cân nhắc **MUA** hoặc **NẮM GIỮ** (Dự kiến tăng {percent_diff:.2f}%)")
                    elif price_diff < 0:
                        st.warning(f"📉 **Xu hướng:** GIẢM. Khuyến nghị: Cân nhắc **BÁN** hoặc **ĐỨNG NGOÀI QUAN SÁT** (Dự kiến giảm {abs(percent_diff):.2f}%)")
                    else:
                        st.info(f"➖ **Xu hướng:** ĐI NGANG. Khuyến nghị: Theo dõi thêm thị trường.")
                        
                    st.caption("Lưu ý: Khuyến nghị này chỉ dựa trên thuật toán AI phân tích dữ liệu lịch sử và chỉ mang tính chất tham khảo. Không phải lời khuyên đầu tư tài chính.")

                    # --- HIỂN THỊ KẾT QUẢ & BIỂU ĐỒ ---
                    st.subheader(f"🚀 Dự báo giá {future_days} ngày tiếp theo")
                    future_df = pd.DataFrame(future_predictions, columns=["Giá dự báo"], index=[f"Ngày {i+1}" for i in range(future_days)])
                    
                    col_table, col_chart = st.columns([1, 2])
                    
                    with col_table:
                        st.table(future_df.style.format("{:,.2f}"))
                        
                    with col_chart:
                        fig, ax = plt.subplots(figsize=(10,5))
                        
                        # Phục hồi lại đường dự báo test so với giá thật
                        ax.plot(y_test_real, label="Giá thật")
                        ax.plot(test_pred, label="Dự báo test")
                        
                        # Nối tiếp đường dự báo tương lai
                        future_x = range(len(y_test_real), len(y_test_real) + future_days)
                        ax.plot(future_x, future_predictions, marker='o', color='red', label=f"Dự báo {future_days} ngày tới")
                        
                        ax.set_title("Biểu đồ giá thực tế và dự báo")
                        ax.legend()
                        st.pyplot(fig)
                    
                    with st.expander("📊 Xem chi tiết đánh giá kỹ thuật của mô hình AI (RMSE, MAE, R2, Loss)"):
                        rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
                        mae = mean_absolute_error(y_test_real, test_pred)
                        r2 = r2_score(y_test_real, test_pred)
                        
                        col_m1, col_m2, col_m3 = st.columns(3)
                        col_m1.metric("RMSE", f"{rmse:.2f}")
                        col_m2.metric("MAE", f"{mae:.2f}")
                        col_m3.metric("R² Score", f"{r2:.2f}")
                        
                        fig_loss, ax_loss = plt.subplots(figsize=(10,4))
                        ax_loss.plot(history.history['loss'], label='Train Loss')
                        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
                        ax_loss.set_title("Loss Train vs Validation")
                        ax_loss.legend()
                        st.pyplot(fig_loss)
