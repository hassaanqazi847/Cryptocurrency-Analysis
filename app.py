import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model
from datetime import timedelta

@st.cache_data(ttl=3600)
def load_data(coin_ticker: str):
    df = yf.download(coin_ticker, start="2020-01-01", end="2025-08-28")
    df = df.reset_index()
    df["Return"] = df["Close"].pct_change()
    df["MA7"] = df["Close"].rolling(7).mean()
    df["MA30"] = df["Close"].rolling(30).mean()
    df["Volatility_7d"] = df["Return"].rolling(7).std()
    df = df.dropna()
    return df

@st.cache_resource
def load_lstm_model():
    return load_model("doge_lstm_model.h5")

@st.cache_resource
def load_rf_model():
    return joblib.load("doge_rf_model.pkl")

@st.cache_resource
def load_xgb_model():
    return joblib.load("doge_xgb_model.pkl")

def prepare_lstm_input(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data[["Close"]].values)
    X_input = scaled.reshape(1, len(scaled), 1)
    return X_input, scaler, scaled

def prepare_ml_input(data):
    features = ["Close", "Volume", "MA7", "MA30", "Return", "Volatility_7d"]
    return data[features].iloc[-1:].values

#*************************************************************** Streamlit UI ***********************************************************

st.title("Crypto Price Predictor (Next Week)")

coin_input = st.text_input("Enter coin ticker (e.g., DOGE-USD)", value="DOGE-USD")
model_choice = st.selectbox("Choose Prediction Model:", ["LSTM", "Random Forest", "XGBoost"])

if st.button("Predict Next Week"):
    data = load_data(coin_input)
    if data.empty:
        st.error("No data found — check the ticker.")
    else:
        future_dates = [data["Date"].iloc[-1] + timedelta(days=i) for i in range(1, 8)]
        predictions = []

        if model_choice == "LSTM":
            model = load_lstm_model()
            X_input, scaler, scaled = prepare_lstm_input(data)
            seq = scaled.tolist()

            for _ in range(7):
                pred_scaled = model.predict(np.array(X_input))
                pred_price = scaler.inverse_transform(pred_scaled)[0][0]
                predictions.append(pred_price)

                seq.append([pred_scaled[0][0]])
                X_input = np.array(seq[-len(scaled):]).reshape(1, len(scaled), 1)

        elif model_choice in ["Random Forest", "XGBoost"]:
            model = load_rf_model() if model_choice == "Random Forest" else load_xgb_model()
            temp_data = data.copy()

            for _ in range(7):
                X_input = prepare_ml_input(temp_data)
                if X_input.shape[0] == 0:
                    break

                pred_price = model.predict(X_input)[0]
                predictions.append(pred_price)

                new_row = temp_data.iloc[-1:].copy()
                new_row["Close"] = pred_price
                new_row["Date"] = new_row["Date"] + timedelta(days=1)

                recent = pd.concat([temp_data.tail(29), new_row], ignore_index=True)
                new_row["Return"] = recent["Close"].pct_change().iloc[-1]
                new_row["MA7"] = recent["Close"].rolling(7).mean().iloc[-1]
                new_row["MA30"] = recent["Close"].rolling(30).mean().iloc[-1]
                new_row["Volatility_7d"] = recent["Return"].rolling(7).std().iloc[-1]

                temp_data = pd.concat([temp_data, new_row], ignore_index=True)

#********************************************************************Results **************************************************************

        pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": predictions})
        st.write("📈 Next Week Predictions:")
        st.dataframe(pred_df)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["Date"], data['Close'], label="Historical Close Price")
        ax.plot(pred_df["Date"], pred_df["Predicted Price"], marker="o", color='red',
                label=f"{model_choice} Prediction (Next 7 Days)")
        ax.legend()
        st.pyplot(fig)
