# 📈 Gold Price Direction Prediction

This project implements a Machine Learning model to predict the weekly price direction of Gold (Increase/Decrease) using **XGBoost**. 
It includes a feature engineering pipeline with technical indicators and external financial markers, deployed as an interactive web application via **Streamlit**.

## 🚀 Features
* **Directional Prediction**: Classifies whether the gold price will rise or fall in the following week.
* **Technical Indicators**: Calculates RSI (14-week) and price volatility.
* **Market Correlation**: Incorporates external data such as the USD Index, S&P 500, Bond Yields, and Oil prices to improve prediction accuracy.
* **Lag Features**: Implements time-series lagging (Lag-1, Lag-2) to capture historical trends.
* **Threshold Optimization**: Automatically finds the optimal probability threshold to maximize the Macro F1-Score.

## 🛠️ Tech Stack
* **Language**: Python.
* **ML Model**: XGBoost Classifier.
* **Web Framework**: Streamlit.
* **Data Libraries**: Pandas, NumPy, Scikit-learn.
* **Visualization**: Matplotlib.

## 📊 Dataset
The model is trained on a weekly historical dataset containing:
* **Gold Prices**: Open, High, Low, Close (OHLC).
* **Economic Indicators**: CPI Inflation, FED Rate, and Real Yields.
* **Market Indices**: VIX Index, SP500, and USD Index.

*Note: Ensure the dataset file `gold_weekly_prices.csv` is in the root directory.*

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/gold-price-prediction.git](https://github.com/your-username/gold-price-prediction.git)
   cd gold-price-prediction
