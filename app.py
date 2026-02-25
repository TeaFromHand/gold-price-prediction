import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="Gold Price Prediction App", layout="wide")

st.title("📊 Gold Price Direction Prediction")
st.markdown("""
This app predicts whether the **Gold Price** will increase or decrease in the following week using an **XGBoost Classifier**.
""")

# --- 1. Load Data ---
@st.cache_data
def load_data():
    # Loading the uploaded dataset
    df = pd.read_csv('gold_weekly_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={'Close': 'Gold_Price'}, inplace=True)
    return df

try:
    df_raw = load_data()
    st.sidebar.success("Dataset loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 2. Feature Engineering ---
def preprocess_data(df):
    df = df.copy()
    # Target variable: 1 if increase, 0 if decrease
    df['Target_Direction'] = (df['Gold_Price'].shift(-1) > df['Gold_Price']).astype(int)

    # Features
    df['Gold_Diff'] = df['Gold_Price'].diff()
    df['Gold_Volat'] = df['High'] - df['Low']

    # RSI (14-week)
    delta = df['Gold_Diff']
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # External Indicators
    external_cols = ['Oil_Price', 'USD_Index', 'Silver_Price', 'SP500', 
                     'Bond_Yield_10Y', 'VIX_Index', 'FED_Rate', 
                     'CPI_Inflation', 'Real_Yield_10Y']
    
    for col in external_cols:
        df[f'{col}_Diff'] = df[col].diff()

    # Lags
    cols_to_lag = ['Gold_Diff', 'Gold_Volat', 'RSI_14'] + [f'{c}_Diff' for c in external_cols]
    for col in cols_to_lag:
        df[f'{col}_Lag1'] = df[col].shift(1)
        df[f'{col}_Lag2'] = df[col].shift(2)

    return df.dropna().reset_index(drop=True)

df_processed = preprocess_data(df_raw)

# --- 3. Sidebar UI ---
st.sidebar.header("Model Parameters")
test_weeks = st.sidebar.slider("Test Set Size (Weeks)", 20, 100, 52)
n_estimators = st.sidebar.number_input("XGBoost Estimators", 100, 1000, 300)

# --- 4. Main Layout ---
col_charts, col_stats = st.columns([2, 1])

with col_charts:
    st.subheader("Historical Gold Price")
    st.line_chart(df_raw.set_index('Date')['Gold_Price'])

# --- 5. Training & Prediction ---
features = [col for col in df_processed.columns if '_Lag' in col]
target = 'Target_Direction'

train_data = df_processed.iloc[:-test_weeks]
test_data = df_processed.iloc[-test_weeks:]

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

if st.button('🚀 Run Prediction Model'):
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, 
        learning_rate=0.01, 
        max_depth=5, 
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Threshold Optimization
    probabilities = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.3, 0.61, 0.02)
    best_threshold = 0.5
    best_f1 = 0

    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)
        score = f1_score(y_test, preds, average='macro')
        if score > best_f1:
            best_f1 = score
            best_threshold = thresh

    final_preds = (probabilities >= best_threshold).astype(int)

    # Display Metrics
    st.divider()
    st.subheader("Model Performance")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Optimal Threshold", f"{best_threshold:.2f}")
    m2.metric("Macro F1-Score", f"{best_f1:.4f}")
    
    st.text("Classification Report:")
    st.code(classification_report(y_test, final_preds))

    # Feature Importance Plot
    st.subheader("Top Influencing Factors")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=True).tail(10)
    
    fig, ax = plt.subplots()
    ax.barh(fi_df['Feature'], fi_df['Importance'], color='gold')
    st.pyplot(fig)
else:
    st.info("Click the button above to train the model and see predictions.")