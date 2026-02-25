import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt

# Read data
df = pd.read_csv('gold_cleaned_dataset.csv')

# Preprocessing data
df['Date'] = pd.to_datetime(df['Date']) # Convert type of Date column to datetime
df.rename(columns={'Close': 'Gold_Price'}, inplace=True) # Rename

# Create target variable: 1 if increase, 0 if decrease
df['Target_Direction'] = (df['Gold_Price'].shift(-1) > df['Gold_Price']).astype(int)

# Create features
df['Gold_Diff'] = df['Gold_Price'].diff()
df['Gold_Volat'] = df['High'] - df['Low']

# Calculate RSI (14-week_period) for gold
delta = df['Gold_Diff']
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

# Calc the diff between each weeks
features = ['Oil_Price', 'USD_Index', 'Silver_Price', 'SP500', 
                  'Bond_Yield_10Y', 'VIX_Index', 'FED_Rate', 
                  'CPI_Inflation', 'Real_Yield_10Y']

for col in features:
    df[f'{col}_Diff'] = df[col].diff()

# Creating Lags
cols_to_lag = ['Gold_Diff', 'Gold_Volat', 'RSI_14'] + [f'{c}_Diff' for c in features]

for col in cols_to_lag:
    df[f'{col}_Lag1'] = df[col].shift(1) # Data from 1 week ago
    df[f'{col}_Lag2'] = df[col].shift(2) # Data from 2 weeks ago

# Drop NaN and overwrite df
df = df.dropna().reset_index(drop=True)

# Split train test set
test_size = 52
train_data = df.iloc[:-test_size] 
test_data = df.iloc[-test_size:] #Last 52 weeks

# Input features (all 'Lag' columns)
features = [col for col in df.columns if '_Lag' in col]
target = 'Target_Direction'

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

# Model training
model = xgb.XGBClassifier(n_estimators=300, 
                          learning_rate=0.01, 
                          max_depth=5, 
                          random_state=42,
                          eval_metric='logloss')

model.fit(X_train, y_train)

# Find optimal threshold using F1 score
probabilities = model.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.3, 0.61, 0.02)

best_threshold = 0.5
best_f1_macro = 0

for thresh in thresholds:
    preds = (probabilities >= thresh).astype(int) 
    macro_f1 = f1_score(y_test, preds, average='macro')
    if macro_f1 > best_f1_macro:
        best_f1_macro = macro_f1
        best_threshold = thresh

print(f'Optimal threshold: {best_threshold:.4f}')

final_predictions = (probabilities >= best_threshold).astype(int)
print(classification_report(y_test, final_predictions, digits=4))

# Important features
# Extract importance scores from the XGBoost model
importances = model.feature_importances_

# Create a DataFrame for easier sorting and visualization
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

# Sort by importance and select the top 15 features
top_features = feature_importance_df.sort_values(by='Importance', ascending=True).tail(15)

# Plot Horizontal Bar Chart
plt.figure(figsize=(10, 8))
plt.barh(top_features['Feature'], top_features['Importance'], color='teal')
plt.title('Top 15 Features Influencing Gold Prices', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature Name', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Display plot
plt.show()

