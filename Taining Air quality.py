import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib


df = pd.read_csv("Airquality_Feed_Newdata.csv", parse_dates=['created_at'])
df = df[['created_at', 'field1', 'field2', 'field3']]
df.columns = ['created_at', 'CO2', 'CH4', 'NH3']
df.dropna(inplace=True)
df.set_index('created_at', inplace=True)


df = df.resample('1H').mean().dropna()



# Add lag features (last 6 hours)
for lag in range(1, 7):
    df[f'CO2_lag{lag}'] = df['CO2'].shift(lag)
    df[f'CH4_lag{lag}'] = df['CH4'].shift(lag)
    df[f'NH3_lag{lag}'] = df['NH3'].shift(lag)

# Add time-based features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek

# Drop rows with NaN from lagging
df.dropna(inplace=True)


feature_cols = [col for col in df.columns if 'lag' in col or col in ['hour', 'day_of_week']]
X = df[feature_cols]
y = df[['CO2', 'CH4', 'NH3']]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


tscv = TimeSeriesSplit(n_splits=5)
best_model = None
best_score = -np.inf

for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    if score > best_score:
        best_score = score
        best_model = model
        best_X_test = X_test
        best_y_test = y_test


y_pred = best_model.predict(best_X_test)
mse = mean_squared_error(best_y_test, y_pred)
r2 = r2_score(best_y_test, y_pred)

print("✅ Model Trained")
print(f"📉 MSE: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")


joblib.dump(best_model, "gas_forecast_model.joblib")
joblib.dump(scaler, "gas_scaler.joblib")
print("\n📦 Model and Scaler saved as 'gas_forecast_model.joblib' and 'gas_scaler.joblib'")

import matplotlib.pyplot as plt
import os

# Create folder to save plots
os.makedirs("plots", exist_ok=True)

# Convert predictions and actuals to DataFrames
y_pred_df = pd.DataFrame(y_pred, columns=['CO2_Predicted', 'CH4_Predicted', 'NH3_Predicted'], index=best_y_test.index)
y_true_df = best_y_test.copy()


for gas in ['CO2', 'CH4', 'NH3']:
    plt.figure(figsize=(10, 4))
    plt.plot(y_true_df[gas].values, label=f'Actual {gas}', linewidth=2)
    plt.plot(y_pred_df[f'{gas}_Predicted'].values, label=f'Predicted {gas}', alpha=0.7, linewidth=2)
    plt.title(f'Actual vs Predicted - {gas}')
    plt.xlabel('Time Steps')
    plt.ylabel(gas)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/actual_vs_predicted_{gas}.png")
    plt.close()


for gas in ['CO2', 'CH4', 'NH3']:
    residuals = y_true_df[gas].values - y_pred_df[f'{gas}_Predicted'].values
    plt.figure(figsize=(8, 4))
    plt.plot(residuals, label=f'{gas} Residuals', color='orange')
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'{gas} Prediction Residuals')
    plt.xlabel('Time Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/residuals_{gas}.png")
    plt.close()

print("✅ All plots saved in 'plots/' folder.")

