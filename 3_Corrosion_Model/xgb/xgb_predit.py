##############################################################################################
##############################################################################################
###########                                                                        ###########
###########                              CMAS_predict                              ###########
###########                                                                        ###########
###########                              xgboost_base                              ###########
###########                                                                        ###########
##############################################################################################
##############################################################################################
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from xgboost import XGBRegressor
import joblib
import os

data = pd.read_csv('property_cmas.csv')
if 'crg' not in data.columns:
    raise KeyError("Target column 'crg' not found in property_cmas.csv")

X = data.drop(columns=[data.columns[0], 'crg'])
y = data['crg']

zs = np.abs(zscore(X))
mask = (zs < 3).all(axis=1)
X_no = X[mask]
y_no = y[mask]
print(f"Removed {len(X) - len(X_no)} rows; kept {len(X_no)} rows.")

X_train, X_test, y_train, y_test = train_test_split(
    X_no, y_no, test_size=0.10, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

xgb_baseline = XGBRegressor(
    random_state=42,
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric='rmse'
)
xgb_baseline.fit(X_train_scaled, y_train)

y_pred = xgb_baseline.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"[Baseline] MSE={mse:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}")

pd.DataFrame([{
    "Model": "XGBoost (Baseline)",
    "Features": "All",
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2,
    "Params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}
}]).to_csv('baseline_metrics.csv', index=False)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label='XGBoost (Baseline)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         linestyle='--', label='Ideal')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('XGBoost (Baseline) Predicted vs. True Values')
plt.legend()
plt.tight_layout()
plt.savefig('xgb_baseline_predictions_vs_true.png', dpi=200)
plt.show()

pd.DataFrame({'True Values': y_test.values, 'Baseline Predictions': y_pred})\
  .to_csv('xgb_baseline_predictions_vs_true.csv', index=False)

joblib.dump(xgb_baseline, 'xgb_baseline_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
with open('train_columns.json', 'w', encoding='utf-8') as f:
    json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)
print("Saved:xgb_baseline_model.pkl, scaler.pkl, train_columns.json")

if os.path.exists('predictions_cmas.csv'):
    pre_data = pd.read_csv('predictions_cmas.csv')
    X_pre = pre_data.drop(columns=[pre_data.columns[0]])

    train_cols = list(X_train.columns)
    missing = set(train_cols) - set(X_pre.columns)
    if missing:
        raise ValueError(f"predictions_cmas.csv is missing required training columns: {sorted(missing)}")
    X_pre = X_pre[train_cols]

    scaler_loaded = joblib.load('scaler.pkl')
    model_loaded  = joblib.load('xgb_baseline_model.pkl')

    X_pre_scaled = scaler_loaded.transform(X_pre)
    y_pre = model_loaded.predict(X_pre_scaled)

    pre_data['XGB_Predicted_crg'] = y_pre
    pre_data.to_csv('preproperties_predictions.csv', index=False)
    print("Predictions saved to preproperties_predictions.csv")
else:
    print("predictions_cmas.csv not found, skipping prediction step.")
