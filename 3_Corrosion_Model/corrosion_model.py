##############################################################################################
##############################################################################################
###########                                                                        ###########
###########                               CMAS_predict                             ###########
###########                                                                        ###########
###########                       CV_MODE = 'kfold'  #####  fix                    ###########
###########                                                                        ###########
##############################################################################################
##############################################################################################

import pandas as pd 
import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.stats import zscore

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import (ShuffleSplit, KFold, RepeatedKFold,
                                     GridSearchCV)

try:
    from xgboost import XGBRegressor
    _xgb_available = True
except Exception as e:
    warnings.warn(f"XGBoost import failed: {e}")
    _xgb_available = False

RANDOM_STATE = 42  #####  fix
np.random.seed(RANDOM_STATE)

USE_SINGLE_SPLIT = False

data = pd.read_csv('property_cmas.csv')
X = data.drop(columns=[data.columns[0], 'crg'])
y = data['crg']

z_scores = np.abs(zscore(X))
mask = (z_scores < 3).all(axis=1)
X_no_outliers = X[mask]
y_no_outliers = y[mask]

deleted_rows = len(X) - len(X_no_outliers)
retained_rows = len(X_no_outliers)
print(f"Deleted {deleted_rows} rows; retained {retained_rows} rows.")

CV_MODE = 'kfold'  #####  fix
TEST_SIZE = 0.10 

n_samples = len(X_no_outliers)
n_splits_k = max(2, min(5, n_samples))
n_splits_rk = max(2, min(5, n_samples))
n_repeats_rk = 2

if CV_MODE == 'shuffle':
    splitter = ShuffleSplit(n_splits=5, test_size=TEST_SIZE, random_state=RANDOM_STATE)
elif CV_MODE == 'kfold':
    splitter = KFold(n_splits=n_splits_k, shuffle=True, random_state=RANDOM_STATE)
elif CV_MODE == 'rkfold':
    splitter = RepeatedKFold(n_splits=n_splits_rk, n_repeats=n_repeats_rk, random_state=RANDOM_STATE)
else:
    raise ValueError(f"Invalid CV_MODE: {CV_MODE!r}. Expected 'shuffle', 'kfold', or 'rkfold'.")

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
param_grid_svr = {
    'C': [1, 10, 100],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['rbf']
}
param_grid_gb = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 150, 200],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0]
}
param_grid_xgb = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5]
}

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    "Support Vector Regression": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}
if _xgb_available:
    models["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1)

def fit_scale_pca(X_tr, X_te, n_pcs=8):
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    pca = PCA(n_components=min(X_tr.shape[1], n_pcs))
    X_tr_pca = pca.fit_transform(X_tr_scaled)
    X_te_pca = pca.transform(X_te_scaled)
    return scaler, pca, X_tr_scaled, X_te_scaled, X_tr_pca, X_te_pca

def grid_for(name, model):
    if name == "Random Forest":
        pg = param_grid_rf
    elif name == "Support Vector Regression":
        pg = param_grid_svr
    elif name == "Gradient Boosting":
        pg = param_grid_gb
    elif name == "XGBoost":
        pg = param_grid_xgb
    else:
        pg = {}
    return GridSearchCV(model, param_grid=pg, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

def eval_once(X_tr, X_te, y_tr, y_te, export_pca=False):
    scaler, pca, X_tr_scaled, X_te_scaled, X_tr_pca, X_te_pca = fit_scale_pca(X_tr, X_te)

    if export_pca:
        pca_components_df = pd.DataFrame(pca.components_, columns=X.columns)
        pca_components_df.to_csv('pca_components_contribution.csv', index=False)
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        pca_variance_df = pd.DataFrame({
            'Principal Component': np.arange(1, len(cumulative_variance_ratio) + 1),
            'Cumulative Explained Variance Ratio': cumulative_variance_ratio
        })
        pca_variance_df.to_csv('pca_explained_variance.csv', index=False)

    results = []
    for name, model in models.items():
        grid = grid_for(name, model)
        try:
            grid.fit(X_tr_scaled, y_tr)
            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_te_scaled)
            mse = mean_squared_error(y_te, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_te, y_pred)
            results.append({
                "Model": name, "Features": "All",
                "MSE": mse, "RMSE": rmse, "R2": r2,
                "Best Params": grid.best_params_
            })

            best_model_pca = grid.best_estimator_
            best_model_pca.fit(X_tr_pca, y_tr)
            y_pred_pca = best_model_pca.predict(X_te_pca)
            mse_pca = mean_squared_error(y_te, y_pred_pca)
            rmse_pca = np.sqrt(mse_pca)
            r2_pca = r2_score(y_te, y_pred_pca)
            results.append({
                "Model": name, "Features": "PCA",
                "MSE": mse_pca, "RMSE": rmse_pca, "R2": r2_pca,
                "Best Params": grid.best_params_
            })
        except Exception as e:
            warnings.warn(f"Model {name} failed to train: {e}")

    if _xgb_available:
        try:
            xgb_baseline = XGBRegressor(random_state=RANDOM_STATE, n_estimators=100,
                                        max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=-1)
            xgb_baseline.fit(X_tr_scaled, y_tr)
            y_pred_xgb = xgb_baseline.predict(X_te_scaled)
            mse_xgb = mean_squared_error(y_te, y_pred_xgb)
            rmse_xgb = np.sqrt(mse_xgb)
            r2_xgb = r2_score(y_te, y_pred_xgb)
            results.append({
                "Model": "XGBoost (Baseline)",
                "Features": "All",
                "MSE": mse_xgb, "RMSE": rmse_xgb, "R2": r2_xgb,
                "Best Params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1}
            })
        except Exception as e:
            warnings.warn(f"XGBoost baseline training failed: {e}")
    return results, scaler, pca

if USE_SINGLE_SPLIT:
    train_idx, test_idx = next(iter(splitter.split(X_no_outliers, y_no_outliers)))
    X_train, X_test = X_no_outliers.iloc[train_idx], X_no_outliers.iloc[test_idx]
    y_train, y_test = y_no_outliers.iloc[train_idx], y_no_outliers.iloc[test_idx]
    print(f"[Split] Mode: {CV_MODE} | Training set: {len(X_train)} | Test set: {len(X_test)}")

    results, scaler, pca = eval_once(X_train, X_test, y_train, y_test, export_pca=True)
    results_df = pd.DataFrame(results)
    print("Regression Model Performance Comparison (Test-Set Evaluation):\n", results_df.sort_values(by='R2', ascending=False))
    results_df.to_csv('optimized_model_comparison.csv', index=False)
else:
    all_rows = []
    first_split_taken = False
    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X_no_outliers, y_no_outliers), start=1):
        X_train, X_test = X_no_outliers.iloc[train_idx], X_no_outliers.iloc[test_idx]
        y_train, y_test = y_no_outliers.iloc[train_idx], y_no_outliers.iloc[test_idx]

        export_pca_flag = (not first_split_taken) 
        results, scaler, pca = eval_once(X_train, X_test, y_train, y_test, export_pca=export_pca_flag)

        for r in results:
            r_copy = r.copy()
            r_copy["Fold"] = fold_id
            all_rows.append(r_copy)

        if not first_split_taken:
            X_train_first, X_test_first = X_train, X_test
            y_train_first, y_test_first = y_train, y_test
            scaler_first, pca_first = scaler, pca
            first_split_taken = True

    cv_df = pd.DataFrame(all_rows)
    cv_summary = cv_df.groupby(["Model", "Features"]).agg(
        MSE_mean=("MSE", "mean"), MSE_std=("MSE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        Folds=("Fold", "nunique")
    ).reset_index().sort_values(by=["R2_mean"], ascending=False)
    print("Cross-Validation Summary (Mean +- SD):\n", cv_summary)
    cv_summary.to_csv("cv_summary.csv", index=False)

    results = cv_df[cv_df["Fold"] == 1].drop(columns=["Fold"])
    results_df = results.sort_values(by='R2', ascending=False)
    results_df.to_csv('optimized_model_comparison.csv', index=False)

    X_train, X_test = X_train_first, X_test_first
    y_train, y_test = y_train_first, y_test_first
    scaler, pca = scaler_first, pca_first

gb_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
gb_model.fit(X_train_scaled, y_train)

feature_importance = gb_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feature_importance_df.to_csv('feature_importance.csv', index=False)

def instantiate_model_by_name(name):
    if name == "Linear Regression":
        return LinearRegression()
    if name == "Decision Tree":
        return DecisionTreeRegressor(random_state=RANDOM_STATE)
    if name == "Random Forest":
        return RandomForestRegressor(random_state=RANDOM_STATE)
    if name == "Support Vector Regression":
        return SVR()
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE)
    if name == "XGBoost":
        if not _xgb_available:
            raise RuntimeError("XGBoost is not available")
        return XGBRegressor(random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1)
    raise ValueError(f"Unknown model name: {name}")

best_row = results_df.iloc[results_df['R2'].values.argmax()]
best_model_name = best_row["Model"]
best_features = best_row["Features"]
best_params = best_row.get("Best Params", {})

if (best_model_name == "XGBoost") and (not _xgb_available):
    tmp = results_df[results_df["Model"] != "XGBoost"]
    if len(tmp) == 0:
        raise RuntimeError("No available model for exporting predictions.")
    best_row = tmp.iloc[tmp['R2'].values.argmax()]
    best_model_name = best_row["Model"]
    best_features = best_row["Features"]
    best_params = best_row.get("Best Params", {})

print(f"[Chosen for export] Model={best_model_name}, Features={best_features}, Params={best_params}")

chosen_model = instantiate_model_by_name(best_model_name)
if isinstance(best_params, dict) and len(best_params) > 0:
    try:
        chosen_model.set_params(**best_params)
    except Exception as e:
        warnings.warn(f"Failed to apply optimal parameters; using default parameters instead: {e}")

if best_features == "PCA":
    Xtr_for_fit = pca.transform(X_train_scaled)
    Xte_for_eval = pca.transform(X_test_scaled)
else:
    Xtr_for_fit = X_train_scaled
    Xte_for_eval = X_test_scaled

chosen_model.fit(Xtr_for_fit, y_train)
y_pred_best = chosen_model.predict(Xte_for_eval)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)
print(f"[Chosen model on test] RMSE={rmse_best:.4f}, R2={r2_best:.4f}")

with open("chosen_model_info.txt", "w", encoding="utf-8") as f:
    f.write(f"Chosen Model: {best_model_name}\n")
    f.write(f"Features: {best_features}\n")
    f.write(f"Best Params: {best_params}\n")
    f.write(f"Test RMSE: {rmse_best:.6f}, Test R2: {r2_best:.6f}\n")

pre_data = pd.read_csv('parameter_compound.csv')
X_pre_raw = pre_data.drop(columns=[pre_data.columns[0]])

train_means = X_train.mean()
X_pre = X_pre_raw.reindex(columns=X_train.columns)
X_pre = X_pre.fillna(train_means)

X_pre_scaled = scaler.transform(X_pre)
if best_features == "PCA":
    X_pre_used = pca.transform(X_pre_scaled)
else:
    X_pre_used = X_pre_scaled

pre_predictions = chosen_model.predict(X_pre_used)
pre_data['CVModel_Predicted_crg'] = pre_predictions
pre_data.to_csv('predictions_cmas.csv', index=False)
print("Prediction results have been saved to 'predictions_cmas.csv' (using the best model selected by cross-validation).")
