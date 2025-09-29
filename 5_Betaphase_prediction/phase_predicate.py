# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

TRAIN_FILE = 'phase_data.csv'        
PRED_FILE  = 'calculated_properties.csv' 
OUT_METRICS_XLSX = 'voting_metrics.xlsx'
OUT_CM_PNG       = 'voting_confusion_matrix.png'
OUT_PRED_XLSX    = 'calculated_properties_with_pred.xlsx'

data = pd.read_csv(TRAIN_FILE)

if data.isnull().values.any():
    print("Warning: training data contains missing values. Dropping missing rows.")
    data = data.dropna()

le = LabelEncoder()
y = le.fit_transform(data['p'])

drop_cols = [c for c in ['Compounds', 'p', 'Compounds_clean'] if c in data.columns]
X = data.drop(columns=drop_cols)

for col in X.columns:
    if not np.issubdtype(X[col].dtype, np.number):
        X[col] = pd.to_numeric(X[col], errors='coerce')

nan_mask = X.isnull().any(axis=1)
if nan_mask.any():
    print(f"Warning: {nan_mask.sum()} rows had non-numeric values after coercion and will be dropped.")
    X = X[~nan_mask]
    y = y[~nan_mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

clf_brf = BalancedRandomForestClassifier(
    n_estimators=300,
    sampling_strategy='all',
    replacement=True,
    bootstrap=False,
    random_state=42
)

clf_rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    class_weight='balanced',  
    random_state=42,
    n_jobs=-1
)

voting = VotingClassifier(
    estimators=[('brf', clf_brf), ('rf', clf_rf)],
    voting='soft',
    weights=[2, 1]  
)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_acc = cross_val_score(voting, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
cv_f1  = cross_val_score(voting, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=-1)

print(f"CV Accuracy (mean ± std): {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
print(f"CV F1       (mean ± std): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

voting.fit(X_train, y_train)

y_pred = voting.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1  = f1_score(y_test, y_pred, average='weighted')
print(f"Holdout Test Accuracy: {acc:.4f}")
print(f"Holdout Test F1:       {f1:.4f}")

pd.DataFrame({
    'Metric': ['CV_Accuracy_mean', 'CV_Accuracy_std', 'CV_F1_mean', 'CV_F1_std', 'Test_Accuracy', 'Test_F1'],
    'Value':  [cv_acc.mean(), cv_acc.std(), cv_f1.mean(), cv_f1.std(), acc, f1]
}).to_excel(OUT_METRICS_XLSX, index=False)
print(f"Saved metrics -> {OUT_METRICS_XLSX}")

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 6), dpi=160)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
            xticklabels=le.classes_, yticklabels=le.classes_, linewidths=0.6, linecolor='gray')
plt.title('Voting Classifier Confusion Matrix (Normalized)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(OUT_CM_PNG, dpi=300)
plt.close()
print(f"Saved confusion matrix -> {OUT_CM_PNG}")

import re
import numpy as np

def _sanitize_series_to_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()

    s = (s
         .str.replace(',', '', regex=False)        
         .str.replace('，', '', regex=False)
         .str.replace('—', '-', regex=False)        
         .str.replace('–', '-', regex=False)
         .str.replace('−', '-', regex=False)
         .str.replace('\u00a0', '', regex=False)  
         .str.replace('∞', '', regex=False)       
    )

    unit_pat = r'(?:\s*(K|°C|℃|J/mol|kJ/mol|eV|W/mK|Pa|GPa))+$'
    s = s.str.replace(unit_pat, '', flags=re.IGNORECASE, regex=True)

    s = s.str.replace(r'[^0-9eE\+\-\.]', '', regex=True)

    out = pd.to_numeric(s, errors='coerce')

    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out

calc_df = pd.read_csv(PRED_FILE)
orig_cols = calc_df.columns.tolist()

calc_X = calc_df.copy()
for col in ['Compounds', 'Compounds_clean', 'p']:
    if col in calc_X.columns:
        calc_X.drop(columns=[col], inplace=True)

missing_features = [c for c in X.columns if c not in calc_X.columns]
for c in missing_features:
    calc_X[c] = np.nan
extra_features = [c for c in calc_X.columns if c not in X.columns]
if extra_features:
    calc_X.drop(columns=extra_features, inplace=True)
calc_X = calc_X[X.columns]  

for col in calc_X.columns:
    calc_X[col] = _sanitize_series_to_numeric(calc_X[col])

feature_medians = X_train.median(numeric_only=True)
calc_X = calc_X.fillna(feature_medians)

nan_after = calc_X.isnull().sum()
if (nan_after > 0).any():
    calc_X = calc_X.fillna(0)
    print("Note: some columns had all-NaN medians; filled remaining NaNs with 0.")

calc_X_out = calc_df.copy()
calc_X_out[X.columns] = calc_X
calc_X_out.to_excel('calculated_properties_cleaned_inputs.xlsx', index=False)
print("Saved cleaned prediction inputs -> calculated_properties_cleaned_inputs.xlsx")

calc_pred_encoded = voting.predict(calc_X)
calc_pred_labels  = le.inverse_transform(calc_pred_encoded)
proba = voting.predict_proba(calc_X)
proba_df = pd.DataFrame(proba, columns=[f'proba_{c}' for c in le.classes_])

out_df = calc_df.copy()
out_df['predicted_p'] = calc_pred_labels
out_df = pd.concat([out_df, proba_df], axis=1)
out_df.to_excel(OUT_PRED_XLSX, index=False)
print(f"Saved predictions -> {OUT_PRED_XLSX}")

print("\nClassification report on holdout test set:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
