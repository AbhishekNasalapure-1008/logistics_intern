import sys, os, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

DATA_PATH = r'E:\Yuva Intern\data\cleaned_logistics_data.csv'
RESULTS_FOLDER = r'E:\Yuva Intern\week4_results'
PROGRESS_FILE = os.path.join(RESULTS_FOLDER, '_progress.txt')
METRICS_JSON = os.path.join(RESULTS_FOLDER, '_metrics.json')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

with open(PROGRESS_FILE, 'w') as f: f.write('start\n')

def checkpoint(msg):
    with open(PROGRESS_FILE, 'a') as f: f.write(msg + '\n')

metrics = {}

# ---- STEP 1: load/verify ----
checkpoint('step1_load')
df = pd.read_csv(DATA_PATH)
metrics['shape_rows'] = int(df.shape[0])
metrics['shape_cols'] = int(df.shape[1])
metrics['missing_total'] = int(df.isnull().sum().sum())
metrics['has_lead_time'] = 'lead_time_days' in df.columns
checkpoint(f"step1_done rows={metrics['shape_rows']} cols={metrics['shape_cols']}")

# ---- STEP 2/3: prep ----
checkpoint('step2_3_prep')
TARGET = 'lead_time_days'
FEATURES_NUMERIC = [v for v in [
    'shipping_costs','traffic_congestion_level','weather_condition_severity',
    'port_congestion_level','supplier_reliability_score','route_risk_level',
    'customs_clearance_time','delay_probability','delivery_time_deviation',
    'historical_demand','fuel_consumption_rate','loading_unloading_time',
    'year','month','day_of_week'] if v in df.columns]
FEATURES_CATEGORICAL = [v for v in ['risk_classification'] if v in df.columns]
ALL_FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL
metrics['n_features'] = len(ALL_FEATURES)
metrics['features_numeric'] = FEATURES_NUMERIC
metrics['features_categorical'] = FEATURES_CATEGORICAL
metrics['target'] = TARGET
metrics['problem_type'] = 'regression'

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()
RANDOM_STATE = 42
TEST_SIZE = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
metrics['train_records'] = int(len(X_train))
metrics['test_records'] = int(len(X_test))

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), FEATURES_NUMERIC),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), FEATURES_CATEGORICAL)
])
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)
ohe = preprocessor.named_transformers_['cat']
cat_names = list(ohe.get_feature_names_out(FEATURES_CATEGORICAL))
FEATURE_NAMES = FEATURES_NUMERIC + cat_names
metrics['feature_names_after_encoding'] = FEATURE_NAMES
metrics['X_train_shape'] = [int(s) for s in X_train_p.shape]
metrics['X_test_shape'] = [int(s) for s in X_test_p.shape]
checkpoint(f"step3_done train={metrics['train_records']} test={metrics['test_records']} feats_enc={len(FEATURE_NAMES)}")

# ---- STEP 4: Linear Regression ----
checkpoint('step4_lr')
lr = LinearRegression()
lr.fit(X_train_p, y_train)
lr_p = lr.predict(X_test_p)
metrics['lr'] = {
    'MAE': float(mean_absolute_error(y_test, lr_p)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, lr_p))),
    'R2': float(r2_score(y_test, lr_p))
}
with open(METRICS_JSON, 'w') as f: json.dump(metrics, f, indent=2, default=str)
checkpoint(f"step4_done RMSE={metrics['lr']['RMSE']:.4f}")

# ---- STEP 5: Decision Tree ----
checkpoint('step5_dt')
dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
dt.fit(X_train_p, y_train)
dt_p = dt.predict(X_test_p)
metrics['dt'] = {
    'MAE': float(mean_absolute_error(y_test, dt_p)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, dt_p))),
    'R2': float(r2_score(y_test, dt_p))
}
with open(METRICS_JSON, 'w') as f: json.dump(metrics, f, indent=2, default=str)
checkpoint(f"step5_done RMSE={metrics['dt']['RMSE']:.4f}")

# ---- STEP 6: Random Forest (smaller for speed) ----
checkpoint('step6_rf')
rf = RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=1)
rf.fit(X_train_p, y_train)
rf_p = rf.predict(X_test_p)
metrics['rf'] = {
    'MAE': float(mean_absolute_error(y_test, rf_p)),
    'RMSE': float(np.sqrt(mean_squared_error(y_test, rf_p))),
    'R2': float(r2_score(y_test, rf_p))
}
rmse_map = {'Linear Regression': metrics['lr']['RMSE'],
            'Decision Tree': metrics['dt']['RMSE'],
            'Random Forest': metrics['rf']['RMSE']}
metrics['best_model_overall'] = min(rmse_map, key=rmse_map.get)
tree_rmse_map = {'Decision Tree': metrics['dt']['RMSE'], 'Random Forest': metrics['rf']['RMSE']}
best_tree_name = min(tree_rmse_map, key=tree_rmse_map.get)
metrics['best_tree_model'] = best_tree_name
default_tree_rmse = tree_rmse_map[best_tree_name]
if best_tree_name == 'Decision Tree':
    default_tree_obj = dt
    default_tree_pred = dt_p
    default_tree_mae = metrics['dt']['MAE']
    default_tree_r2 = metrics['dt']['R2']
else:
    default_tree_obj = rf
    default_tree_pred = rf_p
    default_tree_mae = metrics['rf']['MAE']
    default_tree_r2 = metrics['rf']['R2']
with open(METRICS_JSON, 'w') as f: json.dump(metrics, f, indent=2, default=str)
checkpoint(f"step6_done RF_RMSE={metrics['rf']['RMSE']:.4f} best_tree={best_tree_name} best_overall={metrics['best_model_overall']}")

# ---- STEP 7: Cross-validation (manual KFold loop, 1 fit per fold => fastest) ----
checkpoint('step7_cv')
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
metrics['cv'] = {}
for mk, (model, _Xin) in [
    ('linear_regression', (LinearRegression(), X_train_p)),
    ('decision_tree', (DecisionTreeRegressor(random_state=RANDOM_STATE), X_train_p)),
    ('random_forest', (RandomForestRegressor(n_estimators=50, random_state=RANDOM_STATE, n_jobs=1), X_train_p)),
]:
    fold_mae, fold_rmse, fold_r2 = [], [], []
    for tr_idx, va_idx in kf.split(_Xin, y_train):
        Xtr, Xva = _Xin[tr_idx], _Xin[va_idx]
        ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model.fit(Xtr, ytr)
        pv = model.predict(Xva)
        fold_mae.append(mean_absolute_error(yva, pv))
        fold_rmse.append(np.sqrt(mean_squared_error(yva, pv)))
        fold_r2.append(r2_score(yva, pv))
    fold_mae = np.array(fold_mae); fold_rmse = np.array(fold_rmse); fold_r2 = np.array(fold_r2)
    metrics['cv'][mk] = {
        'cv_mae_mean': float(fold_mae.mean()),
        'cv_mae_std': float(fold_mae.std()),
        'cv_rmse_mean': float(fold_rmse.mean()),
        'cv_rmse_std': float(fold_rmse.std()),
        'cv_r2_mean': float(fold_r2.mean()),
        'cv_r2_std': float(fold_r2.std())
    }
with open(METRICS_JSON, 'w') as f: json.dump(metrics, f, indent=2, default=str)
checkpoint(f"step7_done cv_rf_r2={metrics['cv']['random_forest']['cv_r2_mean']:.4f}")

# ---- STEP 8: Hyperparameter tuning (SMALL grid for speed) ----
checkpoint('step8_tune')
if best_tree_name == 'Decision Tree':
    param_grid = {'max_depth': [5, 10, None],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1, 2]}
    base = DecisionTreeRegressor(random_state=RANDOM_STATE)
else:
    param_grid = {'n_estimators': [50],
                  'max_depth': [10, None],
                  'min_samples_split': [2, 5],
                  'min_samples_leaf': [1],
                  'max_features': ['sqrt']}
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)

grid = GridSearchCV(estimator=base, param_grid=param_grid,
                    scoring='neg_root_mean_squared_error', cv=kf, n_jobs=1, verbose=0)
grid.fit(X_train_p, y_train)
tuned_model = grid.best_estimator_
tuned_p = tuned_model.predict(X_test_p)
tune_rmse = float(np.sqrt(mean_squared_error(y_test, tuned_p)))
metrics['tuning'] = {
    'tuned_model': best_tree_name,
    'best_params': {kk: (vv if vv is not None else 'None') for kk, vv in grid.best_params_.items()},
    'best_cv_rmse': float(-grid.best_score_),
    'MAE': float(mean_absolute_error(y_test, tuned_p)),
    'RMSE': tune_rmse,
    'R2': float(r2_score(y_test, tuned_p)),
    'default_rmse': float(default_tree_rmse),
    'rmse_improvement_pct': float((default_tree_rmse - tune_rmse) / default_tree_rmse * 100)
}
metrics['tuning']['improved'] = bool(metrics['tuning']['RMSE'] < default_tree_rmse)
with open(METRICS_JSON, 'w') as f: json.dump(metrics, f, indent=2, default=str)
checkpoint(f"step8_done tuned_RMSE={tune_rmse:.4f} improved={metrics['tuning']['improved']}")

# ---- STEP 9: Feature Importance ----
checkpoint('step9_fi')
importances = tuned_model.feature_importances_
fi_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': importances}
                    ).sort_values('Importance', ascending=False).reset_index(drop=True)
fi_df.to_csv(os.path.join(RESULTS_FOLDER, 'feature_importance.csv'), index=False)
metrics['top5_features'] = fi_df.head(5).to_dict(orient='records')
metrics['feature_ranks'] = {str(row.Feature): int(row.Index + 1) for row in fi_df.itertuples()}
with open(METRICS_JSON, 'w') as f: json.dump(metrics, f, indent=2, default=str)
checkpoint(f"step9_done top={list(fi_df.head(3)['Feature'])}")

# ---- Final model selection (default tuned whichever is better) ----
if tune_rmse < default_tree_rmse:
    FINAL_MODEL = tuned_model
    final_pred = tuned_p
    final_label = f"Tuned {best_tree_name}"
else:
    FINAL_MODEL = default_tree_obj
    final_pred = default_tree_pred
    final_label = best_tree_name

metrics['final_visualization_model'] = final_label
metrics['final_visualization_rmse'] = float(np.sqrt(mean_squared_error(y_test, final_pred)))

# ---- Comparison CSV ----
comp_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', f"Tuned {best_tree_name}"],
    'MAE': [metrics['lr']['MAE'], metrics['dt']['MAE'], metrics['rf']['MAE'], metrics['tuning']['MAE']],
    'RMSE': [metrics['lr']['RMSE'], metrics['dt']['RMSE'], metrics['rf']['RMSE'], metrics['tuning']['RMSE']],
    'R2': [metrics['lr']['R2'], metrics['dt']['R2'], metrics['rf']['R2'], metrics['tuning']['R2']]
})
comp_df.to_csv(os.path.join(RESULTS_FOLDER, 'model_comparison.csv'), index=False)

# ---- STEP 10: Visualizations ----
checkpoint('step10_viz')
residuals = pd.Series(y_test.values - final_pred)

# 1. actual_vs_predicted.png
plt.figure(figsize=(10, 8))
plt.scatter(y_test.values, final_pred, alpha=0.35, s=20, color='#2E86AB', edgecolors='none')
mn, mx = float(min(y_test.min(), final_pred.min())), float(max(y_test.max(), final_pred.max()))
plt.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Prediction (y=x)')
plt.title(f"Actual vs Predicted Lead Time ({final_label})", fontsize=13, fontweight='bold')
plt.xlabel('Actual Lead Time (days)', fontsize=11)
plt.ylabel('Predicted Lead Time (days)', fontsize=11)
plt.legend(loc='upper left')
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_FOLDER, 'actual_vs_predicted.png'), bbox_inches='tight'); plt.close()

# 2. residual_distribution.png
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=40, kde=True, color='#A23B72')
plt.axvline(float(residuals.mean()), color='red', ls='--', lw=2, label=f"Mean Residual = {float(residuals.mean()):.3f}")
plt.axvline(0, color='black', ls=':', lw=2, label='Zero Error')
plt.title(f"Distribution of Prediction Residuals ({final_label})", fontsize=13, fontweight='bold')
plt.xlabel('Residual (Actual - Predicted) [days]', fontsize=11)
plt.ylabel('Count / Density', fontsize=11)
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_FOLDER, 'residual_distribution.png'), bbox_inches='tight'); plt.close()

# 3. model_comparison.png
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
palette = ['#2E86AB', '#F18F01', '#A23B72', '#2ECC71']
mods = comp_df['Model'].tolist()
for ax, metric in zip(axes, ['MAE', 'RMSE', 'R2']):
    vals = comp_df[metric].tolist()
    bars = ax.bar(mods, vals, color=palette[:len(vals)], edgecolor='white')
    ax.set_title(f"Model Comparison: {metric}", fontsize=11, fontweight='bold')
    ax.set_ylabel(metric, fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.tick_params(axis='x', rotation=25, labelsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_FOLDER, 'model_comparison.png'), bbox_inches='tight'); plt.close()

# 4. feature_importance.png
fi_plot = fi_df.head(min(15, len(fi_df))).iloc[::-1]
plt.figure(figsize=(11, 8))
plt.barh(fi_plot['Feature'], fi_plot['Importance'], color='#2E86AB', edgecolor='white')
plt.title(f"Feature Importance (Top {len(fi_plot)})", fontsize=13, fontweight='bold')
plt.xlabel('Importance (mean impurity decrease)', fontsize=11)
plt.ylabel('Feature', fontsize=11)
for i, v in enumerate(fi_plot['Importance'].values):
    plt.text(v, i, f" {v:.4f}", va='center', fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(RESULTS_FOLDER, 'feature_importance.png'), bbox_inches='tight'); plt.close()

# 5. residual_vs_predicted.png
plt.figure(figsize=(10, 8))
plt.scatter(final_pred, residuals.values, alpha=0.35, s=20, color='#C73E1D', edgecolors='none')
plt.axhline(0, color='red', ls='--', lw=2, label='Zero Residual Line')
plt.title(f"Residuals vs Predicted Lead Time ({final_label})", fontsize=13, fontweight='bold')
plt.xlabel('Predicted Lead Time (days)', fontsize=11)
plt.ylabel('Residual (Actual - Predicted) [days]', fontsize=11)
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_FOLDER, 'residual_vs_predicted.png'), bbox_inches='tight'); plt.close()

# 6. actual_predicted_distribution.png
plt.figure(figsize=(11, 7))
sns.histplot(y_test, bins=30, kde=True, color='#2E86AB', label='Actual Lead Time', alpha=0.5, stat='density')
sns.histplot(final_pred, bins=30, kde=True, color='#A23B72', label='Predicted Lead Time', alpha=0.5, stat='density')
plt.title(f"Distribution Comparison: Actual vs Predicted Lead Time ({final_label})", fontsize=12, fontweight='bold')
plt.xlabel('Lead Time (days)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.legend(fontsize=10); plt.tight_layout(); plt.savefig(os.path.join(RESULTS_FOLDER, 'actual_predicted_distribution.png'), bbox_inches='tight'); plt.close()

checkpoint('step10_viz_done')
with open(METRICS_JSON, 'w') as f:
    json.dump(metrics, f, indent=2, default=str)

# ---- Compact printed summary ----
print('=== WEEK 4 PREDICTIVE MODELING - RESULTS ===')
print(f"Dataset: {metrics['shape_rows']} rows x {metrics['shape_cols']} cols (missing={metrics['missing_total']})")
print(f"Problem: {metrics['problem_type']} | Target: {TARGET} | Features: {metrics['n_features']}")
print(f"Train/Test split: {metrics['train_records']} / {metrics['test_records']} (80/20, random_state=42)")
print(f"Post-encoding feature count: {len(FEATURE_NAMES)}")
print()
for k, label in [('lr','Linear Regression'), ('dt','Decision Tree'), ('rf','Random Forest')]:
    m = metrics[k]
    print(f"{label:22s}: MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")
t = metrics['tuning']
print(f"Tuned {t['tuned_model']:16s}: MAE={t['MAE']:.4f}  RMSE={t['RMSE']:.4f}  R2={t['R2']:.4f} (improved={t['improved']}, {t['rmse_improvement_pct']:+.2f}%)")
print(f"  Best params: {t['best_params']}")
print()
print('--- 5-Fold CV (shuffle=True, rs=42) on TRAIN set ---')
for k, label in [('linear_regression','Linear Regression'),
                 ('decision_tree','Decision Tree'),
                 ('random_forest','Random Forest')]:
    cv = metrics['cv'][k]
    print(f"CV {label:20s}: RMSE={cv['cv_rmse_mean']:.4f} ± {cv['cv_rmse_std']:.4f}  R2={cv['cv_r2_mean']:.4f} ± {cv['cv_r2_std']:.4f}")
print()
print('--- Top-5 Feature Importance ---')
for i, row in fi_df.head(5).iterrows():
    print(f"  {i+1}. {str(row['Feature']):35s}  importance = {row['Importance']:.4f}")
print()
print(f"Best overall model: {metrics['best_model_overall']}")
print(f"Model used in final plots: {final_label} (RMSE = {metrics['final_visualization_rmse']:.4f})")
num_png = len([f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.png')])
print(f"PNG visualizations saved: {num_png}")
print(f"Outputs in: {RESULTS_FOLDER}")

with open(PROGRESS_FILE, 'a') as f: f.write('COMPLETE\n')
with open(METRICS_JSON, 'w') as f:
    json.dump(metrics, f, indent=2, default=str)
