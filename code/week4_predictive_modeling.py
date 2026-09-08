import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

DATA_PATH = r'E:\Yuva Intern\data\cleaned_logistics_data.csv'
RESULTS_FOLDER = r'E:\Yuva Intern\week4_results'
SCRIPT_PATH = r'E:\Yuva Intern\code\week4_predictive_modeling.py'

os.makedirs(RESULTS_FOLDER, exist_ok=True)

print('='*90)
print('WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS SYSTEMS')
print('Project Title: Logistics Delivery Time Analysis and Prediction Using Data Analytics and Machine Learning')
print('='*90)

# ================================================================
# STEP 1: LOAD AND VERIFY THE DATASET
# ================================================================
print('\n' + '-'*90)
print('STEP 1: LOAD AND VERIFY THE DATASET')
print('-'*90)

df = pd.read_csv(DATA_PATH)

print(f'\n1. Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns')

print('\n2. All Column Names:')
for i, col in enumerate(df.columns, 1):
    print(f'   {i:2d}. {col}')

print('\n3. Data Types:')
print(df.dtypes.to_string())

print('\n4. First 5 Rows:')
print(df.head().to_string())

print('\n5. Missing Values Check:')
missing = df.isnull().sum()
if missing.sum() == 0:
    print('   No missing values found in any column.')
else:
    print(missing[missing > 0].to_string())

print('\n6. Confirming lead_time_days exists:')
print(f'   lead_time_days PRESENT: {"lead_time_days" in df.columns}')

print('\n7. Summary Statistics:')
with pd.option_context('display.max_columns', None, 'display.width', 200):
    print(df.describe().to_string())

# ================================================================
# STEP 2: DEFINE THE PREDICTION PROBLEM + TARGET LEAKAGE CHECK
# ================================================================
print('\n' + '-'*90)
print('STEP 2: DEFINE THE PREDICTION PROBLEM')
print('-'*90)

TARGET = 'lead_time_days'
print(f'\nPrediction Objective: Predict {TARGET} (continuous numerical target)')
print(f'This is a REGRESSION problem because {TARGET} takes continuous values (days).')

print('\n--- Target Leakage Analysis ---')
print('Evaluating each variable for target leakage (i.e., directly derived from lead_time_days')
print('or contains information only available AFTER delivery occurs):')

candidate_leakage_vars = [
    'order_fulfillment_status',
    'vehicle_gps_latitude',
    'vehicle_gps_longitude',
    'iot_temperature',
    'cargo_condition_status',
    'driver_behavior_score',
    'fatigue_monitoring_score',
    'disruption_likelihood_score',
    'handling_equipment_availability',
    'warehouse_inventory_level',
    'eta_variation_hours',
]

excluded_leakage = []
for v in candidate_leakage_vars:
    if v in df.columns:
        print(f'   EXCLUDED (post-delivery / in-transit sensor): {v}')
        excluded_leakage.append(v)

print('\n   Variables NOT excluded as leakage (available pre-shipment or at dispatch):')
preferred_features = [
    'shipping_costs',
    'traffic_congestion_level',
    'weather_condition_severity',
    'port_congestion_level',
    'supplier_reliability_score',
    'route_risk_level',
    'customs_clearance_time',
    'delay_probability',
    'delivery_time_deviation',
    'historical_demand',
    'fuel_consumption_rate',
    'loading_unloading_time',
    'risk_classification',
    'year',
    'month',
    'day_of_week',
]

# Keep only those that exist
FEATURES_NUMERIC = [v for v in preferred_features if v in df.columns and v != 'risk_classification']
FEATURES_CATEGORICAL = [v for v in ['risk_classification'] if v in df.columns]
ALL_FEATURES = FEATURES_NUMERIC + FEATURES_CATEGORICAL

for f in ALL_FEATURES:
    print(f'   INCLUDED: {f}')

print(f'\nTotal features selected: {len(ALL_FEATURES)} ({len(FEATURES_NUMERIC)} numeric, {len(FEATURES_CATEGORICAL)} categorical)')

# ================================================================
# STEP 3: DATA PREPARATION
# ================================================================
print('\n' + '-'*90)
print('STEP 3: DATA PREPARATION')
print('-'*90)

X = df[ALL_FEATURES].copy()
y = df[TARGET].copy()

print(f'\nMissing values in features: {X.isnull().sum().sum()}')
print(f'Missing values in target: {y.isnull().sum()}')

RANDOM_STATE = 42
TEST_SIZE = 0.20

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f'\nTrain-Test Split: {100*(1-TEST_SIZE):.0f}% train / {100*TEST_SIZE:.0f}% test')
print(f'   Training records: {len(X_train)}')
print(f'   Testing records:  {len(X_test)}')
print(f'   Features used:    {len(ALL_FEATURES)}')

# Build preprocessor: scale numeric, one-hot encode categorical
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, FEATURES_NUMERIC),
        ('cat', categorical_transformer, FEATURES_CATEGORICAL)
    ]
)

# Apply preprocessing once to get processed arrays (for models that need scaled data, Linear)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after one-hot
ohe = preprocessor.named_transformers_['cat']
cat_feat_names = list(ohe.get_feature_names_out(FEATURES_CATEGORICAL))
FEATURE_NAMES = FEATURES_NUMERIC + cat_feat_names
print(f'   Feature names after encoding: {len(FEATURE_NAMES)}')
print(f'   X_train_processed shape: {X_train_processed.shape}')
print(f'   X_test_processed shape:  {X_test_processed.shape}')

# ================================================================
# STEP 4: BASELINE MODEL - LINEAR REGRESSION
# ================================================================
print('\n' + '-'*90)
print('STEP 4: BASELINE MODEL - LINEAR REGRESSION')
print('-'*90)

print('\nLinear Regression is used as a simple interpretable baseline:')
print('it models a linear weighted sum of all input features.')

lr = LinearRegression()
lr.fit(X_train_processed, y_train)
lr_pred = lr.predict(X_test_processed)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print(f'\n   Linear Regression Test Metrics:')
print(f'   MAE  = {lr_mae:.4f} days')
print(f'   RMSE = {lr_rmse:.4f} days')
print(f'   R2   = {lr_r2:.4f}')

# ================================================================
# STEP 5: DECISION TREE REGRESSOR
# ================================================================
print('\n' + '-'*90)
print('STEP 5: DECISION TREE REGRESSOR')
print('-'*90)

dt = DecisionTreeRegressor(random_state=RANDOM_STATE)
dt.fit(X_train_processed, y_train)
dt_pred = dt.predict(X_test_processed)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

print(f'\n   Decision Tree (default params) Test Metrics:')
print(f'   MAE  = {dt_mae:.4f} days')
print(f'   RMSE = {dt_rmse:.4f} days')
print(f'   R2   = {dt_r2:.4f}')

print(f'\n   Comparison vs Linear Regression Baseline:')
print(f'   MAE  change: {dt_mae - lr_mae:+.4f} days ({"better" if dt_mae < lr_mae else "worse"})')
print(f'   RMSE change: {dt_rmse - lr_rmse:+.4f} days ({"better" if dt_rmse < lr_rmse else "worse"})')
print(f'   R2   change: {dt_r2 - lr_r2:+.4f} ({"better" if dt_r2 > lr_r2 else "worse"})')

# ================================================================
# STEP 6: RANDOM FOREST REGRESSOR
# ================================================================
print('\n' + '-'*90)
print('STEP 6: RANDOM FOREST REGRESSOR')
print('-'*90)

rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_processed, y_train)
rf_pred = rf.predict(X_test_processed)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print(f'\n   Random Forest (n_estimators=100, default) Test Metrics:')
print(f'   MAE  = {rf_mae:.4f} days')
print(f'   RMSE = {rf_rmse:.4f} days')
print(f'   R2   = {rf_r2:.4f}')

models = ['Linear Regression', 'Decision Tree', 'Random Forest']
maes = [lr_mae, dt_mae, rf_mae]
rmses = [lr_rmse, dt_rmse, rf_rmse]
r2s = [lr_r2, dt_r2, rf_r2]

comparison_df = pd.DataFrame({
    'Model': models,
    'MAE': maes,
    'RMSE': rmses,
    'R2': r2s
})

print(f'\n=== Model Comparison Table (Test Set) ===')
print(comparison_df.to_string(index=False))

# Pick best by RMSE (most interpretable and commonly used for regression)
best_idx = np.argmin(rmses)
BEST_MODEL_NAME = models[best_idx]
print(f'\nBest model (lowest RMSE): {BEST_MODEL_NAME}')

# Identify best tree-based model (for Steps 8, 9)
tree_models_rmse = {'Decision Tree': dt_rmse, 'Random Forest': rf_rmse}
best_tree_name = min(tree_models_rmse, key=tree_models_rmse.get)
if best_tree_name == 'Decision Tree':
    best_tree_model_default = dt
    best_tree_pred_default = dt_pred
    best_tree_mae_default, best_tree_rmse_default, best_tree_r2_default = dt_mae, dt_rmse, dt_r2
else:
    best_tree_model_default = rf
    best_tree_pred_default = rf_pred
    best_tree_mae_default, best_tree_rmse_default, best_tree_r2_default = rf_mae, rf_rmse, rf_r2

print(f'Best tree-based model (for tuning + feature importance): {best_tree_name}')

# ================================================================
# STEP 7: CROSS-VALIDATION
# ================================================================
print('\n' + '-'*90)
print('STEP 7: K-FOLD CROSS-VALIDATION')
print('-'*90)

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print(f'\nUsing KFold with n_splits=5, shuffle=True, random_state={RANDOM_STATE}')
print('Cross-validation assesses model stability across different train/validation splits.')

for m_name, m_obj in zip(models, [lr, dt, rf]):
    cv_neg_mae = cross_val_score(m_obj, X_train_processed, y_train, cv=kf,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_neg_mse = cross_val_score(m_obj, X_train_processed, y_train, cv=kf,
                                  scoring='neg_mean_squared_error', n_jobs=-1)
    cv_r2 = cross_val_score(m_obj, X_train_processed, y_train, cv=kf,
                             scoring='r2', n_jobs=-1)
    cv_mae_mean = -cv_neg_mae.mean()
    cv_mae_std = cv_neg_mae.std()
    cv_rmse_mean = np.sqrt(-cv_neg_mse.mean())
    cv_rmse_std = np.sqrt(cv_neg_mse.std())
    cv_r2_mean = cv_r2.mean()
    cv_r2_std = cv_r2.std()
    print(f'\n   {m_name} CV (training set, 5-fold):')
    print(f'   CV MAE  = {cv_mae_mean:.4f} +/- {cv_mae_std:.4f} days')
    print(f'   CV RMSE = {cv_rmse_mean:.4f} +/- {cv_rmse_std:.4f} days')
    print(f'   CV R2   = {cv_r2_mean:.4f} +/- {cv_r2_std:.4f}')

# ================================================================
# STEP 8: HYPERPARAMETER TUNING
# ================================================================
print('\n' + '-'*90)
print('STEP 8: HYPERPARAMETER TUNING (GridSearchCV)')
print('-'*90)

if best_tree_name == 'Decision Tree':
    param_grid = {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    base = DecisionTreeRegressor(random_state=RANDOM_STATE)
    print(f'Tuning Decision Tree with grid: {param_grid}')
else:
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    print(f'Tuning Random Forest with grid: {param_grid}')

grid = GridSearchCV(
    estimator=base,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=kf,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_processed, y_train)
tuned_model = grid.best_estimator_
print(f'\nBest parameters found: {grid.best_params_}')
print(f'Best CV RMSE: {-grid.best_score_:.4f} days')

tuned_pred = tuned_model.predict(X_test_processed)
tuned_mae = mean_absolute_error(y_test, tuned_pred)
tuned_rmse = np.sqrt(mean_squared_error(y_test, tuned_pred))
tuned_r2 = r2_score(y_test, tuned_pred)

print(f'\nTuned {best_tree_name} Test Metrics (after GridSearchCV):')
print(f'   MAE  = {tuned_mae:.4f} days')
print(f'   RMSE = {tuned_rmse:.4f} days')
print(f'   R2   = {tuned_r2:.4f}')

print(f'\nBefore vs After Tuning Comparison:')
print(f'   MAE:  {best_tree_mae_default:.4f} -> {tuned_mae:.4f} ({tuned_mae - best_tree_mae_default:+.4f})')
print(f'   RMSE: {best_tree_rmse_default:.4f} -> {tuned_rmse:.4f} ({tuned_rmse - best_tree_rmse_default:+.4f})')
print(f'   R2:   {best_tree_r2_default:.4f} -> {tuned_r2:.4f} ({tuned_r2 - best_tree_r2_default:+.4f})')

improvement = (best_tree_rmse_default - tuned_rmse) / best_tree_rmse_default * 100
print(f'\nTuning improvement (RMSE reduction): {improvement:+.2f}%')
if improvement > 0:
    print('Result: Tuning IMPROVED performance.')
else:
    print('Result: Tuning DID NOT improve performance (or default was already optimal).')

# Add tuned model to comparison (as a separate row)
comparison_df_full = pd.DataFrame({
    'Model': models + [f'Tuned {best_tree_name}'],
    'MAE': [lr_mae, dt_mae, rf_mae, tuned_mae],
    'RMSE': [lr_rmse, dt_rmse, rf_rmse, tuned_rmse],
    'R2': [lr_r2, dt_r2, rf_r2, tuned_r2]
})

# ================================================================
# STEP 9: FEATURE IMPORTANCE
# ================================================================
print('\n' + '-'*90)
print('STEP 9: FEATURE IMPORTANCE (Best Tree-Based Model)')
print('-'*90)

importances = tuned_model.feature_importances_
fi_df = pd.DataFrame({
    'Feature': FEATURE_NAMES,
    'Importance': importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print('\nFeature Importance (ranked, most to least important):')
for i, row in fi_df.iterrows():
    print(f'   {i+1:2d}. {row["Feature"]:35s} = {row["Importance"]:.4f}')

print('\nImportant note: Feature importance reflects contribution to the model,')
print('NOT causal relationships with lead_time_days.')

# ================================================================
# STEP 10 + 11: VISUALIZATIONS + SAVE RESULTS
# ================================================================
print('\n' + '-'*90)
print('STEP 10 + 11: MODEL VISUALIZATIONS AND SAVING RESULTS')
print('-'*90)

# Use tuned model (or default if tuned is worse) for final viz
if tuned_rmse < best_tree_rmse_default:
    FINAL_MODEL = tuned_model
    final_pred = tuned_pred
    final_model_label = f'Tuned {best_tree_name}'
else:
    FINAL_MODEL = best_tree_model_default
    final_pred = best_tree_pred_default
    final_model_label = best_tree_name

print(f'\nVisualizations will use: {final_model_label}')

# --- 1. actual_vs_predicted.png
print('\nSaving 1. actual_vs_predicted.png')
residuals = y_test - final_pred
plt.figure(figsize=(10, 8))
plt.scatter(y_test, final_pred, alpha=0.35, s=20, color='#2E86AB', edgecolors='none')
min_val = min(y_test.min(), final_pred.min())
max_val = max(y_test.max(), final_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction (y = x)')
plt.title(f'Actual vs Predicted Lead Time ({final_model_label})', fontsize=13, fontweight='bold')
plt.xlabel('Actual Lead Time (days)', fontsize=11)
plt.ylabel('Predicted Lead Time (days)', fontsize=11)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'actual_vs_predicted.png'), bbox_inches='tight')
plt.close()

# --- 2. residual_distribution.png
print('Saving 2. residual_distribution.png')
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=40, kde=True, color='#A23B72')
plt.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean Residual = {residuals.mean():.3f}')
plt.axvline(0, color='black', linestyle=':', linewidth=2, label='Zero Error')
plt.title(f'Distribution of Prediction Residuals ({final_model_label})', fontsize=13, fontweight='bold')
plt.xlabel('Residual (Actual - Predicted) [days]', fontsize=11)
plt.ylabel('Count / Density', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'residual_distribution.png'), bbox_inches='tight')
plt.close()

# --- 3. model_comparison.png
print('Saving 3. model_comparison.png')
metrics = ['MAE', 'RMSE', 'R2']
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
palette = ['#2E86AB', '#F18F01', '#A23B72', '#2ECC71']
model_labels_plot = comparison_df_full['Model'].tolist()
for ax, metric in zip(axes, metrics):
    vals = comparison_df_full[metric].tolist()
    bars = ax.bar(model_labels_plot, vals, color=palette[:len(vals)], edgecolor='white')
    ax.set_title(f'Model Comparison: {metric}', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(metric, fontsize=10)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{v:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.tick_params(axis='x', rotation=25, labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'model_comparison.png'), bbox_inches='tight')
plt.close()

# --- 4. feature_importance.png
print('Saving 4. feature_importance.png')
fi_plot = fi_df.head(15).iloc[::-1]
plt.figure(figsize=(11, 8))
plt.barh(fi_plot['Feature'], fi_plot['Importance'], color='#2E86AB', edgecolor='white')
plt.title(f'Feature Importance (Top {min(15, len(fi_df))})', fontsize=13, fontweight='bold')
plt.xlabel('Importance (mean impurity decrease)', fontsize=11)
plt.ylabel('Feature', fontsize=11)
for i, v in enumerate(fi_plot['Importance']):
    plt.text(v, i, f' {v:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'feature_importance.png'), bbox_inches='tight')
plt.close()

# --- 5. residual_vs_predicted.png
print('Saving 5. residual_vs_predicted.png')
plt.figure(figsize=(10, 8))
plt.scatter(final_pred, residuals, alpha=0.35, s=20, color='#C73E1D', edgecolors='none')
plt.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero Residual Line')
plt.title(f'Residuals vs Predicted Lead Time ({final_model_label})', fontsize=13, fontweight='bold')
plt.xlabel('Predicted Lead Time (days)', fontsize=11)
plt.ylabel('Residual (Actual - Predicted) [days]', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'residual_vs_predicted.png'), bbox_inches='tight')
plt.close()

# --- 6. actual_predicted_distribution.png
print('Saving 6. actual_predicted_distribution.png (optional)')
plt.figure(figsize=(11, 7))
sns.histplot(y_test, bins=30, kde=True, color='#2E86AB', label='Actual Lead Time', alpha=0.5, stat='density')
sns.histplot(final_pred, bins=30, kde=True, color='#A23B72', label='Predicted Lead Time', alpha=0.5, stat='density')
plt.title(f'Distribution Comparison: Actual vs Predicted Lead Time ({final_model_label})', fontsize=12, fontweight='bold')
plt.xlabel('Lead Time (days)', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_FOLDER, 'actual_predicted_distribution.png'), bbox_inches='tight')
plt.close()

# --- Save CSVs
print('\nSaving model_comparison.csv')
comparison_df_full.to_csv(os.path.join(RESULTS_FOLDER, 'model_comparison.csv'), index=False)

print('Saving feature_importance.csv')
fi_df.to_csv(os.path.join(RESULTS_FOLDER, 'feature_importance.csv'), index=False)

num_viz = len([f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.png')])
print(f'\nTotal PNG visualizations saved: {num_viz}')
print(f'All results saved to: {RESULTS_FOLDER}')

# ================================================================
# STEP 12: MODEL EVALUATION SUMMARY (printed, used for final report)
# ================================================================
print('\n' + '-'*90)
print('STEP 12: MODEL EVALUATION AND INTERPRETATION SUMMARY')
print('-'*90)

print('\nQ1. Which model performed best?')
rmse_all = {
    'Linear Regression': lr_rmse,
    'Decision Tree': dt_rmse,
    'Random Forest': rf_rmse,
    f'Tuned {best_tree_name}': tuned_rmse
}
overall_best = min(rmse_all, key=rmse_all.get)
print(f'   Lowest test-set RMSE => {overall_best} (RMSE = {rmse_all[overall_best]:.4f} days)')

print('\nQ2/Q3/Q4. Best model metrics (MAE / RMSE / R2):')
final_best_mae = comparison_df_full.loc[comparison_df_full['Model'] == overall_best, 'MAE'].values[0]
final_best_rmse = comparison_df_full.loc[comparison_df_full['Model'] == overall_best, 'RMSE'].values[0]
final_best_r2 = comparison_df_full.loc[comparison_df_full['Model'] == overall_best, 'R2'].values[0]
print(f'   MAE  = {final_best_mae:.4f} days')
print(f'   RMSE = {final_best_rmse:.4f} days')
print(f'   R2   = {final_best_r2:.4f}')

print('\nQ5. How does it compare with the others?')
print(comparison_df_full.sort_values('RMSE').to_string(index=False))

print('\nQ6. Did cross-validation indicate stable performance?')
print('   CV mean vs single split R2 should be similar; low std => stable.')
print('   (Actual CV mean/std values printed under Step 7.)')

print('\nQ7. Did hyperparameter tuning improve the model?')
print(f'   Default {best_tree_name} RMSE: {best_tree_rmse_default:.4f}')
print(f'   Tuned   {best_tree_name} RMSE: {tuned_rmse:.4f}')
print(f'   Change: {tuned_rmse - best_tree_rmse_default:+.4f} days ({improvement:+.2f}%)')
if tuned_rmse < best_tree_rmse_default:
    print('   Answer: YES, tuning produced a measurable improvement.')
else:
    print('   Answer: NO, tuning did not improve over the default configuration.')

print('\nQ8. What are the most important features?')
for i, row in fi_df.head(5).iterrows():
    print(f'   {i+1}. {row["Feature"]:35s} importance = {row["Importance"]:.4f}')

# ================================================================
# STEP 13: LOGISTICS OPTIMIZATION STRATEGIES
# ================================================================
print('\n' + '-'*90)
print('STEP 13: LOGISTICS OPTIMIZATION STRATEGIES (based on actual results)')
print('-'*90)

top5_features = fi_df.head(5)['Feature'].tolist()
print('\nStrategy 1: Prioritize monitoring and intervention on Top-5 driving features')
print(f'Finding from model/data:')
print(f'   Top-5 features by importance: {", ".join(top5_features)}')
print(f'Possible operational action:')
print(f'   Build a real-time dashboard that tracks these top-5 features per shipment.')
print(f'Expected logistics benefit:')
print(f'   Early warning when key drivers take adverse values, allowing proactive intervention.')

print('\nStrategy 2: Port congestion mitigation')
print(f'Finding from model/data:')
print(f'   port_congestion_level importance rank = {(fi_df[fi_df.Feature=="port_congestion_level"].index[0]+1) if "port_congestion_level" in fi_df.Feature.values else "N/A"}')
print(f'Possible operational action:')
print(f'   Negotiate priority berth windows or off-peak dispatch slots for high-priority cargo.')
print(f'Expected logistics benefit:')
print(f'   Potential smoothing of lead-time variance attributable to port bottlenecks.')

print('\nStrategy 3: Supplier reliability tiering')
print(f'Finding from model/data:')
print(f'   supplier_reliability_score importance rank = {(fi_df[fi_df.Feature=="supplier_reliability_score"].index[0]+1) if "supplier_reliability_score" in fi_df.Feature.values else "N/A"}')
print(f'Possible operational action:')
print(f'   Classify suppliers into A/B/C tiers; prefer A-tier vendors for express shipments.')
print(f'Expected logistics benefit:')
print(f'   Consistent lead-time reduction on the supplier-selection margin.')

print('\nStrategy 4: Customs pre-clearance programs')
print(f'Finding from model/data:')
print(f'   customs_clearance_time importance rank = {(fi_df[fi_df.Feature=="customs_clearance_time"].index[0]+1) if "customs_clearance_time" in fi_df.Feature.values else "N/A"}')
print(f'Possible operational action:')
print(f'   Enroll in Authorized Economic Operator (AEO) / fast-clearance customs programs.')
print(f'Expected logistics benefit:')
print(f'   Reduce clearance-time variability and its downstream effect on lead time.')

print('\nStrategy 5: Congestion-aware routing')
print(f'Finding from model/data:')
print(f'   traffic_congestion_level importance rank = {(fi_df[fi_df.Feature=="traffic_congestion_level"].index[0]+1) if "traffic_congestion_level" in fi_df.Feature.values else "N/A"}')
print(f'Possible operational action:')
print(f'   Integrate real-time traffic feeds into TMS routing; re-route high-value shipments dynamically.')
print(f'Expected logistics benefit:')
print(f'   Dampen traffic-congestion-driven delays on the last-mile and line-haul segments.')

print('\nStrategy 6: Cost-to-service tier mapping')
print(f'Finding from model/data:')
print(f'   shipping_costs importance rank = {(fi_df[fi_df.Feature=="shipping_costs"].index[0]+1) if "shipping_costs" in fi_df.Feature.values else "N/A"}')
print(f'Possible operational action:')
print(f'   Audit shipping cost tiers vs actual speed; de-list tiers that do not deliver lead-time reduction.')
print(f'Expected logistics benefit:')
print(f'   Better cost recovery per service day; reduced waste on ineffective premium options.')

print('\nStrategy 7: High-risk shipment expediting')
print(f'Finding from model/data:')
print(f'   delay_probability importance rank = {(fi_df[fi_df.Feature=="delay_probability"].index[0]+1) if "delay_probability" in fi_df.Feature.values else "N/A"}')
print(f'   risk_classification (one-hot columns) appear in model.')
print(f'Possible operational action:')
print(f'   Activate expedited handling automatically whenever delay_probability > pre-set threshold OR risk_classification = High Risk.')
print(f'Expected logistics benefit:')
print(f'   Contain tail-risk delays before they cascade into SLA breaches.')

print('\nStrategy 8: Seasonal demand and staffing (month / day_of_week)')
print(f'Finding from model/data:')
print(f'   Temporal features (month, day_of_week) are included; Week 3 EDA showed Aug/Sep peak, Oct trough.')
print(f'Possible operational action:')
print(f'   Pre-position fleet + labour capacity in Q3 to match the August-September elevated lead times.')
print(f'Expected logistics benefit:')
print(f'   Smoother operational throughput during historically tight months, reducing peak lead-time spikes.')

print('\n' + '='*90)
print('WEEK 4 PREDICTIVE MODELING COMPLETE')
print(f'Results folder: {RESULTS_FOLDER}')
print(f'Script path:    {SCRIPT_PATH}')
print('='*90)
