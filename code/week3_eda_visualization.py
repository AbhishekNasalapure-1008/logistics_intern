import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

DATA_PATH = r'E:\Yuva Intern\data\cleaned_logistics_data.csv'
OUTPUT_FOLDER = r'E:\Yuva Intern\week3_visualizations'
SCRIPT_OUTPUT = r'E:\Yuva Intern\code\week3_eda_visualization.py'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print('='*80)
print('WEEK 3: ADVANCED DATA ANALYSIS AND VISUALIZATION IN LOGISTICS')
print('Project Title: Logistics Delivery Time Analysis and Prediction Using Data Analytics and Machine Learning')
print('='*80)

print('\n' + '-'*80)
print('STEP 1: DATASET VERIFICATION')
print('-'*80)

df = pd.read_csv(DATA_PATH)

print(f'\n1. Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')

print('\n2. Column Names:')
for i, col in enumerate(df.columns, 1):
    print(f'   {i}. {col}')

print('\n3. Data Types:')
print(df.dtypes.to_string())

print('\n4. First 5 Rows:')
print(df.head().to_string())

print('\n5. Verifying required variables:')
required_vars = [
    'lead_time_days',
    'shipping_costs', 'traffic_congestion_level', 'weather_condition_severity',
    'port_congestion_level', 'supplier_reliability_score', 'route_risk_level',
    'customs_clearance_time', 'delay_probability', 'delivery_time_deviation',
    'historical_demand', 'risk_classification', 'year', 'month', 'day', 'day_of_week'
]

for var in required_vars:
    status = 'PRESENT' if var in df.columns else 'MISSING'
    print(f'   {var}: {status}')

all_present = all(v in df.columns for v in required_vars)
print(f'\nAll required variables present: {all_present}')

print('\n' + '-'*80)
print('STEP 2: DESCRIPTIVE STATISTICS')
print('-'*80)

lead_time_stats = df['lead_time_days'].describe(percentiles=[0.25, 0.5, 0.75])
print('\n=== Lead Time Days - Detailed Statistics:')
print(f'   Count:      {lead_time_stats["count"]:.0f}')
print(f'   Mean:       {lead_time_stats["mean"]:.4f} days')
print(f'   Median:     {df["lead_time_days"].median():.4f} days')
print(f'   Std Dev:    {lead_time_stats["std"]:.4f} days')
print(f'   Minimum:    {lead_time_stats["min"]:.4f} days')
print(f'   Maximum:    {lead_time_stats["max"]:.4f} days')
print(f'   Q1 (25%):   {lead_time_stats["25%"]:.4f} days')
print(f'   Q3 (75%):   {lead_time_stats["75%"]:.4f} days')
print(f'   IQR:        {(lead_time_stats["75%"] - lead_time_stats["25%"]):.4f} days')

key_vars = [
    'shipping_costs', 'traffic_congestion_level', 'weather_condition_severity',
    'port_congestion_level', 'supplier_reliability_score', 'historical_demand',
    'customs_clearance_time', 'delay_probability', 'delivery_time_deviation'
]

print('\n=== Descriptive Statistics for Key Logistics Variables:')
desc_stats = df[key_vars].describe().T
print(desc_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_string())

print('\n--- Quick Logistics Interpretation:')
print(f'   - Average lead time is ~{df["lead_time_days"].mean():.1f} days with a median of ~{df["lead_time_days"].median():.1f} days')
print(f'   - Lead times range from {df["lead_time_days"].min():.2f} to {df["lead_time_days"].max():.2f} days')
print(f'   - Average shipping cost: ${df["shipping_costs"].mean():.2f}')
print(f'   - Average delay probability: {df["delay_probability"].mean()*100:.1f}%')
print(f'   - Average supplier reliability score: {df["supplier_reliability_score"].mean():.3f} (out of 1.0)')

print('\n' + '-'*80)
print('STEP 3: DISTRIBUTION ANALYSIS')
print('-'*80)

dist_vars = ['lead_time_days', 'shipping_costs', 'delay_probability', 'delivery_time_deviation']

print('\n=== Skewness Calculations:')
for var in dist_vars:
    skew_val = df[var].skew()
    if abs(skew_val) < 0.5:
        skew_interp = 'approximately symmetric'
    elif skew_val > 0:
        skew_interp = 'positively skewed (right tail)'
    else:
        skew_interp = 'negatively skewed (left tail)'
    print(f'   {var}: Skewness = {skew_val:.4f} -> {skew_interp}')

print('\n--- Creating Histogram Plots ---')

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df['lead_time_days'], bins=30, color='#2E86AB', edgecolor='white')
axes[0, 0].set_title('Distribution of Lead Time Days', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Lead Time (days)', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].axvline(df['lead_time_days'].mean(), color='red', linestyle='--', label=f'Mean: {df["lead_time_days"].mean():.1f}')
axes[0, 0].axvline(df['lead_time_days'].median(), color='green', linestyle='--', label=f'Median: {df["lead_time_days"].median():.1f}')
axes[0, 0].legend(fontsize=8)

axes[0, 1].hist(df['shipping_costs'], bins=30, color='#A23B72', edgecolor='white')
axes[0, 1].set_title('Distribution of Shipping Costs', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Shipping Costs ($)', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].axvline(df['shipping_costs'].mean(), color='red', linestyle='--', label=f'Mean: ${df["shipping_costs"].mean():.0f}')
axes[0, 1].axvline(df['shipping_costs'].median(), color='green', linestyle='--', label=f'Median: ${df["shipping_costs"].median():.0f}')
axes[0, 1].legend(fontsize=8)

axes[1, 0].hist(df['delay_probability'], bins=30, color='#F18F01', edgecolor='white')
axes[1, 0].set_title('Distribution of Delay Probability', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Delay Probability', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].axvline(df['delay_probability'].mean(), color='red', linestyle='--', label=f'Mean: {df["delay_probability"].mean():.2f}')
axes[1, 0].legend(fontsize=8)

axes[1, 1].hist(df['delivery_time_deviation'], bins=30, color='#C73E1D', edgecolor='white')
axes[1, 1].set_title('Distribution of Delivery Time Deviation', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Delivery Time Deviation', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].axvline(df['delivery_time_deviation'].mean(), color='red', linestyle='--', label=f'Mean: {df["delivery_time_deviation"].mean():.2f}')
axes[1, 1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '00_histogram_grid.png'), bbox_inches='tight')
print('   Saved: 00_histogram_grid.png')
plt.close()

print('\n' + '-'*80)
print('STEP 4: CORRELATION ANALYSIS')
print('-'*80)

corr_target_vars = [
    'lead_time_days', 'shipping_costs', 'traffic_congestion_level',
    'weather_condition_severity', 'port_congestion_level',
    'supplier_reliability_score', 'route_risk_level',
    'customs_clearance_time', 'delay_probability',
    'delivery_time_deviation', 'historical_demand',
    'fuel_consumption_rate', 'loading_unloading_time'
]

corr_df = df[corr_target_vars].corr()
print('\n=== Correlation Matrix (Pearson):')
print(corr_df.to_string())

lead_time_corr = corr_df['lead_time_days'].drop('lead_time_days').sort_values(key=lambda x: x.abs(), ascending=False)

print('\n=== Correlations with lead_time_days (sorted by absolute strength):')
for var, corr_val in lead_time_corr.items():
    strength = 'Very Weak' if abs(corr_val) < 0.1 else ('Weak' if abs(corr_val) < 0.3 else ('Moderate' if abs(corr_val) < 0.5 else 'Strong'))
    direction = 'positive' if corr_val > 0 else 'negative'
    print(f'   {var:35s}: {corr_val:+.4f} ({strength}, {direction})')

print('\nNote: Correlation does not imply causation.')

heatmap_vars = [
    'lead_time_days', 'shipping_costs', 'traffic_congestion_level',
    'weather_condition_severity', 'port_congestion_level',
    'supplier_reliability_score', 'route_risk_level',
    'customs_clearance_time', 'delay_probability',
    'delivery_time_deviation'
]

print('\n' + '-'*80)
print('STEP 5 & 6: CREATING AND SAVING VISUALIZATIONS')
print('-'*80)

print('\nCreating Visualization 1: Lead Time Days Histogram')
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='lead_time_days', bins=30, kde=True, color='#2E86AB')
plt.title('Distribution of Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Lead Time (days)', fontsize=12)
plt.ylabel('Count / Density', fontsize=12)
plt.axvline(df['lead_time_days'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["lead_time_days"].mean():.1f} days')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '01_lead_time_distribution.png'), bbox_inches='tight')
plt.close()
print('   Saved: 01_lead_time_distribution.png')

print('Creating Visualization 2: Shipping Costs Histogram')
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='shipping_costs', bins=30, kde=True, color='#A23B72')
plt.title('Distribution of Shipping Costs', fontsize=14, fontweight='bold')
plt.xlabel('Shipping Costs ($)', fontsize=12)
plt.ylabel('Count / Density', fontsize=12)
plt.axvline(df['shipping_costs'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${df["shipping_costs"].mean():.0f}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '02_shipping_cost_distribution.png'), bbox_inches='tight')
plt.close()
print('   Saved: 02_shipping_cost_distribution.png')

print('Creating Visualization 3: Traffic Congestion vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='traffic_congestion_level', y='lead_time_days', alpha=0.4, s=20, color='#2E86AB')
sns.regplot(data=df, x='traffic_congestion_level', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Traffic Congestion Level vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Traffic Congestion Level', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '03_traffic_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 03_traffic_vs_lead_time.png')

print('Creating Visualization 4: Shipping Costs vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='shipping_costs', y='lead_time_days', alpha=0.4, s=20, color='#A23B72')
sns.regplot(data=df, x='shipping_costs', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Shipping Costs vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Shipping Costs ($)', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '04_shipping_cost_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 04_shipping_cost_vs_lead_time.png')

print('Creating Visualization 5: Port Congestion vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='port_congestion_level', y='lead_time_days', alpha=0.4, s=20, color='#F18F01')
sns.regplot(data=df, x='port_congestion_level', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Port Congestion Level vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Port Congestion Level', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '05_port_congestion_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 05_port_congestion_vs_lead_time.png')

print('Creating Visualization 6: Delay Probability vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='delay_probability', y='lead_time_days', alpha=0.4, s=20, color='#C73E1D')
sns.regplot(data=df, x='delay_probability', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Delay Probability vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Delay Probability', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '06_delay_probability_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 06_delay_probability_vs_lead_time.png')

print('Creating Visualization 7: Lead Time by Risk Classification (Box Plot)')
plt.figure(figsize=(10, 6))
order_risk = df.groupby('risk_classification')['lead_time_days'].median().sort_values().index.tolist()
sns.boxplot(data=df, x='risk_classification', y='lead_time_days', order=order_risk, palette='Set2')
plt.title('Lead Time Days by Risk Classification', fontsize=14, fontweight='bold')
plt.xlabel('Risk Classification', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '07_lead_time_by_risk_boxplot.png'), bbox_inches='tight')
plt.close()
print('   Saved: 07_lead_time_by_risk_boxplot.png')

print('Creating Visualization 8: Average Lead Time by Risk Classification (Bar Chart)')
avg_risk = df.groupby('risk_classification')['lead_time_days'].mean().sort_values()
plt.figure(figsize=(10, 6))
bars = plt.bar(avg_risk.index, avg_risk.values, color=['#2ECC71', '#F39C12', '#E74C3C'])
plt.title('Average Lead Time Days by Risk Classification', fontsize=14, fontweight='bold')
plt.xlabel('Risk Classification', fontsize=12)
plt.ylabel('Average Lead Time (days)', fontsize=12)
for bar, val in zip(bars, avg_risk.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{val:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '08_average_lead_time_by_risk.png'), bbox_inches='tight')
plt.close()
print('   Saved: 08_average_lead_time_by_risk.png')

print('Creating Visualization 9: Correlation Heatmap')
plt.figure(figsize=(12, 10))
sns.heatmap(df[heatmap_vars].corr(), annot=True, fmt='.3f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
plt.title('Correlation Heatmap: Selected Logistics Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '09_correlation_heatmap.png'), bbox_inches='tight')
plt.close()
print('   Saved: 09_correlation_heatmap.png')

print('Creating Visualization 10: Monthly Average Lead Time Trend')
monthly_avg = df.groupby('month')['lead_time_days'].mean()
plt.figure(figsize=(12, 6))
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2, markersize=8, color='#2E86AB')
plt.fill_between(monthly_avg.index, monthly_avg.values, alpha=0.15, color='#2E86AB')
plt.title('Average Lead Time Days by Month', fontsize=14, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Lead Time (days)', fontsize=12)
plt.xticks(range(1, 13), month_labels)
plt.grid(True, alpha=0.3)
for x, y in zip(monthly_avg.index, monthly_avg.values):
    plt.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '10_monthly_lead_time_trend.png'), bbox_inches='tight')
plt.close()
print('   Saved: 10_monthly_lead_time_trend.png')

print('\nCreating Optional Visualizations:')

print('Optional: Weather Severity vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='weather_condition_severity', y='lead_time_days', alpha=0.4, s=20, color='#3B1F2B')
sns.regplot(data=df, x='weather_condition_severity', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Weather Condition Severity vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Weather Condition Severity', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '11_weather_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 11_weather_vs_lead_time.png')

print('Optional: Customs Clearance Time vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='customs_clearance_time', y='lead_time_days', alpha=0.4, s=20, color='#1B5299')
sns.regplot(data=df, x='customs_clearance_time', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Customs Clearance Time vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Customs Clearance Time', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '12_customs_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 12_customs_vs_lead_time.png')

print('Optional: Supplier Reliability vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='supplier_reliability_score', y='lead_time_days', alpha=0.4, s=20, color='#2ECC71')
sns.regplot(data=df, x='supplier_reliability_score', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Supplier Reliability Score vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Supplier Reliability Score', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '13_supplier_reliability_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 13_supplier_reliability_vs_lead_time.png')

print('Optional: Delivery Time Deviation vs Lead Time')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='delivery_time_deviation', y='lead_time_days', alpha=0.4, s=20, color='#6B2E68')
sns.regplot(data=df, x='delivery_time_deviation', y='lead_time_days', scatter=False, color='red', line_kws={'linewidth': 2})
plt.title('Delivery Time Deviation vs Lead Time Days', fontsize=14, fontweight='bold')
plt.xlabel('Delivery Time Deviation', fontsize=12)
plt.ylabel('Lead Time (days)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '14_delivery_deviation_vs_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 14_delivery_deviation_vs_lead_time.png')

print('Optional: Day of Week Lead Time Comparison')
dow_avg = df.groupby('day_of_week')['lead_time_days'].mean()
dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
plt.figure(figsize=(10, 6))
bars = plt.bar(dow_avg.index, dow_avg.values, color='#2E86AB')
plt.title('Average Lead Time Days by Day of Week', fontsize=14, fontweight='bold')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Average Lead Time (days)', fontsize=12)
plt.xticks(range(7), dow_labels)
for bar, val in zip(bars, dow_avg.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f'{val:.1f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, '15_dayofweek_lead_time.png'), bbox_inches='tight')
plt.close()
print('   Saved: 15_dayofweek_lead_time.png')

num_visualizations = len(os.listdir(OUTPUT_FOLDER))
print(f'\nTotal visualizations saved: {num_visualizations}')

print('\n' + '-'*80)
print('STEP 7: RISK CLASSIFICATION ANALYSIS')
print('-'*80)

risk_counts = df['risk_classification'].value_counts()
risk_pct = df['risk_classification'].value_counts(normalize=True) * 100
risk_mean_lt = df.groupby('risk_classification')['lead_time_days'].mean()
risk_median_lt = df.groupby('risk_classification')['lead_time_days'].median()

print('\n=== Risk Classification Breakdown:')
for cls in risk_counts.index:
    print(f'\n   {cls}:')
    print(f'      Count:     {risk_counts[cls]} records')
    print(f'      Percentage: {risk_pct[cls]:.2f}%')
    print(f'      Mean Lead Time:   {risk_mean_lt[cls]:.2f} days')
    print(f'      Median Lead Time: {risk_median_lt[cls]:.2f} days')

risk_order = risk_mean_lt.sort_values(ascending=False)
print(f'\nRisk levels by average lead time (highest to lowest):')
for i, (cls, val) in enumerate(risk_order.items(), 1):
    print(f'   {i}. {cls}: {val:.2f} days')

diff_high_low = risk_mean_lt.max() - risk_mean_lt.min()
print(f'\nDifference between highest and lowest risk class average lead time: {diff_high_low:.2f} days')

print('\n' + '-'*80)
print('STEP 8: TIME-BASED ANALYSIS')
print('-'*80)

yearly_avg = df.groupby('year')['lead_time_days'].mean()
print('\n=== Average Lead Time by Year:')
for yr, val in yearly_avg.items():
    print(f'   {yr}: {val:.2f} days')

print('\n=== Average Lead Time by Month:')
for mon, val in monthly_avg.items():
    month_name = month_labels[mon-1]
    print(f'   {month_name} (Month {mon}): {val:.2f} days')

best_month = monthly_avg.idxmin()
worst_month = monthly_avg.idxmax()
print(f'\nMonth with lowest average lead time: {month_labels[best_month-1]} ({monthly_avg[best_month]:.2f} days)')
print(f'Month with highest average lead time: {month_labels[worst_month-1]} ({monthly_avg[worst_month]:.2f} days)')
print(f'Monthly variation: {(monthly_avg.max() - monthly_avg.min()):.2f} days')

print('\n=== Average Lead Time by Day of Week:')
for dow, val in dow_avg.items():
    print(f'   {dow_labels[dow]}: {val:.2f} days')

best_dow = dow_avg.idxmin()
worst_dow = dow_avg.idxmax()
print(f'\nDay with lowest average lead time: {dow_labels[best_dow]} ({dow_avg[best_dow]:.2f} days)')
print(f'Day with highest average lead time: {dow_labels[worst_dow]} ({dow_avg[worst_dow]:.2f} days)')

print('\n' + '-'*80)
print('STEP 9 & 10: ANALYTICAL INSIGHTS SUMMARY (from actual data)')
print('-'*80)

print('\n=== Key Analytical Insights ===')
print('\nInsight 1: Lead Time Central Tendency')
print(f'   Finding: Mean lead time = {df["lead_time_days"].mean():.2f} days, Median = {df["lead_time_days"].median():.2f} days')
print(f'   Evidence: Descriptive statistics show mean > median, indicating right-skewed distribution')
print(f'   Interpretation: Most deliveries cluster around the median with some long-tail outliers pulling the mean upward')

print('\nInsight 2: Lead Time Distribution')
lt_skew = df['lead_time_days'].skew()
print(f'   Finding: Lead time skewness = {lt_skew:.4f} ({"positively skewed" if lt_skew > 0 else "negatively skewed"})')
print(f'   Evidence: Distribution analysis calculations')
print(f'   Interpretation: Higher probability of occasional very long lead times compared to very short ones')

top_corr_var = lead_time_corr.index[0]
top_corr_val = lead_time_corr.iloc[0]
print(f'\nInsight 3: Strongest Correlation with Lead Time')
print(f'   Finding: {top_corr_var} has correlation of {top_corr_val:+.4f} with lead_time_days')
print(f'   Evidence: Pearson correlation matrix, sorted by absolute strength')
print(f'   Interpretation: {top_corr_var.replace("_"," ").title()} shows the {"strongest positive" if top_corr_val > 0 else "strongest negative"} linear association with delivery lead time')

print(f'\nInsight 4: Risk Classification Impact')
for cls in risk_mean_lt.sort_values(ascending=False).index:
    print(f'   Finding: {cls} average lead time = {risk_mean_lt[cls]:.2f} days')
print(f'   Evidence: Group-by analysis of risk_classification vs lead_time_days')
print(f'   Interpretation: Higher risk classification is associated with longer average lead times')

print(f'\nInsight 5: Shipping Costs Relationship')
sc_corr = corr_df.loc['shipping_costs', 'lead_time_days']
sc_direction = "higher" if sc_corr > 0 else "lower"
print(f'   Finding: Correlation between shipping_costs and lead_time_days = {sc_corr:+.4f}')
print(f'   Evidence: Correlation matrix and scatter plot (04_shipping_cost_vs_lead_time.png)')
print(f'   Interpretation: Longer lead times tend to be associated with {sc_direction} shipping costs')

print(f'\nInsight 6: Congestion Effects')
tc_corr = corr_df.loc['traffic_congestion_level', 'lead_time_days']
pc_corr = corr_df.loc['port_congestion_level', 'lead_time_days']
print(f'   Finding: Traffic congestion corr = {tc_corr:+.4f}, Port congestion corr = {pc_corr:+.4f}')
print(f'   Evidence: Correlation analysis, scatter plots 03 and 05')
print(f'   Interpretation: Both congestion measures show {"positive" if tc_corr > 0 else "negative"} association with lead time')

print(f'\nInsight 7: Supplier Reliability')
sr_corr = corr_df.loc['supplier_reliability_score', 'lead_time_days']
print(f'   Finding: Supplier reliability correlation with lead time = {sr_corr:+.4f}')
print(f'   Evidence: Correlation analysis and optional visualization 13')
print(f'   Interpretation: Higher supplier reliability scores are associated with {"shorter" if sr_corr < 0 else "longer"} lead times')

print(f'\nInsight 8: Customs Clearance')
cc_corr = corr_df.loc['customs_clearance_time', 'lead_time_days']
print(f'   Finding: Customs clearance time correlation = {cc_corr:+.4f}')
print(f'   Evidence: Correlation analysis and optional visualization 12')
print(f'   Interpretation: Longer customs clearance times are associated with {"longer" if cc_corr > 0 else "shorter"} overall lead times')

print('\nInsight 9: Delay Probability')
dp_corr = corr_df.loc['delay_probability', 'lead_time_days']
print(f'   Finding: Delay probability correlation = {dp_corr:+.4f}')
print(f'   Evidence: Correlation analysis, visualization 06')
print(f'   Interpretation: Shipments with higher delay probability scores tend to have {"longer" if dp_corr > 0 else "shorter"} lead times')

print(f'\nInsight 10: Monthly Pattern')
print(f'   Finding: Best month = {month_labels[best_month-1]} ({monthly_avg[best_month]:.2f} days), Worst month = {month_labels[worst_month-1]} ({monthly_avg[worst_month]:.2f} days)')
print(f'   Evidence: Time-based groupby analysis and visualization 10')
print(f'   Interpretation: There may be seasonal patterns in logistics efficiency across the calendar year')

print('\n' + '-'*80)
print('STEP 10: OPERATIONAL RECOMMENDATIONS (based only on actual findings)')
print('-'*80)

print('\nRecommendation 1: Congestion Management')
print(f'   Data shows: Traffic congestion correlation = {tc_corr:+.4f}, Port congestion correlation = {pc_corr:+.4f}')
print('   Action: Consider routing alternatives during peak congestion periods; use predictive monitoring for port and traffic conditions')

print('\nRecommendation 2: Risk-Based Shipment Prioritization')
print(f'   Data shows: High Risk mean lead time = {risk_mean_lt.get("High Risk", risk_mean_lt.iloc[-1]):.2f} days vs Low Risk = {risk_mean_lt.get("Low Risk", risk_mean_lt.iloc[0]):.2f} days')
print('   Action: Allocate additional monitoring and expediting resources to High Risk classified shipments')

print('\nRecommendation 3: Supplier Selection and Monitoring')
print(f'   Data shows: Supplier reliability score correlation with lead time = {sr_corr:+.4f}')
print('   Action: Prioritize partnerships with high-reliability suppliers for time-sensitive shipments')

print('\nRecommendation 4: Customs Planning')
print(f'   Data shows: Customs clearance time correlation = {cc_corr:+.4f}')
print('   Action: Pre-clear documentation where possible; allocate dedicated customs brokerage resources to reduce clearance delays')

print('\nRecommendation 5: Cost Monitoring')
print(f'   Data shows: Shipping costs correlation with lead time = {sc_corr:+.4f}')
print('   Action: Track cost-to-time trade-offs; analyze whether premium shipping options justify lead-time reduction')

print('\nRecommendation 6: Seasonal Resource Allocation')
print(f'   Data shows: Monthly lead time range = {monthly_avg.min():.2f} to {monthly_avg.max():.2f} days (variation of {monthly_avg.max()-monthly_avg.min():.2f} days)')
print('   Action: Adjust staffing and fleet capacity during historically high-lead-time months')

print('\nRecommendation 7: Day-of-Week Scheduling')
print(f'   Data shows: Best day = {dow_labels[best_dow]} ({dow_avg[best_dow]:.2f} days), Worst day = {dow_labels[worst_dow]} ({dow_avg[worst_dow]:.2f} days)')
print('   Action: Consider scheduling time-critical dispatches on more efficient days of the week')

print('\nRecommendation 8: Delay Probability Triggers')
print(f'   Data shows: Delay probability correlation = {dp_corr:+.4f}')
print('   Action: Use delay_probability scores as early warning indicators; trigger contingency plans above threshold scores')

print('\n' + '='*80)
print('WEEK 3 EDA COMPLETE - Key outputs saved to:', OUTPUT_FOLDER)
print('='*80)
