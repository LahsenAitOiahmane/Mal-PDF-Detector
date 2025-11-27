"""
Comprehensive Exploratory Data Analysis (EDA) for PDF Features Dataset
Performs bivariate analysis, multicollinearity checks, and provides recommendations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Load the data
script_dir = Path(__file__).parent
csv_path = script_dir / 'pdf_features.csv'
df = pd.read_csv(csv_path)

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS - PDF FEATURES DATASET")
print("=" * 80)
print(f"\nDataset Shape: {df.shape}")
print(f"Total samples: {len(df)}")
print(f"\nClass Distribution:")
print(df['class'].value_counts().sort_index())
print(f"\nClass Distribution (%):")
print(df['class'].value_counts(normalize=True).sort_index() * 100)

# ============================================================================
# ZERO VARIANCE CHECK (Useless Feature Detector)
# ============================================================================
print("\n" + "=" * 80)
print("ZERO VARIANCE CHECK (Useless Feature Detector)")
print("=" * 80)

# Check for features with zero variance (all same values)
zero_variance_features = []
for col in df.columns:
    if col not in ['file_name', 'class']:  # Skip non-numeric columns
        try:
            if df[col].nunique() <= 1:  # Zero or near-zero variance
                zero_variance_features.append(col)
                print(f"[WARNING] Zero variance feature detected: {col} (all values = {df[col].iloc[0] if len(df) > 0 else 'N/A'})")
        except:
            pass

if zero_variance_features:
    print(f"\n[ACTION REQUIRED] {len(zero_variance_features)} feature(s) with zero variance found.")
    print("These features should be dropped before model training as they provide no information.")
    print(f"Features to drop: {', '.join(zero_variance_features)}")
else:
    print("\n[OK] No zero variance features detected. All features have variation.")

# ============================================================================
# LOGARITHMIC SCALING FOR FILE SIZE
# ============================================================================
# Create log-transformed file_size for better visualization
df['log_file_size'] = np.log1p(df['file_size'])  # log(1+x) handles zeros
print(f"\n[OK] Created log_file_size feature (log(1+file_size)) for better visualization")

# Identify numerical features (exclude file_name and pdf_version for now)
numerical_features = ['file_size', 'log_file_size', 'entropy', 'keyword_JS', 'keyword_JavaScript', 
                      'keyword_AA', 'keyword_OpenAction', 'keyword_Launch', 
                      'keyword_EmbeddedFile', 'keyword_URI', 'keyword_ObjStm', 'class']

# Basic statistics
print("\n" + "=" * 80)
print("BASIC STATISTICS")
print("=" * 80)
print(df[numerical_features].describe())

# ============================================================================
# 1. FEATURE PREDICTIVE POWER ANALYSIS (BIVARIATE)
# ============================================================================
print("\n" + "=" * 80)
print("1. FEATURE PREDICTIVE POWER ANALYSIS (BIVARIATE)")
print("=" * 80)

# Features to analyze (excluding class)
features_to_plot = [f for f in numerical_features if f != 'class']

# Create box plots for each feature
n_features = len(features_to_plot)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feature in enumerate(features_to_plot):
    ax = axes[idx]
    
    # Create box plot
    df.boxplot(column=feature, by='class', ax=ax, grid=False)
    ax.set_title(f'{feature} by Class', fontsize=12, fontweight='bold')
    ax.set_xlabel('Class (0=Benign, 1=Malicious)', fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.set_xticklabels(['Benign', 'Malicious'])
    
    # Calculate and display separation metrics
    benign_data = df[df['class'] == 0][feature]
    malicious_data = df[df['class'] == 1][feature]
    
    benign_median = benign_data.median() # type: ignore
    malicious_median = malicious_data.median() # type: ignore
    separation = abs(malicious_median - benign_median) / (benign_data.std() + malicious_data.std() + 1e-10)
    
    # Add text annotation
    ax.text(0.5, 0.95, f'Median Diff: {abs(malicious_median - benign_median):.2f}\n'
            f'Separation: {separation:.3f}',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Remove extra subplots
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Box Plots: Feature Distributions by Class', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(script_dir / 'boxplots_by_class.png', dpi=300, bbox_inches='tight')
print("\n[OK] Box plots saved to: boxplots_by_class.png")

# Create violin plots for key features (entropy and keyword counts)
key_features = ['entropy', 'keyword_JS', 'keyword_JavaScript', 'keyword_AA', 
                'keyword_OpenAction', 'keyword_Launch', 'keyword_EmbeddedFile', 
                'keyword_URI', 'keyword_ObjStm']

n_key = len(key_features)
n_cols_v = 3
n_rows_v = (n_key + n_cols_v - 1) // n_cols_v

fig, axes = plt.subplots(n_rows_v, n_cols_v, figsize=(18, 6 * n_rows_v))
axes = axes.flatten() if n_key > 1 else [axes]

for idx, feature in enumerate(key_features):
    ax = axes[idx]
    
    # Create violin plot
    data_to_plot = [df[df['class'] == 0][feature].values, # type: ignore
                    df[df['class'] == 1][feature].values] # type: ignore
    
    parts = ax.violinplot(data_to_plot, positions=[0, 1], showmeans=True, showmedians=True)
    
    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('#D3D3D3')
        pc.set_alpha(0.7)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Benign', 'Malicious'])
    ax.set_ylabel(feature, fontsize=10)
    ax.set_title(f'{feature} Distribution by Class', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Remove extra subplots
for idx in range(n_key, len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Violin Plots: Key Feature Distributions by Class', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(script_dir / 'violinplots_by_class.png', dpi=300, bbox_inches='tight')
print("[OK] Violin plots saved to: violinplots_by_class.png")

# Calculate separation metrics for all features
print("\n" + "-" * 80)
print("SEPARATION METRICS (Higher = Better Separation)")
print("-" * 80)

separation_metrics = []
for feature in features_to_plot:
    benign_data = df[df['class'] == 0][feature]
    malicious_data = df[df['class'] == 1][feature]
    
    benign_median = benign_data.median() # type: ignore
    malicious_median = malicious_data.median() # type: ignore
    median_diff = abs(malicious_median - benign_median)
    
    # Effect size (Cohen's d approximation)
    pooled_std = np.sqrt((benign_data.std()**2 + malicious_data.std()**2) / 2)
    cohens_d = median_diff / (pooled_std + 1e-10)
    
    # Mann-Whitney U test statistic (normalized)
    from scipy.stats import mannwhitneyu
    try:
        stat, p_value = mannwhitneyu(benign_data, malicious_data, alternative='two-sided')
        separation_metrics.append({
            'Feature': feature,
            'Median_Diff': median_diff,
            'Cohens_D': cohens_d,
            'Mann_Whitney_p': p_value,
            'Benign_Median': benign_median,
            'Malicious_Median': malicious_median
        })
    except:
        separation_metrics.append({
            'Feature': feature,
            'Median_Diff': median_diff,
            'Cohens_D': cohens_d,
            'Mann_Whitney_p': np.nan,
            'Benign_Median': benign_median,
            'Malicious_Median': malicious_median
        })

separation_df = pd.DataFrame(separation_metrics)
separation_df = separation_df.sort_values('Cohens_D', ascending=False)
print(separation_df.to_string(index=False))

# ============================================================================
# 2. MULTICOLLINEARITY AND REDUNDANCY CHECK (MULTIVARIATE)
# ============================================================================
print("\n" + "=" * 80)
print("2. MULTICOLLINEARITY AND REDUNDANCY CHECK (MULTIVARIATE)")
print("=" * 80)

# Calculate correlation matrix
correlation_matrix = df[numerical_features].corr() # type: ignore

# Plot correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Heatmap: All Numerical Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(script_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("\n[OK] Correlation heatmap saved to: correlation_heatmap.png")

# Analyze correlations with target variable
print("\n" + "-" * 80)
print("CORRELATION WITH TARGET VARIABLE (class)")
print("-" * 80)
target_correlations = correlation_matrix['class'].drop('class').sort_values(key=abs, ascending=False) # type: ignore
print(target_correlations.to_string())

# Find highly correlated feature pairs (excluding class)
print("\n" + "-" * 80)
print("HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.95, excluding class)")
print("-" * 80)

high_corr_pairs = []
features_for_corr = [f for f in numerical_features if f != 'class']
corr_subset = correlation_matrix.loc[features_for_corr, features_for_corr]

for i in range(len(corr_subset.columns)):
    for j in range(i+1, len(corr_subset.columns)):
        corr_value = corr_subset.iloc[i, j]
        # Safely coerce the correlation value to a float-compatible numeric and skip if invalid
        try:
            corr_val_num = np.float64(corr_value)  # type: ignore
        except (ValueError, TypeError):
            continue
        if not np.isnan(corr_val_num) and abs(corr_val_num) > 0.95:
            high_corr_pairs.append({
                'Feature_1': corr_subset.columns[i],
                'Feature_2': corr_subset.columns[j],
                'Correlation': corr_value
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    print(high_corr_df.to_string(index=False))
else:
    print("No feature pairs with |r| > 0.95 found.")

# Check for moderate correlations (0.7 - 0.95)
print("\n" + "-" * 80)
print("MODERATELY CORRELATED FEATURE PAIRS (0.7 < |r| <= 0.95, excluding class)")
print("-" * 80)

moderate_corr_pairs = []
for i in range(len(corr_subset.columns)):
    for j in range(i+1, len(corr_subset.columns)):
        corr_value = corr_subset.iloc[i, j]
        # Safely coerce the correlation value to a float-compatible numeric and skip if invalid
        try:
            corr_val_num = np.float64(corr_value)  # type: ignore
        except (ValueError, TypeError):
            continue
        if not np.isnan(corr_val_num) and 0.7 < abs(corr_val_num) <= 0.95:
            moderate_corr_pairs.append({
                'Feature_1': corr_subset.columns[i],
                'Feature_2': corr_subset.columns[j],
                'Correlation': corr_value
            })

if moderate_corr_pairs:
    moderate_corr_df = pd.DataFrame(moderate_corr_pairs)
    print(moderate_corr_df.to_string(index=False))
else:
    print("No feature pairs with 0.7 < |r| <= 0.95 found.")

# ============================================================================
# 3. SYNTHESIS AND FINAL RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 80)
print("3. SYNTHESIS AND FINAL RECOMMENDATIONS")
print("=" * 80)

# Generate summary report
report_lines = []
report_lines.append("\n" + "=" * 80)
report_lines.append("EDA SUMMARY REPORT")
report_lines.append("=" * 80)

report_lines.append("\n1. MOST PREDICTIVE FEATURES (Top 5 by Cohen's D):")
top_features = separation_df.head(5)
for idx, row in top_features.iterrows():
    report_lines.append(f"   - {row['Feature']}: Cohen's D = {row['Cohens_D']:.3f}, "
                       f"p-value = {row['Mann_Whitney_p']:.2e}")

report_lines.append("\n2. FEATURES HIGHLY CORRELATED WITH TARGET (|r| > 0.3):")
high_target_corr = target_correlations[abs(target_correlations) > 0.3]
for feature, corr in high_target_corr.items(): # type: ignore
    report_lines.append(f"   - {feature}: r = {corr:.3f}")

report_lines.append("\n3. REDUNDANT FEATURES (High inter-feature correlation):")
if high_corr_pairs:
    for pair in high_corr_pairs:
        report_lines.append(f"   - {pair['Feature_1']} <-> {pair['Feature_2']}: r = {pair['Correlation']:.3f}")
        # Determine which feature to keep based on target correlation
        feat1_target_corr = abs(target_correlations.get(pair['Feature_1'], 0)) # type: ignore
        feat2_target_corr = abs(target_correlations.get(pair['Feature_2'], 0)) # type: ignore
        if feat1_target_corr >= feat2_target_corr:
            report_lines.append(f"     Recommendation: Keep {pair['Feature_1']} (higher correlation with target)")
        else:
            report_lines.append(f"     Recommendation: Keep {pair['Feature_2']} (higher correlation with target)")
else:
    report_lines.append("   - No highly redundant features found (|r| > 0.95)")

report_lines.append("\n4. FEATURE ENGINEERING RECOMMENDATIONS:")
# Calculate keyword sum
df['keyword_sum'] = df[['keyword_JS', 'keyword_JavaScript', 'keyword_AA', 
                        'keyword_OpenAction', 'keyword_Launch', 'keyword_EmbeddedFile',
                        'keyword_URI', 'keyword_ObjStm']].sum(axis=1)

# Check correlation of engineered features
keyword_sum_corr = df['keyword_sum'].corr(df['class']) # type: ignore
report_lines.append(f"   - keyword_sum: r with target = {keyword_sum_corr:.3f}")

# Keyword density (keywords per file size)
df['keyword_density'] = df['keyword_sum'] / (df['file_size'] + 1)  # +1 to avoid division by zero
keyword_density_corr = df['keyword_density'].corr(df['class']) # type: ignore
report_lines.append(f"   - keyword_density (keyword_sum/file_size): r with target = {keyword_density_corr:.3f}")

report_lines.append("\n5. FINAL FEATURE RECOMMENDATIONS:")
report_lines.append("\n   PRIORITY 1 (Must Include - High Predictive Power):")
priority1 = separation_df[separation_df['Cohens_D'] > 0.3].head(5)
for idx, row in priority1.iterrows():
    report_lines.append(f"      - {row['Feature']}")

report_lines.append("\n   PRIORITY 2 (Should Include - Moderate Predictive Power):")
priority2 = separation_df[(separation_df['Cohens_D'] > 0.1) & (separation_df['Cohens_D'] <= 0.3)]
for idx, row in priority2.iterrows():
    report_lines.append(f"      - {row['Feature']}")

report_lines.append("\n   PRIORITY 3 (Consider Including - Low but Non-zero Power):")
priority3 = separation_df[(separation_df['Cohens_D'] > 0.05) & (separation_df['Cohens_D'] <= 0.1)]
for idx, row in priority3.iterrows():
    report_lines.append(f"      - {row['Feature']}")

report_lines.append("\n   FEATURES TO EXCLUDE:")
low_power = separation_df[separation_df['Cohens_D'] <= 0.05]
if len(low_power) > 0:
    for idx, row in low_power.iterrows():
        report_lines.append(f"      [EXCLUDE] {row['Feature']} (Cohen's D = {row['Cohens_D']:.3f})")
else:
    report_lines.append("      (None - all features show some separation)")

# Save report
report_text = "\n".join(report_lines)
# Print report with error handling for Windows console
try:
    print(report_text)
except UnicodeEncodeError:
    # Fallback: print without special characters
    safe_report = report_text.encode('ascii', 'ignore').decode('ascii')
    print(safe_report)

with open(script_dir / 'eda_summary_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\n[OK] Summary report saved to: eda_summary_report.txt")

# Save engineered features to CSV for potential use
df_engineered = df.copy()
df_engineered.to_csv(script_dir / 'pdf_features_with_engineered.csv', index=False)
print("[OK] Dataset with engineered features saved to: pdf_features_with_engineered.csv")

print("\n" + "=" * 80)
print("EDA ANALYSIS COMPLETE!")
print("=" * 80)

