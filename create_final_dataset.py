"""
Create Final Optimized Dataset for Model Training
Based on EDA findings - selects top predictive features with engineered features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load the original dataset
script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / 'pdf_features.csv')

# Create results directory structure
results_dir = script_dir / 'results'
csv_dir = results_dir / 'csv'
csv_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CREATING FINAL OPTIMIZED DATASET")
print("=" * 80)
print(f"\nOriginal dataset shape: {df.shape}")

# Feature Engineering: Based on EDA findings
print("\n[1] Engineering features based on EDA analysis...")

# 1. Log-transformed file_size (Priority 1 feature - Cohen's D = 2.153)
df['log_file_size'] = np.log1p(df['file_size'])

# 2. Entropy Density (Priority 1 feature - Cohen's D = 1.224)
# High entropy in small files = packed malware
df['entropy_density'] = df['entropy'] / (df['log_file_size'] + 1)

# 3. Ensure pdf_version is numeric (Priority 1 feature - Cohen's D = 0.350)
df['pdf_version'] = pd.to_numeric(df['pdf_version'], errors='coerce').fillna(0) # type: ignore

print(f"   log_file_size correlation with target: {df['log_file_size'].corr(df['class']):.3f}") # type: ignore
print(f"   entropy_density correlation with target: {df['entropy_density'].corr(df['class']):.3f}") # type: ignore
print(f"   pdf_version correlation with target: {df['pdf_version'].corr(df['class']):.3f}") # type: ignore

# Select final recommended features based on EDA Priority 1 & 2
print("\n[2] Selecting final feature set based on EDA findings...")
print("     Priority 1: keyword_OpenAction, log_file_size, entropy_density, pdf_version")
print("     Priority 2: keyword_JS, keyword_JavaScript, file_size, entropy")

final_features = [
    'file_name',  # Keep for reference
    # Priority 1 features (High Predictive Power)
    'keyword_OpenAction',  # Cohen's D = 2.568, r = 0.595 → Malware
    'log_file_size',  # Cohen's D = 2.153, r = -0.649 → Benign
    'entropy_density',  # Cohen's D = 1.224, r = 0.312 → Malware
    'pdf_version',  # Cohen's D = 0.350, r = -0.289 → Benign
    # Priority 2 features (Moderate Predictive Power)
    'keyword_JS',  # Cohen's D = 0.278, r = 0.118 → Malware
    'keyword_JavaScript',  # Cohen's D = 0.276, r = 0.155 → Malware
    'file_size',  # Cohen's D = 0.243, r = -0.131 → Benign
    'entropy',  # Cohen's D = 0.135, r = -0.113 → Benign
    'class'  # Target variable
]

# Create final dataset
df_final = df[final_features].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Features included: {len(final_features) - 2} features + file_name + class")
print(f"\nFeature correlations with target:")
for feat in final_features:
    if feat not in ['class', 'file_name']:
        corr = df[feat].corr(df['class']) # type: ignore
        direction = "→ Malware" if corr > 0 else "→ Benign"
        print(f"  - {feat:25s} (r = {corr:7.3f} {direction})")

# Save final dataset
output_path = csv_dir / 'pdf_features_final.csv'
df_final.to_csv(output_path, index=False)
print(f"\n[OK] Final optimized dataset saved to: {output_path}")

# Also create a minimal version with only Priority 1 features
print("\n[3] Creating minimal feature set (Priority 1 only - 4 features)...")
minimal_features = [
    'file_name',
    'keyword_OpenAction',  # Priority 1
    'log_file_size',  # Priority 1
    'entropy_density',  # Priority 1
    'pdf_version',  # Priority 1
    'class'
]

df_minimal = df[minimal_features].copy()
output_path_minimal = csv_dir / 'pdf_features_minimal.csv'
df_minimal.to_csv(output_path_minimal, index=False)
print(f"[OK] Minimal feature set saved to: {output_path_minimal}")
print(f"      Features: {len(minimal_features) - 2} (Priority 1 only)")

print("\n" + "=" * 80)
print("DATASET PREPARATION COMPLETE!")
print("=" * 80)
print("\nFiles created:")
print(f"  1. {csv_dir / 'pdf_features_final.csv'} - Full recommended set (8 features: Priority 1 + 2)")
print(f"  2. {csv_dir / 'pdf_features_minimal.csv'} - Minimal set (4 features: Priority 1 only)")

