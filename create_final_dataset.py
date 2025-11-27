"""
Create Final Optimized Dataset for Model Training
Applies feature engineering and selects only recommended features.
"""

import pandas as pd
from pathlib import Path

# Load the original dataset
script_dir = Path(__file__).parent
df = pd.read_csv(script_dir / 'pdf_features.csv')

print("=" * 80)
print("CREATING FINAL OPTIMIZED DATASET")
print("=" * 80)
print(f"\nOriginal dataset shape: {df.shape}")

# Feature Engineering: Create keyword_density
print("\n[1] Engineering keyword_density feature...")
keyword_cols = ['keyword_JS', 'keyword_JavaScript', 'keyword_AA', 
                'keyword_OpenAction', 'keyword_Launch', 'keyword_EmbeddedFile',
                'keyword_URI', 'keyword_ObjStm']

df['keyword_sum'] = df[keyword_cols].sum(axis=1)
df['keyword_density'] = df['keyword_sum'] / (df['file_size'] + 1)  # +1 to avoid division by zero

print(f"   keyword_density correlation with target: {df['keyword_density'].corr(df['class']):.3f}") # type: ignore

# Select final recommended features
print("\n[2] Selecting final feature set...")
final_features = [
    'file_name',  # Keep for reference
    'keyword_OpenAction',
    'keyword_density',
    'keyword_JavaScript',
    'file_size',
    'entropy',
    'keyword_ObjStm',
    'keyword_AA',  # Optional - can be excluded for linear models
    'class'  # Target variable
]

# Create final dataset
df_final = df[final_features].copy()

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Features included: {len(final_features) - 2} features + file_name + class")
print(f"\nFeatures:")
for feat in final_features:
    if feat != 'class' and feat != 'file_name':
        corr = df[feat].corr(df['class']) # type: ignore
        print(f"  - {feat:25s} (r with target: {corr:7.3f})")

# Save final dataset
output_path = script_dir / 'pdf_features_final.csv'
df_final.to_csv(output_path, index=False)
print(f"\n[OK] Final optimized dataset saved to: {output_path}")

# Also create a version without optional features (for linear models)
print("\n[3] Creating minimal feature set (excluding optional features)...")
minimal_features = [
    'file_name',
    'keyword_OpenAction',
    'keyword_density',
    'keyword_JavaScript',
    'file_size',
    'entropy',
    'keyword_ObjStm',
    'class'
]

df_minimal = df[minimal_features].copy()
output_path_minimal = script_dir / 'pdf_features_minimal.csv'
df_minimal.to_csv(output_path_minimal, index=False)
print(f"[OK] Minimal feature set saved to: {output_path_minimal}")
print(f"      Features: {len(minimal_features) - 2} (excludes keyword_AA)")

print("\n" + "=" * 80)
print("DATASET PREPARATION COMPLETE!")
print("=" * 80)
print("\nFiles created:")
print(f"  1. pdf_features_final.csv - Full recommended feature set (7 features)")
print(f"  2. pdf_features_minimal.csv - Minimal feature set (6 features, for linear models)")

