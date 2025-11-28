"""
Professional PDF Malware Detection Training Pipeline
----------------------------------------------------
Author: Senior ML Engineer
Context: Cybersecurity / Malware Analysis
Standards: 
    - 70/15/15 Data Split (Train/Validation/Test)
    - Stratified K-Fold Cross Validation
    - Pipeline Architecture to prevent Data Leakage
    - Bayesian-like Hyperparameter Tuning (RandomizedSearchCV)
    - SHAP Explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import joblib
from pathlib import Path
from time import time

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
)

# --- Configuration ---
RANDOM_SEED = 42
TEST_SIZE = 0.15        # 15% Final Test (Strictly Unseen)
VALIDATION_SIZE = 0.15  # 15% Validation (used implicitly via CV or split)
CV_FOLDS = 5            # 5-Fold Cross Validation
SCORING_METRIC = 'f1'   # Optimize for F1-Score (Balance Precision/Recall)

# --- Paths ---
script_dir = Path(__file__).parent
results_dir = script_dir / 'results'
csv_dir = results_dir / 'csv'
models_dir = results_dir / 'models'
reports_dir = results_dir / 'reports'
images_dir = results_dir / 'images'

# Create directories
for d in [models_dir, reports_dir, images_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load Data
data_path = csv_dir / 'pdf_features_final.csv'
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# --- 1. Data Cleaning & Preparation ---
print("\n[1] Data Preparation...")

# Separate Target and Features
X = df.drop(columns=['class', 'file_name'])
y = df['class']

# Safety Check: Handle Infinite values
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

print(f"    Total Samples: {len(X)}")
print(f"    Features ({X.shape[1]}): {list(X.columns)}")

# --- A. Balanced Dataset Check ---
print("\n[A] Checking Dataset Balance...")
class_counts = y.value_counts().sort_index()
benign_count = class_counts.get(0, 0)
malicious_count = class_counts.get(1, 0)

print(f"    Benign samples: {benign_count}")
print(f"    Malicious samples: {malicious_count}")

MIN_SAMPLES_PER_CLASS = 1000
if benign_count < MIN_SAMPLES_PER_CLASS or malicious_count < MIN_SAMPLES_PER_CLASS:
    print(f"\n    [WARNING] Dataset may be imbalanced for training!")
    print(f"    Recommended: At least {MIN_SAMPLES_PER_CLASS}+ samples per class")
    print(f"    Current: Benign={benign_count}, Malicious={malicious_count}")
    if benign_count < MIN_SAMPLES_PER_CLASS or malicious_count < MIN_SAMPLES_PER_CLASS:
        print(f"    [ERROR] Insufficient samples. Training may be biased.")
        print(f"    Consider collecting more data or using class balancing techniques.")
else:
    print(f"    [OK] Dataset has sufficient samples per class (>= {MIN_SAMPLES_PER_CLASS})")

# --- 2. Professional Data Splitting (70% Train / 15% Val / 15% Test) ---
print("\n[2] Splitting Data (70% Train / 15% Validation / 15% Test)...")

# First, split off the 15% Test Set (Strictly Unseen)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)

# Then split the remaining 85% into 70% Train and 15% Validation
# 70% of total = 0.70 / 0.85 = 0.8235 of the remaining data
train_size_ratio = 0.70 / (1 - TEST_SIZE)  # 0.70 / 0.85 ≈ 0.8235

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=(1 - train_size_ratio), stratify=y_temp, random_state=RANDOM_SEED
)

print(f"    Training Data:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"    Validation Data: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test Data:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# --- 3. Define Pipelines & Hyperparameter Grids ---
print("\n[3] Defining Pipelines and Hyperparameter Grids...")

# A. Logistic Regression (Baseline - Simple)
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, max_iter=1000))
])

# B. SVM (Baseline - Simple)
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True, random_state=RANDOM_SEED))
])
param_svm = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

# C. Random Forest (Baseline - Simple)
pipe_rf = Pipeline([
    ('scaler', StandardScaler()), # RF doesn't strictly need scaling, but it helps SHAP/consistency
    ('clf', RandomForestClassifier(random_state=RANDOM_SEED))
])
param_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__class_weight': ['balanced', None]
}

# D. XGBoost (Advanced)
pipe_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', xgb.XGBClassifier(
        eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_SEED
    ))
])
param_xgb = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__max_depth': [3, 6, 10],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0]
}

models_to_train = [
    ('Logistic Regression', pipe_lr, {}), # No tuning for baseline
    ('SVM', pipe_svm, param_svm),
    ('Random Forest', pipe_rf, param_rf),
    ('XGBoost', pipe_xgb, param_xgb)
]

# --- 4. Training Loop with Cross-Validation ---
print(f"\n[4] Starting Training with {CV_FOLDS}-Fold Cross-Validation...")

best_models = {}
results_data = []

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for name, pipeline, params in models_to_train:
    print(f"\n    Training {name}...")
    start = time()
    
    if params:
        # Use Randomized Search for tuning (uses CV on training set)
        search = RandomizedSearchCV(
            pipeline, params, n_iter=10, scoring=SCORING_METRIC, 
            cv=cv, n_jobs=-1, random_state=RANDOM_SEED, verbose=0
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"      Best Params: {search.best_params_}")
        cv_score = search.best_score_
    else:
        # Simple fit for baseline
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        # Manual CV score for baseline
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=SCORING_METRIC)
        cv_score = scores.mean()

    duration = time() - start
    print(f"      CV {SCORING_METRIC.upper()}: {cv_score:.4f} (Time: {duration:.2f}s)")
    
    # Evaluate on validation set
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    print(f"      Validation F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
    
    best_models[name] = best_model
    results_data.append({
        'Model': name,
        'CV_F1_Score': cv_score,
        'Val_F1_Score': val_f1,
        'Val_AUC': val_auc,
        'Training_Time': duration
    })

# --- 5. Final Evaluation on UNSEEN Test Set ---
print("\n[5] Evaluating on Strictly Unseen Test Set...")

test_results = []
plt.figure(figsize=(10, 8))

for name, model in best_models.items():
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    
    test_results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'AUC': auc_score
    })
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.4f})')

# Finalize ROC Plot
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig(images_dir / 'roc_curve_comparison.png')
print("    Saved ROC Curve to 'roc_curve_comparison.png'")

# Save Metrics CSV
metrics_df = pd.DataFrame(test_results)
metrics_df.to_csv(reports_dir / 'final_model_metrics.csv', index=False)
print("\n--- Final Test Set Metrics ---")
print(metrics_df.round(4).to_string(index=False))

# --- 6. Explainability (SHAP) for the Best Model ---
print("\n[6] Generating SHAP Explanations...")

# Identify best model based on F1
best_model_name = metrics_df.sort_values('F1_Score', ascending=False).iloc[0]['Model']
print(f"    Best Model identified: {best_model_name}")
best_pipeline = best_models[best_model_name]

# Save Best Pipeline
joblib.dump(best_pipeline, models_dir / 'best_model_pipeline.pkl')
print(f"    Saved pipeline to 'best_model_pipeline.pkl'")

# Generate Confusion Matrix for Best Model
y_pred_best = best_pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malicious'], 
            yticklabels=['Benign', 'Malicious'])
plt.title(f'Confusion Matrix ({best_model_name})')
plt.savefig(images_dir / 'confusion_matrix_best.png')

# Run SHAP (Explainability)
# Note: TreeExplainer works best on tree-based models
try:
    if best_model_name in ['XGBoost', 'Random Forest']:
        model_step = 'clf'
        raw_model = best_pipeline.named_steps[model_step]
        
        # We need transformed data for SHAP if scaling was used
        preprocessor = best_pipeline.named_steps['scaler']
        X_test_transformed = preprocessor.transform(X_test)
        X_test_df = pd.DataFrame(X_test_transformed, columns=X.columns)

        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(X_test_df)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.tight_layout()
        plt.savefig(images_dir / 'shap_summary_plot.png')
        print("    Saved SHAP summary plot to 'shap_summary_plot.png'")
        
        # Also create feature importance bar plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(images_dir / 'shap_feature_importance.png')
        print("    Saved SHAP feature importance to 'shap_feature_importance.png'")
    else:
        # For non-tree models, use KernelExplainer (slower but works)
        print(f"    Using KernelExplainer for {best_model_name} (this may take longer)...")
        preprocessor = best_pipeline.named_steps['scaler']
        X_test_transformed = preprocessor.transform(X_test)
        X_test_df = pd.DataFrame(X_test_transformed, columns=X.columns)
        
        # Use a sample for faster computation
        sample_size = min(100, len(X_test_df))
        X_sample = X_test_df.sample(n=sample_size, random_state=RANDOM_SEED)
        
        explainer = shap.KernelExplainer(best_pipeline.predict_proba, X_sample)
        shap_values = explainer.shap_values(X_sample.iloc[:50])  # Limit for speed
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[1], X_sample.iloc[:50], show=False)  # Class 1 (malware)
        plt.tight_layout()
        plt.savefig(images_dir / 'shap_summary_plot.png')
        print("    Saved SHAP summary plot to 'shap_summary_plot.png'")
except Exception as e:
    print(f"    [Warning] Could not generate SHAP plot: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TRAINING COMPLETE. PROFESSIONAL ARTIFACTS GENERATED.")
print("="*80)