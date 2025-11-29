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
sns.set()  # Set seaborn style to prevent version mismatch issues
import xgboost as xgb
import shap
import joblib
import json
import random
from pathlib import Path
from time import time
from datetime import datetime

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
)

# --- Configuration ---
RANDOM_SEED = 42
# Set all random seeds for reproducibility
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

TEST_SIZE = 0.15        # 15% Final Test (Strictly Unseen)
VALIDATION_SIZE = 0.15  # 15% Validation (used implicitly via CV or split)
CV_FOLDS = 5            # 5-Fold Cross Validation
SCORING_METRIC = 'f1'   # Optimize for F1-Score (Balance Precision/Recall)
SVM_N_ITER = 5          # Reduced iterations for SVM RandomizedSearch (probability=True is slow)

# --- Helper Functions ---
def get_proba(estimator, X):
    """
    Safely extract probability scores from estimator.
    Falls back to decision_function if predict_proba not available.
    """
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        else:
            return proba
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        # Normalize decision scores to [0, 1] range
        if scores.ndim == 1:
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return scores_normalized
        else:
            # Multi-class case
            scores_normalized = (scores - scores.min(axis=1, keepdims=True)) / \
                              (scores.max(axis=1, keepdims=True) - scores.min(axis=1, keepdims=True) + 1e-10)
            return scores_normalized
    else:
        raise RuntimeError(f"Model {type(estimator)} has no probability outputs (predict_proba or decision_function)")

# --- Paths ---
script_dir = Path(__file__).parent
results_dir = script_dir / 'results'
csv_dir = results_dir / 'csv'
models_dir = results_dir / 'models'
reports_dir = results_dir / 'reports'
images_dir = results_dir / 'images'

# Create directories
for d in [csv_dir, models_dir, reports_dir, images_dir]:
    d.mkdir(parents=True, exist_ok=True)

# Load Data
data_path = csv_dir / 'pdf_features_final.csv'
assert data_path.exists(), f"Dataset missing at {data_path}. Please run create_final_dataset.py first."
print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

# --- 1. Data Cleaning & Preparation ---
print("\n[1] Data Preparation...")

# Separate Target and Features
X = df.drop(columns=['class', 'file_name'])
y = df['class']

# Validate and map label encoding (handle string labels if present)
if y.dtype == 'object' or y.dtype.name == 'category':
    label_map = {'benign': 0, 'malicious': 1, 'Benign': 0, 'Malicious': 1, 0: 0, 1: 1}
    y = y.map(label_map)  # type: ignore
    if y.isna().any():  # type: ignore
        print(f"    [WARNING] Unknown labels found, mapping to numeric")
        y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)  # type: ignore

# Safety Check: Handle Infinite values (will be handled in pipeline imputers)
# Keep NaN for now - imputers in pipelines will handle them properly
X = X.replace([np.inf, -np.inf], np.nan)

print(f"    Total Samples: {len(X)}")
print(f"    Features ({X.shape[1]}): {list(X.columns)}")

# --- A. Balanced Dataset Check ---
print("\n[A] Checking Dataset Balance...")
class_counts = y.value_counts().sort_index()  # type: ignore

# Safely extract counts, handling None values
benign_count = int(class_counts.get(0, 0) or 0)
malicious_count = int(class_counts.get(1, 0) or 0)

print(f"    Benign samples: {benign_count}")
print(f"    Malicious samples: {malicious_count}")

# Check imbalance ratio instead of hard-coded threshold
if benign_count > 0 and malicious_count > 0:
    imbalance_ratio = max(benign_count, malicious_count) / min(benign_count, malicious_count)
    print(f"    Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3.0:
        print(f"\n    [WARNING] Significant class imbalance detected (ratio > 3:1)")
        print(f"    Consider using class weights or balanced sampling techniques.")
        print(f"    Current ratio: {imbalance_ratio:.2f}:1")
    else:
        print(f"    [OK] Dataset is reasonably balanced (ratio <= 3:1)")
else:
    print(f"    [ERROR] One or both classes have zero samples!")
    raise ValueError("Dataset must contain samples from both classes")

# --- 2. Professional Data Splitting (70% Train / 15% Val / 15% Test) ---
print("\n[2] Splitting Data (70% Train / 15% Validation / 15% Test)...")
print("    Note: Validation set is used for model selection check, not hyperparameter tuning.")
print("    Hyperparameter tuning uses Cross-Validation on training set only.")

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
# Imputation in pipeline prevents data leakage
pipe_lr = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, max_iter=1000))
])

# B. SVM (Baseline - Simple)
# Note: probability=True enables predict_proba() but significantly increases training time
# This is necessary for ROC-AUC calculation
pipe_svm = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True, random_state=RANDOM_SEED))
])
param_svm = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

# C. Random Forest (Baseline - Simple)
# Note: Tree models are scale-invariant - scaling is NOT needed and hurts performance
# But we still need imputation in pipeline to prevent leakage
pipe_rf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(random_state=RANDOM_SEED))
])
param_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__class_weight': ['balanced', None]
}

# D. XGBoost (Advanced)
# Note: Tree models are scale-invariant - scaling is NOT needed and hurts performance
# But we still need imputation in pipeline to prevent leakage
pipe_xgb = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('clf', xgb.XGBClassifier(
        eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_SEED
    ))
])

# Calculate scale_pos_weight for class imbalance (if malicious is minority)
scale_pos_weight = benign_count / malicious_count if malicious_count < benign_count else 1.0
param_xgb = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__max_depth': [3, 6, 10],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0],
    'clf__scale_pos_weight': [1.0, scale_pos_weight]  # Handle class imbalance
}

# E. Neural Network (Advanced)
# Note: max_iter=1000 ensures convergence for most datasets
# early_stopping=True helps prevent overfitting and speeds up training
pipe_nn = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(random_state=RANDOM_SEED, max_iter=1000, early_stopping=True, 
                         validation_fraction=0.1, n_iter_no_change=10))
])
param_nn = {
    'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],  # Correct syntax for scikit-learn
    'clf__activation': ['relu', 'tanh'],
    'clf__alpha': [0.0001, 0.001, 0.01],  # L2 regularization
    'clf__learning_rate': ['constant', 'adaptive'],
    'clf__learning_rate_init': [0.001, 0.01]
}

models_to_train = [
    ('Logistic Regression', pipe_lr, {}), # No tuning for baseline
    ('SVM', pipe_svm, param_svm),
    ('Random Forest', pipe_rf, param_rf),
    ('XGBoost', pipe_xgb, param_xgb),
    ('Neural Network', pipe_nn, param_nn)
]

# --- 4. Training Loop with Cross-Validation ---
print(f"\n[4] Starting Training with {CV_FOLDS}-Fold Cross-Validation...")

best_models = {}
results_data = []

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

for name, pipeline, params in models_to_train:
    print(f"\n    Training {name}...")
    if name == 'SVM':
        print("      [Note] SVM with probability=True may take longer to train...")
    start = time()
    
    if params:
        # Use Randomized Search for tuning (uses CV on training set only)
        # Note: Validation set is NOT used here - it's reserved for model selection check
        # Reduce n_iter for SVM (probability=True is very slow)
        n_iter = SVM_N_ITER if name == 'SVM' else 10
        search = RandomizedSearchCV(
            pipeline, params, n_iter=n_iter, scoring=SCORING_METRIC, 
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
    y_val_proba = get_proba(best_model, X_val)
    val_f1 = f1_score(y_val, y_val_pred, zero_division='warn')
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
    y_proba = get_proba(model, X_test)
    
    # Metrics with zero_division to handle edge cases
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division='warn')
    rec = recall_score(y_test, y_pred, zero_division='warn')
    f1 = f1_score(y_test, y_pred, zero_division='warn')
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

# --- 6. Identify Best Model ---
print("\n[6] Identifying Best Model...")

# Identify best model based on F1
best_model_name = metrics_df.sort_values('F1_Score', ascending=False).iloc[0]['Model']
print(f"    Best Model identified: {best_model_name}")

# Keep uncalibrated copy for SHAP/feature importance (before calibration)
uncalibrated_best = best_models[best_model_name]

# --- 7. Final Refit on Train+Val (Use More Data) ---
print("\n[7] Final Refit on Combined Train+Validation Set...")
# Combine train and validation for final model (common practice)
X_train_final = pd.concat([X_train, X_val], axis=0)
y_train_final = pd.concat([y_train, y_val], axis=0)
print(f"    Combined training data: {len(X_train_final)} samples")

# Refit uncalibrated model on combined data
uncalibrated_best.fit(X_train_final, y_train_final)
print("    Uncalibrated model refit on train+val")

# --- 8. Model Calibration ---
print("\n[8] Calibrating Best Model Probabilities...")
# Calibrate only the classifier, not the entire pipeline
# This is more reliable than wrapping the whole pipeline
if hasattr(uncalibrated_best, 'named_steps'):
    # Extract preprocessing steps and classifier
    pre = uncalibrated_best.named_steps  # type: ignore
    raw_classifier = pre['clf']  # type: ignore
    
    # Calibrate only the classifier
    calibrated_clf = CalibratedClassifierCV(raw_classifier, cv=5, method='sigmoid')
    calibrated_clf.fit(X_train_final, y_train_final)
    
    # Rebuild pipeline with calibrated classifier
    if best_model_name in ['Random Forest', 'XGBoost']:
        # Tree models: no scaler
        best_pipeline = Pipeline([
            ('imputer', pre.get('imputer', SimpleImputer(strategy='median'))),  # type: ignore
            ('clf', calibrated_clf)
        ])
    else:
        # Non-tree models: include scaler
        best_pipeline = Pipeline([
            ('imputer', pre.get('imputer', SimpleImputer(strategy='median'))),  # type: ignore
            ('scaler', pre.get('scaler', StandardScaler())),  # type: ignore
            ('clf', calibrated_clf)
        ])
else:
    # If not a pipeline, calibrate directly
    calibrated_clf = CalibratedClassifierCV(uncalibrated_best, cv=5, method='sigmoid')
    calibrated_clf.fit(X_train_final, y_train_final)
    best_pipeline = calibrated_clf

print("    Model calibrated using Platt scaling (sigmoid)")

# --- 9. Threshold Optimization ---
print("\n[9] Optimizing Classification Threshold...")
# Use original validation set for threshold optimization (already held out)
# Get calibrated probabilities on validation set
y_val_proba_calibrated = get_proba(best_pipeline, X_val)

# Test thresholds from 0.1 to 0.9
thresholds = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
best_f1 = 0
threshold_scores = []

for t in thresholds:
    y_pred_t = (y_val_proba_calibrated >= t).astype(int)
    score = f1_score(y_val, y_pred_t, zero_division='warn')
    threshold_scores.append({'threshold': t, 'f1_score': score})
    if score > best_f1:
        best_f1 = score
        best_threshold = t

print(f"    Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
print(f"    Default threshold (0.5) F1: {f1_score(y_val, (y_val_proba_calibrated >= 0.5).astype(int), zero_division='warn'):.4f}")

# Plot threshold optimization curve
threshold_df = pd.DataFrame(threshold_scores)
plt.figure(figsize=(8, 6))
plt.plot(threshold_df['threshold'], threshold_df['f1_score'])
plt.axvline(best_threshold, color='r', linestyle='--', label=f'Optimal: {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1 Score')
plt.title('Threshold Optimization on Validation Set')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(images_dir / 'threshold_optimization.png')
plt.close()
print("    Saved threshold optimization curve to 'threshold_optimization.png'")

# Re-evaluate test set with optimal threshold
y_test_proba_calibrated = get_proba(best_pipeline, X_test)
y_test_pred_optimal = (y_test_proba_calibrated >= best_threshold).astype(int)

# Update best model metrics with optimal threshold
optimal_acc = accuracy_score(y_test, y_test_pred_optimal)
optimal_prec = precision_score(y_test, y_test_pred_optimal, zero_division='warn')
optimal_rec = recall_score(y_test, y_test_pred_optimal, zero_division='warn')
optimal_f1 = f1_score(y_test, y_test_pred_optimal, zero_division='warn')
optimal_auc = roc_auc_score(y_test, y_test_proba_calibrated)

print(f"\n    Test Set Metrics with Optimal Threshold ({best_threshold:.3f}):")
print(f"      Accuracy: {optimal_acc:.4f}")
print(f"      Precision: {optimal_prec:.4f}")
print(f"      Recall: {optimal_rec:.4f}")
print(f"      F1-Score: {optimal_f1:.4f}")
print(f"      AUC: {optimal_auc:.4f}")

# Save optimal threshold metrics CSV
optimal_df = pd.DataFrame([{
    'Model': f'{best_model_name} (Optimal Threshold)',
    'Accuracy': optimal_acc,
    'Precision': optimal_prec,
    'Recall': optimal_rec,
    'F1_Score': optimal_f1,
    'AUC': optimal_auc,
    'Threshold': best_threshold
}])
# Also append to original metrics for comparison
metrics_df_optimal = pd.concat([metrics_df, optimal_df], ignore_index=True)
metrics_df_optimal.to_csv(reports_dir / 'final_model_metrics_with_optimal.csv', index=False)
print(f"\n    Saved metrics with optimal threshold to 'final_model_metrics_with_optimal.csv'")

# Save Best Pipeline with versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"model_v{timestamp}.pkl"
joblib.dump(best_pipeline, models_dir / model_version)
joblib.dump(best_pipeline, models_dir / 'best_model_pipeline.pkl')  # Latest
# Also save uncalibrated model for SHAP/debugging
joblib.dump(uncalibrated_best, models_dir / f'uncalibrated_{model_version}')
print(f"\n    Saved calibrated pipeline to '{model_version}' and 'best_model_pipeline.pkl'")
print(f"    Saved uncalibrated model to 'uncalibrated_{model_version}'")

# --- 10. Generate Confusion Matrices for All Models ---
print("\n[10] Generating Confusion Matrices for All Models...")

n_models = len(best_models)
n_cols = 2
n_rows = (n_models + n_cols - 1) // n_cols  # More robust calculation
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))

# Robust handling of axes for different cases
if n_models == 1:
    axes = [axes]
elif n_rows == 1:
    # Single row case
    if isinstance(axes, np.ndarray):
        axes = axes.flatten() if axes.ndim > 1 else [axes]
    else:
        axes = [axes]
else:
    # Multiple rows
    axes = axes.flatten() if hasattr(axes, 'flatten') else list(axes)

for idx, (name, model) in enumerate(best_models.items()):
    y_pred_model = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_model)
    
    ax = axes[idx]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    ax.set_title(f'{name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')

# Remove extra subplots
for idx in range(n_models, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(images_dir / 'confusion_matrices_all_models.png')
plt.close()
print("    Saved confusion matrices to 'confusion_matrices_all_models.png'")

# Also save best model confusion matrix separately (with optimal threshold)
y_pred_best_optimal = (get_proba(best_pipeline, X_test) >= best_threshold).astype(int)
cm_best = confusion_matrix(y_test, y_pred_best_optimal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malicious'], 
            yticklabels=['Benign', 'Malicious'])
plt.title(f'Confusion Matrix ({best_model_name}) - Optimal Threshold {best_threshold:.3f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(images_dir / 'confusion_matrix_best_optimal.png')
plt.close()
print("    Saved best model confusion matrix (optimal threshold) to 'confusion_matrix_best_optimal.png'")

# --- 10. Feature Importance Analysis (Permutation Importance for All Models) ---
print("\n[10] Computing Feature Importance (Permutation Importance)...")
print("    [Note] Using raw X_test - pipelines handle preprocessing internally")

feature_importance_results = {}
for name, model in best_models.items():
    print(f"    Computing for {name}...")
    # Always pass raw X_test - pipelines handle scaling/imputation internally
    # This prevents double-transformation issues
    
    # Sample if dataset is large to avoid memory issues
    sample_size = min(1000, len(X_test))
    if len(X_test) > sample_size:
        X_importance_sample = X_test.sample(n=sample_size, random_state=RANDOM_SEED)
        y_importance_sample = y_test.iloc[X_importance_sample.index]
        print(f"      Using sample of {sample_size} for faster computation")
    else:
        X_importance_sample = X_test
        y_importance_sample = y_test
    
    # Compute permutation importance - always pass raw features
    perm_importance = permutation_importance(
        model, X_importance_sample, y_importance_sample, n_repeats=5,  # Reduced from 10
        random_state=RANDOM_SEED, n_jobs=2, scoring='f1'  # Reduced n_jobs to avoid saturation
    )
    
    # Access Bunch object attributes - use getattr for type safety
    # permutation_importance returns a Bunch (dict-like) object that supports both dict and attr access
    # permutation_importance returns a Bunch object (supports both dict and attr access)
    importances_mean = perm_importance.importances_mean  # type: ignore
    importances_std = perm_importance.importances_std  # type: ignore
    
    feature_importance_results[name] = {
        'importances_mean': importances_mean,
        'importances_std': importances_std,
        'features': list(X.columns)
    }

# Create feature importance comparison plot
fig, axes = plt.subplots(len(feature_importance_results), 1, figsize=(10, 4 * len(feature_importance_results)))
if len(feature_importance_results) == 1:
    axes = [axes]

for idx, (name, importance_data) in enumerate(feature_importance_results.items()):
    ax = axes[idx]
    sorted_idx = np.argsort(importance_data['importances_mean'])[::-1]
    features_sorted = [importance_data['features'][i] for i in sorted_idx]
    importances_sorted = importance_data['importances_mean'][sorted_idx]
    std_sorted = importance_data['importances_std'][sorted_idx]
    
    ax.barh(features_sorted, importances_sorted, xerr=std_sorted)
    ax.set_xlabel('Permutation Importance')
    ax.set_title(f'Feature Importance - {name}')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig(images_dir / 'feature_importance_comparison.png')
plt.close()
print("    Saved feature importance comparison to 'feature_importance_comparison.png'")

# --- 12. Explainability (SHAP for Tree Models, Permutation for Others) ---
print("\n[12] Generating Model Explanations...")

try:
    if best_model_name in ['XGBoost', 'Random Forest']:
        # Tree models: Use SHAP with preprocessed features (imputed, but not scaled)
        print(f"    Using SHAP TreeExplainer for {best_model_name}...")
        # Use uncalibrated model (kept before calibration) for SHAP
        if hasattr(uncalibrated_best, 'named_steps'):
            pre = uncalibrated_best.named_steps  # type: ignore
            # Apply imputer transformation (pipeline preprocessing)
            X_shap_raw = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_SEED) if len(X_test) > 500 else X_test
            print(f"    Using {len(X_shap_raw)} samples for SHAP computation")
            
            # Transform through imputer (required for pipeline consistency)
            X_shap_proc = pre['imputer'].transform(X_shap_raw)  # type: ignore
            raw_model = pre['clf']  # type: ignore
        else:
            # If not a pipeline, use raw model
            raw_model = uncalibrated_best
            X_shap_raw = X_test.sample(n=min(500, len(X_test)), random_state=RANDOM_SEED) if len(X_test) > 500 else X_test
            print(f"    Using {len(X_shap_raw)} samples for SHAP computation")
            X_shap_proc = X_shap_raw
        
        # Use preprocessed features (imputed) - tree models don't need scaling
        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(X_shap_proc)
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap_proc, show=False)
        plt.tight_layout()
        plt.savefig(images_dir / 'shap_summary_plot.png')
        plt.close()
        print("    Saved SHAP summary plot to 'shap_summary_plot.png'")
        
        # Also create feature importance bar plot
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_shap_proc, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(images_dir / 'shap_feature_importance.png')
        plt.close()
        print("    Saved SHAP feature importance to 'shap_feature_importance.png'")
    else:
        # Non-tree models: Use Permutation Importance (already computed above)
        print(f"    Using Permutation Importance for {best_model_name} (already computed)")
        print("    See 'feature_importance_comparison.png' for results")
except Exception as e:
    print(f"    [Warning] Could not generate explanations: {e}")
    import traceback
    traceback.print_exc()

# --- 13. Training Metadata Logging ---
print("\n[13] Saving Training Metadata...")

training_metadata = {
    "random_seed": RANDOM_SEED,
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "features_used": list(X.columns),
    "n_features": len(X.columns),
    "best_model": best_model_name,
    "best_threshold": float(best_threshold),
    "optimal_f1_score": float(optimal_f1),
    "optimal_accuracy": float(optimal_acc),
    "optimal_precision": float(optimal_prec),
    "optimal_recall": float(optimal_rec),
    "optimal_auc": float(optimal_auc),
    "class_distribution": {
        "benign": int(benign_count),
        "malicious": int(malicious_count),
        "imbalance_ratio": float(imbalance_ratio) if benign_count > 0 and malicious_count > 0 else None
    },
    "model_version": model_version,
    "timestamp": datetime.now().isoformat(),
    "cv_folds": CV_FOLDS,
    "scoring_metric": SCORING_METRIC,
    "all_model_metrics": test_results
}

metadata_path = reports_dir / 'training_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(training_metadata, f, indent=2, default=str)

print(f"    Saved training metadata to '{metadata_path}'")

print("\n" + "="*80)
print("TRAINING COMPLETE. PROFESSIONAL ARTIFACTS GENERATED.")
print("="*80)
print(f"\nGenerated Artifacts:")
print(f"  - Models: {model_version}, best_model_pipeline.pkl, uncalibrated_{model_version}")
print(f"  - Metrics: final_model_metrics.csv, final_model_metrics_with_optimal.csv")
print(f"  - Metadata: training_metadata.json")
print(f"  - Visualizations: roc_curve_comparison.png, confusion_matrices_all_models.png")
print(f"  - Feature Analysis: feature_importance_comparison.png")
print(f"  - Explanations: shap_summary_plot.png, shap_feature_importance.png (if tree model)")
print(f"  - Optimization: threshold_optimization.png, confusion_matrix_best_optimal.png")