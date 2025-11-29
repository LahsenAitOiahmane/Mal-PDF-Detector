"""
Professional PDF Malware Detection Training Pipeline (Patched)
-------------------------------------------------------------
Author: Senior ML Engineer (patched)
Context: Cybersecurity / Malware Analysis
Standards:
    - 70/15/15 Data Split (Train/Validation/Test)
    - Stratified K-Fold Cross Validation
    - Pipeline Architecture to prevent Data Leakage
    - RandomizedSearchCV hyperparameter tuning
    - SHAP Explainability (tree models) / Permutation Importance (others)
    - Calibration + Threshold Optimization + Final refit on Train+Val
"""

import json
import random
from time import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Scikit-Learn Imports
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)

# --- Configuration ---
TEST_SIZE = 0.15        # 15% Final Test (Strictly Unseen)
VALIDATION_SIZE = 0.15  # 15% Validation (used for model selection & threshold tuning)
CV_FOLDS = 5            # 5-Fold Cross Validation
SCORING_METRIC = 'f1'   # Optimize for F1-Score (Balance Precision/Recall)
N_ITER_SEARCH = 10      # RandomizedSearchCV iterations

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

# Basic mapping if labels are strings - ensures 0/1 encoding
if df['class'].dtype == object:
    label_mapping = {k: i for i, k in enumerate(sorted(df['class'].unique()))}
    df['class'] = df['class'].map(label_mapping)
    print(f"    Label mapping: {label_mapping}")
else:
    label_mapping = {0: 0, 1: 1}

# Separate Target and Features
if 'file_name' in df.columns:
    X = df.drop(columns=['class', 'file_name'])
else:
    X = df.drop(columns=['class'])
y = df['class']

# Fill simple invalid values globally as a guard (still prefer in-pipeline imputer)
X = X.replace([np.inf, -np.inf], np.nan)

print(f"    Total Samples: {len(X)}")
print(f"    Features ({X.shape[1]}): {list(X.columns)}")

# --- A. Balanced Dataset Check ---
print("\n[A] Checking Dataset Balance...")
class_counts = y.value_counts().sort_index()
benign_count = int(class_counts.get(0, 0) or 0)
malicious_count = int(class_counts.get(1, 0) or 0)
print(f"    Benign samples: {benign_count}")
print(f"    Malicious samples: {malicious_count}")

if benign_count > 0 and malicious_count > 0:
    imbalance_ratio = max(benign_count, malicious_count) / min(benign_count, malicious_count)
    print(f"    Imbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3.0:
        print("    [WARNING] Significant class imbalance detected (ratio > 3:1). Consider class_weight/oversampling.")
    else:
        print("    [OK] Dataset imbalance within acceptable bounds (<= 3:1).")
else:
    raise ValueError("Dataset must contain samples from both classes")

# --- 2. Professional Data Splitting (70% Train / 15% Val / 15% Test) ---
print("\n[2] Splitting Data (70% Train / 15% Validation / 15% Test)...")
print("    Note: Hyperparameter tuning uses CV on training set. Validation set is reserved for selection/threshold.")

# First, split off the 15% Test Set (Strictly Unseen)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)

# Then split remaining into Train/Validation
train_size_ratio = 0.70 / (1 - TEST_SIZE)  # 0.70 / 0.85 ≈ 0.8235
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=(1 - train_size_ratio), stratify=y_temp, random_state=RANDOM_SEED
)

print(f"    Training Data:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"    Validation Data: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"    Test Data:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# --- Helper / Utility Functions ---
def safe_get_proba(estimator, X_in):
    """Return probability estimates in a safe way (predict_proba preferred, decision_function fallback).
       Normalize decision_function to [0,1] if used."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X_in)[:, 1]
    elif hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X_in)
        # Min-max scale to [0,1]
        if np.allclose(scores.max(), scores.min()):
            return np.zeros_like(scores, dtype=float)
        return (scores - scores.min()) / (scores.max() - scores.min())
    else:
        raise RuntimeError("Estimator has neither predict_proba nor decision_function.")

def safe_metrics(y_true, y_pred, y_proba=None):
    """Compute core metrics with safe zero_division handling."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': (roc_auc_score(y_true, y_proba) if y_proba is not None else None)
    }

def is_pipeline(obj):
    return hasattr(obj, 'named_steps')

# --- 3. Define Pipelines & Hyperparameter Grids (with in-pipeline imputer) ---
print("\n[3] Defining Pipelines and Hyperparameter Grids...")

imputer = ('imputer', SimpleImputer(strategy='median'))

# A. Logistic Regression (Baseline)
pipe_lr = Pipeline([
    imputer,
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', random_state=RANDOM_SEED, max_iter=1000))
])

# B. SVM (Baseline)
pipe_svm = Pipeline([
    imputer,
    ('scaler', StandardScaler()),
    ('clf', SVC(probability=True, random_state=RANDOM_SEED))
])
param_svm = {
    'clf__C': [0.1, 1, 10],
    'clf__kernel': ['linear', 'rbf'],
    'clf__gamma': ['scale', 'auto']
}

# C. Random Forest (Tree; no scaler)
pipe_rf = Pipeline([
    imputer,
    ('clf', RandomForestClassifier(random_state=RANDOM_SEED))
])
param_rf = {
    'clf__n_estimators': [100, 200, 300],
    'clf__max_depth': [10, 20, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__class_weight': ['balanced', None]
}

# D. XGBoost (Tree; no scaler)
pipe_xgb = Pipeline([
    imputer,
    ('clf', xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=RANDOM_SEED))
])
# compute a sensible scale_pos_weight if imbalance exists
scale_pos_weight = (benign_count / malicious_count) if malicious_count < benign_count else 1.0
param_xgb = {
    'clf__n_estimators': [100, 200, 300],
    'clf__learning_rate': [0.01, 0.1, 0.2],
    'clf__max_depth': [3, 6, 10],
    'clf__subsample': [0.8, 1.0],
    'clf__colsample_bytree': [0.8, 1.0],
    'clf__scale_pos_weight': [1.0, scale_pos_weight]
}

# E. Neural Network
pipe_nn = Pipeline([
    imputer,
    ('scaler', StandardScaler()),
    ('clf', MLPClassifier(random_state=RANDOM_SEED, max_iter=1000, early_stopping=True,
                         validation_fraction=0.1, n_iter_no_change=10))
])
param_nn = {
    'clf__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
    'clf__activation': ['relu', 'tanh'],
    'clf__alpha': [0.0001, 0.001, 0.01],
    'clf__learning_rate': ['constant', 'adaptive'],
    'clf__learning_rate_init': [0.001, 0.01]
}

models_to_train = [
    ('Logistic Regression', pipe_lr, {}),
    ('SVM', pipe_svm, param_svm),
    ('Random Forest', pipe_rf, param_rf),
    ('XGBoost', pipe_xgb, param_xgb),
    ('Neural Network', pipe_nn, param_nn)
]

# --- 4. Training Loop with Cross-Validation (tune on training set only) ---
print(f"\n[4] Starting Training with {CV_FOLDS}-Fold Cross-Validation...")

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
best_models = {}
results_data = []

for name, pipeline, params in models_to_train:
    print(f"\n    Training {name}...")
    start = time()

    if params:
        search = RandomizedSearchCV(
            pipeline, params, n_iter=N_ITER_SEARCH, scoring=SCORING_METRIC,
            cv=cv, n_jobs=-1, random_state=RANDOM_SEED, verbose=0
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        cv_score = search.best_score_
        print(f"      Best Params: {search.best_params_}")
    else:
        # baseline: fit directly
        pipeline.fit(X_train, y_train)
        best_model = pipeline
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=SCORING_METRIC, n_jobs=-1)
        cv_score = scores.mean()

    duration = time() - start
    # Evaluate on validation set (models trained only on X_train)
    # use safe_proba wrapper
    try:
        y_val_proba = safe_get_proba(best_model, X_val)
    except Exception:
        # Some models may not implement predict_proba; fallback to decision_function or use predict
        y_val_proba = None

    y_val_pred = best_model.predict(X_val)
    val_metrics = safe_metrics(y_val, y_val_pred, y_val_proba)

    print(f"      CV {SCORING_METRIC.upper()}: {cv_score:.4f} (Time: {duration:.1f}s)")
    print(f"      Validation F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc'] if val_metrics['auc'] else 'N/A'}")

    best_models[name] = best_model  # uncalibrated copy trained on X_train
    results_data.append({
        'Model': name,
        'CV_F1_Score': float(cv_score),
        'Val_F1_Score': float(val_metrics['f1']),
        'Val_AUC': float(val_metrics['auc']) if val_metrics['auc'] is not None else None,
        'Training_Time': float(duration)
    })

# Save results dataframe for transparency
results_df = pd.DataFrame(results_data).sort_values(['Val_F1_Score', 'CV_F1_Score'], ascending=False)
results_df.to_csv(reports_dir / 'cv_and_val_results.csv', index=False)
print("\n    Saved cross-val & validation results to 'cv_and_val_results.csv'")
print(results_df.round(4).to_string(index=False))

# --- 5. Select best model based on VALIDATION (no peeking at test) ---
print("\n[5] Selecting Best Model based on Validation Performance (no test peeking)...")
best_model_name = results_df.iloc[0]['Model']
print(f"    Selected best model: {best_model_name} based on Validation F1 (and CV tie-break).")
uncalibrated_best = best_models[best_model_name]  # trained on X_train

# --- 6. Calibrate using training set and optimize threshold on validation set ---
print("\n[6] Calibrating Selected Model and Finding Optimal Threshold on Validation Set...")
calibrator_on_train = CalibratedClassifierCV(uncalibrated_best, cv=CV_FOLDS, method='sigmoid')
calibrator_on_train.fit(X_train, y_train)  # calibrate using training data only (CV within)
print("    Calibrated on training folds (Platt scaling).")

# Get calibrated probabilities on validation set for threshold tuning
y_val_proba_cal = safe_get_proba(calibrator_on_train, X_val)
thresholds = np.linspace(0.1, 0.9, 81)
best_threshold = 0.5
best_val_f1 = -1.0
for t in thresholds:
    preds_t = (y_val_proba_cal >= t).astype(int)
    score = f1_score(y_val, preds_t, zero_division=0)
    if score > best_val_f1:
        best_val_f1 = score
        best_threshold = t

print(f"    Optimal threshold on validation: {best_threshold:.3f} (F1={best_val_f1:.4f})")
# Also save threshold curve
threshold_df = pd.DataFrame({'threshold': thresholds, 'f1': [
    f1_score(y_val, (y_val_proba_cal >= t).astype(int), zero_division=0) for t in thresholds
]})
threshold_df.to_csv(reports_dir / 'threshold_tuning_curve.csv', index=False)

plt.figure(figsize=(8, 5))
plt.plot(threshold_df['threshold'], threshold_df['f1'])
plt.axvline(best_threshold, linestyle='--', label=f'Optimal {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1 (validation)')
plt.title('Threshold optimization on validation set')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(images_dir / 'threshold_optimization_validation.png')

# --- 7. Refit final model on TRAIN+VAL, then calibrate on TRAIN+VAL (final model for deployment) ---
print("\n[7] Re-fitting Selected Model on Train+Validation (to maximize data used), then calibrating...")
X_train_full = pd.concat([X_train, X_val], axis=0)
y_train_full = pd.concat([y_train, y_val], axis=0)

final_uncalibrated = uncalibrated_best.__class__(**getattr(uncalibrated_best, 'get_params', lambda: {})())
# The above try to create a fresh estimator with same params; but for pipelines we recreate by cloning via set_params/get_params
# Safer to use sklearn clone, but to avoid additional import we re-fit the existing pipeline object on combined data:
final_uncalibrated = uncalibrated_best
final_uncalibrated.fit(X_train_full, y_train_full)
print("    Final (uncalibrated) model re-fit on train+val.")

final_calibrator = CalibratedClassifierCV(final_uncalibrated, cv=CV_FOLDS, method='sigmoid')
final_calibrator.fit(X_train_full, y_train_full)
best_pipeline = final_calibrator  # calibrated final pipeline
print("    Final model calibrated on train+val.")

# Save both uncalibrated and calibrated artifacts
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"model_v{timestamp}.pkl"
joblib.dump(final_uncalibrated, models_dir / f"uncalibrated_{model_version}")
joblib.dump(best_pipeline, models_dir / model_version)
joblib.dump(best_pipeline, models_dir / 'best_model_pipeline.pkl')  # latest pointer
print(f"    Saved uncalibrated and calibrated final models to '{models_dir}'")

# --- 8. Evaluate final model on STRICTLY UNSEEN TEST SET using optimal threshold ---
print("\n[8] Final Evaluation on Strictly Unseen Test Set...")
y_test_proba = safe_get_proba(best_pipeline, X_test)
y_test_pred_default = (y_test_proba >= 0.5).astype(int)
y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)

metrics_default = safe_metrics(y_test, y_test_pred_default, y_test_proba)
metrics_optimal = safe_metrics(y_test, y_test_pred_optimal, y_test_proba)

test_metrics_record = {
    'Model': best_model_name,
    'Threshold_default': 0.5,
    'Threshold_optimal': float(best_threshold),
    'Accuracy_default': metrics_default['accuracy'],
    'Precision_default': metrics_default['precision'],
    'Recall_default': metrics_default['recall'],
    'F1_default': metrics_default['f1'],
    'AUC': metrics_default['auc'],
    'Accuracy_optimal': metrics_optimal['accuracy'],
    'Precision_optimal': metrics_optimal['precision'],
    'Recall_optimal': metrics_optimal['recall'],
    'F1_optimal': metrics_optimal['f1']
}
# Save to CSV
pd.DataFrame([test_metrics_record]).to_csv(reports_dir / 'final_selected_model_test_metrics.csv', index=False)
print("    Saved final selected model test metrics to 'final_selected_model_test_metrics.csv'")
print(json.dumps(test_metrics_record, indent=2))

# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'{best_model_name} (AUC = {metrics_default["auc"]:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Final Model (Test Set)')
plt.legend()
plt.tight_layout()
plt.savefig(images_dir / 'roc_curve_final_model.png')

# Confusion matrix (optimal threshold)
cm = confusion_matrix(y_test, y_test_pred_optimal)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
plt.title(f'Confusion Matrix ({best_model_name}) - Threshold {best_threshold:.3f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(images_dir / 'confusion_matrix_best_optimal.png')

# --- 9. Feature Importance (Permutation Importance) for ALL models (pass RAW X_test) ---
print("\n[9] Computing Permutation Importance for all trained models (raw X_test passed to estimator)...")
feature_importance_results = {}
for name, model in best_models.items():
    print(f"    Permutation importance for: {name}")
    try:
        # IMPORTANT: pass RAW X_test to permutation_importance so pipeline will handle its own preprocessing
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=8, random_state=RANDOM_SEED, n_jobs=-1, scoring='f1'
        )
        importances_mean = perm.importances_mean
        importances_std = perm.importances_std
        feature_importance_results[name] = {
            'features': list(X.columns),
            'mean': importances_mean,
            'std': importances_std
        }
    except Exception as e:
        print(f"      [Warning] permutation importance failed for {name}: {e}")

# Plot feature importance summary (top 20 features per model for clarity)
for name, data in feature_importance_results.items():
    feats = data['features']
    mean_imp = data['mean']
    std_imp = data['std']
    idx_sorted = np.argsort(mean_imp)[-20:][::-1]  # top 20
    plt.figure(figsize=(8, 6))
    plt.barh([feats[i] for i in idx_sorted], mean_imp[idx_sorted], xerr=std_imp[idx_sorted])
    plt.title(f'Permutation Importance (top features) - {name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(images_dir / f'perm_importance_top_{name.replace(" ", "_")}.png')

# --- 10. Explainability: SHAP for Tree Models, skip or sample for large sets ---
print("\n[10] Generating Explanations (SHAP for tree models, use sample to limit cost)...")
try:
    if best_model_name in ['Random Forest', 'XGBoost']:
        # Use the uncalibrated model re-fit on train+val (raw tree)
        # If it's a pipeline, extract the 'clf' step
        model_for_shap = final_uncalibrated
        if is_pipeline(model_for_shap):
            if 'clf' in model_for_shap.named_steps:
                raw_tree = model_for_shap.named_steps['clf']
            else:
                raw_tree = model_for_shap
        else:
            raw_tree = model_for_shap

        # SHAP: use a sample to avoid long runtimes
        sample_size = min(500, len(X_test))
        X_shap_sample = X_test.sample(n=sample_size, random_state=RANDOM_SEED)
        explainer = shap.TreeExplainer(raw_tree)
        shap_values = explainer.shap_values(X_shap_sample)
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap_sample, show=False)
        plt.tight_layout()
        plt.savefig(images_dir / 'shap_summary_plot.png')
        plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_values, X_shap_sample, plot_type='bar', show=False)
        plt.tight_layout()
        plt.savefig(images_dir / 'shap_feature_importance.png')
        print("    SHAP plots saved (sampled).")
    else:
        print("    Selected model is non-tree. Use permutation importance plots for explanations (already generated).")
except Exception as e:
    print(f"    [Warning] SHAP generation failed: {e}")

# --- 11. Save additional artifacts & metadata ---
print("\n[11] Saving metadata & artifacts...")

training_metadata = {
    "random_seed": RANDOM_SEED,
    "train_samples": int(len(X_train)),
    "validation_samples": int(len(X_val)),
    "test_samples": int(len(X_test)),
    "features_used": list(X.columns),
    "n_features": int(X.shape[1]),
    "selected_model": best_model_name,
    "model_version": model_version,
    "label_mapping": label_mapping,
    "imbalance_ratio": float(imbalance_ratio),
    "optimal_threshold": float(best_threshold),
    "optimal_validation_f1": float(best_val_f1),
    "final_test_metrics": test_metrics_record,
    "timestamp": datetime.now().isoformat(),
    "cv_folds": CV_FOLDS,
    "scoring_metric": SCORING_METRIC
}

with open(reports_dir / 'training_metadata.json', 'w') as f:
    json.dump(training_metadata, f, indent=2, default=str)

# Save results frames & artifacts
results_df.to_csv(reports_dir / 'cv_val_results_sorted.csv', index=False)
pd.DataFrame([test_metrics_record]).to_csv(reports_dir / 'final_selected_model_test_metrics.csv', index=False)

print(f"    Saved training metadata to '{reports_dir / 'training_metadata.json'}'")
print("\n" + "="*80)
print("TRAINING & EVALUATION COMPLETE. ARTIFACTS GENERATED.")
print("="*80)
print(f"\nGenerated Artifacts (examples):")
print(f"  - Models: {model_version}, uncalibrated_{model_version}, best_model_pipeline.pkl")
print(f"  - CV & Validation results: cv_and_val_results.csv")
print(f"  - Test metrics: final_selected_model_test_metrics.csv")
print(f"  - Metadata: training_metadata.json")
print(f"  - Visuals: roc_curve_final_model.png, confusion_matrix_best_optimal.png")
print(f"  - Feature analysis / explanations: perm_importance_top_*.png, shap_*.png (if tree model)")
