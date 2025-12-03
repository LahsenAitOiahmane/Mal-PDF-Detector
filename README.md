## Mal-PDF-Detector – End‑to‑End PDF Malware Detection Pipeline

This directory contains a **complete, production‑grade pipeline** for detecting malicious PDF files, starting from the public **CIC‑Evasive‑PDFMal2022** dataset and ending with a **calibrated, explainable ML model** plus rich forensic reports.

The pipeline has four main stages:
- **1. Data acquisition & raw feature extraction** – turn raw PDFs into a structured CSV.
- **2. Exploratory Data Analysis (EDA)** – understand the data, engineer features, and decide what to keep/drop.
- **3. Final dataset creation** – build a clean, optimized feature table for modeling.
- **4. Model training & explainability** – train multiple models, pick the best one, calibrate it, and generate reports.

All results are saved under the local `results/` folder so nothing leaks outside this directory.

---

## 1. Data Acquisition & Raw Feature Extraction

### 1.1. Source dataset: CIC‑Evasive‑PDFMal2022

First download the benign and malicious PDF files from the **CIC‑Evasive‑PDFMal2022** dataset (or a similar PDF corpus) and place them under a structure like:

- `Data/Benign/` – benign PDF files (with `.pdf` extension)
- `Data/Malicious/` – malicious PDF files (often without `.pdf` extension)

This repository does **not** include the raw PDFs themselves; it only contains the code to extract and analyze them.

### 1.2. Raw feature extraction – `pdf_analyzer.py`

**Script:** `pdf_analyzer.py`  
**Goal:** Read every PDF file and convert it into a single row of numerical features.

**What it does:**
- Scans two folders:
  - `Data/Benign` → labeled as class `0` (**benign**)
  - `Data/Malicious` → labeled as class `1` (**malicious**)
- For each file, it:
  - Reads the file bytes.
  - Computes **file size** (in bytes).
  - Extracts **PDF version** from the header (e.g. `%PDF-1.4` → `1.4`).
  - Computes **Shannon entropy** of the bytes (a measure of randomness/complexity).
  - Counts occurrences of important **PDF keywords**:
    - `/JS`, `/JavaScript`, `/AA`, `/OpenAction`, `/Launch`,
      `/EmbeddedFile`, `/URI`, `/ObjStm`
  - Attaches the **class label**: `0` for benign, `1` for malicious.

**Output:**
- A single CSV:
  - `pdf_features.csv` (in the project root, next to the scripts)
  - Columns:
    - `file_name`, `file_size`, `pdf_version`, `entropy`,
      `keyword_JS`, `keyword_JavaScript`, `keyword_AA`,
      `keyword_OpenAction`, `keyword_Launch`, `keyword_EmbeddedFile`,
      `keyword_URI`, `keyword_ObjStm`, `class`

**How to run:**

```bash
cd Mal-PDF-Detector
python pdf_analyzer.py
```

If paths differ from `Data/Benign` and `Data/Malicious`, edit `main()` in `pdf_analyzer.py`.

---

## 2. Exploratory Data Analysis (EDA) – `eda_analysis.py`

**Script:** `eda_analysis.py`  
**Goal:** Understand the dataset, engineer better features, detect redundancy, and produce recommendations.

**What it does:**
1. **Loads the raw feature CSV**
   - Input: `pdf_features.csv`

2. **Creates an organized results folder structure**
   - `results/images/` – plots (box plots, violin plots, heatmaps, SHAP images)
   - `results/csv/` – engineered/processed datasets
   - `results/reports/` – text/markdown reports

3. **Zero‑variance check**
  - Identifies features that are constant (e.g., always 0) and thus **not useful** for modeling.

4. **Feature engineering (before analysis)**
   - Adds **engineered features**:
     - `log_file_size = log(1 + file_size)`  
       → Reveals patterns hidden by raw size (malicious PDFs are often tiny; benign ones can be large).
     - `keyword_sum` = sum of all keyword counts.
     - `keyword_density = keyword_sum / (log_file_size + 1)`.
     - `entropy_density = entropy / (log_file_size + 1)`  
       → High entropy in a very small file is a strong malware signal.
   - Ensures `pdf_version` is numeric.

5. **Bivariate analysis (feature vs. class)**
   - Generates **box plots** and **violin plots** comparing feature distributions between benign vs malicious.
   - Computes **separation metrics** for each feature:
     - **Median difference** and direction (which class has higher values).
     - **Cohen’s D** (effect size – how strongly the distributions differ).
     - **Mann–Whitney U test p‑value** (statistical significance).
   - Outputs:
     - `results/images/boxplots_by_class.png`
     - `results/images/violinplots_by_class.png`

6. **Multicollinearity and redundancy**
   - Computes a **correlation matrix** of all numerical features (including engineered ones).
   - Highlights:
     - Features highly correlated with the **target** (`class`).
     - Feature pairs that are highly correlated with each other (redundant).
   - Outputs:
     - `results/images/correlation_heatmap.png`

7. **EDA summary report**
   - Writes a human‑readable text report:
     - Top features by Cohen’s D and correlation with the target.
     - Redundant feature pairs and recommendations for which to keep/drop.
     - Performance of engineered features.
     - A prioritized feature list (Priority 1 / Priority 2 / to exclude).
   - Output:
     - `results/reports/eda_summary_report.txt`

8. **Engineered dataset**
   - Saves a CSV with all original + engineered features:
     - `results/csv/pdf_features_with_engineered.csv`

**High‑level findings (from the reports):**
- **Top predictive features:**
  - `keyword_OpenAction` – very strong indicator of malware.
  - `log_file_size` – benign files are typically much larger.
  - `entropy_density` – captures packed/obfuscated malware.
  - `pdf_version` – malware prefers older PDF versions.
- **Hidden gem:** `keyword_ObjStm` – zero difference in medians (Cohen’s D = 0), but a strong negative correlation with the target; benign PDFs often use object streams more.

---

## 3. Final Dataset Creation – `create_final_dataset.py`

**Script:** `create_final_dataset.py`  
**Goal:** Convert the raw + engineered dataset into a clean, minimal, non‑redundant feature table for model training.

**What it does:**
1. **Loads** `pdf_features.csv`.
2. **Re‑creates key engineered features**:
   - `log_file_size`
   - `entropy_density`
   - Numeric `pdf_version`
3. **Selects final feature set** based on EDA and `FINAL_FEATURE_RECOMMENDATIONS.md`:
   - Priority 1 (must‑have):
     - `keyword_OpenAction`
     - `log_file_size`
     - `entropy_density`
     - `pdf_version`
   - Priority 2 (should‑have):
     - `keyword_JavaScript`
     - `entropy`
   - Hidden gem:
     - `keyword_ObjStm`
   - Plus:
     - `file_name` (for traceability)
     - `class` (target label)
4. **Cleans the data:**
   - Replaces any `NaN` values with 0.
   - Detects and replaces any `±inf` values (which can appear from divisions) with 0.
5. **Saves two CSVs:**
   - Full optimized dataset:
     - `results/csv/pdf_features_final.csv`
     - 7 core features + `file_name` + `class`
   - Minimal four‑feature dataset (Priority 1 only):
     - `results/csv/pdf_features_minimal.csv`

**Companion report:** `results/reports/final_dataset_report.txt`
- Summarizes:
  - Original vs final dataset shapes.
  - Which features were included and why.
  - Correlations of each final feature with the target.

---

## 4. Model Training & Explainability – `train_model.py`

**Script:** `train_model.py`  
**Goal:** Train multiple machine‑learning models on the final dataset, select the best one, calibrate it, optimize the decision threshold, and produce detailed metrics and explanations.

### 4.1. Inputs & setup

- Input dataset:
  - `results/csv/pdf_features_final.csv`
- Uses a **fixed random seed** for reproducibility.
- Creates (if not already present):
  - `results/models/` – serialized models (`.pkl`).
  - `results/images/` – plots (ROC curves, SHAP, confusion matrices, etc.).
  - `results/reports/` – CSV metrics, metadata, text reports.

### 4.2. Data preparation

- Splits features and target:
  - `X` = all columns except `class` and `file_name`.
  - `y` = `class`.
- Validates label encoding:
  - Maps textual labels (`'benign'` / `'malicious'`) to integers (0 / 1) if needed.
- Replaces any `±inf` with `NaN` and leaves NaNs to be handled by **in‑pipeline imputers** (avoids data leakage).
- Checks class balance and prints the **imbalance ratio** (e.g., 1.63:1).

### 4.3. Data split (70/15/15)

- Stratified 3‑way split:
  - **Train (70%)** – used for fitting + cross‑validation.
  - **Validation (15%)** – untouched during hyperparameter search; used to:
    - Compare models.
    - Optimize the final decision threshold.
  - **Test (15%)** – **strictly unseen**; used only for final evaluation.

### 4.4. Models and pipelines

All models are wrapped in **scikit‑learn Pipelines** with an appropriate **SimpleImputer** (and **StandardScaler** only where needed) to avoid data leakage.

Models trained:
- **Logistic Regression**
- **Support Vector Machine (SVM)** with `probability=True`
- **Random Forest**
- **XGBoost**
- **Neural Network (MLPClassifier)**

Key configuration:
- Tree models (**Random Forest, XGBoost**) are **not scaled** (scale‑invariant by design) but do use an imputer.
- Linear/NN models (**Logistic Regression, SVM, MLP**) are **imputed + scaled**.
- XGBoost uses a computed `scale_pos_weight` to handle class imbalance.
- Hyperparameters for SVM, RF, XGBoost, and NN are tuned with **RandomizedSearchCV** using **Stratified K‑Fold (5‑fold)** and optimizing **F1‑score**.

### 4.5. Validation & test evaluation

For each model:
- Evaluates on the **validation set**:
  - F1‑score and AUC (ROC‑AUC).
- Evaluates on the **test set**:
  - **Accuracy, Precision, Recall, F1‑Score, ROC‑AUC**.
  - Saves a detailed classification report for each model.

Key outputs:
- `results/reports/final_model_metrics.csv` – metrics per model on the test set.
- `results/images/roc_curve_comparison.png` – ROC curves for all models.
- `results/reports/classification_reports.txt` – pretty printed classification reports.

### 4.6. Best Model Selection

- The **best model** is chosen by the **highest F1‑Score** on the test set.
- In our runs, **XGBoost** is typically selected as the best performer.

### 4.7. Threshold optimization (on validation set only)

- Uses the best (uncalibrated) model trained on the **train split only**.
- Sweeps thresholds from 0.1 to 0.9.
- For each threshold \(t\), computes F1 on the **validation set** and selects the threshold with the highest F1.
- This ensures **no data leakage** because:
  - The validation set is not used to fit the model; only to choose the threshold.

Output:
- `results/images/threshold_optimization.png`
- Optimal threshold (e.g., `0.640`) vs default `0.5` printed and saved.

### 4.8. Final refit and calibration

1. **Refit uncalibrated best model** on:
   - `X_train_final = train + validation` (more data).
2. **Calibrate probabilities** using `CalibratedClassifierCV` with sigmoid (Platt scaling).
   - Produces better‑calibrated probability scores, important for security decisions.
3. **Re‑evaluate on test set** using the **optimized threshold** and calibrated probabilities.

Outputs:
- Updated optimal‑threshold metrics:
  - `results/reports/final_model_metrics_with_optimal.csv`
- Saved models:
  - Versioned file: `results/models/model_vYYYYMMDD_HHMMSS.pkl`
  - Alias: `results/models/best_model_pipeline.pkl`
  - Uncalibrated copy: `results/models/uncalibrated_model_vYYYYMMDD_HHMMSS.pkl`

### 4.9. Confusion matrices & feature importance

- **Confusion matrices**:
  - For each model with default threshold = 0.5:
    - `results/images/confusion_matrices_all_models.png`
  - For the best calibrated model with optimal threshold:
    - `results/images/confusion_matrix_best_optimal.png`
- **Permutation feature importance** (for all models):
  - `results/images/feature_importance_comparison.png`
  - Shows how much shuffling each feature harms performance.

### 4.10. SHAP explainability (tree models)

- For the best tree model (usually XGBoost):
  - Uses **SHAP TreeExplainer** on an imputed (but not scaled) feature matrix.
  - Samples up to 500 test examples to keep runtime reasonable.
- Outputs:
  - `results/images/shap_summary_plot.png`
    - Shows which features drive predictions and in what direction.
  - `results/images/shap_feature_importance.png`
    - Bar chart of global feature importance.

### 4.11. Training metadata

- Saves a comprehensive JSON:
  - `results/reports/training_metadata.json`
- Contains:
  - Dataset sizes (train/val/test).
  - Feature list.
  - Best model name and parameters.
  - Optimal threshold.
  - All key metrics.
  - Random seed, timestamp, and model version.

---

## 5. Key Result Files & How to Read Them

Here is how to interpret the most important output files:

- `results/reports/eda_summary_report.txt`
  - Narrative summary of the EDA:
    - Top features, correlations, redundancy.
    - Which engineered features worked and which did not.
  - Use this to **justify** feature choices in presentations or documentation.

- `FINAL_FEATURE_RECOMMENDATIONS.md`
  - Human‑friendly, final design document:
    - Explains why each feature is included/excluded.
    - Provides minimal (4‑feature) and extended (7‑feature) sets.
  - This is your **reference spec** when building or auditing production models.

- `results/reports/final_dataset_report.txt`
  - Explains how `pdf_features_final.csv` and `pdf_features_minimal.csv` were built.
  - Lists each final feature and its correlation with the class.
  - Use this to understand the **exact input schema** for training/inference.

- `results/reports/model_train_report.txt`
  - Logs the entire training run step‑by‑step:
    - CV scores, best parameters per model.
    - Validation and test performance.
    - Threshold optimization, calibration, and final metrics.
  - Use this as a **debug log** and experiment history.

- `results/reports/classification_reports.txt`
  - Pretty classification reports (per model, and best model with optimal threshold):
    - Precision, recall, F1 for `Benign` and `Malicious`.
    - Overall accuracy and macro/weighted averages.
  - Use this to compare trade‑offs between models.

---

## 6. How to Run the Full Pipeline

From the `Mal-PDF-Detector` directory:

1. **Extract raw features from PDFs**

Windows PowerShell:

```powershell
python pdf_analyzer.py
```

2. **Run EDA and generate recommendations**

```powershell
python eda_analysis.py
```

3. **Create final optimized dataset**

```powershell
python create_final_dataset.py
```

4. **Train models and generate all reports**

```powershell
python train_model.py
```

After this, you will find all artifacts under `results/` (CSV, images, reports, and models).

---

## 7. Methods, Models, and Techniques Explained

This section explains the core machine learning concepts, techniques, and algorithms used in this project.

- **EDA (Exploratory Data Analysis):** Structured exploration of the dataset to understand feature distributions, class differences, anomalies, and relationships (correlations). Results guide feature engineering and selection.
- **Feature Engineering:** Creating more informative features from raw data, e.g., `log_file_size` to normalize file size and `entropy_density` to detect packed/obfuscated content.
- **Train/Validation/Test Split (70/15/15):** Stratified partitioning that keeps class balance across splits. The test split remains strictly unseen for the final evaluation; validation is used for model selection checks and threshold tuning.
- **Pipelines:** Scikit‑learn `Pipeline` objects chain preprocessing (imputation, scaling) with a classifier to prevent data leakage and keep reproducibility.
- **Imputation:** Filling missing values (NaN) safely inside the pipeline with `SimpleImputer` to avoid using information from validation/test during training.
- **Scaling:** `StandardScaler` is applied to linear models and neural networks. Tree‑based models (Random Forest, XGBoost) are scale‑invariant and do not use scaling.
- **Cross‑Validation (Stratified K‑Fold):** Robust model evaluation across multiple folds while preserving class proportions; reduces variance in performance estimates.
- **RandomizedSearchCV:** Efficient hyperparameter search that samples from specified ranges/grids and scores using F1 across CV folds.
- **Threshold Optimization:** Selecting a decision threshold on the validation set that maximizes F1 rather than defaulting to 0.5, which can improve precision/recall trade‑offs.
- **Calibration (Platt Scaling):** Using `CalibratedClassifierCV(method='sigmoid')` to produce well‑calibrated probabilities—important in security contexts.
- **Permutation Importance:** Model‑agnostic feature importance by measuring performance drop when a feature’s values are randomly shuffled.
- **SHAP (TreeExplainer):** Explains individual predictions and global feature influence for tree models, helping validate model behavior.

### Reproducibility

- **Random Seed:** A fixed seed (`42`) is set for NumPy, Python’s `random`, and model components to make results repeatable.
- **Deterministic Splits:** Stratified splitting ensures consistent class ratios across train/validation/test.
- **Environment:** Use the provided `requirements.txt`; versions can affect plots and minor metrics. Record the environment when sharing results.

### Algorithms Used

- **Logistic Regression:** Linear classifier with regularization; fast baseline that works well when features are informative and roughly linearly separable after scaling.
- **Support Vector Machine (SVM):** Maximizes the margin between classes; `rbf` or `linear` kernels used. `probability=True` enables ROC‑AUC and thresholding.
- **Random Forest:** Ensemble of decision trees with bagging; robust to noise and captures nonlinear interactions; scale‑invariant.
- **XGBoost (Gradient Boosted Trees):** Powerful boosting algorithm with regularization and class‑imbalance handling via `scale_pos_weight`; typically achieves top performance in tabular problems.
- **Neural Network (MLPClassifier):** Feed‑forward network with `relu/tanh` activations; benefits from scaling and early stopping to avoid overfitting.

### Key Metrics

- **Accuracy:** Overall correctness; can be misleading with class imbalance.
- **Precision:** Of predicted malware, proportion that is truly malware.
- **Recall:** Proportion of actual malware correctly detected.
- **F1‑Score:** Harmonic mean of precision and recall; optimized in this pipeline.
- **ROC‑AUC:** Threshold‑independent measure of separability across all thresholds.

## 8. Glossary of Key Terms

Brief definitions of important terms used in scripts and reports.

- **Entropy:** Shannon entropy of file bytes; higher values indicate more randomness/complexity.
- **Entropy Density:** `entropy / (log_file_size + 1)`; highlights small, high‑entropy (often packed) files.
- **OpenAction:** PDF keyword that triggers actions upon opening; strong malware indicator.
- **Object Streams (ObjStm):** PDF mechanism to compress object data; negatively correlated with malware in this dataset.
- **Scale Pos Weight:** XGBoost parameter to address class imbalance by weighting the minority class.
- **Data Leakage:** Using information from validation/test during training; prevented via pipelines and proper splitting.


  Creating new input variables (features) from raw data to make patterns easier for models to learn. Examples here: `log_file_size`, `entropy_density`.

- **Cohen’s D (effect size)**  
  A standardized measure of how different two groups are. Roughly:
  - 0.2 = small effect
  - 0.5 = medium
  - 0.8+ = large  
  In this project, it measures how strongly a feature separates benign from malicious PDFs.

- **Correlation (Pearson’s r)**  
  A number between −1 and +1 indicating how strongly a feature is linearly related to the target:
  - r > 0: higher values → more likely malicious.
  - r < 0: higher values → more likely benign.
  - |r| close to 1: strong relationship.

- **Multicollinearity / redundancy**  
  When two features are highly correlated with each other, they carry almost the same information. Keeping both can hurt some models and interpretation; this pipeline identifies such pairs and recommends which one to keep.

- **Shannon entropy**  
  A measure of randomness or complexity of data. Higher entropy in a binary file can indicate compressed or encrypted content (often seen in packed malware).

- **Density feature (e.g., entropy_density)**  
  A ratio that normalizes one quantity by another (e.g., entropy per unit of size). Used here to highlight small but very complex files, which are suspicious.

- **Stratified split**  
  When splitting data into train/validation/test, stratification preserves the original class proportions in each split, reducing sampling bias.

- **Cross‑validation (CV)**  
  A technique to estimate how well a model will generalize: the training set is divided into K folds, and the model is trained/tested K times, each time holding out a different fold. Here we use **Stratified K‑Fold** with 5 folds.

- **RandomizedSearchCV**  
  A scikit‑learn tool that samples random combinations of hyperparameters and evaluates each via cross‑validation. More efficient than grid search across large parameter spaces.

- **Hyperparameters**  
  Model settings chosen before training (e.g., number of trees, learning rate). The training script tunes them automatically via `RandomizedSearchCV`.

- **Accuracy**  
  Fraction of all predictions that are correct. In malware detection, high accuracy can be misleading if the dataset is imbalanced.

- **Precision**  
  Of all files predicted as **malicious**, the fraction that are actually malicious.  
  High precision means **few false alarms**.

- **Recall (Sensitivity / True Positive Rate)**  
  Of all truly malicious files, the fraction the model correctly flags as malicious.  
  High recall means **few missed attacks**.

- **F1‑Score**  
  The harmonic mean of precision and recall.  
  It is high only when both precision and recall are high, making it a good single summary metric for malware detection.

- **ROC‑AUC (Area Under the ROC Curve)**  
  Measures how well the model ranks malicious vs benign across all possible thresholds.  
  - 0.5 = random guessing  
  - 1.0 = perfect ranking  
  Higher is better.

- **Confusion matrix**  
  A 2×2 table summarizing predictions:
  - True Benign predicted Benign (TN)
  - Benign predicted Malicious (FP – false positives)
  - Malicious predicted Benign (FN – false negatives)
  - Malicious predicted Malicious (TP)  
  This shows exactly where the model makes mistakes.

- **Calibration / CalibratedClassifierCV**  
  Adjusting a model so that its output probabilities match real‑world frequencies.  
  Example: among files the model assigns 0.8 malicious probability, about 80% should truly be malicious.  
  This project uses **Platt scaling (sigmoid)** via `CalibratedClassifierCV`.

- **Decision threshold**  
  The cutoff on predicted probability used to decide “malicious vs benign”.  
  Default is 0.5, but this pipeline **optimizes the threshold** on a dedicated validation set to maximize F1.

- **Threshold optimization**  
  Systematically trying many thresholds and selecting the one with the best F1 on validation data (unseen during training), then applying that threshold to the test set and future data.

- **Permutation importance**  
  A model‑agnostic method to measure feature importance:
  - Randomly shuffle one feature’s values and see how much model performance drops.
  - Bigger drop → more important feature.

- **SHAP (SHapley Additive exPlanations)**  
  A technique that assigns each feature a contribution to the prediction for each sample, based on cooperative game theory.  
  - **SHAP summary plot** shows which features are globally most important and how they push predictions toward benign or malicious.
  - Used here for **tree models** (XGBoost/Random Forest) via `TreeExplainer`.

- **Model calibration vs thresholding**  
  Calibration improves the quality of predicted probabilities; threshold optimization chooses where to cut those probabilities to turn them into a yes/no decision. This pipeline does **both** using separated data to avoid leakage.

---

This README should give you (or any future analyst/engineer) a clear, end‑to‑end view of what each script does, how the pieces connect, and how to interpret every major result in the `results/` directory.


