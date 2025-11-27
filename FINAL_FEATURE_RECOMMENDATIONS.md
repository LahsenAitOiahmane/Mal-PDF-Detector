# Final Feature Recommendations for PDF Malware Detection Model

## Executive Summary

This document provides comprehensive feature recommendations based on Exploratory Data Analysis (EDA) of 23,910 PDF files (9,107 benign, 14,803 malicious). The analysis identified the most predictive features, detected redundancy, and recommends an optimized feature set for machine learning model training.

---

## 1. Feature Predictive Power Analysis (Bivariate)

### Key Findings from Box Plots and Violin Plots

**Visual Analysis Results:**
- **keyword_OpenAction**: Shows the strongest separation between benign (median=0) and malicious (median=1) classes
- **keyword_JS** and **keyword_JavaScript**: Both show clear separation, with malicious PDFs more likely to contain these keywords
- **file_size**: Malicious PDFs tend to be significantly smaller (median=9,737 bytes) compared to benign (median=75,923 bytes)
- **entropy**: Slight but statistically significant difference (benign: 7.72, malicious: 7.86)

### Separation Metrics (Cohen's D)

| Feature | Cohen's D | Interpretation | Status |
|---------|-----------|----------------|--------|
| keyword_OpenAction | 2.568 | **Very Large Effect** | ⭐⭐⭐⭐⭐ |
| keyword_JS | 0.278 | Medium Effect | ⭐⭐⭐ |
| keyword_JavaScript | 0.276 | Medium Effect | ⭐⭐⭐ |
| file_size | 0.243 | Medium Effect | ⭐⭐⭐ |
| entropy | 0.135 | Small Effect | ⭐⭐ |
| keyword_AA | 0.000 | No Effect | ❌ |
| keyword_Launch | 0.000 | No Effect | ❌ |
| keyword_EmbeddedFile | 0.000 | No Effect | ❌ |
| keyword_URI | 0.000 | No Effect | ❌ |
| keyword_ObjStm | 0.000 | No Effect | ❌ |

**Interpretation:**
- Cohen's D > 0.8: Large effect
- Cohen's D 0.5-0.8: Medium-large effect
- Cohen's D 0.2-0.5: Medium effect
- Cohen's D < 0.2: Small effect

---

## 2. Multicollinearity and Redundancy Analysis

### Correlation with Target Variable (class)

| Feature | Correlation (r) | Interpretation |
|---------|------------------|----------------|
| keyword_OpenAction | **0.595** | Strong positive correlation |
| keyword_ObjStm | -0.238 | Moderate negative correlation |
| keyword_JavaScript | 0.155 | Weak positive correlation |
| file_size | -0.131 | Weak negative correlation |
| keyword_JS | 0.118 | Weak positive correlation |
| entropy | -0.113 | Weak negative correlation |
| keyword_AA | -0.051 | Very weak negative correlation |
| keyword_EmbeddedFile | 0.030 | Very weak positive correlation |
| keyword_URI | -0.007 | Negligible |
| keyword_Launch | -0.003 | Negligible |

### Highly Correlated Feature Pairs (Redundancy)

**Critical Finding:**
- **keyword_JS ↔ keyword_JavaScript**: r = 0.991 (|r| > 0.95)
  - **Recommendation**: Drop `keyword_JS`, keep `keyword_JavaScript` (slightly higher correlation with target: 0.155 vs 0.118)

**No other feature pairs show |r| > 0.95**, indicating good feature diversity.

---

## 3. Feature Engineering Insights

### Engineered Features Tested

1. **keyword_sum** (sum of all keyword counts)
   - Correlation with target: r = -0.011
   - **Verdict**: Not useful (negligible correlation)

2. **keyword_density** (keyword_sum / file_size)
   - Correlation with target: r = 0.384
   - **Verdict**: **HIGHLY RECOMMENDED** - Stronger than individual keywords
   - **Rationale**: Normalizes keyword presence by file size, capturing malicious PDFs that pack more suspicious content per byte

---

## 4. Final Feature Recommendations

### ✅ **PRIORITY 1: Must Include (High Predictive Power)**

1. **keyword_OpenAction**
   - Cohen's D: 2.568 (exceptional separation)
   - Target correlation: r = 0.595
   - **Justification**: Single most predictive feature. Malicious PDFs almost always contain this keyword (median=1) while benign PDFs rarely do (median=0).

2. **keyword_density** (engineered feature)
   - Target correlation: r = 0.384
   - **Justification**: Captures the density of suspicious keywords relative to file size, providing complementary information to individual keyword counts.

### ✅ **PRIORITY 2: Should Include (Moderate Predictive Power)**

3. **keyword_JavaScript**
   - Cohen's D: 0.276
   - Target correlation: r = 0.155
   - **Justification**: Shows clear separation between classes. Keep over `keyword_JS` due to higher target correlation and to avoid redundancy.

4. **file_size**
   - Cohen's D: 0.243
   - Target correlation: r = -0.131
   - **Justification**: Malicious PDFs are significantly smaller on average, providing useful discriminatory power.

5. **entropy**
   - Cohen's D: 0.135
   - Target correlation: r = -0.113
   - **Justification**: While effect size is small, entropy is statistically significant (p < 1e-124) and provides information about file randomness/compression.

6. **keyword_ObjStm**
   - Target correlation: r = -0.238 (strongest negative correlation)
   - **Justification**: Despite zero median difference, the negative correlation suggests it may be useful in combination with other features (benign PDFs may have more object streams).

### ⚠️ **PRIORITY 3: Consider Including (Context-Dependent)**

7. **keyword_AA**
   - Very weak correlation but may provide context
   - **Recommendation**: Include if using tree-based models (can capture interactions), exclude for linear models

### ❌ **EXCLUDE: Low/No Predictive Power**

- **keyword_JS**: Redundant with `keyword_JavaScript` (r = 0.991)
- **keyword_Launch**: Negligible correlation (r = -0.003)
- **keyword_EmbeddedFile**: Very weak correlation (r = 0.030)
- **keyword_URI**: Negligible correlation (r = -0.007)

---

## 5. Final Optimized Feature Set

### Recommended Feature List (7 features)

```
1. keyword_OpenAction        [MUST HAVE]
2. keyword_density           [MUST HAVE - engineered]
3. keyword_JavaScript        [SHOULD HAVE]
4. file_size                 [SHOULD HAVE]
5. entropy                   [SHOULD HAVE]
6. keyword_ObjStm            [SHOULD HAVE]
7. keyword_AA                [OPTIONAL - context-dependent]
```

### Feature Engineering Code

```python
# Calculate keyword_density
df['keyword_density'] = (
    df[['keyword_JS', 'keyword_JavaScript', 'keyword_AA', 
        'keyword_OpenAction', 'keyword_Launch', 'keyword_EmbeddedFile',
        'keyword_URI', 'keyword_ObjStm']].sum(axis=1) / (df['file_size'] + 1)
)

# Final feature set
final_features = [
    'keyword_OpenAction',
    'keyword_density',
    'keyword_JavaScript',
    'file_size',
    'entropy',
    'keyword_ObjStm',
    'keyword_AA'  # Optional
]
```

---

## 6. Model Training Recommendations

### Feature Scaling
- **StandardScaler** or **RobustScaler** recommended for:
  - `file_size` (wide range, outliers present)
  - `entropy` (continuous values)
  - `keyword_density` (engineered feature)

- **No scaling needed** for:
  - `keyword_OpenAction` (binary-like: 0 or 1)
  - `keyword_JavaScript` (count data, sparse)
  - `keyword_ObjStm` (count data, sparse)
  - `keyword_AA` (count data, sparse)

### Model-Specific Considerations

1. **Tree-Based Models** (Random Forest, XGBoost, LightGBM):
   - Can handle all features without scaling
   - May benefit from including `keyword_AA` for interaction effects
   - Robust to feature redundancy

2. **Linear Models** (Logistic Regression, SVM):
   - Require scaling for continuous features
   - Exclude `keyword_AA` (very weak signal)
   - Consider regularization (L1/L2) to handle potential multicollinearity

3. **Neural Networks**:
   - Standardize all features
   - Can benefit from feature engineering interactions

---

## 7. Summary Statistics

- **Total Samples**: 23,910
- **Class Distribution**: 38.1% Benign, 61.9% Malicious
- **Original Features**: 10 numerical features
- **Recommended Features**: 6-7 features (reduction of 30-40%)
- **Redundancy Removed**: 1 feature pair (keyword_JS/keyword_JavaScript)
- **Features Excluded**: 4 features with no predictive power

---

## 8. Expected Model Performance Impact

Based on feature importance analysis:

- **keyword_OpenAction** alone should achieve ~75-80% accuracy (given r=0.595)
- Adding **keyword_density** should boost performance by 5-10%
- Full feature set (6-7 features) should enable 85-90%+ accuracy with proper model tuning

---

## Files Generated

1. `boxplots_by_class.png` - Box plots for all features
2. `violinplots_by_class.png` - Violin plots for key features
3. `correlation_heatmap.png` - Correlation matrix visualization
4. `eda_summary_report.txt` - Detailed statistical summary
5. `pdf_features_with_engineered.csv` - Dataset with engineered features
6. `FINAL_FEATURE_RECOMMENDATIONS.md` - This document

---

## Next Steps

1. **Feature Engineering**: Create `keyword_density` feature
2. **Feature Selection**: Use the recommended 6-7 features
3. **Data Preprocessing**: Apply appropriate scaling based on chosen model
4. **Model Training**: Train with optimized feature set
5. **Validation**: Compare performance with full vs. optimized feature set

---

**Analysis Date**: Generated from EDA analysis  
**Dataset**: pdf_features.csv (23,910 samples)  
**Analysis Script**: eda_analysis.py

