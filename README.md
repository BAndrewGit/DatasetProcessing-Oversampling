# Dataset Processing Pipeline - Financial Behavior Risk Analysis

## 📋 Project Overview

This project implements a comprehensive data processing and analysis pipeline for **financial behavior risk assessment**. The system processes survey data about financial habits, spending patterns, and economic behaviors to predict and analyze financial risk levels.

### Key Features
-  **Data Preprocessing & Normalization** - Translation, encoding, and feature engineering
-  **Advanced Exploratory Data Analysis (EDA)** - Statistical analysis with multiple visualization techniques
-  **Dimensionality Reduction** - PCA analysis for feature extraction
-  **Clustering Analysis** - K-Means and GMM clustering for behavioral segmentation
-  **Machine Learning Models** - Multiple classification algorithms for risk prediction
-  **Data Augmentation** - CTGAN, SMOTE-Tomek, and WGAN-based synthetic data generation

---

## 🗂️ Project Structure

```
Procesare Dataset/
├── FirstProcessing/          # Initial data processing pipeline
│   ├── main.py               # Entry point for data preprocessing
│   ├── preprocessing.py      # Data normalization and translation (RO→EN)
│   ├── risk_calculation.py   # Risk scoring and clustering algorithms
│   ├── encoder.py            # Feature encoding utilities
│   ├── data_generation.py    # Feature engineering and generation
│   └── file_operations.py    # File I/O and Excel formatting
│
├── EDA/                      # Exploratory Data Analysis modules
│   ├── V1/                   # Initial EDA implementation
│   │   ├── mainEDA.py        # Main EDA workflow
│   │   ├── data_loading.py   # Data loading utilities
│   │   ├── preprocessing.py  # Data preprocessing
│   │   ├── visualization.py  # Visualization functions
│   │   └── model_training.py # ML model training and evaluation
│   │
│   └── V2/                   # Enhanced EDA with advanced analytics (CURRENT)
│       ├── mainEDA2.py       # Main EDA workflow with PCA and clustering
│       ├── config.py         # Configuration settings
│       ├── data_loader.py    # Data loading and preparation
│       ├── plot_generator.py # Comprehensive plotting functions
│       ├── utils.py          # Utility functions
│       ├── PCA/              # Principal Component Analysis
│       │   ├── pca_transformer.py  # PCA fitting and transformation
│       │   └── pca_visualizer.py   # PCA visualization (scree plots, loadings)
│       └── clustering/       # Clustering analysis
│           ├── kmeans_clustering.py      # K-Means implementation
│           ├── gmm_clustering.py         # Gaussian Mixture Models
│           ├── cluster_comparison.py     # Cluster method comparison
│           └── cluster_visualizer.py     # Cluster visualization
│
├── DataAugmentation/         # Synthetic data generation
│   ├── base.py               # Base augmentation class
│   ├── CTGan_Augmentation.py # Conditional GAN augmentation
│   ├── smote_tomek.py        # SMOTE-Tomek hybrid sampling
│   └── WC_GAN.py             # Wasserstein GAN augmentation
│
├── Old/                      # Deprecated experiments (not in use)
│   ├── ADASYN_WCGAN_Augmentation.py
│   ├── SMOTE_VAE_Augmentation.py
│   └── WCGAN_Augmentation.py
│
├── scaler/                   # Saved preprocessing models
│   └── robust_scaler.pkl
│
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyCharm IDE (or any Python IDE)
- Virtual environment (recommended)

### Installation

1. **Open the project in PyCharm:**
   - File → Open → Select `Procesare Dataset` folder

2. **Configure Python Interpreter:**
   - File → Settings → Project → Python Interpreter
   - Create new virtual environment or select existing Python 3.10+

3. **Install dependencies:**
   - Open PyCharm Terminal and run:
     ```bash
     pip install -r requirements.txt
     ```
   - Or use PyCharm's automatic dependency detection

---

## 📊 Workflow

**Complete Workflow** - Run modules in PyCharm following this order:

---

### **Step 1: Data Preprocessing** 

**Run:** `FirstProcessing/main.py` (Right-click → Run 'main')

**Purpose:** Transform raw survey data into machine-learning-ready format

**What it does:**
- **Translation:** Romanian survey responses → English
- **Normalization:** Standardize categorical values and ranges
- **Feature Engineering:**
  - Age grouping and income categorization
  - Product lifetime estimation
  - Essential needs percentage calculation
- **Risk Calculation:**
  - Weighted risk scoring based on 15+ financial behavior features
  - GMM clustering for automatic risk level assignment
  - Outlier detection using Isolation Forest
  - Confidence scoring for risk predictions

**Key Components:**
- **Weights-based Risk Scoring:** Multi-factor weighted model considering:
  - Budget planning habits (0.097)
  - Age demographics (0.078)
  - Family status (0.072-0.065)
  - Financial investments (0.071)
  - Impulse buying patterns (0.063-0.068)
  - Savings goals and obstacles (0.055-0.051)

**Input:** Raw survey CSV/Excel (Romanian language)
**Output:** `encoded_data.csv` / `encoded_data.xlsx` with risk scores

---

### **Step 2: Exploratory Data Analysis** 

#### **Option A: Advanced Analysis (⭐ RECOMMENDED)**

**Run:** `EDA/V2/mainEDA2.py` (Right-click → Run 'mainEDA2')

**Purpose:** Comprehensive analysis with dimensionality reduction and clustering

**Features:**

**Univariate Analysis:**
- Distribution plots for all features
- Grouped by categories (demographic, expenses, behaviors, etc.)

**Bivariate Analysis:**
- Correlation heatmaps
- Feature relationships within groups

**Target Analysis:**
- Risk score distribution
- Feature vs. target relationships

**PCA (Principal Component Analysis):**
- Variance threshold: 80% (configurable in `config.py`)
- Scree plots for component selection
- Loading heatmaps showing feature contributions
- Dimensionality reduction for visualization and clustering

**Clustering Analysis:**
- **K-Means Clustering:**
  - Automatic optimal K selection via silhouette score
  - Range: 2-10 clusters (configurable)
  - Cluster visualization in PCA space
- **Gaussian Mixture Models (GMM):**
  - Probabilistic cluster assignment
  - Soft clustering with confidence scores
- **Comparison Metrics:**
  - Adjusted Rand Index (ARI)
  - Agreement rate between methods
  - Crosstab correspondence analysis
  - Risk score distribution by cluster

**Advanced Visualizations:**
- Parallel coordinates plots
- Radar charts for cluster profiles
- Silhouette score comparisons
- Box plots and bar charts for risk distribution

**Configuration** (`EDA/V2/config.py`):
```python
PCA_VARIANCE_THRESHOLD = 0.80      # 80% variance retention
CLUSTERING_K_RANGE = (2, 11)       # K-Means range
DPI = 300                          # High-quality plots
TARGET = "Risk_Score"              # Target variable
```
---

#### **Option B: Basic Analysis**

**Run:** `EDA/V1/mainEDA.py` (Right-click → Run 'mainEDA')

**Purpose:** Basic exploratory data analysis and model evaluation

**Features:**
- Duplicate detection and removal
- Train/test split or synthetic data comparison
- Model training (Logistic Regression, Random Forest, XGBoost, SVM)
- Performance metrics (F1, ROC-AUC, Classification Report)
- Basic visualization

**Input:** `encoded_data.csv` from Step 1
**Output:** Model metrics + basic plots

**Note:** V1 provides foundational analysis but has been superseded by V2 for more detailed insights.

---

### **Step 3: Data Augmentation (Optional)** 

**When to use:** Imbalanced dataset or need more training samples

#### **Method A: SMOTE-Tomek ( Fast & Reliable)**

**Run:** `DataAugmentation/smote_tomek.py` (Right-click → Run)

**What it does:**
- Hybrid oversampling + undersampling
- SMOTE for minority class synthesis
- Tomek links removal for boundary cleaning
- Validation metrics: F1-weighted, silhouette score, Cohen's kappa
- Feature importance ranking via F-statistics

**Best for:**
- Quick augmentation
- Small datasets (<500 samples)
- Pre-processing for WGAN

**Output:** `augmented_dataset_encoded.csv` / `.xlsx`

---

#### **Method B: CTGAN ( High Quality - Recommended)**

**Run:** `DataAugmentation/CTGan_Augmentation.py` (Right-click → Run)

**What it does:**
- Conditional Tabular GAN using SDV library
- Balanced class generation
- Iterative quality validation
- Minimum confidence threshold: 0.8
- Step-wise generation with metrics tracking

**Best for:**
- High-quality synthetic data
- Medium datasets (500-2000 samples)
- Handling categorical features

**Usage:**
```python
from DataAugmentation.CTGan_Augmentation import CTGANAugmentation

augmentor = CTGANAugmentation(
    target_column="Behavior_Risk_Level",
    step_fraction=0.25,
    max_size=2000
)
# Interactive workflow follows
```

---

## 📁 Data Format

### Input Data (Raw Survey)
- **Format:** CSV/Excel (Romanian language)
- **Columns:** ~22 survey questions covering:
  - Demographics (age, gender, family status, income)
  - Financial attitudes and behaviors
  - Budget planning habits
  - Savings goals and obstacles
  - Impulse buying patterns
  - Credit usage and debt levels
  - Investment behaviors
  - Product lifetime usage

### Processed Data (Encoded)
- **Format:** CSV/Excel (English, encoded)
- **Features:** 80+ one-hot encoded binary features
- **Target Variables:**
  - `Risk_Score` (continuous): Weighted risk metric
  - `Behavior_Risk_Level` (binary): 0 = Low Risk, 1 = High Risk
- **Metadata:**
  - `Confidence`: Prediction confidence (0-1)
  - `Cluster`: GMM cluster assignment
  - `Outlier`: Binary outlier flag

---

## 🔧 Technical Details

### Machine Learning Models (EDA/V1 & V2)
1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble decision trees (150 estimators)
3. **XGBoost** - Gradient boosting (100 estimators)
4. **SVM** - Support Vector Machine (with SGD fallback for large datasets)

### Evaluation Metrics
- F1 Score (macro/weighted)
- ROC-AUC
- Classification Report (precision, recall)
- Confusion Matrix
- Cross-Validation (5-fold Stratified)

### Risk Scoring Methodology
1. **Feature Weighting:** 15 most important features identified via SHAP/XGBoost
2. **Normalization:** RobustScaler to handle outliers
3. **Clustering:** Bayesian GMM (2-8 components) on risk scores
4. **Labeling:** Silhouette-optimized cluster assignment
5. **Validation:** KNN classifier verification

---

## 📊 Key Visualizations

### EDA V2 Generates:
1. **Distribution Plots** - Histograms and KDE for all features
2. **Correlation Heatmaps** - Within-group feature relationships
3. **PCA Scree Plot** - Variance explained by components
4. **Loading Heatmap** - Feature contributions to PCs
5. **Cluster Scatter Plots** - K-Means vs GMM in PCA space
6. **Silhouette Comparison** - Cluster quality metrics
7. **Risk Score Box Plots** - Distribution per cluster
8. **Parallel Coordinates** - Multi-dimensional cluster profiles
9. **Radar Charts** - Normalized cluster characteristics
10. **Crosstab Heatmap** - Cluster correspondence matrix

---

## ⚠️ Known Limitations & Notes

- **Old folder:** Contains deprecated ADASYN, VAE, and early WGAN experiments - not maintained
- **EDA V1:** Still functional but superseded by V2 for comprehensive analysis
- **CPU-Only:** Configured for CPU processing (GAN models force `cuda=False`)
- **Language:** Survey data must be in Romanian for FirstProcessing translation

---

## 🛠️ Configuration Files

### `EDA/V2/config.py`
- Feature grouping definitions
- Columns to exclude from analysis
- PCA variance threshold
- Clustering K range
- Plot DPI settings

### `FirstProcessing/main.py` - CONFIG dict
- Risk scoring weights
- Multi-value column definitions

---

## 📚 Dependencies Highlights

- **Data Processing:** pandas, numpy, scipy
- **Machine Learning:** scikit-learn, xgboost, imbalanced-learn
- **Deep Learning:** torch (CPU), sdv (CTGAN)
- **Visualization:** matplotlib, seaborn
- **Utilities:** openpyxl, joblib, tkinter (file dialogs)

See `requirements.txt` for complete list.

---

## 🎯 Typical Usage Workflow

1. **Prepare Data:**
   ```cmd
   python -m FirstProcessing.main
   ```
   - Select raw survey CSV/Excel
   - Outputs encoded dataset with risk scores

2. **Run Advanced EDA:**
   ```cmd
   python -m EDA.V2.mainEDA2
   ```
   - Select processed dataset
   - Choose output directory
   - Generates comprehensive analysis and plots

3. **Optional - Augment Data:**
   ```cmd
   python -m DataAugmentation.CTGan_Augmentation
   ```
   - Select training dataset
   - Define augmentation parameters
   - Outputs balanced synthetic dataset

4. **Optional - Train Models (V1):**
   ```cmd
   python -m EDA.V1.mainEDA
   ```
   - Select test and train datasets
   - Evaluates model performance

---

## 📝 Output Files

### FirstProcessing
- `encoded_data.csv` / `encoded_data.xlsx` - Processed dataset
- `robust_scaler.pkl` - Saved scaler model

### EDA V2
- `plots/` directory with subdirectories per analysis type
- `pca_loadings.csv` - Feature contributions to PCs
- `pca_transformed.csv` - Dataset in PC space
- `kmeans_clustered_dataset.csv` - K-Means assignments
- `gmm_clustered_dataset.csv` - GMM assignments with probabilities
- `cluster_comparison_means.csv` - Feature means per cluster
- `clustering_summary.txt` - Metrics and statistics

### Data Augmentation
- `augmented_dataset_encoded.csv` / `.xlsx` - Synthetic dataset
- `augmentation_metrics.json` - Quality metrics per iteration

---


