# Customer Segmentation & Campaign Response Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-latest-yellow)

An end-to-end Machine Learning project that segments customers into meaningful groups and predicts marketing campaign acceptance using 10+ ML models, a Neural Network and interactive Streamlit dashboards.

---

## 📌 Project Overview

A real marketing dataset with 2,236 customers and a 14.9% campaign acceptance rate. The goal was to understand **who accepts marketing campaigns and why** — using the complete Data Science pipeline from raw data to production ready dashboards.

This project covers:
- ✅ Data Preprocessing & Feature Engineering
- ✅ Exploratory Data Analysis
- ✅ Dimensionality Reduction (PCA)
- ✅ Unsupervised Learning — Customer Segmentation
- ✅ Supervised Learning — Campaign Prediction
- ✅ Deep Learning — MLP Neural Network
- ✅ Model Comparison & Hyperparameter Tuning
- ✅ Model Explainability — Feature Importance
- ✅ Interactive Streamlit Dashboards

---

## 📁 Project Structure
```
customer-marketing-ml/
│
├── data/
│   ├── cleaned_data.csv          ← preprocessed data
│   ├── scaled_data.csv           ← scaled data for ML
│   └── clustered_data.csv        ← data with cluster labels
│
├── models/
│   ├── feature_importance.csv    ← top features
│   ├── final_leaderboard.csv     ← all model results
│   └── all_models_results.csv    ← base model results
│
├── notebooks/
│   ├── 01_preprocessing.ipynb    ← data loading & preprocessing
│   ├── 02_eda.ipynb              ← exploratory data analysis
│   ├── 03_clustering.ipynb       ← customer segmentation
│   ├── 04_model_training.ipynb   ← ML models & deep learning
│   └── 05_model_comparison.ipynb ← final comparison & export
│
├── dashboards/
│   ├── dashboard_model_comparison.py    ← technical dashboard
│   └── dashboard_business_insights.py  ← business dashboard
│
├── .gitignore
└── README.md
```

---

## 📊 Dataset

**Source:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)

Download the dataset from Kaggle and place it in the `data/` folder as `marketing_campaign.csv` before running the notebooks.

| Category | Features |
|---|---|
| Demographics | Age, Education, Marital Status |
| Financial | Income |
| Purchasing | Wine, Fruits, Meat, Fish, Sweets, Gold |
| Behaviour | Web, Store, Catalog purchases |
| Marketing | Campaign responses (1-5) |

**Target Variable:** `Response` — 1 = accepted campaign, 0 = rejected

---

## ⚙️ Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/YourUsername/customer-segmentation-ml.git
cd customer-segmentation-ml
```

**2. Install required libraries**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow streamlit imbalanced-learn
```

**3. Download the dataset**
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- Download `marketing_campaign.csv`
- Place it in the `data/` folder

**4. Run the notebooks in order**
```
01_preprocessing.ipynb
02_eda.ipynb
03_clustering.ipynb
04_model_training.ipynb
05_model_comparison.ipynb
```

**5. Run the dashboards**
```bash
cd dashboards
streamlit run dashboard_model_comparison.py
streamlit run dashboard_business_insights.py
```

---

## 🔵 Unsupervised Learning — Customer Segmentation

Applied 4 clustering algorithms and evaluated using Silhouette Score:

| Model | Silhouette Score | Notes |
|---|---|---|
| K-Means (K=4) | 0.3398 | ✅ Selected — stable & interpretable |
| Hierarchical (average) | 0.4193 | Good score but less interpretable |
| DBSCAN | 0.4645 | 14 clusters + 2115 noise points — not suitable |
| GMM (spherical) | 0.3002 | Soft assignments |

**4 Customer Segments discovered:**

| Segment | Count | Avg Income | Avg Spend | Accept Rate |
|---|---|---|---|---|
| Premium Customers | 265 (11.9%) | $75,871 | $1,443 | 35.5% |
| Active Spenders | 440 (19.7%) | $69,924 | $1,176 | 14.3% |
| Mid-tier Regulars | 559 (25.0%) | $56,814 | $666 | 16.6% |
| Budget Shoppers | 972 (43.5%) | $34,501 | $85 | 8.6% |

---

## 🟢 Supervised Learning — Model Results

Handled class imbalance (85/15) using SMOTE before training.

**All Models Comparison:**

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost (Tuned) | 0.9527 | 0.9574 | 0.9474 | 0.9524 | 0.9907 |
| Gradient Boosting (Tuned) | 0.9488 | 0.9523 | 0.9447 | 0.9485 | 0.9901 |
| Random Forest (Tuned) | 0.9474 | 0.9381 | 0.9579 | 0.9479 | 0.9882 |
| XGBoost | 0.9474 | 0.9570 | 0.9368 | 0.9468 | 0.9875 |
| Random Forest | 0.9501 | 0.9407 | 0.9605 | 0.9505 | 0.9862 |
| Gradient Boosting | 0.9343 | 0.9388 | 0.9289 | 0.9339 | 0.9848 |
| AdaBoost | 0.8975 | 0.8932 | 0.9026 | 0.8979 | 0.9674 |
| Logistic Regression | 0.8922 | 0.9139 | 0.8658 | 0.8892 | 0.9566 |
| K-Nearest Neighbors | 0.8371 | 0.7678 | 0.9658 | 0.8555 | 0.9221 |
| Decision Tree | 0.8817 | 0.8816 | 0.8816 | 0.8816 | 0.8849 |
| Naive Bayes | 0.7057 | 0.7321 | 0.6474 | 0.6872 | 0.7814 |
| Support Vector Machine | 0.6675 | 0.7530 | 0.4974 | 0.5990 | 0.7216 |

---

## 🧠 Deep Learning — MLP Neural Network
```
Architecture:
Input → Dense(128) → BatchNorm → Dropout(0.3)
      → Dense(64)  → BatchNorm → Dropout(0.3)
      → Dense(32)  → BatchNorm → Dropout(0.2)
      → Dense(1, sigmoid)

Optimizer : Adam (lr=0.001)
Loss      : Binary Crossentropy
Callbacks : EarlyStopping + ReduceLROnPlateau
Epochs    : 47 (early stopping at best epoch 37)
```

| Model | ROC-AUC | Verdict |
|---|---|---|
| XGBoost (Tuned) | 0.9907 | 🏆 Winner |
| MLP Neural Network | 0.9547 | Good but not better |

> XGBoost outperformed the Neural Network — gradient boosting dominates structured tabular data

---

## 🔍 Top Features — Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | TotalCampaignsAccepted | 0.2458 |
| 2 | EnrollmentYear | 0.1043 |
| 3 | Marital_Status_Single | 0.0504 |
| 4 | Teenhome | 0.0469 |
| 5 | AcceptedCmp2 | 0.0366 |

> Customers who accepted previous campaigns are **459% more likely** to accept again

---

## 📊 Dashboards

**Dashboard 1 — Model Comparison (Technical)**
- Model leaderboard with color coding
- Interactive metric comparison charts
- ROC curves for top 3 models
- Confusion matrix
- Feature importance with adjustable slider

**Dashboard 2 — Business Insights**
- Customer segment overview
- Cluster profile comparison
- Income distribution by segment
- Spending by product category
- Purchase channel analysis
- Campaign history analysis
- Live prediction tool — input customer details and get real time campaign acceptance probability

---

## 💡 Key Business Insights

- Past campaign acceptors are **459% more likely** to accept again
- Premium Customers are only **11.9%** of base but have **35.5%** acceptance rate
- Active Spenders make the most purchases but only **14.3%** acceptance — they shop independently
- Campaign 2 had only **1.3%** acceptance vs **7%+** for all other campaigns
- Wines dominate spending at **$304 avg** — nearly double meat at $167
- Budget Shoppers are **43.5%** of customers but generate the lowest campaign ROI

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualizations |
| Scikit-learn | ML models & preprocessing |
| XGBoost | Best performing model |
| TensorFlow/Keras | Neural network |
| Streamlit | Interactive dashboards |
| SMOTE (imbalanced-learn) | Class imbalance handling |

---

## 👨‍💻 Author

**Surya**
- LinkedIn: [Your LinkedIn URL]
- GitHub: [Your GitHub URL]
