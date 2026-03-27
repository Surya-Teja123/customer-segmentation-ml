import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df_scaled   = pd.read_csv('../data/scaled_data.csv')
    leaderboard = pd.read_csv('../models/final_leaderboard.csv')
    feature_imp = pd.read_csv('../models/feature_importance.csv')
    return df_scaled, leaderboard, feature_imp

@st.cache_resource
def load_models():
    with open('../models/best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('../models/tuned_models.pkl', 'rb') as f:
        tuned_models = pickle.load(f)
    with open('../models/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    return best_model, tuned_models, feature_columns

df_scaled, leaderboard, feature_imp = load_data()
best_model, tuned_models, feature_columns = load_models()

@st.cache_data
def prepare_test_data():
    X = df_scaled[feature_columns]
    y = df_scaled['Response']
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    _, X_test, _, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    return X_test, y_test

X_test, y_test = prepare_test_data()

# ─────────────────────────────────────────
# HEADER + METRICS
# ─────────────────────────────────────────
st.title("🤖 Model Comparison Dashboard")
st.caption("Customer Segmentation & Campaign Response Prediction")

best = leaderboard[leaderboard['Model'] == 'XGBoost (Tuned)'].iloc[0]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🎯 Accuracy",  f"{best['Accuracy']:.2%}")
c2.metric("✅ Precision", f"{best['Precision']:.2%}")
c3.metric("🔍 Recall",    f"{best['Recall']:.2%}")
c4.metric("⚖️ F1 Score",  f"{best['F1 Score']:.2%}")
c5.metric("📈 ROC-AUC",   f"{best['ROC-AUC']:.2%}")

st.markdown("---")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Leaderboard",
    "📈 Metric Charts",
    "📉 ROC & Confusion Matrix",
    "🔍 Feature Importance"
])

# ── TAB 1 — LEADERBOARD ──────────────────
with tab1:
    st.subheader("Model Leaderboard — All Models")

    # Add rank and medal column
    lb = leaderboard.copy()
    lb.insert(0, 'Rank', range(1, len(lb)+1))
    lb['Medal'] = lb['Rank'].map({1:'🥇', 2:'🥈', 3:'🥉'}).fillna('')

    # Style with dark text so it's visible on colored backgrounds
    def highlight_rows(row):
        if row['Model'] == 'XGBoost (Tuned)':
            return ['color: #00ff00; font-weight: bold'] * len(row)
        elif 'Tuned' in str(row['Model']):
            return ['color: #ffd700'] * len(row)
        return [''] * len(row)

    cols_to_show = ['Medal', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    styled = lb[cols_to_show].style.apply(highlight_rows, axis=1).format({
        'Accuracy' : '{:.4f}',
        'Precision': '{:.4f}',
        'Recall'   : '{:.4f}',
        'F1 Score' : '{:.4f}',
        'ROC-AUC'  : '{:.4f}'
    })

    st.dataframe(styled, use_container_width=True, height=430)
    st.caption("🟢 Best model &nbsp; 🟡 Tuned models &nbsp; ⬜ Base models")

# ── TAB 2 — METRIC CHARTS ────────────────
with tab2:
    metric_choice = st.selectbox(
        "Select metric:",
        ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    )

    sorted_df  = leaderboard.sort_values(metric_choice, ascending=True)
    bar_colors = ['#2ECC71' if 'Tuned' in str(m) else '#3498DB'
                  for m in sorted_df['Model']]

    fig, ax = plt.subplots(figsize=(7, 3))
    bars = ax.barh(sorted_df['Model'], sorted_df[metric_choice], color=bar_colors)
    ax.set_xlim(0.5, 1.05)
    ax.set_title(f'{metric_choice} — All Models', fontsize=12)
    for bar, val in zip(bars, sorted_df[metric_choice]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='#2ECC71', label='Tuned'),
        Patch(color='#3498DB', label='Base')
    ])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ── TAB 3 — ROC & CONFUSION MATRIX ───────
with tab3:
    col_roc, col_cm = st.columns(2)

    with col_roc:
        st.markdown("**ROC Curves — Top 3 Tuned Models**")
        fig, ax = plt.subplots(figsize=(4, 4))
        colors_roc = ['#2ECC71', '#3498DB', '#E74C3C']
        for (name, model), color in zip(tuned_models.items(), colors_roc):
            RocCurveDisplay.from_estimator(
                model, X_test, y_test,
                ax=ax, name=name, color=color
            )
        ax.plot([0,1], [0,1], 'k--', label='Random')
        ax.set_title('ROC Curves', fontsize=11)
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_cm:
        st.markdown("**Confusion Matrix — XGBoost (Tuned)**")
        y_pred = best_model.predict(X_test)
        cm     = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Rejected', 'Accepted'],
                    yticklabels=['Rejected', 'Accepted'],
                    ax=ax, annot_kws={'size': 14})
        ax.set_title('Confusion Matrix', fontsize=11)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ── TAB 4 — FEATURE IMPORTANCE ───────────
with tab4:
    col_chart, col_table = st.columns([2, 1])

    with col_chart:
        top_n  = st.slider("Number of features:", 5, 20, 10)
        top_df = feature_imp.head(top_n)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top_df['Feature'][::-1],
                top_df['Importance'][::-1], color='#E74C3C')
        ax.set_title(f'Top {top_n} Features', fontsize=11)
        ax.set_xlabel('Importance Score')
        for i, (val, name) in enumerate(zip(
            top_df['Importance'][::-1], top_df['Feature'][::-1]
        )):
            ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_table:
        st.markdown("**Top 10 Features**")
        st.dataframe(
            feature_imp.head(10)[['Feature', 'Importance']].style.format(
                {'Importance': '{:.4f}'}
            ),
            use_container_width=True,
            height=370
        )

st.markdown("---")
st.caption("Tech Stack: Python · Scikit-learn · XGBoost · TensorFlow · Streamlit")