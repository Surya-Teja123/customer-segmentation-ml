import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Business Insights Dashboard",
    page_icon="📊",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('../data/clustered_data.csv')
    return df

@st.cache_resource
def load_models():
    with open('../models/best_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    with open('../models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('../models/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    with open('../models/scale_cols.pkl', 'rb') as f:
        scale_cols = pickle.load(f)
    return best_model, scaler, feature_columns, scale_cols

df = load_data()
best_model, scaler, feature_columns, scale_cols = load_models()

# ─────────────────────────────────────────
# HEADER + TOP METRICS
# ─────────────────────────────────────────
st.title("📊 Business Insights Dashboard")
st.caption("Customer Segmentation & Marketing Intelligence")

c1, c2, c3, c4 = st.columns(4)
c1.metric("👥 Total Customers",   f"{len(df):,}")
c2.metric("✅ Campaign Accepted", f"{df['Response'].sum():,}")
c3.metric("📈 Acceptance Rate",   f"{df['Response'].mean():.1%}")
c4.metric("💰 Avg Income",        f"${df['Income'].mean():,.0f}")

st.markdown("---")

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "👥 Customer Segments",
    "📊 Customer Behaviour",
    "📢 Campaign History",
    "🎯 Campaign Prediction"
])

# ── TAB 1 — CUSTOMER SEGMENTS ────────────
with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Segment Overview**")

        # Segment summary table
        seg_summary = df.groupby('Cluster_Label').agg(
            Count         = ('Income', 'count'),
            Avg_Income    = ('Income', 'mean'),
            Avg_Spend     = ('TotalSpend', 'mean'),
            Campaign_Rate = ('Response', 'mean')
        ).round(2).reset_index()
        seg_summary['Campaign_Rate'] = seg_summary['Campaign_Rate'].map('{:.1%}'.format)
        seg_summary['Avg_Income']    = seg_summary['Avg_Income'].map('${:,.0f}'.format)
        seg_summary['Avg_Spend']     = seg_summary['Avg_Spend'].map('${:,.0f}'.format)
        seg_summary.columns          = ['Segment', 'Count', 'Avg Income', 'Avg Spend', 'Accept Rate']

        st.dataframe(seg_summary, use_container_width=True, 
                     hide_index=True, height=175)

        # Segment size pie chart
        fig, ax = plt.subplots(figsize=(4, 3))
        sizes  = df['Cluster_Label'].value_counts()
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        ax.pie(sizes, labels=sizes.index, autopct='%1.1f%%',
               colors=colors, startangle=90,
               textprops={'fontsize': 8})
        ax.set_title('Segment Distribution', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_right:
        st.markdown("**Segment Profiles**")

        metrics   = ['Avg_Income', 'Avg_Spend', 'Avg_Purchases', 'Campaign_Rate']
        titles    = ['Avg Income ($)', 'Avg Spend ($)', 'Avg Purchases', 'Campaign Accept Rate']
        seg_stats = df.groupby('Cluster_Label').agg(
            Avg_Income    = ('Income', 'mean'),
            Avg_Spend     = ('TotalSpend', 'mean'),
            Avg_Purchases = ('TotalPurchases', 'mean'),
            Campaign_Rate = ('Response', 'mean')
        ).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(8, 5))
        colors    = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']

        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            bars = ax.bar(seg_stats['Cluster_Label'],
                          seg_stats[metric], color=colors)
            ax.set_title(title, fontsize=9)
            ax.set_xticklabels(seg_stats['Cluster_Label'],
                               rotation=15, ha='right', fontsize=7)
            for bar, val in zip(bars, seg_stats[metric]):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + bar.get_height()*0.02,
                        f'{val:.0f}', ha='center', fontsize=7)

        plt.suptitle('Cluster Profiles Comparison', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ── TAB 2 — CUSTOMER BEHAVIOUR ───────────
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Income Distribution by Segment**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        segments = df['Cluster_Label'].unique()
        colors   = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        for seg, color in zip(segments, colors):
            subset = df[df['Cluster_Label'] == seg]['Income']
            ax.hist(subset, bins=20, alpha=0.6, label=seg, color=color)
        ax.set_title('Income Distribution', fontsize=10)
        ax.set_xlabel('Income')
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Spending by Product Category**")
        spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                      'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        spend_means = df[spend_cols].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(
            [c.replace('Mnt','').replace('Products','') for c in spend_cols],
            spend_means,
            color=['#8E44AD','#27AE60','#E74C3C','#3498DB','#F39C12','#1ABC9C']
        )
        ax.set_title('Avg Spend per Category', fontsize=10)
        ax.set_xlabel('Category')
        ax.set_ylabel('Avg Spend ($)')
        for bar, val in zip(bars, spend_means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 2,
                    f'${val:.0f}', ha='center', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        st.markdown("**Purchase Channels**")
        channel_cols  = ['NumWebPurchases', 'NumCatalogPurchases',
                         'NumStorePurchases', 'NumDealsPurchases']
        channel_means = df[channel_cols].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(
            ['Web', 'Catalog', 'Store', 'Deals'],
            channel_means,
            color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
        )
        ax.set_title('Avg Purchases by Channel', fontsize=10)
        ax.set_ylabel('Avg Purchases')
        for bar, val in zip(bars, channel_means):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.05,
                    f'{val:.1f}', ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Spending vs Income by Segment**")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        for seg, color in zip(df['Cluster_Label'].unique(), colors):
            subset = df[df['Cluster_Label'] == seg]
            ax.scatter(subset['Income'], subset['TotalSpend'],
                       alpha=0.4, label=seg, color=color, s=10)
        ax.set_title('Income vs Total Spend', fontsize=10)
        ax.set_xlabel('Income')
        ax.set_ylabel('Total Spend')
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ── TAB 3 — CAMPAIGN HISTORY ─────────────
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Campaign Acceptance by Segment**")
        camp_seg = df.groupby('Cluster_Label')['Response'].mean().reset_index()
        camp_seg.columns = ['Segment', 'Acceptance Rate']
        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E74C3C']
        bars   = ax.bar(camp_seg['Segment'],
                        camp_seg['Acceptance Rate'], color=colors)
        ax.set_title('Campaign Acceptance Rate by Segment', fontsize=10)
        ax.set_ylabel('Acceptance Rate')
        ax.set_ylim(0, 0.5)
        ax.set_xticklabels(camp_seg['Segment'], rotation=15, ha='right', fontsize=8)
        for bar, val in zip(bars, camp_seg['Acceptance Rate']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f'{val:.1%}', ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Overall Campaign Response**")
        fig, ax = plt.subplots(figsize=(5, 3))
        counts = df['Response'].value_counts()
        ax.bar(['Rejected', 'Accepted'], counts,
               color=['#E74C3C', '#2ECC71'])
        ax.set_title('Campaign Response Distribution', fontsize=10)
        for bar, val in zip(ax.patches, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 10,
                    f'{val:,}\n({val/len(df):.1%})',
                    ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Which Segment Accepts Which Campaign**")

        camp_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                    'AcceptedCmp4', 'AcceptedCmp5']

        camp_seg_df = df.groupby('Cluster_Label')[camp_cols].mean()
        camp_seg_df.columns = ['Camp 1', 'Camp 2', 'Camp 3', 'Camp 4', 'Camp 5']

        fig, ax = plt.subplots(figsize=(5, 3.5))
        camp_seg_df.T.plot(kind='bar', ax=ax, 
                        color=['#2ECC71', '#3498DB', '#F39C12', '#E74C3C'],
                        width=0.7)
        ax.set_title('Campaign Acceptance by Segment', fontsize=10)
        ax.set_xlabel('Campaign')
        ax.set_ylabel('Acceptance Rate')
        ax.set_xticklabels(['Camp 1', 'Camp 2', 'Camp 3', 'Camp 4', 'Camp 5'],
                            rotation=0, fontsize=8)
        ax.legend(title='Segment', fontsize=7, title_fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    with col2:
        st.markdown("**Individual Campaign Performance**")
        camp_cols  = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3',
                      'AcceptedCmp4', 'AcceptedCmp5']
        camp_rates = df[camp_cols].mean()
        fig, ax    = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(
            ['Camp 1', 'Camp 2', 'Camp 3', 'Camp 4', 'Camp 5'],
            camp_rates,
            color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6']
        )
        ax.set_title('Acceptance Rate per Campaign', fontsize=10)
        ax.set_ylabel('Acceptance Rate')
        for bar, val in zip(bars, camp_rates):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f'{val:.1%}', ha='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Recency vs Campaign Response**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(df[df['Response']==0]['Recency'], bins=20,
                alpha=0.6, label='Rejected', color='#E74C3C')
        ax.hist(df[df['Response']==1]['Recency'], bins=20,
                alpha=0.6, label='Accepted', color='#2ECC71')
        ax.set_title('Recency vs Response', fontsize=10)
        ax.set_xlabel('Days Since Last Purchase')
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.markdown("**Features Driving Acceptance vs Rejection**")

        top_features = ['Income', 'TotalSpend', 'TotalPurchases',
                        'Recency', 'TotalCampaignsAccepted',
                        'NumCatalogPurchases', 'NumWebPurchases']

        labels   = ['Income', 'Spend', 'Purchases', 'Recency',
                    'PrevCamps', 'Catalog', 'Web']

        accepted = df[df['Response'] == 1][top_features].mean()
        rejected = df[df['Response'] == 0][top_features].mean()

        # Calculate % difference — how much higher/lower accepted is vs rejected
        pct_diff = ((accepted - rejected) / rejected * 100).values

        colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in pct_diff]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(labels, pct_diff, color=colors)

        ax.axhline(y=0, color='white', linewidth=0.8)
        ax.set_title('How Much Higher/Lower Accepted Customers Are', fontsize=9)
        ax.set_ylabel('% Difference vs Rejected')
        ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=7)

        for bar, val in zip(bars, pct_diff):
            offset = 2 if val >= 0 else -8
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + offset,
                    f'{val:+.0f}%', ha='center', fontsize=7)

        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#2ECC71', label='Higher in Accepted'),
            Patch(color='#E74C3C', label='Higher in Rejected')
        ], fontsize=7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ── TAB 4 — CAMPAIGN PREDICTION ──────────
with tab4:
    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.markdown("**Enter Customer Details**")

        age        = st.slider("Age", 18, 90, 45)
        income     = st.number_input("Annual Income ($)", 
                                      min_value=0, max_value=200000, 
                                      value=50000, step=1000)
        recency    = st.slider("Days Since Last Purchase", 0, 100, 30)
        total_spend = st.number_input("Total Spend ($)",
                                       min_value=0, max_value=3000,
                                       value=500, step=50)
        total_purch = st.slider("Total Purchases", 0, 40, 10)
        web_visits  = st.slider("Web Visits per Month", 0, 20, 5)
        children    = st.slider("Total Children at Home", 0, 4, 1)
        prev_camps  = st.slider("Previous Campaigns Accepted", 0, 5, 0)
        education   = st.selectbox("Education Level",
                                    ['Basic', '2n Cycle', 'Graduation',
                                     'Master', 'PhD'])
        enroll_year = st.selectbox("Enrollment Year",
                                    list(range(2010, 2015)))

    with col_result:
        st.markdown("**Prediction Result**")

        if st.button("🎯 Predict Campaign Response", 
                      use_container_width=True):

            # Map education
            edu_map = {'Basic':0, '2n Cycle':1,
                       'Graduation':2, 'Master':3, 'PhD':4}
            edu_val = edu_map[education]

            # Build raw input
            raw_input = {
                'Income'               : income,
                'Age'                  : age,
                'TotalSpend'           : total_spend,
                'TotalPurchases'       : total_purch,
                'Recency'              : recency,
                'NumWebVisitsMonth'    : web_visits,
                'TotalChildren'        : children,
                'TotalCampaignsAccepted': prev_camps,
                'EnrollmentYear'       : enroll_year
            }

            # Scale numerical features
            raw_df  = pd.DataFrame([raw_input])
            scaled  = scaler.transform(raw_df[scale_cols])
            scaled_df = pd.DataFrame(scaled, columns=scale_cols)

            # Build full feature row
            input_row = pd.DataFrame(columns=feature_columns)
            input_row.loc[0] = 0

            # Fill scaled values
            for col in scale_cols:
                if col in feature_columns:
                    input_row[col] = scaled_df[col].values[0]

            # Fill education
            if 'Education' in feature_columns:
                input_row['Education'] = edu_val

            # Predict
            prob       = best_model.predict_proba(input_row)[0][1]
            prediction = int(prob >= 0.5)

            # Show result
            st.markdown("---")
            if prediction == 1:
                st.success("✅ Likely to Accept Campaign")
            else:
                st.error("❌ Unlikely to Accept Campaign")

            # Probability gauge
            st.markdown(f"### Acceptance Probability: {prob:.1%}")
            st.progress(float(prob))

            # Confidence level
            if prob >= 0.75:
                conf = "🟢 High Confidence"
            elif prob >= 0.5:
                conf = "🟡 Moderate Confidence"
            else:
                conf = "🔴 Low Confidence"
            st.markdown(f"**Confidence:** {conf}")

            st.markdown("---")
            st.markdown("**Customer Summary:**")
            summary_data = {
                'Feature': ['Age', 'Income', 'Total Spend',
                            'Total Purchases', 'Prev Campaigns', 'Education'],
                'Value'  : [age, f'${income:,}', f'${total_spend:,}',
                            total_purch, prev_camps, education]
            }
            st.dataframe(pd.DataFrame(summary_data),
                         hide_index=True, use_container_width=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("---")
st.caption("Business Insights Dashboard | Customer Personality Analysis Dataset")