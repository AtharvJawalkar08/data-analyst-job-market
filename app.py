import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Data Analyst Job Market Dashboard", layout="wide")

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_postings.csv")
    return df

df = load_data()
skill_cols = [c for c in df.columns if c.startswith('skill_')]

st.title("📊 Data Analyst Job Market Dashboard")
st.markdown("Explore real Data Analyst job postings: top in-demand skills, country differences, and seniority patterns. Built from 12,894 LinkedIn-scraped postings.")

# --- Sidebar filters ---
st.sidebar.header("Filters")

countries = sorted(df['search_country'].unique())
selected_countries = st.sidebar.multiselect("Country", countries, default=countries)

job_levels = sorted(df['job level'].unique())
selected_levels = st.sidebar.multiselect("Job Level", job_levels, default=job_levels)

# Apply filters
filtered_df = df[
    (df['search_country'].isin(selected_countries)) &
    (df['job level'].isin(selected_levels))
]

st.sidebar.markdown(f"**{len(filtered_df)} postings** match your filters (out of {len(df)} total)")

# --- Top skills chart (updates with filters) ---
st.subheader("Top Skills in Filtered Postings")

if len(filtered_df) > 0:
    skill_pct = (filtered_df[skill_cols].mean() * 100).sort_values(ascending=False).head(15)
    skill_pct.index = [c.replace('skill_', '').replace('_', ' ').title() for c in skill_pct.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(skill_pct.index[::-1], skill_pct.values[::-1], color='#2563EB')
    ax.set_xlabel('% of Postings Mentioning This Skill')
    ax.bar_label(bars, fmt='%.1f%%', padding=3, fontsize=9)
    st.pyplot(fig)
else:
    st.warning("No postings match the selected filters. Try adjusting them.")

# --- Data table ---
st.subheader("Browse Filtered Postings")
display_cols = ['job_title', 'company', 'job_location', 'search_country', 'job level', 'job_type']
st.dataframe(filtered_df[display_cols].head(100))
# --- Salary by experience level (separate dataset) ---
st.subheader("💰 Salary by Experience Level (Global Benchmark)")
st.caption("Note: salary data comes from a separate compensation benchmark dataset, not the postings above.")

@st.cache_data
def load_salary_data():
    df2 = pd.read_csv("data/salary_data_analyst.csv")  # we'll create this next
    return df2

df2_da = load_salary_data()

exp_order = ['Entry', 'Mid', 'Senior', 'Lead']
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df2_da, x='experience_level', y='salary_usd', order=exp_order, palette='Blues', ax=ax2)
ax2.set_xlabel('Experience Level')
ax2.set_ylabel('Salary (USD)')
st.pyplot(fig2)

means = df2_da.groupby('experience_level')['salary_usd'].mean().reindex(exp_order)
cols = st.columns(4)
for col, level, mean_val in zip(cols, exp_order, means):
    col.metric(level, f"${mean_val:,.0f}")

# --- Salary estimator (predictive model) ---
st.subheader("💵 Estimate Your Salary")
st.caption(
    "Trained on 11,000+ Data Analyst salary records. "
    "Model: Random Forest | Test MAE: $6,792 | R²: 0.91 "
    "(vs. $24,301 MAE for a naive mean-prediction baseline)."
)

@st.cache_resource
def load_salary_model():
    return joblib.load("models/salary_model.joblib")

salary_model = load_salary_model()

col1, col2, col3 = st.columns(3)
with col1:
    est_country = st.selectbox(
        "Country", sorted(df2_da['country'].unique()), key="salary_country"
    )
    est_experience_level = st.selectbox(
        "Experience Level", exp_order, key="salary_level"
    )
with col2:
    est_years = st.slider("Years of Experience", 0, 20, 3, key="salary_years")
    est_education = st.selectbox(
        "Education", sorted(df2_da['education_required'].unique()), key="salary_edu"
    )
with col3:
    est_industry = st.selectbox(
        "Industry", sorted(df2_da['industry'].unique()), key="salary_industry"
    )
    est_company_size = st.selectbox(
        "Company Size", sorted(df2_da['company_size'].unique()), key="salary_company"
    )

est_work_mode = st.radio(
    "Work Mode", sorted(df2_da['work_mode'].unique()), horizontal=True
)

if st.button("Estimate Salary"):
    input_df = pd.DataFrame([{
        "country": est_country,
        "experience_level": est_experience_level,
        "education_required": est_education,
        "industry": est_industry,
        "company_size": est_company_size,
        "work_mode": est_work_mode,
        "experience_years": est_years,
    }])
    prediction = salary_model.predict(input_df)[0]
    st.success(f"Estimated salary: **${prediction:,.0f}** per year")
    st.caption(
        "±$6,800 typical error margin based on test-set performance. "
        "This is a model estimate from historical patterns, not a guarantee."
    )