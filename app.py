import streamlit as st
import pandas as pd
import plotly.express as px
import joblib



st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

df = pd.read_csv("data/features.csv")

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_columns = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "ProductDiversity",
    "TotalQuantity",
    "CustomerLifetime",
    "AvgPurchaseInterval",
    "SingleOrderCustomer",
    "IsUKCustomer"
]

X = df[feature_columns]

X_scaled = scaler.transform(X)

df["ChurnProbability"] = (
    model.predict_proba(X_scaled)[:, 1]
)

df["RiskSegment"] = pd.cut(
    df["ChurnProbability"],
    bins=[0, 0.4, 0.7, 1],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)


st.title("📊 Customer Churn Prediction & Retention Intelligence ")

st.markdown("""
Production-ready machine learning system for
customer churn prediction, risk segmentation,
and retention intelligence.
""")

st.sidebar.title("📊 CCP System")

st.sidebar.success("Production Version")

st.sidebar.info("""
Customer Churn Prediction &
Retention Intelligence System
""")


st.subheader("Executive Summary")
# KPIs

total_customers = len(df)

churned_customers = (
    df["churn"] == 1
).sum()

active_customers = (
    df["churn"] == 0
).sum()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

with col2:
    st.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

with col3:
    st.metric(
        "Active Customers",
        f"{active_customers:,}"
    )

st.divider()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Churn Rate",
        f"{df['churn'].mean()*100:.1f}%"
    )

with col2:
    st.metric(
        "Avg Customer Value",
        f"${df['Monetary'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Avg Frequency",
        f"{df['Frequency'].mean():.1f}"
    )

st.divider()

risk_counts = pd.DataFrame({
    "RiskSegment": [
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ],
    "Count": [
        450,
        373,
        164
    ]
})

revenue_at_risk = (
    df[df["churn"] == 1]["Monetary"]
    .sum()
)

largest_segment = (
    risk_counts
    .sort_values(
        "Count",
        ascending=False
    )
    .iloc[0]["RiskSegment"]
)

customers_at_risk = (
    risk_counts["Count"]
    .sum()
)

#executive intelligence

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Revenue At Risk",
        f"${revenue_at_risk:,.0f}"
    )

with col2:
    st.metric(
        "Largest Risk Segment",
        largest_segment
    )

with col3:
    st.metric(
        "Customers At Risk",
        f"{customers_at_risk:,}"
    )

with col4:
    st.metric(
        "Recommended Action",
        "Retention"
    )

st.divider()

st.subheader("Business Overview")

col1, col2 = st.columns(2)

with col1:

    st.info("""
    Dataset: Online Retail II

    Period: Dec 2009 - Dec 2011

    Customers: 4,933

    Prediction Window: 120 Days
    """)

with col2:

    st.success("""
    Objective:

    Predict customer churn using
    historical purchase behavior
    and customer-level features.
    """)

st.subheader("Production Model Performance")

model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        0.750,
        0.742,
        0.740
    ],
    "F1 Score": [
        0.801,
        0.795,
        0.794
    ],
    "ROC AUC": [
        0.790,
        0.786,
        0.802
    ]
})

st.dataframe(
    model_results,
    use_container_width=True
)

st.success("""
Best Production Model:

• Logistic Regression

Reason:
Highest F1 Score and strongest overall business performance.
""")

st.divider()

# st.subheader("Dataset Preview")
# st.dataframe(df.head())

st.subheader("Customer Portfolio Analysis")

col1, col2 = st.columns(2)

# Pie Chart
with col1:

    churn_counts = (
        df["churn"]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = [
        "ChurnStatus",
        "Count"
    ]

    churn_counts["ChurnStatus"] = (
        churn_counts["ChurnStatus"]
        .map({
            0: "Active",
            1: "Churned"
        })
    )

    fig_pie = px.pie(
        churn_counts,
        names="ChurnStatus",
        values="Count",
        title="Customer Distribution"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# Bar Chart
with col2:

    fig_bar = px.bar(
        churn_counts,
        x="ChurnStatus",
        y="Count",
        title="Customer Count by Status"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

st.divider()

st.subheader("Customer Risk Segmentation")

# Risk segmentation using churn label
risk_counts = (
    df["RiskSegment"]
    .value_counts()
    .reset_index()
)

risk_order = [
    "High Risk",
    "Medium Risk",
    "Low Risk"
]

risk_counts["RiskSegment"] = pd.Categorical(
    risk_counts["RiskSegment"],
    categories=risk_order,
    ordered=True
)

risk_counts = risk_counts.sort_values(
    "RiskSegment"
)


risk_counts.columns = [
    "RiskSegment",
    "Count"
]

col1, col2 = st.columns(2)

with col1:

    fig_risk_bar = px.bar(
        risk_counts,
        x="RiskSegment",
        y="Count",
        title="Customer Risk Distribution"
    )

    st.plotly_chart(
        fig_risk_bar,
        use_container_width=True
    )

with col2:

    fig_risk_pie = px.pie(
        risk_counts,
        names="RiskSegment",
        values="Count",
        title="Risk Segment Share"
    )

    st.plotly_chart(
        fig_risk_pie,
        use_container_width=True
    )


st.subheader("Key Business Insights")

st.markdown("""
- High Risk customers represent the largest segment.
- High Risk customers typically have:
    - High Recency
    - Low Frequency
    - Low Monetary Value
- Low Risk customers tend to be loyal repeat buyers.
- Retention efforts should focus on High Risk customers.
""")


st.markdown("---")

st.markdown("""
### Customer Churn Prediction & Retention Intelligence System

Built Using:

- Python
- Scikit-Learn
- Streamlit
- Plotly

Dataset:
Online Retail II
""")

           