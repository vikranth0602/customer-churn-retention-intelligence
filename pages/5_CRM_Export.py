import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CRM Export",
    page_icon="📤",
    layout="wide"
)

st.title("📤 CRM Export Dashboard")

df = pd.read_csv("data/features.csv")

model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")

feature_cols = [
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

X = df[feature_cols]

X_scaled = scaler.transform(X)

df["ChurnProbability"] = model.predict_proba(X_scaled)[:,1]

df["RiskSegment"] = pd.cut(
    df["ChurnProbability"],
    bins=[0,0.4,0.7,1],
    labels=[
        "Low Risk",
        "Medium Risk",
        "High Risk"
    ]
)

st.sidebar.header("Filters")

risk_filter = st.sidebar.selectbox(
    "Risk Segment",
    [
        "All",
        "High Risk",
        "Medium Risk",
        "Low Risk"
    ]
)

min_prob = st.sidebar.slider(
    "Minimum Churn Probability",
    0.0,
    1.0,
    0.7,
    0.05
)

min_value = st.sidebar.number_input(
    "Minimum Monetary Value",
    value=0
)

filtered_df = df.copy()

if risk_filter != "All":
    filtered_df = filtered_df[
        filtered_df["RiskSegment"] == risk_filter
    ]

filtered_df = filtered_df[
    filtered_df["ChurnProbability"] >= min_prob
]

filtered_df = filtered_df[
    filtered_df["Monetary"] >= min_value
]

filtered_df = filtered_df.sort_values(
    by="ChurnProbability",
    ascending=False
)

st.subheader("Campaign Summary")

selected_customers = len(filtered_df)

avg_risk = (
    filtered_df["ChurnProbability"]
    .mean() * 100
)

revenue_at_risk = (
    filtered_df["Monetary"]
    .sum()
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Selected Customers",
        f"{selected_customers:,}"
    )

with col2:
    st.metric(
        "Average Risk",
        f"{avg_risk:.1f}%"
    )

with col3:
    st.metric(
        "Revenue At Risk",
        f"${revenue_at_risk:,.0f}"
    )

st.subheader("Customers Selected For Campaign")

display_cols = [
    "Customer ID",
    "ChurnProbability",
    "RiskSegment",
    "Recency",
    "Frequency",
    "Monetary"
]

st.dataframe(
    filtered_df[display_cols],
    use_container_width=True
)

csv = filtered_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇ Download CRM Campaign List",
    data=csv,
    file_name="crm_campaign_customers.csv",
    mime="text/csv"
)

st.subheader("Recommended Campaign")

if risk_filter == "High Risk":
    st.error("""
    Immediate retention campaign recommended.

    • Discount offers
    • Re-engagement emails
    • VIP outreach
    """)

elif risk_filter == "Medium Risk":
    st.warning("""
    Engagement campaign recommended.

    • Product recommendations
    • Loyalty reminders
    • Promotional emails
    """)

else:
    st.success("""
    Loyalty maintenance campaign.

    • VIP rewards
    • Cross-selling
    • Upselling opportunities
    """)

