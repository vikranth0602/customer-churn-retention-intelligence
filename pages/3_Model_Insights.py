import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Model Insights",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Insights & Feature Importance")

model = joblib.load("models/churn_model.pkl")

feature_names = [
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

importance = model.coef_[0]

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    "Importance",
    key=abs,
    ascending=False
)

fig = px.bar(
    feature_df,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Feature Impact on Churn Prediction"
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Business Interpretation")

st.markdown("""
### Key Insights:

- **Recency** is the strongest churn driver  
  → Customers inactive for long periods are more likely to churn

- **Frequency** reduces churn  
  → More frequent buyers are more loyal

- **Monetary value** also matters  
  → High-value customers behave differently

### Actionable Strategy:

- Target inactive customers with re-engagement campaigns
- Reward frequent buyers with loyalty programs
- Focus retention efforts on high-value customers
""")

st.subheader("Model Summary")

st.info("""
Best Model: Logistic Regression

Why:
- Stable performance
- High interpretability
- Strong F1 Score
- Business-friendly explanation
""")

