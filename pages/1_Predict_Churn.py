import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Predict Churn",
    page_icon="🎯",
    layout="wide"
)

# Load model artifacts
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

st.title("🎯 Customer Churn Prediction")

tab1, tab2 = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction"
    ]
)

with tab1:

    # existing single prediction code

        
    st.markdown(
        "Enter customer information and predict churn probability."
    )

    # Inputs

    col1, col2 = st.columns(2)

    with col1:

        recency = st.number_input(
            "Recency",
            min_value=0.0,
            value=50.0
        )

        frequency = st.number_input(
            "Frequency",
            min_value=1.0,
            value=5.0
        )

        monetary = st.number_input(
            "Monetary",
            min_value=0.0,
            value=1000.0
        )

        avg_order_value = st.number_input(
            "Avg Order Value",
            min_value=0.0,
            value=200.0
        )

        product_diversity = st.number_input(
            "Product Diversity",
            min_value=1.0,
            value=30.0
        )

    with col2:

        total_quantity = st.number_input(
            "Total Quantity",
            min_value=1.0,
            value=500.0
        )

        customer_lifetime = st.number_input(
            "Customer Lifetime",
            min_value=0.0,
            value=100.0
        )

        avg_purchase_interval = st.number_input(
            "Avg Purchase Interval",
            value=30.0
        )

        single_order_customer = st.selectbox(
            "Single Order Customer",
            [0, 1]
        )

        is_uk_customer = st.selectbox(
            "UK Customer",
            [0, 1]
        )

    # Prediction button

    if st.button("Predict Churn"):

        customer = pd.DataFrame([
            [
                recency,
                frequency,
                monetary,
                avg_order_value,
                product_diversity,
                total_quantity,
                customer_lifetime,
                avg_purchase_interval,
                single_order_customer,
                is_uk_customer
            ]
        ], columns=feature_columns
        
        )

        customer_scaled = scaler.transform(customer)

        churn_probability = model.predict_proba(
            customer_scaled
        )[0][1]

        st.subheader("Prediction Results")

        st.metric(
            "Churn Probability",
            f"{churn_probability:.2%}"
        )

        # percentage & recommended actions
        if churn_probability >= 0.70:

            st.error("🔴 High Risk Customer")

            st.markdown("""
            ### Recommended Actions

            - Send retention offer
            - Provide discount coupon
            - Launch re-engagement email campaign
            - Prioritize for customer success outreach
            """)

        elif churn_probability >= 0.40:

            st.warning("🟠 Medium Risk Customer")

            st.markdown("""
            ### Recommended Actions

            - Monitor activity closely
            - Offer personalized promotions
            - Encourage repeat purchases
            """)

        else:

            st.success("🟢 Low Risk Customer")

            st.markdown("""
            ### Recommended Actions

            - Maintain engagement
            - Reward loyalty
            - Upsell premium products
            """)
        
        #reasons
        explanations = []

        if recency > 180:
            explanations.append(
                f"High recency ({recency:.0f} days since last purchase)"
            )

        if frequency <= 2:
            explanations.append(
                f"Low purchase frequency ({frequency:.0f} orders)"
            )

        if customer_lifetime < 90:
            explanations.append(
                f"Short customer lifetime ({customer_lifetime:.0f} days)"
            )

        if product_diversity < 20:
            explanations.append(
                f"Limited product diversity ({product_diversity:.0f} products)"
            )

        if explanations:

            st.subheader("Why This Prediction?")

            for item in explanations:
                st.write(f"• {item}")

with tab2:

    st.subheader("Batch Prediction")

    # paste batch prediction code here
    
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        if st.button("Run Batch Prediction"):

            X = df[feature_columns]

            X_scaled = scaler.transform(X)

            probabilities = model.predict_proba(X_scaled)[:, 1]

            results = df.copy()

            results["ChurnProbability"] = probabilities


            results["RiskSegment"] = pd.cut(
                results["ChurnProbability"],
                bins=[0, 0.4, 0.7, 1],
                labels=[
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ]
            )

            st.subheader("Prediction Results")

            st.dataframe(
                results.head(200)
            )

            csv = results.to_csv(index=False)

            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )