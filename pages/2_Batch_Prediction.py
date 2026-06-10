import streamlit as st
import pandas as pd
import joblib

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


st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

st.title("📂 Batch Churn Prediction")

st.markdown(
    "Upload a CSV file containing customer features."
)


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