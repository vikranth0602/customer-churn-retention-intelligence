# Customer Churn Prediction & Retention Intelligence System

## Live Demo
https://customer-churn-retention-intelligence-ihb97uqy4fh7ebgunb8vnm.streamlit.app/

## Overview

This project predicts customer churn using transactional retail data and machine learning.

The system identifies customers who are likely to stop purchasing and segments them into risk categories to support retention strategies.

Built using the Online Retail II dataset and deployed through an interactive Streamlit dashboard.

---

## Business Problem

Customer retention is often more cost-effective than customer acquisition.

The goal of this project is to:

* Predict customer churn
* Identify high-risk customers
* Support retention campaigns
* Generate actionable business insights

---

## Dataset

Dataset: Online Retail II

Period:
December 2009 – December 2011

Records:
1,067,371 transactions

Customers:
4,933

Countries:
41+

---

## Feature Engineering

Customer-level features were created including:

* Recency
* Frequency
* Monetary Value
* Average Order Value
* Product Diversity
* Total Quantity Purchased
* Customer Lifetime
* Average Purchase Interval
* Single Order Customer Flag
* UK Customer Flag

---

## Models Evaluated

| Model               | Accuracy | F1 Score | ROC AUC |
| ------------------- | -------- | -------- | ------- |
| Logistic Regression | 0.750    | 0.801    | 0.790   |
| Random Forest       | 0.742    | 0.795    | 0.786   |
| XGBoost             | 0.740    | 0.794    | 0.802   |

Selected Production Model:
Logistic Regression

---

## Dashboard Features

* Customer Churn Dashboard
* Risk Segmentation
* Single Customer Prediction
* Batch Prediction
* Download Prediction Results
* Business Recommendations
* Model Performance Reporting

---

## Technology Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* Plotly
* Streamlit
* Joblib

---

## Project Structure

CCP_Application/

* app.py
* pages/
* models/
* data/
* requirements.txt
* README.md

---

## Author

Built as an end-to-end machine learning project focused on customer retention analytics and churn prediction.
