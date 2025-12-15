YouTube Ad Revenue Prediction
 Project Overview

The Content Monetization Modeler is a machine learning–based project that predicts YouTube ad revenue for individual videos using performance, engagement, and contextual features.
The project aims to help content creators, media companies, and analysts estimate potential earnings and make data-driven content decisions.

This project follows a complete end-to-end data science workflow, including data analysis, preprocessing, model building, evaluation, and deployment using Streamlit.

Problem Statement

With YouTube becoming a major revenue platform, predicting ad revenue in advance is essential for:

Revenue forecasting

Content strategy optimization

Business planning

This project builds a regression model to estimate ad_revenue_usd based on historical video performance data.

 Skills & Technologies Used

Python

Pandas, NumPy

Exploratory Data Analysis (EDA)

Data Cleaning & Feature Engineering

Regression Models

Scikit-learn Pipelines

Model Evaluation (R², RMSE, MAE)

Streamlit (Web Application)

Joblib (Model Saving)

 Dataset Information

Dataset Name: YouTube Monetization Modeler

Format: CSV

Size: ~122,000 rows

Source: Synthetic dataset (for learning purposes)

Target Variable: ad_revenue_usd

Key Features

views, likes, comments

watch_time_minutes, video_length_minutes

subscribers

category, device, country

Engineered Feature: engagement_rate

 Project Workflow

Dataset Loading & Understanding

Data Cleaning (missing values, duplicates)

Exploratory Data Analysis (EDA)

Feature Engineering

Data Preprocessing (scaling & encoding)

Model Building (5 regression models)

Model Evaluation (R², RMSE, MAE)

Final Model Selection

Streamlit App Development

🤖 Models Implemented
Model	Description
Linear Regression	Baseline model
Ridge Regression	Regularized regression
Lasso Regression	Feature selection
Decision Tree Regressor	Non-linear modeling
Random Forest Regressor	Best-performing model
 Model Evaluation Metrics

R² Score – Measures explained variance

RMSE – Root Mean Squared Error

MAE – Mean Absolute Error

The Random Forest Regressor provided the best performance and was selected as the final model.

 Streamlit Application

The Streamlit web application allows users to:

Input video performance metrics

Select category, device, and country

Predict estimated YouTube ad revenue instantly

▶️ Run the Application
streamlit run app.py

🗂️ Project Structure
Content Monetization Modeler/
│
├── app.py                 
├── CODE.py                 
├── youtube_ad_revenue_dataset.csv
├── best_model.pkl
├── scaler.pkl
├── features.pkl
├── le_category.pkl
├── le_device.pkl
├── le_country.pkl
├── results.pkl
└── README.md               # Content-Monetization-Modeler
