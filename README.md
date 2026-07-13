# 📈 Sales Forecasting & Demand Intelligence System

An end-to-end Data Science project that forecasts future sales, detects anomalies, segments product demand, and provides an interactive business dashboard using the Superstore Sales dataset.

---

## 📌 Project Overview

Accurate sales forecasting helps businesses optimize inventory, reduce operational costs, and improve customer satisfaction. This project analyzes historical Superstore sales data (2015–2018) to predict future demand using multiple forecasting models and delivers business insights through an interactive Streamlit dashboard.

---

## 🎯 Objectives

* Analyze historical sales trends and seasonality.
* Forecast future sales using multiple time-series and machine learning models.
* Detect unusual sales patterns (anomalies).
* Segment products based on demand behavior.
* Build an interactive dashboard for business users.

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Data Analysis:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost
* **Time Series Forecasting:** SARIMA (Statsmodels), Prophet
* **Dashboard:** Streamlit

---

## 📊 Project Workflow

1. Data Cleaning & Preprocessing
2. Exploratory Data Analysis (EDA)
3. Sales Trend Analysis
4. Time-Series Forecasting

   * SARIMA
   * Prophet
   * XGBoost
5. Model Evaluation (MAE, RMSE, MAPE)
6. Sales Anomaly Detection
7. Product Demand Segmentation using K-Means Clustering
8. Interactive Streamlit Dashboard

---

## 📈 Forecasting Model Performance

| Model       |         MAE |         RMSE |       MAPE |
| ----------- | ----------: | -----------: | ---------: |
| SARIMA      |    13930.02 |     16394.82 |     27.77% |
| **Prophet** | **9839.84** | **14133.08** | **15.67%** |
| XGBoost     |    22087.36 |     26510.80 |     45.46% |

**Best Model:** Prophet

---

## 📊 Streamlit Dashboard Features

### 📌 Sales Overview

* Yearly Sales Analysis
* Monthly Sales Trend
* Region & Category Filters

### 📌 Forecast Explorer

* Prophet-based Sales Forecast
* Category & Region Selection
* Forecast Horizon Selection
* Model Performance Metrics

### 📌 Anomaly Report

* Sales Anomaly Detection
* Anomaly Dates & Sales Values

### 📌 Product Demand Segmentation

* K-Means Clustering
* PCA Visualization
* Stocking Strategy Recommendations

---

## 📁 Project Structure

```text
SalesForecasting_SatyaGangaChilukuri/
│
├── analysis.ipynb
├── app.py
├── train.csv
├── requirements.txt
├── summary.pdf
├── README.md
└── charts/
```

---

## 🚀 How to Run

1. Clone the repository

```bash
git clone https://github.com/yourusername/SalesForecasting_SatyaGangaChilukuri.git
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📌 Business Insights

* Technology generated the highest overall sales.
* Strong seasonal sales peaks occur during September, November, and December.
* Prophet delivered the most accurate sales forecasts.
* Anomaly detection identified unusual sales periods for further investigation.
* Demand segmentation supports optimized inventory and stocking strategies.

---


## Project Links

Google Colab Notebook:
PASTE_YOUR_COLAB_LINK_HERE

Live Streamlit Dashboard:
PASTE_YOUR_STREAMLIT_APP_LINK_HERE

GitHub Repository:
https://github.com/SatyaChilukuri22/SalesForecasting-Demand-Intelligence-System.git

## 👤 Author

**Satya Ganga Chilukuri**

Aspiring Data Analyst | Data Scientist

GitHub: https://github.com/SatyaChilukuri22

