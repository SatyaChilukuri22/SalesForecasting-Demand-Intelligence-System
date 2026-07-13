# ==========================================================
# Sales Forecasting Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# ----------------------------------------------------------
# Dashboard Title
# ----------------------------------------------------------

st.title("📈 Sales Forecasting & Demand Analytics Dashboard")

st.write(
    "Welcome to the Sales Forecasting Dashboard."
)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True
    )

    return df

df = load_data()

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    (
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    )
)

# ----------------------------------------------------------
# Test Page
# ----------------------------------------------------------

st.write("Dataset Shape:", df.shape)

st.dataframe(df.head())