"""
Sales Forecasting & Demand Intelligence Dashboard
Run locally with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv', encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Quarter'] = df['Order Date'].dt.quarter
    return df

df = load_data()

def season_num(m):
    return 0 if m in [12, 1, 2] else 1 if m in [3, 4, 5] else 2 if m in [6, 7, 8] else 3

FEATURES = ['lag1', 'lag2', 'lag3', 'rolling_mean_3', 'month', 'quarter', 'season']

@st.cache_data
def build_monthly_series(category=None, region=None):
    sub = df.copy()
    if category and category != "All":
        sub = sub[sub['Category'] == category]
    if region and region != "All":
        sub = sub[sub['Region'] == region]
    monthly = sub.set_index('Order Date').resample('MS')['Sales'].sum().asfreq('MS').fillna(0)
    return monthly

@st.cache_data
def train_prophet_and_forecast(category, region, horizon):
    import logging
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
    from prophet import Prophet

    monthly = build_monthly_series(category, region)
    prophet_df = monthly.reset_index()
    prophet_df.columns = ['ds', 'y']

    n_test = min(3, len(prophet_df) // 4) or 1
    train_df = prophet_df.iloc[:-n_test]
    test_df = prophet_df.iloc[-n_test:]

    # --- Fit on train, evaluate against held-out months ---
    m_test = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m_test.fit(train_df)
    future_test = m_test.make_future_dataframe(periods=n_test, freq='MS')
    fc_test = m_test.predict(future_test)
    test_pred = fc_test.set_index('ds')['yhat'].reindex(test_df['ds']).values

    mae = float(np.mean(np.abs(test_df['y'].values - test_pred)))
    rmse = float(np.sqrt(np.mean((test_df['y'].values - test_pred) ** 2)))

    # --- Refit on full history, forecast the requested horizon forward ---
    m_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m_full.fit(prophet_df)
    future_full = m_full.make_future_dataframe(periods=horizon, freq='MS')
    fc_full = m_full.predict(future_full)
    future_rows = fc_full.iloc[-horizon:]

    preds = [
        (row['ds'], max(row['yhat'], 0), max(row['yhat_lower'], 0), max(row['yhat_upper'], 0))
        for _, row in future_rows.iterrows()
    ]

    return monthly, preds, mae, rmse

@st.cache_data
def compute_anomalies():
    weekly = df.set_index('Order Date').resample('W')['Sales'].sum().asfreq('W').fillna(0).to_frame('Sales')
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.07, random_state=42)
    weekly['iso_anomaly'] = iso.fit_predict(weekly[['Sales']]) == -1

    window = 8
    weekly['rolling_mean'] = weekly['Sales'].rolling(window, center=True, min_periods=4).mean()
    weekly['rolling_std'] = weekly['Sales'].rolling(window, center=True, min_periods=4).std()
    weekly['z_score'] = (weekly['Sales'] - weekly['rolling_mean']) / weekly['rolling_std']
    weekly['z_anomaly'] = weekly['z_score'].abs() > 2
    return weekly

@st.cache_data
def compute_clusters():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    d = df.copy()
    d['YearMonth'] = d['Order Date'].dt.to_period('M')
    rows = []
    for sub, g in d.groupby('Sub-Category'):
        total_sales = g['Sales'].sum()
        yearly = g.groupby('Year')['Sales'].sum().sort_index()
        yoy = (yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] / (len(yearly) - 1) if len(yearly) > 1 and yearly.iloc[0] > 0 else 0
        monthly = g.groupby('YearMonth')['Sales'].sum()
        volatility = monthly.std()
        avg_order_value = g.groupby('Order ID')['Sales'].sum().mean()
        rows.append({'Sub-Category': sub, 'TotalSales': total_sales, 'YoYGrowth': yoy,
                      'Volatility': volatility, 'AvgOrderValue': avg_order_value})
    feat_df = pd.DataFrame(rows).set_index('Sub-Category')

    X_scaled = StandardScaler().fit_transform(feat_df)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    feat_df['Cluster'] = km.fit_predict(X_scaled)

    profile = feat_df.groupby('Cluster')[['TotalSales', 'YoYGrowth', 'Volatility', 'AvgOrderValue']].mean()
    medians = profile.median()

    def label_cluster(row):
        vol_label = 'High Volatility' if row['Volatility'] > medians['Volatility'] else 'Stable Demand'
        if row['YoYGrowth'] > medians['YoYGrowth'] * 1.3:
            return 'Growing Demand'
        elif row['YoYGrowth'] < medians['YoYGrowth'] * 0.7:
            return 'Declining Demand'
        else:
            vol_size = 'High Volume' if row['TotalSales'] > medians['TotalSales'] else 'Low Volume'
            return f'{vol_size}, {vol_label}'

    labels = {c: label_cluster(profile.loc[c]) for c in profile.index}
    feat_df['ClusterLabel'] = feat_df['Cluster'].map(labels)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    feat_df['PC1'], feat_df['PC2'] = coords[:, 0], coords[:, 1]
    return feat_df

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📊 Sales Intelligence")
page = st.sidebar.radio("Navigate", [
    "1. Sales Overview",
    "2. Forecast Explorer",
    "3. Anomaly Report",
    "4. Product Demand Segments",
])

# ============================================================
# PAGE 1 — SALES OVERVIEW
# ============================================================
if page == "1. Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        region_filter = st.selectbox("Filter by Region", ["All"] + sorted(df['Region'].unique().tolist()))
    with col2:
        category_filter = st.selectbox("Filter by Category", ["All"] + sorted(df['Category'].unique().tolist()))

    filtered = df.copy()
    if region_filter != "All":
        filtered = filtered[filtered['Region'] == region_filter]
    if category_filter != "All":
        filtered = filtered[filtered['Category'] == category_filter]

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
    k2.metric("Total Orders", f"{filtered['Order ID'].nunique():,}")
    k3.metric("Avg Order Value", f"${filtered.groupby('Order ID')['Sales'].sum().mean():,.0f}")

    st.subheader("Total Sales by Year")
    yearly = filtered.groupby('Year')['Sales'].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(yearly.index.astype(str), yearly.values, color='#4C72B0')
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales ($)")
    ax.set_title("Total Sales by Year")
    st.pyplot(fig)

    st.subheader("Monthly Sales Trend")
    monthly = filtered.set_index('Order Date').resample('MS')['Sales'].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker='o', color='#4C72B0')
    ax.set_xlabel("Month")
    ax.set_ylabel("Sales ($)")
    ax.set_title("Monthly Sales Trend")
    st.pyplot(fig)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Sales by Region")
        region_sales = filtered.groupby('Region')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(region_sales.index, region_sales.values, color='#55A868')
        ax.set_title("Sales by Region")
        ax.set_ylabel("Sales ($)")
        plt.xticks(rotation=30)
        st.pyplot(fig)
    with c2:
        st.subheader("Sales by Category")
        category_sales = filtered.groupby('Category')['Sales'].sum()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(category_sales.index, category_sales.values, color='#C44E52')
        ax.set_title("Sales by Category")
        ax.set_ylabel("Sales ($)")
        plt.xticks(rotation=15)
        st.pyplot(fig)

# ============================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================
elif page == "2. Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Forecasts generated with Prophet, refit on the full history for each selected segment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        dim = st.selectbox("Forecast dimension", ["Category", "Region"])
    with col2:
        if dim == "Category":
            options = ["All"] + sorted(df['Category'].unique().tolist())
            selected = st.selectbox("Select Category", options)
            category, region = selected, "All"
        else:
            options = ["All"] + sorted(df['Region'].unique().tolist())
            selected = st.selectbox("Select Region", options)
            category, region = "All", selected
    with col3:
        last_hist_date = df['Order Date'].max().to_period('M').to_timestamp('M')
        future_month_labels = [(last_hist_date + pd.DateOffset(months=i)).strftime('%b %Y') for i in [1, 2, 3]]
        horizon_label = st.select_slider("Forecast horizon (through month)", options=future_month_labels, value=future_month_labels[-1])
        horizon = future_month_labels.index(horizon_label) + 1

    with st.spinner("Generating forecast..."):
        monthly, preds, mae, rmse = train_prophet_and_forecast(category, region, horizon)

    fig, ax = plt.subplots(figsize=(11, 5))
    hist_tail = monthly.iloc[-15:]
    ax.plot(hist_tail.index, hist_tail.values, label="Actual", marker='o', color='#4C72B0')
    fdates = [p[0] for p in preds]
    fvals = [p[1] for p in preds]
    flower = [p[2] for p in preds]
    fupper = [p[3] for p in preds]
    ax.plot(fdates, fvals, label="Forecast", marker='s', linestyle='--', color='#55A868')
    ax.fill_between(fdates, flower, fupper, color='#55A868', alpha=0.2, label="Confidence Interval")
    ax.legend()
    ax.set_title(f"Prophet Forecast: {selected} ({dim})")
    ax.set_ylabel("Sales ($)")
    st.pyplot(fig)

    st.subheader("Forecast Values")
    fc_table = pd.DataFrame({
        "Month": [d.strftime('%B %Y') for d in fdates],
        "Forecasted Sales": [round(v, 2) for v in fvals],
        "Lower Bound": [round(v, 2) for v in flower],
        "Upper Bound": [round(v, 2) for v in fupper],
    })
    st.table(fc_table)

    m1, m2 = st.columns(2)
    m1.metric("Model MAE (holdout)", f"${mae:,.0f}")
    m2.metric("Model RMSE (holdout)", f"${rmse:,.0f}")

# ============================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================
elif page == "3. Anomaly Report":
    st.title("Anomaly Report")
    st.caption("Isolation Forest and rolling Z-score (>2 std dev from 8-week rolling mean) applied to weekly sales.")

    weekly = compute_anomalies()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly.index, weekly['Sales'], color='#4C72B0', label='Weekly Sales', linewidth=1.2)
    iso_pts = weekly[weekly['iso_anomaly']]
    z_pts = weekly[weekly['z_anomaly']]
    ax.scatter(iso_pts.index, iso_pts['Sales'], color='#C44E52', marker='^', s=90, label='Isolation Forest', zorder=5)
    ax.scatter(z_pts.index, z_pts['Sales'], color='#DD8452', marker='x', s=90, label='Z-Score', zorder=5)
    ax.legend()
    ax.set_title("Weekly Sales with Detected Anomalies")
    st.pyplot(fig)

    st.subheader("Detected Anomalies")
    report = weekly[weekly['iso_anomaly'] | weekly['z_anomaly']].copy()
    report['Flagged By'] = report.apply(
        lambda r: 'Both' if r['iso_anomaly'] and r['z_anomaly'] else ('Isolation Forest' if r['iso_anomaly'] else 'Z-Score'),
        axis=1)
    report = report[['Sales', 'z_score', 'Flagged By']].sort_values('Sales', ascending=False)
    report.columns = ['Sales', 'Z-Score', 'Flagged By']
    report.index.name = 'Week Of'
    st.dataframe(report.style.format({'Sales': '${:,.0f}', 'Z-Score': '{:.2f}'}))

# ============================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================
elif page == "4. Product Demand Segments":
    st.title("Product Demand Segments")
    st.caption("K-Means clustering (k=4) on sub-category level: total sales, YoY growth, volatility, avg order value.")

    clusters = compute_clusters()

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Set2.colors
    for c in sorted(clusters['Cluster'].unique()):
        sub = clusters[clusters['Cluster'] == c]
        label = f"Cluster {c}: {sub['ClusterLabel'].iloc[0]}"
        ax.scatter(sub['PC1'], sub['PC2'], s=140, color=colors[c], label=label)
        for name, row in sub.iterrows():
            ax.annotate(name, (row['PC1'], row['PC2']), fontsize=8, xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.set_title("Sub-Category Clusters (PCA projection)")
    st.pyplot(fig)

    st.subheader("Sub-Category → Cluster Table")
    display_df = clusters[['TotalSales', 'YoYGrowth', 'Volatility', 'AvgOrderValue', 'ClusterLabel']].copy()
    display_df.columns = ['Total Sales', 'YoY Growth', 'Volatility', 'Avg Order Value', 'Cluster']
    st.dataframe(display_df.style.format({
        'Total Sales': '${:,.0f}', 'YoY Growth': '{:.1%}',
        'Volatility': '${:,.0f}', 'Avg Order Value': '${:,.0f}'
    }))

    st.subheader("Recommended Stocking Strategy")
    strategy_map = {
        'High Volume, Stable Demand': 'Maintain steady safety stock, standard reorder-point replenishment.',
        'Growing Demand': 'Forecast at account level; lean buffer stock; rely on supplier lead-time agreements.',
        'Low Volume, Stable Demand': 'Minimize holding cost; lean/just-in-time ordering or drop-ship.',
        'Declining Demand': 'Reduce standing inventory; shift to order-on-demand; investigate root cause.',
    }
    for label, strat in strategy_map.items():
        if label in clusters['ClusterLabel'].values:
            st.markdown(f"**{label}:** {strat}")
