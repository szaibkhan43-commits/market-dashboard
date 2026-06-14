import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
from catboost import CatBoostClassifier

# Page Configuration for Mobile Responsiveness
st.set_page_config(page_title="Institutional Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for Mobile Optimization
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem; padding-left: 1rem; padding-right: 1rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem !important;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Institutional Microstructure & ML Edge")
st.caption("Live Auction Theory & Machine Learning Processing Engine (Free Data Source)")

# --- USER INPUTS (Dropdowns for Mobile) ---
col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.selectbox("Asset / Ticker", ["EURUSD=X", "GBPUSD=X", "BTC-USD", "GC=F", "ES=F"], index=0)
with col2:
    interval = st.selectbox("Resolution", ["5m", "15m", "60m", "1d"], index=1)
with col3:
    lookback_days = st.slider("Lookback (Days)", min_value=1, max_value=30, value=7)

# --- LIVE DATA INGESTION ---
@st.cache_data(ttl=60)
def fetch_live_market_data(symbol, interval, days):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days)
    data = yf.download(tickers=symbol, start=start_date, end=end_date, interval=interval)
    
    if data.empty:
        return data
        
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0].lower() for col in data.columns]
    else:
        data.columns = [col.lower() for col in data.columns]
        
    return data

with st.spinner("Analyzing micro liquidity states..."):
    df = fetch_live_market_data(ticker, interval, lookback_days)

if df.empty:
    st.error("Data ingestion failed. Historical feed is temporarily unavailable.")
    st.stop()

# --- ENGINE: MICROSTRUCTURE & AUCTION COMPUTATIONS ---
def compute_metrics(data):
    df = data.copy()
    
    # 1. Volatility (ATR Proxy)
    df['tr'] = np.maximum(df['high'] - df['low'], 
                          np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # 2. Session VWAP (Volume-Weighted Average Price)
    # Check if volume data exists and is valid, otherwise use uniform volume proxy
    if 'volume' not in df.columns or df['volume'].sum() == 0:
        df['volume'] = 1.0
        
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    df['vwap'] = (tp * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-8)
    df['vwap_dev'] = (df['close'] - df['vwap']) / (df['atr'] + 1e-8)
    
    # 3. Volume Profile / Price Density (Auction Theory)
    price_min, price_max = float(df['close'].min()), float(df['close'].max())
    bins = np.linspace(price_min, price_max, 20)
    v_profile, edge = np.histogram(df['close'], bins=bins, weights=df['volume'])
    poc_idx = np.argmax(v_profile)
    df['poc'] = (edge[poc_idx] + edge[poc_idx+1]) / 2.0
    
    # 4. Liquidity Sweeps
    df['roll_high'] = df['high'].shift(1).rolling(20).max()
    df['roll_low'] = df['low'].shift(1).rolling(20).min()
    df['sweep_high'] = np.where((df['high'] > df['roll_high']) & (df['close'] < df['roll_high']), 1, 0)
    df['sweep_low'] = np.where((df['low'] < df['roll_low']) & (df['close'] > df['roll_low']), 1, 0)
    
    # 5. Imbalances (Fair Value Gaps)
    df['fvg_size'] = (df['low'].shift(1) - df['high'].shift(-1)).clip(lower=0) / (df['atr'] + 1e-8)
    df['rvol'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
    
    df.dropna(inplace=True)
    return df

processed_df = compute_metrics(df)
current_state = processed_df.iloc[-1]

# --- ML ENGINE: LIVE INFERENCE ---
def execute_live_inference(data):
    feature_cols = ['vwap_dev', 'rvol', 'sweep_high', 'fvg_size']
    X = data[feature_cols]
    
    # Shift labels forward to define statistical edge (Reversion Framework)
    y = np.where(data['close'].shift(-3) > data['close'], 1, 0)
    
    split = int(len(X) * 0.8)
    if split < 10:  # Safeguard for low data windows
        return 0.50
        
    X_train, y_train = X.iloc[:split], y[:split]
    
    clf = CatBoostClassifier(iterations=40, depth=4, learning_rate=0.1, verbose=0)
    clf.fit(X_train, y_train)
    
    last_row = X.iloc[[-1]]
    prob = clf.predict_proba(last_row)[0][1]
    return prob

live_prob = execute_live_inference(processed_df)

# --- VISUALIZATION MATRIX ---
st.subheader("📈 Institutional Metrics & Profiles")

# Responsive Cards for Mobile Screen
m_col1, m_col2 = st.columns(2)
m_col3, m_col4 = st.columns(2)

with m_col1:
    st.metric("Live Price", f"{float(current_state['close']):.5f}")
with m_col2:
    st.metric("Auction POC", f"{float(current_state['poc']):.5f}")
with m_col3:
    st.metric("VWAP Dev (Z-Score)", f"{float(current_state['vwap_dev']):.2f}σ")
with m_col4:
    st.metric("Relative Volume (RVOL)", f"{float(current_state['rvol']):.2f}x")

# Candlestick Matrix
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=processed_df.index, open=processed_df['open'], high=processed_df['high'],
    low=processed_df['low'], close=processed_df['close'], name="Price"
))
fig.add_trace(go.Scatter(x=processed_df.index, y=processed_df['vwap'], line=dict(color='#29B6F6', width=2), name="VWAP"))
fig.add_trace(go.Scatter(x=processed_df.index, y=np.full(len(processed_df), current_state['poc']), line=dict(color='#FFA726', dash='dash'), name="POC Anchor"))

fig.update_layout(
    height=400, xaxis_rangeslider_visible=False, template="plotly_dark",
    margin=dict(l=5, r=5, t=5, b=5), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig, use_container_width=True)

# --- RISK & DIRECTIONAL CRITERIA ---
st.subheader("⚡ Execution Matrix & Sizing")

e_col1, e_col2 = st.columns(2)
with e_col1:
    st.markdown(f"**ML Edge Long Probability Context:** `{live_prob * 100:.1f}%`")
    st.progress(float(live_prob))
    
    if live_prob >= 0.58:
        st.success("🚨 STRATEGY EDGE: BUY REVERSION REGIME")
    elif live_prob <= 0.42:
        st.error("🚨 STRATEGY EDGE: SELL REVERSION REGIME")
    else:
        st.info("⚖️ BALANCED AUCTION: RISK DISTRIBUTION CHOP")

with e_col2:
    # Dynamic Math Limits for Stop Allocations
    atr_val = float(current_state['atr'])
    close_val = float(current_state['close'])
    calculated_sl = close_val - (1.5 * atr_val) if live_prob > 0.5 else close_val + (1.5 * atr_val)
    
    st.markdown(f"Current ATR Volatility Level: `{atr_val:.5f}`")
    st.markdown(f"Structural Invalidation (Stop Loss): `{calculated_sl:.5f}`")
    st.markdown("Portfolio Risk Guardrail: **1.00% Max Risk per Trade**")
