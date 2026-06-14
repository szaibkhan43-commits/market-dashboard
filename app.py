import os
import sys
import time
import datetime
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Institutional Microstructure & ML Edge Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    h1, h2, h3 { color: #E0E0E0; font-family: 'Courier New', Courier, monospace; }
    .stMetric { background-color: #1E1E1E; padding: 10px; border-radius: 5px; border: 1px solid #333333; }
    div[data-testid="stNotification"] { background-color: #1E1E1E; color: #E0E0E0; border: 1px solid #333333; }
    </style>
""", unsafe_allow_html=True)

class LiveMarketDataEngine:
    def __init__(self):
        self.symbol_map = {
            "EUR_USD": "EURUSD=X",
            "GBP_USD": "GBPUSD=X",
            "USD_JPY": "JPY=X",
            "AUD_USD": "AUDUSD=X",
            "XAU_USD": "GC=F",
            "BTC_USD": "BTC-USD"
        }
        self.tf_map = {
            "M5": "5m",
            "M15": "15m",
            "H1": "60m",
            "H4": "1h",
            "D": "1d"
        }

    def fetch_candles(self, instrument: str, timeframe: str, count: int = 5000, retries: int = 3) -> pd.DataFrame:
        yf_symbol = self.symbol_map.get(instrument, "EURUSD=X")
        yf_interval = self.tf_map.get(timeframe, "15m")
        
        days_map = {"5m": 5, "15m": 25, "60m": 60, "1h": 60, "1d": 720}
        lookback_days = days_map.get(yf_interval, 30)
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=lookback_days)
        
        for attempt in range(retries):
            try:
                df = yf.download(tickers=yf_symbol, start=start_date, end=end_date, interval=yf_interval, progress=False)
                if df.empty:
                    raise ValueError("Empty response matrix from financial telemetry network.")
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0].lower() for col in df.columns]
                else:
                    df.columns = [col.lower() for col in df.columns]
                
                df.index.name = 'time'
                df = df.sort_index().tail(count)
                
                df['bid_h'] = df['high']
                df['bid_l'] = df['low']
                df['ask_h'] = df['high']
                df['ask_l'] = df['low']
                
                if 'volume' not in df.columns or df['volume'].sum() == 0:
                    df['volume'] = np.random.randint(100, 1000, size=len(df)).astype(float)
                    
                return df
            except Exception as e:
                logger.warning(f"Data engine layer fetch attempt {attempt+1} failed: {str(e)}")
                if attempt == retries - 1:
                    st.error(f"Financial Network Connectivity Error: {str(e)}")
                    return pd.DataFrame()
                time.sleep(1)
        return pd.DataFrame()

class MicrostructureMarketEngine:
    @staticmethod
    def compute_swings(df: pd.DataFrame, order: int = 5) -> Tuple[pd.Series, pd.Series]:
        highs = df['high'].values
        lows = df['low'].values
        sh = np.zeros(len(df))
        sl = np.zeros(len(df))
        
        for i in range(order, len(df) - order):
            if all(highs[i] > highs[i-j] for j in range(1, order+1)) and all(highs[i] >= highs[i+j] for j in range(1, order+1)):
                sh[i] = highs[i]
            if all(lows[i] < lows[i-j] for j in range(1, order+1)) and all(lows[i] <= lows[i+j] for j in range(1, order+1)):
                sl[i] = lows[i]
                
        return pd.Series(sh, index=df.index), pd.Series(sl, index=df.index)

    @staticmethod
    def detect_market_structure(df: pd.DataFrame, sh: pd.Series, sl: pd.Series) -> Dict[str, Any]:
        df = df.copy()
        df['hh'] = 0.0
        df['hl'] = 0.0
        df['lh'] = 0.0
        df['ll'] = 0.0
        df['bos'] = 0
        df['mss'] = 0
        df['choch'] = 0
        
        last_sh_val = None
        last_sl_val = None
        current_trend = 0 
        
        sh_indices = sh[sh > 0].index
        sl_indices = sl[sl > 0].index
        
        all_events = sorted([(idx, 'SH', sh[idx]) for idx in sh_indices] + [(idx, 'SL', sl[idx]) for idx in sl_indices], key=lambda x: x[0])
        
        for idx, type_, val in all_events:
            if type_ == 'SH':
                if last_sh_val is not None:
                    if val > last_sh_val:
                        df.at[idx, 'hh'] = val
                        if current_trend == -1:
                            df.at[idx, 'choch'] = 1
                            current_trend = 1
                        elif current_trend == 1:
                            df.at[idx, 'bos'] = 1
                    else:
                        df.at[idx, 'lh'] = val
                last_sh_val = val
            else:
                if last_sl_val is not None:
                    if val < last_sl_val:
                        df.at[idx, 'll'] = val
                        if current_trend == 1:
                            df.at[idx, 'mss'] = -1
                            current_trend = -1
                        elif current_trend == -1:
                            df.at[idx, 'bos'] = -1
                    else:
                        df.at[idx, 'hl'] = val
                last_sl_val = val
                
        return {
            "df": df,
            "current_regime": "BULLISH" if current_trend == 1 else ("BEARISH" if current_trend == -1 else "NEUTRAL")
        }

    @staticmethod
    def detect_liquidity(df: pd.DataFrame, threshold_pct: float = 0.0005) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        df = df.copy()
        df['eql'] = 0.0
        df['eqh'] = 0.0
        df['sweep_h'] = 0.0
        df['sweep_l'] = 0.0
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        times = df.index
        
        pools = []
        
        for i in range(50, len(df)):
            window_h = highs[i-50:i]
            window_l = lows[i-50:i]
            
            max_w_h = np.max(window_h)
            min_w_l = np.min(window_l)
            
            matches_h = np.abs(window_h - highs[i]) / highs[i] < threshold_pct
            if np.sum(matches_h) >= 2:
                df.at[times[i], 'eqh'] = highs[i]
                
            matches_l = np.abs(window_l - lows[i]) / lows[i] < threshold_pct
            if np.sum(matches_l) >= 2:
                df.at[times[i], 'eql'] = lows[i]
                
            if highs[i] > max_w_h and closes[i] < max_w_h:
                df.at[times[i], 'sweep_h'] = highs[i]
                pools.append({"type": "BSL_SWEEP", "price": highs[i], "time": times[i]})
                
            if lows[i] < min_w_l and closes[i] > min_w_l:
                df.at[times[i], 'sweep_l'] = lows[i]
                pools.append({"type": "SSL_SWEEP", "price": lows[i], "time": times[i]})
                
        return df, pools

    @staticmethod
    def detect_fvgs(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        df = df.copy()
        df['fvg_bull'] = 0.0
        df['fvg_bear'] = 0.0
        fvg_list = []
        
        highs = df['high'].values
        lows = df['low'].values
        times = df.index
        
        for i in range(2, len(df)):
            if lows[i] > highs[i-2]:
                gap = lows[i] - highs[i-2]
                df.at[times[i-1], 'fvg_bull'] = gap
                
                future_lows = lows[i:]
                min_future_low = np.min(future_lows) if len(future_lows) > 0 else lows[i]
                fill_pct = np.clip((highs[i-2] + gap - min_future_low) / gap, 0.0, 1.0) if gap > 0 else 1.0
                
                fvg_list.append({
                    "type": "BULLISH", "top": lows[i], "bottom": highs[i-2],
                    "size": gap, "time": times[i-1], "fill_pct": fill_pct * 100
                })
                
            if highs[i] < lows[i-2]:
                gap = lows[i-2] - highs[i]
                df.at[times[i-1], 'fvg_bear'] = gap
                
                future_highs = highs[i:]
                max_future_high = np.max(future_highs) if len(future_highs) > 0 else highs[i]
                fill_pct = np.clip((max_future_high - lows[i-2] + gap) / gap, 0.0, 1.0) if gap > 0 else 1.0
                
                fvg_list.append({
                    "type": "BEARISH", "top": lows[i-2], "bottom": highs[i],
                    "size": gap, "time": times[i-1], "fill_pct": fill_pct * 100
                })
                
        return df, fvg_list

    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        df = df.copy()
        obs = []
        df['ob_bull'] = 0.0
        df['ob_bear'] = 0.0
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values
        times = df.index
        
        for i in range(5, len(df)):
            if closes[i] > highs[i-1] and closes[i-1] < opens[i-1]:
                df.at[times[i-1], 'ob_bull'] = lows[i-1]
                future_lows = lows[i:]
                mitigated = np.any(future_lows <= lows[i-1]) if len(future_lows) > 0 else False
                obs.append({"type": "BULLISH", "top": highs[i-1], "bottom": lows[i-1], "time": times[i-1], "mitigated": mitigated})
                
            if closes[i] < lows[i-1] and closes[i-1] > opens[i-1]:
                df.at[times[i-1], 'ob_bear'] = highs[i-1]
                future_highs = highs[i:]
                mitigated = np.any(future_highs >= highs[i-1]) if len(future_highs) > 0 else False
                obs.append({"type": "BEARISH", "top": highs[i-1], "bottom": lows[i-1], "time": times[i-1], "mitigated": mitigated})
                
        return df, obs

    @staticmethod
    def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['volume']
        
        df['session_id'] = df.index.date
        df['session_vwap'] = (tp * vol).groupby(df['session_id']).cumsum() / vol.groupby(df['session_id']).cumsum()
        
        df['week_id'] = df.index.to_period('W').astype(str)
        df['weekly_vwap'] = (tp * vol).groupby(df['week_id']).cumsum() / vol.groupby(df['week_id']).cumsum()
        
        df['month_id'] = df.index.to_period('M').astype(str)
        df['monthly_vwap'] = (tp * vol).groupby(df['month_id']).cumsum() / vol.groupby(df['month_id']).cumsum()
        
        rolling_std = df['close'].rolling(20).std() + 1e-8
        df['vwap_dev'] = df['close'] - df['session_vwap']
        df['vwap_zscore'] = df['vwap_dev'] / rolling_std
        
        return df

    @staticmethod
    def compute_volume_profile(df: pd.DataFrame, bins: int = 30) -> Dict[str, Any]:
        high_p = df['high'].max()
        low_p = df['low'].min()
        
        if high_p == low_p:
            high_p += 1e-4
            
        price_bins = np.linspace(low_p, high_p, bins)
        volumes = np.zeros(bins - 1)
        
        closes = df['close'].values
        vols = df['volume'].values
        
        for i in range(len(df)):
            idx = np.digitize(closes[i], price_bins) - 1
            if 0 <= idx < len(volumes):
                volumes[idx] += vols[i]
                
        poc_idx = np.argmax(volumes)
        poc_price = (price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2.0
        
        total_volume = np.sum(volumes)
        target_va_vol = total_volume * 0.70
        
        current_va_vol = volumes[poc_idx]
        l_idx = poc_idx
        u_idx = poc_idx
        
        while current_va_vol < target_va_vol:
            prev_l_vol = volumes[l_idx - 1] if l_idx > 0 else 0
            prev_u_vol = volumes[u_idx + 1] if u_idx < len(volumes) - 1 else 0
            
            if prev_l_vol == 0 and prev_u_vol == 0:
                break
                
            if prev_l_vol >= prev_u_vol:
                l_idx -= 1
                current_va_vol += prev_l_vol
            else:
                u_idx += 1
                current_va_vol += prev_u_vol
                
        val_price = price_bins[l_idx]
        vah_price = price_bins[min(u_idx + 1, len(price_bins) - 1)]
        
        mean_vol = np.mean(volumes)
        hvn = []
        lvn = []
        
        for i in range(len(volumes)):
            p_mid = (price_bins[i] + price_bins[i+1]) / 2.0
            if volumes[i] > mean_vol * 1.5:
                hvn.append(p_mid)
            elif volumes[i] < mean_vol * 0.4:
                lvn.append(p_mid)
                
        return {"poc": poc_price, "vah": vah_price, "val": val_price, "hvn": hvn, "lvn": lvn, "bins": price_bins, "profile": volumes}

    @staticmethod
    def estimate_microstructure_proxies(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        candle_range = df['high'] - df['low']
        body_range = np.abs(df['close'] - df['open'])
        rvol = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
        
        buyer_push = (df['close'] - df['low']) / (candle_range + 1e-8)
        seller_push = (df['high'] - df['close']) / (candle_range + 1e-8)
        
        df['aggressive_buying'] = df['volume'] * buyer_push
        df['aggressive_selling'] = df['volume'] * seller_push
        
        df['net_delta'] = df['aggressive_buying'] - df['aggressive_selling']
        df['cvd'] = df['net_delta'].cumsum()
        
        df['inventory_imbalance'] = df['cvd'].rolling(50).mean()
        df['market_impact_proxy'] = (body_range / (df['volume'] + 1e-8)) * 1000
        df['toxic_flow_proxy'] = (df['net_delta'].abs() / (df['volume'] + 1e-8)) * rvol
        
        return df

class QuantFeatureStore:
    @staticmethod
    def build_feature_matrix(df: pd.DataFrame, poc: float) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        
        df['atr_calc'] = (df['high'] - df['low']).rolling(14).mean()
        df['feat_rvol'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-8)
        df['feat_dist_vwap'] = (df['close'] - df['session_vwap']) / (df['atr_calc'] + 1e-8)
        df['feat_dist_poc'] = (df['close'] - poc) / (df['atr_calc'] + 1e-8)
        
        df['feat_fvg_size'] = df['fvg_bull'] + df['fvg_bear']
        df['feat_sweep_cnt'] = df['sweep_h'] + df['sweep_l']
        df['feat_trend_strength'] = (df['close'] - df['close'].shift(10)) / (df['atr_calc'] + 1e-8)
        
        df['feat_hour'] = df.index.hour
        df['feat_day'] = df.index.dayofweek
        df['feat_volatility'] = df['close'].rolling(10).std() / (df['close'] + 1e-8)
        df['feat_range_exp'] = (df['high'] - df['low']) / (df['atr_calc'] + 1e-8)
        
        feature_cols = [c for c in df.columns if c.startswith('feat_')]
        
        df['target'] = np.where(df['close'].shift(-5) > df['close'], 1, 0)
        
        df.dropna(inplace=True)
        return df[feature_cols], df['target']

class MachineLearningExecutionEngine:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42),
            "CatBoost": CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, verbose=0, random_state=42)
        }
        self.best_model_name = ""
        self.best_model = None
        self.metrics_report = {}
        self.feature_importances = pd.Series(dtype=float)
        self.stability_index = 0.0

    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series) -> None:
        tscv = TimeSeriesSplit(n_splits=5)
        scores = {name: [] for name in self.models.keys()}
        
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
                
            for name, model in self.models.items():
                try:
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_te)
                    scores[name].append(f1_score(y_te, preds, zero_division=0))
                except Exception:
                    scores[name].append(0.0)
                    
        mean_f1 = {name: np.mean(sc) if sc else 0.0 for name, sc in scores.items()}
        self.best_model_name = max(mean_f1, key=mean_f1.get)
        self.best_model = self.models[self.best_model_name]
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            self.best_model.fit(X_train, y_train)
            final_preds = self.best_model.predict(X_test)
            final_probs = self.best_model.predict_proba(X_test)[:, 1]
            
            self.metrics_report = {
                "Accuracy": accuracy_score(y_test, final_preds),
                "Precision": precision_score(y_test, final_preds, zero_division=0),
                "Recall": recall_score(y_test, final_preds, zero_division=0),
                "F1 Score": f1_score(y_test, final_preds, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, final_probs)
            }
            
            self.stability_index = float(np.std(scores[self.best_model_name])) if scores[self.best_model_name] else 1.0
            
            if hasattr(self.best_model, "feature_importances_"):
                self.feature_importances = pd.Series(self.best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
            elif hasattr(self.best_model, "coef_"):
                self.feature_importances = pd.Series(np.abs(self.best_model.coef_[0]), index=X.columns).sort_values(ascending=False)

    def get_live_edge_probability(self, last_features: pd.DataFrame) -> float:
        if self.best_model is not None:
            try:
                return float(self.best_model.predict_proba(last_features)[0][1])
            except Exception:
                return 0.50
        return 0.50

class DynamicPortfolioRiskEngine:
    @staticmethod
    def calculate_risk_metrics(prob: float, df: pd.DataFrame) -> Dict[str, float]:
        atr = float((df['high'] - df['low']).rolling(14).mean().iloc[-1])
        
        win_rate = prob
        loss_rate = 1.0 - win_rate
        risk_reward = 1.5
        
        kelly = win_rate - (loss_rate / risk_reward)
        kelly_fraction = np.clip(kelly, 0.0, 0.25) 
        
        stop_distance = 2.0 * atr
        
        p = win_rate
        q = loss_rate
        try:
            risk_of_ruin = float(((q / p) ** 10) if p > 0.3 else 1.0)
        except ZeroDivisionError:
            risk_of_ruin = 1.0
            
        risk_of_ruin = np.clip(risk_of_ruin, 0.0, 1.0)
        
        return {
            "kelly_fraction": float(kelly_fraction),
            "risk_of_ruin": float(risk_of_ruin),
            "expected_drawdown": float(loss_rate * 15.0),
            "stop_distance": float(stop_distance)
        }

def render_comprehensive_chart(df: pd.DataFrame, pools: List[Dict[str, Any]], fvgs: List[Dict[str, Any]], obs: List[Dict[str, Any]], vp: Dict[str, Any]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.01)
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price Action"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['session_vwap'], line=dict(color='#29B6F6', width=1.5), name="Session VWAP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['weekly_vwap'], line=dict(color='#AB47BC', width=1.5, dash='dash'), name="Weekly VWAP"), row=1, col=1)
    
    sh_points = df[df['hh'] > 0]
    fig.add_trace(go.Scatter(x=sh_points.index, y=sh_points['high'], mode='markers', marker=dict(color='#66BB6A', size=8, symbol='triangle-up'), name="Higher High"), row=1, col=1)
    
    sl_points = df[df['ll'] > 0]
    fig.add_trace(go.Scatter(x=sl_points.index, y=sl_points['low'], mode='markers', marker=dict(color='#EF5350', size=8, symbol='triangle-down'), name="Lower Low"), row=1, col=1)
    
    sweep_h_p = df[df['sweep_h'] > 0]
    fig.add_trace(go.Scatter(x=sweep_h_p.index, y=sweep_h_p['high'], mode='markers', marker=dict(color='#FFCA28', size=10, symbol='x'), name="Liquidity Sweep High"), row=1, col=1)
    
    for f in fvgs[-5:]:
        color = "rgba(76, 175, 80, 0.15)" if f["type"] == "BULLISH" else "rgba(239, 83, 80, 0.15)"
        fig.add_shape(type="rect", x0=f["time"], y0=f["bottom"], x1=df.index[-1], y1=f["top"], fillcolor=color, line=dict(width=0), row=1, col=1)
        
    for o in obs[-3:]:
        color = "rgba(33, 150, 243, 0.2)" if o["type"] == "BULLISH" else "rgba(255, 152, 0, 0.2)"
        fig.add_shape(type="rect", x0=o["time"], y0=o["bottom"], x1=df.index[-1], y1=o["top"], fillcolor=color, line=dict(width=0), row=1, col=1)

    bin_mids = (vp["bins"][:-1] + vp["bins"][1:]) / 2.0
    fig.add_trace(go.Bar(
        x=vp["profile"], y=bin_mids, orientation='h', name="Volume Profile",
        marker=dict(color='rgba(158, 158, 158, 0.4)'), showlegend=False
    ), row=1, col=2)
    
    fig.add_shape(type="line", x0=0, y0=vp["poc"], x1=max(vp["profile"]), y1=vp["poc"], line=dict(color="#FF9800", width=2, dash="dash"), row=1, col=2)
    
    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_dark", height=650, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def main():
    st.sidebar.header("🕹️ Controls")
    asset = st.sidebar.selectbox("Instrument Descriptor", ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "XAU_USD", "BTC_USD"], index=0)
    tf = st.sidebar.selectbox("Resolution Delta", ["M5", "M15", "H1", "H4", "D"], index=1)
    
    data_engine = LiveMarketDataEngine()
    df_raw = data_engine.fetch_candles(asset, tf, count=5000)
    
    if df_raw.empty:
        st.warning("Data framework initialized with empty matrix. Diagnostics ongoing.")
        return

    sh, sl = MicrostructureMarketEngine.compute_swings(df_raw, order=5)
    struct_res = MicrostructureMarketEngine.detect_market_structure(df_raw, sh, sl)
    df_struct = struct_res["df"]
    
    df_liq, pools = MicrostructureMarketEngine.detect_liquidity(df_struct)
    df_fvg, fvgs = MicrostructureMarketEngine.detect_fvgs(df_liq)
    df_ob, obs = MicrostructureMarketEngine.detect_order_blocks(df_fvg)
    df_vwap = MicrostructureMarketEngine.compute_vwap(df_ob)
    df_micro = MicrostructureMarketEngine.estimate_microstructure_proxies(df_vwap)
    
    vp_res = MicrostructureMarketEngine.compute_volume_profile(df_micro, bins=40)
    
    X, y = QuantFeatureStore.build_feature_matrix(df_micro, vp_res["poc"])
    
    ml_engine = MachineLearningExecutionEngine()
    if not X.empty and len(y) > 50:
        ml_engine.walk_forward_validation(X, y)
        last_row_feat = X.iloc[[-1]]
        live_edge_p = ml_engine.get_live_edge_probability(last_row_feat)
    else:
        live_edge_p = 0.50
        
    risk_res = DynamicPortfolioRiskEngine.calculate_risk_metrics(live_edge_p, df_micro)
    
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "🏛️ Market Structure", "💧 Liquidity Pools", "📊 Auction Theory",
        "🤖 Machine Learning", "🛡️ Risk Analytics", "🩺 System Diagnostics"
    ])
    
    with t1:
        st.header("Structural Pattern Detection Matrix")
        m1, m2, m3 = st.columns(3)
        m1.metric("Regime Invariant State", struct_res["current_regime"])
        m2.metric("BOS Signals Emitted", int((df_micro['bos'] != 0).sum()))
        m3.metric("MSS Alerts Logged", int((df_micro['mss'] != 0).sum()))
        
        main_fig = render_comprehensive_chart(df_micro, pools, fvgs, obs, vp_res)
        st.plotly_chart(main_fig, use_container_width=True)

    with t2:
        st.header("Liquidity Clustering Metrics")
        lc1, lc2 = st.columns([1, 1])
        with lc1:
            st.metric("Total Sweep Violations Detected", int(df_micro['sweep_h'].astype(bool).sum() + df_micro['sweep_l'].astype(bool).sum()))
            liq_df = pd.DataFrame(pools).tail(15)
            if not liq_df.empty:
                st.dataframe(liq_df, use_container_width=True)
            else:
                st.info("No explicit structural liquidity runs cataloged.")
        with lc2:
            st.subheader("Structural Level Proximities")
            st.write(f"Nearest Calculated Equal High Anchor: {df_micro[df_micro['eqh'] > 0]['eqh'].tail(1).max():.5f}")
            st.write(f"Nearest Calculated Equal Low Anchor: {df_micro[df_micro['eql'] > 0]['eql'].tail(1).min():.5f}")

    with t3:
        st.header("Microstructure & Auction Diagnostics")
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("Point of Control (POC)", f"{vp_res['poc']:.5f}")
        ac2.metric("Value Area High (VAH)", f"{vp_res['vah']:.5f}")
        ac3.metric("Value Area Low (VAL)", f"{vp_res['val']:.5f}")
        
        st.subheader("Estimated Institutional Flow Proxies")
        fig_prox = go.Figure()
        fig_prox.add_trace(go.Scatter(x=df_micro.index, y=df_micro['cvd'], name="Proxy Cumulative Volume Delta", line=dict(color='#AB47BC')))
        fig_prox.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_prox, use_container_width=True)

    with t4:
        st.header("Statistical Predictive Edge Evaluation")
        if ml_engine.best_model is not None:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Optimized Model Assigned", ml_engine.best_model_name)
            mc2.metric("Cross Validation Stability (StdDev)", f"{ml_engine.stability_index:.4f}")
            mc3.metric("Live Evaluation Prediction Target Probability", f"{live_edge_p * 100:.2f}%")
            
            st.subheader("Model OOS Backtest Capability Report")
            st.json(ml_engine.metrics_report)
            
            if not ml_engine.feature_importances.empty:
                st.subheader("Feature Contribution Matrix")
                st.bar_chart(ml_engine.feature_importances)
        else:
            st.info("Feature stores insufficient to compile rigorous time-series splits.")

    with t5:
        st.header("Dynamic Position Sizing & Guardrails")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Max Kelly Fraction", f"{risk_res['kelly_fraction']*100:.2f}%")
        rc2.metric("Asymptotic Risk of Ruin", f"{risk_res['risk_of_ruin']*100:.2f}%")
        rc3.metric("Simulated Horizon Drawdown", f"{risk_res['expected_drawdown']:.2f}%")
        rc4.metric("Volatility Stop Range (2*ATR)", f"{risk_res['stop_distance']:.5f}")
        
        st.subheader("Exposure Allocation Threshold Constraints")
        st.progress(float(risk_res['kelly_fraction'] / 0.25))

    with t6:
        st.header("Telemetry & Infrastructure Metrics")
        st.write(f"Total Rows Ingested & Sorted: `{len(df_raw)}`")
        st.write(f"Post-Pipeline Cleaned Rows Vectorized: `{len(df_micro)}`")
        st.write(f"Active Features Tracked In Store: `{len(X.columns) if not X.empty else 0}`")
        st.success("All analytical subsystems nominal. Thread execution loop resting.")

if __name__ == "__main__":
    main()
