import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime
import os
import csv

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Nifty Swing Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== CONFIG ==================
CAPITAL = 200000
MAX_RISK_RS = 2000
LOT_SIZE = 65
SL_PREMIUM_PCT = 0.45
SIGNAL_LOG_FILE = "nifty_signal_log.csv"

# ================== FUNCTIONS ==================
@st.cache_data(ttl=300)
def get_data():
    nifty = yf.download("^NSEI", period="8mo", interval="1d", progress=False, auto_adjust=True)
    vix = yf.download("^INDIAVIX", period="8mo", interval="1d", progress=False, auto_adjust=True)

    # Fix multi-level columns
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    nifty = nifty[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    vix = vix[['Close']].dropna()
    return nifty, vix

def calculate_supertrend(df, period=10, multiplier=3):
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    hl2 = (high + low) / 2
    atr = ta.volatility.average_true_range(high, low, close, window=period)

    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)

    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1, index=df.index)  # default to 1

    for i in range(period, len(df)):
        if close.iloc[i] > upperband.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

            if direction.iloc[i] == 1 and lowerband.iloc[i] < lowerband.iloc[i-1]:
                lowerband.iloc[i] = lowerband.iloc[i-1]
            if direction.iloc[i] == -1 and upperband.iloc[i] > upperband.iloc[i-1]:
                upperband.iloc[i] = upperband.iloc[i-1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lowerband.iloc[i]
        else:
            supertrend.iloc[i] = upperband.iloc[i]

    # Forward fill any remaining NaNs
    supertrend = supertrend.ffill()
    direction = direction.ffill().fillna(1).astype(int)

    return supertrend, direction

def get_support_resistance(df):
    last_close = float(df["Close"].iloc[-1])
    prev = df.iloc[-2]

    high = float(prev["High"])
    low = float(prev["Low"])
    close = float(prev["Close"])

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)

    swing_high = float(df["High"].tail(15).max())
    swing_low = float(df["Low"].tail(15).min())

    base = int(last_close // 50) * 50
    round_levels = [base + i * 50 for i in range(-6, 7)]

    supports = sorted(set([
        round(s1), round(s2), round(swing_low)
    ] + [l for l in round_levels if l < last_close - 20]), reverse=True)[:3]

    resistances = sorted(set([
        round(r1), round(r2), round(swing_high)
    ] + [l for l in round_levels if l > last_close + 20]))[:3]

    return {
        "pivot": round(pivot),
        "supports": supports,
        "resistances": resistances
    }

def suggest_strikes(close, bias):
    atm = int(round(close / 50) * 50)

    if "BULLISH" in bias:
        return {
            "primary": f"{atm} CE (ATM)",
            "aggressive": f"{atm+50} / {atm+100} CE",
            "conservative": f"{atm-50} CE"
        }
    elif "BEARISH" in bias:
        return {
            "primary": f"{atm} PE (ATM)",
            "aggressive": f"{atm-50} / {atm-100} PE",
            "conservative": f"{atm+50} PE"
        }
    else:
        return {
            "primary": "No clear strike",
            "aggressive": "-",
            "conservative": "-"
        }

def analyze(nifty, vix_df):
    df = nifty.copy()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["EMA20"] = ta.trend.ema_indicator(close, window=20)
    df["EMA50"] = ta.trend.ema_indicator(close, window=50)
    df["RSI"] = ta.momentum.rsi(close, window=14)

    macd = ta.trend.MACD(close)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()

    df["Supertrend"], df["ST_direction"] = calculate_supertrend(df)
    df["Vol_MA20"] = volume.rolling(20).mean()

    # Drop rows that still have NaN after calculations
    df = df.dropna()

    last = df.iloc[-1]
    close_price = float(last["Close"])

    current_vix = float(vix_df["Close"].iloc[-1]) if len(vix_df) > 0 else 15.0
    vix_status = "High ⚠️" if current_vix > 18 else "Moderate" if current_vix > 14 else "Low ✅"

    bullish_score = 0
    bearish_score = 0

    if close_price > float(last["EMA20"]) > float(last["EMA50"]):
        bullish_score += 2
    elif close_price < float(last["EMA20"]) < float(last["EMA50"]):
        bearish_score += 2

    rsi = float(last["RSI"])
    if rsi > 55:
        bullish_score += 1
    elif rsi < 45:
        bearish_score += 1

    if float(last["MACD"]) > float(last["MACD_signal"]) and float(last["MACD_hist"]) > 0:
        bullish_score += 1.5
    elif float(last["MACD"]) < float(last["MACD_signal"]) and float(last["MACD_hist"]) < 0:
        bearish_score += 1.5

    st_dir = int(last["ST_direction"])
    if st_dir == 1:
        bullish_score += 1.5
    else:
        bearish_score += 1.5

    vol_status = "Above Avg" if float(last["Volume"]) > float(last["Vol_MA20"]) * 1.1 else "Normal/Low"

    if bullish_score >= bearish_score + 2 and current_vix < 19:
        bias = "BULLISH 🟢"
        option_pref = "CE (Call)"
        strength = "Strong" if bullish_score >= 5 else "Moderate"
    elif bearish_score >= bullish_score + 2 and current_vix < 19:
        bias = "BEARISH 🔴"
        option_pref = "PE (Put)"
        strength = "Strong" if bearish_score >= 5 else "Moderate"
    else:
        bias = "NEUTRAL / MIXED ⚪"
        option_pref = "Stay out"
        strength = "Weak / High VIX"

    sr = get_support_resistance(df)
    strikes = suggest_strikes(close_price, bias)

    return {
        "date": str(df.index[-1].date()),
        "close": round(close_price, 1),
        "ema20": round(float(last["EMA20"]), 1),
        "ema50": round(float(last["EMA50"]), 1),
        "rsi": round(rsi, 1),
        "macd_status": "Bullish" if float(last["MACD"]) > float(last["MACD_signal"]) else "Bearish",
        "supertrend": "Green" if st_dir == 1 else "Red",
        "volume_status": vol_status,
        "vix": round(current_vix, 2),
        "vix_status": vix_status,
        "bias": bias,
        "strength": strength,
        "option_pref": option_pref,
        "supports": sr["supports"],
        "resistances": sr["resistances"],
        "pivot": sr["pivot"],
        "strikes": strikes
    }

def log_signal(info):
    file_exists = os.path.isfile(SIGNAL_LOG_FILE)
    with open(SIGNAL_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Date", "Close", "Bias", "Strength", "VIX", "Option", "Primary Strike"])
        writer.writerow([
            info["date"], info["close"],
            info["bias"].replace("🟢", "").replace("🔴", "").replace("⚪", "").strip(),
            info["strength"], info["vix"], info["option_pref"], info["strikes"]["primary"]
        ])

# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown(f"**Capital:** ₹{CAPITAL:,}")
st.sidebar.markdown(f"**Max Risk/Trade:** ₹{MAX_RISK_RS:,}")
st.sidebar.markdown(f"**Lot Size:** {LOT_SIZE}")
st.sidebar.markdown("---")

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info("Educational tool only.\nNot financial advice.")

# ================== MAIN ==================
st.title("📈 Nifty 50 Swing Analyzer")
st.caption(f"Last updated: {datetime.now().strftime('%d %b %Y, %I:%M %p IST')}")

try:
    with st.spinner("Fetching latest market data..."):
        nifty, vix = get_data()
        info = analyze(nifty, vix)
        log_signal(info)

    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nifty Close", f"{info['close']:,.1f}")
    col2.metric("Bias", info['bias'])
    col3.metric("India VIX", f"{info['vix']}", info['vix_status'])
    col4.metric("Preferred", info['option_pref'])

    st.markdown("---")

    left, right = st.columns([1.4, 1])

    with left:
        st.subheader("📊 Analysis Details")
        st.markdown(f"""
        **Date:** {info['date']}  
        **Strength:** {info['strength']}  
        **EMA 20 / 50:** {info['ema20']} / {info['ema50']}  
        **RSI (14):** {info['rsi']}  
        **MACD:** {info['macd_status']}  
        **Supertrend:** {info['supertrend']}  
        **Volume:** {info['volume_status']}
        """)

        st.markdown("#### Key Levels")
        st.write(f"**Pivot:** {info['pivot']}")
        st.write(f"**Support:** {', '.join(map(str, info['supports']))}")
        st.write(f"**Resistance:** {', '.join(map(str, info['resistances']))}")

        st.markdown("#### Suggested Strikes")
        st.success(f"**Primary:** {info['strikes']['primary']}")
        st.info(f"**Aggressive:** {info['strikes']['aggressive']}")
        st.warning(f"**Conservative:** {info['strikes']['conservative']}")

    with right:
        st.subheader("🧮 Position Size Calculator")
        premium = st.number_input("Enter Live Premium of the option", min_value=1.0, value=120.0, step=1.0)

        if premium > 0:
            risk_per_lot = premium * SL_PREMIUM_PCT * LOT_SIZE
            max_lots = int(MAX_RISK_RS / risk_per_lot)
            capital_required = premium * LOT_SIZE * max_lots

            st.metric("Risk per Lot (45% SL)", f"₹{risk_per_lot:,.0f}")
            st.metric("Max Lots Allowed", f"{max_lots}")
            st.metric("Capital Required", f"₹{capital_required:,.0f}")

            if max_lots == 0:
                st.error("Even 1 lot exceeds your ₹2,000 risk limit.")
            elif max_lots == 1:
                st.success("Recommended: Take **1 lot** only.")
            else:
                st.success(f"You can safely take up to **{max_lots} lots**.")

    st.markdown("---")
    st.caption("⚠️ Educational tool only. Not financial advice. Paper trade first.")

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Try clicking the Refresh button in the sidebar.")