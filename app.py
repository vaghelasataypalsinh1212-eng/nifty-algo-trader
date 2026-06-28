import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="NIFTY 50 Algo Backtest",
    page_icon="📈",
    layout="wide"
)

st.title("📈 NIFTY 50 Algo Backtest Engine")
st.markdown("---")

# ─── SIDEBAR CONTROLS ───────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

capital = st.sidebar.number_input(
    "Capital (₹)", min_value=10000,
    max_value=10000000, value=100000, step=10000
)

risk_per_trade = st.sidebar.slider(
    "Risk per Trade (%)", min_value=0.5,
    max_value=5.0, value=1.0, step=0.5
)

daily_loss_limit = st.sidebar.slider(
    "Daily Max Loss (%)", min_value=1.0,
    max_value=10.0, value=3.0, step=0.5
)

weekly_loss_limit = st.sidebar.slider(
    "Weekly Max Loss (%)", min_value=2.0,
    max_value=20.0, value=6.0, step=1.0
)

timeframe = st.sidebar.selectbox(
    "Timeframe", ["1h", "5m"]
)

lookback_days = st.sidebar.selectbox(
    "Lookback Period",
    [30, 60, 90, 180, 365],
    index=2
)

run_backtest = st.sidebar.button("🚀 Run Backtest", type="primary")

# ─── DATA LAYER ──────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker, period_days, interval):
    end = datetime.today()
    start = end - timedelta(days=period_days)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval=interval,
        progress=False,
        auto_adjust=True
    )
    if df.empty:
        return pd.DataFrame()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

# ─── STRATEGY LAYER (PLUG-IN) ────────────────────────────────────
def apply_strategy(df):
    """
    Strategy plug-in zone.
    Apni strategy yahan add karo.
    Return karo df with 'signal' column:
        1  = BUY
       -1  = SELL
        0  = NO TRADE
    Aur 'sl' column = stop loss price
    Aur 'tp' column = take profit price
    """
    df = df.copy()

    # Example placeholder — EMA crossover (replace with your strategy)
    df["ema_fast"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["Close"].ewm(span=21, adjust=False).mean()

    df["signal"] = 0
    df.loc[df["ema_fast"] > df["ema_slow"], "signal"] = 1
    df.loc[df["ema_fast"] < df["ema_slow"], "signal"] = -1

    atr = df["High"] - df["Low"]
    df["sl"] = np.where(
        df["signal"] == 1,
        df["Close"] - atr,
        df["Close"] + atr
    )
    df["tp"] = np.where(
        df["signal"] == 1,
        df["Close"] + (atr * 2),
        df["Close"] - (atr * 2)
    )
    return df

# ─── BACKTEST ENGINE ─────────────────────────────────────────────
def run_backtest_engine(df, capital, risk_pct, daily_loss_pct, weekly_loss_pct):
    risk_amount = capital * (risk_pct / 100)
    daily_limit = capital * (daily_loss_pct / 100)
    weekly_limit = capital * (weekly_loss_pct / 100)

    trades = []
    equity = capital
    equity_curve = [capital]
    daily_loss = 0.0
    weekly_loss = 0.0
    last_date = None
    last_week = None

    for i in range(1, len(df)):
        row = df.iloc[i]
        current_date = df.index[i].date()
        current_week = df.index[i].isocalendar()[1]

        if last_date != current_date:
            daily_loss = 0.0
            last_date = current_date

        if last_week != current_week:
            weekly_loss = 0.0
            last_week = current_week

        if daily_loss >= daily_limit:
            equity_curve.append(equity)
            continue
        if weekly_loss >= weekly_limit:
            equity_curve.append(equity)
            continue

        signal = row["signal"]
        if signal == 0:
            equity_curve.append(equity)
            continue

        entry = row["Close"]
        sl = row["sl"]
        tp = row["tp"]

        sl_distance = abs(entry - sl)
        if sl_distance == 0:
            equity_curve.append(equity)
            continue

        qty = int(risk_amount / sl_distance)
        if qty <= 0:
            equity_curve.append(equity)
            continue

        if signal == 1:
            pnl = (tp - entry) * qty
        else:
            pnl = (entry - tp) * qty

        pnl = round(pnl, 2)
        equity += pnl

        if pnl < 0:
            daily_loss += abs(pnl)
            weekly_loss += abs(pnl)

        trades.append({
            "date": df.index[i],
            "signal": "BUY" if signal == 1 else "SELL",
            "entry": round(entry, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "qty": qty,
            "pnl": pnl,
            "equity": round(equity, 2)
        })
        equity_curve.append(equity)

    return trades, equity_curve

# ─── METRICS CALCULATOR ──────────────────────────────────────────
def calculate_metrics(trades, capital):
    if not trades:
        return {}

    df_t = pd.DataFrame(trades)
    total_trades = len(df_t)
    wins = df_t[df_t["pnl"] > 0]
    losses = df_t[df_t["pnl"] <= 0]

    win_rate = round((len(wins) / total_trades) * 100, 2)
    total_pnl = round(df_t["pnl"].sum(), 2)
    avg_win = round(wins["pnl"].mean(), 2) if len(wins) > 0 else 0
    avg_loss = round(losses["pnl"].mean(), 2) if len(losses) > 0 else 0
    expectancy = round((win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss), 2)

    equity_vals = df_t["equity"].values
    peak = np.maximum.accumulate(equity_vals)
    drawdown = (equity_vals - peak) / peak * 100
    max_drawdown = round(drawdown.min(), 2)

    net_return = round(((df_t["equity"].iloc[-1] - capital) / capital) * 100, 2)

    return {
        "Total Trades": total_trades,
        "Win Rate (%)": win_rate,
        "Total P&L (₹)": total_pnl,
        "Net Return (%)": net_return,
        "Avg Win (₹)": avg_win,
        "Avg Loss (₹)": avg_loss,
        "Expectancy (₹)": expectancy,
        "Max Drawdown (%)": max_drawdown
    }

# ─── MAIN UI ─────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Price Chart")
    with st.spinner("Data load ho raha hai..."):
        df_raw = load_data("^NSEI", lookback_days, timeframe)

    if df_raw.empty:
        st.error("Data load nahi hua. Internet check karo ya timeframe badlo.")
    else:
        fig = go.Figure(data=[go.Candlestick(
            x=df_raw.index,
            open=df_raw["Open"],
            high=df_raw["High"],
            low=df_raw["Low"],
            close=df_raw["Close"],
            name="NIFTY 50"
        )])
        fig.update_layout(
            height=400,
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("⚙️ Capital Settings")
    st.metric("Capital", f"₹{capital:,}")
    st.metric("Risk/Trade", f"₹{int(capital * risk_per_trade / 100):,}")
    st.metric("Daily Loss Limit", f"₹{int(capital * daily_loss_limit / 100):,}")
    st.metric("Weekly Loss Limit", f"₹{int(capital * weekly_loss_limit / 100):,}")

st.markdown("---")

if run_backtest:
    if df_raw.empty:
        st.error("Pehle data load karo.")
    else:
        with st.spinner("Backtest chal raha hai..."):
            df_strategy = apply_strategy(df_raw)
            trades, equity_curve = run_backtest_engine(
                df_strategy, capital,
                risk_per_trade, daily_loss_limit, weekly_loss_limit
            )
            metrics = calculate_metrics(trades, capital)

        if not trades:
            st.warning("Koi trade nahi mila is strategy mein.")
        else:
            st.subheader("📈 Results")
            m_cols = st.columns(4)
            metric_list = list(metrics.items())
            for idx, (k, v) in enumerate(metric_list):
                with m_cols[idx % 4]:
                    st.metric(k, v)

            st.subheader("📉 Equity Curve")
            eq_fig = go.Figure()
            eq_fig.add_trace(go.Scatter(
                y=equity_curve,
                mode="lines",
                name="Equity",
                line=dict(color="#00ff88", width=2)
            ))
            eq_fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(eq_fig, use_container_width=True)

            st.subheader("📋 Trade Log")
            df_trades = pd.DataFrame(trades)
            df_trades["date"] = df_trades["date"].astype(str)
            st.dataframe(df_trades, use_container_width=True)

else:
    st.info("⬅️ Left side mein settings karo aur 'Run Backtest' press karo.")

st.markdown("---")
st.caption("Paper Trading mode — coming soon | Real broker API — coming soon")
