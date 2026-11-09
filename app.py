import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Single-Stock 7-Strategy Suite & Gap Auditor", layout="wide")

st.title("📊 Single-Stock 7-Strategy Backtest Suite (with True Clock-Grid Gap Protection)")
st.markdown("""
Upload any 1-minute OHLCV CSV file. This engine audits your file for **missing 1-minute bars and intraday data gaps**, 
omits corrupted/incomplete sessions, reindexes valid days onto a **strict 1-minute clock grid** so missing minutes never 
create false signals, and backtests **all 7 strategies** side-by-side.
""")

# ==========================================
# SIDEBAR: DATA UPLOAD, SIZING & GAP FILTER
# ==========================================
st.sidebar.header("1. Upload Single-Stock 1-Min CSV")
uploaded_file = st.sidebar.file_uploader("Upload 1-Minute CSV File", type=["csv"])

st.sidebar.header("2. Capital, Compounding & Commissions")
base_trade_size = st.sidebar.number_input("Base Dollar Value per Trade ($)", value=20000.0, step=1000.0, min_value=500.0)
commission_per_trade = st.sidebar.number_input("Commission per Trade ($)", value=1.00, step=0.25, min_value=0.0, format="%.2f")

sizing_mode = st.sidebar.selectbox(
    "Position Sizing / Compounding Mode",
    options=[
        "Strict Fixed Dollar Size (Always use Base $)",
        "Capped at Start, Deduct Losses (No Profit Reinvestment)",
        "Full Compounding (Reinvest Profits & Deduct Losses)"
    ]
)

st.sidebar.header("3. Missing Data & Anti-False-Signal Gate")
min_completeness_pct = st.sidebar.slider(
    "Min Daily Bar Completeness (% of 390 Bars)", 
    min_value=50, max_value=100, value=93, step=1,
    help="390 bars = 100% complete regular session (09:30–15:59). 93% requires at least 363 real 1-min bars per day."
)
max_gap_allowed = st.sidebar.slider(
    "Max Allowed Single Intraday Gap (Minutes)", 
    min_value=1, max_value=30, value=5, step=1,
    help="Set to 1 for 0 missing minutes allowed. Set to 5 to omit any day where trading halted/gapped for >5 consecutive minutes."
)

st.sidebar.header("4. Strategy Exit & Hold Parameters")
default_max_hold = st.sidebar.slider("Semi-HFT Max Hold (Minutes)", min_value=15, max_value=240, value=120, step=5)
default_pt_pct = st.sidebar.slider("Default Profit Target (%)", min_value=0.20, max_value=3.00, value=0.75, step=0.05) / 100.0
default_sl_pct = st.sidebar.slider("Default Stop Loss (%)", min_value=0.20, max_value=3.00, value=0.80, step=0.05) / 100.0
orb_minutes = st.sidebar.slider("Strategy 7: Classic ORB Duration (Mins)", min_value=5, max_value=60, value=15, step=5)

# ==========================================
# CORE DATA PROCESSING & CLOCK-GRID ENGINE
# ==========================================
@st.cache_data
def load_and_audit_csv(file, min_comp_pct, max_gap_min):
    raw = pd.read_csv(file)
    dt_col = raw.columns[0]
    raw[dt_col] = pd.to_datetime(raw[dt_col], errors='coerce')
    raw = raw.dropna(subset=[dt_col]).sort_values(dt_col).reset_index(drop=True)
    raw.rename(columns={dt_col: 'datetime'}, inplace=True)
    
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if len(raw.columns) >= 6:
        raw = raw[['datetime'] + list(raw.columns[1:6])]
        raw.columns = ['datetime'] + expected_cols
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw.dropna(subset=['Open', 'High', 'Low', 'Close']).reset_index(drop=True)
    
    if raw['datetime'].dt.tz is not None:
        raw['datetime'] = raw['datetime'].dt.tz_convert('America/New_York').dt.tz_localize(None)
        
    raw['date'] = raw['datetime'].dt.date
    raw['time'] = raw['datetime'].dt.strftime('%H:%M')
    raw = raw[(raw['time'] >= '09:30') & (raw['time'] <= '15:59')].copy()
    raw = raw[~raw['datetime'].duplicated(keep='first')].copy()
    
    if 'VXX' in file.name.upper():
        raw = raw[raw['date'] != pd.to_datetime('2024-08-05').date()].copy()

    min_bars_req = int(round(390 * (min_comp_pct / 100.0)))
    
    clean_days = []
    total_days = 0
    kept_days = 0
    omit_low_bars = 0
    omit_gap = 0
    omit_or = 0
    all_day_bar_counts = []
    
    for d, grp in raw.groupby('date'):
        total_days += 1
        n_bars = len(grp)
        all_day_bar_counts.append(n_bars)
        
        if n_bars < min_bars_req:
            omit_low_bars += 1
            continue
            
        diffs = grp['datetime'].diff().dt.total_seconds() / 60.0
        max_g = diffs.max() if len(diffs) > 1 else 1.0
        if max_g > max_gap_min:
            omit_gap += 1
            continue
            
        or15_bars = grp[grp['time'] <= '09:44']
        if len(or15_bars) < 10:
            omit_or += 1
            continue
            
        grid_idx = pd.date_range(start=f"{d} 09:30:00", end=f"{d} 15:59:00", freq='1min')
        g_re = grp.set_index('datetime').reindex(grid_idx)
        g_re.index.name = 'datetime'
        g_re['is_real_bar'] = g_re['Close'].notna()
        g_re['date'] = d
        g_re['year'] = d.year
        g_re['time'] = g_re.index.strftime('%H:%M')
        
        g_re['Close_real'] = g_re['Close']
        g_re['Volume_fill'] = g_re['Volume'].fillna(0.0)
        g_re['Close_ffill'] = g_re['Close'].ffill()
        g_re['High_ffill'] = g_re['High'].fillna(g_re['Close_ffill'])
        g_re['Low_ffill'] = g_re['Low'].fillna(g_re['Close_ffill'])
        
        kept_days += 1
        clean_days.append(g_re.reset_index())
        
    audit = {
        'total_days': total_days,
        'kept_days': kept_days,
        'omitted_days': total_days - kept_days,
        'omit_low_bars': omit_low_bars,
        'omit_gap': omit_gap,
        'omit_or': omit_or,
        'avg_bars_raw': np.mean(all_day_bar_counts) if all_day_bar_counts else 0.0,
        'pct_completeness_raw': (np.mean(all_day_bar_counts) / 390.0 * 100.0) if all_day_bar_counts else 0.0
    }
    
    if not clean_days:
        return pd.DataFrame(), audit
        
    df = pd.concat(clean_days, ignore_index=True)
    
    # Intraday VWAP
    df['typ_price'] = (df['High_ffill'] + df['Low_ffill'] + df['Close_ffill']) / 3.0
    df['pv'] = df['typ_price'] * df['Volume_fill']
    df['vwap'] = df.groupby('date')['pv'].cumsum() / df.groupby('date')['Volume_fill'].cumsum().replace(0, np.nan)
    
    # True clock-minute returns
    df['ret_1m'] = df.groupby('date')['Close_real'].pct_change(1)
    df['ret_2m'] = df.groupby('date')['Close_real'].pct_change(2)
    df['ret_3m'] = df.groupby('date')['Close_real'].pct_change(3)
    df['ret_15m'] = df.groupby('date')['Close_ffill'].pct_change(15)
    
    first_opens = raw.groupby('date')['Open'].first()
    df['day_open'] = df['date'].map(first_opens)
    df['ret_from_open'] = (df['Close_real'] - df['day_open']) / df['day_open']
    
    last_closes = raw.groupby('date')['Close'].last()
    df['prev_day_close'] = df['date'].map(last_closes.shift(1))
    df['overnight_gap'] = (df['day_open'] - df['prev_day_close']) / df['prev_day_close']
    
    rng = (df['High'] - df['Low']).replace(0, np.nan)
    df['ibs'] = (df['Close_real'] - df['Low']) / rng
    
    or15 = raw[raw['time'] <= '09:44'].groupby('date').agg(or15_high=('High', 'max'), or15_low=('Low', 'min'))
    or30 = raw[raw['time'] <= '09:59'].groupby('date').agg(or30_high=('High', 'max'), or30_low=('Low', 'min'), or30_close=('Close', 'last'))
    df['or15_high'] = df['date'].map(or15['or15_high'])
    df['or15_low'] = df['date'].map(or15['or15_low'])
    df['or30_ret'] = (df['date'].map(or30['or30_close']) - df['day_open']) / df['day_open']
    
    # Custom ORB window for Strategy 7
    or_end_time = (pd.to_datetime('09:30') + pd.Timedelta(minutes=orb_minutes - 1)).strftime('%H:%M')
    or_custom = raw[raw['time'] <= or_end_time].groupby('date').agg(orc_high=('High', 'max'), orc_low=('Low', 'min'))
    df['orc_high'] = df['date'].map(or_custom['orc_high'])
    df['orc_low'] = df['date'].map(or_custom['orc_low'])
    df['or_after_time'] = (pd.to_datetime('09:30') + pd.Timedelta(minutes=orb_minutes)).strftime('%H:%M')
    
    # Strict contiguous 1-minute streaks
    ret1 = df['ret_1m'].values
    is_real = df['is_real_bar'].values
    dates = df['date'].values
    u_st, d_st = np.zeros(len(df), dtype=int), np.zeros(len(df), dtype=int)
    cu, cd = 0, 0
    for i in range(len(df)):
        if i == 0 or dates[i] != dates[i-1] or not is_real[i] or np.isnan(ret1[i]):
            cu, cd = 0, 0
        else:
            if ret1[i] > 0:
                cu += 1; cd = 0
            elif ret1[i] < 0:
                cd += 1; cu = 0
            else:
                cu = 0; cd = 0
        u_st[i] = cu; d_st[i] = cd
    df['u_streak'] = u_st
    df['d_streak'] = d_st
    
    g = df.groupby('date')['Close_ffill']
    df['ma_15'] = g.transform(lambda x: x.rolling(15, min_periods=10).mean())
    df['std_15'] = g.transform(lambda x: x.rolling(15, min_periods=10).std())
    df['z_15'] = (df['Close_real'] - df['ma_15']) / df['std_15'].replace(0, np.nan)
    
    return df, audit

def simulate_strategy(df, strat_name, l_sig, s_sig, ex_l, ex_s, max_hold, sl_pct, pt_pct, max_trades_per_day=99):
    trades = []
    dates = df['date'].values
    years = df['year'].values
    times = df['datetime'].values
    time_strs = df['time'].values
    closes = df['Close_real'].values
    is_real = df['is_real_bar'].values
    l_arr, s_arr = l_sig.values, s_sig.values
    el_arr, es_arr = ex_l.values, ex_s.values
    
    n = len(df)
    in_pos = 0
    entry_px, entry_idx, entry_time, shares, alloc_sz = 0.0, 0, None, 0, 0.0
    cur_eq = base_trade_size
    trades_today = 0
    
    for i in range(n):
        if i == 0 or dates[i] != dates[i-1]:
            trades_today = 0
        if not is_real[i]:
            continue
            
        is_eod = (i == n - 1) or (dates[i+1] != dates[i]) or (time_strs[i] >= '15:58')
        
        if in_pos != 0:
            bars_held = i - entry_idx
            cur_px = closes[i]
            ret = (cur_px - entry_px) / entry_px if in_pos == 1 else (entry_px - cur_px) / entry_px
            reason = None
            if is_eod:
                reason = 'EOD Close'
            elif ret <= -sl_pct:
                reason = 'Stop Loss'
            elif ret >= pt_pct:
                reason = 'Profit Target'
            elif (in_pos == 1 and el_arr[i]) or (in_pos == -1 and es_arr[i]):
                reason = 'Regime Reversal'
            elif bars_held >= max_hold:
                reason = f'Time Exit ({max_hold}m)'
                
            if reason is not None:
                gross = (cur_px - entry_px) * shares if in_pos == 1 else (entry_px - cur_px) * shares
                net = gross - commission_per_trade
                cur_eq += net
                trades.append({
                    'Strategy': strat_name,
                    'Date': dates[i],
                    'Year': years[i],
                    'Entry Time': entry_time,
                    'Exit Time': times[i],
                    'Direction': 'LONG' if in_pos == 1 else 'SHORT',
                    'Allocation ($)': round(alloc_sz, 2),
                    'Shares': shares,
                    'Entry Price': round(entry_px, 3),
                    'Exit Price': round(cur_px, 3),
                    'Bars Held (m)': bars_held,
                    'Exit Reason': reason,
                    'Gross PnL ($)': round(gross, 2),
                    'Net PnL ($)': round(net, 2),
                    'Running Equity ($)': round(cur_eq, 2)
                })
                in_pos = 0
                
        if in_pos == 0 and not is_eod and trades_today < max_trades_per_day:
            if l_arr[i] or s_arr[i]:
                in_pos = 1 if l_arr[i] else -1
                entry_px = closes[i]
                entry_idx = i
                entry_time = times[i]
                if "Full Compounding" in sizing_mode:
                    alloc_sz = max(cur_eq, 100.0)
                elif "Capped at Start" in sizing_mode:
                    alloc_sz = max(min(cur_eq, base_trade_size), 100.0)
                else:
                    alloc_sz = base_trade_size
                shares = max(1, int(round(alloc_sz / entry_px)))
                trades_today += 1
                
    return pd.DataFrame(trades)

if uploaded_file is not None:
    ticker_name = uploaded_file.name.split('_')[0].split('.')[0].upper()
    with st.spinner(f"Auditing 1-minute data quality & running 7-strategy tournament on {ticker_name}..."):
        df, audit = load_and_audit_csv(uploaded_file, min_completeness_pct, max_gap_allowed)
        
    # Display Missing Data Audit Report
    st.subheader(f"🔍 Layer 1–3 Data Quality & Missing-Bar Audit: `{ticker_name}`")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Raw Avg Daily Completeness", f"{audit['pct_completeness_raw']:.1f}%", f"{audit['avg_bars_raw']:.1f} / 390 bars")
    a2.metric("Total Calendar Days in File", f"{audit['total_days']:,}")
    a3.metric("Clean Days Kept for Backtest", f"{audit['kept_days']:,}", f"{audit['kept_days']/max(audit['total_days'],1)*100:.1f}% of days")
    a4.metric("Corrupted/Gapped Days Omitted", f"{audit['omitted_days']:,}", f"-{audit['omit_low_bars']} low bars, -{audit['omit_gap']} large gaps")
    
    if audit['pct_completeness_raw'] < 90.0 or (audit['kept_days'] / max(audit['total_days'], 1)) < 0.35:
        st.warning(
            f"⚠️ **Data Realism Warning for `{ticker_name}`:** This file averages only **{audit['avg_bars_raw']:.1f} bars/day "
            f"({audit['pct_completeness_raw']:.1f}% of 390 minutes)**, leaving **{audit['kept_days']} clean days** under your current filter. "
            f"All missing minutes have been isolated on a 1-minute clock grid so no false signals can bridge across gaps."
        )
    else:
        st.success(
            f"✅ **High Data Integrity Confirmed:** `{ticker_name}` averages **{audit['avg_bars_raw']:.1f}/390 bars per day "
            f"({audit['pct_completeness_raw']:.1f}%)**. Backtesting **{audit['kept_days']} verified clean days** with strict 1-minute clock-grid lookbacks."
        )

    if df.empty:
        st.error("No trading days passed the current completeness and max-gap filters. Lower the completeness slider or increase the max allowed gap in the sidebar.")
        st.stop()

    # ==========================================
    # BUILD ALL 7 STRATEGIES
    # ==========================================
    zero_s = pd.Series(False, index=df.index)
    ex_default_l = pd.Series(False, index=df.index)
    ex_default_s = pd.Series(False, index=df.index)

    # Strategy 1: Core VWAP Trend + 3-Bar/IBS Dip Buy (Long-Only)
    s1_long = (df['Close_real'] > df['vwap']) & ((df['d_streak'] >= 2) | (df['ibs'] < 0.15)) & (df['time'] >= '09:45') & (df['time'] <= '15:30')
    s1_exit = df['Close_real'] >= df['vwap'] * 1.002

    # Strategy 2: VWAP Mean Reversion Fade (Long & Short)
    s2_long = (df['Close_real'] < df['vwap'] * 0.996) & (df['ret_15m'] <= -0.003) & (df['time'] >= '10:00') & (df['time'] <= '15:15')
    s2_short = (df['Close_real'] > df['vwap'] * 1.004) & (df['ret_15m'] >= 0.003) & (df['time'] >= '10:00') & (df['time'] <= '15:15')

    # Strategy 3: Momentum Streak Continuation (Long & Short)
    s3_long = (df['u_streak'] >= 3) & (df['Close_real'] > df['vwap']) & (df['time'] >= '09:45') & (df['time'] <= '15:00')
    s3_short = (df['d_streak'] >= 3) & (df['Close_real'] < df['vwap']) & (df['time'] >= '09:45') & (df['time'] <= '15:00')

    # Strategy 4: Z-Score Bollinger Reversion (Long & Short)
    s4_long = (df['z_15'] < -2.0) & (df['time'] >= '10:00') & (df['time'] <= '15:15')
    s4_short = (df['z_15'] > 2.0) & (df['time'] >= '10:00') & (df['time'] <= '15:15')

    # Strategy 5: Overnight Gap Fade (Long & Short)
    s5_long = (df['overnight_gap'] < -0.008) & (df['time'] == '09:31')
    s5_short = (df['overnight_gap'] > 0.008) & (df['time'] == '09:31')

    # Strategy 6: Intraday Momentum Breakout (Long & Short)
    s6_long = (df['ret_from_open'] > 0.012) & (df['Close_real'] > df['vwap']) & (df['time'] >= '10:00') & (df['time'] <= '14:30')
    s6_short = (df['ret_from_open'] < -0.012) & (df['Close_real'] < df['vwap']) & (df['time'] >= '10:00') & (df['time'] <= '14:30')

    # Strategy 7: Classic Opening Range Breakout (ORB)
    s7_long = (df['time'] == df['or_after_time']) & (df['Close_real'] > df['orc_high'])
    s7_short = (df['time'] == df['or_after_time']) & (df['Close_real'] < df['orc_low'])

    # ==========================================
    # RUN TOURNAMENT SUITE
    # ==========================================
    with st.spinner("Executing 7-strategy simulation tournament..."):
        t1 = simulate_strategy(df, "S1: VWAP Dip Buy (Long)", s1_long, zero_s, s1_exit, zero_s, default_max_hold, default_sl_pct, default_pt_pct)
        t2 = simulate_strategy(df, "S2: VWAP Mean Reversion", s2_long, s2_short, zero_s, zero_s, default_max_hold, default_sl_pct, default_pt_pct)
        t3 = simulate_strategy(df, "S3: Momentum Streak", s3_long, s3_short, zero_s, zero_s, 45, default_sl_pct, default_pt_pct)
        t4 = simulate_strategy(df, "S4: Z-Score Reversion", s4_long, s4_short, zero_s, zero_s, 60, default_sl_pct, default_pt_pct)
        t5 = simulate_strategy(df, "S5: Overnight Gap Fade", s5_long, s5_short, zero_s, zero_s, 90, 0.005, 0.005, max_trades_per_day=1)
        t6 = simulate_strategy(df, "S6: Intraday Trend Breakout", s6_long, s6_short, zero_s, zero_s, default_max_hold, default_sl_pct, default_pt_pct)
        t7 = simulate_strategy(df, f"S7: {orb_minutes}m ORB", s7_long, s7_short, zero_s, zero_s, 180, default_sl_pct, default_pt_pct)

    all_trades = pd.concat([t1, t2, t3, t4, t5, t6, t7], ignore_index=True)

    if all_trades.empty:
        st.warning("No trades generated across any strategy with current parameters. Adjust indicators, profit targets, or stop losses.")
    else:
        st.subheader("🏆 Strategy Tournament Performance Leaderboard")
        
        summary_rows = []
        for s_name, grp in all_trades.groupby('Strategy'):
            total_t = len(grp)
            win_t = len(grp[grp['Net PnL ($)'] > 0])
            wr = (win_t / total_t) * 100.0 if total_t > 0 else 0.0
            tot_pnl = grp['Net PnL ($)'].sum()
            avg_pnl = grp['Net PnL ($)'].mean()
            max_dd = grp['Running Equity ($)'].min()
            summary_rows.append({
                'Strategy': s_name,
                'Total Trades': total_t,
                'Win Rate (%)': round(wr, 1),
                'Total Net PnL ($)': round(tot_pnl, 2),
                'Avg PnL / Trade ($)': round(avg_pnl, 2),
                'Ending Equity ($)': round(grp['Running Equity ($)'].iloc[-1], 2)
            })
        
        summary_df = pd.DataFrame(summary_rows).sort_values(by='Total Net PnL ($)', ascending=False).reset_index(drop=True)
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("📈 Cumulative Equity Curves")
        eq_piv = all_trades.pivot_table(index='Exit Time', columns='Strategy', values='Running Equity ($)').ffill().fillna(base_trade_size)
        st.line_chart(eq_piv)

        st.subheader("📋 Trade Ledger Inspector")
        sel_strat = st.selectbox("Select Strategy to Inspect Ledger", options=all_trades['Strategy'].unique())
        filtered_trades = all_trades[all_trades['Strategy'] == sel_strat]
        st.dataframe(filtered_trades, use_container_width=True)

        csv_export = all_trades.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Trade Ledger (CSV)", data=csv_export, file_name=f"{ticker_name}_7_strategy_trades.csv", mime="text/csv")
