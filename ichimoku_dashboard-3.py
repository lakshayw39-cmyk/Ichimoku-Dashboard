"""
Ichimoku Dual-Strategy — LIVE DASHBOARD (long + short)
======================================================
Run from a terminal (NOT a notebook):

    python3 -m streamlit run ichimoku_dashboard.py

STRATEGY 1 — "Chikou v2" (inside/below cloud breakout):
  trend already aligned (TK bullish & rising) + clean Chikou breakout
  + ADX gate, triggered at/below cloud top. Mirrored for shorts.

STRATEGY 2 — "TK Cross" (early-reversal, from the classic TK+Chikou method):
  LONG : a recent Tenkan/Kijun GOLD cross BELOW the kumo (early reversal
         heads-up) -> wait -> enter on a CLEAN Chikou breakout candle,
         Tenkan sloping up, price ABOVE SMA(50), ADX gate.
  SHORT: recent TK DEATH cross ABOVE the kumo -> clean Chikou breakdown,
         Tenkan sloping down, price BELOW SMA(50), ADX gate.
  Signals on the wrong side of SMA(50) are ignored.
  Stop: beyond Kijun or the swing (whichever is wider). Exits: 50% at 1:1,
  runner rides the Kijun trail ("ride the trend as long as Kijun holds").

Both strategies share: screener tab (7-condition matrix on latest bar)
and portfolio backtest tab. Sidebar filters apply to all tabs.
"""

import time
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# yfinance screener is optional — older versions don't have it
try:
    from yfinance import screen, EquityQuery
    HAS_SCREENER = True
except ImportError:
    HAS_SCREENER = False

# ─── PAGE ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ichimoku Strategies — Live",
                   page_icon="☁️", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.2rem;}
  [data-testid="stMetric"] {
      background: rgba(28, 33, 48, 0.55);
      border: 1px solid rgba(120, 140, 190, 0.25);
      border-radius: 10px; padding: 10px 14px;
  }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS (bar-based, timeframe-agnostic) ───────────────────────────────
SLOPE_LOOKBACK   = 3
CHIKOU_SHIFT     = 26
SWING_LOOKBACK   = 10
ADX_PERIOD       = 14
SMA_PERIOD       = 50
STOP_BUFFER        = 0.005
KIJUN_TRAIL_BUFFER = 0.005
PARTIAL_RR         = 1.0
PARTIAL_FRACTION   = 0.50
INITIAL_CAPITAL    = 100_000
POSITION_SIZE      = 5_000
MAX_POSITIONS      = 20
MIN_BARS           = 140

TIMEFRAMES = {
    "15 min": ("15m", 59,   None),
    "1 Hour": ("1h",  729,  None),
    "4 Hour": ("1h",  729,  "4h"),
    "1 Day":  ("1d",  3650, None),
}

# ─── FALLBACK UNIVERSE (used if Yahoo screener is unavailable) ───────────────
FALLBACK_UNIVERSE = [
    ("NVDA","NASDAQ",3400),("MSFT","NASDAQ",3300),("AAPL","NASDAQ",3200),
    ("GOOGL","NASDAQ",2300),("AMZN","NASDAQ",2200),("META","NASDAQ",1500),
    ("AVGO","NASDAQ",1100),("TSLA","NASDAQ",1000),("BRK-B","NYSE",990),
    ("LLY","NYSE",800),("WMT","NYSE",780),("JPM","NYSE",700),("V","NYSE",620),
    ("XOM","NYSE",520),("ORCL","NYSE",510),("MA","NYSE",500),("UNH","NYSE",480),
    ("COST","NASDAQ",420),("PG","NYSE",400),("HD","NYSE",390),("NFLX","NASDAQ",380),
    ("JNJ","NYSE",370),("ABBV","NYSE",340),("BAC","NYSE",330),("CRM","NYSE",290),
    ("KO","NYSE",290),("TMUS","NASDAQ",280),("CVX","NYSE",270),("AMD","NASDAQ",260),
    ("MRK","NYSE",250),("CSCO","NASDAQ",250),("ADBE","NASDAQ",240),("PEP","NASDAQ",230),
    ("ACN","NYSE",230),("LIN","NASDAQ",220),("MCD","NYSE",220),("TMO","NYSE",210),
    ("WFC","NYSE",210),("GE","NYSE",200),("ABT","NYSE",200),("IBM","NYSE",200),
    ("QCOM","NASDAQ",190),("PM","NYSE",190),("AXP","NYSE",190),("CAT","NYSE",180),
    ("INTU","NASDAQ",180),("DIS","NYSE",180),("MS","NYSE",180),("NOW","NYSE",180),
    ("ISRG","NASDAQ",180),("VZ","NYSE",170),("GS","NYSE",170),("TXN","NASDAQ",170),
    ("BKNG","NASDAQ",160),("AMGN","NASDAQ",160),("RTX","NYSE",160),("T","NYSE",160),
    ("PLTR","NASDAQ",150),("SPGI","NYSE",150),("UBER","NYSE",150),("CMCSA","NASDAQ",150),
    ("NEE","NYSE",150),("PFE","NYSE",150),("UNP","NYSE",140),("LOW","NYSE",140),
    ("BLK","NYSE",140),("HON","NASDAQ",140),("PGR","NYSE",140),("AMAT","NASDAQ",140),
    ("ETN","NYSE",130),("TJX","NYSE",130),("BSX","NYSE",130),("SYK","NYSE",130),
    ("COP","NYSE",130),("BX","NYSE",130),("PANW","NASDAQ",130),("DHR","NYSE",130),
    ("ADP","NASDAQ",120),("FI","NYSE",120),("VRTX","NASDAQ",120),("MU","NASDAQ",120),
    ("BMY","NYSE",110),("GILD","NASDAQ",110),("SBUX","NASDAQ",110),("LRCX","NASDAQ",110),
    ("MDT","NYSE",110),("ADI","NASDAQ",110),("MMC","NYSE",110),("PLD","NYSE",100),
    ("INTC","NASDAQ",100),("KLAC","NASDAQ",100),("ANET","NYSE",100),("UPS","NYSE",100),
    ("SO","NYSE",100),("CB","NYSE",100),("ELV","NYSE",95),("DE","NYSE",95),
    ("AMT","NYSE",95),("REGN","NASDAQ",95),("ICE","NYSE",90),("SHW","NYSE",90),
    ("MO","NYSE",90),("APH","NYSE",90),("CI","NYSE",90),("DUK","NYSE",90),
    ("WM","NYSE",85),("SCHW","NYSE",85),("ZTS","NYSE",85),("CME","NASDAQ",85),
    ("AON","NYSE",80),("GEV","NYSE",80),("CL","NYSE",80),("SNPS","NASDAQ",80),
    ("CDNS","NASDAQ",80),("EQIX","NASDAQ",80),("MCK","NYSE",80),("PH","NYSE",80),
    ("CVS","NYSE",80),("ITW","NYSE",75),("TGT","NYSE",75),("MSI","NYSE",75),
    ("CRWD","NASDAQ",75),("CEG","NASDAQ",75),("PNC","NYSE",75),("USB","NYSE",70),
    ("EOG","NYSE",70),("APD","NYSE",70),("FDX","NYSE",70),("CSX","NASDAQ",70),
    ("BDX","NYSE",70),("WELL","NYSE",70),("MMM","NYSE",70),("ORLY","NASDAQ",70),
    ("AJG","NYSE",65),("TDG","NYSE",65),("MAR","NASDAQ",65),("GD","NYSE",65),
    ("EMR","NYSE",65),("NOC","NYSE",65),("ABNB","NASDAQ",65),("COF","NYSE",60),
    ("NSC","NYSE",60),("ECL","NYSE",60),("ROP","NASDAQ",60),("SLB","NYSE",60),
    ("HLT","NYSE",60),("ADSK","NASDAQ",60),("MELI","NASDAQ",60),("PYPL","NASDAQ",60),
    ("AZO","NYSE",60),("WMB","NYSE",60),("TFC","NYSE",55),("PSA","NYSE",55),
    ("CARR","NYSE",55),("SPG","NYSE",55),("OKE","NYSE",55),("SRE","NYSE",55),
    ("DLR","NYSE",55),("AEP","NASDAQ",55),("GM","NYSE",55),("KMI","NYSE",55),
    ("JCI","NYSE",55),("TRV","NYSE",55),("FTNT","NASDAQ",55),("BK","NYSE",55),
    ("URI","NYSE",50),("AFL","NYSE",50),("CPRT","NASDAQ",50),("ALL","NYSE",50),
    ("MET","NYSE",50),("PCAR","NASDAQ",50),("NXPI","NASDAQ",50),("O","NYSE",50),
    ("AMP","NYSE",50),("DASH","NASDAQ",50),("SQ","NYSE",50),("MRVL","NASDAQ",50),
    ("RY.TO","TSX",180),("SHOP.TO","TSX",130),("TD.TO","TSX",110),
    ("ENB.TO","TSX",90),("BN.TO","TSX",80),("BMO.TO","TSX",75),
    ("CP.TO","TSX",75),("CNR.TO","TSX",75),("BNS.TO","TSX",65),
    ("CNQ.TO","TSX",60),("TRI.TO","TSX",60),("CM.TO","TSX",50),
    ("SU.TO","TSX",50),("ATD.TO","TSX",50),("CSU.TO","TSX",50),
    ("TRP.TO","TSX",45),("MFC.TO","TSX",45),("WCN.TO","TSX",45),
    ("NTR.TO","TSX",35),("FNV.TO","TSX",30),("SLF.TO","TSX",30),
    ("T.TO","TSX",30),("AEM.TO","TSX",30),("WPM.TO","TSX",25),
]

# ─── INDICATORS ───────────────────────────────────────────────────────────────
def ichimoku(df):
    hi, lo = df["High"], df["Low"]
    tenkan = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
    kijun  = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    senA   = ((tenkan + kijun) / 2).shift(26)
    senB   = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    return pd.DataFrame({
        "tenkan": tenkan, "kijun": kijun,
        "cloud_top": np.maximum(senA, senB),
        "cloud_bot": np.minimum(senA, senB),
    }, index=df.index)


def adx(df, period=14):
    hi, lo, cl = df["High"], df["Low"], df["Close"]
    up_move, down_move = hi.diff(), -lo.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_cl = cl.shift(1)
    tr = pd.concat([(hi - lo), (hi - prev_cl).abs(), (lo - prev_cl).abs()],
                   axis=1).max(axis=1)
    atr      = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm  = pd.Series(plus_dm,  index=df.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    plus_di  = 100 * (plus_dm  / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False).mean()


def _arrays(df):
    """Common indicator arrays used by both strategies."""
    df = df.sort_index()
    ichi = ichimoku(df)
    return {
        "df": df, "dates": df.index, "n": len(df),
        "o": df["Open"].values, "h": df["High"].values,
        "l": df["Low"].values,  "c": df["Close"].values,
        "tenkan": ichi["tenkan"].values, "kijun": ichi["kijun"].values,
        "cloud_top": ichi["cloud_top"].values,
        "cloud_bot": ichi["cloud_bot"].values,
        "adx": adx(df, ADX_PERIOD).values,
        "sma": df["Close"].rolling(SMA_PERIOD).mean().values,
    }

# ─── STRATEGY 1: "Chikou v2" (inside/below cloud breakout) ───────────────────
def conds_v2(A, i, adx_min, clean_margin, direction, cross_lookback=None):
    """Return ordered list of (label, bool) for the 7 v2 conditions at bar i."""
    t, k = A["tenkan"][i], A["kijun"][i]
    tp, kp = A["tenkan"][i-SLOPE_LOOKBACK], A["kijun"][i-SLOPE_LOOKBACK]
    o, h, l, c = A["o"][i], A["h"][i], A["l"][i], A["c"][i]
    if direction == "long":
        ref, cref = A["h"][i-CHIKOU_SHIFT], A["cloud_top"][i]
    else:
        ref, cref = A["l"][i-CHIKOU_SHIFT], A["cloud_bot"][i]
    if any(np.isnan(v) for v in [t, k, tp, kp, A["adx"][i], ref, cref]):
        return None
    rng = h - l
    if direction == "long":
        return [
            ("TK > KJ", t > k), ("Tenkan rising", t > tp), ("Kijun rising", k > kp),
            ("Chikou breakout", c > ref),
            ("Clean break", c > ref*(1+clean_margin) and c > o and rng > 0 and c >= (h+l)/2),
            (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
            ("In/below cloud", c <= cref),
        ]
    return [
        ("TK < KJ", t < k), ("Tenkan falling", t < tp), ("Kijun falling", k < kp),
        ("Chikou breakdown", c < ref),
        ("Clean break", c < ref*(1-clean_margin) and c < o and rng > 0 and c <= (h+l)/2),
        (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
        ("In/above cloud", c >= cref),
    ]


def stop_v2(A, i, direction):
    if direction == "long":
        return np.min(A["l"][i-SWING_LOOKBACK:i+1]) * (1 - STOP_BUFFER)
    return np.max(A["h"][i-SWING_LOOKBACK:i+1]) * (1 + STOP_BUFFER)

# ─── STRATEGY 2: "TK Cross" (video method + SMA50 filter) ────────────────────
def _cross_flags(A, direction):
    """Bool array: TK gold cross below cloud (long) / death cross above (short)."""
    t, k = A["tenkan"], A["kijun"]
    n = A["n"]
    out = np.zeros(n, dtype=bool)
    for j in range(1, n):
        if any(np.isnan(v) for v in [t[j], k[j], t[j-1], k[j-1]]):
            continue
        if direction == "long":
            crossed = t[j] > k[j] and t[j-1] <= k[j-1]
            cb = A["cloud_bot"][j]
            out[j] = crossed and not np.isnan(cb) and t[j] < cb   # cross BELOW kumo
        else:
            crossed = t[j] < k[j] and t[j-1] >= k[j-1]
            ct = A["cloud_top"][j]
            out[j] = crossed and not np.isnan(ct) and t[j] > ct   # cross ABOVE kumo
    return out


def conds_v3(A, i, adx_min, clean_margin, direction, cross_lookback=15,
             cross_flags=None):
    """Ordered (label, bool) list for the 7 TK-Cross conditions at bar i."""
    t, k = A["tenkan"][i], A["kijun"][i]
    tp = A["tenkan"][i-SLOPE_LOOKBACK]
    o, h, l, c = A["o"][i], A["h"][i], A["l"][i], A["c"][i]
    sma = A["sma"][i]
    ref = A["h"][i-CHIKOU_SHIFT] if direction == "long" else A["l"][i-CHIKOU_SHIFT]
    if any(np.isnan(v) for v in [t, k, tp, A["adx"][i], ref, sma]):
        return None
    if cross_flags is None:
        cross_flags = _cross_flags(A, direction)
    j0 = max(1, i - cross_lookback)
    recent_cross = bool(cross_flags[j0:i+1].any())
    rng = h - l
    if direction == "long":
        return [
            (f"Gold cross <kumo ({cross_lookback} bars)", recent_cross),
            ("TK > KJ now", t > k),
            ("Tenkan rising", t > tp),
            ("Chikou breakout", c > ref),
            ("Clean break", c > ref*(1+clean_margin) and c > o and rng > 0 and c >= (h+l)/2),
            ("Price > SMA50", c > sma),
            (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
        ]
    return [
        (f"Death cross >kumo ({cross_lookback} bars)", recent_cross),
        ("TK < KJ now", t < k),
        ("Tenkan falling", t < tp),
        ("Chikou breakdown", c < ref),
        ("Clean break", c < ref*(1-clean_margin) and c < o and rng > 0 and c <= (h+l)/2),
        ("Price < SMA50", c < sma),
        (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
    ]


def stop_v3(A, i, direction):
    """Video rule: stop beyond Kijun or the swing — take the wider (safer)."""
    if direction == "long":
        base = min(A["kijun"][i], np.min(A["l"][i-SWING_LOOKBACK:i+1]))
        return base * (1 - STOP_BUFFER)
    base = max(A["kijun"][i], np.max(A["h"][i-SWING_LOOKBACK:i+1]))
    return base * (1 + STOP_BUFFER)


STRATEGIES = {
    "Chikou v2": {"conds": conds_v2, "stop": stop_v2, "uses_cross": False},
    "TK Cross":  {"conds": conds_v3, "stop": stop_v3, "uses_cross": True},
}

# ─── SIGNAL SCAN (shared) ─────────────────────────────────────────────────────
def find_signals(strat, ticker, df, adx_min, clean_margin, direction,
                 cross_lookback=15):
    if df is None or len(df) < MIN_BARS:
        return []
    A = _arrays(df)
    S = STRATEGIES[strat]
    cross_flags = (_cross_flags(A, direction) if S["uses_cross"] else None)
    events = []
    start = 52 + CHIKOU_SHIFT + max(SLOPE_LOOKBACK, SWING_LOOKBACK,
                                    ADX_PERIOD, SMA_PERIOD) + 2
    for i in range(start, A["n"]):
        cl = S["conds"](A, i, adx_min, clean_margin, direction,
                        cross_lookback=cross_lookback, cross_flags=cross_flags) \
             if S["uses_cross"] else \
             S["conds"](A, i, adx_min, clean_margin, direction)
        if cl is None or not all(v for _, v in cl):
            continue
        entry = A["c"][i]
        stop = S["stop"](A, i, direction)
        risk = (entry - stop) if direction == "long" else (stop - entry)
        if risk <= 0:
            continue
        tp1 = entry + PARTIAL_RR*risk if direction == "long" else entry - PARTIAL_RR*risk
        events.append({
            "ticker": ticker, "direction": direction, "strategy": strat,
            "entry_idx": i, "entry_date": A["dates"][i],
            "entry_price": entry, "init_stop": stop, "tp1": tp1,
            "risk_per_share": risk,
            "_close": A["c"], "_high": A["h"], "_low": A["l"],
            "_kijun": A["kijun"], "_dates": A["dates"], "_n": A["n"],
        })
    return events


def latest_conditions(strat, df, adx_min, clean_margin, direction,
                      cross_lookback=15):
    """Condition list on the last closed bar, for the screener."""
    if df is None or len(df) < MIN_BARS:
        return None
    A = _arrays(df)
    i = A["n"] - 1
    if i < 52 + CHIKOU_SHIFT + SMA_PERIOD:
        return None
    S = STRATEGIES[strat]
    if S["uses_cross"]:
        cl = S["conds"](A, i, adx_min, clean_margin, direction,
                        cross_lookback=cross_lookback)
    else:
        cl = S["conds"](A, i, adx_min, clean_margin, direction)
    if cl is None:
        return None
    return cl, float(A["c"][i]), float(A["adx"][i])

# ─── TRADE SIM + PORTFOLIO (shared, direction-aware) ─────────────────────────
def simulate_trade(e, shares):
    close, high, low = e["_close"], e["_high"], e["_low"]
    kijun, dates = e["_kijun"], e["_dates"]
    n, i0 = e["_n"], e["entry_idx"]
    tp1, stop = e["tp1"], e["init_stop"]
    is_long = e["direction"] == "long"
    sh1 = int(shares * PARTIAL_FRACTION)
    sh2 = shares - sh1
    hit_11, legs = False, []
    for t in range(i0 + 1, n):
        if not hit_11:
            stopped = close[t] < stop if is_long else close[t] > stop
            if stopped:
                return {"exit_date": dates[t],
                        "legs": [(close[t], dates[t], "stop", shares)]}
            reached = high[t] >= tp1 if is_long else low[t] <= tp1
            if reached:
                hit_11 = True
                legs.append((tp1, dates[t], "tp1_partial", sh1))
                if sh2 <= 0:
                    return {"exit_date": dates[t], "legs": legs}
                continue
        else:
            if np.isnan(kijun[t]):
                continue
            trail_hit = (close[t] < kijun[t] * (1 - KIJUN_TRAIL_BUFFER)
                         if is_long else
                         close[t] > kijun[t] * (1 + KIJUN_TRAIL_BUFFER))
            if trail_hit:
                legs.append((close[t], dates[t], "kijun_trail", sh2))
                return {"exit_date": dates[t], "legs": legs}
    if not hit_11:
        return {"exit_date": dates[n-1],
                "legs": [(close[n-1], dates[n-1], "open_end", shares)]}
    legs.append((close[n-1], dates[n-1], "open_end", sh2))
    return {"exit_date": dates[n-1], "legs": legs}


def run_backtest(all_events):
    all_events.sort(key=lambda x: x["entry_date"])
    cash, open_positions, closed = INITIAL_CAPITAL, [], []
    for e in all_events:
        ed = e["entry_date"]
        still = []
        for p in open_positions:
            if p["exit_date"] <= ed:
                cash += p["_exit_value"]
                closed.append(p)
            else:
                still.append(p)
        open_positions = still
        if len(open_positions) >= MAX_POSITIONS or cash < POSITION_SIZE:
            continue
        shares = int(POSITION_SIZE // e["entry_price"])
        if shares <= 0:
            continue
        cost = shares * e["entry_price"]
        cash -= cost
        sim = simulate_trade(e, shares)
        legs, exit_date = sim["legs"], sim["exit_date"]
        if e["direction"] == "long":
            pnl = sum(px*sh for px, _, _, sh in legs) - cost
        else:
            pnl = sum((e["entry_price"] - px)*sh for px, _, _, sh in legs)
        avg_exit = sum(px*sh for px, _, _, sh in legs) / shares
        hold_hrs = (exit_date - ed).total_seconds() / 3600
        open_positions.append({
            "ticker": e["ticker"], "direction": e["direction"],
            "entry_date": ed,
            "entry_price": round(e["entry_price"], 2), "shares": shares,
            "stop_loss": round(e["init_stop"], 2),
            "tp1_level": round(e["tp1"], 2),
            "exit_date": exit_date, "exit_price": round(avg_exit, 2),
            "exit_reason": "+".join(r for _, _, r, _ in legs),
            "pnl": round(pnl, 2),
            "return_%": round(pnl / cost * 100, 2),
            "hold_hours": round(hold_hrs, 1),
            "_exit_value": cost + pnl,
        })
    closed.extend(open_positions)
    for p in closed:
        p.pop("_exit_value", None)
    return closed

# ─── DATA (cached) ────────────────────────────────────────────────────────────
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_universe(exchanges, min_cap_b, max_tickers):
    rows, source = [], "Yahoo screener"
    if HAS_SCREENER:
        codes = {"NYSE": ["NYQ"], "NASDAQ": ["NMS", "NGM", "NCM"], "TSX": ["TOR"]}
        for ex in exchanges:
            query = EquityQuery('and', [
                EquityQuery('is-in', ['exchange'] + codes[ex]),
                EquityQuery('gt', ['intradaymarketcap', min_cap_b * 1e9]),
            ])
            offset = 0
            while True:
                try:
                    res = screen(query, offset=offset, size=250,
                                 sortField="intradaymarketcap", sortAsc=False)
                    quotes = res.get("quotes", [])
                    if not quotes:
                        break
                    for q in quotes:
                        if "symbol" in q:
                            rows.append({"ticker": q["symbol"], "exchange": ex,
                                         "mktcap_b": q.get("marketCap", 0) / 1e9})
                    if len(quotes) < 250:
                        break
                    offset += 250
                    time.sleep(0.4)
                except Exception:
                    break
    if not rows:
        source = "built-in fallback list (Yahoo screener unavailable)"
        rows = [{"ticker": t, "exchange": ex, "mktcap_b": cap}
                for t, ex, cap in FALLBACK_UNIVERSE
                if ex in exchanges and cap >= min_cap_b]
    uni = (pd.DataFrame(rows).drop_duplicates("ticker")
             .sort_values("mktcap_b", ascending=False)
             .head(max_tickers).reset_index(drop=True))
    return uni, source


def _normalize(df, resample_rule):
    if df.index.tz is not None:
        df = df.tz_convert("UTC")
    else:
        df = df.tz_localize("UTC")
    df = df.sort_index()
    if resample_rule:
        df = df.resample(resample_rule).agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last",
        }).dropna(how="any")
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_prices(tickers, interval, days, resample_rule):
    price_map = {}
    batch_size = 25 if interval != "1d" else 50
    batches = [list(tickers)[i:i+batch_size]
               for i in range(0, len(tickers), batch_size)]
    prog = st.progress(0.0, text="Downloading price data ...")
    for bi, batch in enumerate(batches):
        try:
            raw = yf.download(batch, period=f"{days}d", interval=interval,
                              auto_adjust=True, progress=False, threads=True)
            if raw is not None and not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    lvl = 1 if raw.columns.names[0] in ("Price", None) else 0
                    for tk in raw.columns.get_level_values(lvl).unique():
                        try:
                            sub = raw.xs(tk, axis=1, level=lvl).dropna(how="all")
                            if len(sub):
                                price_map[tk] = _normalize(sub, resample_rule)
                        except Exception:
                            pass
                elif len(batch) == 1:
                    price_map[batch[0]] = _normalize(raw, resample_rule)
        except Exception:
            pass
        prog.progress((bi + 1) / len(batches),
                      text=f"Downloading price data ... batch {bi+1}/{len(batches)}")
        time.sleep(0.2)
    prog.empty()
    return {t: d for t, d in price_map.items()
            if d is not None and len(d) >= MIN_BARS}

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
st.sidebar.title("☁️ Ichimoku Suite")
st.sidebar.caption("Chikou v2 · TK Cross")

tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=1)
interval, max_days, resample_rule = TIMEFRAMES[tf_label]

dir_choice = st.sidebar.radio("Direction", ["Long", "Short", "Both"],
                              index=0, horizontal=True)
DIRECTIONS = {"Long": ["long"], "Short": ["short"],
              "Both": ["long", "short"]}[dir_choice]

today = datetime.now(timezone.utc).date()
earliest = today - timedelta(days=max_days)
picked = st.sidebar.date_input(
    "Date range", value=(earliest, today),
    min_value=earliest, max_value=today,
    help=f"Yahoo caps {tf_label} history at {max_days} days back.")
if isinstance(picked, tuple) and len(picked) == 2:
    date_from, date_to = picked
else:
    date_from, date_to = earliest, today

min_cap = st.sidebar.slider("Min market cap ($B)", 1, 200, 5)
exchanges = st.sidebar.multiselect("Exchanges", ["NYSE", "NASDAQ", "TSX"],
                                   default=["NYSE", "NASDAQ", "TSX"])
max_tickers = st.sidebar.slider("Universe size (top-N by cap)", 25, 1000, 100,
                                step=25, help="Smaller = faster.")

st.sidebar.divider()
adx_min = st.sidebar.slider("ADX minimum", 10, 40, 20)
clean_margin = st.sidebar.slider("Clean margin (%)", 0.0, 2.0, 0.5, 0.1) / 100
cross_lookback = st.sidebar.slider("TK cross lookback (bars)", 5, 40, 15,
                                   help="TK Cross strategy: how recently the "
                                        "gold/death cross must have happened.")

st.sidebar.divider()
auto_refresh = st.sidebar.toggle("🔴 Live auto-refresh", value=False)
refresh_min = st.sidebar.slider("Refresh every (min)", 1, 30, 5,
                                disabled=not auto_refresh)
if st.sidebar.button("↻ Force refresh (clear cache)", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

if auto_refresh:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=refresh_min * 60 * 1000, key="live_refresh")
    except ImportError:
        st.sidebar.info("For auto-refresh: python3 -m pip install streamlit-autorefresh")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.title("Ichimoku Strategy Suite — Live Dashboard")
st.caption(f"**{tf_label}** · **{dir_choice}** · {date_from} → {date_to} · "
           f"cap ≥ ${min_cap}B · ADX ≥ {adx_min} · margin {clean_margin*100:.1f}% · "
           f"updated {datetime.now().strftime('%H:%M:%S')}")

if not exchanges:
    st.info("Select at least one exchange in the sidebar.")
    st.stop()

try:
    with st.spinner("Building universe ..."):
        universe, uni_source = fetch_universe(tuple(sorted(exchanges)),
                                              min_cap, max_tickers)
    if universe.empty:
        st.error("No tickers matched. Lower the market-cap filter or add exchanges.")
        st.stop()
    st.caption(f"Universe: **{len(universe)}** tickers · source: {uni_source}")

    days_needed = min((today - date_from).days + 1, max_days)
    price_map = fetch_prices(tuple(universe["ticker"]), interval,
                             days_needed, resample_rule)
    if not price_map:
        st.error("No price data downloaded — Yahoo may be throttling. "
                 "Press '↻ Force refresh' in a minute, or reduce Universe size.")
        st.stop()
    st.caption(f"Price data loaded for **{len(price_map)}** tickers.")
except Exception:
    st.error("Something failed while fetching data:")
    st.code(traceback.format_exc())
    st.stop()

cap_lookup = universe.set_index("ticker")["mktcap_b"].to_dict()

# ─── TAB RENDERERS ────────────────────────────────────────────────────────────
def render_screener(strat):
    st.subheader(f"{strat} — latest {tf_label} bar · {dir_choice.lower()}")
    rows, near_rows = [], []
    for tk, df in price_map.items():
        for d in DIRECTIONS:
            try:
                res = latest_conditions(strat, df, adx_min, clean_margin, d,
                                        cross_lookback=cross_lookback)
            except Exception:
                continue
            if res is None:
                continue
            cl, close_px, adx_val = res
            passed = sum(v for _, v in cl)
            row = {"Ticker": tk, "Dir": "🟢 LONG" if d == "long" else "🔴 SHORT",
                   "Cap $B": round(cap_lookup.get(tk, 0), 1),
                   "Close": round(close_px, 2), "ADX": round(adx_val, 1),
                   **{lbl: ("✅" if v else "—") for lbl, v in cl},
                   "Passed": f"{passed}/7"}
            if passed == 7:
                rows.append(row)
            elif passed >= 5:
                near_rows.append(row)

    c1, c2 = st.columns(2)
    c1.metric("🔥 Full signals now", len(rows))
    c2.metric("👀 Near-setups (5–6 of 7)", len(near_rows))
    if rows:
        st.success("Entry conditions ALL met on the last closed bar:")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No ticker meets all 7 conditions on the latest bar right now.")
    with st.expander(f"Near-setups — watchlist ({len(near_rows)})"):
        if near_rows:
            st.dataframe(pd.DataFrame(near_rows).sort_values("Passed", ascending=False),
                         use_container_width=True, hide_index=True)
        else:
            st.write("Nothing close right now.")


def render_backtest(strat):
    with st.spinner(f"Backtesting {strat} ..."):
        all_events = []
        for tk, df in price_map.items():
            for d in DIRECTIONS:
                try:
                    all_events.extend(find_signals(strat, tk, df, adx_min,
                                                   clean_margin, d,
                                                   cross_lookback=cross_lookback))
                except Exception:
                    continue
        lo = pd.Timestamp(date_from, tz="UTC")
        hi = pd.Timestamp(date_to, tz="UTC") + pd.Timedelta(days=1)
        all_events = [e for e in all_events if lo <= e["entry_date"] < hi]
        trades = run_backtest(all_events)

    if not trades:
        st.warning("No trades in this window with the current filters. "
                   "Try widening the date range or loosening ADX / margin / cross lookback.")
        return

    tdf = pd.DataFrame(trades).sort_values("entry_date").reset_index(drop=True)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    total_pnl = tdf["pnl"].sum()
    pf = (wins["pnl"].sum() / abs(losses["pnl"].sum())
          if losses["pnl"].sum() != 0 else float("inf"))

    eq = tdf.sort_values("exit_date").copy()
    eq["equity"] = INITIAL_CAPITAL + eq["pnl"].cumsum()
    eq["peak"] = eq["equity"].cummax()
    eq["dd_pct"] = (eq["equity"] - eq["peak"]) / eq["peak"] * 100

    m = st.columns(6)
    m[0].metric("Total P&L", f"${total_pnl:,.0f}",
                f"{total_pnl/INITIAL_CAPITAL*100:+.2f}%")
    m[1].metric("Trades", len(tdf))
    m[2].metric("Win rate", f"{len(wins)/len(tdf)*100:.1f}%")
    m[3].metric("Profit factor", f"{pf:.2f}")
    m[4].metric("Expectancy", f"${tdf['pnl'].mean():,.0f}")
    m[5].metric("Max drawdown", f"{eq['dd_pct'].min():.2f}%")

    if dir_choice == "Both" and tdf["direction"].nunique() > 1:
        s = st.columns(2)
        for col, d, icon in [(s[0], "long", "🟢"), (s[1], "short", "🔴")]:
            sub = tdf[tdf["direction"] == d]
            if len(sub):
                wr = (sub["pnl"] > 0).mean() * 100
                col.metric(f"{icon} {d.capitalize()} side",
                           f"${sub['pnl'].sum():,.0f}",
                           f"{len(sub)} trades · {wr:.0f}% win")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["exit_date"], y=eq["equity"], mode="lines",
                             name="Equity", line=dict(width=2, color="#2dd4a7")))
    fig.add_hline(y=INITIAL_CAPITAL, line_dash="dot", line_color="gray")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                      title="Equity curve (realized, by exit time)",
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True, key=f"eq_{strat}")

    cA, cB = st.columns([1, 2])
    with cA:
        st.markdown("**Exit breakdown**")
        br = (tdf.groupby(["direction", "exit_reason"])["pnl"]
                 .agg(["count", "sum"]).reset_index())
        br.columns = ["Dir", "Exit path", "Trades", "P&L $"]
        st.dataframe(br, use_container_width=True, hide_index=True)
    with cB:
        st.markdown("**P&L by month**")
        monthly = (tdf.assign(month=pd.to_datetime(tdf["entry_date"], utc=True)
                              .dt.to_period("M").astype(str))
                      .groupby("month")["pnl"].sum().reset_index())
        fig2 = go.Figure(go.Bar(x=monthly["month"], y=monthly["pnl"],
                                marker_color=np.where(monthly["pnl"] >= 0,
                                                      "#2dd4a7", "#f26d6d")))
        fig2.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                           template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True, key=f"mo_{strat}")

    st.markdown("**All trades**")
    show = tdf.copy()
    show["entry_date"] = pd.to_datetime(show["entry_date"], utc=True).dt.strftime("%Y-%m-%d %H:%M")
    show["exit_date"]  = pd.to_datetime(show["exit_date"],  utc=True).dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(show, use_container_width=True, hide_index=True, height=420)
    st.download_button("⬇ Download trades CSV",
                       show.to_csv(index=False).encode(),
                       file_name=f"{strat.replace(' ','_').lower()}_{tf_label.replace(' ','')}"
                                 f"_{dir_choice.lower()}_trades.csv",
                       mime="text/csv", key=f"dl_{strat}")

# ─── TABS ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "🔎 Screener — Chikou v2", "📊 Backtest — Chikou v2",
    "🔎 Screener — TK Cross",  "📈 Backtest — TK Cross",
])
with t1:
    render_screener("Chikou v2")
with t2:
    render_backtest("Chikou v2")
with t3:
    render_screener("TK Cross")
with t4:
    render_backtest("TK Cross")
