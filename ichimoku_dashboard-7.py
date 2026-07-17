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

# Kumo Breakout strategy (fixed-exit style, from the standalone script)
KUMO_SL_PCT        = 0.10   # 10% stop loss (both sides)
KUMO_TP_PCT        = 0.30   # 30% take profit (longs)
KUMO_TP_SHORT_PCT  = 0.10   # 10% take profit (fade shorts)
KUMO_HOLD_DAYS     = 14     # max holding period in trading days
BIG_RED_MULT       = 1.5    # "big red" = body >= 1.5x avg body of last 10 bars
BARS_PER_DAY       = {"15 min": 26, "1 Hour": 7, "4 Hour": 2, "1 Day": 1}

# Breakout -> Pullback -> Trigger strategy (state machine)
PULLBACK_MAX_AGE   = 15      # bars the armed breakout stays valid before expiry
TENKAN_TOL         = 0.005   # low may dip 0.5% below Tenkan and still "touch" it

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


def _arrays(df, htf_rule="4h"):
    """Common indicator arrays used by all strategies."""
    df = df.sort_index()
    ichi = ichimoku(df)
    A = {
        "df": df, "dates": df.index, "n": len(df),
        "o": df["Open"].values, "h": df["High"].values,
        "l": df["Low"].values,  "c": df["Close"].values,
        "tenkan": ichi["tenkan"].values, "kijun": ichi["kijun"].values,
        "cloud_top": ichi["cloud_top"].values,
        "cloud_bot": ichi["cloud_bot"].values,
        "adx": adx(df, ADX_PERIOD).values,
        "sma": df["Close"].rolling(SMA_PERIOD).mean().values,
        "htf_label": htf_rule,
    }
    # Higher-timeframe cloud position (lagged one HTF bar -> no lookahead)
    try:
        htf = df.resample(htf_rule, label="left", closed="left").agg({
            "Open": "first", "High": "max", "Low": "min", "Close": "last",
        }).dropna(how="any")
        hichi = ichimoku(htf)
        bull = (htf["Close"] > hichi["cloud_top"]).shift(1)
        bear = (htf["Close"] < hichi["cloud_bot"]).shift(1)
        A["htf_bull"] = (bull.reindex(df.index, method="ffill")
                             .fillna(False).values.astype(bool))
        A["htf_bear"] = (bear.reindex(df.index, method="ffill")
                             .fillna(False).values.astype(bool))
    except Exception:
        A["htf_bull"] = np.zeros(len(df), dtype=bool)
        A["htf_bear"] = np.zeros(len(df), dtype=bool)
    return A

# ─── STRATEGY 1: "Chikou v2" (inside/below cloud breakout) ───────────────────
def conds_v2(A, i, adx_min, clean_margin, direction, cross_lookback=None,
             flags=None, hold_bars=None):
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
def _cross_flags(A, direction, require_cloud_side=True):
    """Bool array of TK crosses. If require_cloud_side: gold cross must be
    below the kumo (long) / death cross above it (short). Otherwise any
    cross location counts."""
    t, k = A["tenkan"], A["kijun"]
    n = A["n"]
    out = np.zeros(n, dtype=bool)
    for j in range(1, n):
        if any(np.isnan(v) for v in [t[j], k[j], t[j-1], k[j-1]]):
            continue
        if direction == "long":
            crossed = t[j] > k[j] and t[j-1] <= k[j-1]
            if not require_cloud_side:
                out[j] = crossed
            else:
                cb = A["cloud_bot"][j]
                out[j] = crossed and not np.isnan(cb) and t[j] < cb
        else:
            crossed = t[j] < k[j] and t[j-1] >= k[j-1]
            if not require_cloud_side:
                out[j] = crossed
            else:
                ct = A["cloud_top"][j]
                out[j] = crossed and not np.isnan(ct) and t[j] > ct
    return out


def conds_v3(A, i, adx_min, clean_margin, direction, cross_lookback=15,
             flags=None, hold_bars=None):
    """Ordered (label, bool) list for the 7 TK-Cross conditions at bar i."""
    t, k = A["tenkan"][i], A["kijun"][i]
    tp = A["tenkan"][i-SLOPE_LOOKBACK]
    o, h, l, c = A["o"][i], A["h"][i], A["l"][i], A["c"][i]
    sma = A["sma"][i]
    ref = A["h"][i-CHIKOU_SHIFT] if direction == "long" else A["l"][i-CHIKOU_SHIFT]
    if any(np.isnan(v) for v in [t, k, tp, A["adx"][i], ref, sma]):
        return None
    if flags is None:
        flags = _cross_flags(A, direction, require_cloud_side=True)
    j0 = max(1, i - cross_lookback)
    recent_cross = bool(flags[j0:i+1].any())
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

# ─── STRATEGY 3: "Under Kumo + HTF" ──────────────────────────────────────────
# LONG : TK gold cross (anywhere, within lookback) + price UNDER the cloud
#        + Chikou breakout + higher-timeframe close ABOVE its cloud.
# SHORT: mirror — death cross + price ABOVE cloud + Chikou breakdown
#        + HTF close BELOW its cloud. Enter on the qualifying candle close.
def conds_htf(A, i, adx_min, clean_margin, direction, cross_lookback=15,
              flags=None, hold_bars=None):
    t, k = A["tenkan"][i], A["kijun"][i]
    o, h, l, c = A["o"][i], A["h"][i], A["l"][i], A["c"][i]
    ref = A["h"][i-CHIKOU_SHIFT] if direction == "long" else A["l"][i-CHIKOU_SHIFT]
    cb, ct = A["cloud_bot"][i], A["cloud_top"][i]
    if any(np.isnan(v) for v in [t, k, ref, cb, ct]):
        return None
    if flags is None:
        flags = _cross_flags(A, direction, require_cloud_side=False)
    j0 = max(1, i - cross_lookback)
    recent_cross = bool(flags[j0:i+1].any())
    htf = A["htf_label"]
    if direction == "long":
        return [
            (f"Gold cross ({cross_lookback} bars)", recent_cross),
            ("TK > KJ now", t > k),
            ("Price under kumo", c < cb),
            ("Chikou breakout", c > ref * (1 + clean_margin)),
            (f"HTF({htf}) above kumo", bool(A["htf_bull"][i])),
        ]
    return [
        (f"Death cross ({cross_lookback} bars)", recent_cross),
        ("TK < KJ now", t < k),
        ("Price above kumo", c > ct),
        ("Chikou breakdown", c < ref * (1 - clean_margin)),
        (f"HTF({htf}) below kumo", bool(A["htf_bear"][i])),
    ]


# ─── STRATEGY 4: "Kumo Break" (breakout long / engulfing-fade short) ────────
# LONG : candle CLOSES ACROSS the cloud top (prev close <= cloud, now above)
#        + Tenkan > Kijun + both lines sloping up. Fixed exits:
#        10% SL / 30% TP / 14-trading-day time stop (from the original script).
# SHORT: the FAILED-BREAKOUT FADE — if a bearish engulfing candle forms
#        within 1–3 candles AFTER a valid breakout, short the engulfing close.
#        Mirrored fixed exits (10% SL above, 30% TP below, time stop).
def _kumo_breakout_flags(A, direction=None):
    """Bars where the full long breakout signal fired (direction-agnostic:
    the short fade also keys off bullish breakouts)."""
    n = A["n"]
    out = np.zeros(n, dtype=bool)
    c, ct = A["c"], A["cloud_top"]
    t, k = A["tenkan"], A["kijun"]
    for i in range(SLOPE_LOOKBACK + 1, n):
        vals = [c[i], c[i-1], ct[i], ct[i-1], t[i], k[i],
                t[i-SLOPE_LOOKBACK], k[i-SLOPE_LOOKBACK]]
        if any(np.isnan(v) for v in vals):
            continue
        out[i] = (c[i-1] <= ct[i-1] and c[i] > ct[i]        # closed across kumo
                  and t[i] > k[i]                            # TK bullish
                  and t[i] > t[i-SLOPE_LOOKBACK]             # tenkan rising
                  and k[i] > k[i-SLOPE_LOOKBACK])            # kijun rising
    return out


def _is_reversal(o, c, i, mult=None):
    """Bearish reversal candle: body-engulfing red candle OR a 'big red'
    candle whose body >= BIG_RED_MULT x the average body of the prior 10."""
    if mult is None:
        mult = BIG_RED_MULT
    if i < 11 or any(np.isnan(v) for v in [o[i], c[i], o[i-1], c[i-1]]):
        return False
    if c[i] >= o[i]:                      # must be red
        return False
    # engulfing
    prev_top, prev_bot = max(o[i-1], c[i-1]), min(o[i-1], c[i-1])
    if o[i] >= prev_top and c[i] <= prev_bot:
        return True
    # big red
    bodies = np.abs(c[i-10:i] - o[i-10:i])
    avg = np.nanmean(bodies)
    return bool(avg > 0 and (o[i] - c[i]) >= mult * avg)


def _fade_entry_flags(A, breakout_flags, hold_bars):
    """Short-entry bars: the FIRST reversal candle after each breakout,
    within the long's holding window (the long exits there and flips)."""
    n = A["n"]
    o, c = A["o"], A["c"]
    fade = np.zeros(n, dtype=bool)
    for j in np.where(breakout_flags)[0]:
        end = min(j + hold_bars, n - 1)
        for m in range(j + 1, end + 1):
            if _is_reversal(o, c, m):
                fade[m] = True
                break
    return fade


def conds_kumo(A, i, adx_min, clean_margin, direction, cross_lookback=15,
               flags=None, hold_bars=14):
    if flags is None or not isinstance(flags, dict):
        flags = {"breakout": _kumo_breakout_flags(A)}
    bo = flags["breakout"]
    if direction == "long":
        c, ct = A["c"], A["cloud_top"]
        t, k = A["tenkan"][i], A["kijun"][i]
        vals = [c[i], c[i-1], ct[i], ct[i-1], t, k,
                A["tenkan"][i-SLOPE_LOOKBACK], A["kijun"][i-SLOPE_LOOKBACK]]
        if any(np.isnan(v) for v in vals):
            return None
        return [
            ("Closed across kumo", bool(c[i-1] <= ct[i-1] and c[i] > ct[i])),
            ("TK > KJ", bool(t > k)),
            ("Tenkan rising", bool(t > A["tenkan"][i-SLOPE_LOOKBACK])),
            ("Kijun rising", bool(k > A["kijun"][i-SLOPE_LOOKBACK])),
        ]
    # short: flip on the first reversal candle during the long's life
    if "fade" not in flags:
        flags["fade"] = _fade_entry_flags(A, bo, hold_bars)
    recent_breakout = bool(bo[max(0, i-hold_bars):i].any())
    return [
        (f"Kumo breakout ≤{hold_bars} bars ago", recent_breakout),
        ("Reversal candle (engulf/big red)", bool(flags["fade"][i])),
    ]


def stop_kumo(A, i, direction):
    entry = A["c"][i]
    return entry * (1 - KUMO_SL_PCT) if direction == "long" \
        else entry * (1 + KUMO_SL_PCT)


# ─── STRATEGY 5: "Shomega" (video method — TK cross heads-up + Chikou break) ──
# The exact sequencing from the pasted lesson ("shomega / 潮目 — the tide turns"):
#   LONG : identify a prior DOWNTREND -> a Tenkan/Kijun GOLD cross that happened
#          BELOW the kumo (early heads-up, do NOT enter yet) -> WAIT -> enter on
#          the candle where Chikou CLEANLY breaks ABOVE the past price
#          (close > High[26]). Stop below Kijun or the swing low (wider = safer).
#   SHORT: prior UPTREND -> TK DEATH cross ABOVE the kumo -> wait -> enter when
#          Chikou breaks BELOW the past price (close < Low[26]). Mirror stop.
# Exits (shared scale engine): 50% at 1:1, runner rides the Kijun trail
# ("ride the trend as long as Kijun is respected").
def conds_shomega(A, i, adx_min, clean_margin, direction, cross_lookback=15,
                  flags=None, hold_bars=None):
    t, k = A["tenkan"][i], A["kijun"][i]
    o, h, l, c = A["o"][i], A["h"][i], A["l"][i], A["c"][i]
    cb, ct = A["cloud_bot"][i], A["cloud_top"][i]
    ref = A["h"][i-CHIKOU_SHIFT] if direction == "long" else A["l"][i-CHIKOU_SHIFT]
    if any(np.isnan(v) for v in [t, k, cb, ct, ref, A["adx"][i]]):
        return None
    if flags is None:                        # gold-below / death-above the kumo
        flags = _cross_flags(A, direction, require_cloud_side=True)
    j0 = max(1, i - cross_lookback)
    recent_cross = bool(flags[j0:i+1].any())
    rng = h - l
    if direction == "long":
        # "identify a downtrend" -> at the cross the past-price ref (High[26])
        # was still capping price; now Chikou clears it = the tide turns.
        return [
            (f"TK gold cross <kumo ({cross_lookback}b)", recent_cross),
            ("Chikou > price 26 back", c > ref),
            ("Clean breakout", c > ref*(1+clean_margin) and c > o
                               and rng > 0 and c >= (h+l)/2),
            ("Not extended (≤ cloud top or just above)", c <= ct*(1+clean_margin)
                                                          or c < ct),
            (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
        ]
    return [
        (f"TK death cross >kumo ({cross_lookback}b)", recent_cross),
        ("Chikou < price 26 back", c < ref),
        ("Clean breakdown", c < ref*(1-clean_margin) and c < o
                            and rng > 0 and c <= (h+l)/2),
        ("Not extended (≥ cloud bot or just below)", c >= cb*(1-clean_margin)
                                                      or c > cb),
        (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
    ]


# ─── STRATEGY 6: "Twin Line" (Tenkan>Kijun and BOTH sloping up) ──────────────
# Bare-bones trend filter you asked for: enter LONG when the blue line
# (Tenkan) is above the red line (Kijun) AND both are sloping up over the
# slope lookback. SHORT mirrors (Tenkan<Kijun, both sloping down). ADX gate
# kept for quality. Same scale-out exits as the other trend strategies.
def conds_twin(A, i, adx_min, clean_margin, direction, cross_lookback=None,
               flags=None, hold_bars=None):
    t, k = A["tenkan"][i], A["kijun"][i]
    tp, kp = A["tenkan"][i-SLOPE_LOOKBACK], A["kijun"][i-SLOPE_LOOKBACK]
    if any(np.isnan(v) for v in [t, k, tp, kp, A["adx"][i]]):
        return None
    if direction == "long":
        return [
            ("Tenkan > Kijun", t > k),
            ("Tenkan sloping up", t > tp),
            ("Kijun sloping up", k > kp),
            (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
        ]
    return [
        ("Tenkan < Kijun", t < k),
        ("Tenkan sloping down", t < tp),
        ("Kijun sloping down", k < kp),
        (f"ADX ≥ {adx_min:g}", A["adx"][i] >= adx_min),
    ]


# ─── STRATEGY 7: "Break-Pull" (Breakout -> Pullback -> Trigger) ──────────────
# A three-phase STATE MACHINE (multi-bar), unlike the single-bar strategies.
# It does not fit the per-bar conds() shape, so it has its own signal finder
# (find_signals_breakpull) plugged into the same portfolio/exit engine.
#
#   Phase 1 — Breakout / arm:
#       close crosses ABOVE cloud top (prev close <= top, now above)
#       + Chikou confirm (close > close[26])
#       + regime gate (close > SMA50)
#       -> armed, record breakout_bar, expiry = i + PULLBACK_MAX_AGE
#   Phase 2 — Pullback:
#       while armed & i <= expiry:
#         * if close < cloud_top  -> DISARM (breakout failed / regime broken)
#         * pullback qualifies when low <= Tenkan*(1+TENKAN_TOL)
#           AND cloud still holds (low > cloud_top) -> mark pullback_seen,
#           record pullback_high & pullback_low
#   Phase 3 — Trigger / entry:
#       after pullback_seen: enter when close > pullback_high AND close > Tenkan
#       -> fill NEXT bar open (no look-ahead). Stop = pullback_low - buffer.
#       Exit = Kijun trail (close < Kijun), invalidation exit if close < cloud_top.
# SHORT is the mirror (breakdown below cloud bottom, rally to Tenkan, etc.).
def find_signals_breakpull(strat, ticker, df, adx_min, clean_margin, direction,
                           cross_lookback=15, htf_rule="4h", hold_bars=14):
    if df is None or len(df) < MIN_BARS:
        return []
    A = _arrays(df, htf_rule=htf_rule)
    c, o, h, l = A["c"], A["o"], A["h"], A["l"]
    ct, cb = A["cloud_top"], A["cloud_bot"]
    ten, kij, sma = A["tenkan"], A["kijun"], A["sma"]
    n = A["n"]
    is_long = direction == "long"
    start = 52 + CHIKOU_SHIFT + max(SLOPE_LOOKBACK, SMA_PERIOD, ADX_PERIOD) + 2

    events = []
    armed = False
    expiry = -1
    pullback_seen = False
    pb_high = pb_low = np.nan

    for i in range(start, n):
        cloud_edge = ct[i] if is_long else cb[i]
        cloud_edge_p = ct[i-1] if is_long else cb[i-1]
        chikou_ref = c[i-CHIKOU_SHIFT]
        if any(np.isnan(v) for v in [cloud_edge, cloud_edge_p, ten[i],
                                     kij[i], sma[i], chikou_ref]):
            continue

        # ── Phase 1: arm on a fresh cloud breakout ──
        if not armed:
            if is_long:
                broke = c[i-1] <= cloud_edge_p and c[i] > cloud_edge
                chikou_ok = c[i] > chikou_ref          # close > close[26]
                regime_ok = c[i] > sma[i]
            else:
                broke = c[i-1] >= cloud_edge_p and c[i] < cloud_edge
                chikou_ok = c[i] < chikou_ref
                regime_ok = c[i] < sma[i]
            if broke and chikou_ok and regime_ok:
                armed = True
                expiry = i + PULLBACK_MAX_AGE
                pullback_seen = False
                pb_high = pb_low = np.nan
            continue

        # ── armed: expiry / invalidation ──
        if i > expiry:
            armed = False
            continue
        # regime broken -> disarm (close back through the cloud edge)
        if (is_long and c[i] < ct[i]) or (not is_long and c[i] > cb[i]):
            armed = False
            continue

        # ── Phase 2: look for the pullback to Tenkan (cloud must still hold) ──
        if not pullback_seen:
            if is_long:
                touched = l[i] <= ten[i] * (1 + TENKAN_TOL)
                holds = l[i] > ct[i]
            else:
                touched = h[i] >= ten[i] * (1 - TENKAN_TOL)
                holds = h[i] < cb[i]
            if touched and holds:
                pullback_seen = True
                pb_high, pb_low = h[i], l[i]
            continue

        # ── Phase 3: trigger — reclaim of the pullback extreme + beyond Tenkan ──
        if is_long:
            trigger = c[i] > pb_high and c[i] > ten[i]
        else:
            trigger = c[i] < pb_low and c[i] < ten[i]
        if not trigger:
            # keep extending the pullback extreme while we wait
            if is_long:
                pb_high = max(pb_high, h[i]); pb_low = min(pb_low, l[i])
            else:
                pb_low = min(pb_low, l[i]);   pb_high = max(pb_high, h[i])
            continue

        # fire — fill at NEXT bar open (no look-ahead)
        if i + 1 >= n:
            armed = False
            continue
        entry = o[i+1]
        stop = (pb_low * (1 - STOP_BUFFER) if is_long
                else pb_high * (1 + STOP_BUFFER))
        risk = (entry - stop) if is_long else (stop - entry)
        if risk > 0:
            tp1 = (entry + PARTIAL_RR*risk if is_long
                   else entry - PARTIAL_RR*risk)
            events.append({
                "ticker": ticker, "direction": direction, "strategy": strat,
                "entry_idx": i + 1, "entry_date": A["dates"][i+1],
                "entry_price": entry, "init_stop": stop, "tp1": tp1,
                "risk_per_share": risk, "exit_style": "scale",
                "max_bars": hold_bars,
                "_close": c, "_high": h, "_low": l, "_open": o,
                "_kijun": kij, "_dates": A["dates"], "_n": n,
            })
        armed = False        # reset; look for the next breakout

    return events


def latest_breakpull_state(df, adx_min, clean_margin, direction, htf_rule="4h"):
    """For the screener: report the current phase (armed / pullback / trigger)
    on the most recent bars, plus whether a trigger just fired on the last bar."""
    if df is None or len(df) < MIN_BARS:
        return None
    A = _arrays(df, htf_rule=htf_rule)
    c, h, l = A["c"], A["h"], A["l"]
    ct, cb = A["cloud_top"], A["cloud_bot"]
    ten, kij, sma = A["tenkan"], A["kijun"], A["sma"]
    n = A["n"]
    is_long = direction == "long"
    start = 52 + CHIKOU_SHIFT + max(SLOPE_LOOKBACK, SMA_PERIOD, ADX_PERIOD) + 2
    if n - 1 < start:
        return None

    armed = False; expiry = -1; pullback_seen = False
    pb_high = pb_low = np.nan; fired_last = False
    for i in range(start, n):
        cloud_edge, cloud_edge_p = (ct[i], ct[i-1]) if is_long else (cb[i], cb[i-1])
        chikou_ref = c[i-CHIKOU_SHIFT]
        if any(np.isnan(v) for v in [cloud_edge, cloud_edge_p, ten[i], kij[i],
                                     sma[i], chikou_ref]):
            continue
        if not armed:
            if is_long:
                broke = c[i-1] <= cloud_edge_p and c[i] > cloud_edge
                ok = c[i] > chikou_ref and c[i] > sma[i]
            else:
                broke = c[i-1] >= cloud_edge_p and c[i] < cloud_edge
                ok = c[i] < chikou_ref and c[i] < sma[i]
            if broke and ok:
                armed = True; expiry = i + PULLBACK_MAX_AGE
                pullback_seen = False; pb_high = pb_low = np.nan
            continue
        if i > expiry:
            armed = False; continue
        if (is_long and c[i] < ct[i]) or (not is_long and c[i] > cb[i]):
            armed = False; continue
        if not pullback_seen:
            if is_long:
                touched, holds = l[i] <= ten[i]*(1+TENKAN_TOL), l[i] > ct[i]
            else:
                touched, holds = h[i] >= ten[i]*(1-TENKAN_TOL), h[i] < cb[i]
            if touched and holds:
                pullback_seen = True; pb_high, pb_low = h[i], l[i]
            continue
        trig = (c[i] > pb_high and c[i] > ten[i]) if is_long \
            else (c[i] < pb_low and c[i] < ten[i])
        if trig:
            fired_last = (i == n - 1)
            armed = False
        else:
            if is_long:
                pb_high = max(pb_high, h[i]); pb_low = min(pb_low, l[i])
            else:
                pb_low = min(pb_low, l[i]); pb_high = max(pb_high, h[i])

    i = n - 1
    if armed and not pullback_seen:
        phase = "🟡 Armed — waiting for pullback to Tenkan"
    elif armed and pullback_seen:
        phase = "🟠 Pullback seen — waiting for trigger"
    elif fired_last:
        phase = "🟢 TRIGGER fired on last bar"
    else:
        phase = "⚪ Idle — no active setup"
    bars_left = (expiry - i) if armed else 0
    return {
        "phase": phase, "fired": fired_last, "armed": armed,
        "pullback_seen": pullback_seen, "bars_left": max(0, bars_left),
        "close": float(c[i]), "tenkan": float(ten[i]) if not np.isnan(ten[i]) else np.nan,
        "adx": float(A["adx"][i]) if not np.isnan(A["adx"][i]) else np.nan,
        "pb_high": None if np.isnan(pb_high) else float(pb_high),
        "pb_low": None if np.isnan(pb_low) else float(pb_low),
    }


STRATEGIES = {
    "Chikou v2":  {"conds": conds_v2,  "stop": stop_v2,   "flags": None},
    "TK Cross":   {"conds": conds_v3,  "stop": stop_v3,
                   "flags": lambda A, d: _cross_flags(A, d, True)},
    "Under Kumo": {"conds": conds_htf, "stop": stop_v2,
                   "flags": lambda A, d: _cross_flags(A, d, False)},
    "Kumo Break": {"conds": conds_kumo, "stop": stop_kumo,
                   "flags": lambda A, d: {"breakout": _kumo_breakout_flags(A)},
                   "exit_style": "pct"},
    "Shomega":    {"conds": conds_shomega, "stop": stop_v3,
                   "flags": lambda A, d: _cross_flags(A, d, True)},
    "Twin Line":  {"conds": conds_twin, "stop": stop_v2, "flags": None},
    "Break-Pull": {"conds": None, "stop": stop_v2, "flags": None,
                   "custom_finder": True},   # uses find_signals_breakpull
}

# ─── SIGNAL SCAN (shared) ─────────────────────────────────────────────────────
def find_signals(strat, ticker, df, adx_min, clean_margin, direction,
                 cross_lookback=15, htf_rule="4h", hold_bars=14):
    if df is None or len(df) < MIN_BARS:
        return []
    S = STRATEGIES[strat]
    if S.get("custom_finder"):               # Break-Pull state machine
        return find_signals_breakpull(strat, ticker, df, adx_min, clean_margin,
                                      direction, cross_lookback=cross_lookback,
                                      htf_rule=htf_rule, hold_bars=hold_bars)
    A = _arrays(df, htf_rule=htf_rule)
    flags = S["flags"](A, direction) if S["flags"] is not None else None
    events = []
    start = 52 + CHIKOU_SHIFT + max(SLOPE_LOOKBACK, SWING_LOOKBACK,
                                    ADX_PERIOD, SMA_PERIOD) + 2
    for i in range(start, A["n"]):
        cl = S["conds"](A, i, adx_min, clean_margin, direction,
                        cross_lookback=cross_lookback, flags=flags,
                        hold_bars=hold_bars)
        if cl is None or not all(v for _, v in cl):
            continue
        entry = A["c"][i]
        if S.get("exit_style") == "pct":
            stop = stop_kumo(A, i, direction)
            tp1 = (entry * (1 + KUMO_TP_PCT) if direction == "long"
                   else entry * (1 - KUMO_TP_SHORT_PCT))
            risk = abs(entry - stop)
        else:
            stop = S["stop"](A, i, direction)
            risk = (entry - stop) if direction == "long" else (stop - entry)
            if risk <= 0:
                continue
            tp1 = (entry + PARTIAL_RR*risk if direction == "long"
                   else entry - PARTIAL_RR*risk)
        events.append({
            "ticker": ticker, "direction": direction, "strategy": strat,
            "entry_idx": i, "entry_date": A["dates"][i],
            "entry_price": entry, "init_stop": stop, "tp1": tp1,
            "risk_per_share": risk,
            "exit_style": S.get("exit_style", "scale"),
            "max_bars": hold_bars,
            "_close": A["c"], "_high": A["h"], "_low": A["l"],
            "_open": A["o"],
            "_kijun": A["kijun"], "_dates": A["dates"], "_n": A["n"],
        })
    return events


def latest_conditions(strat, df, adx_min, clean_margin, direction,
                      cross_lookback=15, htf_rule="4h", hold_bars=14):
    """Condition list on the last closed bar, for the screener."""
    if df is None or len(df) < MIN_BARS:
        return None
    A = _arrays(df, htf_rule=htf_rule)
    i = A["n"] - 1
    if i < 52 + CHIKOU_SHIFT + SMA_PERIOD:
        return None
    S = STRATEGIES[strat]
    flags = S["flags"](A, direction) if S["flags"] is not None else None
    cl = S["conds"](A, i, adx_min, clean_margin, direction,
                    cross_lookback=cross_lookback, flags=flags,
                    hold_bars=hold_bars)
    if cl is None:
        return None
    return cl, float(A["c"][i]), float(A["adx"][i])

# ─── TRADE SIM + PORTFOLIO (shared, direction-aware) ─────────────────────────
def simulate_pct(e, shares):
    """Fixed-exit style (Kumo Break): intrabar 10% SL / TP, else time exit.
    LONGS additionally exit at the close of a reversal candle (bearish
    engulfing or big red) — the bar where the fade short takes over."""
    high, low, close, dates = e["_high"], e["_low"], e["_close"], e["_dates"]
    opn = e.get("_open")
    n, i0 = e["_n"], e["entry_idx"]
    sl, tp = e["init_stop"], e["tp1"]
    is_long = e["direction"] == "long"
    end = min(i0 + e["max_bars"], n - 1)
    for t in range(i0 + 1, end + 1):
        if is_long:
            if low[t] <= sl:
                return {"exit_date": dates[t], "legs": [(sl, dates[t], "stop_loss", shares)]}
            if high[t] >= tp:
                return {"exit_date": dates[t], "legs": [(tp, dates[t], "take_profit", shares)]}
            if opn is not None and _is_reversal(opn, close, t):
                return {"exit_date": dates[t],
                        "legs": [(close[t], dates[t], "reversal_flip", shares)]}
        else:
            if high[t] >= sl:
                return {"exit_date": dates[t], "legs": [(sl, dates[t], "stop_loss", shares)]}
            if low[t] <= tp:
                return {"exit_date": dates[t], "legs": [(tp, dates[t], "take_profit", shares)]}
    t_end = max(i0 + 1, end)
    return {"exit_date": dates[t_end],
            "legs": [(close[t_end], dates[t_end], "time_exit", shares)]}


def simulate_trade(e, shares):
    if e.get("exit_style") == "pct":
        return simulate_pct(e, shares)
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
st.sidebar.caption("Chikou v2 · TK Cross · Under Kumo · Kumo Break · "
                   "Shomega · Twin Line · Break-Pull")

tf_label = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=1)
interval, max_days, resample_rule = TIMEFRAMES[tf_label]

# higher timeframe used by the "Under Kumo" strategy (resampled, 1-bar lag)
HTF_MAP = {"15 min": "1h", "1 Hour": "4h", "4 Hour": "1D", "1 Day": "1W"}
HTF_RULE = HTF_MAP[tf_label]
KUMO_HOLD_BARS = KUMO_HOLD_DAYS * BARS_PER_DAY[tf_label]
st.sidebar.caption(f"Under Kumo HTF confirm: **{HTF_RULE}** cloud · "
                   f"Kumo Break max hold: **{KUMO_HOLD_BARS}** bars")

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
def render_breakpull_screener():
    st.subheader(f"Break-Pull — live phase tracker · {tf_label} · {dir_choice.lower()}")
    st.caption("🟡 **Armed** (cloud breakout + Chikou + SMA50) → 🟠 **Pullback** "
               "to Tenkan (cloud must hold) → 🟢 **Trigger** on reclaim of the "
               "pullback high above Tenkan. Fill = next bar open · stop = pullback "
               "low − buffer · exit = Kijun trail / close back inside cloud.")
    fired, active, idle_ct = [], [], 0
    for tk, df in price_map.items():
        for d in DIRECTIONS:
            try:
                s = latest_breakpull_state(df, adx_min, clean_margin, d,
                                           htf_rule=HTF_RULE)
            except Exception:
                continue
            if s is None:
                continue
            row = {"Ticker": tk, "Dir": "🟢 LONG" if d == "long" else "🔴 SHORT",
                   "Cap $B": round(cap_lookup.get(tk, 0), 1),
                   "Phase": s["phase"], "Close": round(s["close"], 2),
                   "Tenkan": None if np.isnan(s["tenkan"]) else round(s["tenkan"], 2),
                   "PB high": s["pb_high"] and round(s["pb_high"], 2),
                   "PB low": s["pb_low"] and round(s["pb_low"], 2),
                   "Bars left": s["bars_left"] if s["armed"] else "—",
                   "ADX": None if np.isnan(s["adx"]) else round(s["adx"], 1)}
            if s["fired"]:
                fired.append(row)
            elif s["armed"]:
                active.append(row)
            else:
                idle_ct += 1

    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Triggers on last bar", len(fired))
    c2.metric("🟡🟠 Active setups (armed)", len(active))
    c3.metric("⚪ Idle names", idle_ct)
    if fired:
        st.success("Entry TRIGGER fired on the last closed bar — fill at next open:")
        st.dataframe(pd.DataFrame(fired), use_container_width=True, hide_index=True)
    else:
        st.info("No Break-Pull trigger on the latest bar right now.")
    with st.expander(f"Armed setups — watchlist ({len(active)})"):
        if active:
            order = {"🟠 Pullback seen — waiting for trigger": 0,
                     "🟡 Armed — waiting for pullback to Tenkan": 1}
            adf = pd.DataFrame(active)
            adf["_o"] = adf["Phase"].map(lambda p: order.get(p, 2))
            st.dataframe(adf.sort_values("_o").drop(columns="_o"),
                         use_container_width=True, hide_index=True)
        else:
            st.write("Nothing armed right now.")


def render_screener(strat):
    if strat == "Break-Pull":
        render_breakpull_screener()
        return
    st.subheader(f"{strat} — latest {tf_label} bar · {dir_choice.lower()}")
    if strat == "Kumo Break":
        st.caption("🟢 LONG = candle closes across the kumo (TK bullish, both lines "
                   "rising); exits: 10% SL / 30% TP / time stop / **flips on a "
                   "reversal candle** · 🔴 SHORT = the flip — first bearish "
                   "engulfing or big red candle during the long → short at its "
                   f"close, {KUMO_TP_SHORT_PCT*100:.0f}% profit target, "
                   f"{KUMO_SL_PCT*100:.0f}% SL, time stop")
    if strat == "Shomega":
        st.caption("潮目 — 'the tide turns'. 🟢 LONG = a TK **gold cross below "
                   "the kumo** (heads-up) then a clean **Chikou breakout above "
                   "price 26 bars back**, not over-extended past the cloud top. "
                   "🔴 SHORT mirrors. Stop beyond Kijun/swing; 50% off at 1:1, "
                   "runner on the Kijun trail.")
    if strat == "Twin Line":
        st.caption("🟢 LONG = **Tenkan (blue) > Kijun (red)** and **both lines "
                   "sloping up** over the slope lookback (+ ADX gate). 🔴 SHORT = "
                   "Tenkan < Kijun and both sloping down. Stop at the swing; "
                   "50% off at 1:1, runner on the Kijun trail.")
    rows, near_rows = [], []
    total_conds = 7
    for tk, df in price_map.items():
        for d in DIRECTIONS:
            try:
                res = latest_conditions(strat, df, adx_min, clean_margin, d,
                                        cross_lookback=cross_lookback,
                                        htf_rule=HTF_RULE,
                                        hold_bars=KUMO_HOLD_BARS)
            except Exception:
                continue
            if res is None:
                continue
            cl, close_px, adx_val = res
            total_conds = len(cl)
            passed = sum(v for _, v in cl)
            row = {"Ticker": tk, "Dir": "🟢 LONG" if d == "long" else "🔴 SHORT",
                   "Cap $B": round(cap_lookup.get(tk, 0), 1),
                   "Close": round(close_px, 2), "ADX": round(adx_val, 1),
                   **{lbl: ("✅" if v else "—") for lbl, v in cl},
                   "Passed": f"{passed}/{len(cl)}"}
            total = len(cl)
            near_min = total - 2 if total >= 5 else total - 1
            if passed == total:
                rows.append(row)
            elif passed >= near_min:
                near_rows.append(row)

    c1, c2 = st.columns(2)
    c1.metric("🔥 Full signals now", len(rows))
    c2.metric("👀 Near-setups (1–2 conditions missing)", len(near_rows))
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
                                                   cross_lookback=cross_lookback,
                                                   htf_rule=HTF_RULE,
                                                   hold_bars=KUMO_HOLD_BARS))
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
tabs = st.tabs([
    "🔎 Chikou v2", "📊 Chikou v2 BT",
    "🔎 TK Cross",  "📈 TK Cross BT",
    "🔎 Under Kumo", "☁️ Under Kumo BT",
    "🔎 Kumo Break", "🚀 Kumo Break BT",
    "🌊 Shomega",    "🌊 Shomega BT",
    "➰ Twin Line",  "➰ Twin Line BT",
    "🎯 Break-Pull", "🎯 Break-Pull BT",
])
pairs = [
    ("Chikou v2", "screener"), ("Chikou v2", "backtest"),
    ("TK Cross", "screener"),  ("TK Cross", "backtest"),
    ("Under Kumo", "screener"),("Under Kumo", "backtest"),
    ("Kumo Break", "screener"),("Kumo Break", "backtest"),
    ("Shomega", "screener"),   ("Shomega", "backtest"),
    ("Twin Line", "screener"), ("Twin Line", "backtest"),
    ("Break-Pull", "screener"),("Break-Pull", "backtest"),
]
for tab, (strat, kind) in zip(tabs, pairs):
    with tab:
        if kind == "screener":
            render_screener(strat)
        else:
            render_backtest(strat)
