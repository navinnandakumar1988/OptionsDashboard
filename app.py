# app.py — Options Dashboard (Excel Parity), fixed day-first parsing

from pathlib import Path
from datetime import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------- PAGE ----------
st.set_page_config(page_title="Options Dashboard (Excel Parity)", layout="wide")

# ---------- OPTIONAL AUTO-REFRESH ----------
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=60_000, key="optidash_refresh_60s")
except Exception:
    pass

# ---------- FILES ----------
from pathlib import Path
import streamlit as st
import pandas as pd

def _csv_source(sym: str):
    """
    1) If Streamlit Secrets provides a URL (cloud), use it
    2) Else if local /files exists (your n8n writes here), use it
    3) Else fallback to repo ./data/ (tiny samples)
    """
    url = st.secrets.get("data_urls", {}).get(sym)
    if url:
        return url  # pandas can read URLs directly

    local = Path(f"/files/{sym}_reference_data_V9.csv")
    if local.exists():
        return local

    return Path(f"data/{sym}_reference_data_V9.csv")

FILES = {
    "NIFTY": _csv_source("NIFTY"),
    "BANKNIFTY": _csv_source("BANKNIFTY"),
}

CE_COLOR = "#1f77b4"
PE_COLOR = "#ff7f0e"
LABEL_ORDER = ["ITM3", "ITM2", "ITM1", "ATM", "OTM1", "OTM2", "OTM3"]

def apply_market_rangebreaks(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(pattern="hour", bounds=[15.5, 9.25]),
        ]
    )
    return fig

def apply_ce_pe_colors(fig: go.Figure) -> go.Figure:
    for tr in fig.data:
        nm = getattr(tr, "name", "")
        if nm == "CE":
            if hasattr(tr, "line"):   tr.line.color   = CE_COLOR
            if hasattr(tr, "marker"): tr.marker.color = CE_COLOR
        elif nm == "PE":
            if hasattr(tr, "line"):   tr.line.color   = PE_COLOR
            if hasattr(tr, "marker"): tr.marker.color = PE_COLOR
    return fig

def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data(ttl=60)
def load_csv(symbol: str) -> pd.DataFrame:
    src = FILES[symbol]
    return pd.read_csv(str(src), low_memory=False)

    # ---------- Robust DateTime build ----------
    # Prefer explicit Date+Time; else DateTime; else best-guess date/time columns
    if {"Date", "Time"}.issubset(df.columns):
        s = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
    elif "DateTime" in df.columns:
        s = df["DateTime"].astype(str).str.strip()
    else:
        # Try common alternates (e.g., O=Date, A=Time in some exports)
        date_col = first_present(df, ["Date", "TradeDate", "DATE", "O"])
        time_col = first_present(df, ["Time", "TradeTime", "TIME", "A"])
        if date_col and time_col:
            s = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        else:
            # Last resort: use first column as is
            s = df.iloc[:, 0].astype(str).str.strip()

    # ---------- Parse day-first FIRST (your files are dd-mm-yyyy hh:mm) ----------
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    if dt.isna().all():
        # If the entire column failed (unlikely), try generic inference as a fallback
        dt = pd.to_datetime(s, errors="coerce")

    if dt.isna().all():
        ex = s.head(3).tolist()
        raise ValueError(f"Could not parse DateTime. Examples: {ex}")

    df["DateTime"] = dt
    df = df.dropna(subset=["DateTime"]).copy()

    df["Minute"]   = df["DateTime"].dt.floor("min")
    df["Date"]     = df["DateTime"].dt.date
    df["TOD"]      = df["DateTime"].dt.time

    if "OptionType" in df.columns:
        df["OptionType"] = df["OptionType"].astype(str).str.upper().str.strip()
    if "OptionLabel" in df.columns:
        df["OptionLabel"] = df["OptionLabel"].astype(str).str.upper().str.replace(" ", "", regex=False)

    # Row-level helpers
    if {"ChangeInOI", "OptionVolume"}.issubset(df.columns):
        vol = df["OptionVolume"].replace(0, np.nan)
        df["RowRatio"] = (df["ChangeInOI"] * 75.0) / vol
        df["RowRatio"] = df["RowRatio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["COI75"]     = df["ChangeInOI"] * 75.0

    return df

# ---------- SIDEBAR ----------
st.sidebar.header("Filters")
symbol = st.sidebar.radio("Instrument", ["NIFTY", "BANKNIFTY"], index=0)

df = load_csv(symbol)

with st.sidebar.expander("Available dates in the file:", expanded=False):
    dates_list = pd.to_datetime(df["Date"]).sort_values().unique()
    st.write(", ".join(pd.Series(dates_list).dt.strftime("%Y-%m-%d").tolist()))

min_dt, max_dt = df["DateTime"].min(), df["DateTime"].max()
start, end = st.sidebar.date_input("Date range", value=(min_dt.date(), max_dt.date()))
if isinstance(start, (list, tuple)):
    start, end = start
mask = (df["DateTime"] >= pd.to_datetime(start)) & (df["DateTime"] <= pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1))
df = df.loc[mask].copy()

st.sidebar.markdown("**Intraday time filter (IST)**")
t_start, t_end = st.sidebar.slider("Time window", value=(time(9, 15), time(15, 30)), format="HH:mm")
df = df[(df["TOD"] >= t_start) & (df["TOD"] <= t_end)].copy()

if "OptionType" in df.columns:
    types = [t for t in ["CE", "PE"] if t in df["OptionType"].unique().tolist()]
    sel_types = st.sidebar.multiselect("Option Type", options=types, default=types)
    if sel_types:
        df = df[df["OptionType"].isin(sel_types)]

if "OptionLabel" in df.columns:
    labs = [x for x in LABEL_ORDER if x in df["OptionLabel"].unique().tolist()]
    sel_labs = st.sidebar.multiselect("OptionLabel (strike bucket)", options=labs, default=labs)
    if sel_labs:
        df = df[df["OptionLabel"].isin(sel_labs)]

# ---------- KPI STRIP ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Σ Call Volume", f"{df.loc[df.get('OptionType','')=='CE','OptionVolume'].sum():,.0f}" if "OptionVolume" in df.columns else "—")
c2.metric("Σ Put Volume",  f"{df.loc[df.get('OptionType','')=='PE','OptionVolume'].sum():,.0f}" if "OptionVolume" in df.columns else "—")
c3.metric("Σ CE COI",      f"{df.loc[df.get('OptionType','')=='CE','ChangeInOI'].sum():,.0f}" if "ChangeInOI" in df.columns else "—")
c4.metric("Σ PE COI",      f"{df.loc[df.get('OptionType','')=='PE','ChangeInOI'].sum():,.0f}" if "ChangeInOI" in df.columns else "—")

# ---------- CHART 1: ROC_IV ----------
if {"ROC_IV", "OptionType"}.issubset(df.columns):
    g = df.groupby(["Minute", "OptionType"], as_index=False, sort=False)["ROC_IV"].sum()
    fig = px.line(g, x="Minute", y="ROC_IV", color="OptionType",
                  title="Sum of ROC_IV (CE vs PE) over Time")
    fig = apply_ce_pe_colors(fig); fig = apply_market_rangebreaks(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------- CHART 2: ROC_PremiumDecay ----------
if {"ROC_PremiumDecay", "OptionType"}.issubset(df.columns):
    g = df.groupby(["Minute", "OptionType"], as_index=False, sort=False)["ROC_PremiumDecay"].sum()
    fig = px.line(g, x="Minute", y="ROC_PremiumDecay", color="OptionType",
                  title="Sum of ROC_PremiumDecay (CE vs PE) over Time")
    fig = apply_ce_pe_colors(fig); fig = apply_market_rangebreaks(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------- CHART 3: COI per Volume (cumulative intraday) ----------
if {"RowRatio", "OptionType"}.issubset(df.columns):
    minute_sum = df.groupby(["Minute", "OptionType"], as_index=False, sort=False)["RowRatio"].sum()
    minute_sum = minute_sum.sort_values(["OptionType", "Minute"]).reset_index(drop=True)
    minute_sum["Date"] = minute_sum["Minute"].dt.date
    minute_sum["COI_per_Volume"] = minute_sum.groupby(["OptionType", "Date"])["RowRatio"].cumsum()
    fig = px.line(minute_sum, x="Minute", y="COI_per_Volume", color="OptionType",
                  title="COI per Volume — Cumulative intraday (Σ per-row ratios) — CE vs PE")
    fig = apply_ce_pe_colors(fig); fig = apply_market_rangebreaks(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------- CHART 4: ROC of COI (cumulative intraday) ----------
if {"COI75", "OptionType"}.issubset(df.columns):
    coi_min = df.groupby(["Minute", "OptionType"], as_index=False, sort=False)["COI75"].sum()
    coi_min = coi_min.sort_values(["OptionType", "Minute"]).reset_index(drop=True)
    coi_min["Date"] = coi_min["Minute"].dt.date
    coi_min["ROC_COI"] = coi_min.groupby(["OptionType", "Date"])["COI75"].cumsum()
    fig = px.line(coi_min, x="Minute", y="ROC_COI", color="OptionType",
                  title="ROC of COI — Cumulative intraday (CE vs PE)")
    fig = apply_ce_pe_colors(fig); fig = apply_market_rangebreaks(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------- CHART 5: Spot (ABS at timestamp) vs Neutral (smoothed) ----------
spot_col    = first_present(df, ["SpotPrice", "Spot", "Spot_Price"])
neutral_col = first_present(df, ["Neutral_Point", "NeutralPoint", "Neutral", "CP"])

if spot_col and neutral_col and not df.empty:
    # Use the latest reading within each minute for Spot (absolute value)
    df_sorted = df.sort_values("DateTime")
    g = df_sorted.groupby("Minute", as_index=False, sort=False).agg(
        Spot_abs=(spot_col, lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[-1]
                  if pd.to_numeric(s, errors="coerce").dropna().size else np.nan),
        Neutral_mean=(neutral_col, "mean"),
    ).sort_values("Minute").reset_index(drop=True)

    # Neutral remains your cumulative intraday average (smoothing like Excel SUMIF≤time)
    g["Date"] = g["Minute"].dt.date
    g["Neutral_mean_cs"]  = g.groupby("Date")["Neutral_mean"].cumsum()
    g["Neutral_mean_cnt"] = g.groupby("Date")["Neutral_mean"].cumcount() + 1
    g["Neutral_mean_run"] = g["Neutral_mean_cs"] / g["Neutral_mean_cnt"]

    # Spot displayed as ABS and rounded to whole numbers
    g["Spot_abs_rounded"] = pd.to_numeric(g["Spot_abs"], errors="coerce").round(0)

    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(
        x=g["Minute"], y=g["Spot_abs_rounded"],
        name="Spot (absolute, rounded)", mode="lines",
        hovertemplate="%{y:.0f}"
    ))
    fig5.add_trace(go.Scatter(
        x=g["Minute"], y=g["Neutral_mean_run"],
        name="Neutral (avg, smoothed)", mode="lines",
        hovertemplate="%{y:.0f}"
    ))
    fig5.update_layout(
        title="Spot (ABS at timestamp) vs Neutral — intraday",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title=None, yaxis_title=None
    )
    fig5.update_yaxes(tickformat=".0f")
    fig5 = apply_market_rangebreaks(fig5)
    st.plotly_chart(fig5, use_container_width=True)


# ---------- CHART 6: Support / Resistance / Neutral (smoothed) vs Spot (ABS) ----------
sup_col  = first_present(df, ["Support", "SupportValue", "CN_Support", "CN"])
res_col  = first_present(df, ["Resistance", "ResistanceValue", "CP_Resistance", "CP"])
spot_col = first_present(df, ["SpotPrice", "Spot", "Spot_Price"])
neu_col  = first_present(df, ["Neutral_Point", "NeutralPoint", "Neutral", "CP_Neutral", "CPN", "CPR", "CNR"])

if sup_col and res_col and spot_col and neu_col and not df.empty:
    df_sorted = df.sort_values("DateTime")
    base = df_sorted.groupby("Minute", as_index=False, sort=False).agg(
        Support_mean=(sup_col, "mean"),
        Resistance_mean=(res_col, "mean"),
        Neutral_mean=(neu_col, "mean"),
        Spot_abs=(spot_col, lambda s: pd.to_numeric(s, errors="coerce").dropna().iloc[-1]
                  if pd.to_numeric(s, errors="coerce").dropna().size else np.nan),
    ).sort_values("Minute").reset_index(drop=True)

    # Keep S/R/Neutral as cumulative intraday averages (your Excel SUMIF≤time approach)
    base["Date"] = base["Minute"].dt.date
    for col in ["Support_mean", "Resistance_mean", "Neutral_mean"]:
        base[f"{col}_cs"]  = base.groupby("Date")[col].cumsum()
        base[f"{col}_cnt"] = base.groupby("Date")[col].cumcount() + 1
        base[f"{col}_run"] = base[f"{col}_cs"] / base[f"{col}_cnt"]

    # Spot displayed as ABS and rounded
    base["Spot_abs_rounded"] = pd.to_numeric(base["Spot_abs"], errors="coerce").round(0)

    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(x=base["Minute"], y=base["Support_mean_run"],    name="Support (avg)",    mode="lines", hovertemplate="%{y:.0f}"))
    fig6.add_trace(go.Scatter(x=base["Minute"], y=base["Resistance_mean_run"], name="Resistance (avg)", mode="lines", hovertemplate="%{y:.0f}"))
    fig6.add_trace(go.Scatter(x=base["Minute"], y=base["Neutral_mean_run"],    name="Neutral (avg)",    mode="lines", hovertemplate="%{y:.0f}"))
    fig6.add_trace(go.Scatter(x=base["Minute"], y=base["Spot_abs_rounded"],    name="Spot (absolute, rounded)", mode="lines", hovertemplate="%{y:.0f}"))

    fig6.update_layout(
        title="Support / Resistance / Neutral (smoothed) vs Spot (ABS)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title=None, yaxis_title=None
    )
    fig6.update_yaxes(tickformat=".0f")
    fig6 = apply_market_rangebreaks(fig6)
    st.plotly_chart(fig6, use_container_width=True)


# ---------- CHART 7: Spot (rounded) with Signals ----------
# Requirements:
# - Up (green ▲): "BUY CE" or "Write PE"
# - Down (red ▼): "BUY PE" or "Write CE"
# - Star (★): both an up and a down signal at the same timestamp
# - Hover shows: Signal + OptionLabel + Strike (integer if possible)
# Columns used (robust detection): Minute (time), Spot/SpotPrice, Signal (BR), OptionType (J), 
# OptionLabel (M), Strike/StrikePrice (I)

# ---------- CHART 7: Spot (rounded) with Signals ----------
# Columns: use robust detection so nothing else in the app needs to change
spot_col   = first_present(df, ["SpotPrice", "Spot", "Spot_Price"])
strike_col = first_present(df, ["StrikePrice", "Strike", "Strike_Price", "Strike Price", "I"])
signal_col = first_present(df, ["Signal", "Signals", "BR"])
type_col   = first_present(df, ["OptionType", "Type", "J"])
label_col  = first_present(df, ["OptionLabel", "Label", "M"])

if all([spot_col, strike_col, signal_col, type_col, label_col]) and not df.empty:
    df_chart = df.copy()

    # Normalise inputs we rely on
    df_chart["Signal"]      = df_chart[signal_col].astype(str).str.upper().str.strip()
    df_chart["OptionType"]  = df_chart[type_col].astype(str).str.upper().str.strip()
    df_chart["OptionLabel"] = (
        df_chart[label_col].astype(str).str.upper().str.replace(" ", "", regex=False)
    )
    # Strike as neat integer (blank if not a number)
    strike_num = pd.to_numeric(df_chart[strike_col], errors="coerce")
    df_chart["StrikeTxt"] = strike_num.round(0).astype("Int64").astype(str).str.replace("<NA>", "", regex=False)

    # Spot rounded (so the y-value feels like Excel’s chart)
    df_chart["SpotRounded"] = pd.to_numeric(df_chart[spot_col], errors="coerce").round(0)

    # Keep only rows that actually have a BUY/WRITE signal
    sigdf = df_chart[df_chart["Signal"].isin(["BUY", "WRITE"])].copy()

    # Build the hover text the way you requested
    sigdf["hover_txt"] = (
        sigdf["Signal"] + " - " + sigdf["OptionLabel"] + " " + sigdf["OptionType"] + " @" + sigdf["StrikeTxt"]
    )

    # CE/PE side classification
    green_mask = ((sigdf["Signal"] == "BUY") & (sigdf["OptionType"] == "CE")) | \
                 ((sigdf["Signal"] == "WRITE") & (sigdf["OptionType"] == "PE"))
    red_mask   = ((sigdf["Signal"] == "BUY") & (sigdf["OptionType"] == "PE")) | \
                 ((sigdf["Signal"] == "WRITE") & (sigdf["OptionType"] == "CE"))

    # We'll use your existing "Minute" column on the x-axis (already created earlier)
    xcol = "Minute"

    green = sigdf.loc[green_mask, [xcol, spot_col, "hover_txt"]].rename(columns={spot_col: "y"})
    red   = sigdf.loc[red_mask,   [xcol, spot_col, "hover_txt"]].rename(columns={spot_col: "y"})

    # Times where both green & red exist -> star marker with combined hover
    both_times = pd.Index(green[xcol]).intersection(red[xcol])
    star = (
        sigdf[sigdf[xcol].isin(both_times)]
        .groupby(xcol, as_index=False)
        .agg(y=(spot_col, "first"), hover_txt=("hover_txt", lambda s: "<br>".join(sorted(set(s)))))
    )

    # Base chart (spot)
    fig7 = go.Figure()
    fig7.add_trace(go.Scatter(
        x=df_chart[xcol], y=df_chart["SpotRounded"],
        mode="lines", name="Spot (rounded)"
    ))

    # Green ▲
    if not green.empty:
        fig7.add_trace(go.Scatter(
            x=green[xcol], y=green["y"], mode="markers",
            name="Signal ▲ (up)",
            marker=dict(symbol="triangle-up", size=12, color="green"),
            text=green["hover_txt"], hoverinfo="text"
        ))

    # Red ▼
    if not red.empty:
        fig7.add_trace(go.Scatter(
            x=red[xcol], y=red["y"], mode="markers",
            name="Signal ▼ (down)",
            marker=dict(symbol="triangle-down", size=12, color="red"),
            text=red["hover_txt"], hoverinfo="text"
        ))

    # ★ when both sides at same timestamp
    if not star.empty:
        fig7.add_trace(go.Scatter(
            x=star[xcol], y=star["y"], mode="markers",
            name="Multiple Signals ★",
            marker=dict(symbol="star", size=14, color="gold"),
            text=star["hover_txt"], hoverinfo="text"
        ))

    fig7.update_layout(
        title="Spot Price (rounded) with Signals",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title="Time", yaxis_title="Spot"
    )
    fig7 = apply_market_rangebreaks(fig7)
    st.plotly_chart(fig7, use_container_width=True)
# ---------- end Chart 7 ----------





# ---------- CHART 8: Cumulative Money Flow (AP) ----------
ap_col = first_present(df, ["AP", "CumulativeMoneyFlow", "MoneyFlowCum"])
if ap_col and "OptionType" in df.columns:
    g = df.groupby(["Minute", "OptionType"], as_index=False, sort=False)[ap_col].sum()
    g = g.sort_values(["OptionType", "Minute"]).reset_index(drop=True)
    g["Date"] = g["Minute"].dt.date
    g["AP_cum"] = g.groupby(["OptionType", "Date"])[ap_col].cumsum()
    fig = px.line(g, x="Minute", y="AP_cum", color="OptionType",
                  title="Cumulative Money Flow (AP) — CE vs PE")
    fig = apply_ce_pe_colors(fig); fig = apply_market_rangebreaks(fig)
    st.plotly_chart(fig, use_container_width=True)
