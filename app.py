"""GEX (gamma exposure) dashboard using CBOE's free delayed options feed."""

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from scipy.stats import norm

CBOE_URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/options/{}.json"
CBOE_CHART_URL = "https://cdn-api.cboe.com/api/global/delayed_quotes/charts"
PERIODS = {"Intraday": None, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
MA_COLORS = {20: "#9085e9", 50: "#e87ba4", 100: "#5598e7", 200: "#c3c2b7"}
INDEX_SYMBOLS = {"SPX", "NDX", "RUT", "VIX", "XSP", "DJX", "OEX"}
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
OPT_RE = re.compile(r"^(?P<root>\D+)(?P<exp>\d{6})(?P<type>[CP])(?P<strike>\d{8})$")

POS = "#2a78d6"  # positive gamma (blue)
NEG = "#eb6834"  # negative gamma (orange)
SPOT = "#888780"
FLIP = "#eda100"

st.set_page_config(page_title="GEX Dashboard", page_icon="📊", layout="wide")


@st.cache_data(ttl=900, max_entries=10, show_spinner="Downloading option chain from CBOE…")
def fetch_chain(ticker: str):
    ticker = ticker.upper().strip().lstrip("^_")
    if not TICKER_RE.match(ticker):
        raise ValueError(f"'{ticker}' is not a valid ticker symbol.")
    candidates = [f"_{ticker}", ticker] if ticker in INDEX_SYMBOLS else [ticker, f"_{ticker}"]
    for sym in candidates:
        r = requests.get(CBOE_URL.format(sym), timeout=30)
        if r.status_code == 200:
            payload = r.json()
            data = payload["data"]
            df = pd.DataFrame(data["options"])
            parts = df["option"].str.extract(OPT_RE)
            df["root"] = parts["root"]
            df["type"] = parts["type"]
            df["strike"] = parts["strike"].astype(float) / 1000
            df["expiration"] = pd.to_datetime(parts["exp"], format="%y%m%d")
            quote = {k: v for k, v in data.items() if k != "options"}
            quote["cboe_symbol"] = sym
            return df.dropna(subset=["type"]), quote, payload["timestamp"]
    raise ValueError(f"No CBOE data found for '{ticker}'.")


@st.cache_data(ttl=900, max_entries=20, show_spinner="Downloading price history…")
def fetch_prices(cboe_symbol: str, intraday: bool) -> pd.DataFrame:
    kind = "intraday" if intraday else "historical"
    r = requests.get(f"{CBOE_CHART_URL}/{kind}/{cboe_symbol}.json", timeout=30)
    r.raise_for_status()
    rows = r.json()["data"]
    if intraday:
        df = pd.DataFrame([{"time": x["datetime"], **x["price"], "volume": x["volume"]["stock_volume"],
                            "opt_volume": x["volume"]["total_options_volume"]} for x in rows])
    else:
        df = pd.DataFrame(rows).rename(columns={"date": "time"})
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["time"] = pd.to_datetime(df["time"])
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    # Old index rows have no open; fall back to the close.
    df["open"] = df["open"].where(df["open"] > 0, df["close"])
    return df


def add_indicators(df: pd.DataFrame, ma_periods, ma_kind: str) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    for n in ma_periods:
        df[f"ma{n}"] = close.ewm(span=n, adjust=False).mean() if ma_kind == "EMA" else close.rolling(n).mean()
    df["macd"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["macd"] - df["signal"]
    # Wilder's RSI.
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    return df


def years_to_expiry(expiration: pd.Series, am_settled: pd.Series, now_et: pd.Timestamp) -> np.ndarray:
    # PM-settled options stop at 16:00 ET; AM-settled index monthlies settle at the 09:30 open.
    # Floor at ~1 hour so 0DTE gamma stays finite.
    close = expiration + pd.to_timedelta(np.where(am_settled, 9.5, 16.0), unit="h")
    secs = (close - now_et).dt.total_seconds().to_numpy()
    return np.maximum(secs, 3600) / (365 * 24 * 3600)


def bs_gamma(spot, strike, iv, t, r=0.04, q=0.0):
    """Black-Scholes gamma; spot may be a column vector to evaluate a price grid."""
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + (r - q + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
        g = np.exp(-q * t) * norm.pdf(d1) / (spot * iv * np.sqrt(t))
    return np.nan_to_num(g)


def gex_dollars(gamma, oi, spot, sign):
    # Dollar gamma per 1% move: dealers assumed long calls / short puts.
    return sign * gamma * oi * 100 * spot**2 * 0.01


def zero_crossing(x, y):
    idx = np.where(np.diff(np.sign(y)) != 0)[0]
    if len(idx) == 0:
        return None
    # Pick the crossing nearest the middle of the grid (i.e. current spot).
    i = idx[np.argmin(np.abs(idx - len(x) // 2))]
    return x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i])


def fmt_bn(v):
    sign = "-" if v < 0 else ""
    return f"{sign}${abs(v) / 1e9:,.2f}B" if abs(v) >= 1e8 else f"{sign}${abs(v) / 1e6:,.1f}M"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", "SPX", help="Indices (SPX, NDX, RUT, VIX) or stocks/ETFs (SPY, QQQ, AAPL…)")
    if st.button("Refresh data", width="stretch"):
        fetch_chain.clear()

try:
    chain, quote, ts = fetch_chain(ticker)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

spot = float(quote["current_price"])
# CBOE's timestamp is UTC; expirations are naive ET dates, so compare in naive ET.
now_et = pd.Timestamp(ts, tz="UTC").tz_convert("America/New_York").tz_localize(None)

expirations = sorted(chain["expiration"].unique())
with st.sidebar:
    exp_mode = st.radio("Expirations", ["All", "Next N", "Pick"], horizontal=True)
    if exp_mode == "Next N":
        n = st.slider("Number of expirations", 1, min(20, len(expirations)), 4)
        selected = expirations[:n]
    elif exp_mode == "Pick":
        labels = [pd.Timestamp(e).strftime("%Y-%m-%d") for e in expirations]
        picked = st.multiselect("Expirations", labels, default=labels[:1])
        selected = [expirations[labels.index(p)] for p in picked]
    else:
        selected = expirations
    window = st.slider("Strike window (± % of spot)", 2, 40, 8)
    split = st.toggle("Split calls / puts", value=False)
    top_n = st.slider("Top GEX levels on price chart", 0, 15, 6)
    st.subheader("Indicators")
    ma_periods = st.multiselect("Moving averages", list(MA_COLORS), default=[20, 50])
    ma_kind = st.radio("MA type", ["SMA", "EMA"], horizontal=True)
    show_macd = st.toggle("MACD (12, 26, 9)", value=True)
    show_rsi = st.toggle("RSI (14)", value=True)
    show_vol = st.toggle("Volume", value=True,
                         help="Share volume. Indices have none, so intraday shows option volume instead.")
    st.caption("Source: CBOE delayed quotes (~15 min delay). Estimates only, not investment advice; see the disclaimer at the bottom.")

df = chain[chain["expiration"].isin(selected) & (chain["open_interest"] > 0)].copy()
if df.empty:
    st.warning("No open interest for the selected expirations.")
    st.stop()

df["sign"] = np.where(df["type"] == "C", 1.0, -1.0)
# Index monthlies whose root is the bare index symbol (SPX, NDX, RUT…) are AM-settled; weeklies (SPXW…) are PM.
is_index = quote["cboe_symbol"].startswith("_")
df["t"] = years_to_expiry(df["expiration"], is_index & (df["root"] == quote["cboe_symbol"].lstrip("_")), now_et)
# CBOE rounds gamma to 4 decimals, so recompute it from each contract's IV where available.
df["gamma"] = np.where(df["iv"] > 0, bs_gamma(spot, df["strike"], df["iv"], df["t"]), df["gamma"])
df["gex"] = gex_dollars(df["gamma"], df["open_interest"], spot, df["sign"])

by_strike = df.pivot_table(index="strike", columns="type", values="gex", aggfunc="sum").fillna(0)
for c in ("C", "P"):
    if c not in by_strike:
        by_strike[c] = 0.0
by_strike["net"] = by_strike["C"] + by_strike["P"]

# Gamma profile: total GEX re-computed across a range of hypothetical spot prices.
df["iv_f"] = df["iv"].where(df["iv"] > 0, np.nan)
prof_df = df.dropna(subset=["iv_f"])
grid = np.linspace(spot * 0.85, spot * 1.15, 121)
gam = bs_gamma(grid[:, None], prof_df["strike"].to_numpy(), prof_df["iv_f"].to_numpy(), prof_df["t"].to_numpy())
profile = (gam * (prof_df["open_interest"] * prof_df["sign"]).to_numpy() * 100 * grid[:, None] ** 2 * 0.01).sum(axis=1)
flip = zero_crossing(grid, profile)

total_gex = df["gex"].sum()
call_wall = by_strike["C"].idxmax()
put_wall = by_strike["P"].idxmin()

# ---------------------------------------------------------------- header
st.title(f"{quote['symbol'].lstrip('^')} Gamma Exposure")
st.caption(f"CBOE data as of {now_et:%Y-%m-%d %H:%M} ET · last trade {quote.get('last_trade_time', '—')} ET · "
           f"{len(selected)} of {len(expirations)} expirations · {len(df):,} contracts with OI")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{spot:,.2f}", f"{quote.get('price_change_percent', 0):+.2f}%")
c2.metric("Net GEX / 1% move", fmt_bn(total_gex),
          "Long gamma (dampening)" if total_gex > 0 else "Short gamma (amplifying)",
          delta_color="normal" if total_gex > 0 else "inverse")
c3.metric("Zero-gamma flip", f"{flip:,.2f}" if flip else "n/a",
          f"{(flip / spot - 1) * 100:+.2f}% from spot" if flip else None, delta_color="off")
c4.metric("Call wall", f"{call_wall:,.0f}", f"{(call_wall / spot - 1) * 100:+.2f}%", delta_color="off")
c5.metric("Put wall", f"{put_wall:,.0f}", f"{(put_wall / spot - 1) * 100:+.2f}%", delta_color="off")


def add_vline(fig, x, label, color, dash):
    fig.add_vline(x=x, line_color=color, line_dash=dash, line_width=2,
                  annotation_text=label, annotation_position="top", annotation_font_color=color)


lo, hi = spot * (1 - window / 100), spot * (1 + window / 100)
view = by_strike[(by_strike.index >= lo) & (by_strike.index <= hi)]

# ---------------------------------------------------------------- price chart
st.subheader("Price with GEX levels")
period = st.segmented_control("Period", list(PERIODS), default="3M", label_visibility="collapsed") or "3M"
try:
    px = fetch_prices(quote["cboe_symbol"], period == "Intraday")
except Exception as e:
    st.warning(f"Price history unavailable: {e}")
    px = None

if px is not None and not px.empty:
    # Compute on the full history so long MAs are warmed up, then trim to the period.
    px = add_indicators(px, sorted(ma_periods), ma_kind)
    if PERIODS[period]:
        px = px.tail(PERIODS[period])

    # Key levels first, then the largest |net GEX| strikes within the strike window.
    if call_wall == put_wall:
        levels = [(call_wall, "Call + put wall", SPOT, "dash", 2.5)]
    else:
        levels = [(call_wall, "Call wall", POS, "dash", 2.5), (put_wall, "Put wall", NEG, "dash", 2.5)]
    if flip:
        levels.append((flip, "Zero gamma", FLIP, "dot", 2.5))
    near = by_strike[(by_strike.index >= lo) & (by_strike.index <= hi)]["net"]
    biggest = near.abs().max() if len(near) else 0
    taken = {call_wall, put_wall}
    for strike, val in near.reindex(near.abs().sort_values(ascending=False).index).items():
        if sum(1 for lv in levels if lv[1].startswith("GEX")) >= top_n:
            break
        if strike in taken:
            continue
        levels.append((strike, f"GEX {fmt_bn(val)}", POS if val > 0 else NEG, "solid",
                       0.75 + 1.75 * abs(val) / biggest))

    # Indices report no share volume; intraday bars carry total option volume we can use instead.
    vol_col, vol_name = None, None
    if show_vol and px["volume"].sum() > 0:
        vol_col, vol_name = "volume", "Volume"
    elif show_vol and "opt_volume" in px and px["opt_volume"].sum() > 0:
        vol_col, vol_name = "opt_volume", "Opt. vol"
    panes = ["price"] + (["volume"] if vol_col else []) + (["macd"] if show_macd else []) + (["rsi"] if show_rsi else [])
    heights = {"price": 0.6, "volume": 0.14, "macd": 0.2, "rsi": 0.2}
    pc = make_subplots(rows=len(panes), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                       row_heights=[heights[p] for p in panes])
    row = {p: i + 1 for i, p in enumerate(panes)}
    pc.add_candlestick(row=1, col=1, showlegend=False, x=px["time"], open=px["open"], high=px["high"], low=px["low"], close=px["close"],
                       name=quote["symbol"].lstrip("^"), increasing_line_color="#1baf7a",
                       decreasing_line_color="#e34948", increasing_fillcolor="#1baf7a",
                       decreasing_fillcolor="#e34948", line_width=1)
    for n in sorted(ma_periods):
        pc.add_scatter(x=px["time"], y=px[f"ma{n}"], mode="lines", name=f"{ma_kind} {n}",
                       line=dict(color=MA_COLORS[n], width=1.5), hovertemplate="%{y:,.2f}", row=1, col=1)
    if vol_col:
        r = row["volume"]
        up = px["close"] >= px["open"]
        pc.add_bar(x=px["time"], y=px[vol_col], name=vol_name, showlegend=False, row=r, col=1,
                   marker_color=np.where(up, "#1baf7a", "#e34948"), marker_line_width=0, opacity=0.6,
                   hovertemplate="%{y:,.0f}")
        pc.update_yaxes(title=vol_name, row=r, col=1, tickformat="~s")
    if show_macd:
        r = row["macd"]
        pc.add_bar(x=px["time"], y=px["hist"], name="Histogram", showlegend=False, row=r, col=1,
                   marker_color=np.where(px["hist"] >= 0, "#1baf7a", "#e34948"), marker_line_width=0,
                   hovertemplate="%{y:.2f}")
        pc.add_scatter(x=px["time"], y=px["macd"], mode="lines", name="MACD", showlegend=False, row=r, col=1,
                       line=dict(color="#3987e5", width=1.5), hovertemplate="%{y:.2f}")
        pc.add_scatter(x=px["time"], y=px["signal"], mode="lines", name="Signal", showlegend=False, row=r, col=1,
                       line=dict(color="#eda100", width=1.5), hovertemplate="%{y:.2f}")
        pc.update_yaxes(title="MACD", row=r, col=1)
        pc.add_annotation(text="<span style='color:#3987e5'>MACD</span> · <span style='color:#eda100'>Signal</span>",
                          xref="x domain", yref=f"y{r} domain", x=0, y=1, showarrow=False,
                          xanchor="left", yanchor="bottom", font=dict(size=11))
    if show_rsi:
        r = row["rsi"]
        pc.add_scatter(x=px["time"], y=px["rsi"], mode="lines", name="RSI", showlegend=False, row=r, col=1,
                       line=dict(color="#9085e9", width=1.5), hovertemplate="%{y:.1f}")
        for y in (70, 30):
            pc.add_hline(y=y, line_color=SPOT, line_dash="dot", line_width=1, row=r, col=1)
        pc.update_yaxes(title="RSI", range=[0, 100], tickvals=[30, 50, 70], row=r, col=1)
    for y, label, color, dash, width in levels:
        pc.add_hline(row=1, col=1, y=y, line_color=color, line_dash=dash, line_width=width, opacity=0.9,
                     annotation_text=f"{y:,.0f} · {label}", annotation_position="right",
                     annotation_font_color=color, annotation_font_size=11)
    ys = [lv[0] for lv in levels] + [px["low"].min(), px["high"].max()]
    pad = (max(ys) - min(ys)) * 0.03
    pc.update_yaxes(range=[min(ys) - pad, max(ys) + pad], title="Price", row=1, col=1)
    rangebreaks = [dict(bounds=["sat", "mon"])]
    if period == "Intraday":
        rangebreaks.append(dict(bounds=[16.25, 9.5], pattern="hour"))
    pc.update_xaxes(rangeslider_visible=False, rangebreaks=rangebreaks)
    pc.update_layout(height=420 + 150 * (len(panes) - 1), margin=dict(t=30, b=10, l=10, r=150),
                     hovermode="x unified", bargap=0.2,
                     legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(pc, width="stretch")
    st.caption("Levels reflect today's open interest and the expiration filter; they are not historical levels.")

# ---------------------------------------------------------------- strike chart
fig = go.Figure()
hover = "Strike %{x:,.0f}<br>%{customdata}<extra>%{fullData.name}</extra>"
if split:
    for col, name, color in (("C", "Calls", POS), ("P", "Puts", NEG)):
        fig.add_bar(x=view.index, y=view[col] / 1e9, name=name, marker_color=color,
                    customdata=[fmt_bn(v) for v in view[col]], hovertemplate=hover)
    fig.update_layout(barmode="relative")
else:
    fig.add_bar(x=view.index, y=view["net"] / 1e9, name="Net GEX",
                marker_color=np.where(view["net"] >= 0, POS, NEG),
                customdata=[fmt_bn(v) for v in view["net"]], hovertemplate=hover)
add_vline(fig, spot, f"Spot {spot:,.0f}", SPOT, "solid")
if flip and lo <= flip <= hi:
    add_vline(fig, flip, f"Flip {flip:,.0f}", FLIP, "dash")
fig.update_layout(height=460, margin=dict(t=40, b=10, l=10, r=10), bargap=0.15,
                  yaxis_title="GEX ($B per 1% move)", xaxis_title="Strike",
                  legend=dict(orientation="h", y=1.08, x=0), showlegend=split, hovermode="closest")
fig.update_traces(marker_line_width=0)

st.subheader("GEX by strike")
st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------- open interest / volume
st.subheader("Open interest & volume by strike")
sel = chain[chain["expiration"].isin(selected)]
totals = sel.groupby("type")[["open_interest", "volume"]].sum().reindex(["C", "P"]).fillna(0)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Call OI", f"{totals.at['C', 'open_interest']:,.0f}")
m2.metric("Put OI", f"{totals.at['P', 'open_interest']:,.0f}")
m3.metric("Put/call OI", f"{totals.at['P', 'open_interest'] / max(totals.at['C', 'open_interest'], 1):.2f}")
m4.metric("Call volume", f"{totals.at['C', 'volume']:,.0f}")
m5.metric("Put volume", f"{totals.at['P', 'volume']:,.0f}")
m6.metric("Put/call volume", f"{totals.at['P', 'volume'] / max(totals.at['C', 'volume'], 1):.2f}")

oi_metric = st.segmented_control("Show", ["Open interest", "Volume"], default="Open interest",
                                 label_visibility="collapsed") or "Open interest"
field = "open_interest" if oi_metric == "Open interest" else "volume"
ov = (sel[(sel["strike"] >= lo) & (sel["strike"] <= hi)]
      .pivot_table(index="strike", columns="type", values=field, aggfunc="sum")
      .reindex(columns=["C", "P"]).fillna(0))
of = go.Figure()
# Calls plotted up, puts mirrored down so the two sides share one axis.
of.add_bar(x=ov.index, y=ov["C"], name="Calls", marker_color=POS, marker_line_width=0,
           hovertemplate="Strike %{x:,.0f}<br>%{y:,.0f} contracts<extra>Calls</extra>")
of.add_bar(x=ov.index, y=-ov["P"], name="Puts", marker_color=NEG, marker_line_width=0, customdata=ov["P"],
           hovertemplate="Strike %{x:,.0f}<br>%{customdata:,.0f} contracts<extra>Puts</extra>")
add_vline(of, spot, f"Spot {spot:,.0f}", SPOT, "solid")
of.update_layout(barmode="relative", height=420, margin=dict(t=40, b=10, l=10, r=10), bargap=0.15,
                 yaxis_title=f"{oi_metric} (puts < 0)", xaxis_title="Strike",
                 yaxis_tickformat="~s", legend=dict(orientation="h", y=1.08, x=0), hovermode="closest")
st.plotly_chart(of, width="stretch")
st.caption("Volume is the most recent session's; open interest is as of the prior close (OCC updates it overnight).")

left, right = st.columns(2)

# ---------------------------------------------------------------- profile chart
with left:
    st.subheader("Gamma profile vs. price")
    pf = go.Figure()
    pf.add_scatter(x=grid, y=np.where(profile >= 0, profile, np.nan) / 1e9, mode="lines",
                   line=dict(color=POS, width=2), fill="tozeroy", name="Positive",
                   hovertemplate="Price %{x:,.0f}<br>%{y:.2f}B<extra></extra>")
    pf.add_scatter(x=grid, y=np.where(profile < 0, profile, np.nan) / 1e9, mode="lines",
                   line=dict(color=NEG, width=2), fill="tozeroy", name="Negative",
                   hovertemplate="Price %{x:,.0f}<br>%{y:.2f}B<extra></extra>")
    add_vline(pf, spot, "Spot", SPOT, "solid")
    if flip:
        add_vline(pf, flip, "Flip", FLIP, "dash")
    pf.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10), showlegend=False,
                     xaxis_title="Underlying price", yaxis_title="Total GEX ($B per 1%)", hovermode="x")
    st.plotly_chart(pf, width="stretch")

# ---------------------------------------------------------------- expiration chart
with right:
    st.subheader("GEX by expiration")
    by_exp = df.groupby("expiration")["gex"].sum().head(20)
    ef = go.Figure()
    ef.add_bar(x=by_exp.index.strftime("%b %d '%y"), y=by_exp / 1e9,
               marker_color=np.where(by_exp >= 0, POS, NEG), marker_line_width=0,
               customdata=[fmt_bn(v) for v in by_exp], hovertemplate="%{x}<br>%{customdata}<extra></extra>")
    ef.update_layout(height=380, margin=dict(t=40, b=10, l=10, r=10),
                     yaxis_title="Net GEX ($B per 1%)", xaxis_type="category")
    st.plotly_chart(ef, width="stretch")

# ---------------------------------------------------------------- table
with st.expander("Top strikes table"):
    top = by_strike.reindex(by_strike["net"].abs().sort_values(ascending=False).index).head(25)
    st.dataframe(
        (top[["C", "P", "net"]] / 1e6).rename(columns={"C": "Call GEX ($M)", "P": "Put GEX ($M)", "net": "Net GEX ($M)"}),
        width="stretch", column_config={c: st.column_config.NumberColumn(format="%.1f") for c in
                                                 ["Call GEX ($M)", "Put GEX ($M)", "Net GEX ($M)"]},
    )

st.divider()
st.caption(
    "**Disclaimer:** For informational and educational purposes only; not investment advice. "
    "GEX figures are estimates: they assume dealers are long calls and short puts, which is a convention, "
    "not observed positioning. Data is CBOE's delayed public feed, provided for personal use, and may be "
    "delayed, incomplete or wrong."
)
