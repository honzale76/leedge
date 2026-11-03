import warnings
import xml.etree.ElementTree as ET

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

warnings.filterwarnings("ignore")

st.set_page_config(page_title="London Desk", layout="wide")

# ---------------------------------------------------------------------------
# Barevná schémata (vybírají se v bočním panelu). Další přidáš sem jako nový řádek.
# ---------------------------------------------------------------------------
PRESETS = {
    "Zelená": {"bg": "#0a0e0d", "card": "#111715", "border": "#1e2a26", "accent": "#3ecf8e"},
    "Modrá": {"bg": "#090d14", "card": "#0f1623", "border": "#1c2a40", "accent": "#5b9dff"},
    "Fialová": {"bg": "#0d0a14", "card": "#15111f", "border": "#2a2140", "accent": "#a78bfa"},
    "Zlatá": {"bg": "#0c0b09", "card": "#161411", "border": "#2b261d", "accent": "#e5b84b"},
}
MUTED = "#93a79e"
LEVEL_COLORS = ["#f0c05a", "#c9923a", "#7aa2ff", "#4f78d6"]

with st.sidebar:
    st.header("Nastavení")
    scheme = st.selectbox("Barevné schéma", list(PRESETS))
    asia_start, asia_end = st.slider("Asia session (londýnský čas, hodiny)", 0, 12, (0, 8))
    day_start = st.selectbox(
        "Začátek obchodního dne (londýnský čas)",
        [0, 21, 22, 23],
        index=0,
        format_func=lambda h: f"{h:02d}:00",
        help="Od této hodiny se počítá 'včerejší' high/low a profil objemu. Mnoho brokerů používá 22:00.",
    )

P = PRESETS[scheme]

CSS = """
<style>
html {font-size: 14px;}
.stApp {background: %BG%;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: %CARD%; border-right: 1px solid %BORDER%;}
.block-container {max-width: 1500px; padding: 1rem 1.5rem 2rem;}
[data-testid="stVerticalBlock"] {gap: 0.6rem;}
[data-testid="stMetric"] {
    background: %CARD%; border: 1px solid %BORDER%; border-radius: 10px; padding: 8px 12px;
}
[data-testid="stMetricLabel"] p {font-size: 0.72rem; color: %MUTED%;}
[data-testid="stMetricValue"] {font-size: 1.25rem; line-height: 1.3;}
[data-testid="stMetricDelta"] {font-size: 0.7rem;}
h1, h2, h3, h4, h5 {letter-spacing: -0.01em; margin: 0; padding: 0.2rem 0 0.1rem;}
h4 {font-size: 1rem !important; font-weight: 700;}
[data-testid="stCaptionContainer"] {font-size: 0.7rem;}
[data-testid="stMarkdownContainer"] p {margin-bottom: 0.25rem; font-size: 0.85rem;}
[data-testid="stExpander"] {border: 1px solid %BORDER%; border-radius: 10px; background: %CARD%;}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: %CARD%; border-color: %BORDER% !important; border-radius: 10px;
}
</style>
"""
for token, value in {"%BG%": P["bg"], "%CARD%": P["card"], "%BORDER%": P["border"], "%MUTED%": MUTED}.items():
    CSS = CSS.replace(token, value)
st.markdown(CSS, unsafe_allow_html=True)

PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "XAU/USD (zlato)": "GC=F",  # futures na zlato, dobrá náhrada spotu
}
# Futures se skutečným objemem (pro Volume POC)
FUTURES = {
    "EUR/USD": "6E=F",
    "GBP/USD": "6B=F",
    "XAU/USD (zlato)": "GC=F",
}
DXY = "DX-Y.NYB"
MONTHS = ["Led", "Úno", "Bře", "Dub", "Kvě", "Čvn", "Čvc", "Srp", "Zář", "Říj", "Lis", "Pro"]


# ---------------------------------------------------------------------------
# Pomocné funkce pro hodnocení bullish / bearish
# ---------------------------------------------------------------------------
def sign(x: float, threshold: float = 0.0) -> int:
    return 1 if x > threshold else (-1 if x < -threshold else 0)


def badge(score: int) -> str:
    return {1: ":green[▲ Bullish]", -1: ":red[▼ Bearish]", 0: ":gray[● Neutrální]"}[score]


def verdict(total: int) -> str:
    if total >= 3:
        return ":green[▲ Bullish]"
    if total >= 1:
        return ":green[▲ Mírně bullish]"
    if total == 0:
        return ":gray[● Neutrální]"
    if total > -3:
        return ":red[▼ Mírně bearish]"
    return ":red[▼ Bearish]"


# ---------------------------------------------------------------------------
# Stahování dat
# ---------------------------------------------------------------------------
def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


@st.cache_data(ttl=3600)
def load(ticker: str, years: int = 15) -> pd.DataFrame:
    return _download(ticker, f"{years}y", "1d")


def _to_london(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/London")
    return df


@st.cache_data(ttl=900)
def load_hourly(ticker: str) -> pd.DataFrame:
    """Hodinové svíčky za posledních 10 dní, přepočtené na londýnský čas."""
    return _to_london(_download(ticker, "10d", "1h"))


@st.cache_data(ttl=900)
def load_intraday(ticker: str) -> pd.DataFrame:
    """15minutové svíčky futures za posledních 7 dní (pro objemový profil)."""
    return _to_london(_download(ticker, "7d", "15m"))


def seasonality(df: pd.DataFrame, years: int = 10) -> pd.Series:
    """Průměrný měsíční pohyb v % za posledních X let."""
    monthly = df["Close"].resample("ME").last()
    returns = monthly.pct_change().dropna() * 100
    returns = returns[returns.index >= returns.index.max() - pd.DateOffset(years=years)]
    avg = returns.groupby(returns.index.month).mean()
    avg.index = [MONTHS[m - 1] for m in avg.index]
    return avg


def day_slices(frame: pd.DataFrame, day_start: int):
    """Rozdělí svíčky na aktuální a předchozí obchodní den (den začíná v hodině day_start)."""
    keys = pd.Series((frame.index - pd.Timedelta(hours=day_start)).date, index=frame.index)
    ordered = sorted(set(keys))
    current = frame[keys == ordered[-1]]
    previous = frame[keys == ordered[-2]] if len(ordered) > 1 else frame.iloc[0:0]
    return current, previous


def session_levels(hourly: pd.DataFrame, asia_start: int = 0, asia_end: int = 8, day_start: int = 0) -> dict:
    """Asia range, včerejší high/low a dnešní London range (vše v londýnském čase)."""
    current = hourly.index[-1].date()
    day = hourly[hourly.index.date == current]
    asia = day[(day.index.hour >= asia_start) & (day.index.hour < asia_end)]
    london = day[day.index.hour >= 8]
    _, prev_day = day_slices(hourly, day_start)

    def hi(d):
        return float(d["High"].max()) if not d.empty else None

    def lo(d):
        return float(d["Low"].min()) if not d.empty else None

    return {
        "date": current,
        "price": float(hourly["Close"].iloc[-1]),
        "asia_high": hi(asia),
        "asia_low": lo(asia),
        "prev_high": hi(prev_day),
        "prev_low": lo(prev_day),
        "london_high": hi(london),
        "london_low": lo(london),
    }


def volume_profile(bars: pd.DataFrame, bins: int = 40):
    """Profil objemu: POC (cena s největším objemem) a value area (70 % objemu)."""
    bars = bars[bars["Volume"] > 0]
    if bars.empty:
        return None
    low, high = float(bars["Low"].min()), float(bars["High"].max())
    if high <= low:
        return None
    edges = np.linspace(low, high, bins + 1)
    volume = np.zeros(bins)
    for _, bar in bars.iterrows():
        i0 = int(np.clip(np.searchsorted(edges, bar["Low"], side="right") - 1, 0, bins - 1))
        i1 = int(np.clip(np.searchsorted(edges, bar["High"], side="right") - 1, 0, bins - 1))
        volume[i0 : i1 + 1] += bar["Volume"] / (i1 - i0 + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    poc_index = int(volume.argmax())
    target = 0.7 * volume.sum()
    lo_i = hi_i = poc_index
    accumulated = volume[poc_index]
    while accumulated < target and (lo_i > 0 or hi_i < bins - 1):
        below = volume[lo_i - 1] if lo_i > 0 else -1
        above = volume[hi_i + 1] if hi_i < bins - 1 else -1
        if above >= below:
            hi_i += 1
            accumulated += volume[hi_i]
        else:
            lo_i -= 1
            accumulated += volume[lo_i]
    return {
        "poc": float(centers[poc_index]),
        "vah": float(edges[hi_i + 1]),
        "val": float(edges[lo_i]),
        "centers": centers,
        "volume": volume,
    }


def estimate_delta(bars: pd.DataFrame):
    """Hrubý odhad delty: objem rostoucích svíček minus objem klesajících. Ne skutečný orderflow."""
    up = float(bars.loc[bars["Close"] >= bars["Open"], "Volume"].sum())
    down = float(bars.loc[bars["Close"] < bars["Open"], "Volume"].sum())
    total = up + down
    return up - down, (up - down) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Grafy
# ---------------------------------------------------------------------------
def _finish(chart, height):
    return (
        chart.properties(height=height)
        .configure(background="rgba(0,0,0,0)")
        .configure_axis(labelColor=MUTED, gridColor=P["border"], domainColor=P["border"])
        .configure_legend(labelColor=MUTED)
        .configure_view(strokeWidth=0)
    )


def line_chart(frame: pd.DataFrame, height: int = 220):
    """Čárový graf s osou přizpůsobenou datům. První sloupec je plná čára, ostatní čárkované úrovně."""
    data = frame.copy()
    data.index.name = "Cas"
    long = data.reset_index().melt(id_vars="Cas", var_name="Rada", value_name="Hodnota").dropna()
    lo, hi = long["Hodnota"].min(), long["Hodnota"].max()
    pad = (hi - lo) * 0.08 or abs(hi) * 0.001
    names = list(data.columns)
    color = alt.Color(
        "Rada:N",
        scale=alt.Scale(domain=names, range=[P["accent"]] + LEVEL_COLORS[: len(names) - 1]),
        legend=alt.Legend(orient="bottom", title=None) if len(names) > 1 else None,
    )
    x = alt.X("Cas:T", title=None)
    y = alt.Y("Hodnota:Q", scale=alt.Scale(domain=[lo - pad, hi + pad]), title=None)
    base = alt.Chart(long)
    layers = [
        base.transform_filter(alt.datum.Rada == names[0]).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    ]
    if len(names) > 1:
        layers.append(
            base.transform_filter(alt.datum.Rada != names[0])
            .mark_line(strokeWidth=1, strokeDash=[4, 3])
            .encode(x=x, y=y, color=color)
        )
    st.altair_chart(_finish(alt.layer(*layers), height), use_container_width=True, theme=None)


def profile_chart(prof: dict, price: float, height: int = 260):
    """Vodorovný profil objemu s čarami POC, VAH, VAL a aktuální ceny."""
    data = pd.DataFrame({"Cena": prof["centers"], "Objem": prof["volume"]})
    low = min(data["Cena"].min(), price)
    high = max(data["Cena"].max(), price)
    pad = (high - low) * 0.05
    y = alt.Y("Cena:Q", scale=alt.Scale(domain=[low - pad, high + pad]), title=None)
    area = (
        alt.Chart(data)
        .mark_area(orient="horizontal", opacity=0.45, color=P["accent"], line={"color": P["accent"]})
        .encode(y=y, x=alt.X("Objem:Q", title=None, axis=None))
    )
    names = ["POC", "VAH", "VAL", "Cena"]
    rules = (
        alt.Chart(pd.DataFrame({"Cena": [prof["poc"], prof["vah"], prof["val"], price], "Název": names}))
        .mark_rule(strokeDash=[4, 3], strokeWidth=1.5)
        .encode(
            y=y,
            color=alt.Color(
                "Název:N",
                scale=alt.Scale(domain=names, range=["#f0c05a", "#7aa2ff", "#7aa2ff", "#ffffff"]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
        )
    )
    st.altair_chart(_finish(alt.layer(area, rules), height), use_container_width=True, theme=None)


# ---------------------------------------------------------------------------
# COT, novinky, kalendář
# ---------------------------------------------------------------------------
# Kódy kontraktů CFTC: Euro FX, British Pound, Gold
COT_CODES = {
    "EUR/USD": "099741",
    "GBP/USD": "096742",
    "XAU/USD (zlato)": "088691",
}


@st.cache_data(ttl=6 * 3600)
def load_cot(code: str, weeks: int = 104) -> pd.DataFrame:
    """Stáhne týdenní COT (velcí spekulanti, Legacy report) z CFTC."""
    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    params = {
        "cftc_contract_market_code": code,
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": weeks,
        "$select": "report_date_as_yyyy_mm_dd,noncomm_positions_long_all,noncomm_positions_short_all",
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["long"] = pd.to_numeric(df["noncomm_positions_long_all"])
    df["short"] = pd.to_numeric(df["noncomm_positions_short_all"])
    df["net"] = df["long"] - df["short"]
    df = df.sort_values("date").set_index("date")
    return df[["long", "short", "net"]]


HEADERS = {"User-Agent": "Mozilla/5.0 (osobní dashboard London Desk)"}

# Měna a klíčová slova, podle kterých se v novinkách pozná "náš" instrument
PAIR_INFO = {
    "EUR/USD": {"ccy": "EUR", "base": [" eur ", " euro ", " ecb ", "lagarde", "eurozone", "euro zone"]},
    "GBP/USD": {"ccy": "GBP", "base": [" gbp ", " pound ", "sterling", " boe ", "bank of england", " uk "]},
    "XAU/USD (zlato)": {"ccy": None, "base": [" gold ", " xau ", "bullion"]},
}
USD_WORDS = [" usd ", "dollar", " fed ", "fomc", "powell", " dxy ", "treasury", " us "]
BULL_WORDS = ["rise", "rises", "gain", "gains", "jump", "jumps", "climb", "climbs", "rally", "rallies",
              "surge", "surges", "hawkish", "strong", "beat", "beats", "higher", "advance", "firm", "boost"]
BEAR_WORDS = ["fall", "falls", "drop", "drops", "slide", "slides", "decline", "declines", "slump", "slumps",
              "dovish", "weak", "miss", "misses", "lower", "tumble", "tumbles", "retreat", "pressure", "cut"]
NEWS_FEEDS = {
    "FXStreet": "https://www.fxstreet.com/rss/news",
    "ForexLive": "https://www.forexlive.com/feed/news",
}


@st.cache_data(ttl=900)
def load_news() -> pd.DataFrame:
    """Stáhne titulky z volných RSS feedů."""
    items = []
    for source, url in NEWS_FEEDS.items():
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            for entry in root.iter("item"):
                items.append(
                    {
                        "source": source,
                        "title": (entry.findtext("title") or "").strip(),
                        "link": (entry.findtext("link") or "").strip(),
                        "time": pd.to_datetime(entry.findtext("pubDate"), utc=True, errors="coerce"),
                    }
                )
        except Exception:
            continue
    df = pd.DataFrame(items, columns=["source", "title", "link", "time"])
    return df.sort_values("time", ascending=False).reset_index(drop=True)


def classify_headline(title: str, base_words: list):
    """Vrátí (relevantní?, skóre -1/0/+1) podle slov v titulku. Hrubý odhad, ne AI."""
    text = f" {title.lower()} "
    is_base = any(w in text for w in base_words)
    is_usd = any(w in text for w in USD_WORDS)
    if not (is_base or is_usd):
        return False, 0
    raw = sum(w in text.split() or f" {w} " in text for w in BULL_WORDS) - sum(
        w in text.split() or f" {w} " in text for w in BEAR_WORDS
    )
    if raw == 0:
        return True, 0
    direction = 1 if raw > 0 else -1
    if is_base and not is_usd:
        return True, direction
    if is_usd and not is_base:
        return True, -direction  # silný dolar = pro náš pár negativní
    return True, 0


@st.cache_data(ttl=3600)
def load_calendar() -> pd.DataFrame:
    """Týdenní ekonomický kalendář (volný feed ve formátu Forex Factory)."""
    response = requests.get(
        "https://nfs.faireconomy.media/ff_calendar_thisweek.json", headers=HEADERS, timeout=20
    )
    response.raise_for_status()
    df = pd.DataFrame(response.json())
    df["time"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Prague")
    for col in ["forecast", "previous"]:
        if col not in df:
            df[col] = ""
    return df


def session_status() -> str:
    now_london = pd.Timestamp.now(tz="Europe/London")
    hour_float = now_london.hour + now_london.minute / 60
    if now_london.weekday() >= 5:
        return ":gray[●] Trh je o víkendu zavřený, úrovně jsou z posledního obchodního dne."
    if hour_float < 8:
        minutes = int((8 - hour_float) * 60)
        return f":orange[●] Asia session. London open (09:00 u nás) za {minutes // 60} h {minutes % 60} min."
    if hour_float < 16.5:
        return ":green[●] London session právě běží (do 17:30 u nás)."
    return ":gray[●] London session už skončila."


# ---------------------------------------------------------------------------
# Hlavička: název a výběr páru
# ---------------------------------------------------------------------------
head_left, head_right = st.columns([1, 2])
head_left.markdown("#### London Desk")
head_left.caption("Ranní přehled před London open (08:00 London = 09:00 u nás)")
pair = head_right.radio("Pár", list(PAIRS.keys()), horizontal=True, label_visibility="collapsed")

df = load(PAIRS[pair])
if df.empty:
    st.error("Nepodařilo se stáhnout data. Zkus to za chvíli znovu.")
    st.stop()

is_gold = "XAU" in pair
unit, unit_label, decimals = (1, "$", 2) if is_gold else (10000, "pips", 4)

# ---------------------------------------------------------------------------
# Výpočty a hodnocení (každý faktor dá skóre -1 bearish / 0 neutrální / +1 bullish)
# ---------------------------------------------------------------------------
signals = {}  # název -> (skóre, krátký popis)
errors = {}

last = float(df["Close"].iloc[-1])
prev = float(df["Close"].iloc[-2])
change = (last / prev - 1) * 100

dxy = load(DXY, years=2)
d_last = d_change = None
if not dxy.empty:
    d_last = float(dxy["Close"].iloc[-1])
    d_change = (d_last / float(dxy["Close"].iloc[-2]) - 1) * 100
    s = -sign(d_change, 0.1)  # slabý dolar = pro naše páry bullish
    text = "Dolar slábne" if s > 0 else ("Dolar posiluje" if s < 0 else "Dolar beze změny")
    signals["Dolar (DXY)"] = (s, f"{text} ({d_change:+.2f} %)")

# COT
cot = None
try:
    cot = load_cot(COT_CODES[pair])
    net = int(cot["net"].iloc[-1])
    net_change = int(cot["net"].iloc[-1] - cot["net"].iloc[-2])
    long_pct = cot["long"].iloc[-1] / (cot["long"].iloc[-1] + cot["short"].iloc[-1]) * 100
    cot_lo, cot_hi = cot["net"].min(), cot["net"].max()
    cot_index = (cot["net"].iloc[-1] - cot_lo) / (cot_hi - cot_lo) * 100 if cot_hi > cot_lo else 50
    if net_change > 0 and cot_index >= 50:
        s = 1
    elif net_change < 0 and cot_index <= 50:
        s = -1
    else:
        s = 0
    signals["COT"] = (s, f"Změna {net_change:+,} za týden, index {cot_index:.0f}/100".replace(",", " "))
except Exception as error:
    errors["COT"] = str(error)

# Seasonalita
seas = seasonality(df)
this_month = MONTHS[pd.Timestamp.today().month - 1]
if this_month in seas.index:
    value = float(seas[this_month])
    signals["Seasonalita"] = (sign(value, 0.1), f"{this_month}: průměr {value:+.2f} %")

# Novinky
news_rows = []
news_total = 0
try:
    news = load_news()
    for _, item in news.iterrows():
        relevant, score = classify_headline(item["title"], PAIR_INFO[pair]["base"])
        if relevant:
            news_rows.append((item, score))
        if len(news_rows) >= 10:
            break
    if news_rows:
        news_total = sum(score for _, score in news_rows)
        signals["Novinky"] = (sign(news_total, 1), f"Skóre {news_total:+d} z {len(news_rows)} titulků")
except Exception as error:
    errors["Novinky"] = str(error)

# London session (cena vůči Asia range)
hourly = lv = None
try:
    hourly = load_hourly(PAIRS[pair])
    lv = session_levels(hourly, asia_start, asia_end, day_start)
    p = lv["price"]
    if lv["asia_high"] is not None and p > lv["asia_high"]:
        signals["London session"] = (1, "Cena nad Asia High")
    elif lv["asia_low"] is not None and p < lv["asia_low"]:
        signals["London session"] = (-1, "Cena pod Asia Low")
    else:
        signals["London session"] = (0, "Cena uvnitř Asia range")
except Exception as error:
    errors["London session"] = str(error)

# Volume POC (z futures)
prof = None
poc_info = {}
try:
    intraday = load_intraday(FUTURES[pair])
    cur_bars, prev_bars = day_slices(intraday, day_start)
    prof = volume_profile(prev_bars)
    if prof is None:
        raise ValueError("Za včerejšek nejsou k dispozici objemová data.")
    fut_price = float(intraday["Close"].iloc[-1])
    delta, delta_ratio = estimate_delta(cur_bars if not cur_bars.empty else prev_bars)
    if fut_price > prof["vah"]:
        s, text = 1, "Cena nad value area"
    elif fut_price < prof["val"]:
        s, text = -1, "Cena pod value area"
    else:
        s, text = 0, "Cena uvnitř value area"
        if abs(delta_ratio) > 0.1:
            s = sign(delta_ratio)
            text += ", převaha kupců" if s > 0 else ", převaha prodejců"
    signals["Volume POC"] = (s, text)
    poc_info = {"price": fut_price, "delta": delta, "delta_ratio": delta_ratio}
except Exception as error:
    errors["Volume POC"] = str(error)


def title_with_badge(title: str, key: str) -> str:
    return f"#### {title}  {badge(signals[key][0])}" if key in signals else f"#### {title}"


# ---------------------------------------------------------------------------
# Horní řádek: cena, DXY, stav session
# ---------------------------------------------------------------------------
top1, top2, top3, top4 = st.columns(4)
top1.metric(pair, f"{last:,.4f}" if last < 100 else f"{last:,.2f}", f"{change:+.2f} %")
if d_last is not None:
    top2.metric("DXY (index dolaru)", f"{d_last:.2f}", f"{d_change:+.2f} %")
top3.metric("Poslední data", str(df.index[-1].date()))
top4.markdown(session_status())

# ---------------------------------------------------------------------------
# Celkový obraz: souhrn a jednotlivé faktory
# ---------------------------------------------------------------------------
st.markdown("#### Celkový obraz")
ORDER = ["Dolar (DXY)", "COT", "Seasonalita", "Novinky", "London session", "Volume POC"]
cards = st.columns(len(ORDER) + 1)

total = sum(score for score, _ in signals.values())
with cards[0].container(border=True):
    st.caption("Celkově")
    st.markdown(f"**{verdict(total)}**")
    st.caption(f"Skóre {total:+d} z {len(signals)} faktorů")

for column, name in zip(cards[1:], ORDER):
    with column.container(border=True):
        st.caption(name)
        if name in signals:
            st.markdown(badge(signals[name][0]))
            st.caption(signals[name][1])
        else:
            st.markdown(":gray[n/a]")
            st.caption("Data se nenačetla")
st.caption(
    "Každý faktor dává ▲ bullish, ▼ bearish nebo ● neutrální podle jednoduchých pravidel. "
    "Slouží jako rychlá orientace, není to obchodní signál."
)

# ---------------------------------------------------------------------------
# Řádek 1: London session (vlevo) a kalendář (vpravo)
# ---------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.markdown(title_with_badge("London session", "London session"))
    if lv is None:
        st.warning(f"London session data se nepodařilo načíst: {errors.get('London session', '')}")
    else:
        def show(col, name, key):
            level = lv[key]
            if level is None:
                col.metric(name, "n/a")
                return
            distance = (level - lv["price"]) * unit
            col.metric(name, f"{level:,.{decimals}f}", f"{distance:+.1f} {unit_label}", delta_color="off")

        c1, c2, c3, c4, c5 = st.columns(5)
        show(c1, "Asia High", "asia_high")
        show(c2, "Asia Low", "asia_low")
        if lv["asia_high"] is not None and lv["asia_low"] is not None:
            c3.metric("Asia range", f"{(lv['asia_high'] - lv['asia_low']) * unit:.1f} {unit_label}")
        else:
            c3.metric("Asia range", "n/a")
        show(c4, "Včerejší High", "prev_high")
        show(c5, "Včerejší Low", "prev_low")

        frame = hourly["Close"].tail(48).to_frame("Cena")
        for label, key in [("Asia High", "asia_high"), ("Asia Low", "asia_low"),
                           ("Včerejší High", "prev_high"), ("Včerejší Low", "prev_low")]:
            if lv[key] is not None:
                frame[label] = lv[key]
        frame.index = frame.index.tz_convert("Europe/Prague").tz_localize(None)
        line_chart(frame, height=230)
        st.caption(
            f"Data ze dne {lv['date']}. Asia = {asia_start:02d}:00-{asia_end:02d}:00 londýnského času "
            f"({asia_start + 1:02d}:00-{asia_end + 1:02d}:00 u nás), obchodní den začíná v {day_start:02d}:00. "
            "Vzdálenost je od aktuální ceny. Cena nad Asia High = bullish, pod Asia Low = bearish."
        )

with right:
    st.markdown("#### Kalendář (čas v ČR)")
    try:
        cal = load_calendar()
        ccys = ["USD"] + ([PAIR_INFO[pair]["ccy"]] if PAIR_INFO[pair]["ccy"] else [])
        cal = cal[cal["country"].isin(ccys) & cal["impact"].isin(["High", "Medium"])].copy()
        cal["Dopad"] = cal["impact"].map({"High": "Vysoký", "Medium": "Střední"})
        cal["Den"] = cal["time"].dt.strftime("%a %d.%m.")
        cal["Čas"] = cal["time"].dt.strftime("%H:%M")
        cal = cal.rename(columns={"country": "Měna", "title": "Událost", "forecast": "Předpověď", "previous": "Předchozí"})
        cols = ["Čas", "Měna", "Událost", "Dopad", "Předpověď", "Předchozí"]

        today = pd.Timestamp.now(tz="Europe/Prague").date()
        today_df = cal[cal["time"].dt.date == today]
        if today_df.empty:
            st.caption("Dnes nejsou žádné důležité události pro tento pár.")
        else:
            st.dataframe(today_df[cols], hide_index=True, use_container_width=True,
                         height=min(36 * (len(today_df) + 1) + 3, 300))
        with st.expander("Zbytek týdne"):
            st.dataframe(cal[["Den"] + cols], hide_index=True, use_container_width=True, height=300)
    except Exception as error:
        st.warning(f"Kalendář se nepodařilo načíst: {error}")

# ---------------------------------------------------------------------------
# Řádek 2: cena, seasonalita, COT
# ---------------------------------------------------------------------------
col_price, col_seas, col_cot = st.columns(3)

with col_price:
    st.markdown(title_with_badge("Cena (90 dní) a dolar", "Dolar (DXY)"))
    line_chart(df["Close"].tail(90).to_frame("Cena"), height=200)
    st.caption("Slábnoucí dolar (DXY dolů) je pro EUR/USD, GBP/USD i zlato bullish, posilující bearish.")

with col_seas:
    st.markdown(title_with_badge("Seasonalita (10 let)", "Seasonalita"))
    st.bar_chart(seas, height=200, color=P["accent"])
    if "Seasonalita" in signals:
        st.caption(f"{signals['Seasonalita'][1]}. Nad +0,1 % = bullish, pod -0,1 % = bearish.")

with col_cot:
    st.markdown(title_with_badge("COT: velcí spekulanti", "COT"))
    if cot is None:
        st.warning(f"COT data se nepodařilo načíst: {errors.get('COT', '')}")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Čistá pozice", f"{net:,}".replace(",", " "))
        m2.metric("Změna / týden", f"{net_change:+,}".replace(",", " "))
        m3.metric("Podíl long", f"{long_pct:.0f} %")
        m4.metric("COT index", f"{cot_index:.0f}/100")
        line_chart(cot["net"].to_frame("Čistá pozice"), height=130)
        st.caption(
            f"Report k {cot.index[-1].date()} (vychází v pátek). Bullish: pozice rostou a index je nad 50. "
            "Bearish: pozice klesají a index je pod 50."
        )

# ---------------------------------------------------------------------------
# Řádek 3: Volume POC a novinky
# ---------------------------------------------------------------------------
poc_col, news_col = st.columns([2, 3])

with poc_col:
    st.markdown(title_with_badge("Volume POC (včerejší profil)", "Volume POC"))
    if prof is None:
        st.warning(f"Objemová data se nepodařilo načíst: {errors.get('Volume POC', '')}")
    else:
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("POC", f"{prof['poc']:,.{decimals}f}")
        q2.metric("VAH", f"{prof['vah']:,.{decimals}f}")
        q3.metric("VAL", f"{prof['val']:,.{decimals}f}")
        q4.metric("Delta (odhad)", f"{poc_info['delta_ratio'] * 100:+.0f} %")
        profile_chart(prof, poc_info["price"], height=250)
        st.caption(
            f"Profil z futures {FUTURES[pair]} (15min svíčky, skutečný objem), cena futures {poc_info['price']:,.{decimals}f}. "
            "Cena nad value area = bullish, pod = bearish, uvnitř = rozhoduje delta. "
            "Delta je odhad podle směru svíček, ne skutečný orderflow."
        )

with news_col:
    st.markdown(title_with_badge("Novinky", "Novinky"))
    if "Novinky" in errors:
        st.warning(f"Novinky se nepodařilo načíst: {errors['Novinky']}")
    elif not news_rows:
        st.caption("Žádné relevantní titulky se nenašly.")
    else:
        labels = {1: ":green[Bullish]", -1: ":red[Bearish]", 0: ":gray[Neutrální]"}
        news_left, news_right = st.columns(2)
        for index, (item, score) in enumerate(news_rows):
            when = item["time"].tz_convert("Europe/Prague").strftime("%d.%m. %H:%M") if pd.notna(item["time"]) else ""
            title = item["title"].replace("[", "(").replace("]", ")")
            target = news_left if index < 5 else news_right
            target.markdown(f"{labels[score]} · [{title}]({item['link']}) — {item['source']}, {when}")
        st.caption("Hodnocení je jen podle slov v titulku (např. rises/falls), není to AI a může se mýlit.")

st.caption(
    "Data: Yahoo Finance, CFTC, FXStreet, ForexLive, Forex Factory kalendář. "
    "Minulost nezaručuje budoucnost, stránka není investiční doporučení."
)
