#!/usr/bin/env python3
"""
Stock Signals Bot v6
--------------------
MODE=morning   (weekdays ~6am Central, on yesterday's closing prices)
MODE=scan      (9am, 12pm, 2pm Central, on live prices ~15 min delayed)
  Scans the S&P 500 + S&P 400 (~900 stocks, $5+):
    1. RULE     - uptrend + momentum setup (the version that tested best).
    2. MODEL    - a prediction model scores every stock: the odds it beats the
                  market over the next month, learned from ~900 stocks x 5 years.
                  Its confidence levels come from months it was NOT trained on.
    3. CHECKS   - healthy business, news tone + big events (Yahoo, Google News,
                  SEC filings), no earnings within 2 weeks.
    4. SIGNALS  - BUY / UPTREND (hold) / SELL / WAIT with stop and exit prices.
    5. PAPER    - every BUY is tracked as a pretend trade until its sell signal.
  Scans also re-check the morning's BUYs and your HOLDINGS, and always notify.

MODE=compare   (run by hand)
  Tests rule versions and the model on the most recent ~1.5 years, which they
  never saw while being set up. Writes comparison.md / comparison.json.

History is used to estimate odds. It cannot know the future. Not financial advice.

Env vars (optional): NTFY_TOPIC, DASHBOARD_URL, HOLDINGS="F:11.20,AAPL:180.5",
                     SEC_CONTACT, BOT_EMAIL_USER, BOT_EMAIL_PASS, BOT_EMAIL_TO, MODE
"""
import datetime as dt
import io
import json
import os
import re
import smtplib
import ssl
import time
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---- Settings you can change ---------------------------------------------
INDEX_PAGES = [  # stock lists pulled fresh each run
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
]
EXTRA_TICKERS = []        # add any others you want scanned, e.g. ["SOFI", "HOOD"]
MARKET_TICKER = "SPY"     # stands in for the overall market (S&P 500)
MIN_PRICE = 5.0           # skip stocks under this price (holdings are always checked)
SHORTLIST = 60            # how many top stocks get the news + business check
TOP_N = 5                 # buy ideas listed in the notification

# Buy rule (version B from the comparison: tested best per signal count)
RSI_MAX = 70
STRETCH_ATR = 0           # 0 = off. Skipping "stretched" stocks tested worse.
MARKET_FILTER = False     # tested worse in the last 18 months; kept as an option
VALIDATION = "none"       # per-stock history test: "none", "loose" or "strict"

# Prediction model
USE_MODEL = True          # require the model to agree before a BUY
MODEL_MIN_TIER = "medium" # "high" = fewest, strongest BUYs; "medium"; "low" = model ignored
HORIZON = 20              # the model predicts the next 20 trading days (~1 month)

STOP_ATR = 2.0            # stop-loss = buy price - 2 x normal daily swing
TRAIL_ATR = 3.0           # trailing exit = highest close since buying - 3 x normal daily swing
EARNINGS_DAYS = 14        # no new BUY if earnings are this close
NEWS_DAYS = 7             # how far back headlines/filings count
TEST_SHARE = 0.30         # last 30% of history is the unseen test
COST = 0.001              # 0.1% per round trip for spreads/fees in the backtests
CHART_DAYS = 120          # days of price shown on the dashboard chart
FALLBACK = [              # used only if the stock lists can't be downloaded
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM",
    "V", "MA", "UNH", "LLY", "XOM", "CVX", "COST", "WMT", "HD", "NFLX", "CRM",
    "ORCL", "ADBE", "DIS", "BA", "KO", "PEP", "PLTR", "F", "GM", "SOFI", "T", "VZ",
    "PFE", "INTC", "BAC", "WFC", "C", "HOOD", "SNAP", "PINS", "RIVN", "LCID", "NIO",
    "PLUG", "AAL", "CCL", "NCLH", "KEY", "HBAN", "CMCSA", "WBD", "UBER", "SHOP",
]
# --------------------------------------------------------------------------

UA = {"User-Agent": "Mozilla/5.0 (stock-signals-bot)"}
SEC_UA = {"User-Agent": "stock-signals-bot " + os.getenv("SEC_CONTACT", "github.com/LaloMtz3/stock-bot")}
CT = ZoneInfo("America/Chicago")
PAPER_FILE = "paper_trades.json"
TODAY_FILE = "today.json"
TIER_RANK = {"low": 0, "medium": 1, "high": 2}

HOLDINGS = {}
for _p in os.getenv("HOLDINGS", "").split(","):
    if ":" in _p:
        _k, _v = _p.split(":", 1)
        try:
            HOLDINGS[_k.strip().upper().replace(".", "-")] = float(_v)
        except ValueError:
            pass


def now_ct():
    return dt.datetime.now(CT)


def session_label():
    n = now_ct()
    if n.time() < dt.time(8, 30):
        return "morning"
    return n.strftime("%-I%p").lower().replace(":00", "")


# ============================ stock list + data ===========================
def universe() -> dict:
    import requests
    out = {}
    for url in INDEX_PAGES:
        try:
            html = requests.get(url, headers=UA, timeout=30).text
            best = None
            for tb in pd.read_html(io.StringIO(html)):
                col = next((c for c in tb.columns if str(c).strip().lower() in ("symbol", "ticker", "ticker symbol")), None)
                if col is not None and (best is None or len(tb) > len(best[0])):
                    best = (tb, col)
            if best is None or len(best[0]) < 100:
                print(f"list looked wrong: {url}")
                continue
            tb, col = best
            ncol = next((c for c in tb.columns if str(c).strip().lower() in ("security", "company")), None)
            scol = next((c for c in tb.columns if "sector" in str(c).lower()), None)
            for _, r in tb.iterrows():
                t = str(r[col]).strip().upper().replace(".", "-")
                if t and t != "NAN":
                    out[t] = {"name": str(r[ncol]) if ncol is not None else t,
                              "sector": str(r[scol]) if scol is not None else None}
            print(f"{url.rsplit('/', 1)[-1]}: {len(tb)} stocks")
        except Exception as e:
            print(f"could not load {url}: {e}")
    if len(out) < 300:
        print("using fallback list")
        for t in FALLBACK:
            out.setdefault(t, {"name": t, "sector": None})
    for t in EXTRA_TICKERS + list(HOLDINGS):
        out.setdefault(t.upper(), {"name": t.upper(), "sector": None})
    return out


def download(tickers, period="5y", chunk=100, completed_days_only=False):
    """Returns {ticker: DataFrame}. completed_days_only drops today's unfinished bar."""
    import yfinance as yf
    frames, tickers = {}, list(tickers)
    cutoff = None
    if completed_days_only and now_ct().time() < dt.time(15, 15):
        cutoff = now_ct().date()
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        for attempt in range(3):
            try:
                data = yf.download(part, period=period, group_by="ticker", auto_adjust=True,
                                   progress=False, threads=True)
                break
            except Exception as e:
                print(f"download retry {attempt + 1}: {e}")
                time.sleep(10)
        else:
            continue
        for t in part:
            try:
                df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(subset=["Close"])
                if cutoff is not None and len(df) and df.index[-1].date() >= cutoff:
                    df = df[[d.date() < cutoff for d in df.index]]
                if len(df):
                    frames[t] = df
            except Exception:
                pass
        print(f"downloaded {min(i + chunk, len(tickers))}/{len(tickers)}")
        time.sleep(2)
    return frames


def market_trend(frames):
    df = frames.get(MARKET_TICKER)
    if df is None or len(df) < 220:
        return None
    c = df["Close"]
    return c > c.rolling(200).mean()


# ============================ rule indicators =============================
def indicators(df, market=None, rsi_max=None, stretch=None, use_market=None):
    rsi_max = RSI_MAX if rsi_max is None else rsi_max
    stretch = STRETCH_ATR if stretch is None else stretch
    use_market = MARKET_FILTER if use_market is None else use_market
    c, h, l = df["Close"], df["High"], df["Low"]
    out = pd.DataFrame(index=df.index)
    out["close"] = c
    out["sma50"] = c.rolling(50).mean()
    out["sma200"] = c.rolling(200).mean()
    out["ret5"] = c.pct_change(5)
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    out["rsi"] = 100 - 100 / (1 + gain / loss)
    prev = c.shift()
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    out["atr"] = tr.rolling(14).mean()
    out["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    out["stretch"] = (c - out["sma50"]) / out["atr"]
    if market is not None:
        out["market_ok"] = market.reindex(out.index).ffill().fillna(False).astype(bool)
    else:
        out["market_ok"] = True
    entry = ((c > out["sma50"]) & (out["sma50"] > out["sma200"]) & (out["ret5"] > 0)
             & out["rsi"].between(40, rsi_max))
    if stretch:
        entry &= out["stretch"] <= stretch
    if use_market:
        entry &= out["market_ok"]
    out["entry"] = entry
    return out


def simulate(ind):
    c, sma50, atr, entry = (ind["close"].values, ind["sma50"].values,
                            ind["atr"].values, ind["entry"].values)
    n = len(c)
    pos = np.zeros(n)
    trades, in_pos, e_px, e_i, peak, stop = [], False, 0.0, 0, 0.0, 0.0
    for i in range(n):
        if np.isnan(sma50[i]) or np.isnan(atr[i]):
            continue
        if in_pos:
            peak = max(peak, c[i])
            if c[i] < max(stop, peak - TRAIL_ATR * atr[i]) or c[i] < sma50[i]:
                trades.append(dict(entry_i=e_i, exit_i=i, ret=c[i] / e_px - 1 - COST))
                in_pos = False
        elif entry[i]:
            in_pos, e_px, e_i, peak = True, c[i], i, c[i]
            stop = c[i] - STOP_ATR * atr[i]
        pos[i] = 1.0 if in_pos else 0.0
    daily = pd.Series(c).pct_change().fillna(0).values
    strat = np.concatenate([[0.0], pos[:-1] * daily[1:]])
    return dict(strat=strat, hold=daily, trades=trades, open=in_pos,
                open_entry_i=e_i if in_pos else None)


def stats(rets, trades):
    eq = np.cumprod(1 + rets)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0
    total = float(eq[-1] - 1) if len(eq) else 0.0
    wins = [t for t in trades if t["ret"] > 0]
    return dict(ret=total * 100, maxdd=dd * 100, trades=len(trades),
                win=(len(wins) / len(trades) * 100) if trades else None,
                avg=float(np.mean([t["ret"] for t in trades]) * 100) if trades else None,
                score=total / abs(dd) if dd < 0 else total * 10)


def passes(strategy, hold, mode):
    if mode == "none":
        return True
    if strategy["trades"] < 2 or strategy["ret"] <= 0:
        return False
    if mode == "strict":
        return strategy["score"] >= hold["score"]
    return strategy["maxdd"] > hold["maxdd"]


def backtest(ind):
    sim = simulate(ind)
    n = len(ind)
    cut = int(n * (1 - TEST_SHARE))
    res = {}
    for name, a, b in (("train", 200, cut), ("test", cut, n)):
        tr = [t for t in sim["trades"] if a <= t["entry_i"] < b]
        res[name] = dict(strategy=stats(sim["strat"][a:b], tr), hold=stats(sim["hold"][a:b], []))
    res["validated"] = all(passes(res[p]["strategy"], res[p]["hold"], VALIDATION) for p in ("train", "test"))
    res["history_ok"] = all(passes(res[p]["strategy"], res[p]["hold"], "loose") for p in ("train", "test"))
    res["sim"] = sim
    return res


# ============================ prediction model ============================
FEATS = ["ret5", "ret20", "ret60", "ret120", "mom_12_1", "d50", "d200", "slope50", "rsi",
         "atrp", "vol20", "vratio", "dhigh", "rs60", "rs120", "mkt200", "mkt20",
         "mom_rank", "rs_rank"]


def stock_features(df, spy):
    """Everything here uses only prices known at each day's close."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    f["ret5"], f["ret20"] = c.pct_change(5), c.pct_change(20)
    f["ret60"], f["ret120"] = c.pct_change(60), c.pct_change(120)
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    s50, s200 = c.rolling(50).mean(), c.rolling(200).mean()
    f["d50"], f["d200"] = c / s50 - 1, c / s200 - 1
    f["slope50"] = s50 / s50.shift(20) - 1
    d = c.diff()
    f["rsi"] = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean())
    prev = c.shift()
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    f["atrp"] = tr.rolling(14).mean() / c
    f["vol20"] = c.pct_change().rolling(20).std()
    f["vratio"] = v / v.rolling(20).mean()
    f["dhigh"] = c / c.rolling(252).max() - 1
    s = spy.reindex(f.index).ffill()
    f["rs60"] = f["ret60"] - s["ret60"]
    f["rs120"] = f["ret120"] - s["ret120"]
    f["mkt200"], f["mkt20"] = s["d200"], s["ret20"]
    fwd = c.shift(-HORIZON) / c - 1
    f["y_ex"] = fwd - s["fwd"]              # beat the market over the next month?
    return f


def build_panel(frames):
    spy_df = frames.get(MARKET_TICKER)
    if spy_df is None:
        return None
    sc = spy_df["Close"]
    spy = pd.DataFrame({"ret20": sc.pct_change(20), "ret60": sc.pct_change(60),
                        "ret120": sc.pct_change(120), "d200": sc / sc.rolling(200).mean() - 1,
                        "fwd": sc.shift(-HORIZON) / sc - 1})
    parts = []
    for t, df in frames.items():
        if t == MARKET_TICKER or len(df) < 300:
            continue
        f = stock_features(df, spy)
        f["ticker"] = t
        parts.append(f.iloc[252:])
    if not parts:
        return None
    p = pd.concat(parts)
    p.index.name = "date"
    p = p.reset_index()
    p["mom_rank"] = p.groupby("date")["mom_12_1"].rank(pct=True)
    p["rs_rank"] = p.groupby("date")["rs60"].rank(pct=True)
    for col in FEATS + ["y_ex"]:
        p[col] = p[col].astype("float32")
    return p


def fit_model(rows):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = rows[rows["y_ex"].notna() & (rows["date"].dt.dayofweek == 2)]   # weekly sample
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=31,
                                       min_samples_leaf=300, l2_regularization=1.0,
                                       random_state=7)
    m.fit(rows[FEATS], (rows["y_ex"] > 0).astype(int))
    return m


def calibrate(model, rows):
    """Bucket held-out predictions into 10 groups and record what actually happened."""
    rows = rows[rows["y_ex"].notna()].copy()
    rows["p"] = model.predict_proba(rows[FEATS])[:, 1]
    edges = np.unique(np.quantile(rows["p"], np.linspace(0, 1, 11)))
    rows["b"] = np.clip(np.searchsorted(edges, rows["p"], side="right") - 1, 0, len(edges) - 2)
    out = []
    for b, g in rows.groupby("b"):
        out.append(dict(lo=float(edges[b]), hi=float(edges[b + 1]), n=int(len(g)),
                        hit=float((g["y_ex"] > 0).mean()), avg_ex=float(g["y_ex"].mean())))
    return out


def tier_of(cal_bucket):
    if cal_bucket is None:
        return "low"
    if cal_bucket["hit"] >= 0.56 and cal_bucket["avg_ex"] > 0.005:
        return "high"
    if cal_bucket["hit"] >= 0.52 and cal_bucket["avg_ex"] > 0:
        return "medium"
    return "low"


def bucket_for(p, cal):
    for c in cal:
        if c["lo"] <= p <= c["hi"]:
            return c
    return cal[0] if p < cal[0]["lo"] else cal[-1]


def score_today(panel):
    """Train on older data, measure confidence on the last ~6 months it never trained on,
    then score the latest day for every stock."""
    labeled = panel[panel["y_ex"].notna()]
    last_lab = labeled["date"].max()
    dates = np.sort(labeled["date"].unique())
    calib_start = dates[max(0, len(dates) - 126)]
    embargo = dates[max(0, len(dates) - 126 - 30)]
    model = fit_model(labeled[labeled["date"] < embargo])
    cal = calibrate(model, labeled[labeled["date"] >= calib_start])
    latest = panel[panel["date"] == panel.groupby("ticker")["date"].transform("max")].copy()
    latest = latest.dropna(subset=["ret120", "mom_12_1"])
    latest["p"] = model.predict_proba(latest[FEATS])[:, 1]
    latest["p_rank"] = latest["p"].rank(pct=True)
    out = {}
    for _, r in latest.iterrows():
        b = bucket_for(float(r["p"]), cal)
        out[r["ticker"]] = dict(prob=round(float(r["p"]) * 100, 1), rank=round(float(r["p_rank"]) * 100),
                                tier=tier_of(b), hit=round(b["hit"] * 100, 1),
                                avg_ex=round(b["avg_ex"] * 100, 2))
    info = dict(trained_until=str(pd.Timestamp(embargo).date()),
                checked_on=f"{pd.Timestamp(calib_start).date()} to {pd.Timestamp(last_lab).date()}",
                buckets=[{k: round(v, 4) if isinstance(v, float) else v for k, v in c.items()} for c in cal])
    return out, info


# ============================ news + filings ==============================
FIN_POS = ["beat", "beats", "tops", "raises guidance", "raised guidance", "upgrade", "upgraded",
           "record", "surge", "soar", "jumps", "outperform", "buyback", "repurchase", "approval",
           "approved", "wins contract", "awarded", "strong demand", "dividend increase",
           "raises dividend", "expands", "partnership", "rallies", "all-time high"]
FIN_NEG = ["miss", "misses", "cuts guidance", "cut guidance", "lowers guidance", "downgrade",
           "downgraded", "lawsuit", "sued", "probe", "investigation", "subpoena", "recall",
           "layoffs", "job cuts", "bankruptcy", "chapter 11", "fraud", "plunge", "tumbles",
           "slumps", "sinks", "warning", "delist", "default", "resigns", "steps down",
           "short seller", "offering", "dilution", "halted", "restatement", "underperform"]
EVENTS = [  # (pattern, label, blocks a BUY?)
    (r"downgrad", "analyst downgrade", True),
    (r"lawsuit|sued|probe|investigation|subpoena|class action", "legal/regulatory trouble", True),
    (r"(cut|lower|slash)\w* (its )?(guidance|forecast|outlook)", "guidance cut", True),
    (r"(ceo|cfo|chief executive|chief financial).{0,40}(resign|step|depart|exit|ousted|fired)", "top executive leaving", True),
    (r"(stock|share|equity) offering|dilut", "new shares being sold", True),
    (r"bankrupt|chapter 11|going concern|default", "financial distress", True),
    (r"restat|accounting (error|irregular)", "accounting problem", True),
    (r"upgrad", "analyst upgrade", False),
    (r"(rais|boost|lift)\w* (its )?(guidance|forecast|outlook)", "guidance raised", False),
    (r"acqui|merger|to buy |buyout|takeover", "deal / acquisition news", False),
    (r"earnings|quarterly results|q[1-4] results", "earnings news", False),
]
SEC_ITEMS = {"1.01": ("major agreement", False), "1.03": ("bankruptcy filing", True),
             "2.01": ("acquisition completed", False), "2.02": ("earnings released", False),
             "2.05": ("restructuring costs", True), "2.06": ("asset write-down", True),
             "3.01": ("delisting notice", True), "4.02": ("past financials unreliable", True),
             "5.02": ("executive/board change", False)}
_CIK = None


def _rss(url, source):
    import requests
    try:
        r = requests.get(url, headers=UA, timeout=12)
        items = ET.fromstring(r.content).findall(".//item")[:15]
    except Exception:
        return []
    out = []
    for i in items:
        title = (i.findtext("title") or "").strip()
        if source == "Google News" and " - " in title:
            title, pub = title.rsplit(" - ", 1)
            src = pub.strip()
        else:
            src = source
        try:
            when = parsedate_to_datetime(i.findtext("pubDate")).astimezone(CT)
        except Exception:
            when = None
        out.append(dict(title=title, link=i.findtext("link") or "", source=src,
                        when=when.strftime("%Y-%m-%d") if when else None, _dt=when))
    return out


def sec_filings(ticker):
    import requests
    global _CIK
    try:
        if _CIK is None:
            data = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_UA, timeout=20).json()
            _CIK = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}
        cik = _CIK.get(ticker)
        if not cik:
            return []
        j = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=SEC_UA, timeout=20).json()
        rec = j["filings"]["recent"]
        out, cutoff = [], now_ct().date() - dt.timedelta(days=NEWS_DAYS)
        for form, date, items, acc in zip(rec["form"], rec["filingDate"], rec["items"], rec["accessionNumber"]):
            if form != "8-K" or dt.date.fromisoformat(date) < cutoff:
                continue
            for it in [x.strip() for x in str(items).split(",")]:
                if it in SEC_ITEMS:
                    label, bad = SEC_ITEMS[it]
                    out.append(dict(title=f"SEC 8-K filing: {label}", source="SEC filing", when=date,
                                    link=f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/",
                                    event=label, bad=bad))
        time.sleep(0.15)
        return out
    except Exception:
        return []


def headline_score(title, sia):
    t = title.lower()
    pos = sum(1 for w in FIN_POS if w in t)
    neg = sum(1 for w in FIN_NEG if w in t)
    lex = (pos - neg) / max(1, pos + neg)
    return 0.5 * sia.polarity_scores(title)["compound"] + 0.5 * lex


def news(ticker, name, sia):
    from urllib.parse import quote
    q = quote(f'"{name}" stock' if name and name != ticker else f"{ticker} stock")
    heads = (_rss(f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US", "Yahoo Finance")
             + _rss(f"https://news.google.com/rss/search?q={q}+when:{NEWS_DAYS}d&hl=en-US&gl=US&ceid=US:en", "Google News"))
    cutoff = now_ct() - dt.timedelta(days=NEWS_DAYS)
    seen, recent = set(), []
    for h in heads:
        key = re.sub(r"[^a-z0-9 ]", "", h["title"].lower())[:60]
        if key in seen or (h["_dt"] and h["_dt"] < cutoff):
            continue
        seen.add(key)
        recent.append(h)
    events = []
    for h in recent:
        t = h["title"].lower()
        for pat, label, bad in EVENTS:
            if re.search(pat, t):
                events.append(dict(event=label, bad=bad, title=h["title"], source=h["source"]))
                break
    filings = sec_filings(ticker)
    events += [dict(event=f["event"], bad=f["bad"], title=f["title"], source="SEC filing") for f in filings]
    sent = float(np.mean([headline_score(h["title"], sia) for h in recent])) if recent else 0.0
    shown = [{k: h[k] for k in ("title", "link", "source", "when")} for h in (filings + recent)[:6]]
    uniq = {}
    for e in events:
        uniq.setdefault(e["event"], e)
    return round(sent, 2), shown, list(uniq.values()), len(recent)


# ============================ business data ===============================
def fundamentals(tk):
    try:
        i = tk.info or {}
    except Exception:
        return {}
    return {k: i.get(k) for k in ("profitMargins", "revenueGrowth", "debtToEquity",
                                  "forwardPE", "shortName", "sector")}


def quality(f):
    if not f or f.get("profitMargins") is None:
        return None, ["business data unavailable"]
    notes, bad = [], 0
    pm, rg, de, pe = (f.get(k) for k in ("profitMargins", "revenueGrowth", "debtToEquity", "forwardPE"))
    if pm is not None:
        notes.append(f"profit margin {pm*100:.0f}%"); bad += pm <= 0
    if rg is not None:
        notes.append(f"revenue growth {rg*100:+.0f}%"); bad += rg < -0.05
    if de is not None:
        notes.append(f"debt/equity {de:.0f}%"); bad += de > 200
    if pe is not None and pe > 0:
        notes.append(f"forward P/E {pe:.0f}"); bad += pe > 80
    return bad == 0, notes


def next_earnings(tk):
    try:
        cal = tk.calendar
        d = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if d:
            d = d[0] if isinstance(d, (list, tuple)) else d
            d = pd.Timestamp(d).date()
            return d if d >= now_ct().date() else None
    except Exception:
        pass
    return None


# ============================ signals =====================================
SIGNAL_RANK = {"buy": 0, "hold": 1, "wait": 2, "sell": 3}


def evaluate(t, meta, ind, bt, model=None, f=None, earn=None, sent=0.0, heads=None,
             events=None, checked=False):
    last = ind.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])
    sim = bt["sim"]
    q_ok, q_notes = quality(f) if checked else (None, ["not on today's shortlist, so not checked"])
    earn_soon = bool(earn and 0 <= (earn - now_ct().date()).days <= EARNINGS_DAYS)
    stop, target = price - STOP_ATR * atr, price + 3 * atr
    peak = float(ind["close"].iloc[sim["open_entry_i"]:].max()) if sim["open"] else float(ind["close"].iloc[-22:].max())
    trail = peak - TRAIL_ATR * atr
    exit_now = price < float(last["sma50"]) or price < trail
    setup = bool(last["entry"])
    tier = (model or {}).get("tier", "low")
    bad_events = [e["event"] for e in (events or []) if e["bad"]]

    reasons, blockers, warnings = [], [], []
    if setup:
        reasons.append("uptrend setup today")
    if model and tier != "low":
        reasons.append(f"model: {tier} confidence")
    if setup and USE_MODEL and model and TIER_RANK.get(tier, 0) < TIER_RANK[MODEL_MIN_TIER]:
        blockers.append(f"model not convinced ({tier} confidence)")
    if setup and USE_MODEL and not model:
        warnings.append("model unavailable today: rule only")
    if not bt["validated"]:
        blockers.append("no proven edge on this stock")
    if q_ok is False:
        blockers.append("weak fundamentals")
    if sent < -0.15:
        blockers.append("negative news tone")
    for ev in bad_events:
        blockers.append(f"news: {ev}")
    if earn_soon:
        blockers.append(f"earnings on {earn} (gap risk)")
    if not checked and setup:
        blockers.append("not checked today")
    if model and tier == "low" and model.get("rank", 50) <= 20:
        warnings.append("model outlook weak")

    if exit_now:
        signal = "sell"
    elif setup and not blockers:
        signal = "buy"
    elif sim["open"]:
        signal = "hold"
    else:
        signal = "wait"

    score = (model["prob"] / 10 if model else 5.0) + (1.0 if setup else 0.0) \
        + (1.0 if q_ok else (-1.0 if q_ok is False else 0.0)) + 1.5 * sent \
        - (1.0 if earn_soon else 0.0) - (2.0 if exit_now else 0.0) - len(bad_events)

    def pack(p):
        s, h = bt[p]["strategy"], bt[p]["hold"]
        return {"strategy": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items() if k != "score"},
                "hold": {"ret": round(h["ret"], 1), "maxdd": round(h["maxdd"], 1)}}

    dates = ind.index
    return dict(
        ticker=t, name=(f or {}).get("shortName") or meta.get("name") or t,
        sector=(f or {}).get("sector") or meta.get("sector"),
        price=round(price, 2), asof=dates[-1].strftime("%Y-%m-%d"),
        ret1=round(float(ind["close"].pct_change().iloc[-1] * 100), 2),
        ret5=round(float(last["ret5"] * 100), 2),
        rsi=round(float(last["rsi"]), 1), vol_ratio=round(float(last["vol_ratio"]), 2),
        stretch=round(float(last["stretch"]), 2),
        sma50=round(float(last["sma50"]), 2), atr=round(atr, 2),
        stop=round(stop, 2), target=round(target, 2), trail=round(trail, 2),
        signal=signal, rank=SIGNAL_RANK[signal], score=round(score, 2),
        reasons=reasons, blockers=blockers, warnings=warnings, model=model,
        checked=checked, quality=q_ok, quality_notes=q_notes,
        earnings=str(earn) if earn else None, sent=round(sent, 2), heads=heads or [],
        events=[{"event": e["event"], "bad": e["bad"], "source": e["source"]} for e in (events or [])],
        validated=bt["history_ok"], train=pack("train"), test=pack("test"),
        recent_trades=[dict(buy=dates[x["entry_i"]].strftime("%Y-%m-%d"),
                            sell=dates[x["exit_i"]].strftime("%Y-%m-%d"),
                            ret=round(x["ret"] * 100, 1)) for x in sim["trades"][-4:]],
    )


# ============================ full scan ===================================
def run_scan(live):
    """live=False: morning scan on completed days. live=True: intraday prices."""
    import yfinance as yf
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    uni = universe()
    print(f"universe: {len(uni)} stocks")
    frames = download(list(uni) + [MARKET_TICKER], completed_days_only=not live)
    mkt = market_trend(frames)
    market_ok = bool(mkt.iloc[-1]) if mkt is not None else True

    models, model_info = {}, None
    if USE_MODEL:
        try:
            panel = build_panel({t: d for t, d in frames.items()
                                 if t == MARKET_TICKER or float(d["Close"].iloc[-1]) >= MIN_PRICE or t in HOLDINGS})
            models, model_info = score_today(panel)
            del panel
            print(f"model scored {len(models)} stocks")
        except Exception as e:
            print(f"model failed, continuing without it: {e}")

    stage = {}
    for t, df in frames.items():
        if t == MARKET_TICKER and t not in uni:
            continue
        try:
            if len(df) < 400 or (float(df["Close"].iloc[-1]) < MIN_PRICE and t not in HOLDINGS):
                continue
            ind = indicators(df, mkt)
            stage[t] = (ind, backtest(ind))
        except Exception as e:
            print(f"skip {t}: {e}")
    print(f"backtested {len(stage)} stocks")

    prelim = {t: evaluate(t, uni.get(t, {}), ind, bt, models.get(t)) for t, (ind, bt) in stage.items()}
    ranked = sorted(prelim, key=lambda t: prelim[t]["score"], reverse=True)
    cands = [t for t in ranked if prelim[t]["reasons"] and prelim[t]["signal"] != "sell"
             and "uptrend setup today" in prelim[t]["reasons"] and t not in HOLDINGS
             and not any(b.startswith("model not") for b in prelim[t]["blockers"])]
    must = [t for t in ranked if t in HOLDINGS] + cands[:SHORTLIST * 2]
    shortlist = must + [t for t in ranked if t not in must][:max(0, SHORTLIST - len(must))]
    print(f"shortlist: {len(shortlist)} stocks")
    sia = SentimentIntensityAnalyzer()
    rows = dict(prelim)
    for t in shortlist:
        try:
            tk = yf.Ticker(t)
            name = uni.get(t, {}).get("name") or t
            sent, heads, events, n_news = news(t, name, sia)
            ind, bt = stage[t]
            rows[t] = evaluate(t, uni.get(t, {}), ind, bt, models.get(t), fundamentals(tk),
                               next_earnings(tk), sent, heads, events, checked=True)
            rows[t]["news_count"] = n_news
            time.sleep(0.2)
        except Exception as e:
            print(f"closer look failed for {t}: {e}")
    rows = sorted(rows.values(), key=lambda r: (r["rank"], -r["score"]))

    cal = None
    for t, (ind, _) in stage.items():
        if cal is None or ind.index[-1] > cal[-1] or len(ind.index) > len(cal):
            cal = ind.index
    cal = cal[-CHART_DAYS:]
    for r in rows:
        s = stage[r["ticker"]][0]["close"].reindex(cal).ffill()
        r["closes"] = [None if pd.isna(x) else round(float(x), 2) for x in s]
    return rows, [d.strftime("%Y-%m-%d") for d in cal], market_ok, stage, model_info


# ============================ paper trading ===============================
def load_paper():
    try:
        return json.load(open(PAPER_FILE))
    except Exception:
        return {"open": [], "closed": []}


def update_paper(book, stage, rows, kind, close_trades):
    if close_trades:
        still = []
        for p in book["open"]:
            got = stage.get(p["ticker"])
            if not got:
                still.append(p)
                continue
            ind = got[0]
            after = ind[ind.index > pd.Timestamp(p["last_checked"])]
            closed = False
            for d, r in after.iterrows():
                if pd.isna(r["atr"]) or pd.isna(r["sma50"]):
                    continue
                p["peak"] = max(p["peak"], float(r["close"]))
                trail = p["peak"] - TRAIL_ATR * float(r["atr"])
                if r["close"] < max(p["stop"], trail) or r["close"] < r["sma50"]:
                    p.update(exit_date=d.strftime("%Y-%m-%d"), exit=round(float(r["close"]), 2),
                             ret=round((float(r["close"]) / p["entry"] - 1) * 100, 2),
                             reason="stop-loss" if r["close"] < p["stop"] else
                             ("trailing exit" if r["close"] < trail else "below 50-day avg"))
                    book["closed"].append(p)
                    closed = True
                    break
            if not closed:
                if len(after):
                    p["last_checked"] = after.index[-1].strftime("%Y-%m-%d")
                    p["now"] = round(float(after["close"].iloc[-1]), 2)
                still.append(p)
        book["open"] = still
    held = {p["ticker"] for p in book["open"]}
    for r in rows:
        if r["signal"] == "buy" and r["ticker"] not in held:
            book["open"].append(dict(ticker=r["ticker"], kind=kind, entry_date=r["asof"], entry=r["price"],
                                     stop=r["stop"], peak=r["price"], last_checked=r["asof"], now=r["price"],
                                     tier=(r.get("model") or {}).get("tier")))
            held.add(r["ticker"])
    if not close_trades:            # live scan: refresh current prices of open trades
        px = {r["ticker"]: r["price"] for r in rows}
        for p in book["open"]:
            p["now"] = px.get(p["ticker"], p.get("now"))

    def summ(c):
        wins = [x for x in c if x["ret"] > 0]
        return dict(closed=len(c), win_rate=round(len(wins) / len(c) * 100, 1) if c else None,
                    avg_ret=round(float(np.mean([x["ret"] for x in c])), 2) if c else None,
                    avg_win=round(float(np.mean([x["ret"] for x in wins])), 2) if wins else None,
                    avg_loss=round(float(np.mean([x["ret"] for x in c if x["ret"] <= 0])), 2)
                    if len(wins) < len(c) else None)
    c = book["closed"]
    book["summary"] = dict(open=len(book["open"]), **summ(c),
                           morning=summ([x for x in c if x.get("kind", "morning") == "morning"]),
                           midday=summ([x for x in c if x.get("kind") == "midday"]))
    json.dump(book, open(PAPER_FILE, "w"), indent=1)
    return book


# ============================ today's tracking ============================
def load_today():
    try:
        s = json.load(open(TODAY_FILE))
        if s.get("date") == str(now_ct().date()):
            return s
    except Exception:
        pass
    return {"date": str(now_ct().date()), "buys": {}, "cancelled": [], "sell_alerts": []}


def track_today(state, rows, session):
    """Record new BUYs, cancel BUYs that broke down, raise holdings sell alerts."""
    by_t = {r["ticker"]: r for r in rows}
    new, cancelled, sells, status = [], [], [], []
    for r in rows:
        if r["signal"] == "buy" and r["ticker"] not in state["buys"]:
            state["buys"][r["ticker"]] = dict(price=r["price"], stop=r["stop"], sma50=r["sma50"], time=session)
            new.append(r)
    for t, b in state["buys"].items():
        r = by_t.get(t)
        if not r:
            continue
        since = round((r["price"] / b["price"] - 1) * 100, 2)
        b["now"], b["since"] = r["price"], since
        if t in state["cancelled"]:
            continue
        if session != "morning" and (r["price"] < b["stop"] or r["price"] < b["sma50"]):
            state["cancelled"].append(t)
            cancelled.append((t, r["price"], b["stop"] if r["price"] < b["stop"] else b["sma50"]))
        elif session != "morning" and t not in [n["ticker"] for n in new]:
            status.append((t, r["price"], since))
    for t, buy in HOLDINGS.items():
        r = by_t.get(t)
        if not r:
            continue
        stop = buy - STOP_ATR * r["atr"]
        why = []
        if r["price"] < stop:
            why.append(f"below your stop ${stop:.2f}")
        if r["price"] < r["trail"]:
            why.append(f"below trailing exit ${r['trail']:.2f}")
        if r["price"] < r["sma50"]:
            why.append(f"below 50-day avg ${r['sma50']:.2f}")
        if why and t not in state["sell_alerts"]:
            state["sell_alerts"].append(t)
            sells.append(f"SELL {t} ${r['price']:.2f} ({(r['price'] / buy - 1) * 100:+.1f}% vs your ${buy:g}): " + ", ".join(why))
    for r in rows:
        b = state["buys"].get(r["ticker"])
        if b:
            r["signal_time"], r["signal_price"] = b["time"], b["price"]
            r["since"] = b.get("since", 0.0)
            r["cancelled"] = r["ticker"] in state["cancelled"]
    json.dump(state, open(TODAY_FILE, "w"), indent=1)
    return new, cancelled, sells, status


def message(session, market_ok, new, cancelled, sells, status, book, n):
    lines = list(sells)
    if cancelled:
        lines += [f"BUY CANCELLED {t}: now ${p:.2f}, fell below ${lvl:.2f}" for t, p, lvl in cancelled]
    if new:
        tag = "" if session == "morning" else " (midday, not confirmed until close)"
        lines.append(f"{len(new)} new BUY{'s' if len(new) > 1 else ''}{tag}:")
        for r in new[:TOP_N]:
            m = r.get("model") or {}
            conf = f", {m['tier']} confidence ({m['hit']:.0f}% of similar setups beat the market)" if m else ""
            lines.append(f"{r['ticker']} ${r['price']:.2f}, stop ${r['stop']:.2f}{conf}")
    if status:
        lines.append("Today's BUYs: " + ", ".join(f"{t} ${p:.2f} ({s:+.1f}%)" for t, p, s in status[:6]))
    if not (sells or cancelled or new):
        lines.append("No new BUYs and no sell alerts." if session != "morning"
                     else "No buy setups passed all checks today. Waiting is a valid choice.")
    if MARKET_FILTER and not market_ok:
        lines.append("Market is in a downtrend: no new buys.")
    s = book.get("summary", {})
    if s.get("closed"):
        lines.append(f"Paper trades: {s['closed']} closed, {s['win_rate']}% winners, avg {s['avg_ret']:+.1f}%")
    return "\n".join(lines) + f"\n\nScanned {n} stocks. Odds from history, not guarantees."


# ============================ comparison ==================================
VARIANTS = [
    ("A. Old rules (v4)", 70, 0, False, "strict"),
    ("B. Rule only (current rule)", 70, 0, False, "none"),
    ("C. Calmer entry (RSI<60, not stretched)", 60, 3.0, False, "none"),
    ("D. Calmer entry + market filter", 60, 3.0, True, "none"),
    ("F. D + strict per-stock test", 60, 3.0, True, "strict"),
]


def trade_stats(name, trades, months, n_stocks=None):
    r = np.array([x["ret"] for x in trades]) if trades else np.array([0.0])
    wins, losses = r[r > 0], r[r <= 0]
    return dict(version=name, stocks_allowed=n_stocks, trades=len(trades),
                trades_per_month=round(len(trades) / months, 1),
                win_rate=round(float((r > 0).mean() * 100), 1) if trades else None,
                avg_trade=round(float(r.mean() * 100), 2) if trades else None,
                avg_win=round(float(wins.mean() * 100), 2) if len(wins) else None,
                avg_loss=round(float(losses.mean() * 100), 2) if len(losses) else None,
                profit_factor=round(float(wins.sum() / -losses.sum()), 2) if len(losses) and losses.sum() < 0 else None,
                median_days=int(np.median([x["days"] for x in trades])) if trades else None)


def compare():
    from sklearn.metrics import roc_auc_score
    uni = universe()
    frames = download(list(uni) + [MARKET_TICKER])
    frames = {t: d for t, d in frames.items() if t == MARKET_TICKER or
              (len(d) >= 400 and float(d["Close"].iloc[-1]) >= MIN_PRICE)}
    mkt = market_trend(frames)
    spy = frames[MARKET_TICKER]
    cut_i = int(len(spy) * (1 - TEST_SHARE))
    cut_date = spy.index[cut_i]
    months = (len(spy) - cut_i) / 21
    results, b_trades = [], []
    for name, rsi_max, stretch, use_mkt, mode in VARIANTS:
        trades, n_stocks = [], 0
        for t, df in frames.items():
            if t == MARKET_TICKER:
                continue
            ind = indicators(df, mkt, rsi_max=rsi_max, stretch=stretch, use_market=use_mkt)
            sim = simulate(ind)
            cut = int(np.searchsorted(ind.index, cut_date))
            s = stats(sim["strat"][200:cut], [x for x in sim["trades"] if 200 <= x["entry_i"] < cut])
            if not passes(s, stats(sim["hold"][200:cut], []), mode):
                continue
            n_stocks += 1
            for x in sim["trades"]:
                if x["entry_i"] >= cut:
                    trades.append(dict(t=t, date=ind.index[x["entry_i"]], ret=x["ret"],
                                       days=x["exit_i"] - x["entry_i"]))
        results.append(trade_stats(name, trades, months, n_stocks))
        if name.startswith("B."):
            b_trades = trades
        print(results[-1])

    # ---- the prediction model, trained only on data before the test period
    panel = build_panel(frames)
    dates = np.sort(panel["date"].unique())
    train_end = dates[max(0, int(np.searchsorted(dates, np.datetime64(cut_date))) - 30)]
    model = fit_model(panel[panel["date"] < train_end])
    test = panel[panel["date"] >= np.datetime64(cut_date)].dropna(subset=["ret120", "mom_12_1"]).copy()
    test["p"] = model.predict_proba(test[FEATS])[:, 1]
    test["p_rank"] = test.groupby("date")["p"].rank(pct=True)
    lab = test[test["y_ex"].notna()]
    auc = float(roc_auc_score(lab["y_ex"] > 0, lab["p"]))
    lab = lab.assign(dec=np.minimum((lab["p_rank"] * 10).astype(int), 9))
    deciles = [dict(decile=int(d) + 1, hit=round(float((g["y_ex"] > 0).mean() * 100), 1),
                    avg_ex=round(float(g["y_ex"].mean() * 100), 2), n=int(len(g)))
               for d, g in lab.groupby("dec")]
    pr = test.set_index(["ticker", "date"])["p_rank"]
    for lbl, cut_rank in (("G. Rule + model (top half)", 0.5), ("H. Rule + model (top 30%)", 0.7),
                          ("I. Rule + model (top 10%)", 0.9)):
        keep = [x for x in b_trades if pr.get((x["t"], x["date"]), 0) >= cut_rank]
        results.append(trade_stats(lbl, keep, months))
        print(results[-1])
    model_report = dict(auc=round(auc, 3), trained_until=str(pd.Timestamp(train_end).date()),
                        test_from=str(cut_date.date()), deciles=deciles)

    lines = [f"# Strategy comparison ({now_ct():%Y-%m-%d})", "",
             f"Unseen test period: {cut_date:%Y-%m-%d} to today. Costs of {COST*100:.1f}% per trade included.",
             f"The model was trained only on data up to {model_report['trained_until']}.", "",
             "| Version | Trades | Per month | Win rate | Avg trade | Avg win | Avg loss | Profit factor | Median days |",
             "|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v, s="": "–" if v is None else f"{v}{s}"
    for x in results:
        lines.append(f"| {x['version']} | {x['trades']} | {fmt(x['trades_per_month'])} | {fmt(x['win_rate'], '%')} | "
                     f"{fmt(x['avg_trade'], '%')} | {fmt(x['avg_win'], '%')} | {fmt(x['avg_loss'], '%')} | "
                     f"{fmt(x['profit_factor'])} | {fmt(x['median_days'])} |")
    lines += ["", f"## Prediction model (AUC {auc:.3f}; 0.5 = coin flip)", "",
              "Stocks ranked each day by the model, split into 10 groups (10 = most confident).",
              "Next-month result vs the S&P 500:", "",
              "| Group | Beat the market | Avg vs market | Samples |", "|---|---|---|---|"]
    lines += [f"| {d['decile']} | {d['hit']}% | {d['avg_ex']:+.2f}% | {d['n']} |" for d in deciles]
    open("comparison.md", "w").write("\n".join(lines) + "\n")
    json.dump(clean({"rules": results, "model": model_report}), open("comparison.json", "w"), indent=1)
    print("\n".join(lines))


# ============================ output ======================================
def clean(o):
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return None if o != o or o in (float("inf"), float("-inf")) else o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, dt.date)):
        return str(o)[:10]
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def send_push(text, title, priority=None):
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        return
    import requests
    h = {"Title": title, "Tags": "chart_with_upwards_trend"}
    if priority:
        h["Priority"] = priority
    if os.getenv("DASHBOARD_URL"):
        h["Click"] = os.environ["DASHBOARD_URL"]
    requests.post(f"https://ntfy.sh/{topic}", data=text.encode("utf-8"), headers=h, timeout=15)


def send_email(text, subject):
    user, pw, to = (os.getenv(k) for k in ("BOT_EMAIL_USER", "BOT_EMAIL_PASS", "BOT_EMAIL_TO"))
    if not (user and pw and to):
        return
    msg = MIMEText(text)
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    with smtplib.SMTP("smtp.office365.com", 587) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, pw)
        s.send_message(msg)


def main_scan(live):
    session = session_label() if live else "morning"
    rows, dates, market_ok, stage, model_info = run_scan(live)
    state = {"date": str(now_ct().date()), "buys": {}, "cancelled": [], "sell_alerts": []} if not live else load_today()
    new, cancelled, sells, status = track_today(state, rows, session)
    book = update_paper(load_paper(), stage, rows, "morning" if not live else "midday", close_trades=not live)
    with open("data.json", "w", encoding="utf-8") as fh:
        json.dump(clean({"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "session": session, "live": live, "market_ok": market_ok,
                         "settings": dict(rsi_max=RSI_MAX, stretch_atr=STRETCH_ATR, market_filter=MARKET_FILTER,
                                          validation=VALIDATION, use_model=USE_MODEL, model_min_tier=MODEL_MIN_TIER,
                                          earnings_days=EARNINGS_DAYS),
                         "model": model_info, "paper": book, "dates": dates, "stocks": rows}),
                  fh, separators=(",", ":"))
    text = message(session, market_ok, new, cancelled, sells, status, book, len(rows))
    print(text)
    urgent = "high" if (sells or cancelled) else None
    title = "Stock signals: morning" if not live else f"Stock update: {session}"
    send_push(text, title, urgent)
    send_email(text, f"{title} {now_ct():%Y-%m-%d}")


if __name__ == "__main__":
    mode = os.getenv("MODE", "morning").lower()
    if mode in ("scan", "intraday"):
        main_scan(live=True)
    elif mode == "compare":
        compare()
    else:
        main_scan(live=False)
