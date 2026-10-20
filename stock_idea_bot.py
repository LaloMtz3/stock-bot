#!/usr/bin/env python3
"""
Stock Signals Bot v4
--------------------
MODE=morning  (default, weekdays ~8am Central)
  Scans the S&P 500 + S&P 400 (about 900 US companies, $5+ stocks):
    1. BACKTEST - replays the buy/sell rule on each stock's last 5 years. Tuned on the
                  first ~3.5 years, checked on the last ~1.5 years it never "saw".
    2. SHORTLIST - stocks that look promising get a closer look:
    3. QUALITY  - profitable? growing? not drowning in debt?
    4. TODAY    - news tone, earnings within a week?
    5. SIGNALS  - BUY / HOLD / SELL / WAIT with stop-loss and trailing-exit prices.
  Writes data.json for the dashboard and sends a phone notification.

MODE=intraday  (9am, 12pm and 2pm Central)
  Checks only the stocks in HOLDINGS and sends a SELL alert if one has crossed its
  stop-loss, trailing exit or 50-day average. Each stock alerts at most once per day.

History is used to estimate odds. It cannot know the future. Not financial advice.

Env vars (all optional): NTFY_TOPIC, DASHBOARD_URL, HOLDINGS="F:11.20,AAPL:180.5",
                         BOT_EMAIL_USER, BOT_EMAIL_PASS, BOT_EMAIL_TO, MODE
"""
import datetime as dt
import io
import json
import os
import smtplib
import ssl
import time
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---- Settings you can change ---------------------------------------------
INDEX_PAGES = [  # stock lists pulled fresh each morning
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
]
EXTRA_TICKERS = []        # add any others you want scanned, e.g. ["SOFI", "HOOD"]
MIN_PRICE = 5.0           # skip stocks under this price (holdings are always checked)
SHORTLIST = 60            # how many top stocks get the news + business check
TOP_N = 5                 # buy ideas listed in the notification
TEST_SHARE = 0.30         # last 30% of history is the unseen test
STOP_ATR = 2.0            # stop-loss = buy price - 2 x normal daily swing
TRAIL_ATR = 3.0           # trailing exit = recent high - 3 x normal daily swing
EARNINGS_DAYS = 7         # no new BUY if earnings are this close
CHART_DAYS = 120          # days of price shown on the dashboard chart
# Used only if the stock lists can't be downloaded:
FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM",
    "V", "MA", "UNH", "LLY", "XOM", "CVX", "COST", "WMT", "HD", "NFLX", "CRM",
    "ORCL", "ADBE", "DIS", "BA", "KO", "PEP", "PLTR", "F", "GM", "SOFI", "T", "VZ",
    "PFE", "INTC", "BAC", "WFC", "C", "HOOD", "SNAP", "PINS", "RIVN", "LCID", "NIO",
    "PLUG", "AAL", "CCL", "NCLH", "KEY", "HBAN", "CMCSA", "WBD", "UBER", "SHOP",
]
# --------------------------------------------------------------------------

UA = {"User-Agent": "Mozilla/5.0 (stock-signals-bot)"}
CT = ZoneInfo("America/Chicago")

HOLDINGS = {}
for _p in os.getenv("HOLDINGS", "").split(","):
    if ":" in _p:
        _k, _v = _p.split(":", 1)
        try:
            HOLDINGS[_k.strip().upper().replace(".", "-")] = float(_v)
        except ValueError:
            pass


# ============================ stock list ==================================
def universe() -> dict:
    """Returns {ticker: {"name":..., "sector":...}}."""
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


def download(tickers, period="5y", chunk=100):
    """Returns {ticker: DataFrame} downloaded in batches to stay polite to Yahoo."""
    import yfinance as yf
    frames, tickers = {}, list(tickers)
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
                if len(df):
                    frames[t] = df
            except Exception:
                pass
        print(f"downloaded {min(i + chunk, len(tickers))}/{len(tickers)}")
        time.sleep(2)
    return frames


# ============================ indicators ==================================
def indicators(df: pd.DataFrame) -> pd.DataFrame:
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
    # Buy rule: long-term uptrend, short-term uptrend, positive momentum, not overheated
    out["entry"] = ((c > out["sma50"]) & (out["sma50"] > out["sma200"])
                    & (out["ret5"] > 0) & out["rsi"].between(40, 70))
    return out


# ============================ backtest ====================================
def simulate(ind: pd.DataFrame) -> dict:
    """Replay the rule day by day using only data known at each day's close."""
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
            trail = peak - TRAIL_ATR * atr[i]
            if c[i] < max(stop, trail) or c[i] < sma50[i]:
                trades.append(dict(entry_i=e_i, exit_i=i, ret=c[i] / e_px - 1))
                in_pos = False
        elif entry[i]:
            in_pos, e_px, e_i, peak = True, c[i], i, c[i]
            stop = c[i] - STOP_ATR * atr[i]
        pos[i] = 1.0 if in_pos else 0.0
    daily = pd.Series(c).pct_change().fillna(0).values
    strat = np.concatenate([[0.0], pos[:-1] * daily[1:]])
    return dict(strat=strat, hold=daily, trades=trades, open=in_pos)


def stats(rets, trades):
    eq = np.cumprod(1 + rets)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0
    total = float(eq[-1] - 1) if len(eq) else 0.0
    wins = [t for t in trades if t["ret"] > 0]
    return dict(ret=total * 100, maxdd=dd * 100, trades=len(trades),
                win=(len(wins) / len(trades) * 100) if trades else None,
                avg=float(np.mean([t["ret"] for t in trades]) * 100) if trades else None,
                score=total / abs(dd) if dd < 0 else total * 10)


def backtest(ind):
    sim = simulate(ind)
    n = len(ind)
    cut = int(n * (1 - TEST_SHARE))
    res = {}
    for name, a, b in (("train", 200, cut), ("test", cut, n)):
        tr = [t for t in sim["trades"] if a <= t["entry_i"] < b]
        res[name] = dict(strategy=stats(sim["strat"][a:b], tr), hold=stats(sim["hold"][a:b], []))
    res["validated"] = all(res[p]["strategy"]["ret"] > 0
                           and res[p]["strategy"]["score"] >= res[p]["hold"]["score"]
                           and res[p]["strategy"]["trades"] >= 2 for p in ("train", "test"))
    res["sim"] = sim
    return res


# ============================ outside data ================================
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
            return pd.Timestamp(d).date()
    except Exception:
        pass
    return None


def news(ticker, sia):
    import requests
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, timeout=10, headers=UA)
        items = ET.fromstring(r.content).findall(".//item")[:8]
        heads = [{"title": i.findtext("title", ""), "link": i.findtext("link", "")} for i in items]
    except Exception:
        return 0.0, []
    if not heads:
        return 0.0, []
    return sum(sia.polarity_scores(h["title"])["compound"] for h in heads) / len(heads), heads[:4]


# ============================ signals =====================================
def evaluate(t, meta, ind, bt, f=None, earn=None, sent=0.0, heads=None, checked=False):
    last = ind.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])
    sim = bt["sim"]
    q_ok, q_notes = quality(f) if checked else (None, ["not on today's shortlist, so not checked"])
    earn_soon = bool(earn and 0 <= (earn - dt.datetime.now(CT).date()).days <= EARNINGS_DAYS)
    stop, target = price - STOP_ATR * atr, price + 3 * atr
    trail = float(ind["close"].iloc[-22:].max() - TRAIL_ATR * atr)
    exit_now = price < float(last["sma50"]) or price < trail

    reasons, blockers = [], []
    if bool(last["entry"]):
        reasons.append("entry setup present today")
    if not bt["validated"]:
        blockers.append("rule has no proven edge on this stock")
    if q_ok is False:
        blockers.append("weak fundamentals")
    if sent < -0.15:
        blockers.append("negative news tone")
    if earn_soon:
        blockers.append(f"earnings on {earn} (gap risk)")
    if not checked and bool(last["entry"]) and bt["validated"]:
        blockers.append("not checked today")

    if exit_now:
        signal = "sell"
    elif bool(last["entry"]) and not blockers:
        signal = "buy"
    elif sim["open"] and bt["validated"]:
        signal = "hold"
    else:
        signal = "wait"

    tst = bt["test"]["strategy"]
    score = (2.0 if bt["validated"] else 0.0) \
        + (1.0 if q_ok else (-1.0 if q_ok is False else 0.0)) \
        + 1.5 * sent + (1.0 if bool(last["entry"]) else 0.0) \
        + max(-1.0, min(1.0, tst["ret"] / 20)) \
        - (1.0 if earn_soon else 0.0) - (2.0 if exit_now else 0.0)

    def pack(p):
        s, h = bt[p]["strategy"], bt[p]["hold"]
        return {"strategy": {k: (round(v, 1) if isinstance(v, float) else v)
                             for k, v in s.items() if k != "score"},
                "hold": {"ret": round(h["ret"], 1), "maxdd": round(h["maxdd"], 1)}}

    dates = ind.index
    return dict(
        ticker=t, name=(f or {}).get("shortName") or meta.get("name") or t,
        sector=(f or {}).get("sector") or meta.get("sector"),
        price=round(price, 2),
        ret1=round(float(ind["close"].pct_change().iloc[-1] * 100), 2),
        ret5=round(float(last["ret5"] * 100), 2),
        rsi=round(float(last["rsi"]), 1), vol_ratio=round(float(last["vol_ratio"]), 2),
        sma50=round(float(last["sma50"]), 2), atr=round(atr, 2),
        stop=round(stop, 2), target=round(target, 2), trail=round(trail, 2),
        signal=signal, score=round(score, 2), reasons=reasons, blockers=blockers,
        checked=checked, quality=q_ok, quality_notes=q_notes,
        earnings=str(earn) if earn else None, sent=round(sent, 2), heads=heads or [],
        validated=bt["validated"], train=pack("train"), test=pack("test"),
        recent_trades=[dict(buy=dates[x["entry_i"]].strftime("%Y-%m-%d"),
                            sell=dates[x["exit_i"]].strftime("%Y-%m-%d"),
                            ret=round(x["ret"] * 100, 1)) for x in sim["trades"][-4:]],
    )


def morning():
    import yfinance as yf
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    uni = universe()
    print(f"universe: {len(uni)} stocks")
    frames = download(uni)

    # Stage 1: price rules + 5-year backtest for everything
    stage = {}
    for t, df in frames.items():
        try:
            if len(df) < 400:
                continue
            if float(df["Close"].iloc[-1]) < MIN_PRICE and t not in HOLDINGS:
                continue
            ind = indicators(df)
            stage[t] = (ind, backtest(ind))
        except Exception as e:
            print(f"skip {t}: {e}")
    print(f"backtested {len(stage)} stocks")

    # Stage 2: closer look only at the shortlist (keeps Yahoo happy)
    prelim = {t: evaluate(t, uni.get(t, {}), ind, bt) for t, (ind, bt) in stage.items()}
    must = {t for t, r in prelim.items()
            if t in HOLDINGS or (r["reasons"] and r["validated"] and r["signal"] != "sell")}
    ranked = [t for t, _ in sorted(prelim.items(), key=lambda kv: kv[1]["score"], reverse=True)]
    must = [t for t in ranked if t in HOLDINGS] + [t for t in ranked if t in must and t not in HOLDINGS][:SHORTLIST * 2]
    shortlist = must + [t for t in ranked if t not in must][:max(0, SHORTLIST - len(must))]
    print(f"shortlist: {len(shortlist)} stocks")
    sia = SentimentIntensityAnalyzer()
    rows = dict(prelim)
    for t in shortlist:
        try:
            tk = yf.Ticker(t)
            sent, heads = news(t, sia)
            ind, bt = stage[t]
            rows[t] = evaluate(t, uni.get(t, {}), ind, bt, fundamentals(tk),
                               next_earnings(tk), sent, heads, checked=True)
            time.sleep(0.3)
        except Exception as e:
            print(f"closer look failed for {t}: {e}")
    rows = sorted(rows.values(), key=lambda r: r["score"], reverse=True)

    # Chart data on one shared calendar to keep the file small
    cal = None
    for t, (ind, _) in stage.items():
        if cal is None or ind.index[-1] > cal[-1] or len(ind.index) > len(cal):
            cal = ind.index
    cal = cal[-CHART_DAYS:]
    for r in rows:
        s = stage[r["ticker"]][0]["close"].reindex(cal).ffill()
        r["closes"] = [None if pd.isna(x) else round(float(x), 2) for x in s]
    return rows, [d.strftime("%Y-%m-%d") for d in cal]


# ============================ intraday checks =============================
STATE_FILE = "alerts_state.json"


def intraday():
    if not HOLDINGS:
        print("No HOLDINGS set, nothing to check. Add a HOLDINGS secret like F:11.20,AAPL:180.50")
        return
    today = dt.datetime.now(CT).date()
    try:
        state = json.load(open(STATE_FILE))
    except Exception:
        state = {}
    if state.get("date") != str(today):
        state = {"date": str(today), "sent": []}
    frames = download(HOLDINGS, period="1y")
    alerts = []
    for t, buy in HOLDINGS.items():
        df = frames.get(t)
        if df is None or len(df) < 60:
            print(f"{t}: no data")
            continue
        if df.index[-1].date() != today:
            print(f"{t}: no trading today (market closed?)")
            continue
        ind = indicators(df)
        price = float(ind["close"].iloc[-1])
        atr = float(ind["atr"].iloc[-2])           # yesterday's full-day swing
        sma50 = float(ind["sma50"].iloc[-2])
        stop = buy - STOP_ATR * atr
        trail = float(ind["close"].iloc[-22:].max() - TRAIL_ATR * atr)
        why = []
        if price < stop:
            why.append(f"below your stop-loss ${stop:.2f}")
        if price < trail:
            why.append(f"below trailing exit ${trail:.2f}")
        if price < sma50:
            why.append(f"below 50-day average ${sma50:.2f}")
        print(f"{t}: ${price:.2f} (bought ${buy:g}) {'; '.join(why) or 'OK'}")
        if why and t not in state["sent"]:
            alerts.append(f"SELL {t} ${price:.2f} ({(price / buy - 1) * 100:+.1f}% vs your ${buy:g}): "
                          + ", ".join(why))
            state["sent"].append(t)
    json.dump(state, open(STATE_FILE, "w"))
    if alerts:
        text = "\n".join(alerts) + "\n\nPrices can be ~15 min delayed. Check your brokerage before acting."
        print(text)
        send_push(text, title="SELL alert", priority="high")


# ============================ output ======================================
def clean(o):
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return None if o != o or o in (float("inf"), float("-inf")) else o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def push_text(rows):
    lines = []
    for r in rows:
        b = HOLDINGS.get(r["ticker"])
        if b and (r["signal"] == "sell" or r["price"] < b - STOP_ATR * r["atr"]):
            lines.append(f"SELL ALERT {r['ticker']} ${r['price']:.2f} "
                         f"({(r['price'] / b - 1) * 100:+.1f}% vs your ${b:g})")
    buys = [r for r in rows if r["signal"] == "buy"]
    if buys:
        lines.append(f"{len(buys)} BUY setups today. Top {min(TOP_N, len(buys))}:")
        lines += [f"{r['ticker']} ${r['price']:.2f}, stop ${r['stop']:.2f}" for r in buys[:TOP_N]]
    else:
        lines.append("No buy setups passed all checks today. Waiting is a valid choice.")
    return "\n".join(lines) + f"\n\nScanned {len(rows)} stocks. Odds from history, not guarantees."


def send_push(text, title="Stock signals today", priority=None):
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


def send_email(text):
    user, pw, to = (os.getenv(k) for k in ("BOT_EMAIL_USER", "BOT_EMAIL_PASS", "BOT_EMAIL_TO"))
    if not (user and pw and to):
        return
    msg = MIMEText(text)
    msg["Subject"] = f"Stock signals {dt.date.today().isoformat()}"
    msg["From"], msg["To"] = user, to
    with smtplib.SMTP("smtp.office365.com", 587) as s:
        s.starttls(context=ssl.create_default_context())
        s.login(user, pw)
        s.send_message(msg)


if __name__ == "__main__":
    if os.getenv("MODE", "morning").lower() == "intraday":
        intraday()
    else:
        rows, dates = morning()
        with open("data.json", "w", encoding="utf-8") as fh:
            json.dump(clean({"generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "dates": dates, "stocks": rows}), fh, separators=(",", ":"))
        text = push_text(rows)
        print(text)
        send_push(text)
        send_email(text)
