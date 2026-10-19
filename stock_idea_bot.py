#!/usr/bin/env python3
"""
Stock Idea Bot v3 - history-tested buy/sell signals
----------------------------------------------------
Every weekday morning, for each stock on the watchlist:

  1. QUALITY  - checks the business: profitable? growing? not drowning in debt?
  2. BACKTEST - replays the exact buy/sell rule on that stock's last 5 years.
                The rule is tuned on the first ~3.5 years and then checked on the
                last ~1.5 years it never "saw" (out-of-sample). A stock only gets
                a BUY signal if the rule made money in both periods AND handled
                risk better than simply buying and holding.
  3. TODAY    - is the entry setup present right now? Is the news tone OK?
                Is an earnings report coming within a week (gap risk)?
  4. SIGNALS  - BUY / HOLD / SELL / WAIT, with stop-loss and trailing-exit prices.

Results go to data.json (for the dashboard), a phone push via ntfy, and email.

This uses history to estimate odds. It cannot know the future. Not advice.

Setup:  pip install -r requirements.txt
Env vars (all optional): NTFY_TOPIC, DASHBOARD_URL, HOLDINGS="F:11.20,AAPL:180.5",
                         BOT_EMAIL_USER, BOT_EMAIL_PASS, BOT_EMAIL_TO
"""
import datetime as dt
import json
import os
import smtplib
import ssl
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText

import numpy as np
import pandas as pd

# ---- Edit these ----------------------------------------------------------
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "JPM",
    "V", "MA", "UNH", "LLY", "XOM", "CVX", "COST", "WMT", "HD", "NFLX", "CRM",
    "ORCL", "ADBE", "DIS", "BA", "KO", "PEP", "PLTR", "F", "GM", "SOFI", "T", "VZ",
    "PFE", "INTC", "BAC", "WFC", "C", "HOOD", "SNAP", "PINS", "RIVN", "LCID", "NIO",
    "PLUG", "AAL", "CCL", "NCLH", "KEY", "HBAN", "CMCSA", "WBD", "UBER", "SHOP",
]
TOP_N = 5
TEST_SHARE = 0.30        # last 30% of history is the out-of-sample check
STOP_ATR = 2.0           # initial stop = entry - 2 x daily swing
TRAIL_ATR = 3.0          # trailing exit = highest close since entry - 3 x daily swing
EARNINGS_DAYS = 7        # no new BUY if earnings are this close
# --------------------------------------------------------------------------

HOLDINGS = {}
for _p in os.getenv("HOLDINGS", "").split(","):
    if ":" in _p:
        _k, _v = _p.split(":", 1)
        try:
            HOLDINGS[_k.strip().upper()] = float(_v)
        except ValueError:
            pass
WATCHLIST = sorted(set(WATCHLIST) | set(HOLDINGS))


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
    # Entry rule: long-term uptrend, short-term uptrend, positive momentum, not overheated
    out["entry"] = ((c > out["sma50"]) & (out["sma50"] > out["sma200"])
                    & (out["ret5"] > 0) & out["rsi"].between(40, 70))
    return out


# ============================ backtest ====================================
def simulate(ind: pd.DataFrame) -> dict:
    """Replay the rule day by day. Decisions use only data up to that day's close;
    trades fill at that close. Returns daily strategy returns and a trade list."""
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
    strat = np.concatenate([[0.0], pos[:-1] * daily[1:]])  # held yesterday -> earn today
    return dict(strat=strat, hold=daily, trades=trades, open=in_pos,
                open_entry=e_px if in_pos else None, open_stop=stop if in_pos else None,
                open_peak=peak if in_pos else None)


def stats(rets: np.ndarray, trades: list) -> dict:
    eq = np.cumprod(1 + rets)
    dd = float((eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0
    total = float(eq[-1] - 1) if len(eq) else 0.0
    wins = [t for t in trades if t["ret"] > 0]
    return dict(ret=total * 100, maxdd=dd * 100,
                trades=len(trades),
                win=(len(wins) / len(trades) * 100) if trades else None,
                avg=(np.mean([t["ret"] for t in trades]) * 100) if trades else None,
                score=total / abs(dd) if dd < 0 else (total * 10))  # return per unit of pain


def backtest(ind: pd.DataFrame) -> dict:
    sim = simulate(ind)
    n = len(ind)
    cut = int(n * (1 - TEST_SHARE))
    res = {}
    for name, a, b in (("train", 200, cut), ("test", cut, n)):
        tr = [t for t in sim["trades"] if a <= t["entry_i"] < b]
        s = stats(sim["strat"][a:b], tr)
        h = stats(sim["hold"][a:b], [])
        res[name] = dict(strategy=s, hold=h)
    ok = all(res[p]["strategy"]["ret"] > 0
             and res[p]["strategy"]["score"] >= res[p]["hold"]["score"]
             and res[p]["strategy"]["trades"] >= 2
             for p in ("train", "test"))
    res["validated"] = bool(ok)
    res["sim"] = sim
    return res


# ============================ outside data ================================
def fundamentals(tk) -> dict:
    try:
        i = tk.info or {}
    except Exception:
        return {}
    keep = ("profitMargins", "revenueGrowth", "debtToEquity", "forwardPE",
            "shortName", "sector", "marketCap")
    return {k: i.get(k) for k in keep}


def quality(f: dict):
    """Returns (passed: bool|None, notes: list)."""
    if not f or f.get("profitMargins") is None:
        return None, ["fundamentals unavailable"]
    notes, bad = [], 0
    pm, rg, de, pe = (f.get("profitMargins"), f.get("revenueGrowth"),
                      f.get("debtToEquity"), f.get("forwardPE"))
    if pm is not None:
        notes.append(f"profit margin {pm*100:.0f}%")
        bad += pm <= 0
    if rg is not None:
        notes.append(f"revenue growth {rg*100:+.0f}%")
        bad += rg < -0.05
    if de is not None:
        notes.append(f"debt/equity {de:.0f}%")
        bad += de > 200
    if pe is not None and pe > 0:
        notes.append(f"forward P/E {pe:.0f}")
        bad += pe > 80
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


def news(ticker: str, sia):
    import requests
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        items = ET.fromstring(r.content).findall(".//item")[:8]
        heads = [{"title": i.findtext("title", ""), "link": i.findtext("link", "")} for i in items]
    except Exception:
        return 0.0, []
    if not heads:
        return 0.0, []
    s = sum(sia.polarity_scores(h["title"])["compound"] for h in heads) / len(heads)
    return s, heads[:4]


# ============================ main logic ==================================
def evaluate(t, df, f, earn, sent, heads):
    ind = indicators(df)
    last = ind.iloc[-1]
    price, atr = float(last["close"]), float(last["atr"])
    bt = backtest(ind)
    sim = bt["sim"]
    q_ok, q_notes = quality(f)
    today = dt.date.today()
    earn_soon = bool(earn and 0 <= (earn - today).days <= EARNINGS_DAYS)

    # Levels if you bought today
    stop, target = price - STOP_ATR * atr, price + 3 * atr
    trail_now = float(ind["close"].iloc[-22:].max() - TRAIL_ATR * atr)
    exit_now = price < float(last["sma50"]) or price < trail_now

    reasons, blockers = [], []
    if bool(last["entry"]):
        reasons.append("entry setup present today")
    if bt["validated"]:
        reasons.append("rule beat buy-and-hold (risk-adjusted) in both test periods")
    else:
        blockers.append("rule has no proven edge on this stock")
    if q_ok is False:
        blockers.append("weak fundamentals")
    if sent < -0.15:
        blockers.append("negative news tone")
    if earn_soon:
        blockers.append(f"earnings on {earn} (gap risk)")

    if exit_now:
        signal = "sell"
    elif bool(last["entry"]) and not blockers:
        signal = "buy"
    elif sim["open"] and bt["validated"]:
        signal = "hold"
    else:
        signal = "wait"

    # Ranking score: proven edge + quality + news + momentum
    tst = bt["test"]["strategy"]
    score = 0.0
    score += 2.0 if bt["validated"] else 0.0
    score += 1.0 if q_ok else (-1.0 if q_ok is False else 0.0)
    score += 1.5 * sent
    score += 1.0 if bool(last["entry"]) else 0.0
    score += max(-1.0, min(1.0, tst["ret"] / 20))
    score -= 1.0 if earn_soon else 0.0
    score -= 2.0 if exit_now else 0.0

    def pack(p):
        s, h = bt[p]["strategy"], bt[p]["hold"]
        return {"strategy": {k: (round(v, 1) if isinstance(v, float) else v) for k, v in s.items() if k != "score"},
                "hold": {"ret": round(h["ret"], 1), "maxdd": round(h["maxdd"], 1)}}

    dates = ind.index
    recent = [dict(buy=dates[x["entry_i"]].strftime("%Y-%m-%d"),
                   sell=dates[x["exit_i"]].strftime("%Y-%m-%d"),
                   ret=round(x["ret"] * 100, 1)) for x in sim["trades"][-6:]]
    return dict(
        ticker=t, name=f.get("shortName") or t, sector=f.get("sector"),
        price=round(price, 2),
        ret1=round(float(ind["close"].pct_change().iloc[-1] * 100), 2),
        ret5=round(float(last["ret5"] * 100), 2),
        rsi=round(float(last["rsi"]), 1), vol_ratio=round(float(last["vol_ratio"]), 2),
        sma50=round(float(last["sma50"]), 2), atr=round(atr, 2),
        stop=round(stop, 2), target=round(target, 2), trail=round(trail_now, 2),
        signal=signal, score=round(score, 2), reasons=reasons, blockers=blockers,
        quality=q_ok, quality_notes=q_notes, earnings=str(earn) if earn else None,
        sent=round(sent, 2), heads=heads,
        validated=bt["validated"], train=pack("train"), test=pack("test"),
        recent_trades=recent,
        dates=[d.strftime("%Y-%m-%d") for d in dates[-120:]],
        closes=[round(float(x), 2) for x in ind["close"].iloc[-120:]],
    )


def run():
    import yfinance as yf
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    data = yf.download(WATCHLIST, period="5y", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)
    rows = []
    for t in WATCHLIST:
        try:
            df = data[t].dropna()
            if len(df) < 400:
                continue
            tk = yf.Ticker(t)
            sent, heads = news(t, sia)
            rows.append(evaluate(t, df, fundamentals(tk), next_earnings(tk), sent, heads))
        except Exception as e:
            print(f"skip {t}: {e}")
    return sorted(rows, key=lambda r: r["score"], reverse=True)


# ============================ output ======================================
def clean(o):
    if isinstance(o, (float, np.floating)):
        o = float(o)
        return None if o != o or o in (float("inf"), float("-inf")) else o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
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
                         f"({(r['price']/b-1)*100:+.1f}% vs your ${b:g})")
    buys = [r for r in rows if r["signal"] == "buy"][:TOP_N]
    if buys:
        lines.append("BUY setups (history-tested):")
        lines += [f"{r['ticker']} ${r['price']:.2f}, stop ${r['stop']:.2f}" for r in buys]
    else:
        lines.append("No buy setups passed all checks today. Waiting is a valid choice.")
    return "\n".join(lines) + "\n\nOdds from history, not guarantees."


def send_push(text):
    topic = os.getenv("NTFY_TOPIC")
    if not topic:
        return
    import requests
    h = {"Title": "Stock signals today", "Tags": "chart_with_upwards_trend"}
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
    rows = run()
    with open("data.json", "w", encoding="utf-8") as fh:
        json.dump(clean({"generated": dt.datetime.utcnow().isoformat() + "Z",
                         "test_share": TEST_SHARE, "stocks": rows}), fh)
    text = push_text(rows)
    print(text)
    send_push(text)
    send_email(text)
