import json, datetime as dt
import numpy as np, pandas as pd, yfinance as yf

LOOK = 3      # عدد أيام البحث عن إشارة
PIVOT = 10

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1/n, adjust=False).mean() / dn.ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + rs)

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High-df.Low, (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def pivots(df):
    lo, hi = df.Low.values, df.High.values
    n = len(df); LL = np.full(n, np.nan); LH = np.full(n, np.nan)
    ll = lh = np.nan
    for i in range(n):
        j = i - PIVOT
        if j >= PIVOT:
            if lo[j] <= lo[j-PIVOT:j+PIVOT+1].min(): ll = lo[j]
            if hi[j] >= hi[j-PIVOT:j+PIVOT+1].max(): lh = hi[j]
        LL[i], LH[i] = ll, lh
    return LL, LH

def analyze(df):
    if len(df) < 210: return None
    c = df.Close
    e50, e200 = ema(c, 50), ema(c, 200)
    macd = ema(c, 12) - ema(c, 26); sig = ema(macd, 9)
    r, a = rsi(c), atr(df)
    vol_ok = df.Volume > df.Volume.rolling(20).mean() * 1.2
    LL, LH = pivots(df)
    n = len(df)
    for i in range(n-1, max(n-LOOK, 1)-1, -1):
        ok = (macd.iloc[i] > sig.iloc[i] and macd.iloc[i-1] <= sig.iloc[i-1]
              and c.iloc[i] > e50.iloc[i] > e200.iloc[i]
              and 50 < r.iloc[i] < 70 and vol_ok.iloc[i])
        if not ok: continue
        en = float(c.iloc[i])
        stop = max(LL[i], en-2.5*a.iloc[i]) if (not np.isnan(LL[i]) and LL[i] < en) else en-2*a.iloc[i]
        risk = en - stop
        t1 = en + 2*risk
        if not np.isnan(LH[i]) and LH[i] > en:
            base = stop if np.isnan(LL[i]) else LL[i]
            t2 = LH[i] + 0.618*(LH[i]-base)
        else:
            t2 = en + 3*risk
        return dict(date=str(df.index[i].date()), entry=en, stop=float(stop), t1=float(t1),
                    t2=float(t2), rsi=float(r.iloc[i]), last=float(c.iloc[-1]))
    return None

def main():
    items = []
    for line in open("tickers.txt", encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            code, _, name = line.partition(",")
            items.append((code.strip(), name.strip() or code.strip()))
    tick = [f"{c}.SR" for c, _ in items]
    data = yf.download(tick, period="2y", interval="1d", group_by="ticker",
                       auto_adjust=False, progress=False, threads=True)
    signals, failed = [], []
    for (code, name), t in zip(items, tick):
        try:
            df = data[t].dropna(subset=["Close"])
            res = analyze(df)
            if res: signals.append(dict(code=code, name=name, **res))
        except Exception:
            failed.append(code)
    signals.sort(key=lambda s: s["date"], reverse=True)
    now = dt.datetime.now(dt.timezone(dt.timedelta(hours=3)))
    out = dict(updated=now.strftime("%Y-%m-%d %H:%M"), scanned=len(items),
               failed=failed, signals=signals)
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"scanned={len(items)} signals={len(signals)} failed={failed}")

if __name__ == "__main__":
    main()
