"""
Data Fetcher — yfinance with ETF fallbacks + period retries.
All fallback params are keyword-only with defaults so signature is always compatible.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=4)

try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    logger.error("yfinance not installed!")


# ─── SYNC HELPERS ────────────────────────────────────────────────────────────

def _try_fetch(ticker: str, period: str, interval: str):
    """Return DataFrame or None — never raises."""
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, auto_adjust=True)
        if hist is not None and not hist.empty and len(hist) >= 20:
            return hist
    except Exception as e:
        logger.debug(f"yfinance {ticker}/{period}: {e}")
    return None


def _sync_history(ticker: str, period: str, interval: str,
                  fallbacks: Dict[str, str] = {}) -> dict:
    """
    Fetch OHLCV history.
    Strategy: primary ticker → shorter periods → ETF fallback → raise.
    """
    period_chain = {"2y": ["2y","1y","6mo","3mo"],
                    "1y": ["1y","6mo","3mo"],
                    "6mo":["6mo","3mo"]}.get(period, [period, "1y", "6mo"])

    hist        = None
    used_ticker = ticker

    # 1. Try primary across period chain
    for p in period_chain:
        hist = _try_fetch(ticker, p, interval)
        if hist is not None:
            break

    # 2. Try ETF fallback if primary failed
    if hist is None and ticker in fallbacks:
        fb = fallbacks[ticker]
        logger.info(f"{ticker} unavailable — trying ETF fallback {fb}")
        used_ticker = fb
        for p in period_chain:
            hist = _try_fetch(fb, p, interval)
            if hist is not None:
                break

    if hist is None:
        raise ValueError(f"No history returned for {ticker} (tried all periods + fallbacks)")

    hist = hist.dropna(subset=["Close"])

    closes = [round(float(x), 6) for x in hist["Close"].tolist()]
    opens  = [round(float(x), 6) for x in hist["Open"].tolist()]
    highs  = [round(float(x), 6) for x in hist["High"].tolist()]
    lows   = [round(float(x), 6) for x in hist["Low"].tolist()]
    vols   = [int(x) for x in hist["Volume"].fillna(0).tolist()]
    dates  = [d.strftime("%Y-%m-%d") for d in hist.index]

    returns = []
    for i in range(1, len(closes)):
        r = (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] else 0.0
        returns.append(round(r, 8))

    return {
        "ticker":   used_ticker,
        "period":   period,
        "interval": interval,
        "dates":    dates,
        "close":    closes,
        "open":     opens,
        "high":     highs,
        "low":      lows,
        "volume":   vols,
        "returns":  returns,
        "count":    len(closes),
    }


def _sync_quote(ticker: str, display_name: str,
                fallbacks: Dict[str, str] = {}) -> dict:
    """Fetch real-time quote with ETF fallback."""
    used_ticker = ticker
    price       = 0.0

    def _get_price(tk):
        t = yf.Ticker(tk)
        i = t.fast_info
        return float(getattr(i, "last_price", 0) or 0), i

    try:
        price, info = _get_price(ticker)
        if price <= 0:
            raise ValueError("zero price")
    except Exception:
        if ticker in fallbacks:
            used_ticker = fallbacks[ticker]
            try:
                price, info = _get_price(used_ticker)
            except Exception as e:
                raise ValueError(f"Both {ticker} and {used_ticker} failed: {e}")
        else:
            # Return placeholder so UI doesn't break
            return _placeholder_quote(display_name, ticker)

    open_  = float(getattr(info, "open",           price) or price)
    high   = float(getattr(info, "day_high",       price) or price)
    low_   = float(getattr(info, "day_low",        price) or price)
    prev   = float(getattr(info, "previous_close", price) or price)
    volume = float(getattr(info, "three_month_average_volume", 0) or 0)
    chg    = price - prev
    chg_pct= (chg / prev * 100) if prev else 0.0

    # Technical indicators from 30d history
    ann_vol = rsi_val = adx_val = sharpe = 0.0
    try:
        h30 = yf.Ticker(used_ticker).history(period="30d", interval="1d", auto_adjust=True)
        if len(h30) >= 5:
            ret     = h30["Close"].pct_change().dropna()
            ann_vol = float(ret.std() * np.sqrt(252) * 100)
            rsi_val = _compute_rsi(h30["Close"].tolist())
            adx_val = _compute_adx(h30)
            sharpe  = _compute_sharpe(ret)
    except Exception:
        pass

    return {
        "symbol":     display_name,
        "ticker":     used_ticker,
        "price":      round(price,   6),
        "open":       round(open_,   6),
        "high":       round(high,    6),
        "low":        round(low_,    6),
        "prev_close": round(prev,    6),
        "change":     round(chg,     6),
        "change_pct": round(chg_pct, 4),
        "volume":     int(volume),
        "ann_vol":    round(ann_vol,  2),
        "rsi":        round(rsi_val,  2),
        "adx":        round(adx_val,  2),
        "sharpe":     round(sharpe,   3),
        "timestamp":  datetime.utcnow().isoformat(),
    }


def _placeholder_quote(display_name: str, ticker: str) -> dict:
    """Return a zero quote when data is unavailable — prevents 500 errors."""
    return {
        "symbol": display_name, "ticker": ticker,
        "price": 0.0, "open": 0.0, "high": 0.0, "low": 0.0,
        "prev_close": 0.0, "change": 0.0, "change_pct": 0.0,
        "volume": 0, "ann_vol": 0.0, "rsi": 50.0, "adx": 0.0, "sharpe": 0.0,
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Data temporarily unavailable",
    }


# ─── ASYNC CLASS ─────────────────────────────────────────────────────────────

class DataFetcher:

    async def get_quote(self, ticker: str, display_name: str,
                        fallbacks: Dict[str, str] = {}) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, _sync_quote, ticker, display_name, fallbacks)

    async def get_history(self, ticker: str, period: str, interval: str,
                          fallbacks: Dict[str, str] = {}) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor, _sync_history, ticker, period, interval, fallbacks)

    def future_dates(self, last_date_str: str, steps: int) -> List[str]:
        try:
            last = datetime.strptime(last_date_str, "%Y-%m-%d")
        except Exception:
            last = datetime.utcnow()
        dates, d = [], last
        while len(dates) < steps:
            d += timedelta(days=1)
            if d.weekday() < 5:
                dates.append(d.strftime("%Y-%m-%d"))
        return dates


# ─── TECHNICAL INDICATORS ────────────────────────────────────────────────────

def _compute_rsi(closes: list, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0)); losses.append(max(-d, 0))
    ag = np.mean(gains[-period:])
    al = np.mean(losses[-period:])
    return float(100 - 100 / (1 + ag / (al + 1e-9)))


def _compute_adx(hist, period: int = 14) -> float:
    try:
        hi, lo, cl = hist["High"].values, hist["Low"].values, hist["Close"].values
        n = len(cl)
        if n < period + 2:
            return 25.0
        tr_l, pdm_l, ndm_l = [], [], []
        for i in range(1, n):
            tr  = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
            pdm = max(hi[i]-hi[i-1], 0); ndm = max(lo[i-1]-lo[i], 0)
            if pdm > ndm:   ndm = 0
            elif ndm > pdm: pdm = 0
            else:           pdm = ndm = 0
            tr_l.append(tr); pdm_l.append(pdm); ndm_l.append(ndm)
        atr  = np.mean(tr_l[-period:])
        apdi = np.mean(pdm_l[-period:]) / (atr + 1e-9) * 100
        andi = np.mean(ndm_l[-period:]) / (atr + 1e-9) * 100
        return float(np.clip(abs(apdi-andi) / (apdi+andi+1e-9) * 100, 0, 100))
    except Exception:
        return 25.0


def _compute_sharpe(returns, rf_annual: float = 0.05) -> float:
    r = np.array(returns, dtype=float)
    if r.std() < 1e-9:
        return 0.0
    return float((r.mean() - rf_annual/252) / r.std() * np.sqrt(252))
