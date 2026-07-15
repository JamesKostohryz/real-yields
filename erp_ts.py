"""
Pull the issuer bond block (and the ticker) from a shared Google Sheet.

The user keeps ONE tab: a TICKER cell at the top and a pasted TradingView bond
block below. The sheet is shared "anyone with the link can view", so the job can
fetch it with no credentials — the same pull pattern as FRED/Yahoo.

Two public endpoints are tried, in order:
  1. /export?format=csv   — raw stored values (full precision, decimal fractions)
  2. /gviz/tq?tqx=out:csv — formatted values (fallback; parser handles percents)
"""
from __future__ import annotations

import csv
import io
import pandas as pd

from . import debt_analytics as da


def csv_to_df(text):
    """Parse CSV text into a positional string DataFrame, tolerant of ragged rows
    (a TICKER row narrower than the bond block) and quoted percent fields."""
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(str(c).strip() for c in r)]  # drop blank lines
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(rows, dtype=str)


def _csv_urls(sheet_id, gid=None):
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}"
    g = f"&gid={gid}" if gid not in (None, "") else ""
    g2 = f"?gid={gid}" if gid not in (None, "") else ""
    return [
        f"{base}/export?format=csv{g2}",
        f"{base}/gviz/tq?tqx=out:csv{g}",
    ]


def fetch_csv_text(sheet_id, gid=None, timeout=30):
    """Return the raw CSV text of the tab, or None if unreachable."""
    import requests
    for url in _csv_urls(sheet_id, gid):
        try:
            r = requests.get(url, timeout=timeout, allow_redirects=True)
            if r.status_code == 200 and r.text and r.text.strip():
                # a private sheet returns an HTML sign-in page, not CSV
                if "<html" in r.text[:200].lower():
                    continue
                return r.text
        except Exception:
            continue
    return None


def read_ticker(raw):
    """Find a 'TICKER' label in the top-left block and return the value beside it."""
    nr = min(len(raw), 12)
    nc = min(raw.shape[1], 6)
    for i in range(nr):
        for j in range(nc):
            v = str(raw.iat[i, j]).strip().lower().rstrip(":")
            if v == "ticker":
                for k in range(j + 1, raw.shape[1]):
                    w = str(raw.iat[i, k]).strip()
                    if w and w.lower() != "nan":
                        return w.upper()
    return None


def bonds_and_ticker(sheet_id, gid=None):
    """Fetch the sheet and return (bonds_DataFrame_or_None, ticker_or_None)."""
    text = fetch_csv_text(sheet_id, gid)
    if not text:
        return None, None
    raw = csv_to_df(text)
    if raw.empty:
        return None, None
    ticker = read_ticker(raw)
    bonds = da.parse_tradingview_bonds(raw)
    if bonds is None or len(bonds) == 0:
        bonds = None
    return bonds, ticker
