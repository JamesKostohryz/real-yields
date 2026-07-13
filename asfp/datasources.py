"""
Data fetchers for the ASFP tool. These run in the automated runner (GitHub
Actions), where normal networking and the FRED API key are available.

Sources
-------
* GSW nominal Svensson params : feds200628.csv  (weekly, public)
* GSW real Svensson params    : feds200805.csv  (weekly, public)
* Cleveland Fed expected infl : FRED EXPINF{h}YR (monthly, needs API key)
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import requests

GSW_NOMINAL_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
GSW_REAL_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"
FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

_PARAMS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]


def fetch_text(url, timeout=90):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "asfp-tool/1.0"})
    r.raise_for_status()
    return r.text


def _parse_gsw(text):
    """Return (DataFrame, date_column_name) from a GSW CSV, skipping the
    descriptive preamble and locating the true header row."""
    lines = text.splitlines()
    hdr = None
    for i, l in enumerate(lines):
        up = l.upper()
        if "BETA0" in up and "DATE" in up:
            hdr = i
            break
    if hdr is None:                       # fallback: first line starting "Date,"
        for i, l in enumerate(lines):
            if l.strip().lower().startswith("date,"):
                hdr = i
                break
    if hdr is None:
        raise ValueError("Could not locate header row in GSW CSV")
    df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])))
    df.columns = [c.strip() for c in df.columns]
    datecol = df.columns[0]
    df[datecol] = pd.to_datetime(df[datecol], errors="coerce")
    df = df.dropna(subset=[datecol]).sort_values(datecol)
    return df, datecol


def latest_gsw_params(text):
    """Most recent row's Svensson parameters as a dict, plus the as-of date."""
    df, datecol = _parse_gsw(text)
    missing = [p for p in _PARAMS if p not in df.columns]
    if missing:
        raise ValueError(f"GSW file missing parameter columns: {missing}")
    sub = df.dropna(subset=_PARAMS)
    if sub.empty:
        raise ValueError("No rows with complete Svensson parameters")
    row = sub.iloc[-1]
    return dict(
        date=row[datecol].date().isoformat(),
        b0=float(row["BETA0"]), b1=float(row["BETA1"]),
        b2=float(row["BETA2"]), b3=float(row["BETA3"]),
        t1=float(row["TAU1"]), t2=float(row["TAU2"]),
    )


# daily constant-maturity series used to roll the weekly GSW curves to "today"
DGS_MAP = {1: "DGS1", 2: "DGS2", 3: "DGS3", 5: "DGS5", 7: "DGS7",
           10: "DGS10", 20: "DGS20", 30: "DGS30"}          # nominal
DFII_MAP = {5: "DFII5", 7: "DFII7", 10: "DFII10", 20: "DFII20", 30: "DFII30"}  # real


def fetch_fred_latest(api_key, series_id, timeout=60):
    """Latest non-missing observation of a FRED series -> (value, date)."""
    params = dict(series_id=series_id, api_key=api_key, file_type="json",
                  sort_order="desc", limit=1)
    r = requests.get(FRED_OBS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    for o in r.json().get("observations", []):
        v = o.get("value", ".")
        if v not in (".", "", None):
            return float(v), o.get("date")
    return None, None


def fetch_fred_asof(api_key, series_id, asof_date, timeout=60):
    """Latest non-missing observation on or before `asof_date` -> (value, date)."""
    params = dict(series_id=series_id, api_key=api_key, file_type="json",
                  sort_order="desc", limit=1, observation_end=asof_date)
    r = requests.get(FRED_OBS_URL, params=params, timeout=timeout)
    r.raise_for_status()
    for o in r.json().get("observations", []):
        v = o.get("value", ".")
        if v not in (".", "", None):
            return float(v), o.get("date")
    return None, None


def fetch_curve_delta(api_key, series_map, since_date):
    """Daily change in each series since `since_date` (the GSW as-of date).
    Returns (maturities, deltas, latest_date) for the points that have data."""
    mats, deltas, latest = [], [], None
    for m, sid in series_map.items():
        cur, cdate = fetch_fred_latest(api_key, sid)
        old, _ = fetch_fred_asof(api_key, sid, since_date)
        if cur is None or old is None:
            continue
        mats.append(float(m))
        deltas.append(cur - old)
        if cdate and (latest is None or cdate > latest):
            latest = cdate
    return mats, deltas, latest


def fetch_expinf(api_key, timeout=60):
    """Cleveland Fed expected inflation from FRED, interpolated onto years 1..30.

    Returns (values_1_to_30 ndarray, as_of_date_str).
    """
    vals, dates = {}, []
    for h in range(1, 31):
        params = dict(series_id=f"EXPINF{h}YR", api_key=api_key,
                      file_type="json", sort_order="desc", limit=1)
        try:
            r = requests.get(FRED_OBS_URL, params=params, timeout=timeout)
            if r.status_code != 200:
                continue
            obs = r.json().get("observations", [])
        except Exception:
            continue
        if not obs:
            continue
        v = obs[0].get("value", ".")
        if v in (".", "", None):
            continue
        vals[h] = float(v)
        dates.append(obs[0].get("date"))
    if not vals:
        raise RuntimeError("No EXPINF observations returned from FRED — check the API key")
    hs = np.array(sorted(vals))
    ys = np.array([vals[h] for h in hs])
    grid = np.arange(1, 31)
    interp = np.interp(grid, hs, ys)
    as_of = max(d for d in dates if d) if dates else None
    return interp, as_of
