#!/usr/bin/env python3
"""
refresh_bonds.py
====================================================================
Regenerate  bonds/<TICKER>.csv  automatically from the EODHD API,
in the EXACT 10-column TradingView format the existing pipeline
(asfp/debt_analytics.parse_tradingview_bonds) already reads.

This RETIRES the manual TradingView copy-paste. Nothing downstream
changes: run_company.py still reads bonds/<T>.csv, fits the modal
rating + multiplicative offset, and writes cod_<T>_annual.csv exactly
as before — it just now reads a file that refreshed itself.

It also does one thing the manual paste never did reliably: it drops
the spun-off media debt (Time Warner / DIRECTV / AOL / WarnerMedia)
BY NAME, so AT&T's credit curve is fit only on AT&T's own obligations
instead of leaning on a statistical outlier filter to remove them.

Pipeline per issuer:
  1. enumerate the issuer's USD bonds from EODHD Search (no hand list),
  2. keep only the family's own debt (include / exclude name rules),
  3. pull each bond's recent yield-to-worst + price (robust MEDIAN of
     the last few daily prints — these bonds are matrix-priced so a
     single tick is noisy),
  4. read coupon + maturity out of the bond name,
  5. write bonds/<TICKER>.csv in the 10-column format.

Columns written (matched to parse_tradingview_bonds, which reads by
name-prefix and is unit-agnostic via its _num/parse_amount helpers):
  Symbol, YTW %, Price %, Coupon %, Maturity date, Outstanding amt,
  Face value, S&P rating, Fitch rating, Issuer
  - YTW %, Price %, Coupon % are DECIMAL FRACTIONS (4.35% -> 0.0435),
    matching the existing committed files.
  - S&P rating is set to the issuer's configured rating so the
    downstream modal-rating picks it up (EODHD Search carries no
    per-bond rating).
  - Outstanding amt is left blank (EODHD Search carries no amount);
    the cost-of-debt fit does not use it, and portfolio_summary
    tolerates the blank (headline market-value-of-debt just reads nan).

Runs headless in GitHub Actions. stdlib + (optional) nothing else.
EODHD key from env EODHD_API_KEY (a GitHub Actions secret). NEVER
committed. Self-test:  python refresh_bonds.py --self-test  (no key).
====================================================================
"""
import os
import re
import sys
import csv
import json
import statistics
import datetime as _dt
from urllib.request import urlopen
from urllib.parse import quote

EODHD_BASE = "https://eodhd.com/api"
YTW_WINDOW_DAYS = 7
MIN_YEARS, MAX_YEARS = 0.4, 40.0

# ---- issuer families -------------------------------------------------------
ISSUERS = {
    "T": {
        "name": "AT&T Inc.",
        "rating": "BBB",
        "search_queries": ["AT&T", "BellSouth", "SBC Communications",
                            "Ameritech", "Cingular"],
        "include_substrings": ["AT&T", "AT & T", "BELLSOUTH", "SBC ",
                               "SBC COMM", "AMERITECH", "CINGULAR",
                               "PACIFIC BELL", "SOUTHWESTERN BELL"],
        "exclude_substrings": ["TIME WARNER", "WARNERMEDIA", "WARNER MEDIA",
                               "AOL", "DIRECTV", "DISCOVERY", "WBD",
                               "WARNER BROS"],
        "currency": "USD",
    },
    # add more issuers here, same shape.
}

# ---- EODHD REST (thin; shapes proven against the live MCP server) ----------
def _get_json(url):
    with urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def eodhd_search_bonds(query, api_key, limit=500):
    url = (f"{EODHD_BASE}/search/{quote(query)}"
           f"?api_token={api_key}&bonds_only=1&type=bond&limit={limit}&fmt=json")
    try:
        data = _get_json(url)
    except Exception as e:                                   # noqa: BLE001
        print(f"    [WARN] search '{query}' failed: {e}")
        return []
    return data if isinstance(data, list) else []


def eodhd_bond_prints(isin, api_key, days=YTW_WINDOW_DAYS):
    start = (_dt.date.today() - _dt.timedelta(days=days * 3)).isoformat()
    url = (f"{EODHD_BASE}/eod/{isin}.BOND"
           f"?api_token={api_key}&from={start}&order=d&fmt=json")
    try:
        rows = _get_json(url)
    except Exception as e:                                   # noqa: BLE001
        print(f"    [WARN] price pull {isin} failed: {e}")
        return []
    return rows if isinstance(rows, list) else []


# ---- parsing / filtering ---------------------------------------------------
_MAT_RE = re.compile(r"(\d{1,2})([A-Za-z]{3})(\d{4})")
_COUPON_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}


def parse_maturity(name, as_of):
    if not name:
        return None
    m = _MAT_RE.search(name.replace(" ", ""))
    if not m:
        return None
    d, mon, y = int(m.group(1)), m.group(2).title(), int(m.group(3))
    if mon not in _MONTHS:
        return None
    try:
        return _dt.date(y, _MONTHS[mon], d)
    except ValueError:
        return None


def parse_coupon(name):
    m = _COUPON_RE.search(name or "")
    return float(m.group(1)) / 100.0 if m else None


def keep_bond(name, cfg):
    up = (name or "").upper()
    if any(x in up for x in cfg["exclude_substrings"]):
        return False, "excluded (spun-off/unrelated)"
    if not any(x in up for x in cfg["include_substrings"]):
        return False, "no include match"
    return True, "included"


def robust_prints(rows):
    """Return (median_ytw_frac, latest_price_frac, n) from recent daily rows."""
    ys, latest_price = [], None
    for r in rows:
        y = r.get("yield")
        if y not in (None, "", 0):
            try:
                ys.append(float(y))
            except (TypeError, ValueError):
                pass
        if latest_price is None:
            p = r.get("price")
            if p not in (None, ""):
                try:
                    latest_price = float(p)
                except (TypeError, ValueError):
                    pass
    if not ys:
        return None, None, 0
    return statistics.median(ys) / 100.0, latest_price, len(ys)


# ---- build the CSV ---------------------------------------------------------
BOND_COLS = ["Symbol", "YTW %", "Price %", "Coupon %", "Maturity date",
             "Outstanding amt", "Face value", "S&P rating", "Fitch rating",
             "Issuer"]


def build_rows(ticker, api_key, as_of=None):
    cfg = ISSUERS[ticker]
    as_of = as_of or _dt.date.today()

    seen, cands = set(), []
    for q in cfg["search_queries"]:
        for b in eodhd_search_bonds(q, api_key):
            isin = b.get("ISIN") or b.get("Code")
            if isin and isin not in seen:
                seen.add(isin); cands.append(b)
    print(f"  enumerated {len(cands)} candidate bonds")

    rows, kept, dropped = [], 0, 0
    for b in cands:
        name = b.get("Name", "") or ""
        isin = b.get("ISIN") or b.get("Code")
        ccy = b.get("Currency", "")
        ok, _ = keep_bond(name, cfg)
        if not ok:
            dropped += 1; continue
        if ccy and cfg["currency"] and ccy != cfg["currency"]:
            dropped += 1; continue
        mat = parse_maturity(name, as_of)
        coup = parse_coupon(name)
        if mat is None or coup is None:
            dropped += 1; continue          # no maturity/coupon in name -> skip
        yrs = (mat - as_of).days / 365.25
        if not (MIN_YEARS <= yrs <= MAX_YEARS):
            dropped += 1; continue
        ytw, price, n = robust_prints(eodhd_bond_prints(isin, api_key))
        if ytw is None:
            dropped += 1; continue
        rows.append({
            "Symbol": f"{isin} {name}".strip(),
            "YTW %": round(ytw, 6),
            "Price %": round((price if price is not None else 100.0) / 100.0, 6),
            "Coupon %": round(coup, 6),
            "Maturity date": mat.isoformat(),
            "Outstanding amt": "",
            "Face value": "1,000.00 USD",
            "S&P rating": cfg["rating"],
            "Fitch rating": "—",
            "Issuer": cfg["name"],
        })
        kept += 1
    print(f"  kept {kept} AT&T-family USD bonds, dropped {dropped}")
    if kept < 3:
        raise RuntimeError(f"{ticker}: only {kept} usable bonds — refusing to "
                           f"overwrite bonds/{ticker}.csv with a thin list")
    rows.sort(key=lambda r: r["Maturity date"])
    return rows


def write_bonds_csv(ticker, rows, bonds_dir="bonds"):
    os.makedirs(bonds_dir, exist_ok=True)
    path = os.path.join(bonds_dir, f"{ticker.upper()}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOND_COLS)
        w.writeheader(); w.writerows(rows)
    return path


def main():
    if "--self-test" in sys.argv:
        return self_test()
    ticker = (os.environ.get("TICKER") or
              (sys.argv[1] if len(sys.argv) > 1 else "")).strip().upper()
    # Graceful no-ops so this step is safe to run for ANY company job:
    if not ticker:
        print("refresh_bonds: no TICKER given — skipping (nothing to refresh).")
        return
    if ticker not in ISSUERS:
        print(f"refresh_bonds: {ticker} not configured for auto-refresh — "
              f"leaving bonds/{ticker}.csv as-is.")
        return
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        print("refresh_bonds: EODHD_API_KEY not set — skipping refresh, "
              "leaving the committed bond file as-is.")
        return
    rows = build_rows(ticker, api_key)
    path = write_bonds_csv(ticker, rows,
                           bonds_dir=os.environ.get("BONDS_DIR", "bonds"))
    print(f"  wrote {path} ({len(rows)} bonds)")


# ---- self-test: baked-in live AT&T bonds, no network, no key ---------------
def self_test():
    print("SELF-TEST — refresh_bonds for AT&T (live fixtures, no network)\n")
    as_of = _dt.date(2026, 7, 17)
    fx = [  # ISIN, Name, currency, ytw% (as EODHD returns, percent)
        ("US00206RDQ20", "AT&T INC 4.25% 01Mar2027", "USD", 4.35, 99.93),
        ("US00206RGL06", "AT&T INC 4.1% 15Feb2028",  "USD", 4.56, 99.29),
        ("US00206RGQ92", "AT&T INC 4.3% 15Feb2030",  "USD", 4.80, 98.39),
        ("US00206RCP55", "AT&T INC 4.5% 15May2035",  "USD", 5.48, 93.16),
        ("US00206RFW79", "AT&T INC 4.9% 15Aug2037",  "USD", 5.50, 94.39),
        ("US00206RDH21", "AT&T INC 5.15% 15Mar2042", "USD", 6.12, 90.33),
        ("US00206RCG56", "AT&T INC 4.8% 15Jun2044",  "USD", 6.31, 83.92),
        ("US00206RDS85", "AT&T INC 5.45% 01Mar2047", "USD", 6.38, 90.17),
        ("US00206RDK59", "AT&T INC 4.55% 09Mar2049", "USD", 6.41, 77.71),
        ("US00206RDT68", "AT&T INC 5.7% 01Mar2057",  "USD", 6.35, 91.20),
        ("US887317AA00", "TIME WARNER 6.1% 15Jul2040", "USD", 11.0, 65.0),   # decoy
        ("US25459HAA00", "DIRECTV 5.15% 15Mar2042", "USD", 18.2, 32.9),      # decoy
        ("US00206RZZ00", "AT&T INC 3.5% 01Jan2030 EUR", "EUR", 3.5, 99.0),   # decoy
    ]
    pmap = {i: (y, p) for i, _, _, y, p in fx}

    global eodhd_search_bonds, eodhd_bond_prints
    eodhd_search_bonds = lambda q, k, limit=500: (
        [{"Code": i, "Name": n, "ISIN": i, "Currency": c} for i, n, c, _, _ in fx]
        if q == "AT&T" else [])
    eodhd_bond_prints = lambda i, k, days=7: (
        [{"date": "2026-07-16", "price": pmap[i][1], "yield": pmap[i][0], "volume": 0}]
        if i in pmap else [])

    rows = build_rows("T", "FAKE", as_of)
    path = write_bonds_csv("T", rows, bonds_dir="/tmp/bonds_selftest")
    print(f"\n  wrote {path}")
    print("  issuers in file:", sorted({r['Issuer'] for r in rows}))
    print("  n rows:", len(rows), "(decoys Time Warner / DIRECTV / EUR must be gone)")
    print("  first 3 rows:")
    for r in rows[:3]:
        print("   ", r["Maturity date"], r["Coupon %"], "cpn  ytw", r["YTW %"])
    assert len(rows) == 10, "decoys should be excluded"
    assert all("AT&T" in r["Issuer"] for r in rows)
    print("\n  SELF-TEST PASSED (10 clean AT&T bonds, media + non-USD excluded)")


if __name__ == "__main__":
    main()
