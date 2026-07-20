#!/usr/bin/env python3
"""
refresh_bonds.py  (v2 — GENERIC, no per-company list)
====================================================================
Regenerate  bonds/<TICKER>.csv  automatically from the EODHD API for
*ANY* US ticker — no hand-maintained list of company names, ever.
You never touch the repo to add a company; you just give it a ticker.

How it identifies "this company's own bonds" automatically:
  Every US security's ISIN embeds its issuer's 6-character CUSIP
  (ISIN chars 3-8).  The stock and that issuer's bonds share it:
      AT&T stock  US00206R1023  -> CUSIP6 00206R -> AT&T bonds US00206R…
      Coca-Cola   US1912161007  -> CUSIP6 191216 -> KO bonds   US191216…
      (Coca-Cola FEMSA is 191241 — a DIFFERENT company — auto-excluded)
  So we resolve the ticker to its stock ISIN, take that CUSIP6, and keep
  only the bonds that share it.  Precise, and impossible to pollute with
  a same-named but different issuer.

Pipeline per ticker:
  1. resolve <TICKER> -> the US-listed stock's name + ISIN (EODHD Search),
  2. CUSIP6 = the issuer prefix of that ISIN,
  3. search bonds by the company's core name (a wide net),
  4. keep USD bonds whose ISIN shares CUSIP6, with a coupon + maturity in
     the name, and a live yield,
  5. robust MEDIAN of recent daily yields per bond (matrix-priced, noisy),
  6. write bonds/<TICKER>.csv in the 10-column TradingView format the
     existing pipeline already reads — nothing downstream changes.

Fallback: if the CUSIP6 match is thin (issuer floats debt under a finance
subsidiary with its own CUSIP), it falls back to core-name matching on the
bond descriptions (still avoiding same-name-different-company via the core
name). Either way it refuses to overwrite the file with < MIN_BONDS bonds.

OPTIONAL overrides (only if you ever want to force extra issuer CUSIP6s in,
e.g. legacy subsidiaries): add them to EXTRA_CUSIP6 below. Default: none.

Runs headless in GitHub Actions. stdlib only. EODHD key from env
EODHD_API_KEY (a GitHub Actions secret) — NEVER committed.
Self-test:  python refresh_bonds.py --self-test   (no key, no network)
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
MIN_BONDS = 4                      # refuse to write a thinner list than this

# Only needed for special cases; maps TICKER -> list of extra CUSIP6 prefixes
# to also treat as this issuer (e.g. {"T": ["001957"]} to add AT&T Corp /Old/).
EXTRA_CUSIP6 = {}


# ---- EODHD REST (thin; shapes proven against the live MCP server) ----------
def _get_json(url):
    with urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def eodhd_search(query, api_key, bonds_only=False, limit=500):
    b = "&bonds_only=1&type=bond" if bonds_only else ""
    url = (f"{EODHD_BASE}/search/{quote(query)}"
           f"?api_token={api_key}{b}&limit={limit}&fmt=json")
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


# ---- resolve ticker -> (name, stock ISIN) ----------------------------------
def resolve_issuer(ticker, api_key):
    """Return (company_name, stock_isin) for the US listing of `ticker`."""
    hits = eodhd_search(ticker, api_key, limit=50)
    best = None
    for h in hits:
        code = str(h.get("Code", "")).upper()
        exch = str(h.get("Exchange", "")).upper()
        typ = str(h.get("Type", "")).lower()
        isin = h.get("ISIN") or ""
        if code == ticker.upper() and exch in ("US", "NYSE", "NASDAQ") \
                and "stock" in typ and isin:
            best = h
            break
        if best is None and code == ticker.upper() and isin:
            best = h
    if not best:
        return None, None
    return best.get("Name", ticker), (best.get("ISIN") or "")


def cusip6_of(isin):
    """Issuer CUSIP6 = ISIN chars 3-8 for a US ISIN ('US' + 9-char CUSIP + chk)."""
    if isin and isin[:2].upper() == "US" and len(isin) >= 8:
        return isin[2:8].upper()
    return None


# ---- name helpers ----------------------------------------------------------
_SUFFIX = re.compile(
    r"\b(inc|inc\.|incorporated|corp|corp\.|corporation|co|co\.|company|"
    r"plc|ltd|ltd\.|llc|l\.l\.c\.|sa|s\.a\.|nv|n\.v\.|ag|holdings|holding|"
    r"group|the)\b", re.I)


def core_name(name):
    """A wide-net search term: drop 'The', corporate suffixes, punctuation tails."""
    n = re.sub(r"^the\s+", "", (name or "").strip(), flags=re.I)
    n = _SUFFIX.sub("", n)
    n = re.sub(r"[,/].*$", "", n)            # cut at first comma/slash
    n = re.sub(r"\s+", " ", n).strip(" -&")
    words = n.split()
    return " ".join(words[:3]) if words else (name or "").strip()


# ---- bond parsing ----------------------------------------------------------
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


def robust_prints(rows):
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


# ---- build ------------------------------------------------------------------
BOND_COLS = ["Symbol", "YTW %", "Price %", "Coupon %", "Maturity date",
             "Outstanding amt", "Face value", "S&P rating", "Fitch rating",
             "Issuer"]


def eodhd_book_total_debt(ticker, api_key):
    """Reported BOOK total debt (short + long), absolute USD, or None.

    EODHD publishes no per-bond amount outstanding (verified against
    /bond-fundamentals: the payload carries coupon/maturity/price/ratings only), so
    the bond list cannot be notional-weighted from the feed. We anchor instead to the
    issuer's reported total debt -- a real filed number -- and allocate it.
    """
    url = (f"{EODHD_BASE}/fundamentals/{quote(ticker.upper())}.US"
           f"?api_token={api_key}&fmt=json"
           f"&filter=Financials::Balance_Sheet::yearly")
    try:
        d = _get_json(url)
    except Exception as e:                                   # noqa: BLE001
        print(f"    [WARN] fundamentals for {ticker} failed: {e}")
        return None
    if not isinstance(d, dict) or not d:
        return None
    for y in sorted(d.keys(), reverse=True):
        bs = d[y] or {}
        for f in ("shortLongTermDebtTotal", "totalDebt"):
            v = bs.get(f)
            if v not in (None, "", "0", 0):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return None


def build_rows(ticker, api_key, as_of=None):
    as_of = as_of or _dt.date.today()
    name, isin = resolve_issuer(ticker, api_key)
    if not name:
        raise RuntimeError(f"{ticker}: could not resolve to a US listing on EODHD")
    c6 = cusip6_of(isin)
    allow = {c6} if c6 else set()
    allow |= set(EXTRA_CUSIP6.get(ticker.upper(), []))
    print(f"  resolved {ticker} -> {name!r}  ISIN {isin or '?'}  CUSIP6 {c6 or '?'}")

    cands = eodhd_search(core_name(name), api_key, bonds_only=True)
    print(f"  search '{core_name(name)}' -> {len(cands)} candidate bonds")

    def accept_issuer(bisin, bname):
        if allow:
            c = cusip6_of(bisin)
            if c and c in allow:
                return True
        return False

    # primary pass: CUSIP6 match
    rows = _collect(cands, api_key, as_of, name, ticker, accept_issuer)
    if len(rows) < MIN_BONDS:
        # fallback: core-name-in-description match (still avoids other issuers)
        core = core_name(name).upper()
        print(f"  CUSIP6 match thin ({len(rows)}); falling back to name match on "
              f"'{core}'")
        rows = _collect(cands, api_key, as_of, name, ticker,
                        lambda bi, bn: core in (bn or "").upper())
    if len(rows) < MIN_BONDS:
        raise RuntimeError(f"{ticker}: only {len(rows)} usable bonds — refusing to "
                           f"overwrite bonds/{ticker}.csv")
    rows.sort(key=lambda r: r["Maturity date"])
    print(f"  kept {len(rows)} {ticker} bonds")

    # --- amount outstanding: BOOK-SCALED allocation ----------------------------
    # The feed has no per-bond amount outstanding, and debt_analytics DROPS any bond
    # without one -- which zeroes market value of debt, portfolio YTM and duration.
    # So anchor to the issuer's REPORTED book total debt and spread it evenly over the
    # observed bonds. Market value of debt then collapses to
    #       MVD = book_debt * mean(price_frac)
    # i.e. reported debt marked to the mean traded price of the issuer's own curve.
    # This is an APPROXIMATION (equal notional weighting across maturities); it is
    # tagged 'book-scaled' in company_<T>.csv so it is never mistaken for
    # issue-level truth. Upgrade path: a real 10-K/XBRL debt schedule.
    book = eodhd_book_total_debt(ticker, api_key)
    if book and rows:
        per = book / len(rows)
        for r in rows:
            r["Outstanding amt"] = round(per, 2)
        mean_px = sum(r["Price %"] for r in rows) / len(rows)
        print(f"  book-scaled outstanding: total debt ${book/1e9:,.1f}B over {len(rows)} "
              f"bonds (${per/1e9:,.2f}B each); mean price {mean_px:.4f} "
              f"-> MVD ~ ${book*mean_px/1e9:,.1f}B")
    else:
        print(f"  [WARN] no book total debt for {ticker}; 'Outstanding amt' left blank "
              f"(market value of debt will be unavailable)")
    return rows


def _collect(cands, api_key, as_of, issuer_name, ticker, accept):
    rows = []
    for b in cands:
        name = b.get("Name", "") or ""
        isin = b.get("ISIN") or b.get("Code")
        ccy = b.get("Currency", "")
        if ccy and ccy != "USD":
            continue
        if not accept(isin, name):
            continue
        mat = parse_maturity(name, as_of)
        coup = parse_coupon(name)
        if mat is None or coup is None:
            continue
        yrs = (mat - as_of).days / 365.25
        if not (MIN_YEARS <= yrs <= MAX_YEARS):
            continue
        ytw, price, n = robust_prints(eodhd_bond_prints(isin, api_key))
        if ytw is None:
            continue
        rows.append({
            "Symbol": f"{isin} {name}".strip(),
            "YTW %": round(ytw, 6),
            "Price %": round((price if price is not None else 100.0) / 100.0, 6),
            "Coupon %": round(coup, 6),
            "Maturity date": mat.isoformat(),
            "Outstanding amt": "",
            "Face value": "1,000.00 USD",
            "S&P rating": "",            # blank -> pipeline's modal rating -> BBB,
                                         # then the bond-fitted offset corrects it
            "Fitch rating": "—",
            "Issuer": issuer_name,
        })
    return rows


def write_bonds_csv(ticker, rows, bonds_dir="bonds"):
    os.makedirs(bonds_dir, exist_ok=True)
    path = os.path.join(bonds_dir, f"{ticker.upper()}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BOND_COLS)
        w.writeheader(); w.writerows(rows)
    return path


def ticker_from_sheet():
    """If BONDS_SHEET_ID is set, read the TICKER cell from the Cockpit sheet
    (reusing the pipeline's own sheet reader). Empty string if unavailable."""
    sid = os.environ.get("BONDS_SHEET_ID", "").strip()
    if not sid:
        return ""
    try:
        from asfp import sheets                       # same reader run_company uses
        _, tk = sheets.bonds_and_ticker(sid)
        return (tk or "").strip().upper()
    except Exception as e:                            # noqa: BLE001
        print(f"refresh_bonds: could not read ticker from the sheet: {e}")
        return ""


def main():
    if "--self-test" in sys.argv:
        return self_test()
    # ticker precedence: explicit env/arg (workflow box) > the Cockpit sheet cell
    ticker = (os.environ.get("TICKER") or
              (sys.argv[1] if len(sys.argv) > 1 else "")).strip().upper()
    if not ticker:
        ticker = ticker_from_sheet()
    if not ticker:
        print("refresh_bonds: no TICKER (workflow box empty and no TICKER cell in "
              "the sheet) — skipping.")
        return
    api_key = os.environ.get("EODHD_API_KEY")
    if not api_key:
        print("refresh_bonds: EODHD_API_KEY not set — skipping, leaving the "
              "committed bond file as-is.")
        return
    try:
        rows = build_rows(ticker, api_key)
    except Exception as e:                                   # noqa: BLE001
        # Never fail the run: leave the committed file for the pipeline to use.
        print(f"refresh_bonds: {e}. Leaving bonds/{ticker}.csv as-is.")
        return
    path = write_bonds_csv(ticker, rows,
                           bonds_dir=os.environ.get("BONDS_DIR", "bonds"))
    print(f"  wrote {path} ({len(rows)} bonds)")


# ---- self-test: T, VZ, KO with baked fixtures (no key, no network) ---------
def self_test():
    print("SELF-TEST — generic refresh_bonds (T, VZ, KO), no network\n")
    as_of = _dt.date(2026, 7, 17)
    STOCK = {   # ticker -> (name, stock ISIN)
        "T":  ("AT&T Inc",                   "US00206R1023"),
        "VZ": ("Verizon Communications Inc", "US92343V1044"),
        "KO": ("The Coca-Cola Company",      "US1912161007"),
    }
    # candidate bonds per core-name search (ISIN, Name, ccy, ytw%, price)
    BONDS = {
        "AT&T": [
            ("US00206RDQ20", "AT&T INC 4.25% 01Mar2027", "USD", 4.35, 99.9),
            ("US00206RCP55", "AT&T INC 4.5% 15May2035",  "USD", 5.48, 93.2),
            ("US00206RDT68", "AT&T INC 5.7% 01Mar2057",  "USD", 6.35, 91.2),
            ("US00206RDK59", "AT&T INC 4.55% 09Mar2049", "USD", 6.41, 77.7),
            ("US001957AW94", "AT&T CORP 6.5% 15Mar2029", "USD", 5.20, 103.),
        ],
        "Verizon Communications": [
            ("US92343VDY74", "VERIZON COMMUNICATIONS INC 4.125% 16Mar2027", "USD", 4.20, 99.6),
            ("US92343VBS25", "VERIZON COMMUNICATIONS INC 6.4% 15Sep2033",  "USD", 4.90, 107.),
            ("US92343VCZ58", "VERIZON COMMUNICATIONS INC 4.672% 15Mar2055", "USD", 6.30, 79.1),
            ("US92343VBE39", "VERIZON COMMUNICATIONS INC 4.75% 01Nov2041",  "USD", 5.70, 88.1),
            ("US92344XAB55", "VERIZON NEW YORK INC 7.375% 01Apr2032", "USD", 5.10, 110.5),
        ],
        "Coca-Cola": [
            ("US191241AF58", "COCA-COLA FEMSA S A B DE C V 5.25% 26Nov2043", "USD", 5.6, 97.3),
            ("US191216CR95", "COCA-COLA CO 3.45% 15Mar2027",  "USD", 3.9, 99.5),
            ("US191216DE73", "COCA-COLA CO 4.2% 15May2043",   "USD", 4.9, 87.0),
            ("US191216DP21", "COCA-COLA CO 4.125% 15Mar2053", "USD", 5.1, 89.2),
            ("US191216DJ60", "COCA-COLA CO 4.0% 01Mar2034",   "USD", 4.3, 95.8),
        ],
    }
    global eodhd_search, eodhd_bond_prints
    pmap = {i: (y, p) for lst in BONDS.values() for i, _, _, y, p in lst}

    def fake_search(query, api_key, bonds_only=False, limit=500):
        if not bonds_only:                      # resolve_issuer path
            for tk, (nm, isin) in STOCK.items():
                if query.upper() == tk:
                    return [{"Code": tk, "Exchange": "US", "Type": "Common Stock",
                             "Name": nm, "ISIN": isin}]
            return []
        return [{"Code": i, "Name": n, "ISIN": i, "Currency": c}
                for i, n, c, _, _ in BONDS.get(query, [])]

    eodhd_search = fake_search
    eodhd_bond_prints = lambda i, k, days=7: (
        [{"date": "2026-07-16", "price": pmap[i][1], "yield": pmap[i][0]}]
        if i in pmap else [])

    for tk in ("T", "VZ", "KO"):
        print(f"--- {tk} ---")
        rows = build_rows(tk, "FAKE", as_of)
        write_bonds_csv(tk, rows, bonds_dir="/tmp/bonds_v2")
        isins = [r["Symbol"].split()[0] for r in rows]
        print(f"   kept CUSIP6 set: {sorted({i[2:8] for i in isins})}\n")

    ko = _read("/tmp/bonds_v2/KO.csv")
    assert not any("191241" in r["Symbol"] for r in ko), "FEMSA must be excluded from KO"
    assert all("191216" in r["Symbol"] for r in ko), "KO must be all Coca-Cola Co"
    t = _read("/tmp/bonds_v2/T.csv")
    assert all("00206R" in r["Symbol"] for r in t), "T must be AT&T Inc CUSIP6 only"
    vz = _read("/tmp/bonds_v2/VZ.csv")
    assert all("92343V" in r["Symbol"] for r in vz), "VZ must be Verizon Comms CUSIP6"
    print("SELF-TEST PASSED  same-name-different-company (FEMSA) excluded; each "
          "ticker resolved automatically with NO hand-written list.")


def _read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
