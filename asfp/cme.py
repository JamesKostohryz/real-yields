"""
CME E-mini S&P 500 (ES) long-dated option settlements -> ATM implied-vol term
structure, to extend the observed index-vol curve past the ~3y LEAPS reach toward ~5y.

The quarterly ES options list expiries out to ~5 years (Dec 2029/2030). Each settlement
gives strikes + settle prices; we take the near-the-money strikes, pull the matching
quarterly FUTURES settle as the forward, and invert Black-76 to an ATM implied vol at
each far expiry. Those (years, vol_points) pairs feed the market-ERP front.

Two layers:
  * PURE, unit-tested math (Black-76 price + bisection implied vol, expiry-code parsing,
    ATM extraction from a settlements list). No I/O — deterministic and testable offline.
  * A best-effort NETWORK fetcher that hits CME's public settlement web service on the CI
    runner (which can reach CME; the build sandbox cannot). It is heavily LOGGED and
    hard-VALIDATED: every candidate endpoint's outcome is printed, and only implied vols
    in a sane range at genuinely long tenors are returned. Anything unexpected -> [] (the
    ERP just leans on the bond blend past 3y, exactly as before). So a wrong guess about
    CME's product id / format degrades safely and the log tells us how to fix it.

CME product ids are not stable/guessable, so the fetcher tries a few candidates and logs
which (if any) returns far-dated data; override once known via env:
    CME_OPT_PRODUCTS="138,3554"   CME_FUT_PRODUCT="133"
"""
from __future__ import annotations

import math
import os
import datetime as dt
from statistics import NormalDist

_N = NormalDist().cdf
MONTH_CODE = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
              "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
MONTH_ABBR = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
              7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}


# ----------------------------------------------------------------- pure math
def black76_call(F, K, T, r, sigma):
    """Black-76 call price on a future F (settle in index points), strike K, T years,
    cc rate r (decimal), vol sigma (decimal)."""
    if T <= 0 or sigma <= 0:
        return math.exp(-r * T) * max(F - K, 0.0)
    srt = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / srt
    d2 = d1 - srt
    return math.exp(-r * T) * (F * _N(d1) - K * _N(d2))


def implied_vol_call(price, F, K, T, r, lo=1e-4, hi=5.0, iters=100):
    """Implied vol (decimal) from a call price by bisection (price is monotone in vol).
    Returns None if the price is outside the no-arbitrage envelope."""
    if price is None or price <= 0 or F <= 0 or K <= 0 or T <= 0:
        return None
    intrinsic = math.exp(-r * T) * max(F - K, 0.0)
    upper = math.exp(-r * T) * F
    if price <= intrinsic + 1e-9 or price >= upper:      # off the curve -> unreliable
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if black76_call(F, K, T, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def third_friday(year, month):
    """3rd Friday of a month — the standard quarterly expiry date."""
    d = dt.date(year, month, 1)
    first_fri = d + dt.timedelta(days=(4 - d.weekday()) % 7)
    return first_fri + dt.timedelta(days=14)


def parse_expiry_code(code, base_year):
    """'Z9' -> (year, month, MONTH_ABBR label). Single-digit year resolved forward from
    base_year (the run year), so 9 -> 2029, 0 -> 2030 when base_year is 2026."""
    code = code.strip().upper()
    m = MONTH_CODE.get(code[0])
    if m is None or len(code) < 2 or not code[1].isdigit():
        return None
    d = int(code[1])
    year = (base_year // 10) * 10 + d
    if year < base_year:
        year += 10
    return year, m, f"{MONTH_ABBR[m]} {year % 100:02d}"


def atm_iv_from_settlements(settlements, forward, T, r, n_atm=3, max_moneyness=0.10):
    """ATM implied vol (decimal) from a settlements list.

    settlements: iterable of dicts with a strike, a Call/Put type, and a settle price
    (string or number). We keep CALL settles within `max_moneyness` of `forward`, invert
    each to a vol, and median the `n_atm` nearest the money. Returns None if nothing
    plausible survives."""
    calls = []
    for s in settlements:
        typ = str(s.get("type", s.get("optionType", ""))).lower()
        if not (typ.startswith("c")):
            continue
        try:
            K = float(str(s.get("strike", s.get("strikePrice", ""))).replace(",", ""))
            px = float(str(s.get("settle", s.get("settlement", ""))).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if K <= 0 or px <= 0:
            continue
        if abs(K / forward - 1.0) > max_moneyness:
            continue
        calls.append((abs(K - forward), K, px))
    if not calls:
        return None
    calls.sort()
    vols = []
    for _, K, px in calls[:n_atm]:
        iv = implied_vol_call(px, forward, K, T, r)
        if iv is not None and 0.03 <= iv <= 1.5:
            vols.append(iv)
    if not vols:
        return None
    vols.sort()
    return vols[len(vols) // 2]


# ----------------------------------------------------------------- network (runner)
_HDRS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.html",
}
_BASE = "https://www.cmegroup.com/CmeWS/mvc/Settlements"


def _get_json(url, timeout, log):
    """GET a URL and return parsed JSON, logging a short diagnostic either way."""
    import requests
    try:
        r = requests.get(url, headers=_HDRS, timeout=timeout)
    except Exception as e:
        log(f"    {url[:90]} -> EXC {type(e).__name__}")
        return None
    ct = r.headers.get("content-type", "")
    if r.status_code != 200:
        log(f"    {url[:90]} -> HTTP {r.status_code}")
        return None
    if "json" not in ct.lower():
        log(f"    {url[:90]} -> non-JSON ({ct[:30]}); first bytes: {r.text[:100]!r}")
        return None
    try:
        return r.json()
    except Exception:
        log(f"    {url[:90]} -> JSON parse failed; first bytes: {r.text[:100]!r}")
        return None


def _futures_forwards(fut_product, timeout, log):
    """{month_label: forward_settle} from the futures settlement service, e.g. 'DEC 29'->8594.75."""
    url = f"{_BASE}/Futures/Settlements/{fut_product}/FUT?strategy=DEFAULT"
    j = _get_json(url, timeout, log)
    out = {}
    if not j:
        return out
    for s in j.get("settlements", []):
        try:
            mlabel = str(s.get("month", "")).upper().strip()
            settle = float(str(s.get("settle", "")).replace(",", ""))
            if mlabel and settle > 0:
                out[mlabel] = settle
        except (TypeError, ValueError):
            continue
    log(f"    futures {fut_product}: {len(out)} contract settles")
    return out


def fetch_cme_settlement_vols(disc_rate_pct=4.5, timeout=20, log=print, base_year=None):
    """Best-effort long-dated ATM implied vols from CME ES option settlements.
    Returns [(years, vol_points), …] (typically 2y..5y) or [] on any failure.

    Heavily logged: prints every endpoint tried and its outcome, so the first CI run
    reveals CME's actual product ids / format even if these guesses are wrong. Override
    once known: env CME_OPT_PRODUCTS (comma list) and CME_FUT_PRODUCT."""
    try:
        import requests  # noqa: F401
    except Exception as e:
        log(f"  cme: requests unavailable ({e}); skipping"); return []

    if base_year is None:
        base_year = dt.date.today().year
    r = disc_rate_pct / 100.0

    fut_product = os.environ.get("CME_FUT_PRODUCT", "133").strip()
    opt_products = [p.strip() for p in
                    os.environ.get("CME_OPT_PRODUCTS", "138,3554,137,136").split(",") if p.strip()]
    # candidate far quarterly expiries (Mar/Jun/Sep/Dec) ~2-5y out
    codes = ["Z8", "H9", "M9", "U9", "Z9", "H0", "M0", "U0", "Z0", "Z1"]

    log(f"  cme: probing futures={fut_product} opts={opt_products} (base_year={base_year})")
    forwards = _futures_forwards(fut_product, timeout, log)
    if not forwards:
        log("  cme: no futures forwards -> cannot pin ATM; skipping (see log above)")
        return []

    out, seen_t = [], set()
    for pid in opt_products:
        got_any = False
        for code in codes:
            exp = parse_expiry_code(code, base_year)
            if not exp:
                continue
            year, month, mlabel = exp
            F = forwards.get(mlabel)
            if not F:
                continue                                  # no matching future for this expiry
            url = (f"{_BASE}/Options/Settlements/{pid}/OOF"
                   f"?optionExpiration={pid}-{code}&strategy=DEFAULT&pageSize=1500")
            j = _get_json(url, timeout, log)
            if not j:
                continue
            settles = j.get("settlements", [])
            if not settles:
                continue
            T = (third_friday(year, month) - dt.date.today()).days / 365.0
            if T < 1.2:                                   # too near — LEAPS already cover it
                continue
            iv = atm_iv_from_settlements(settles, F, T, r)
            if iv is None:
                log(f"    opt {pid}-{code}: {len(settles)} rows, F={F:.0f}, ATM IV not recoverable")
                continue
            tkey = round(T, 1)
            if tkey in seen_t:
                continue
            seen_t.add(tkey)
            got_any = True
            out.append((round(T, 3), round(iv * 100.0, 2)))
            log(f"    opt {pid}-{code}: T={T:.2f}y F={F:.0f} -> ATM IV {iv*100:.2f}")
        if got_any:
            log(f"  cme: product {pid} yielded {len(out)} far-dated IV points")
            break                                         # this product works; stop probing
    out.sort()
    if not out:
        log("  cme: no long-dated ATM vols recovered (endpoints/format differ — see log)")
    return out
