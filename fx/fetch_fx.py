#!/usr/bin/env python3
"""fetch_fx.py -- the committed exchange-rate history the AEG engine translates foreign filers with.

WHY THIS EXISTS, AND WHY IT IS A HISTORY RATHER THAN A SPOT RATE.

Standing rule 11 says a foreign filer is valued in its own currency and that currency enters the
valuation exactly once, at the final translation of value per share. That is true of the discounted
stream. But `market_data.apply_market_data()` also writes a HISTORICAL year-end price series into the
workbook -- one price at each fiscal year-end across the filing history -- and `Econ Statements` row
95 divides reported common equity by market capitalization off it, year by year. Equity comes from
the statements; market cap comes from the vendor's quoted price. For a foreign filer those are two
different currencies, and the error lands in row 98, a REAL quantity in the economic restatement.

It works today only because the statements are currently converted to dollars. The moment they stay
in local currency -- the whole point of the foreign-filer work -- that row breaks, and it breaks
silently, because a book-to-market ratio has no scale any gate can check. So the requirement is one
exchange rate per fiscal year-end DATE across the filing history.

Traced cell by cell in
AEG-Project/docs/engine/SPEC-Foreign-Filer-Data-Requirements-2026-09-06.md section 2, which corrects
two other specs that say to reduce fx_rates.py to a single spot lookup.

WHY A COMMITTED FEED AND NOT JUST fx_rates.py. That module fetches FRED live at run time. A
valuation is point in time (standing rule 3) and must reproduce years later; it cannot if the
exchange rate underneath it is refetched on every run. Nothing fails -- the number is simply
different. Every run here writes a DATED IMMUTABLE VINTAGE, exactly as the consumer price feed does.

THE CONVENTION. Everything published is `usd_per_unit`: US dollars per ONE unit of the foreign
currency, the same normalization `fx_rates.usd_per_unit_spot/avg()` expose, so the two can never
disagree about direction. FRED quotes most pairs the other way round and those are inverted here.

FAILS LOUD, NOT OPEN. An unreachable source, a series that has stopped advancing, a history that has
silently shortened, or a monthly average that disagrees with its own daily series all end the run
non-zero. A quietly wrong exchange rate is the ~7x invisible error class (failure mode A,
00-START-HERE.md); it must never be committed.
"""
from __future__ import annotations
import csv
import datetime as dt
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
HIST = os.path.join(OUT, "fx_history")
UA = {"User-Agent": "aeg-fx-feed/1"}
FREDCSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"
FREDAPI = ("https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}"
           "&file_type=json&observation_start={start}&observation_end={end}")


class FXError(RuntimeError):
    pass


def _get(url, tries=3, timeout=90):
    """One HTTP GET with bounded retries. Same lesson as fx_rates and the CPI feed: a single slow
    read must not cost the run, and a real outage must raise rather than return something empty."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                       # noqa: BLE001 -- re-raised below
            last = e
            if i < tries - 1:
                time.sleep(2 ** i)
    raise FXError(f"unreachable after {tries} tries: {url.split('?')[0]} :: {last}")


def fetch_fred(sid, start="1900-01-01"):
    """{'YYYY-MM-DD': value}. Authenticated API when FRED_API_KEY is present -- the keyless graph
    endpoint is throttled for cloud IPs and is not reliably reachable from CI runners, which cost
    three YMM valuations on 2026-09-05 before it was found."""
    end = dt.date.today().isoformat()
    key = os.environ.get("FRED_API_KEY", "").strip()
    out = {}
    if key:
        payload = _get(FREDAPI.format(sid=sid, key=key, start=start, end=end))
        import json
        for o in json.loads(payload).get("observations", []):
            if o.get("value") not in (None, "", ".", "null"):
                out[o["date"][:10]] = float(o["value"])
    else:
        for line in _get(FREDCSV.format(sid=sid, start=start, end=end)).strip().splitlines()[1:]:
            p = line.split(",")
            if len(p) == 2 and p[1] not in ("", ".", "null"):
                out[p[0][:10]] = float(p[1])
    if not out:
        raise FXError(f"FRED returned no observations for {sid}")
    return out


def to_usd_per_unit(raw, direction):
    """Normalize to US dollars per one unit of the foreign currency.

    Identical to `aeg-valuation/fx_rates._to_usd_per_unit`, deliberately: one convention in the
    system, defined the same way in both places. A zero would be a division error rather than a
    plausible rate, so it is refused rather than allowed to become an infinity nobody notices."""
    if direction == "usd_per_foreign":
        return raw
    if direction == "foreign_per_usd":
        if raw == 0:
            raise FXError("a quoted rate of zero cannot be inverted")
        return 1.0 / raw
    raise FXError(f"unknown quote direction {direction!r} "
                  f"(expected 'foreign_per_usd' or 'usd_per_foreign')")


def build(reg):
    """Fetch every registered currency. Returns (daily, monthly, notes) keyed by currency, both
    already normalized to usd_per_unit."""
    daily, monthly, notes = {}, {}, []
    short, mismatch = [], []
    start = reg.get("fetch_from", "1900-01-01")
    tol = float(reg.get("monthly_vs_daily_tolerance", 0.005))

    # The identity currencies need a calendar to be an identity ON, and it must be the union of
    # every real series so a USD filer's lookup can never miss a date a foreign filer's would hit.
    real = {c: v for c, v in reg["currencies"].items() if not v.get("identity")}
    ident = [c for c, v in reg["currencies"].items() if v.get("identity")]

    for ccy in sorted(real):
        c = real[ccy]
        direction = c["direction"]
        d_raw = fetch_fred(c["daily"], start)
        m_raw = fetch_fred(c["monthly"], start)
        daily[ccy] = {d: to_usd_per_unit(v, direction) for d, v in d_raw.items()}
        monthly[ccy] = {m[:7]: to_usd_per_unit(v, direction) for m, v in m_raw.items()}

        first_d, last_d = min(daily[ccy]), max(daily[ccy])
        print(f"  {ccy}  {c['daily']:<9} {len(daily[ccy]):>6} daily   {first_d} -> {last_d}")
        print(f"       {c['monthly']:<9} {len(monthly[ccy]):>6} monthly {min(monthly[ccy])} "
              f"-> {max(monthly[ccy])}")

        # COVERAGE GUARD. A series that silently shortens would arrive as a hole at the earliest
        # statement years, and nothing would fail -- the workbook would tie on a history that had
        # quietly lost its front end.
        expect = c.get("history_from")
        if expect:
            got = min(monthly[ccy])
            if got > expect:
                short.append(f"{ccy}: expected history from {expect}, got {got}")
            elif got < expect:
                print(f"       NOTE: history now reaches {got}, earlier than the declared "
                      f"{expect}. Update registry.history_from.")

        # CROSS-CHECK, AND THIS IS THE ONE THAT CATCHES A SWAPPED SERIES OR A WRONG DIRECTION.
        # The Board publishes the monthly average independently of the daily series. If this feed
        # has the direction backwards, or has paired a currency's daily series with another's
        # monthly one, the two will not agree -- and nothing else in the system would notice,
        # because a plausible-looking exchange rate has no internal identity to violate.
        #
        # THE COMPARISON IS DONE IN FRED'S OWN QUOTED SPACE, BEFORE INVERSION, AND THAT IS NOT A
        # DETAIL. `mean(1/x)` is not `1/mean(x)` -- Jensen's inequality -- and the gap grows with
        # within-month variance. Comparing the inverted series flagged exactly three months on the
        # first run: China 1989-12, Mexico 1994-12 and Brazil 1999-01. Those are the yuan's 21%
        # devaluation, the Tequila crisis and the real's float: real events, violent intra-month
        # moves, and nothing whatever wrong with either series. The check would have been a false
        # alarm precisely in the months a reader would most want to trust it.
        #
        # THE SAME ARITHMETIC IS A TRAP FOR CALLERS OF THIS FEED, so it is stated here and in the
        # registry: to average a period, use fx_monthly.csv. DO NOT average the usd_per_unit column
        # of fx_daily.csv -- that computes mean(1/x) and will disagree with every published average,
        # by up to several percent in a devaluation month. The published monthly value is
        # 1/mean(quoted rate), which is what the Board publishes and exactly what
        # `fx_rates.usd_per_unit_avg()` computes (it averages the raw series, then inverts), so the
        # feed and that module agree by construction rather than by luck.
        by_month = {}
        for d, v in d_raw.items():
            by_month.setdefault(d[:7], []).append(v)
        monthly_raw = {m[:7]: v for m, v in m_raw.items()}
        checked = worst = 0
        worst_m = None
        for m, vs in by_month.items():
            if m not in monthly_raw or len(vs) < 15:        # skip partial months
                continue
            own = statistics.fmean(vs)
            if own == 0:
                continue
            rel = abs(monthly_raw[m] - own) / own
            checked += 1
            if rel > worst:
                worst, worst_m = rel, m
        if checked == 0:
            mismatch.append(f"{ccy}: no complete month could be cross-checked")
        elif worst > tol:
            mismatch.append(f"{ccy}: monthly average disagrees with its own daily mean by "
                            f"{worst:.4%} at {worst_m} (tolerance {tol:.2%})")
        else:
            print(f"       cross-check OK: {checked} complete months, worst disagreement "
                  f"{worst:.4%} at {worst_m}")
        notes.append((ccy, checked, worst, worst_m))

    if short:
        raise FXError("exchange-rate history has SHORTENED against the registry -- refusing to "
                      "publish a feed whose early years have quietly gone missing: "
                      + "; ".join(short))
    if mismatch:
        raise FXError("the Board's monthly average disagrees with its own daily series -- this is "
                      "what a swapped series id or a reversed quote direction looks like, and it "
                      "has no other symptom: " + "; ".join(mismatch))

    # Identity currencies, on the union calendar of everything real.
    all_days = sorted({d for s in daily.values() for d in s})
    all_months = sorted({m for s in monthly.values() for m in s})
    for ccy in ident:
        daily[ccy] = {d: 1.0 for d in all_days}
        monthly[ccy] = {m: 1.0 for m in all_months}
        print(f"  {ccy}  identity  {len(daily[ccy]):>6} daily / {len(monthly[ccy])} monthly = 1.0 exactly")

    return daily, monthly, notes


def main():
    import yaml
    reg = yaml.safe_load(open(os.path.join(HERE, "registry.yaml"), encoding="utf-8"))
    stale_days = int(reg.get("stale_after_days", 21))
    today = dt.date.today()
    print(f"[fx] fetching {len(reg['currencies'])} currencies (usd_per_unit convention)")
    daily, monthly, _notes = build(reg)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(HIST, exist_ok=True)

    def write_monthly(path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["currency", "month", "usd_per_unit"])
            for ccy in sorted(monthly):
                for m in sorted(monthly[ccy]):
                    w.writerow([ccy, m, f"{monthly[ccy][m]:.10g}"])

    def write_daily(path):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["currency", "date", "usd_per_unit"])
            for ccy in sorted(daily):
                for d in sorted(daily[ccy]):
                    w.writerow([ccy, d, f"{daily[ccy][d]:.10g}"])

    write_monthly(os.path.join(OUT, "fx_monthly.csv"))
    write_monthly(os.path.join(HIST, f"fx_monthly_{today.isoformat()}.csv"))
    write_daily(os.path.join(OUT, "fx_daily.csv"))
    write_daily(os.path.join(HIST, f"fx_daily_{today.isoformat()}.csv"))

    stale = []
    with open(os.path.join(OUT, "fx_provenance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["currency", "daily_series", "monthly_series", "direction", "verified",
                    "first_month", "history_from_declared", "n_monthly", "n_daily",
                    "last_daily", "age_days", "stale", "fetched_utc"])
        for ccy in sorted(daily):
            c = reg["currencies"][ccy]
            last = max(daily[ccy])
            age = (today - dt.date.fromisoformat(last)).days
            # An identity series is as fresh as the real series it was built on, by construction,
            # so it is reported but never flagged on its own account.
            is_stale = (not c.get("identity")) and age > stale_days
            if is_stale:
                stale.append(f"{ccy} ({last}, {age}d)")
            w.writerow([ccy, c.get("daily", "identity"), c.get("monthly", "identity"),
                        c.get("direction", "identity"),
                        str(bool(c.get("verified"))).lower(),
                        min(monthly[ccy]), c.get("history_from", ""),
                        len(monthly[ccy]), len(daily[ccy]), last, age,
                        str(is_stale).lower(),
                        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")])

    print(f"[fx] wrote fx_monthly.csv, fx_daily.csv, fx_provenance.csv, "
          f"fx_history/fx_{{monthly,daily}}_{today.isoformat()}.csv")
    if stale:
        raise FXError(f"exchange rates stale beyond {stale_days} days: {', '.join(stale)} -- "
                      f"rates publish every business day, so this means the fetch is broken")
    unverified = [c for c, v in reg["currencies"].items() if not v.get("verified")]
    if unverified:
        print(f"[fx] QUOTE DIRECTION NOT YET CONFIRMED BY A REAL FILING (no valuation may publish "
              f"on these): {', '.join(sorted(unverified))}")


if __name__ == "__main__":
    try:
        main()
    except FXError as e:
        print(f"::error::{e}")
        sys.exit(1)
