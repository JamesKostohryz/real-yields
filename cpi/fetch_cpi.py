#!/usr/bin/env python3
"""fetch_cpi.py -- the monthly consumer-price feed the AEG engine deflates foreign filers with.

WHY THIS EXISTS. Standing rule 11, ruled by James 2026-09-05: a company that reports in a foreign
currency is NOT converted to dollars. Its statements are deflated by its OWN country's consumer
price index, exactly as a US filer's are deflated by CPI-U. This module fetches those indices,
publishes them as dated immutable vintages, and says loudly when one has gone stale.

THE DESIGN POINT THAT MATTERS, AND IT IS NOT THE FETCHING. Consumer price indices are REVISED --
seasonally, at rebasing, and sometimes in back history. A valuation is point in time (standing rule
3) and must reproduce years later. If the engine read a live feed, or a single file that gets
overwritten, a published valuation could not be reproduced: the deflator underneath it would have
moved, nothing would fail, and the number would simply be different. So every run writes a DATED
SNAPSHOT into outputs/cpi_history/ and every valuation records the cpi_vintage it stood on. Same
discipline as screen_history/, and for the same reason.

SOURCES. FRED for the dollar and the euro; the OECD's own SDMX interface for everything else.
Note that FRED's MIRRORS of the OECD series are stale by one to five years -- CHNCPIALLMINMEI stops
at 2025-04 -- while the OECD's own interface is current to 2026-07. The data was never missing; the
intermediary had stopped updating. World Bank FP.CPI.TOTL is annual-only and is a cross-check, never
a primary source.

FAILS LOUD, NOT OPEN. A source that cannot be reached, or a series that has stopped advancing, ends
the run non-zero. It never commits a quietly stale file, because a stale deflator does not announce
itself -- it just restates history slightly wrong, which is failure mode A in 00-START-HERE.md.
"""
from __future__ import annotations
import csv, datetime as dt, json, os, statistics, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
HIST = os.path.join(OUT, "cpi_history")
UA = {"User-Agent": "aeg-cpi-feed/1"}
OECD = ("https://sdmx.oecd.org/public/rest/data/{flow}/"
        "{iso3}.M.N.CPI.IX._T.N._Z?startPeriod={start}&dimensionAtObservation=AllDimensions"
        "&format=jsondata")
FREDCSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}&coed={end}"
FREDAPI = ("https://api.stlouisfed.org/fred/series/observations?series_id={sid}&api_key={key}"
           "&file_type=json&observation_start={start}&observation_end={end}")


class CPIError(RuntimeError):
    pass


def _get(url, tries=3, timeout=60):
    """One HTTP GET with bounded retries. The lesson from fx_rates: a single slow read must not
    cost the run, and a real outage must still raise rather than return something empty."""
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:                       # noqa: BLE001 - re-raised below
            last = e
            if i < tries - 1:
                time.sleep(2 ** i)
    raise CPIError(f"unreachable after {tries} tries: {url.split('?')[0]} :: {last}")


def fetch_oecd(iso3, flow, start="1990-01"):
    """{'YYYY-MM': index} from the OECD's own SDMX interface. The response is 'AllDimensions'
    flat: observation keys are colon-joined dimension positions whose LAST element indexes the
    TIME_PERIOD value list, and the period list is NOT in chronological order -- it must be read
    from the structure rather than assumed."""
    d = json.loads(_get(OECD.format(flow=flow, iso3=iso3, start=start)))
    ds = (d.get("data", {}).get("dataSets") or [])
    if not ds:
        raise CPIError(f"OECD returned no dataSets for {iso3}")
    dims = d["data"]["structures"][0]["dimensions"]["observation"]
    periods = next(x["values"] for x in dims if x["id"] == "TIME_PERIOD")
    out = {}
    for key, val in (ds[0].get("observations") or {}).items():
        if not val or val[0] is None:
            continue
        out[periods[int(key.split(":")[-1])]["id"]] = float(val[0])
    if not out:
        raise CPIError(f"OECD returned no observations for {iso3} on {flow} -- a country sits in exactly ONE of the two COICOP dataflows and the wrong one returns empty, not an error. Check registry `flow`.")
    return out


def fetch_fred(sid, start="1990-01-01"):
    """{'YYYY-MM': index}. Uses the AUTHENTICATED API when FRED_API_KEY is present -- the
    unauthenticated graph endpoint is not reachable from this project's CI runners, which cost
    three YMM valuations on 2026-09-05 before it was found."""
    end = dt.date.today().isoformat()
    key = os.environ.get("FRED_API_KEY", "").strip()
    out = {}
    if key:
        for o in json.loads(_get(FREDAPI.format(sid=sid, key=key, start=start, end=end))).get(
                "observations", []):
            if o.get("value") not in (None, "", ".", "null"):
                out[o["date"][:7]] = float(o["value"])
    else:
        for line in _get(FREDCSV.format(sid=sid, start=start, end=end)).strip().splitlines()[1:]:
            p = line.split(",")
            if len(p) == 2 and p[1] not in ("", ".", "null"):
                out[p[0][:7]] = float(p[1])
    if not out:
        raise CPIError(f"FRED returned no observations for {sid}")
    return out


def build(reg):
    """Fetch every registered currency. Returns {ccy: {'YYYY-MM': index}}."""
    monthly = {}
    for ccy, c in sorted(reg["currencies"].items()):
        src = c["source"]
        if src == "oecd":
            monthly[ccy] = fetch_oecd(c["series"], reg["oecd_flows"][c["flow"]])
        elif src == "fred":
            monthly[ccy] = fetch_fred(c["series"])
        else:
            raise CPIError(f"{ccy}: unknown source {src!r} (expected 'oecd' or 'fred')")
        print(f"  {ccy}  {src:<5} {c['series']:<22} {c.get('flow',''):<6} "
              f"{len(monthly[ccy]):>4} months  latest {max(monthly[ccy])}")
    return monthly


def annual(monthly):
    """{ccy: {year: (mean index, n_months)}}. The mean of the months present -- which is how CPI-U's
    own annual average is built -- and n_months is carried so the engine can tell a complete year
    from the year in progress. That column is what lets a foreign anchor be extended into the
    current year the way the US path already is."""
    out = {}
    for ccy, series in monthly.items():
        by_year = {}
        for ym, v in series.items():
            by_year.setdefault(int(ym[:4]), []).append(v)
        out[ccy] = {y: (statistics.fmean(vs), len(vs)) for y, vs in by_year.items()}
    return out


def main():
    import yaml
    reg = yaml.safe_load(open(os.path.join(HERE, "registry.yaml")))
    stale_days = int(reg.get("stale_after_days", 120))
    today = dt.date.today()
    print(f"[cpi] fetching {len(reg['currencies'])} currencies")
    monthly = build(reg)
    ann = annual(monthly)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(HIST, exist_ok=True)

    with open(os.path.join(OUT, "cpi_monthly.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["currency", "month", "index"])
        for ccy in sorted(monthly):
            for ym in sorted(monthly[ccy]):
                w.writerow([ccy, ym, f"{monthly[ccy][ym]:.6f}"])

    rows = [["currency", "year", "index", "n_months"]]
    for ccy in sorted(ann):
        for y in sorted(ann[ccy]):
            m, n = ann[ccy][y]
            rows.append([ccy, y, f"{m:.6f}", n])
    for path in (os.path.join(OUT, "cpi_annual.csv"),
                 os.path.join(HIST, f"cpi_annual_{today.isoformat()}.csv")):
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(rows)

    stale = []
    with open(os.path.join(OUT, "cpi_provenance.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["currency", "source", "series", "base_period", "verified", "last_month",
                    "age_days", "stale", "n_months_current_year", "fetched_utc"])
        for ccy in sorted(monthly):
            c = reg["currencies"][ccy]
            last = max(monthly[ccy])
            y, m = int(last[:4]), int(last[5:7])
            # age from the END of the month the observation covers
            eom = dt.date(y + (m == 12), (m % 12) + 1, 1) - dt.timedelta(days=1)
            age = (today - eom).days
            is_stale = age > stale_days
            if is_stale:
                stale.append(f"{ccy} ({last}, {age}d)")
            w.writerow([ccy, c["source"], c["series"], c["base_period"],
                        str(bool(c.get("verified"))).lower(), last, age,
                        str(is_stale).lower(), ann[ccy][today.year][1] if today.year in ann[ccy] else 0,
                        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")])

    print(f"[cpi] wrote cpi_monthly.csv, cpi_annual.csv, cpi_provenance.csv, "
          f"cpi_history/cpi_annual_{today.isoformat()}.csv")
    if stale:
        print(f"::warning::CPI stale beyond {stale_days} days: {', '.join(stale)}")
    unverified = [c for c, v in reg["currencies"].items() if not v.get("verified")]
    if unverified:
        print(f"[cpi] NOT YET RECONCILED TO A FILING (no valuation may publish on these): "
              f"{', '.join(sorted(unverified))}")


if __name__ == "__main__":
    try:
        main()
    except CPIError as e:
        print(f"::error::{e}"); sys.exit(1)
