"""Tests for parsing the Google-Sheet bond block (both CSV export flavors)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import sheets, debt_analytics as da

HDR = ["Symbol", "YTW %", "Price %", "Coupon %", "Maturity date",
       "Outstanding amt", "Face value", "S&P rating", "Fitch rating", "Issuer"]

# /export flavor: a TICKER cell above the block, clean headers, DECIMAL values
EXPORT = (
    "TICKER,T\n"
    ",\n"
    + ",".join(HDR) + "\n"
    "US1AApple 3% 2049,0.0582,0.6373,0.0295,2049-09-11,1.5 B USD,1000,AA+,,Apple Inc.\n"
    "US2AApple 4.1% 2062,0.0575,0.7412,0.041,2062-08-08,1.25 B USD,1000,AA+,,Apple Inc.\n"
)

# /gviz flavor: doubled headers, PERCENT strings, no ticker cell
GVIZ = (
    ",".join(f'"{h} {h}"' for h in HDR) + "\n"
    '"US1AApple 3% 2049","5.82%","63.73%","2.95%","2049-09-11","1.5 B USD","1,000.00 USD","AA+","","Apple Inc."\n'
    '"US2AApple 4.1% 2062","5.75%","74.12%","4.10%","2062-08-08","1.25 B USD","1,000.00 USD","AA+","","Apple Inc."\n'
)


def _parse(text):
    raw = sheets.csv_to_df(text)
    return sheets.read_ticker(raw), da.parse_tradingview_bonds(raw)


def test_export_flavor_ticker_and_decimals():
    tk, b = _parse(EXPORT)
    assert tk == "T"
    assert len(b) == 2
    assert abs(float(b["ytw"].iloc[0]) - 0.0582) < 1e-6
    assert abs(float(b["price_frac"].iloc[0]) - 0.6373) < 1e-6
    assert abs(float(b["coupon"].iloc[0]) - 0.0295) < 1e-6


def test_gviz_flavor_percent_strings_and_doubled_headers():
    tk, b = _parse(GVIZ)
    assert tk is None                       # no ticker cell in this flavor
    assert len(b) == 2
    # "5.82%" -> 0.0582, "63.73%" -> 0.6373  (percent strings decoded)
    assert abs(float(b["ytw"].iloc[0]) - 0.0582) < 1e-6
    assert abs(float(b["price_frac"].iloc[0]) - 0.6373) < 1e-6


def test_market_value_matches_across_flavors():
    _, be = _parse(EXPORT)
    _, bg = _parse(GVIZ)
    mv_e = da.market_value(be)[1]
    mv_g = da.market_value(bg)[1]
    assert abs(mv_e - mv_g) < 1.0            # identical bonds, either flavor
    assert mv_e > 0
