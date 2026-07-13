import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from asfp import erp

GRID = np.arange(1, 31, dtype=float)
# aggregate IG index spread rising ~0.46 -> 0.96 (like the live credit grid)
IG = np.interp(GRID, [1, 5, 10, 20, 30], [0.46, 0.71, 0.92, 0.96, 0.96])

def test_anchor_from_vix():
    # VIX 18 -> anchor 3.24%
    d = erp.build_market_erp(GRID, IG, a_mkt=(18.0**2)/100.0)
    assert abs(d.loc[1, "market_erp"] - 3.24) < 1e-6      # yr1 == anchor

def test_declines_to_floor():
    d = erp.build_market_erp(GRID, IG, a_mkt=3.24)
    assert d.loc[1, "market_erp"] > d.loc[30, "market_erp"]        # mean-reverts down
    assert d.loc[30, "market_erp"] > 0.5                            # positive floor
    # floor is credit-RP(30) - liq + tail
    assert abs(d.loc[30, "floor"] - ((0.96 - 0.6*0.30) - 0.30 + 0.50)) < 1e-6

if __name__ == "__main__":
    d = erp.build_market_erp(GRID, IG, a_mkt=3.24)
    print("VIX=18 -> anchor 3.24%")
    print(d[["market_erp","market_credit_rp","floor"]].loc[[1,2,5,10,20,30]].round(3).to_string())
