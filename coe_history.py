"""
coe_history.py — GENERALIZED historical cost-of-equity series, per company (any ticker).
Real & nominal, monthly, back to the ticker's return history.

Design (locked with James 2026-07-23):
  COE = real_yield(reliable, as-is) + market_ERP(as-is) + idiosyncratic_ERP.
  idiosyncratic_ERP: from the stock's realized vol IN EXCESS OF the index (residual after a
    rolling market beta), de-noised as a 12-month INTERQUARTILE-MEAN slow base blended with a
    3-month fast overlay, given its OWN gentle countercyclical coefficient (keyed to the real-yield
    regime, NOT the market beta), then priced by a UNIVERSAL price-of-idiosyncratic-risk K.
  K is calibrated ONCE (from AAPL, whose live option-implied idio is 1.915%) and reused for ALL
  tickers -> zero per-name hand-calibration. When a ticker has a live option-implied idio, use it
  to validate/refine K; otherwise the realized-based leg with universal K is the historical + fallback.
  Nominal = real + expected inflation (yields-repo Epi_10y; NOT Shiller).
Pure/injectable (prices, market legs, epi passed in) so the whole thing is testable offline; the
runner run_coe_history.py adds the live monthly-price fetch. The __main__ here is a LOCAL demo only.
"""
import numpy as np, pandas as pd

R_NEUTRAL=2.0; LAM=0.12; W_SLOW=0.60; K_UNIVERSAL=0.0897   # K from AAPL anchor (idio-vol 21.8% -> 1.915%)
BETA_WIN=60; BETA_MIN=24

def _iqmean(x):
    x=np.sort(x); return np.mean(x[3:9])          # drop top/bottom 3 of 12, mean middle 6

def idio_erp_series(px, spx, eff_tips, k=K_UNIVERSAL):
    """px, spx, eff_tips are pd.Series on a common monthly PeriodIndex. Returns idio ERP (%)."""
    ra=np.log(px/px.shift(1)); rs=np.log(spx/spx.shift(1))
    common=ra.dropna().index
    ra=ra.reindex(common); rs=rs.reindex(common)
    beta=np.full(len(ra),np.nan); rav=ra.values; rsv=rs.values
    for i in range(len(rav)):
        lo=max(0,i-BETA_WIN+1); xs=rav[lo:i+1]; ys=rsv[lo:i+1]
        if len(xs)>=BETA_MIN and np.var(ys)>0: beta[i]=np.cov(xs,ys)[0,1]/np.var(ys)
    e=rav-np.where(np.isnan(beta),1.0,beta)*rsv
    v=pd.Series(np.abs(e)*np.sqrt(12.0)*np.sqrt(np.pi/2.0)*100.0,index=common)   # monthly ann. idio-vol primitive
    slow=v.rolling(12).apply(_iqmean,raw=True); fast=v.rolling(3).mean()
    blend=W_SLOW*slow+(1-W_SLOW)*fast
    et=eff_tips.reindex(common)
    cc=1.0+LAM*np.clip((R_NEUTRAL-et)/R_NEUTRAL,-0.5,1.5)
    return (k*blend*cc).rename('idio_erp')

def build_coe(ticker, px, spx, v4, epi, k=K_UNIVERSAL):
    idio=idio_erp_series(px,spx,v4['eff_tips_ry'],k)
    out=pd.DataFrame(index=idio.index)
    out['real_yield']=v4['eff_tips_ry'].reindex(out.index)
    out['market_erp']=v4['eff_erp'].reindex(out.index)
    out['idio_erp']=idio
    out['coe_real']=out['real_yield']+out['market_erp']+out['idio_erp']
    out['exp_infl']=epi.reindex(out.index)
    out['coe_nominal']=out['coe_real']+out['exp_infl']
    out['ticker']=ticker
    return out.dropna(subset=['coe_real'])

# ---- LOCAL demo helpers (staging only; the runner injects real data) ----
def load_shared():
    v4=pd.read_csv('/tmp/calib/FINAL_decomposition_v4_1877_2026.csv')
    v4['m']=pd.to_datetime(v4['date']).dt.to_period('M'); v4=v4.set_index('m')
    epi=pd.read_csv('/tmp/ryfetch/expected_inflation_termstructure_v2.csv')
    epi['m']=pd.PeriodIndex(epi['mkey'],freq='M'); epi=epi.set_index('m')['Epi_10y'].astype(float)
    spx=np.array([float(x) for x in open('spx_adj.csv').read().split(',')])
    spx=pd.Series(spx,index=pd.period_range('1980-12','2026-07',freq='M'))
    return v4,epi,spx

def load_ticker(ticker,start):
    a=np.array([float(x) for x in open(f'{ticker.lower()}_adj.csv').read().split(',')])
    return pd.Series(a,index=pd.period_range(start,periods=len(a),freq='M'))
