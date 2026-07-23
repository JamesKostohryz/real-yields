"""
build_erp_daily.py  —  ERP/COE daily-close builder (v4 engine, DAILY cadence)
=============================================================================
Produces, for any as-of date from that day's inputs:
  (1) the 1..30 forward term structure  -> values individual stocks (+ per-name idio, added downstream)
  (2) its duration-collapsed EFFECTIVE (real rf / ERP / COE) -> values the S&P 500 index itself,
      = the live current-month observation of the monthly historical series (finalizes at month-end).

DESIGN (locked with James, 2026-07-22):
  * Daily-moving inputs: real-yield curve (Treasury daily real 5/7/10/20/30 + 1y short-end rule),
    and the normalized earnings yield (SP500 normalized earnings LEVEL / real price; price is daily).
  * Slow inputs, HELD between their native monthly updates and re-anchored each month:
    vol_scale (vs), fair_ey state, cost overlay, corp-premium floor, and the prior-month (D,fey) state.
  * The effective uses the INCOMING monthly state (fey_in, D_in) -- identical to the historical
    monthly engine, so the current-month point is consistent with the 1877-> series.
  * The term-structure snapshot uses the UPDATED fair_ey (fey_out) -- identical to how the committed
    TODAY curve was built.
  * Forward transform: zero->1y-forward bootstrap  f_t=(1+z_t)^t/(1+z_{t-1})^{t-1}-1  (engine convention).

ACCEPTANCE GATE: hermetic self-test below reproduces the committed June artifacts
(effective 2.349/3.887/6.236, dur 24.30; TODAY_forward_curve_2026-06 spot_coe) to <1bp, no external files.
"""
import pandas as pd, numpy as np
from scipy.stats import norm

# ---------- LOCKED v4 parameters (METHODOLOGY_effective_ERP_COE.md sec.12) ----------
R_NEUTRAL=2.0; H_CONV=20.0; VARP=3.0; C=7.5; VOLNORM=13.0
G_REAL=0.0175; D_LO=12.0; D_KNEE=30.0; D_MAX=60.0
LO,HI=0.5,4.5; BETA_IN=0.30; KMAX=5.0; SCALE=1.3; CASH_HURDLE=1.5
CORP_PREM_DEFAULT=1.8   # BAA-AAA credit file ends 2021; default floor, non-binding at current ERP
RVb={1:.195,2:.18,3:.172,5:.158,7:.148,10:.138,15:.126,20:.118,25:.112,30:.108}
def rvbase(T): ks=sorted(RVb); return float(np.interp(T,ks,[RVb[k] for k in ks]))
def gdecay(T): return float(np.interp(T,[1,10,20,30],[1.12,1.0,0.9,0.85]))
def ey_eff_avg(ey,T,fey):
    g0=ey-fey; return (fey+g0*(1-T/(2*H_CONV))) if T<=H_CONV else (fey+g0*H_CONV/(2*T))
def pund(eyv,yT,T,vs): return norm.cdf(-((eyv-yT)/100.0)*np.sqrt(T)/(rvbase(T)*vs))
def base_val_T(ey,yT,T,vs,fey):
    return VARP*vs*gdecay(T)+C*max(0.0,pund(ey_eff_avg(ey,T,fey),yT,T,vs)-pund(fey,yT,T,vs))
def Rresp(y):
    d=max(0.0,LO-y,y-HI); return BETA_IN*(R_NEUTRAL-y)+KMAX*(1-np.exp(-d/SCALE))
def dur(r):
    r=r/100.0; raw=max((1+r)/max(r-G_REAL,1e-4),D_LO)
    return float(raw if raw<=D_KNEE else D_KNEE+(D_MAX-D_KNEE)*(1-np.exp(-(raw-D_KNEE)/(D_MAX-D_KNEE))))
TMAX=120; Tg=np.arange(1,TMAX+1); Tclip=np.minimum(Tg,30)
_qg=np.linspace(0.55,1.28,4000); _mg=np.array([np.dot((q**Tg)/(q**Tg).sum(),Tg) for q in _qg])
_wc={round(d,1):(lambda q:(q**Tg)/(q**Tg).sum())(float(np.interp(np.clip(d,_mg[0],_mg[-1]),_mg,_qg))) for d in np.round(np.arange(D_LO,60.01,0.1),1)}
def wget(D): return _wc[round(float(np.clip(D,D_LO,60.0)),1)]
def cost_of_year(yr): return (1.5+(0.5-1.5)*((yr-1995)/(2026.5-1995))**1.3)

def fwd_from_spot(spot):   # zero -> 1y-forward bootstrap (engine convention)
    f=[]
    for i in range(len(spot)):
        if i==0: f.append(spot[0])
        else: f.append((1+spot[i])**(i+1)/(1+spot[i-1])**i-1)
    return np.array(f)

def build_asof(real_tips_5pt, norm_ey, vs, fey_in, D_in, cost, corp_prem=CORP_PREM_DEFAULT):
    """One daily step from the incoming monthly state. Returns effective + fwd term structure."""
    ks=[1,5,10,20,30]; yv=np.interp(Tclip,ks,[real_tips_5pt[k] for k in ks])
    w=wget(D_in); tips_eff=float(w@yv)
    bv=np.array([base_val_T(norm_ey,yv[i],Tclip[i],vs,fey_in) for i in range(TMAX)])
    bvc=float(w@bv); Rc=Rresp(tips_eff)
    erp_risk=max(corp_prem,bvc+Rc); eff_erp=erp_risk+cost; eff_coe=tips_eff+eff_erp
    D_out=0.6*D_in+0.4*dur(eff_coe); fey_out=0.7*fey_in+0.3*eff_coe
    # term-structure snapshot uses the UPDATED fair_ey (fey_out), common Rresp at the effective yield
    yvT=yv[:30].copy()
    erpT=np.array([max(corp_prem, base_val_T(norm_ey,yvT[i],float(Tclip[i]),vs,fey_out)+Rc)+cost for i in range(30)])
    coeT=yvT+erpT
    fr=fwd_from_spot(yvT/100.0)*100.0; fc=fwd_from_spot(coeT/100.0)*100.0; fe=fc-fr
    return dict(eff_tips=tips_eff,eff_erp=eff_erp,eff_coe=eff_coe,D_out=D_out,fey_out=fey_out,
                spot_real=yvT,spot_erp=erpT,spot_coe=coeT,fwd_real=fr,fwd_erp=fe,fwd_coe=fc)

# ---------- vol_scale helper (monthly re-anchor; NOT needed by the hermetic gate) ----------
def vol_scale_from_shiller(asof_month, path='/tmp/shiller/shiller.csv'):
    sh=pd.read_csv(path); sh['date']=pd.to_datetime(sh['Date'])
    for c in ['SP500','Dividend','Consumer Price Index']: sh[c]=pd.to_numeric(sh[c],errors='coerce')
    sh=sh.sort_values('date').reset_index(drop=True)
    P=sh['SP500'].values; Dv=sh['Dividend'].fillna(0).values; CPI=sh['Consumer Price Index'].ffill().values
    gg=np.ones(len(P))
    for t in range(1,len(P)):
        dm=(Dv[t-1]/12.0) if Dv[t-1]>0 else 0.0; gg[t]=(P[t]+dm)/P[t-1] if P[t-1]>0 else 1.0
    with np.errstate(divide='ignore',invalid='ignore'):
        rtv=np.cumprod(gg)/(CPI/CPI[0]); lr=pd.Series(np.log(rtv)).diff()
    rv=(lr.rolling(36).std()*np.sqrt(12)*100).bfill().ffill().values
    sh['rv']=rv; row=sh[sh.date==asof_month]
    return float(np.clip((row['rv'].iloc[0] if len(row) else 13.0)/VOLNORM,0.8,2.0))

# ================== ACCEPTANCE SELF-TEST (June 2026, HERMETIC — no external files) ==================
# Embedded June-2026 reference so the gate runs green in CI with zero file dependencies.
VS_JUNE=0.9348                    # vol_scale at 2026-06 (clip(rvol/13)); precomputed from Shiller
JUNE_TIPS={1:1.07,5:1.885,10:2.204,20:2.745,30:2.73}
JUNE_NORM_EY=3.138
JUNE_STATE=dict(fey_in=6.02, D_in=24.72, cost=0.503)          # May->June incoming state
JUNE_EFF=dict(eff_tips=2.349, eff_erp=3.887, eff_coe=6.236)   # committed effective
SPOT_COE_REF=[5.0710,5.4490,5.7660,6.0550,6.3290,6.4310,6.5180,6.5840,6.6390,6.6840,6.7140,6.7350,6.7490,6.7570,6.7580,6.7530,6.7430,6.7300,6.7130,6.6930,6.6090,6.5280,6.4500,6.3750,6.3040,6.2370,6.1730,6.1110,6.0530,5.9970]

def run_gate():
    r=build_asof(JUNE_TIPS, JUNE_NORM_EY, VS_JUNE, **JUNE_STATE)
    ok_eff = all(abs(r[k]-JUNE_EFF[k])<0.01 for k in JUNE_EFF)
    sp=max(abs(r['spot_coe'][i]-SPOT_COE_REF[i]) for i in range(30))   # canonical handoff (engine bootstraps forwards)
    print("SELF-TEST June: eff tips=%.3f erp=%.3f coe=%.3f dur=%.2f fey_out=%.3f"%(r['eff_tips'],r['eff_erp'],r['eff_coe'],r['D_out'],r['fey_out']))
    print("  effective ties (<1bp): %s"%ok_eff)
    print("  SPOT coe max|delta| vs embedded June ref = %.4f pp  (canonical handoff)"%sp)
    assert ok_eff and sp<0.01, "ACCEPTANCE FAILED"
    print("  ACCEPTANCE PASSED")

if __name__=='__main__':
    run_gate()
