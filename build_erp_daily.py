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
    Since 2026-08-18 `vs` is a 30-long TERM STRUCTURE vs(T), not a scalar; scalars are still
    accepted and behave exactly as before. See _vs_at() and vol_scale_v3.py section (4).
  * The effective uses the INCOMING monthly state (fey_in, D_in) -- identical to the historical
    monthly engine, so the current-month point is consistent with the 1877-> series.
  * The term-structure snapshot uses the UPDATED fair_ey (fey_out) -- identical to how the committed
    TODAY curve was built.
  * Forward transform: zero->1y-forward bootstrap  f_t=(1+z_t)^t/(1+z_{t-1})^{t-1}-1  (engine convention).

ACCEPTANCE GATE: hermetic self-test below, no external files and no network. Current dated
reference 2026-08-18 (2.349/3.472/5.821, preset B, vs(T) from VIX1Y=22.94); the superseded June
scalar reference (2.349/3.400/5.748) is retained as an executable BACK-COMPATIBILITY assertion,
so a change to the engine is distinguishable from a change to the input.
"""
import pandas as pd, numpy as np
from scipy.stats import norm

# ---------- LOCKED v4 parameters (METHODOLOGY_effective_ERP_COE.md sec.12) ----------
R_NEUTRAL=2.0; H_CONV=20.0; VARP=3.0; C=7.5; VOLNORM=13.0
G_REAL=0.0175; D_LO=12.0; D_KNEE=30.0; D_MAX=60.0
LO,HI=0.5,4.5; BETA_IN=0.30; KMAX=5.0; SCALE=1.3; CASH_HURDLE=1.5
CORP_PREM_DEFAULT=1.8   # BAA-AAA credit file ends 2021; default floor, non-binding at current ERP
# Plateau presets (AEG-ERP-TASK6-BUILD-SPEC-2026-08-12.md sec.4; landed 2026-08-12).
# Pure-risk long-run targets the curve blends toward as T grows. Front end (T<=3, option-
# implied) is untouched by any preset; the blend ramps in from T=3 to T=30 and is full-
# weight (100% preset) at T>=30 and beyond, matching how gdecay/gap_decay already go flat
# past year 30. corp_prem stays a separate, lower hard floor underneath the blended value
# (Task 6 sec.3 -- the floor and the preset are two different numbers, not one).
PLATEAU_PRESETS={"A":3.35,"B":2.40,"C":2.05}   # pure-risk plateau, total = +cost (~0.50 today)
PLATEAU_DEFAULT="B"
def plateau_w(T): return float(np.interp(T,[1,3,10,20,30],[0.0,0.0,0.35,0.75,1.0]))
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

# ---------- vs may be a SCALAR or a 30-long vs(T) term structure ----------
# Landed 2026-08-18 (session 17), approved by James. Before this date `vs` was always one number
# charged at every tenor 1..30 and the only cross-tenor variation in the engine was gdecay(T), a
# hard-coded shape that decayed 15% from year 1 to year 30 regardless of the market state. vs(T)
# makes that shape state-dependent. Construction and every caveat: vol_scale_v3.py section (4).
# BACKWARD COMPATIBILITY IS EXACT, NOT APPROXIMATE: a scalar `vs` takes the float() branch below
# and every downstream line is unchanged, so any pre-2026-08-18 call reproduces bit-for-bit. The
# hermetic gate keeps a scalar reference alongside the new vector one for precisely this reason.
def _vs_at(vs, T):
    """The vs to charge at tenor T. Scalar -> itself at every tenor. Vector -> element T, with
    tenors past 30 reusing the 30-year value, which is the same Tclip=min(T,30) convention the
    rest of the engine already uses for the 120-point grid."""
    if np.ndim(vs) == 0:
        return float(vs)
    v = np.asarray(vs, dtype=float)
    if v.shape[0] != 30:
        raise ValueError(f"vs(T) must be a 30-long vector (tenors 1..30); got shape {v.shape}")
    return float(v[min(int(T), 30) - 1])


def build_asof(real_tips_5pt, norm_ey, vs, fey_in, D_in, cost, corp_prem=CORP_PREM_DEFAULT, preset=PLATEAU_DEFAULT):
    """One daily step from the incoming monthly state. Returns effective + fwd term structure."""
    if preset not in PLATEAU_PRESETS: raise KeyError(f"unknown preset {preset!r}; use one of {list(PLATEAU_PRESETS)}")
    preset_val=PLATEAU_PRESETS[preset]
    ks=[1,5,10,20,30]; yv=np.interp(Tclip,ks,[real_tips_5pt[k] for k in ks])
    w=wget(D_in); tips_eff=float(w@yv)
    bv=np.array([base_val_T(norm_ey,yv[i],Tclip[i],_vs_at(vs,Tclip[i]),fey_in) for i in range(TMAX)])
    bv_blend=np.array([(1-plateau_w(Tg[i]))*bv[i]+plateau_w(Tg[i])*preset_val for i in range(TMAX)])
    bvc=float(w@bv_blend); Rc=Rresp(tips_eff)
    erp_risk=max(corp_prem,bvc+Rc); eff_erp=erp_risk+cost; eff_coe=tips_eff+eff_erp
    D_out=0.6*D_in+0.4*dur(eff_coe); fey_out=0.7*fey_in+0.3*eff_coe
    # term-structure snapshot uses the UPDATED fair_ey (fey_out), common Rresp at the effective yield
    yvT=yv[:30].copy()
    erpT=np.array([max(corp_prem, (1-plateau_w(i+1))*base_val_T(norm_ey,yvT[i],float(Tclip[i]),_vs_at(vs,Tclip[i]),fey_out)+plateau_w(i+1)*preset_val+Rc)+cost for i in range(30)])
    coeT=yvT+erpT
    fr=fwd_from_spot(yvT/100.0)*100.0; fc=fwd_from_spot(coeT/100.0)*100.0; fe=fc-fr
    return dict(eff_tips=tips_eff,eff_erp=eff_erp,eff_coe=eff_coe,D_out=D_out,fey_out=fey_out,
                spot_real=yvT,spot_erp=erpT,spot_coe=coeT,fwd_real=fr,fwd_erp=fe,fwd_coe=fc,
                preset=preset,preset_pure_risk=preset_val)

# ---------- vol_scale helper (monthly re-anchor; NOT needed by the hermetic gate) ----------
# SUPERSEDED 2026-08-18 (session 13, approved by James): this Shiller-semi-deviation method
# with its flat [0.8,2.0] clip is being replaced at the NEXT monthly re-anchor by
# vol_scale_v3.vol_scale_from_vix1y() -- VIX1Y as sole primary, normalized by a fixed
# full-record median (22.62), soft-clipped (knees 0.70/1.55, asymptotes 0.40/2.50) instead of
# flat-stopped, sourced via a six-tier chain with alarms (see vol_scale_v3.py and
# AEG-Project/docs/AEG-Market-VolScale-*-2026-08-18.md for the full rationale and evidence).
# Kept here unmodified for audit/comparison -- the committed June-2026 reference below
# (VS_JUNE=0.9348) was produced by THIS function and must keep reproducing exactly; nothing
# about this replacement is retroactive.
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

# ================== ACCEPTANCE SELF-TEST (HERMETIC — no external files, no network) ==================
# Embedded references so the gate runs green in CI with zero file dependencies.
JUNE_TIPS={1:1.07,5:1.885,10:2.204,20:2.745,30:2.73}
JUNE_NORM_EY=3.138
JUNE_STATE=dict(fey_in=6.02, D_in=24.72, cost=0.503)          # May->June incoming state

# ---------------------------------------------------------------- LEGACY reference (SUPERSEDED)
# Kept EXECUTABLE, not merely kept in a comment. Its job is no longer to state what the engine
# publishes -- it is the backward-compatibility assertion that the scalar code path is still
# bit-identical, so that nothing which used the old path can have silently changed underneath the
# vs(T) landing. If this ever fails, the engine moved, not the input.
VS_JUNE=0.9348                    # vol_scale at 2026-06 from vol_scale_from_shiller (SUPERSEDED)
JUNE_EFF=dict(eff_tips=2.349, eff_erp=3.400, eff_coe=5.748)   # committed effective, PRESET B
SPOT_COE_REF=[5.0490,5.4178,5.7276,5.9352,6.1229,6.1375,6.1380,6.1209,6.0964,6.0667,6.0419,6.0145,5.9859,5.9571,5.9291,5.9021,5.8780,5.8575,5.8411,5.8292,5.7816,5.7385,5.6996,5.6647,5.6336,5.6063,5.5823,5.5614,5.5436,5.5284]

# ---------------------------------------------------------------- CURRENT reference, 2026-08-18
# THE LANDING. This is the first change in the vol_scale sequence that MOVES A PUBLISHED NUMBER.
#
#   ACCEPTANCE REFERENCE CHANGE LOG
#   2026-06-01   2.349 / 3.400 / 5.748   scalar vs=0.9348 from vol_scale_from_shiller (36-month
#                                        trailing standard deviation of real S&P total returns
#                                        / 13.0, hard-clipped to [0.80, 2.00]).  SUPERSEDED.
#   2026-08-18   2.349 / 3.472 / 5.821   vs(T) term structure from VIX1Y=22.94, anchor-clipped,
#                                        frozen past year 10.  Session 17, approved by James.
#
#   THE MOVE, DECOMPOSED (identical June inputs throughout; ONLY vs changes; the three parts are
#   nested and sum to the total with zero residual -- tools/volscale_landing_decomposition.py):
#       source change   Shiller -> VIX1Y/22.62 soft-clipped ........ +0.0801 pp
#       asymptote       lower asymptote 0.40 -> 0.5283 ............. +0.0000 pp  (see note)
#       tenor reshaping scalar -> anchor-clipped vs(T) ............. -0.0078 pp
#       TOTAL eff_erp 3.3999 -> 3.4722 ............................. +0.0723 pp
#       eff_coe 5.7484 -> 5.8207.
#   The asymptote contributes zero AT THIS VIX1Y only, because 22.94/22.62 = 1.014 sits inside the
#   knees where the clip is the identity. It is NOT zero in general and the brief's claim that it
#   is "free in sample" is wrong: raising the lower asymptote steepens the whole tanh arm below the
#   0.70 knee, so it moves 120 of 4,931 historical days (2.43%), all calm ones, by up to +0.0091 pp.
#
#   WHAT THIS REFERENCE DELIBERATELY MIXES, STATED RATHER THAN BURIED: it pairs the JUNE-2026 real
#   curve, normalized earnings yield and incoming monthly state with the AUGUST-18 VIX1Y, because
#   holding everything else at June is what isolates the vs change. It is a regression gate, NOT a
#   valuation of any date, and must never be quoted as one.
VS_AUG_VIX1Y=22.94                # CBOE VIX1Y close 2026-08-18, the landing date
VS_AUG=[1.0141467728,1.0108379625,1.0086373506,1.0070864667,1.0059468983,1.0050825747,
        1.0044103262,1.0038765815,1.0034453816,1.0030917752,1.0030917752,1.0030917752,
        1.0030917752,1.0030917752,1.0030917752,1.0030917752,1.0030917752,1.0030917752,
        1.0030917752,1.0030917752,1.0030917752,1.0030917752,1.0030917752,1.0030917752,
        1.0030917752,1.0030917752,1.0030917752,1.0030917752,1.0030917752,1.0030917752]
AUG_EFF=dict(eff_tips=2.349, eff_erp=3.472, eff_coe=5.821)    # PRESET B
SPOT_COE_REF_AUG=[5.2855,5.6299,5.9240,6.1106,6.2807,6.2817,6.2707,6.2434,6.2097,6.1715,6.1411,6.1083,6.0743,6.0400,6.0064,5.9734,5.9431,5.9162,5.8932,5.8746,5.8235,5.7766,5.7337,5.6944,5.6588,5.6266,5.5977,5.5718,5.5488,5.5284]

def run_gate():
    # --- CURRENT: the landed vs(T) reference
    r=build_asof(JUNE_TIPS, JUNE_NORM_EY, VS_AUG, **JUNE_STATE)
    ok_eff = all(abs(r[k]-AUG_EFF[k])<0.01 for k in AUG_EFF)
    sp=max(abs(r['spot_coe'][i]-SPOT_COE_REF_AUG[i]) for i in range(30))
    print("SELF-TEST 2026-08-18 vs(T): eff tips=%.3f erp=%.3f coe=%.3f dur=%.2f fey_out=%.3f"%(r['eff_tips'],r['eff_erp'],r['eff_coe'],r['D_out'],r['fey_out']))
    print("  effective ties (<1bp): %s"%ok_eff)
    print("  SPOT coe max|delta| vs embedded 2026-08 ref = %.4f pp  (canonical handoff)"%sp)
    assert ok_eff and sp<0.01, "ACCEPTANCE FAILED (2026-08-18 vs(T) reference)"

    # --- LEGACY: the scalar path must still be bit-identical
    rl=build_asof(JUNE_TIPS, JUNE_NORM_EY, VS_JUNE, **JUNE_STATE)
    ok_l = all(abs(rl[k]-JUNE_EFF[k])<0.01 for k in JUNE_EFF)
    spl=max(abs(rl['spot_coe'][i]-SPOT_COE_REF[i]) for i in range(30))
    print("BACK-COMPAT June scalar: eff tips=%.3f erp=%.3f coe=%.3f   ties: %s, spot max|delta| = %.4f pp"%(rl['eff_tips'],rl['eff_erp'],rl['eff_coe'],ok_l,spl))
    assert ok_l and spl<0.01, "BACK-COMPAT FAILED (scalar path moved)"

    # --- a constant vs(T) vector must equal the scalar EXACTLY, not approximately
    rc=build_asof(JUNE_TIPS, JUNE_NORM_EY, [VS_JUNE]*30, **JUNE_STATE)
    d=abs(rc['eff_erp']-rl['eff_erp'])
    print("  constant vs(T) vector == scalar: |delta| = %.2e pp"%d)
    assert d==0.0, "VECTOR/SCALAR PATHS DIVERGED"

    # --- preset invariance: a vs move is worth the same pp under A, B and C
    dd=[build_asof(JUNE_TIPS,JUNE_NORM_EY,VS_AUG,preset=p,**JUNE_STATE)['eff_erp']
        -build_asof(JUNE_TIPS,JUNE_NORM_EY,VS_JUNE,preset=p,**JUNE_STATE)['eff_erp'] for p in ("A","B","C")]
    print("  preset invariance A/B/C: %+.9f %+.9f %+.9f  spread %.2e"%(dd[0],dd[1],dd[2],max(dd)-min(dd)))
    assert max(dd)-min(dd)<5e-10, "PRESET INVARIANCE BROKEN"
    print("  ACCEPTANCE PASSED")

if __name__=='__main__':
    run_gate()
