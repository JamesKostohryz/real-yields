#!/usr/bin/env python3
"""U.S. Real Yield Curve 1877-2026 (v2) — complete reproducible build.
Regenerates nominal curve, expected-inflation term structure, ex-ante & TIPS-like
real yields, and uncertainty bands from source files. See METHODS doc for methodology.
Requires: pandas, numpy, statsmodels, python-calamine. Set SRC/OUT and run."""
import numpy as np, pandas as pd, warnings, os
warnings.filterwarnings("ignore")
import statsmodels.api as sm

SRC = os.environ.get("SRC", "./repo_data")
OUT = os.environ.get("OUT", "./out")
os.makedirs(OUT, exist_ok=True)
F = {  # filename -> role
 "cpi_gfd":"GFD_CPI.xlsx","cpi_fred":"CPIAUCNS.csv","gsw":"feds200628.csv",
 "y1":"1Y.xlsx","y5":"5Y.xlsx","y10":"10Y.xlsx","y20":"20Y.xlsx","y30":"30Y.xlsx",
 "tbill":"3M_TBILL.xlsx","rates":"Interest_Rate_Data.xlsx","spf":"Inflation.xlsx",
 "groen":"Groen.xlsx","shiller":"ie_data.xls","add10":"Additional-CPIE10.xlsx"}
p = lambda k: f"{SRC}/{F[k]}"
def me(s): s=s.copy(); s.index=pd.to_datetime(s.index); return s.resample("ME").last()
def gfd(path,sheet,hdr=2):
    r=pd.read_excel(path,sheet_name=sheet,header=hdr,engine="calamine")[["Date","Close"]]
    r["d"]=pd.to_datetime(r["Date"],errors="coerce"); r["v"]=pd.to_numeric(r["Close"],errors="coerce")
    s=me(r.dropna(subset=["d","v"]).set_index("d")["v"]); s.index=s.index.to_period("M"); return s

# ---------- 1. CPI -> YoY inflation ----------
def load_infl():
    g=pd.read_excel(p("cpi_gfd"),sheet_name="Price Data",engine="calamine")[["Date","Close"]]
    g["date"]=pd.to_datetime(g["Date"],format="%m/%d/%Y",errors="coerce"); g["cpi"]=pd.to_numeric(g["Close"],errors="coerce")
    g=g.dropna(subset=["date","cpi"]); g=g[(g["date"].dt.year>=1875)&(g["date"].dt.year<=1912)].set_index("date")["cpi"].resample("ME").last()
    fr=pd.read_csv(p("cpi_fred")); fr.columns=["date","cpi"]; fr["date"]=pd.to_datetime(fr["date"]); fr=fr.set_index("date")["cpi"].resample("ME").last()
    scale=fr.asof(pd.Timestamp("1913-01-31"))/g.asof(pd.Timestamp("1912-12-31"))
    cpi=pd.concat([g*scale,fr]).sort_index(); cpi=cpi[~cpi.index.duplicated()]
    return (100*(cpi/cpi.shift(12)-1)).dropna()

# ---------- 2. Expected-inflation term structure (UCSV + regime) ----------
def kll(y,sig,q):
    n=len(y); a=np.zeros(n); P=np.zeros(n); ap=np.zeros(n); Pp=np.zeros(n); af=y[0]; Pf=1.0
    for t in range(n):
        H=sig[t]**2; Q=q*sig[t]**2; apr=af; Ppr=Pf+Q; ap[t]=apr; Pp[t]=Ppr
        K=Ppr/(Ppr+H); af=apr+K*(y[t]-apr); Pf=(1-K)*Ppr; a[t]=af; P[t]=Pf
    asm=a.copy()
    for t in range(n-2,-1,-1): C=P[t]/Pp[t+1]; asm[t]=a[t]+C*(asm[t+1]-ap[t+1])
    return asm
def exp_infl(infl,sp1,sp10):
    y=infl.asfreq("ME").interpolate(limit_area="inside"); yv=y.values; idx=y.index.to_period("M")
    sig=np.clip(pd.Series(np.abs(np.diff(yv,prepend=yv[0]))).ewm(halflife=24).mean().values,0.3,None)
    best=None
    for q in [0.001,0.003,0.006,0.01,0.02,0.04,0.08]:
        tr=pd.Series(kll(yv,sig,q),index=idx); j=pd.concat([tr.rename("m"),sp10.rename("s")],axis=1).dropna()
        m=(j["m"]-j["s"]).abs().mean(); best=(m,q) if best is None or m<best[0] else best
    trend=kll(yv,sig,best[1]); mu=trend[(idx.year>=1876)&(idx.year<=1913)].mean()
    yr=idx.year+(idx.month-1)/12; w=np.clip((yr-1933)/(1960-1933),0,1); endp=w*trend+(1-w)*mu; lvl=yv
    lam=lambda N,r: np.mean(r**np.arange(1,N+1)); best=None
    for r in [0.90,0.92,0.94,0.95,0.96,0.97,0.98]:
        E1=pd.Series(endp+lam(12,r)*(lvl-endp),index=idx); j=pd.concat([E1.rename("m"),sp1.rename("s")],axis=1).dropna()
        m=(j["m"]-j["s"]).abs().mean(); best=(m,r) if best is None or m<best[0] else best
    rho=best[1]; ei=pd.DataFrame(index=idx)
    for t,N in {1:12,5:60,10:120,20:240,30:360}.items(): ei[t]=endp+lam(N,rho)*(lvl-endp)
    return ei

# ---------- 3. Nominal curve (NS fit pre-1961, GSW Svensson 1961+, pre-1920 blend) ----------
def ns_load(tau,lam=1.4): x=tau/lam; L1=(1-np.exp(-x))/x; L2=L1-np.exp(-x); return np.array([1.,L1,L2])
def svensson(n,b0,b1,b2,b3,t1,t2):
    a=n/t1; y=b0+b1*(1-np.exp(-a))/a+b2*((1-np.exp(-a))/a-np.exp(-a))
    if pd.notna(b3) and b3!=0 and pd.notna(t2) and t2>0: c=n/t2; y+=b3*((1-np.exp(-c))/c-np.exp(-c))
    return y
def nominal_curve():
    bill=gfd(p("tbill"),"USA Government 90-day T-Bills",hdr=8)
    y1=gfd(p("y1"),"IGUSA1D"); y5=gfd(p("y5"),"IGUSA5D"); y10=gfd(p("y10"),"IGUSA10D")
    y20=gfd(p("y20"),"IGUSA20D"); y30=gfd(p("y30"),"IGUSA30D")
    cp=pd.read_excel(p("rates"),sheet_name="Commercial Paper 1831",header=1,engine="calamine"); cp.columns=[str(c).strip() for c in cp.columns]
    cp["date"]=pd.to_datetime(dict(year=pd.to_numeric(cp["Year"],errors="coerce"),month=pd.to_numeric(cp["Month"],errors="coerce"),day=1),errors="coerce")+pd.offsets.MonthEnd(0)
    cpm=pd.to_numeric(cp.dropna(subset=["date"]).set_index("date")["Yield"],errors="coerce").dropna(); cpm.index=cpm.index.to_period("M")
    credit=(pd.concat([cpm.rename("cp"),bill.rename("b")],axis=1).dropna().eval("cp-b")).mean()   # CP-bill basis
    idx=pd.period_range("1876-09","2026-06",freq="M")
    pan=pd.DataFrame(index=idx)
    pan["0.25"]=bill.reindex(idx).fillna((cpm-credit).reindex(idx))
    pan["1"]=y1.reindex(idx); pan["5"]=y5.reindex(idx); pan["10"]=y10.reindex(idx); pan["20"]=y20.reindex(idx); pan["30"]=y30.reindex(idx)
    tm={"0.25":0.375,"1":1,"5":5,"10":10,"20":20,"30":30}
    def fit(row,b2=None):
        pts=[(tm[k],row[k]) for k in tm if pd.notna(row[k])]
        if len(pts)<2: return None
        T=np.array([ns_load(t) for t,_ in pts]); yv=np.array([v for _,v in pts])
        if len(pts)>=3 and b2 is None: return np.linalg.lstsq(T,yv,rcond=None)[0]
        bb=0.0 if b2 is None else b2; return np.r_[np.linalg.lstsq(T[:,:2],yv-T[:,2]*bb,rcond=None)[0],bb]
    b2bar=np.nanmean([fit(pan.loc[q])[2] for q in idx if q<pd.Period("1961-08","M") and pan.loc[q].notna().sum()>=3 and fit(pan.loc[q]) is not None])
    # GSW Svensson params 1961+
    df=pd.read_csv(p("gsw"),skiprows=9,low_memory=False); df["Date"]=pd.to_datetime(df["Date"],errors="coerce"); df=df[df["Date"].notna()]
    for c in ["BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    mp=df.set_index("Date")[["BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]].resample("ME").last(); mp.index=mp.index.to_period("M")
    rows=[]
    for q in idx:
        if q>=pd.Period("1961-08","M"):
            r=mp.loc[q] if q in mp.index else None
            if r is None or pd.isna(r.BETA0): continue
            rows.append({"k":q,**{t:svensson(t,r.BETA0,r.BETA1,r.BETA2,r.BETA3,r.TAU1,r.TAU2) for t in [1,5,10,20,30]}})
        else:
            npts=int(pan.loc[q][list(tm)].notna().sum())
            if npts==0: continue
            b=fit(pan.loc[q],b2=(b2bar if npts<3 else None))
            if b is None: continue
            rows.append({"k":q,**{t:float(ns_load(t)@b) for t in [1,5,10,20,30]}})
    nom=pd.DataFrame(rows).set_index("k")
    # pre-1920 blend with Homer-Sylla (Shiller)
    d=pd.read_excel(p("shiller"),sheet_name="Data",header=7,engine="calamine")
    d=d.rename(columns={d.columns[0]:"df",d.columns[6]:"gs10"}); d["df"]=pd.to_numeric(d["df"],errors="coerce"); d=d[d["df"].notna()]
    yr=np.floor(d["df"]+1e-6).astype(int); mo=np.round((d["df"]-yr)*100).astype(int).clip(1,12)
    d["k"]=pd.to_datetime(dict(year=yr,month=mo,day=1)).dt.to_period("M"); d["gs10"]=pd.to_numeric(d["gs10"],errors="coerce")
    sh=d.dropna(subset=["gs10"]).set_index("k")["gs10"]
    gap=(sh.reindex(nom.index)-nom[10]); pre=nom.index<pd.Period("1920-01","M"); adj=(0.5*gap).where(pre,0).fillna(0)
    for t in [1,5,10,20,30]: nom[t]=nom[t]+adj
    nom.attrs["nom_sd"]=(0.5*gap.abs()).where(pre,0.0).fillna(0.0)
    return nom

# ---------- 4. assemble: real, IRP, TIPS-like, bands ----------
def build():
    infl=load_infl()
    spf=pd.read_excel(p("spf"),sheet_name="INFLATION",engine="calamine"); spf.columns=[c.strip() for c in spf.columns]
    spf["k"]=pd.PeriodIndex(pd.to_datetime(dict(year=spf["YEAR"],month=3*spf["QUARTER"]-1,day=1)),freq="M")
    sp1=spf.dropna(subset=["INFCPI1YR"]).set_index("k")["INFCPI1YR"]; sp10=spf.dropna(subset=["INFCPI10YR"]).set_index("k")["INFCPI10YR"]
    ei=exp_infl(infl,sp1,sp10); nom=nominal_curve(); nom_sd=nom.attrs["nom_sd"]
    idx=nom.index
    # IRP calibration (Groen breakeven - unified survey Eπ ≈ kappa*Eπ)
    g=pd.read_excel(p("groen"),header=14).rename(columns={0:"date"}); g=g.rename(columns={g.columns[0]:"date"})
    g["date"]=pd.to_datetime(g["date"],errors="coerce"); g=g[g["date"].notna()]; g["be"]=pd.to_numeric(g["10-yr breakeven backcast"],errors="coerce")
    g=g.assign(k=g["date"].dt.to_period("M")).set_index("k")["be"]
    # unified SURVEY 10y expected inflation: Blue Chip (Additional, pre-1991Q4) + SPF (1991Q4+)
    a=pd.read_excel(p("add10"),sheet_name="Sheet1",header=13,engine="calamine").dropna(how="all")
    a.columns=[str(c).strip() for c in a.columns]; a=a[a["Survey Date"].notna()]
    a[["YEAR","QUARTER"]]=a["Survey Date"].str.split(":",expand=True).astype(int); a["Combined"]=pd.to_numeric(a["Combined"],errors="coerce")
    a["k"]=pd.PeriodIndex(pd.to_datetime(dict(year=a["YEAR"],month=3*a["QUARTER"]-1,day=1)),freq="M")
    pre=a[a["YEAR"]*10+a["QUARTER"]<19914].dropna(subset=["Combined"]).set_index("k")["Combined"]
    uni=pd.concat([pre, sp10[sp10.index>=pd.Period("1991-11","M")]]).sort_index(); uni=uni[~uni.index.duplicated()]
    Xk=pd.concat([(g-uni).rename("irp"),uni.rename("e")],axis=1).dropna()   # IRP=breakeven-survey Eπ
    kappa=(Xk["irp"]*Xk["e"]).sum()/(Xk["e"]**2).sum(); resid=(Xk["irp"]-kappa*Xk["e"]).std()
    E10=ei[10]
    irp10=(kappa*E10).clip(lower=0)
    gfac={1:.45,5:.78,10:1.,20:1.13,30:1.2}
    imerr={1:.54,5:.48,10:.42,20:.42,30:.42}; mdiv={1:.23,5:.10,10:.02,20:.17,30:.22}
    out=pd.DataFrame(index=idx); out["date"]=idx.to_timestamp()
    yrs=idx.year
    for t in [1,5,10,20,30]:
        real=nom[t]-ei[t].reindex(idx)
        imu=imerr[t]*np.where(yrs<1940,1.8,np.where(yrs<1961,1.3,1.0))
        se=np.sqrt(nom_sd.values**2+imu**2+mdiv[t]**2)
        irp_t=gfac[t]*irp10.reindex(idx)
        irp_unc=gfac[t]*resid*np.where((yrs>=1971)&(yrs<=2013),1.0,1.8)
        out[f"nom{t}"]=nom[t]; out[f"real{t}_exante"]=real; out[f"real{t}_exante_se"]=se
        out[f"irp{t}"]=irp_t; out[f"real{t}_tips"]=real-irp_t; out[f"real{t}_tips_se"]=np.sqrt(se**2+irp_unc**2)
    out=out.dropna(subset=[f"real{t}_exante" for t in [1,5,10,20,30]],how="all")
    out.to_csv(f"{OUT}/real_yield_curve_v2.csv",index=False)
    print(f"built {len(out)} months x 5 tenors -> {OUT}/real_yield_curve_v2.csv")
    print(f"  kappa(IRP)={kappa:.3f}  coverage {out['date'].min().date()}..{out['date'].max().date()}")
    return out

if __name__=="__main__": build()
