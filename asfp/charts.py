"""
Rating-fan diagnostic chart (roadmap A3).

Situate an issuer's estimated debt term structure within the family of rating
curves (AAA..CCC), overlay the issuer's individual bonds coloured by their own
rating, and auto-flag bonds that imply a rating far from the issuer's — the
"inspect this" signal (secured/subordinated, callable YTW, liquidity, stale).

Consumes the pipeline credit grid (treasury_nominal + spread_<rating>), so the
chart is plotted in NOMINAL yield space — the space the bonds are quoted in.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
RATING_COLOR = {"AAA": "#1a7f4b", "AA": "#3f9e5a", "A": "#8bb43a", "BBB": "#e0a92e",
                "BB": "#e07b2e", "B": "#d0492e", "CCC": "#a01f2e"}


def nominal_family(cg, ratings=None):
    """Nominal yield curve per rating = treasury_nominal + spread_<rating>.
    `cg` is the credit grid DataFrame (index = tenor). Returns dict rating->array."""
    ratings = ratings or [r for r in RATING_ORDER if f"spread_{r}" in cg.columns]
    tsy = cg["treasury_nominal"].to_numpy()
    return {r: tsy + cg[f"spread_{r}"].to_numpy() for r in ratings}


def _coarse_rating(s):
    """Map 'BBB+', 'AA-', 'BB' ... to a coarse family bucket; None if unknown."""
    if not s:
        return None
    s = str(s).upper().strip().replace("+", "").replace("-", "")
    for r in RATING_ORDER:
        if s == r:
            return r
    if s in ("CC", "C", "D", "CCC"):
        return "CCC"
    return None


def fit_company_curve(tenors, family, bonds, modal_rating):
    """Company curve = modal-rating nominal curve x multiplicative median offset
    fitted from the issuer's own bonds (mirrors the deployed cost-of-debt tool)."""
    base = family[modal_rating]
    fit = np.interp(bonds["years"], tenors, base)
    ratio = (bonds["ytw"].to_numpy() * 100.0) / fit          # ytw decimal -> %
    offset = float(np.median(ratio[np.isfinite(ratio)]))
    return base * offset, offset


def implied_rating(year, ytw_pct, tenors, family):
    """Nearest rating curve to a bond, by yield distance at its tenor."""
    dists = {r: abs(ytw_pct - np.interp(year, tenors, fam)) for r, fam in family.items()}
    return min(dists, key=dists.get)


def rating_fan_chart(cg, bonds, issuer, out_path, illustrative_family=False,
                     max_annotations=3, min_resid=1.0):
    """Render the diagnostic fan. `cg`: credit grid; `bonds`: parsed issuer bonds.
    Returns a dict of flagged outliers."""
    tenors = cg.index.to_numpy()
    family = nominal_family(cg)
    fam_ratings = [r for r in RATING_ORDER if r in family]

    # issuer's modal coarse rating (fall back to BBB)
    coarse = [c for c in (bonds["sp_rating"].map(_coarse_rating)) if c]
    modal = max(set(coarse), key=coarse.count) if coarse else "BBB"
    if modal not in family:
        modal = "BBB" if "BBB" in family else fam_ratings[len(fam_ratings) // 2]

    company, offset = fit_company_curve(tenors, family, bonds, modal)

    # classify bonds & flag outliers. A bond is flagged only if it BOTH implies a
    # rating >=2 notches from the issuer's modal rating AND sits materially off the
    # issuer's own fitted curve (min_resid). The second condition kills short-end
    # false positives where the rating curves bunch together.
    idx = {r: i for i, r in enumerate(RATING_ORDER)}
    flags = []
    byr, ycol = bonds["years"].to_numpy(), bonds["ytw"].to_numpy() * 100.0
    for i in range(len(bonds)):
        imp = implied_rating(byr[i], ycol[i], tenors, family)
        notch = idx[imp] - idx[modal]
        resid = ycol[i] - float(np.interp(byr[i], tenors, company))
        if abs(notch) >= 2 and abs(resid) >= min_resid:
            flags.append(dict(
                cusip=bonds["cusip"].iloc[i], years=float(byr[i]), ytw=float(ycol[i]),
                implied=imp, modal=modal, notches=int(notch), resid=float(resid),
                desc=str(bonds["description"].iloc[i])[:46]))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(11.5, 7.4), dpi=150)
    # y-window: keep the fan legible; clamp extreme distressed bonds to the top
    ymax = float(min(np.nanmax(ycol) + 0.4, family[fam_ratings[-1]].max() + 1.5))
    ymin = float(min(np.nanmin(ycol), min(f.min() for f in family.values())) - 0.4)
    clamped = ycol > ymax
    yplot = np.where(clamped, ymax, ycol)

    for r in fam_ratings:
        ax.plot(tenors, family[r], color=RATING_COLOR[r], lw=1.5, alpha=0.9, zorder=2)
        ax.annotate(r, xy=(tenors[-1], family[r][-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=RATING_COLOR[r], fontweight="bold")
    ax.plot(tenors, company, color="#12233b", lw=3.4, zorder=5,
            label=f"{issuer} estimated curve  (x{offset:.2f} vs {modal})")

    for i in range(len(bonds)):
        c = _coarse_rating(bonds["sp_rating"].iloc[i])
        ax.scatter(byr[i], yplot[i], s=42, marker=("^" if clamped[i] else "o"),
                   facecolor=RATING_COLOR.get(c, "#12233b"),
                   edgecolor="white", linewidth=0.7, zorder=6)
    ax.scatter([], [], s=42, facecolor="#12233b", edgecolor="white",
               label=f"{issuer} bonds (YTW, coloured by S&P)")
    if clamped.any():
        ax.scatter([], [], s=42, marker="^", facecolor="#7a1020", edgecolor="white",
                   label=f"{int(clamped.sum())} distressed/legacy bonds above axis")

    # annotate only the few most extreme flags, staggered to avoid overlap
    top = sorted(flags, key=lambda f: abs(f["resid"]), reverse=True)[:max_annotations]
    for k, f in enumerate(top):
        yy = min(f["ytw"], ymax)
        ty = ymax - 0.3 - k * (ymax - ymin) * 0.10
        ax.annotate(f"implies {f['implied']} ({f['resid']:+.1f}pp) — inspect",
                    xy=(f["years"], yy), xytext=(tenors[-1] * 0.42, ty),
                    fontsize=8, color="#7a1020", va="center",
                    arrowprops=dict(arrowstyle="->", color="#7a1020", lw=1.0,
                                    connectionstyle="arc3,rad=0.15"))

    ax.set_title(f"{issuer} debt term structure within the rating family",
                 fontsize=13.5, fontweight="bold", color="#12233b", pad=12)
    ax.set_xlabel("Maturity (years)"); ax.set_ylabel("Nominal yield to worst (%)")
    ax.set_xlim(0, tenors[-1] + 3); ax.set_ylim(ymin, ymax + 0.3)
    ax.grid(True, color="#e6e9ef", lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9.3)
    note = []
    if flags:
        note.append(f"{len(flags)} of {len(bonds)} bonds imply a rating "
                    f"≥2 notches off {modal} — inspect (subordinated/subsidiary, callable, illiquid).")
    if illustrative_family:
        note.append("Rating-family levels illustrative; issuer bonds are real data.")
    if note:
        ax.text(0.5, -0.12, "  ".join(note), transform=ax.transAxes, ha="center",
                fontsize=8.5, color="#8792a6")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dict(issuer=issuer, modal_rating=modal, offset=offset, n_flagged=len(flags),
                n_clamped=int(clamped.sum()), flags=flags)
