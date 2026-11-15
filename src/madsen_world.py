"""
madsen_world.py - narrow-vein (Madsen-class) world generator.

Longitudinal-section representation of a sub-vertical vein system:
the 2D grid is the vein PLANE (x = strike, z = elevation). Two coupled
fields live on it:
    TH(x,z) : true vein thickness (m); pinch-and-swell, zero in gaps/dykes
    G(x,z)  : in-vein gold grade (g/t); high nugget, shoot-controlled

Truth statistics follow the SRK-2023 / WRLG re-model of Madsen: narrow
(~<=2-3 m) discontinuous structures, plunging high-grade shoots, barren
dykes; honest cutoff ~7.4 g/t at planning price and realistic dilution/costs.

The PLANNING model (interpretation A) is built from the SAME pierce-point
drilling the way the 2019 FS model was: long-range smooth interpolation,
generous width flooring, envelope connection -- reproducing the documented
overstatement (reserve ~9 g/t at 4.75 g/t cutoff with 13 % assumed dilution).

Everything is validated against the calibration sheet at the bottom.
"""

import numpy as np

# ----------------------------------------------------------------------
# Grid: longitudinal section of one vein system (a "zone" of the mine)
# ----------------------------------------------------------------------
NX, NZ = 96, 40
DX, DZ = 4.0, 4.0                    # 384 m strike x 160 m vertical
LX, LZ = NX * DX, NZ * DZ
SX, SZL = 16, 8                      # stopes: 24 m strike x 20 m height
CX, CZ = NX // SX, NZ // SZL
xs = (np.arange(NX) + 0.5) * DX
zs = (np.arange(NZ) + 0.5) * DZ
XX, ZZ = np.meshgrid(xs, zs, indexing="ij")
COORDS = np.column_stack([XX.ravel(), ZZ.ravel()])
NCELL = NX * NZ
SG = 2.80                            # t/m3

# ----------------------------------------------------------------------
# Economics (calibration sheet §1, §2)
# ----------------------------------------------------------------------
FX = 0.75                            # US$ per C$
PRICE_PLAN_US = 1275.0               # US$/oz (2019 FS basis)
PRICE_REAL_US = 1800.0               # US$/oz (realized 2021-22 era)
RECOVERY = 0.96
GRAMS_PER_OZ = 31.1035

def rev_per_gram_C(price_us):
    return price_us / GRAMS_PER_OZ / FX * RECOVERY      # C$ per in-situ gram

REV_PLAN = rev_per_gram_C(PRICE_PLAN_US)                # ~52.5 C$/g
REV_REAL = rev_per_gram_C(PRICE_REAL_US)                # ~74.1 C$/g

# unit operating costs (C$/t milled): belief (FS) vs truth (Macassa class)
COST_BELIEF = {"LHOS": 205.0, "CF": 250.0}              # blends to ~C$223/t
COST_TRUE   = {"LHOS": 285.0, "CF": 330.0}              # blends to ~C$305/t

CUTOFF_FS = 4.75                     # g/t, FS cutoff (Austin/S.Austin/8 Zone)
DIL_PLANNED = 0.13                   # planner's assumed total dilution

# geometric dilution truth: overbreak per wall (m), by method
ELOS_TRUE = {"LHOS": 0.28, "CF": 0.13}
MIN_WIDTH = {"LHOS": 2.0, "CF": 2.0}                    # m, minimum mining width

# ----------------------------------------------------------------------
# Geostatistical machinery (shared Cholesky factors)
# ----------------------------------------------------------------------
def _cov(a, b, rx, rz, plunge_deg):
    ux, uz = np.cos(np.deg2rad(plunge_deg)), np.sin(np.deg2rad(plunge_deg))
    dx = a[:, None, 0] - b[None, :, 0]
    dz = a[:, None, 1] - b[None, :, 1]
    ha = (dx * ux + dz * uz) / rx
    hb = (-dx * uz + dz * ux) / rz
    return np.exp(-3.0 * (ha ** 2 + hb ** 2))


class Machinery:
    """Cholesky factors for the structured parts of thickness and grade,
    plus long-range factors for smooth 'interpretation A' style conditioning."""

    PLUNGE = -55.0                    # Red Lake style steep shoot plunge

    def __init__(self):
        jitter = 1e-8 * np.eye(NCELL)
        # thickness: smooth-ish structure, ranges 55 m along plunge, 16 m across
        self.L_th = np.linalg.cholesky(
            _cov(COORDS, COORDS, 55, 16, self.PLUNGE) + jitter)
        # grade structured part: 32 m x 9 m, HIGH relative nugget added later
        self.L_g = np.linalg.cholesky(
            _cov(COORDS, COORDS, 32, 9, self.PLUNGE) + jitter)
        # structure PRESENCE: long along plunge, tight across (connectivity field)
        self.L_p = np.linalg.cholesky(
            _cov(COORDS, COORDS, 75, 13, self.PLUNGE) + jitter)


_M = None
def machinery():
    global _M
    if _M is None:
        _M = Machinery()
    return _M

# ----------------------------------------------------------------------
# Truth generator (interpretation-B-like statistics: SRK/WRLG world)
# ----------------------------------------------------------------------
COVERAGE = 0.42                      # fraction of vein plane with structure
TH_BASE, TH_LN_MED, TH_LN_SIG = 0.9, 1.35, 0.50   # th = base + lognormal (m)
NUGGET_G = 0.50                      # relative nugget on grade field
MU_LOG_G, SIG_LOG_G = np.log(6.5), 0.75    # in-vein grade lognormal


def _shoot_trend(rng, n_shoots):
    """Plunging high-GT shoots: additive ridges in latent space."""
    m = machinery()
    ux, uz = np.cos(np.deg2rad(m.PLUNGE)), np.sin(np.deg2rad(m.PLUNGE))
    t = np.zeros(NCELL)
    for _ in range(n_shoots):
        x0 = rng.uniform(40, LX - 40)
        z0 = rng.uniform(50, LZ - 20)
        amp = rng.uniform(0.9, 1.5)
        length = rng.uniform(90, 150)
        width = rng.uniform(16, 26)
        dx, dz = COORDS[:, 0] - x0, COORDS[:, 1] - z0
        al = dx * ux + dz * uz
        ac = -dx * uz + dz * ux
        t += amp * np.exp(-(al / length) ** 2 - (ac / width) ** 2)
    return t


def _dykes(rng, n):
    """Barren dyke corridors crossing the section; TH -> 0 inside."""
    mask = np.zeros(NCELL, dtype=bool)
    dykes = []
    for _ in range(n):
        x0 = rng.uniform(30, LX - 30)
        dip = rng.uniform(60, 85) * rng.choice([-1, 1])
        half_w = rng.uniform(4, 7)
        ux, uz = np.sin(np.deg2rad(dip)), np.cos(np.deg2rad(dip))
        d = np.abs((COORDS[:, 0] - x0) * uz - (COORDS[:, 1] - 0) * ux * 0
                   + (COORDS[:, 0] - x0) * 0 - (COORDS[:, 1] - LZ/2) * ux)
        # distance from line through (x0, LZ/2) with direction (ux, uz):
        d = np.abs(-(COORDS[:, 0] - x0) * uz + (COORDS[:, 1] - LZ / 2) * ux)
        mask |= d < half_w
        dykes.append((x0, dip, half_w))
    return mask, dykes


def make_truth(seed):
    m = machinery()
    rng = np.random.default_rng(seed)
    shoots = _shoot_trend(rng, rng.integers(2, 4))

    # structure presence: anisotropic indicator, shoots raise local presence
    p_lat = m.L_p @ rng.standard_normal(NCELL) + 0.85 * (shoots - shoots.mean())
    tau = np.quantile(p_lat, 1 - COVERAGE)
    present = p_lat > tau
    # thickness where present: base + lognormal, swelling inside shoots
    th_f = m.L_th @ rng.standard_normal(NCELL)
    TH = np.where(present,
                  TH_BASE + TH_LN_MED * np.exp(TH_LN_SIG * th_f
                                               + 0.25 * (shoots - shoots.mean())),
                  0.0)
    TH = np.minimum(TH, 6.0)

    g_struct = m.L_g @ rng.standard_normal(NCELL)
    g_lat = (np.sqrt(1 - NUGGET_G) * g_struct
             + np.sqrt(NUGGET_G) * rng.standard_normal(NCELL)
             + 0.55 * shoots)
    G = np.exp(MU_LOG_G + SIG_LOG_G * (g_lat - 0.55 * shoots.mean()))
    G[TH <= 0] = 0.0

    dyke_mask, dykes = _dykes(rng, rng.integers(2, 4))
    TH[dyke_mask] = 0.0
    G[dyke_mask] = 0.0

    return {
        "TH": TH.reshape(NX, NZ), "G": G.reshape(NX, NZ),
        "shoots": shoots.reshape(NX, NZ), "dykes": dykes,
        "dyke_mask": dyke_mask.reshape(NX, NZ), "rng": rng,
    }

# ----------------------------------------------------------------------
# Pierce-point drilling (surface program, FS era)
# ----------------------------------------------------------------------
PIERCE_DX_M, PIERCE_DZ_M = 32.0, 28.0        # ~30 m pattern (Indicated class)
TH_OBS_REL_ERR = 0.15
G_OBS_REL_SD = 0.35                          # nugget + tiny support


def drill_pierce_points(world, rng=None):
    rng = rng or world["rng"]
    ixs = np.arange(3, NX - 2, int(round(PIERCE_DX_M / DX)))
    izs = np.arange(2, NZ - 1, int(round(PIERCE_DZ_M / DZ)))
    pts = []
    for ix in ixs:
        for iz in izs:
            th = world["TH"][ix, iz]
            g = world["G"][ix, iz]
            th_o = max(th * (1 + rng.normal(0, TH_OBS_REL_ERR)), 0.0)
            g_o = (g * np.exp(rng.normal(0, G_OBS_REL_SD)
                              - 0.5 * G_OBS_REL_SD ** 2)) if th > 0 else 0.0
            pts.append((ix, iz, th_o, g_o))
    return pts

# ----------------------------------------------------------------------
# Interpretation A: the 2019-FS-style smooth model from pierce points
#   - long-range interpolation (connects everything)
#   - width floored to minimum mining width inside a lenient envelope
#   - grade by inverse-distance (preserves highs across large volumes)
# ----------------------------------------------------------------------
A_RANGE_X, A_RANGE_Z = 130.0, 45.0            # over-long ranges


def interpretation_A(pierce):
    P = np.array([(xs[ix], zs[iz]) for (ix, iz, _, _) in pierce])
    th_o = np.array([p[2] for p in pierce])
    g_o = np.array([p[3] for p in pierce])

    C_dd = _cov(P, P, A_RANGE_X, A_RANGE_Z, Machinery.PLUNGE) + 1e-6 * np.eye(len(P))
    C_xd = _cov(COORDS, P, A_RANGE_X, A_RANGE_Z, Machinery.PLUNGE)
    W = C_xd @ np.linalg.inv(C_dd)

    TH_A = np.clip(W @ th_o, 0, None)
    env = TH_A > 1.15                                   # lenient ore envelope
    TH_A = np.where(env, np.maximum(TH_A, MIN_WIDTH["LHOS"]), 0.0)

    # grade: IDW in anisotropic distance, mineralized pierce points only
    ux, uz = np.cos(np.deg2rad(Machinery.PLUNGE)), np.sin(np.deg2rad(Machinery.PLUNGE))
    keep = th_o > 0.2
    Pk, gk = P[keep], np.minimum(g_o[keep], 28.0)      # top-cut, per real MREs
    dx = COORDS[:, None, 0] - Pk[None, :, 0]
    dz = COORDS[:, None, 1] - Pk[None, :, 1]
    da = (dx * ux + dz * uz) / A_RANGE_X
    db = (-dx * uz + dz * ux) / A_RANGE_Z
    dist = np.sqrt(da ** 2 + db ** 2) + 1e-4
    Wg = 1.0 / dist ** 2
    G_A = (Wg @ gk) / Wg.sum(axis=1)
    G_A = np.where(env, G_A, 0.0)
    return TH_A.reshape(NX, NZ), G_A.reshape(NX, NZ)

# ----------------------------------------------------------------------
# Stope aggregation + geometric dilution economics
# ----------------------------------------------------------------------
def stope_vein_stats(TH, G):
    """Per stope: mean thickness over mineralized cells, vein tonnes,
    tonnes-weighted vein grade."""
    th = TH.reshape(SX, CX, SZL, CZ)
    g = G.reshape(SX, CX, SZL, CZ)
    area = DX * DZ
    vt = th * area * SG / 1000.0                       # kt per cell strip
    VT = vt.sum(axis=(1, 3))                           # vein kt per stope
    metal = (vt * g).sum(axis=(1, 3))
    VG = np.divide(metal, VT, out=np.zeros_like(VT), where=VT > 1e-9)
    TH_bar = np.divide((th * (th > 0)).sum(axis=(1, 3)),
                       np.maximum((th > 0).sum(axis=(1, 3)), 1))
    frac_min = (th > 0).mean(axis=(1, 3))              # mineralized fraction
    return VT, VG, TH_bar, frac_min


def diluted(VT, VG, TH_bar, method, elos=None, planned=False):
    """Geometric dilution: mined width = max(th, min_width) + 2*ELOS.
    planned=True applies the FS's flat 13% instead of geometry."""
    if planned:
        DT = VT * (1 + DIL_PLANNED)
        DG = VG / (1 + DIL_PLANNED)
        return DT, DG
    e = ELOS_TRUE[method] if elos is None else elos
    w_mined = np.maximum(TH_bar, MIN_WIDTH[method]) + 2 * e
    factor = np.divide(w_mined, np.maximum(TH_bar, 1e-6))
    factor = np.clip(factor, 1.0, 3.5)
    DT = VT * factor
    DG = VG / factor
    return DT, DG


def stope_realized(TH, G, design_env, method):
    """Per-cell realized mining of the DESIGN envelope against the TRUTH.
    Returns per-stope: mined kt, metal kg, plus loss buckets:
      overbreak kt (ELOS walls on real vein), gap kt (barren cells inside
      the design envelope, mined at design width at ~zero grade)."""
    e = ELOS_TRUE[method]
    minw = MIN_WIDTH[method]
    area = DX * DZ
    th = TH.ravel(); g = G.ravel(); env = design_env.ravel()
    mined_w = np.where(th > 0, np.maximum(th, minw) + 2 * e, minw + 2 * e)
    mined_w = np.where(env, mined_w, 0.0)              # only design is mined
    mined_t = mined_w * area * SG / 1000.0             # kt per cell
    ore_t = np.where(env, th, 0.0) * area * SG / 1000.0
    metal = ore_t * g                                   # kt * g/t = kg
    over_t = np.where(env & (th > 0), mined_t - ore_t, 0.0)
    gap_t = np.where(env & (th <= 0), mined_t, 0.0)

    def agg(v):
        return v.reshape(SX, CX, SZL, CZ).sum(axis=(1, 3))
    return agg(mined_t), agg(metal), agg(ore_t), agg(over_t), agg(gap_t)


def stope_profit_kC(DT_kt, DG, method, rev_per_g, cost_table):
    rev = DT_kt * 1000 * DG * rev_per_g
    cost = DT_kt * 1000 * cost_table[method]
    return (rev - cost) / 1000.0                       # kC$


# ----------------------------------------------------------------------
# Ensemble of interpretations (for the agents; A-family and B-family)
# ----------------------------------------------------------------------
def make_ensemble(pierce, n_real=60, frac_A=0.4, seed=0):
    """Conditional realizations honoring pierce data under BOTH families.
    B-family: truth-process statistics (narrow, discontinuous, own shoots).
    A-family: long-range smooth statistics (Ginto-like).
    Conditioning by kriging-residual substitution per family."""
    m = machinery()
    rng = np.random.default_rng(seed)
    P = np.array([(xs[ix], zs[iz]) for (ix, iz, _, _) in pierce])
    idx = np.array([ix * NZ + iz for (ix, iz, _, _) in pierce])
    th_o = np.array([p[2] for p in pierce])
    g_o = np.array([p[3] for p in pierce])
    g_lat_o = np.where(th_o > 0.2,
                       (np.log(np.maximum(g_o, 0.1)) - MU_LOG_G) / SIG_LOG_G,
                       -1.5)

    fams = []
    reals_TH, reals_G = [], []
    for k in range(n_real):
        fam = "A" if rng.random() < frac_A else "B"
        fams.append(fam)
        if fam == "B":
            w = make_truth(rng.integers(1 << 30))
            THu, Gu = w["TH"].ravel(), w["G"].ravel()
            rx, rz = 55, 16
        else:
            # smooth family: long-range thickness, generous widths
            THu = np.clip(2.3 + 1.0 * (m.L_th @ rng.standard_normal(NCELL)),
                          0.0, 6.0)
            g_lat = 0.8 * (m.L_g @ rng.standard_normal(NCELL))
            Gu = np.exp(MU_LOG_G + SIG_LOG_G * g_lat)
            Gu[THu <= 0] = 0
            rx, rz = A_RANGE_X, A_RANGE_Z
        # condition thickness and latent grade at pierce points
        C_dd = _cov(P, P, rx, rz, m.PLUNGE) + 1e-6 * np.eye(len(P))
        Wk = _cov(COORDS, P, rx, rz, m.PLUNGE) @ np.linalg.inv(C_dd)
        THc = np.clip(THu + Wk @ (th_o - THu[idx]), 0, None)
        g_lat_u = (np.log(np.maximum(Gu, 0.1)) - MU_LOG_G) / SIG_LOG_G
        g_lat_c = g_lat_u + Wk @ (g_lat_o - g_lat_u[idx])
        Gc = np.exp(MU_LOG_G + SIG_LOG_G * g_lat_c)
        Gc[THc <= 0] = 0
        reals_TH.append(THc.reshape(NX, NZ))
        reals_G.append(Gc.reshape(NX, NZ))
    return np.array(reals_TH), np.array(reals_G), fams


# ======================================================================
# Calibration validation
# ======================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Building machinery ...")
    machinery()

    N_WORLDS = 8
    rows = []
    for s in range(N_WORLDS):
        w = make_truth(100 + s)
        pierce = drill_pierce_points(w)
        TH_A, G_A = interpretation_A(pierce)

        # --- planner's reserve (interpretation A, 13% dilution, FS cutoff)
        VT_A, VG_A, THb_A, _ = stope_vein_stats(TH_A, G_A)
        DT_A, DG_A = diluted(VT_A, VG_A, THb_A, "LHOS", planned=True)
        res_mask = DG_A >= CUTOFF_FS
        res_kt = DT_A[res_mask].sum()
        res_g = (DT_A * DG_A)[res_mask].sum() / max(res_kt, 1e-9)
        res_koz = res_kt * 1000 * res_g / GRAMS_PER_OZ / 1000.0

        # --- truth realized mining of the design envelope (per-cell)
        design_env = TH_A > 0
        mined_t, metal, ore_t, over_t, gap_t = stope_realized(
            w["TH"], w["G"], design_env, "LHOS")
        head_all = np.divide(metal, np.maximum(mined_t, 1e-9))

        mined = res_mask & (mined_t > 0.05)
        head_g = metal[mined].sum() / max(mined_t[mined].sum(), 1e-9)
        # V1: FIRST FEED = the 6 best stopes by the planner's own ranking
        order = np.argsort(-(DG_A * res_mask).ravel())[:6]
        ff = np.unravel_index(order, DG_A.shape)
        ff_head = metal[ff].sum() / max(mined_t[ff].sum(), 1e-9)
        ff_plan = (DT_A * DG_A)[ff].sum() / max(DT_A[ff].sum(), 1e-9)

        # V4: operational overbreak dilution on real structures only
        dil_op = over_t[mined].sum() / max(ore_t[mined].sum(), 1e-9)
        # internal waste: barren design cells (gaps/dykes) as share of feed
        gap_share = gap_t[mined].sum() / max(mined_t[mined].sum(), 1e-9)
        # total factor vein->mill
        dil_total = (mined_t[mined].sum() / max(ore_t[mined].sum(), 1e-9)) - 1

        # honest truth ounces: per-stope realized grade >= honest cutoff
        honest_cut = COST_TRUE["LHOS"] / REV_PLAN       # diluted g/t
        true_ok = head_all >= honest_cut
        true_koz = metal[true_ok].sum() * 1000 / GRAMS_PER_OZ / 1000.0
        # undiluted-equivalent honest cutoff for CRUX comparison:
        # overbreak on coherent structures + minor internal waste (gaps are
        # a separate ore-loss channel, not a cutoff adjustment)
        f_mine = (1 + dil_op) * 1.08

        rows.append(dict(
            res_g=res_g, res_kt=res_kt, res_koz=res_koz,
            head_g=head_g, shortfall=1 - head_g / max(res_g, 1e-9),
            ff_short=1 - ff_head / max(ff_plan, 1e-9),
            dil=dil_op, gap=gap_share, dil_total=dil_total,
            true_koz=true_koz, writedown=1 - true_koz / max(res_koz, 1e-9),
            honest_cut_undil=honest_cut * f_mine,
        ))
        r = rows[-1]
        print(f"world {s}: A-res {r['res_kt']:5.0f} kt @ {r['res_g']:5.2f} g/t | "
              f"LOM head {r['head_g']:.2f} ({r['shortfall']:+.0%}) | "
              f"1st-feed {r['ff_short']:+.0%} | overbreak {r['dil']:.0%} "
              f"gap-waste {r['gap']:.0%} | write-down {r['writedown']:+.0%}")

    A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    print("\n===== CALIBRATION vs MADSEN SHEET =====")
    print(f"{'metric':44s} {'target':>14s} {'simulated':>12s}")
    print(f"{'A-model reserve grade (4.75 cutoff, 13% dil)':44s} "
          f"{'~9.0 g/t':>14s} {A['res_g'].mean():>10.2f} g/t")
    print(f"{'Q1 head grade shortfall, first stopes (V2)':44s} "
          f"{'45-65%':>14s} {A['ff_short'].mean():>11.0%}")
    print(f"{'LOM head grade shortfall vs plan':44s} "
          f"{'(context)':>14s} {A['shortfall'].mean():>11.0%}")
    print(f"{'overbreak dilution on real structures (V4)':44s} "
          f"{'18-28%':>14s} {np.nanmean(A['dil']):>11.0%}")
    print(f"{'internal waste (gaps/dykes) share of feed (V3)':44s} "
          f"{'material':>14s} {np.nanmean(A['gap']):>11.0%}")
    print(f"{'honest diluted cutoff at plan price':44s} "
          f"{'~5.8 g/t dil':>14s} {COST_TRUE['LHOS']/REV_PLAN:>10.2f} g/t")
    print(f"{'undiluted honest cutoff (CRUX ~7.45)':44s} "
          f"{'~7.4 g/t':>14s} {np.nanmean(A['honest_cut_undil']):>10.2f} g/t")
    print(f"{'economic write-down A -> honest truth (V7)':44s} "
          f"{'30-60%':>14s} {A['writedown'].mean():>11.0%}")

    # ---------------- figures ----------------
    w = make_truth(103)
    pierce = drill_pierce_points(w)
    TH_A, G_A = interpretation_A(pierce)
    GT_T = w["TH"] * w["G"]
    GT_A = TH_A * G_A
    vmax = np.percentile(GT_T[GT_T > 0], 97) if (GT_T > 0).any() else 1

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.8), constrained_layout=True)
    for ax, F, title in [
        (axes[0], GT_T, "TRUTH: grade x thickness (narrow, discontinuous, dyked)"),
        (axes[1], GT_A, "INTERPRETATION A from the same pierce points\n"
                        "(long ranges, floored widths — the 2019-FS-style model)")]:
        im = ax.imshow(F.T, origin="lower", extent=[0, LX, 0, LZ],
                       cmap="inferno", vmin=0, vmax=vmax, aspect="auto")
        for (ix, iz, th_o, g_o) in pierce:
            ax.plot(xs[ix], zs[iz], "o", ms=3,
                    color="cyan" if th_o > 0.2 else "white",
                    mec="k", mew=0.3)
        for i in range(1, SX):
            ax.axvline(i * CX * DX, color="w", lw=0.3, alpha=0.3)
        for j in range(1, SZL):
            ax.axhline(j * CZ * DZ, color="w", lw=0.3, alpha=0.3)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Strike (m)", fontsize=8)
        ax.set_ylabel("Elevation (m)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=axes, shrink=0.85, label="GT (g/t x m)")
    fig.suptitle("Same drillholes, two deposits: the Madsen epistemic gap",
                 fontsize=12)
    fig.savefig("out/fig13_madsen_truth_vs_interpA.png", dpi=160)

    # grade-tonnage: what the planner sells vs what reality pays
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    VT_A, VG_A, THb_A, _ = stope_vein_stats(TH_A, G_A)
    DT_A, DG_A = diluted(VT_A, VG_A, THb_A, "LHOS", planned=True)
    VT_T, VG_T, THb_T, _ = stope_vein_stats(w["TH"], w["G"])
    DT_T, DG_T = diluted(VT_T, VG_T, THb_T, "LHOS")
    cuts = np.linspace(2, 14, 60)
    koz_A = [(DT_A * DG_A)[DG_A >= c].sum() * 1000 / GRAMS_PER_OZ / 1000
             for c in cuts]
    koz_T = [(DT_T * DG_T)[DG_T >= c].sum() * 1000 / GRAMS_PER_OZ / 1000
             for c in cuts]
    ax.plot(cuts, koz_A, lw=2, color="#8d99ae",
            label="Interpretation A (planned, 13% dilution)")
    ax.plot(cuts, koz_T, lw=2, color="#d1495b",
            label="Truth (geometric dilution, LHOS)")
    ax.axvline(CUTOFF_FS, color="green", ls="--", lw=1.2,
               label=f"FS cutoff {CUTOFF_FS} g/t (believed costs)")
    ax.axvline(COST_TRUE['LHOS'] / REV_PLAN, color="purple", ls=":",
               lw=1.4, label=f"Honest cutoff {COST_TRUE['LHOS']/REV_PLAN:.1f} g/t "
                             f"(true costs, diluted)")
    ax.set_xlabel("Diluted cutoff grade (g/t)"); ax.set_ylabel("Contained koz above cutoff")
    ax.set_title("The two grade-tonnage curves the market never saw side by side",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)
    fig.savefig("out/fig14_madsen_grade_tonnage.png", dpi=160)

    # ensemble sanity: both families, all honoring pierce data
    print("\nBuilding 12-member mixed-interpretation ensemble (sanity) ...")
    rTH, rG, fams = make_ensemble(pierce, n_real=12, seed=5)
    idx = np.array([ix * NZ + iz for (ix, iz, _, _) in pierce])
    th_o = np.array([p[2] for p in pierce])
    mism = max(abs(rTH[k].ravel()[idx] - th_o).max() for k in range(12))
    print(f"families: {fams}")
    print(f"max thickness mismatch at pierce points: {mism:.2e} m")
    np.savez_compressed("out/madsen_world_sample.npz",
                        TH=w["TH"], G=w["G"], TH_A=TH_A, G_A=G_A,
                        pierce=np.array(pierce))
    print("Wrote fig13, fig14, madsen_world_sample.npz")
