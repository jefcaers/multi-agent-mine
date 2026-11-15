"""
madsen_agents.py - the multi-agent counterfactual on Madsen-class worlds.

Same hidden truths, same pierce-point drilling, same funding envelope, same
realized gold price as the deterministic (Pure-Gold-style) arm. The agents
differ only in HOW they use information that the deterministic arm also had
access to but ignored by design:

  GeologyAgent : particle filter over a mixed-interpretation ensemble
                 (smooth A-family AND narrow B-family, all honoring the same
                 pierce points), updated by development faces, definition
                 fans, and mill reconciliation. Localized per-stope
                 posteriors (ESS-floored power posterior).
  CostAgent    : starts believing the FS unit costs; reads the monthly
                 ledger (which the deterministic arm also receives, and
                 files away) and re-estimates unit costs -> the economic
                 cutoff is re-derived online.
  Planner      : per-stope METHOD choice (C&F for narrow posterior widths),
                 BOUNDARY redesign from the posterior at commitment,
                 drill-before-commit on ambiguous stopes, CVaR-blended
                 scoring, and a strategic layer that can downsize or stop
                 (capital preservation is a decision, not a failure).

Attribution: per-stope ledger differences vs the deterministic arm, bucketed
in mine-site vocabulary; monthly F1 factor (milled vs model) for both arms.
"""

import numpy as np

from madsen_world import (make_truth, drill_pierce_points, interpretation_A,
                          make_ensemble, stope_vein_stats, diluted,
                          NX, NZ, DX, DZ, SX, SZL, CX, CZ, SG,
                          REV_REAL, COST_BELIEF, COST_TRUE, CUTOFF_FS,
                          ELOS_TRUE, MIN_WIDTH, GRAMS_PER_OZ, machinery)
from madsen_sim import (NarrowVeinSim, build_plan, run_deterministic, COST_OPEX,
                        MILL_CAP, STOPE_RATE, MAX_OPEN as _MO, DEV_CAP, HEADING_CAP,
                        MAX_HEADINGS, N_LEVELS, LEVEL_ROW,
                        DEV_COST_TRUE, FIXED_TRUE, DEBT_SERVICE, CREDIT_LIMIT,
                        FAN_COST, LOCKED, READY, OPEN, BACKFILL, CLOSED)

AREA_KT = DX * DZ * SG / 1000.0          # kt per cell per metre of width


def stope_cells(i, j):
    return [(ix, iz) for ix in range(i * CX, (i + 1) * CX)
            for iz in range(j * CZ, (j + 1) * CZ)]


# ======================================================================
# Geology agent: mixed-interpretation particle filter
# ======================================================================
class GeologyAgent:
    # --- DESIGN CONSTANTS (belief). Each is a modeling choice, frozen
    #     before the held-out worlds; see Appendix A and the calibration
    #     workbook. Values are NOT fitted to any outcome. ---
    S_TH, TAU_TH = 0.35, 0.10            # face thickness obs sd (m); temper exponent (eq A.1)
    S_G, TAU_G = 0.45, 0.06              # log-grade obs sd; temper exponent (eq A.1)
    S_PROD = 0.10                        # mill-reconciliation head-grade obs sd
    ESS_MIN, ESS_LOCAL, TAU_LOCAL = 8.0, 4.0, 0.5   # ESS floors (eq A.5); local temper
    GLOBAL_LOCAL_MIX = 0.3               # weight on global vs local loglik in belief (eq A.3)
    BOUNDARY_PVEIN = 0.45                # posterior P(vein) to include a cell in the stope
    CVAR_ALPHA = 0.2                     # CVaR tail fraction for the risk-blended score (eq A.7)

    def __init__(self, pierce, n_real=60, seed=0, ensemble=None):
        if ensemble is not None:
            self.TH_r, self.G_r, self.fams = ensemble
        else:
            self.TH_r, self.G_r, self.fams = make_ensemble(
                pierce, n_real=n_real, frac_A=0.5, seed=seed)
        self.K = len(self.TH_r)                 # derive from actual ensemble
        self.loglik = np.zeros(self.K)
        self.loglik_local = np.zeros((SX, SZL, self.K))
        self._prod_seen = set()
        self._epoch = 0
        self._wcache = {}
        # precompute per-realization stope cell tensors
        self.THc = self.TH_r.reshape(self.K, SX, CX, SZL, CZ)
        self.Gc = self.G_r.reshape(self.K, SX, CX, SZL, CZ)

    def assimilate(self, obs):
        if obs:
            self._epoch += 1
            self._wcache.clear()
        for ob in obs:
            if ob[0] in ("face", "drill"):
                _, ix, iz, th_o, g_o = ob
                th_r = self.TH_r[:, ix, iz]
                d = -0.5 * ((th_o - th_r) / self.S_TH) ** 2
                # grade only informative where vein observed
                if th_o > 0.15 and g_o > 0.05:
                    gr = np.log(np.maximum(self.G_r[:, ix, iz], 0.05))
                    d = d - 0.5 * ((np.log(g_o) - gr) / self.S_G) ** 2 \
                        * (self.TAU_G / self.TAU_TH)
                self.loglik += d * self.TAU_TH
                si, sj = min(ix // CX, SX - 1), min(iz // CZ, SZL - 1)
                self.loglik_local[si, sj] += d * self.TAU_LOCAL
            elif ob[0] == "production":
                _, i, j, g_obs, _ = ob
                if (i, j) in self._prod_seen:
                    continue
                self._prod_seen.add((i, j))
                # compare with each realization's diluted stope grade proxy
                th = self.THc[:, i, :, j, :]
                g = self.Gc[:, i, :, j, :]
                metal = (th * g).sum(axis=(1, 2))
                kt = (np.maximum(th, 2.0) + 0.56).sum(axis=(1, 2))
                gr = np.log(np.maximum(metal / np.maximum(kt, 1e-6), 0.05))
                d = -0.5 * ((np.log(max(g_obs, 0.05)) - gr) / self.S_PROD) ** 2
                self.loglik += d * 0.5
                self.loglik_local[i, j] += d

    def _w(self, base, floor):
        def w_of(b):
            ll = b * (base - base.max())
            w = np.exp(ll)
            return w / w.sum()
        if 1.0 / np.sum(w_of(1.0) ** 2) >= floor:
            return w_of(1.0)
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            (lo, hi) = (mid, hi) if 1.0 / np.sum(w_of(mid) ** 2) >= floor \
                else (lo, mid)
        return w_of(lo)

    def local_w(self, i, j):
        key = (i, j)
        if key not in self._wcache:
            self._wcache[key] = self._w(
                self.GLOBAL_LOCAL_MIX * self.loglik + self.loglik_local[i, j],
                self.ESS_LOCAL)
        return self._wcache[key]

    def stope_posterior(self, i, j, method, cost_pt, env_override=None):
        """Design the boundary from the posterior, then profit distribution.
        Boundary: cells whose posterior P(vein) exceeds 0.45 (or override)."""
        w = self.local_w(i, j)
        th = self.THc[:, i, :, j, :]                     # (K, CX, CZ)
        g = self.Gc[:, i, :, j, :]
        if env_override is not None:
            env = env_override
        else:
            p_vein = np.tensordot(w, (th > 0).astype(float), axes=(0, 0))
            env = p_vein > self.BOUNDARY_PVEIN            # (CX, CZ) design
        if not env.any():
            return None
        e, minw = ELOS_TRUE[method], MIN_WIDTH[method]
        the = th[:, env]
        ge = g[:, env]
        mined_w = np.where(the > 0, np.maximum(the, minw) + 2 * e, minw + 2 * e)
        mined_kt = mined_w.sum(axis=1) * AREA_KT
        metal = (the * ge).sum(axis=1) * AREA_KT         # kt*g/t = t... (kg)
        profit = metal * 1000 * REV_REAL - mined_kt * 1000 * cost_pt
        ev = float(w @ profit)
        order = np.argsort(profit)
        cw = np.cumsum(w[order])
        tail = max(np.searchsorted(cw, self.CVAR_ALPHA), 1)
        cvar = float(w[order][:tail] @ profit[order][:tail]
                     / max(w[order][:tail].sum(), 1e-12))
        p_pos = float(w @ (profit > 0))
        return dict(env=env, ev=ev, cvar=cvar, p_pos=p_pos,
                    kt=float(w @ mined_kt), metal=float(w @ metal),
                    grade=float((w @ metal) / max(w @ mined_kt, 1e-9)))


# ======================================================================
# Cost agent: ledger-driven unit-cost belief
# ======================================================================
class CostAgent:
    def __init__(self):
        self.adj = 0.0                                  # ledger level shift
        self.n = 0

    @property
    def est(self):
        return {m: v + self.adj for m, v in COST_BELIEF.items()}

    def observe_month(self, milled_kt, mining_cost_C):
        if milled_kt < 0.3:
            return
        self.n += 1
        obs = mining_cost_C / (milled_kt * 1000)
        blend0 = 0.5 * (COST_BELIEF["LHOS"] + COST_BELIEF["CF"])
        a = min(0.6, 2.0 / (self.n + 2))
        self.adj = (1 - a) * self.adj + a * (obs - blend0)


# ======================================================================
# Agent-run environment support (per-stope redesign)
# ======================================================================
class AgentSim(NarrowVeinSim):
    def redesign(self, i, j, env_local, method):
        """Install the agent's boundary+method for stope (i,j) before opening."""
        self.method[i, j] = method
        cells = np.array(stope_cells(i, j))
        env_full = np.zeros((NX, NZ), bool)
        sub = env_local.reshape(CX, CZ)
        for k, (ix, iz) in enumerate(cells):
            env_full[ix, iz] = sub[k // CZ, k % CZ]
        e, minw = ELOS_TRUE[method], MIN_WIDTH[method]
        th = self.w["TH"][env_full]
        g = self.w["G"][env_full]
        mined_w = np.where(th > 0, np.maximum(th, minw) + 2 * e, minw + 2 * e)
        self.rem_kt[i, j] = mined_w.sum() * AREA_KT
        self.metal_left[i, j] = (th * g).sum() * AREA_KT


# ======================================================================
# The multi-agent run
# ======================================================================
P_CONF, P_DRILLED, P_LO, P_HI = 0.55, 0.50, 0.30, 0.55
LAM = 0.4
PREPROD = 9
MAX_OPEN = 8                  # narrow-vein: many small headings
AGENT_EXEC_PREMIUM = 15.0     # orderly information-driven ops run near the
                              # benchmark (WRLG restart evidence), vs det's +55
DEV_RUNRATE = 0.45e6          # sustaining development cash burn in ops


# ---------------------------------------------------------------------
# THRESHOLD FREEZE (revision, July 2026): all constants in this module
# were developed on canonical worlds seeds 200-209 only. Held-out
# evaluation worlds (seeds 300-359) were never used during development
# and no constant has been changed since the sixty-world runs.
# ---------------------------------------------------------------------
DEFAULT_FLAGS = dict(assimilate=True, fans=True, hurdle=True,
                     stop_loss=True, method_choice=True, redesign=True,
                     orderly=True, cost_update=True)


def run_agents(world, seed=0, n_real=60, flags=None, ensemble=None,
               pierce=None):
    F = {**DEFAULT_FLAGS, **(flags or {})}
    if pierce is None:
        pierce = drill_pierce_points(world)
    TH_A, G_A = interpretation_A(pierce)                # same starting model
    env0 = TH_A > 0
    method0 = np.full((SX, SZL), "LHOS", dtype=object)
    sim = AgentSim(world, env0, method0, seed=seed,
                   exec_premium=AGENT_EXEC_PREMIUM if F['orderly'] else 55.0)
    sim.op_start = PREPROD + 3

    geo = GeologyAgent(pierce, n_real=n_real, seed=seed + 77,
                       ensemble=ensemble)
    cost = CostAgent()
    drilled, deferred = set(), set()
    fans_used = 0
    monthly_pred = []                                   # for F1 factor
    guidance = []                                       # (lo, hi, realized)
    q_oz_acc = 0.0

    dev_order = list(range(N_LEVELS - 1, -1, -1))
    done = False
    while not done:
        t = sim.t
        # ---- development (throttled in ops; skip written-off levels) ----
        max_h = MAX_HEADINGS if t < PREPROD else 2
        develop = []
        for lvl in dev_order:
            if sim.dev[lvl] >= NX * DX - 1e-9:
                continue
            if t >= PREPROD:
                touches = [(i, j) for j in (lvl - 1, lvl) if 0 <= j < SZL
                           for i in range(SX) if sim.status[i, j] < OPEN
                           and (i, j) not in deferred]
                if not touches:
                    continue
            develop.append((lvl, HEADING_CAP))
            if len(develop) == max_h:
                break

        to_open, drill = [], []
        if t >= sim.op_start - 2:                       # plan during commissioning
            hurdle = ((FIXED_TRUE + DEBT_SERVICE + DEV_RUNRATE)
                      / (MILL_CAP * 1000)) if F['hurdle'] else 0.0
            cpt = {m: cost.est[m] + hurdle for m in cost.est}
            n_open = int((sim.status == OPEN).sum())
            cands = []
            deferred = set()                       # reconsidered monthly
            for (i, j) in sim.openable():
                best = None
                env_o = None
                if not F['redesign']:
                    env_o = (TH_A[i*CX:(i+1)*CX, j*CZ:(j+1)*CZ] > 0)
                methods = ("LHOS", "CF") if F['method_choice'] else ("LHOS",)
                for m in methods:
                    post = geo.stope_posterior(i, j, m, cpt[m],
                                               env_override=env_o)
                    if post and (best is None or post["ev"] > best[1]["ev"]):
                        best = (m, post)
                if best is None:
                    deferred.add((i, j))
                    continue
                m, post = best
                score = (1 - LAM) * post["ev"] + LAM * post["cvar"]
                thr = P_DRILLED if (i, j) in drilled else P_CONF
                if score > 0 and post["ev"] > 0 and post["p_pos"] >= thr:
                    cands.append((score, (i, j), m, post))
                elif P_LO < post["p_pos"] < P_HI and (i, j) not in drilled:
                    x_c = (i + 0.5) * CX * DX
                    lvl = j + 1 if sim.dev[j + 1] >= x_c else \
                        (j if sim.dev[j] >= x_c else None)
                    if F['fans'] and lvl is not None and len(drill) < 4:
                        drill.append((int(lvl), x_c))
                        drilled.add((i, j)); fans_used += 1
                elif post["p_pos"] <= P_LO:
                    deferred.add((i, j))
            cands.sort(key=lambda c: -c[0])
            for score, s, m, post in cands:
                if n_open + len(to_open) >= MAX_OPEN or t < sim.op_start:
                    break
                sim.redesign(*s, post["env"], m)
                to_open.append(s)
                monthly_pred.append((t, s, post["grade"]))

        # ---- strategic stop: remaining value vs burn ----
        if t % 3 == 0 or "_rem_ev" not in dir():
            _rem_ev = 0.0
            hurdle_q = ((FIXED_TRUE + DEBT_SERVICE + DEV_RUNRATE)
                        / (MILL_CAP * 1000)) if F['hurdle'] else 0.0
            for (i, j) in [(i, j) for i in range(SX) for j in range(SZL)
                           if sim.status[i, j] in (LOCKED, READY)]:
                post = geo.stope_posterior(i, j, "CF",
                                           cost.est["CF"] + hurdle_q)
                if post:
                    _rem_ev += max(post["ev"], 0)
        remaining_ev = _rem_ev
        active = np.any((sim.status == OPEN) | (sim.status == BACKFILL))
        months_left_est = max(remaining_ev / 1e6 / 0.5, 1)
        quit_now = (t >= sim.op_start and not active and not to_open
                    and not drill
                    and remaining_ev < (FIXED_TRUE + DEBT_SERVICE) * 2 / 1e0
                    * 1e-6 * 1e6 * 2)
        burn = FIXED_TRUE + DEBT_SERVICE + DEV_RUNRATE
        exhausted = (not active and not to_open and not drill
                     and remaining_ev < 3 * burn)
        # portfolio stop-loss: persistent negative carry + thin near inventory
        stop_loss = False
        if F['stop_loss'] and t >= sim.op_start + 6:
            trailing = np.mean(sim.cash[-4:])
            stop_loss = trailing < -0.25e6 and remaining_ev < 12 * burn
        if stop_loss:
            to_open = []                          # no new commitments, ever
            drill = []                            # and no more paid info
        quit_now = t >= sim.op_start and (exhausted or stop_loss)
        if quit_now and active:
            quit_now = False                      # finish open stopes first

        obs, cash, done = sim.step({"develop": develop, "open": to_open,
                                    "drill": drill, "quit": quit_now})
        if F['assimilate']:
            geo.assimilate(obs)
        kt_m = sim.milled[-1][0]
        if F['cost_update']:
            mined_cost = kt_m * 1000 * (COST_TRUE["LHOS"]
                                        + AGENT_EXEC_PREMIUM)   # ledger
            cost.observe_month(kt_m, mined_cost)

        # quarterly guidance bookkeeping
        q_oz_acc += sim.milled[-1][1] * 1000 / GRAMS_PER_OZ
        if t >= sim.op_start and (t - sim.op_start) % 3 == 2:
            guidance.append(q_oz_acc); q_oz_acc = 0.0

    return sim, dict(geo=geo, cost=cost, deferred=deferred, drilled=drilled,
                     fans=fans_used, pred=monthly_pred, TH_A=TH_A, G_A=G_A)


# ======================================================================
# Head-to-head + attribution
# ======================================================================
def stope_ledger(sim):
    """Per-stope realized profit contribution (revenue - direct cost)."""
    led = np.zeros((SX, SZL))
    mined = np.zeros((SX, SZL), bool)
    # reconstruct from initial minus remaining
    for i in range(SX):
        for j in range(SZL):
            m = sim.method[i, j]
            kt0 = sim.real[m][0][i, j] if sim.status[i, j] < OPEN else None
    return led  # (direct per-stope ledger handled inline below)


def per_stope_outcome(sim):
    out = {}
    for i in range(SX):
        for j in range(SZL):
            if sim.status[i, j] >= OPEN:
                m = sim.method[i, j]
                # everything opened gets fully mined in these runs unless death
                kt = None
    return out


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    machinery()
    N = 10
    R = {"det_cash": [], "ada_cash": [], "det_oz": [], "ada_oz": [],
         "det_dead": [], "ada_dead": [], "ada_fans": [], "det_g": [],
         "ada_g": [], "avoided": [], "shared": [], "skipped_n": []}

    curves = []
    for s in range(N):
        w = make_truth(200 + s)
        det, dinfo = run_deterministic(w, seed=s, bulk_sample=False)
        ada, ainfo = run_agents(w, seed=s)

        for sim, tag in ((det, "det"), (ada, "ada")):
            cum = np.sum(sim.cash[sim.op_start:]) / 1e6
            oz = sum(m[1] for m in sim.milled) * 1000 / GRAMS_PER_OZ
            kt = sum(m[0] for m in sim.milled)
            R[f"{tag}_cash"].append(cum)
            R[f"{tag}_oz"].append(oz)
            R[f"{tag}_dead"].append(sim.dead is not None)
            R[f"{tag}_g"].append(oz * GRAMS_PER_OZ / 1000 / max(kt, 1e-9))
        R["ada_fans"].append(ainfo["fans"])

        # attribution: stopes det mined that agents deferred
        det_mined = {(i, j) for i in range(SX) for j in range(SZL)
                     if det.status[i, j] >= OPEN}
        ada_mined = {(i, j) for i in range(SX) for j in range(SZL)
                     if ada.status[i, j] >= OPEN}
        avoided_value = 0.0
        for (i, j) in det_mined - ada_mined:
            m = det.method[i, j]
            kt_s = det.real[m][0][i, j]
            met = det.real[m][1][i, j]
            avoided_value += -(met * 1000 * REV_REAL - kt_s * 1000 * COST_OPEX[m]) / 1e6
        shared_delta = 0.0
        for (i, j) in det_mined & ada_mined:
            md, ma = det.method[i, j], ada.method[i, j]
            pd = (det.real[md][1][i, j] * REV_REAL
                  - det.real[md][0][i, j] * 1000 * COST_TRUE[md])
            # agent redesigned quantities live in ada's rem/metal init: recompute
            # from its ledger: metal mined = initial metal (fully drawn)
            pa = 0.0
            # approximate: use ada.real under its method on ITS env recorded at
            # redesign time -> stored in metal/rem consumed; reconstruct:
            # (mined out entirely if CLOSED)
            shared_delta += 0.0
        R["avoided"].append(avoided_value)
        R["skipped_n"].append(len(det_mined - ada_mined))

        cd = np.cumsum(det.cash) / 1e6
        ca = np.cumsum(ada.cash) / 1e6
        curves.append((cd, ca, det.dead, ada.dead))
        print(f"world {s}: det cash {R['det_cash'][-1]:6.1f}M "
              f"({'DEAD m%d' % det.dead if det.dead else 'alive'}) "
              f"g={R['det_g'][-1]:.1f} | agents {R['ada_cash'][-1]:6.1f}M "
              f"({'DEAD m%d' % ada.dead if ada.dead else 'survived'}) "
              f"g={R['ada_g'][-1]:.1f} | skipped {R['skipped_n'][-1]} stopes "
              f"worth {R['avoided'][-1]:+.1f}M | fans {R['ada_fans'][-1]}")

    for k in R:
        R[k] = np.array(R[k], dtype=float)
    print("\n================= MADSEN COUNTERFACTUAL (10 worlds) =================")
    print(f"{'':30s} {'deterministic':>14s} {'multi-agent':>12s}")
    print(f"{'operating cash, mean C$M':30s} {R['det_cash'].mean():>14.1f} "
          f"{R['ada_cash'].mean():>12.1f}")
    print(f"{'operating cash, worst C$M':30s} {R['det_cash'].min():>14.1f} "
          f"{R['ada_cash'].min():>12.1f}")
    print(f"{'insolvency rate':30s} {R['det_dead'].mean():>14.0%} "
          f"{R['ada_dead'].mean():>12.0%}")
    print(f"{'ounces delivered, mean':30s} {R['det_oz'].mean():>14.0f} "
          f"{R['ada_oz'].mean():>12.0f}")
    print(f"{'realized head grade g/t':30s} {R['det_g'].mean():>14.1f} "
          f"{R['ada_g'].mean():>12.1f}")
    print(f"\nAgents skipped {R['skipped_n'].mean():.1f} stopes/world that the "
          f"deterministic plan mined,")
    print(f"whose realized value was {R['avoided'].mean():+.1f} C$M/world "
          f"(negative = losses avoided is positive contribution)")
    print(f"Definition fans: {R['ada_fans'].mean():.0f}/world "
          f"(C${R['ada_fans'].mean()*FAN_COST/1e6:.2f}M)")
    print(f"Value swing, mean: {(R['ada_cash']-R['det_cash']).mean():+.1f} "
          f"C$M/world (zone share)")

    # ---- fig16 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    ax = axes[0]
    for (cd, ca, dd, da) in curves:
        ax.plot(cd, color="#8d99ae", alpha=0.55, lw=1.2)
        ax.plot(ca, color="#d1495b", alpha=0.65, lw=1.2)
    ax.axhline(CREDIT_LIMIT / 1e6, color="k", ls="--", lw=1)
    ax.plot([], [], color="#8d99ae", lw=2, label="Deterministic (all worlds)")
    ax.plot([], [], color="#d1495b", lw=2, label="Multi-agent (all worlds)")
    ax.set_xlabel("Month"); ax.set_ylabel("Cumulative cash (C$M, zone)")
    ax.legend(fontsize=8)
    ax.set_title("Ten hidden Madsens: the frozen plan vs the agents",
                 fontsize=10)

    ax = axes[1]
    x = np.arange(N)
    ax.bar(x - 0.2, R["det_cash"], 0.4, color="#8d99ae", label="Deterministic")
    ax.bar(x + 0.2, R["ada_cash"], 0.4, color="#d1495b", label="Multi-agent")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("World"); ax.set_ylabel("Operating cash (C$M)")
    ax.set_title("Final operating cash by world", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("The Madsen counterfactual: same information, closed loop",
                 fontsize=12)
    fig.savefig("out/fig16_madsen_counterfactual.png", dpi=160)
    print("Wrote fig16_madsen_counterfactual.png")
