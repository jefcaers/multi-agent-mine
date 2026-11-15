"""
madsen_sim.py - narrow-vein sublevel mining simulator on the Madsen-class
world, plus the DETERMINISTIC (Pure-Gold-style) arm and its validation
against the failure-trajectory targets V2/V5/V6/V8.

Scale note: the panel is ONE ZONE of a Madsen-class mine (~1/6 of the mill
feed). All capacities and financing are zone-share scaled and documented.

Period = 1 month.
"""

import numpy as np

from madsen_world import (make_truth, drill_pierce_points, interpretation_A,
                          stope_vein_stats, stope_realized, diluted,
                          NX, NZ, DX, DZ, SX, SZL, CX, CZ, SG,
                          REV_PLAN, REV_REAL, COST_BELIEF, COST_TRUE,
                          CUTOFF_FS, DIL_PLANNED, GRAMS_PER_OZ, machinery)

# ---------------- capacities (zone share of an 800 tpd mine) ----------------
MILL_CAP = 6.0            # kt/month milled from this zone
STOPE_RATE = 1.4          # kt/month drawn from one active stope
MAX_OPEN = 5              # simultaneous active stopes (narrow-vein normal)
DEV_CAP = 150.0           # m/month lateral development
HEADING_CAP = 60.0        # m/month per heading
MAX_HEADINGS = 3
CURE = 1                  # backfill cure (months)
N_LEVELS = SZL + 1        # 9 sublevel drives
LEVEL_ROW = [min(j * CZ, NZ - 1) for j in range(N_LEVELS)]

# ---------------- costs (C$; belief vs truth) ----------------
DEV_COST_BELIEF = 4200.0      # C$/m
DEV_COST_TRUE = 5200.0
FIXED_BELIEF = 0.22e6         # C$/month zone-share site fixed
FIXED_TRUE = 0.33e6
DEBT_SERVICE = 0.30e6         # C$/month zone-share of capex debt
CREDIT_LIMIT = -14.0e6       # cum op cash floor: credit + emergency equity raises (zone share)
FAN_COST = 18000.0            # definition fan
OBS_TH_SD, OBS_G_SD = 0.12, 0.20   # face-mapping observation noise (rel.)
EXEC_PREMIUM = 55.0           # C$/t: PG actual execution above the Macassa
                              # benchmark (rushed contractors, rehandling);
                              # world-level honest-cutoff math stays on 305
COST_OPEX = {m: v + EXEC_PREMIUM for m, v in COST_TRUE.items()}

LOCKED, READY, OPEN, BACKFILL, CLOSED = 0, 1, 2, 3, 4


class NarrowVeinSim:
    """Environment. Mines the DESIGN envelope of whatever plan is supplied;
    reality is the hidden truth world."""

    def __init__(self, world, design_env, method_map, seed=0,
                 exec_premium=EXEC_PREMIUM):
        self.exec_premium = exec_premium
        self.w = world
        self.env = design_env                          # (NX,NZ) bool
        self.rng = np.random.default_rng(seed)
        self.method = method_map                       # (SX,SZL) 'LHOS'/'CF'
        # precompute per-stope realized quantities under each method
        self.real = {}
        for m in ("LHOS", "CF"):
            self.real[m] = stope_realized(world["TH"], world["G"], design_env, m)
        self.reset()

    def reset(self):
        self.t = 0
        self.dev = np.zeros(N_LEVELS)
        self.status = np.full((SX, SZL), LOCKED, int)
        self.rem_kt = np.zeros((SX, SZL))
        self.metal_left = np.zeros((SX, SZL))
        for i in range(SX):
            for j in range(SZL):
                m = self.method[i, j]
                self.rem_kt[i, j] = self.real[m][0][i, j]
                self.metal_left[i, j] = self.real[m][1][i, j]
        self.cure_left = np.zeros((SX, SZL), int)
        self.cash = []
        self.op_start = 0           # months before this are capitalized
        self.milled = []            # (kt, metal_kg) per period
        self.log = []
        self.exposed = set()
        self._refresh()
        self.dead = None            # insolvency month

    # ---- readiness / adjacency (same semantics as base sim) ----
    def _reach(self, i, j):
        need = (i + 1) * CX * DX
        return self.dev[j] >= need - 1e-9 and self.dev[j + 1] >= need - 1e-9

    def _neigh_ok(self, i, j):
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < SX and 0 <= b < SZL and self.status[a, b] in (OPEN, BACKFILL):
                return False
        return True

    def _refresh(self):
        for i in range(SX):
            for j in range(SZL):
                if self.status[i, j] == LOCKED and self._reach(i, j):
                    self.status[i, j] = READY

    def can_open(self, i, j):
        return self.status[i, j] == READY and self._neigh_ok(i, j)

    def openable(self):
        return [(i, j) for i in range(SX) for j in range(SZL)
                if self.can_open(i, j)]

    # ---- observations ----
    def _face_obs(self, cells, kind, obs):
        for (ix, iz) in cells:
            if (ix, iz) in self.exposed:
                continue
            self.exposed.add((ix, iz))
            th = self.w["TH"][ix, iz]
            g = self.w["G"][ix, iz]
            th_o = max(th * (1 + self.rng.normal(0, OBS_TH_SD)), 0.0)
            g_o = max(g * (1 + self.rng.normal(0, OBS_G_SD)), 0.0) if th > 0 else 0.0
            obs.append((kind, ix, iz, th_o, g_o))

    # ---- one period ----
    def step(self, action):
        action = {**{"develop": [], "open": [], "drill": [], "quit": False},
                  **(action or {})}
        obs, cash = [], 0.0

        # development
        total, heads = 0.0, 0
        for (lvl, m) in action["develop"]:
            if heads >= MAX_HEADINGS or total >= DEV_CAP:
                break
            m = float(np.clip(m, 0, min(HEADING_CAP, DEV_CAP - total,
                                        NX * DX - self.dev[lvl])))
            if m <= 0:
                continue
            row = LEVEL_ROW[lvl]
            x0, x1 = self.dev[lvl], self.dev[lvl] + m
            cells = [(ix, row) for ix in range(int(x0 // DX),
                                               min(int(np.ceil(x1 / DX)), NX))]
            self._face_obs(cells, "face", obs)
            self.dev[lvl] = x1
            cash -= m * DEV_COST_TRUE
            total += m; heads += 1

        # definition fans
        for (lvl, x_m) in action["drill"]:
            if self.dev[lvl] < x_m:
                continue
            ix = int(np.clip(x_m / DX, 0, NX - 1))
            lo = LEVEL_ROW[max(lvl - 1, 0)]
            hi = LEVEL_ROW[min(lvl + 1, N_LEVELS - 1)]
            self._face_obs([(ix, iz) for iz in range(lo, hi + 1)], "drill", obs)
            cash -= FAN_COST

        self._refresh()
        for (i, j) in action["open"]:
            if self.can_open(i, j) and self.rem_kt[i, j] > 0.02:
                self.status[i, j] = OPEN
                self.log.append((self.t, "open", i, j))

        # extraction under mill cap
        budget = MILL_CAP
        kt_p, metal_p = 0.0, 0.0
        for i in range(SX):
            for j in range(SZL):
                if self.status[i, j] != OPEN or budget <= 0:
                    continue
                pull = min(STOPE_RATE, self.rem_kt[i, j], budget)
                if pull <= 0:
                    continue
                frac = pull / self.rem_kt[i, j]
                metal = self.metal_left[i, j] * frac
                self.rem_kt[i, j] -= pull
                self.metal_left[i, j] -= metal
                budget -= pull
                kt_p += pull; metal_p += metal
                g_head = metal / max(pull, 1e-9)
                g_obs = max(g_head * (1 + self.rng.normal(0, 0.04)), 0.0)
                obs.append(("production", i, j, g_obs, pull))
                mth = self.method[i, j]
                cash += metal * 1000 * REV_REAL          # kg -> g x C$/g
                cash -= pull * 1000 * (COST_TRUE[mth] + self.exec_premium)
                if self.rem_kt[i, j] <= 0.02:
                    self.status[i, j] = BACKFILL
                    self.cure_left[i, j] = CURE
                    self.log.append((self.t, "backfill", i, j))

        curing = self.status == BACKFILL
        self.cure_left[curing] -= 1
        done_c = curing & (self.cure_left <= 0)
        self.status[done_c] = CLOSED
        self._refresh()

        cash -= FIXED_TRUE + DEBT_SERVICE
        self.cash.append(cash)
        self.milled.append((kt_p, metal_p))
        self.t += 1

        cum_op = np.sum(self.cash[self.op_start:])
        if self.dead is None and self.t > self.op_start and cum_op < CREDIT_LIMIT:
            self.dead = self.t
        active = np.any((self.status == OPEN) | (self.status == BACKFILL))
        done = (self.dead is not None) or (action["quit"] and not active) \
               or self.t >= 96
        return obs, cash, done

    def head_grade_series(self):
        kt = np.array([m[0] for m in self.milled])
        kg = np.array([m[1] for m in self.milled])
        with np.errstate(invalid="ignore", divide="ignore"):
            return kt, np.where(kt > 0, kg / kt, np.nan)


# ======================================================================
# The deterministic (Pure-Gold-style) arm
# ======================================================================
def build_plan(TH_A, G_A):
    """FS-style plan: reserve stopes from interpretation A, LHOS everywhere,
    top-down mining sequence (ramp access from surface: upper levels first,
    hangingwall of the plan = McVeigh analogue)."""
    VT, VG, THb, _ = stope_vein_stats(TH_A, G_A)
    DT, DG = diluted(VT, VG, THb, "LHOS", planned=True)
    reserve = DG >= CUTOFF_FS
    method = np.full((SX, SZL), "LHOS", dtype=object)
    seq = [(i, j) for j in range(SZL - 1, -1, -1) for i in range(SX)
           if reserve[i, j]]
    return reserve, method, seq, DT, DG


def planned_trajectory(seq, DT, DG):
    """The plan the deterministic arm PROMISES: its own scheduler run against
    interpretation A with believed costs and planned dilution."""
    kt_left = {s: DT[s] for s in seq}
    oz, cashf = [], []
    dev_m_left = N_LEVELS * NX * DX
    q = list(seq)
    open_now = []
    for t in range(96):
        dev = min(DEV_CAP, dev_m_left); dev_m_left -= dev
        # naive readiness: stopes become available as development proceeds
        frontier = (N_LEVELS * NX * DX - dev_m_left) / (N_LEVELS * NX * DX)
        avail = int(frontier * len(seq) + 1e-9)
        while len(open_now) < MAX_OPEN and len(seq) - len(q) < avail and q:
            open_now.append(q.pop(0))
        budget, kt_t, kg_t = MILL_CAP, 0.0, 0.0
        for s in list(open_now):
            pull = min(STOPE_RATE, kt_left[s], budget)
            kt_left[s] -= pull; budget -= pull
            kt_t += pull; kg_t += pull * DG[s]
            if kt_left[s] <= 0.02:
                open_now.remove(s)
        oz.append(kg_t * 1000 / GRAMS_PER_OZ)
        cashf.append(kg_t * REV_PLAN * 1000
                     - kt_t * 1000 * COST_BELIEF["LHOS"]
                     - dev * DEV_COST_BELIEF - FIXED_BELIEF - DEBT_SERVICE)
        if not q and not open_now:
            break
    return np.array(oz), np.array(cashf)


def run_deterministic(world, seed=0, bulk_sample=True, pierce=None):
    if pierce is None:
        pierce = drill_pierce_points(world)
    TH_A, G_A = interpretation_A(pierce)
    env = TH_A > 0
    reserve, method, seq, DT_A, DG_A = build_plan(TH_A, G_A)
    sim = NarrowVeinSim(world, env, method, seed=seed)

    # ---- V8: pre-production bulk sample -------------------------------
    bulk = None
    if bulk_sample:
        # 2 test stopes: high-A-grade AND best locally-supported (nearest
        # mineralized pierce points) -- where you have access and confidence
        P = [(ix, iz) for (ix, iz, th_o, _) in pierce if th_o > 0.2]
        support = np.zeros((SX, SZL))
        for (ix, iz) in P:
            support[min(ix // CX, SX - 1), min(iz // CZ, SZL - 1)] += 1
        med = np.median(DG_A[reserve])
        cand = [(support[i, j] - 0.05 * abs(DG_A[i, j] - med), (i, j))
                for j in range(SZL - 1, SZL - 3, -1) for i in range(SX)
                if reserve[i, j]]
        cand.sort(reverse=True)
        picks = [s for _, s in cand[:2]]
        mined_t, metal, *_ = sim.real["LHOS"]
        got_oz = sum(metal[s] for s in picks) * 1000 / GRAMS_PER_OZ
        plan_oz = sum((DT_A * DG_A)[s] for s in picks) * 1000 / GRAMS_PER_OZ
        bulk = got_oz / max(plan_oz, 1e-9) - 1
        for s in picks:                                 # bulk stopes consumed
            seq.remove(s) if s in seq else None
            sim.status[s] = CLOSED
            sim.rem_kt[s] = 0

    # ---- open-loop execution ------------------------------------------
    PREPROD = 9                                         # capitalized months
    sim.op_start = PREPROD + 3          # commissioning months capitalized too
    dev_order = list(range(N_LEVELS - 1, -1, -1))       # top-down
    q = list(seq)
    done = False
    while not done:
        develop = []
        max_h = MAX_HEADINGS if sim.t < PREPROD else 2   # dev throttled in ops
        for lvl in dev_order:
            if sim.dev[lvl] < NX * DX - 1e-9:
                develop.append((lvl, HEADING_CAP))
            if len(develop) == max_h:
                break
        to_open = []
        if sim.t >= PREPROD:
            n_open = int((sim.status == OPEN).sum())
            for s in list(q):
                if n_open + len(to_open) >= MAX_OPEN:
                    break
                if sim.can_open(*s):
                    to_open.append(s); q.remove(s)
        quit_now = (sim.t >= PREPROD and not q
                    and int((sim.status == OPEN).sum()) == 0 and not to_open)
        obs, cash, done = sim.step({"develop": develop, "open": to_open,
                                    "quit": quit_now})
    return sim, dict(TH_A=TH_A, G_A=G_A, DT_A=DT_A, DG_A=DG_A,
                     reserve=reserve, seq=seq, bulk=bulk, pierce=pierce)


# ======================================================================
# Validation of the failure trajectory
# ======================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    machinery()
    N = 10
    rows = []
    for s in range(N):
        w = make_truth(200 + s)
        sim, info = run_deterministic(w, seed=s)
        kt, g = sim.head_grade_series()
        res = info["reserve"]
        plan_g = ((info["DT_A"] * info["DG_A"])[res].sum()
                  / max(info["DT_A"][res].sum(), 1e-9))
        q1 = slice(0, 12)
        # first producing quarter: first 3 months with feed
        prod_months = np.where(kt > 0.05)[0]
        prod_months = prod_months[prod_months >= sim.op_start]
        q1m = prod_months[:3] if len(prod_months) >= 3 else prod_months
        q1_g = (np.nansum(g[q1m] * kt[q1m]) / max(kt[q1m].sum(), 1e-9)
                if len(q1m) else np.nan)
        cum = np.cumsum(sim.cash[sim.op_start:]) / 1e6
        opk = kt[sim.op_start:]
        tot_kt = opk.sum()
        # realized all-in OPERATING cost/t (post pre-production window)
        rev_total = sum(m[1] for m in sim.milled[sim.op_start:]) * 1000 * REV_REAL
        cost_total = rev_total - np.sum(sim.cash[sim.op_start:])
        cost_pt = cost_total / max(tot_kt * 1000, 1)
        # planned all-in cost/t from the promised trajectory
        oz_p, cash_p = planned_trajectory(list(info['seq']), info['DT_A'],
                                          info['DG_A'])
        kt_p_tot = info['DT_A'][info['reserve']].sum()
        plan_rev = oz_p.sum() * GRAMS_PER_OZ / 1000 * REV_PLAN * 1000
        plan_cost_pt = (plan_rev - cash_p.sum()) / max(kt_p_tot * 1000, 1)
        rows.append(dict(
            plan_g=plan_g, q1_short=1 - q1_g / plan_g if plan_g else np.nan,
            cost_pt=cost_pt, over=cost_pt / max(plan_cost_pt, 1) - 1,
            dead=sim.dead if sim.dead else np.nan,
            min_cum=cum.min(), end_cum=cum[-1], bulk=info["bulk"],
            months=sim.t, kt=tot_kt,
        ))
        r = rows[-1]
        print(f"world {s}: plan {r['plan_g']:5.2f} g/t | Q1 {r['q1_short']:+.0%}"
              f" | C$/t {r['cost_pt']:.0f} (plan {plan_cost_pt:.0f}, "
            f"{r['over']:+.0%}) | insolvent m"
            f"{r['dead'] if not np.isnan(r['dead']) else '--'} "
            f"| bulk {r['bulk']:+.0%}")

    A = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}
    dead_frac = np.mean(~np.isnan(A["dead"]))
    print("\n===== FAILURE-TRAJECTORY VALIDATION (deterministic arm) =====")
    print(f"{'metric':46s} {'target':>16s} {'simulated':>12s}")
    print(f"{'Q1 head-grade shortfall vs plan (V2)':46s} {'45-65%':>16s} "
          f"{np.nanmean(A['q1_short']):>11.0%}")
    print(f"{'all-in operating cost overrun vs plan (V5)':46s} {'40-70%':>16s} "
          f"{np.nanmean(A['over']):>11.0%}")
    print(f"{'worlds reaching insolvency (V6)':46s} {'majority':>16s} "
          f"{dead_frac:>11.0%}")
    print(f"{'median insolvency month (V6)':46s} {'~18-30':>16s} "
          f"{np.nanmedian(A['dead']):>11.0f}")
    print(f"{'bulk sample reconciles POSITIVE (V8)':46s} {'frequent':>16s} "
          f"{np.mean(A['bulk'] > 0):>11.0%}")
    print(f"{'... while the mine still fails (V8 joint)':46s} {'the trap':>16s} "
          f"{np.mean((A['bulk'] > 0) & ~np.isnan(A['dead'])):>11.0%}")

    # ---- fig15: the death spiral on one world ----
    w = make_truth(203)
    sim, info = run_deterministic(w, seed=3)
    kt, g = sim.head_grade_series()
    oz_plan, cash_plan = planned_trajectory(info["seq"], info["DT_A"], info["DG_A"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), constrained_layout=True,
                             sharex=True)
    T = sim.t
    oz_real = np.array([m[1] for m in sim.milled]) * 1000 / GRAMS_PER_OZ
    axes[0].plot(range(len(oz_plan)), np.array(oz_plan) / 1, color="#2a9d8f",
                 lw=2, label="FS plan (promised)")
    axes[0].bar(range(T), oz_real, color="#d1495b", label="Realized")
    axes[0].set_ylabel("oz/month"); axes[0].legend(fontsize=8)
    axes[0].set_title("Ounces: the promise vs the mine", fontsize=10)

    res = info["reserve"]
    plan_g = ((info["DT_A"] * info["DG_A"])[res].sum()
              / max(info["DT_A"][res].sum(), 1e-9))
    axes[1].axhline(plan_g, color="#2a9d8f", lw=2, label=f"Reserve grade {plan_g:.1f} g/t")
    axes[1].plot(range(T), g, "o-", ms=3, color="#d1495b", label="Realized head grade")
    axes[1].axhline(COST_TRUE['LHOS'] / REV_REAL, color="purple", ls=":",
                    label="True breakeven (realized price)")
    axes[1].set_ylabel("g/t"); axes[1].legend(fontsize=8)

    cum_plan = np.cumsum(cash_plan) / 1e6
    cum_real = np.cumsum(sim.cash) / 1e6
    axes[2].plot(range(len(cum_plan)), cum_plan, color="#2a9d8f", lw=2,
                 label="Planned cumulative cash")
    axes[2].plot(range(T), cum_real, color="#d1495b", lw=2, label="Realized")
    axes[2].axhline(CREDIT_LIMIT / 1e6, color="k", ls="--", lw=1,
                    label="Credit limit (zone share)")
    if sim.dead:
        axes[2].axvline(sim.dead, color="k", lw=1)
        axes[2].text(sim.dead + 0.5, CREDIT_LIMIT / 1e6 * 0.5, "insolvency",
                     fontsize=8)
    axes[2].set_ylabel("C$M"); axes[2].set_xlabel("Month")
    axes[2].legend(fontsize=8)
    fig.suptitle("The deterministic arm reproduces the Madsen death spiral "
                 "(one hidden truth)", fontsize=12)
    fig.savefig("out/fig15_death_spiral.png", dpi=160)
    print("\nWrote fig15_death_spiral.png")
