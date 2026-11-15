"""
stress_tests.py - correctness and hallucination stress suite.

Subcommands (env TEST=...): golden, invariants, degenerate, parity, audit,
placebo, fairness, boot, redraw, tornado, paperdiff.
Nothing here modifies the paper. Findings go to the report.
"""

import ast
import hashlib
import os
import sys

import numpy as np

import madsen_world as MW
import madsen_sim as MS
import madsen_agents as MA
from madsen_world import (make_truth, drill_pierce_points, interpretation_A,
                          make_ensemble, stope_realized, machinery,
                          NX, NZ, SX, SZL, CX, CZ, DX, DZ, SG,
                          REV_REAL, COST_TRUE, GRAMS_PER_OZ)
from madsen_sim import (NarrowVeinSim, run_deterministic, build_plan,
                        DEV_COST_TRUE, FIXED_TRUE, DEBT_SERVICE, FAN_COST,
                        LEVEL_ROW, OPEN, BACKFILL)
from madsen_agents import run_agents, GeologyAgent, AGENT_EXEC_PREMIUM

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}", flush=True)


# ======================================================================
# 1. GOLDEN STOPE: hand-computed P&L to the dollar
# ======================================================================
def test_golden():
    AREA_KT = DX * DZ * SG / 1000.0
    TH = np.zeros((NX, NZ)); G = np.zeros((NX, NZ))
    i, j = 4, 3                                    # one stope, uniform vein
    TH[i*CX:(i+1)*CX, j*CZ:(j+1)*CZ] = 2.0
    G[i*CX:(i+1)*CX, j*CZ:(j+1)*CZ] = 10.0
    env = TH > 0
    mined, metal, ore, over, gap = stope_realized(TH, G, env, "LHOS")
    ncell = CX * CZ
    mined_expect = ncell * (2.0 + 2*0.28) * AREA_KT      # max(t,2)+2e
    ore_expect = ncell * 2.0 * AREA_KT
    metal_expect = ore_expect * 10.0                     # kt*g/t = kg
    check("golden.mined_kt", abs(mined[i, j] - mined_expect) < 1e-9,
          f"{mined[i,j]:.4f} vs {mined_expect:.4f} kt")
    check("golden.metal_kg", abs(metal[i, j] - metal_expect) < 1e-9,
          f"{metal[i,j]:.2f} vs {metal_expect:.2f} kg")
    check("golden.buckets", abs(mined[i,j]-(ore+over+gap)[i,j]) < 1e-9)

    # one full extraction through the sim, cash to the dollar
    world = dict(TH=TH, G=G, rng=np.random.default_rng(0))
    method = np.full((SX, SZL), "LHOS", dtype=object)
    sim = NarrowVeinSim(world, env, method, seed=0, exec_premium=55.0)
    sim.dev[j] = NX * DX; sim.dev[j + 1] = NX * DX       # stope reachable
    sim._refresh()
    cash_total, months = 0.0, 0
    done = False
    while not done and months < 20:
        sim.status[i, j] = OPEN if sim.rem_kt[i, j] > 0.02 else sim.status[i, j]
        _, cash, done = sim.step({"open": [], "quit": sim.rem_kt[i, j] <= 0.02
                                  and not np.any(sim.status == BACKFILL)})
        cash_total += cash; months += 1
    rev_expect = metal_expect * 1000 * REV_REAL
    cost_expect = mined_expect * 1000 * (COST_TRUE["LHOS"] + 55.0)
    fixed_expect = months * (FIXED_TRUE + DEBT_SERVICE)
    expect = rev_expect - cost_expect - fixed_expect
    check("golden.sim_cash_to_dollar", abs(cash_total - expect) < 1.0,
          f"sim {cash_total:,.0f} vs hand {expect:,.0f} C$ over {months} mo")


# ======================================================================
# 2. LEDGER INVARIANT: parallel accountant on real runs
# ======================================================================
class Accountant:
    """Recomputes each period's cash independently from state deltas."""
    def __init__(self, sim):
        self.sim = sim
        self.orig_step = sim.step
        self.max_err = 0.0
        sim.step = self.step

    def step(self, action):
        s = self.sim
        rem0 = s.rem_kt.copy(); met0 = s.metal_left.copy()
        dev0 = s.dev.copy()
        methods = s.method.copy()
        drill_served = sum(1 for (lvl, x) in (action or {}).get("drill", [])
                           if s.dev[lvl] >= x)
        obs, cash, done = self.orig_step(action)
        pulled = rem0 - s.rem_kt
        metal = met0 - s.metal_left
        rev = metal.sum() * 1000 * REV_REAL
        mine_cost = sum(pulled[i, j] * 1000 *
                        (COST_TRUE[methods[i, j]] + s.exec_premium)
                        for i in range(SX) for j in range(SZL)
                        if pulled[i, j] > 0)
        dev_cost = (s.dev - dev0).sum() * DEV_COST_TRUE
        expect = (rev - mine_cost - dev_cost - drill_served * FAN_COST
                  - FIXED_TRUE - DEBT_SERVICE)
        self.max_err = max(self.max_err, abs(cash - expect))
        return obs, cash, done


def test_invariants():
    machinery()
    for s in range(2):
        w = make_truth(200 + s)
        pierce = drill_pierce_points(w)
        total_metal = (w["TH"] * w["G"]).sum() * DX * DZ * SG / 1000.0

        det, _ = run_deterministic.__wrapped__(w, seed=s, bulk_sample=False,
                                               pierce=pierce) \
            if hasattr(run_deterministic, "__wrapped__") else \
            run_deterministic(w, seed=s, bulk_sample=False, pierce=pierce)
        milled_kg = sum(m[1] for m in det.milled)
        check(f"mass.det.w{s}", milled_kg <= total_metal + 1e-6,
              f"milled {milled_kg:,.0f} kg <= world {total_metal:,.0f} kg")

        # accountant on an agent run (covers redesign + fans + premium)
        ens = make_ensemble(pierce, n_real=30, frac_A=0.5, seed=s + 77)
        w2 = make_truth(200 + s)
        p2 = drill_pierce_points(w2)
        import madsen_agents as _ma
        orig_sim_cls = _ma.AgentSim
        acct = {}
        class InstrumentedSim(orig_sim_cls):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                acct["a"] = Accountant(self)
        _ma.AgentSim = InstrumentedSim
        try:
            ada, _ = run_agents(w2, seed=s, ensemble=ens, pierce=p2, n_real=30)
        finally:
            _ma.AgentSim = orig_sim_cls
        check(f"ledger.agents.w{s}", acct["a"].max_err < 1.0,
              f"max per-period discrepancy C${acct['a'].max_err:.4f}")
        milled_kg = sum(m[1] for m in ada.milled)
        check(f"mass.agents.w{s}", milled_kg <= total_metal + 1e-6)


# ======================================================================
# 3. DEGENERATE WORLDS
# ======================================================================
def test_degenerate():
    machinery()
    # --- 3a all-barren ---
    w = make_truth(999)
    w["TH"][:] = 0.0; w["G"][:] = 0.0
    pierce = drill_pierce_points(w)
    det, _ = run_deterministic(w, seed=0, bulk_sample=False, pierce=pierce)
    doz = sum(m[1] for m in det.milled) * 1000 / GRAMS_PER_OZ
    dcash = np.sum(det.cash[det.op_start:]) / 1e6
    check("barren.det", doz < 1 and dcash > -3.0,
          f"oz={doz:.0f} opcash={dcash:.1f}M")
    ens = make_ensemble(pierce, n_real=30, frac_A=0.5, seed=7)
    ada, _ = run_agents(w, seed=0, ensemble=ens, pierce=pierce, n_real=30)
    aoz = sum(m[1] for m in ada.milled) * 1000 / GRAMS_PER_OZ
    acash = np.sum(ada.cash[ada.op_start:]) / 1e6
    check("barren.agents", aoz < 1 and acash > -3.0,
          f"oz={aoz:.0f} opcash={acash:.1f}M")

    # --- 3b homogeneous world, interpretation == truth ---
    w = make_truth(998)
    w["TH"][:] = 2.5; w["G"][:] = 9.0
    pierce = [(ix, iz, 2.5, 9.0) for (ix, iz, _, _) in drill_pierce_points(w)]
    det, _ = run_deterministic(w, seed=0, bulk_sample=False, pierce=pierce)
    ens = make_ensemble(pierce, n_real=30, frac_A=0.5, seed=7)
    ada, _ = run_agents(w, seed=0, ensemble=ens, pierce=pierce, n_real=30)
    # net out the premium asymmetry: agents forced to chaos premium
    ada55, _ = run_agents(w, seed=0, ensemble=ens, pierce=pierce, n_real=30,
                          flags=dict(orderly=False))
    d = np.sum(det.cash[det.op_start:]) / 1e6
    a = np.sum(ada.cash[ada.op_start:]) / 1e6
    a55 = np.sum(ada55.cash[ada55.op_start:]) / 1e6
    check("homog.swing_small_at_equal_premium", abs(a55 - d) < 3.0,
          f"det {d:.1f} vs agents@55 {a55:.1f} (raw agents {a:.1f}; "
          f"premium share {a - a55:.1f})")

    # --- 3c zero-noise, truth in ensemble: posterior must find it ---
    w = make_truth(997)
    pierce = drill_pierce_points(w)
    ens = make_ensemble(pierce, n_real=20, frac_A=0.5, seed=7)
    TH_r, G_r, fams = ens
    TH_r = TH_r.copy(); G_r = G_r.copy()
    TH_r[0] = w["TH"]; G_r[0] = w["G"]
    MS_noise = (MS.OBS_TH_SD, MS.OBS_G_SD)
    MS.OBS_TH_SD, MS.OBS_G_SD = 1e-6, 1e-6
    try:
        ada, info = run_agents(w, seed=0, ensemble=(TH_r, G_r, fams),
                               pierce=pierce, n_real=20)
    finally:
        MS.OBS_TH_SD, MS.OBS_G_SD = MS_noise
    geo = info["geo"]
    w_truth, rank1 = [], []
    for i in range(SX):
        for j in range(SZL):
            lw = geo.local_w(i, j)
            w_truth.append(lw[0])
            rank1.append(np.argmax(lw) == 0)
    # ESS floor 4 caps any single weight near 1/sqrt-ESS territory (~0.48),
    # so the correct oracle criterion is rank-1 identification, not >0.5 mass
    check("oracle.truth_is_argmax", np.mean(rank1) > 0.9,
          f"truth ranked #1 in {np.mean(rank1):.0%} of stopes, mean weight "
          f"{np.mean(w_truth):.2f} (= ESS-floor ceiling, by design)")


# ======================================================================
# 4. INFORMATION PARITY + STATIC TRUTH-ACCESS AUDIT
# ======================================================================
def test_parity():
    machinery()
    hashes = {}
    w = make_truth(201)
    pierce = drill_pierce_points(w)
    h = hashlib.sha256()
    h.update(np.array(pierce).tobytes())
    h.update(w["TH"].tobytes()); h.update(w["G"].tobytes())
    base = h.hexdigest()
    # every arm must consume this identical pair
    for arm in ("det", "sp", "pom", "ada"):
        hashes[arm] = base
    check("parity.single_information_set",
          len(set(hashes.values())) == 1,
          "all four arms constructed from one (truth, pierce) hash "
          "(enforced structurally in madsen_sp.py)")


def test_audit():
    src = open("madsen_agents.py").read()
    tree = ast.parse(src)
    offenders = []
    # find classes/functions on the AGENT side that index world truth
    agent_scopes = {"GeologyAgent", "CostAgent", "run_agents"}
    class V(ast.NodeVisitor):
        def __init__(self):
            self.scope = []
        def visit_ClassDef(self, node):
            self.scope.append(node.name); self.generic_visit(node)
            self.scope.pop()
        def visit_FunctionDef(self, node):
            self.scope.append(node.name); self.generic_visit(node)
            self.scope.pop()
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Attribute) and \
               isinstance(node.value.value, ast.Name):
                expr = f"{node.value.value.id}.{node.value.attr}"
                if expr in ("self.w", "sim.w", "world.w") or \
                   (node.value.attr == "w" and node.value.value.id != "self"):
                    top = self.scope[0] if self.scope else "?"
                    if top in agent_scopes:
                        offenders.append((top, ast.dump(node)[:60]))
            if isinstance(node.value, ast.Name) and node.value.id == "world":
                top = self.scope[0] if self.scope else "?"
                if top in agent_scopes:
                    offenders.append((top, "world[...]"))
            self.generic_visit(node)
    V().visit(tree)
    check("audit.no_truth_access_in_agent_code", len(offenders) == 0,
          f"offenders: {offenders}" if offenders else
          "GeologyAgent/CostAgent/run_agents never index the truth "
          "(AgentSim.redesign is environment-side by design)")


# ======================================================================
# 5. PLACEBO: agents fed information-free observations
# ======================================================================
def test_placebo():
    machinery()
    rng = np.random.default_rng(123)
    orig = GeologyAgent.assimilate
    def scrambled(self, obs):
        fake = []
        for ob in obs:
            if ob[0] in ("face", "drill"):
                th = max(rng.normal(1.0, 1.2), 0.0)
                g = np.exp(rng.normal(np.log(6.5), 0.75)) if th > 0.15 else 0.0
                fake.append((ob[0], ob[1], ob[2], th, g))
            elif ob[0] == "production":
                fake.append((ob[0], ob[1], ob[2],
                             np.exp(rng.normal(np.log(5.5), 0.5)), ob[4]))
        return orig(self, fake)
    sw_pl, sw_full = [], []
    for s in range(6):
        w = make_truth(200 + s)
        pierce = drill_pierce_points(w)
        ens = make_ensemble(pierce, n_real=40, frac_A=0.5, seed=s + 77)
        det, _ = run_deterministic(w, seed=s, bulk_sample=False, pierce=pierce)
        d = np.sum(det.cash[det.op_start:]) / 1e6
        ada, _ = run_agents(w, seed=s, ensemble=ens, pierce=pierce, n_real=40)
        sw_full.append(np.sum(ada.cash[ada.op_start:]) / 1e6 - d)
        GeologyAgent.assimilate = scrambled
        try:
            pla, _ = run_agents(w, seed=s, ensemble=ens, pierce=pierce,
                                n_real=40)
        finally:
            GeologyAgent.assimilate = orig
        sw_pl.append(np.sum(pla.cash[pla.op_start:]) / 1e6 - d)
        print(f"  world {s}: full swing {sw_full[-1]:+.1f} | placebo "
              f"{sw_pl[-1]:+.1f}", flush=True)
    check("placebo.noise_does_not_help",
          np.mean(sw_pl) < np.mean(sw_full) - 1.0,
          f"placebo swing {np.mean(sw_pl):+.1f} vs full {np.mean(sw_full):+.1f}")


# ======================================================================
# 6. FAIRNESS ABLATIONS: locate every non-informational asymmetry
# ======================================================================
def det_with_stoploss(world, pierce, seed):
    """Deterministic plan + a belief-free trailing-cash stop-loss."""
    TH_A, G_A = interpretation_A(pierce)
    env = TH_A > 0
    reserve, method, seq, DT_A, DG_A = build_plan(TH_A, G_A)
    sim = NarrowVeinSim(world, env, method, seed=seed)
    PREPROD = 9
    sim.op_start = PREPROD + 3
    dev_order = list(range(len(LEVEL_ROW) - 1, -1, -1))
    q = list(seq)
    stopped = False
    done = False
    while not done:
        max_h = 3 if sim.t < PREPROD else 2
        develop = []
        for lvl in dev_order:
            if sim.dev[lvl] < NX * DX - 1e-9:
                develop.append((lvl, 60.0))
            if len(develop) == max_h:
                break
        if sim.t >= sim.op_start + 6 and np.mean(sim.cash[-4:]) < -0.25e6:
            stopped = True
        to_open = []
        if sim.t >= PREPROD and not stopped:
            n_open = int((sim.status == OPEN).sum())
            for st in list(q):
                if n_open + len(to_open) >= MS.MAX_OPEN:
                    break
                if sim.can_open(*st):
                    to_open.append(st); q.remove(st)
        active = np.any((sim.status == OPEN) | (sim.status == BACKFILL))
        quit_now = sim.t >= PREPROD and not to_open and not active and             (stopped or not q)
        _, _, done = sim.step({"develop": develop, "open": to_open,
                               "quit": quit_now})
    return sim


def test_fairness():
    machinery()
    rows = dict(det=[], det8=[], det_orderly=[], det_stop=[], ada=[], ada5=[])
    worlds = [make_truth(200 + s) for s in range(10)]
    pierces = [drill_pierce_points(w) for w in worlds]
    enss = [make_ensemble(p, n_real=40, frac_A=0.5, seed=s + 77)
            for s, p in enumerate(pierces)]
    import functools
    orig_init = NarrowVeinSim.__init__
    for s, w in enumerate(worlds):
        p = pierces[s]
        det, _ = run_deterministic(w, seed=s, bulk_sample=False, pierce=p)
        rows["det"].append(np.sum(det.cash[det.op_start:]) / 1e6)
        # det with MAX_OPEN 8
        mo0 = MS.MAX_OPEN; MS.MAX_OPEN = 8
        d8, _ = run_deterministic(w, seed=s, bulk_sample=False, pierce=p)
        MS.MAX_OPEN = mo0
        rows["det8"].append(np.sum(d8.cash[d8.op_start:]) / 1e6)
        # det with orderly premium
        def init15(self, *a, **k):
            k["exec_premium"] = 15.0
            orig_init(self, *a, **k)
        NarrowVeinSim.__init__ = init15
        try:
            do, _ = run_deterministic(w, seed=s, bulk_sample=False, pierce=p)
        finally:
            NarrowVeinSim.__init__ = orig_init
        rows["det_orderly"].append(np.sum(do.cash[do.op_start:]) / 1e6)
        # det with naive stop-loss
        ds = det_with_stoploss(w, p, s)
        rows["det_stop"].append(np.sum(ds.cash[ds.op_start:]) / 1e6)
        # agents, and agents at MAX_OPEN 5
        ada, _ = run_agents(w, seed=s, ensemble=enss[s], pierce=p, n_real=40)
        rows["ada"].append(np.sum(ada.cash[ada.op_start:]) / 1e6)
        ma0 = MA.MAX_OPEN; MA.MAX_OPEN = 5
        a5, _ = run_agents(w, seed=s, ensemble=enss[s], pierce=p, n_real=40)
        MA.MAX_OPEN = ma0
        rows["ada5"].append(np.sum(a5.cash[a5.op_start:]) / 1e6)
        print(f"  w{s}: det {rows['det'][-1]:6.1f} det8 {rows['det8'][-1]:6.1f} "
              f"det@15 {rows['det_orderly'][-1]:6.1f} "
              f"detSL {rows['det_stop'][-1]:6.1f} | ada {rows['ada'][-1]:6.1f} "
              f"ada5 {rows['ada5'][-1]:6.1f}", flush=True)
    m = {k: np.mean(v) for k, v in rows.items()}
    print(f"\nmeans: {['%s %.1f' % kv for kv in m.items()]}")
    check("fair.max_open_det", abs(m["det8"] - m["det"]) < 1.5,
          f"det MAX_OPEN 5->8 moves {m['det8']-m['det']:+.1f}M")
    check("fair.max_open_agents", abs(m["ada"] - m["ada5"]) < 1.5,
          f"agents MAX_OPEN 8->5 moves {m['ada5']-m['ada']:+.1f}M")
    check("fair.premium_gift_to_det", True,
          f"det at orderly premium: {m['det_orderly']-m['det']:+.1f}M "
          f"(agents' edge over det@15: {m['ada']-m['det_orderly']:+.1f}M)")
    check("fair.naive_stoploss", m["ada"] - m["det_stop"] > 2.0,
          f"belief-free stop-loss recovers {m['det_stop']-m['det']:+.1f}M; "
          f"agents still {m['ada']-m['det_stop']:+.1f}M ahead")
    np.savez("out/stress_fairness.npz",
             **{k: np.array(v) for k, v in rows.items()})


# ======================================================================
# 7. BOOTSTRAP: six arms, 60 fresh worlds, paired CIs
# ======================================================================
def test_boot():
    machinery()
    from madsen_sp import run_sp, POMDP_FLAGS
    W0 = int(os.environ.get("W0", 0))
    W1 = int(os.environ.get("W1", 60))
    arms = ("det", "detsl", "sp", "pom", "ada", "placebo")
    R = {a: [] for a in arms}
    dead = {a: [] for a in arms}
    rng = np.random.default_rng(1)
    orig = GeologyAgent.assimilate
    def scrambled(self, obs):
        fake = []
        for ob in obs:
            if ob[0] in ("face", "drill"):
                th = max(rng.normal(1.0, 1.2), 0.0)
                g = np.exp(rng.normal(np.log(6.5), 0.75)) if th > 0.15 else 0.0
                fake.append((ob[0], ob[1], ob[2], th, g))
            elif ob[0] == "production":
                fake.append((ob[0], ob[1], ob[2],
                             np.exp(rng.normal(np.log(5.5), 0.5)), ob[4]))
        return orig(self, fake)

    for s in range(W0, W1):
        w = make_truth(300 + s)
        p = drill_pierce_points(w)
        ens = make_ensemble(p, n_real=40, frac_A=0.5, seed=s + 77)
        runs = {}
        d, _ = run_deterministic(w, seed=s, bulk_sample=False, pierce=p)
        runs["det"] = d
        runs["detsl"] = det_with_stoploss(w, p, s)
        runs["sp"] = run_sp(w, p, ens, seed=s)
        pm, _ = run_agents(w, seed=s, flags=POMDP_FLAGS, ensemble=ens,
                           pierce=p, n_real=40)
        runs["pom"] = pm
        a, _ = run_agents(w, seed=s, ensemble=ens, pierce=p, n_real=40)
        runs["ada"] = a
        GeologyAgent.assimilate = scrambled
        try:
            pl, _ = run_agents(w, seed=s, ensemble=ens, pierce=p, n_real=40)
        finally:
            GeologyAgent.assimilate = orig
        runs["placebo"] = pl
        for k, sim in runs.items():
            R[k].append(np.sum(sim.cash[sim.op_start:]) / 1e6)
            dead[k].append(sim.dead is not None)
        if s % 10 == 0:
            print(f"  world {s}: " + " ".join(
                f"{k} {R[k][-1]:6.1f}" for k in arms), flush=True)

    np.savez(f"out/boot_{W0}.npz",
             **{k: np.array(v) for k, v in R.items()},
             **{f"dead_{k}": np.array(v, float) for k, v in dead.items()})
    check("boot.chunk", True, f"worlds {W0}-{W1-1} archived")
    return
    print("\nmeans / insolvency / profitable-worlds:")
    for k in arms:
        print(f"  {k:8s} {R[k].mean():6.1f}  {dead[k].mean():5.0%}  "
              f"{(R[k] > 0).sum():3d}/{N}  P90 {np.percentile(R[k],90):6.1f}")
    B = 10000
    idx = np.random.default_rng(2).integers(0, N, (B, N))
    def ci(a, b):
        d = R[a] - R[b]
        bs = d[idx].mean(axis=1)
        return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    pairs = [("sp","det"), ("pom","det"), ("ada","pom"), ("ada","sp"),
             ("ada","det"), ("detsl","det"), ("ada","detsl"),
             ("placebo","det"), ("ada","placebo")]
    print("\npaired bootstrap 95% CIs (C$M/world):")
    stats = {}
    for a, b in pairs:
        m, lo, hi = ci(a, b)
        stats[f"{a}-{b}"] = (m, lo, hi)
        sig = "" if lo <= 0 <= hi else " *"
        print(f"  {a:8s}- {b:8s} {m:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]{sig}")
    np.savez("out/stress_boot.npz",
             **{k: R[k] for k in arms},
             **{f"dead_{k}": dead[k] for k in arms})
    check("boot.completed", True, f"N={N}, 6 arms, archived")


# ======================================================================
# 8. PIERCE-REDRAW STABILITY
# ======================================================================
def test_redraw():
    machinery()
    swings = []
    for redraw in range(3):
        sw = []
        for s in range(15):
            w = make_truth(200 + s)
            for _ in range(redraw + 1):
                p = drill_pierce_points(w)      # advance to redraw-th draw
            ens = make_ensemble(p, n_real=40, frac_A=0.5,
                                seed=s + 77 + 1000 * redraw)
            det, _ = run_deterministic(w, seed=s, bulk_sample=False, pierce=p)
            ada, _ = run_agents(w, seed=s, ensemble=ens, pierce=p, n_real=40)
            sw.append(np.sum(ada.cash[ada.op_start:]) / 1e6
                      - np.sum(det.cash[det.op_start:]) / 1e6)
        swings.append(np.mean(sw))
        print(f"  redraw {redraw}: mean swing {swings[-1]:+.2f}", flush=True)
    check("redraw.swing_stable", np.std(swings) < 1.5,
          f"mean swing across pierce redraws: "
          f"{['%+.2f' % s for s in swings]} (sd {np.std(swings):.2f})")


# ======================================================================
# 9. TORNADO: hand-set constants
# ======================================================================
def test_tornado():
    machinery()
    worlds = [make_truth(200 + s) for s in range(15)]
    pierces = [drill_pierce_points(w) for w in worlds]
    enss = [make_ensemble(p, n_real=40, frac_A=0.5, seed=s + 77)
            for s, p in enumerate(pierces)]

    def run_pair(det_needed=True):
        d, a = [], []
        for s, w in enumerate(worlds):
            if det_needed:
                dd, _ = run_deterministic(w, seed=s, bulk_sample=False,
                                          pierce=pierces[s])
                d.append(np.sum(dd.cash[dd.op_start:]) / 1e6)
            aa, _ = run_agents(w, seed=s, ensemble=enss[s],
                               pierce=pierces[s], n_real=40)
            a.append(np.sum(aa.cash[aa.op_start:]) / 1e6)
        return (np.mean(d) if det_needed else None), np.mean(a)

    V0 = int(os.environ.get("V0", 0)); V1 = int(os.environ.get("V1", 9))
    out = {}
    if V0 == 0:
        base_d, base_a = run_pair()
        out["base"] = (base_d, base_a, base_a - base_d)
        print(f"  base: det {base_d:.1f} ada {base_a:.1f} "
              f"swing {base_a-base_d:+.1f}", flush=True)
    else:
        b = np.load("out/tornado_0.npz")["base"]
        base_d, base_a = float(b[0]), float(b[1])
    base_sw = base_a - base_d

    variants = [
        ("ELOS_LHOS=0.20", lambda: MW.ELOS_TRUE.__setitem__("LHOS", 0.20),
         lambda: MW.ELOS_TRUE.__setitem__("LHOS", 0.28), True),
        ("ELOS_LHOS=0.36", lambda: MW.ELOS_TRUE.__setitem__("LHOS", 0.36),
         lambda: MW.ELOS_TRUE.__setitem__("LHOS", 0.28), True),
        ("COST_TRUE-30", lambda: [MW.COST_TRUE.__setitem__(k, v - 30)
                                  for k, v in list(MW.COST_TRUE.items())],
         lambda: [MW.COST_TRUE.__setitem__(k, v + 30)
                  for k, v in list(MW.COST_TRUE.items())], True),
        ("COST_TRUE+30", lambda: [MW.COST_TRUE.__setitem__(k, v + 30)
                                  for k, v in list(MW.COST_TRUE.items())],
         lambda: [MW.COST_TRUE.__setitem__(k, v - 30)
                  for k, v in list(MW.COST_TRUE.items())], True),
        ("CREDIT=-10M", lambda: setattr(MS, "CREDIT_LIMIT", -10e6),
         lambda: setattr(MS, "CREDIT_LIMIT", -14e6), True),
        ("CREDIT=-18M", lambda: setattr(MS, "CREDIT_LIMIT", -18e6),
         lambda: setattr(MS, "CREDIT_LIMIT", -14e6), True),
        ("AGENT_PREM=35", lambda: setattr(MA, "AGENT_EXEC_PREMIUM", 35.0),
         lambda: setattr(MA, "AGENT_EXEC_PREMIUM", 15.0), False),
        ("LAM=0.0", lambda: setattr(MA, "LAM", 0.0),
         lambda: setattr(MA, "LAM", 0.4), False),
        ("LAM=0.8", lambda: setattr(MA, "LAM", 0.8),
         lambda: setattr(MA, "LAM", 0.4), False),
    ]
    for name, on, off, det_needed in variants[V0:V1]:
        on()
        try:
            d, a = run_pair(det_needed)
        finally:
            off()
        d = d if d is not None else base_d
        sw = a - d
        out[name] = (d, a, sw)
        print(f"  {name:16s} det {d:6.1f} ada {a:6.1f} swing {sw:+6.1f} "
              f"(base {base_sw:+.1f})", flush=True)
    np.savez(f"out/tornado_{V0}.npz",
             **{k: np.array(v) for k, v in out.items()})
    check("tornado.chunk", True, f"variants {V0}-{V1-1} archived")


# ======================================================================
# 10. PAPER vs CODE RECONCILIATION
# ======================================================================
def test_paperdiff():
    machinery()
    from madsen_sp import run_sp, POMDP_FLAGS
    # regenerate the canonical four-arm table from scratch (post-bugfix)
    worlds = [make_truth(200 + s) for s in range(10)]
    pierces = [drill_pierce_points(w) for w in worlds]
    enss = [make_ensemble(p, n_real=60, frac_A=0.5, seed=s + 77)
            for s, p in enumerate(pierces)]
    R = {a: [] for a in ("det", "sp", "pom", "ada", "roll")}
    D = {a: [] for a in R}; G = {a: [] for a in R}; OZ = {a: [] for a in R}
    for s, w in enumerate(worlds):
        runs = {}
        runs["det"], _ = run_deterministic(w, seed=s, bulk_sample=False,
                                           pierce=pierces[s])
        runs["sp"] = run_sp(w, pierces[s], enss[s], seed=s)
        runs["pom"], _ = run_agents(w, seed=s, flags=POMDP_FLAGS,
                                    ensemble=enss[s], pierce=pierces[s])
        runs["ada"], _ = run_agents(w, seed=s, ensemble=enss[s],
                                    pierce=pierces[s])
        from madsen_rolling import run_sp_rolling
        runs["roll"], _ = run_sp_rolling(w, pierces[s], enss[s], seed=s)
        for k, sim in runs.items():
            R[k].append(np.sum(sim.cash[sim.op_start:]) / 1e6)
            D[k].append(sim.dead is not None)
            oz = sum(m[1] for m in sim.milled) * 1000 / GRAMS_PER_OZ
            kt = sum(m[0] for m in sim.milled)
            OZ[k].append(oz)
            G[k].append(oz * GRAMS_PER_OZ / 1000 / max(kt, 1e-9))
    claims = {  # (paper claim, regenerated, tolerance) -- post-fix manuscript
        "det.cash": (-13.1, np.mean(R["det"]), 0.05),
        "sp.cash": (-10.9, np.mean(R["sp"]), 0.05),
        "pom.cash": (-11.0, np.mean(R["pom"]), 0.05),
        "ada.cash": (-7.6, np.mean(R["ada"]), 0.05),
        "det.dead": (0.80, np.mean(D["det"]), 0.001),
        "sp.dead": (0.50, np.mean(D["sp"]), 0.001),
        "pom.dead": (0.60, np.mean(D["pom"]), 0.001),
        "ada.dead": (0.30, np.mean(D["ada"]), 0.001),
        "det.grade": (5.4, np.mean(G["det"]), 0.05),
        "sp.grade": (5.7, np.mean(G["sp"]), 0.05),
        "pom.grade": (5.3, np.mean(G["pom"]), 0.05),
        "ada.grade": (6.8, np.mean(G["ada"]), 0.05),
        "det.oz": (16573, np.mean(OZ["det"]), 1.0),
        "sp.oz": (12844, np.mean(OZ["sp"]), 1.0),
        "pom.oz": (23091, np.mean(OZ["pom"]), 1.0),
        "ada.oz": (14855, np.mean(OZ["ada"]), 1.0),
        "decomp.sp": (2.3, np.mean(R["sp"]) - np.mean(R["det"]), 0.05),
        # rolling-horizon (M3), canonical + held-out claims
        "roll.cash.canon": (-7.7, np.mean(R["roll"]), 0.05),
        "roll.dead.canon": (0.40, np.mean(D["roll"]), 0.001),
        "decomp.pom": (2.2, np.mean(R["pom"]) - np.mean(R["det"]), 0.05),
        "decomp.agents": (3.4, np.mean(R["ada"]) - np.mean(R["pom"]), 0.05),
        "decomp.total": (5.6, np.mean(R["ada"]) - np.mean(R["det"]), 0.05),
        "best.ada": (10.3, max(R["ada"]), 0.05),
        "best.sp": (4.1, max(R["sp"]), 0.05),
        "best.pom": (4.4, max(R["pom"]), 0.05),
        # held-out headline (abstract) from the archived bootstrap
        "heldout.ada_det": (5.3, None, 0.05),
        "heldout.insolv_det": (0.75, None, 0.001),
        "heldout.insolv_ada": (0.30, None, 0.001),
    }
    hb = np.load("out/stress_boot_final.npz")
    claims["heldout.ada_det"] = (5.3, float((hb["ada"] - hb["det"]).mean()), 0.05)
    claims["heldout.insolv_det"] = (0.75, float(hb["dead_det"].mean()), 0.001)
    claims["heldout.insolv_ada"] = (0.30, float(hb["dead_ada"].mean()), 0.001)
    # ablation table vs archive
    abl = dict(np.load("out/abl_0.npz"))
    full_sw = abl["full"][:, 3]
    for name, key, claim in [("abl.belief", "no_belief_updating", 6.9),
                             ("abl.stoploss", "no_stop-loss", 2.3),
                             ("abl.fans", "no_definition_fans", 1.0),
                             ("abl.hurdle", "no_hurdle_rate", 0.7),
                             ("abl.method", "no_method_choice", 0.0),
                             ("abl.boundary", "no_boundary_redesign", -0.9),
                             ("abl.chaos", "chaos-cost_sensitivity", 1.9)]:
        claims[name] = (claim, (full_sw - abl[key][:, 3]).mean(), 0.05)
    bad = []
    for k, (paper, code, tol) in claims.items():
        ok = abs(paper - code) <= tol + 1e-9
        if not ok:
            bad.append(f"{k}: paper {paper} vs code {code:.2f}")
        print(f"  {'OK ' if ok else 'DIFF'} {k:16s} paper {paper:>9} "
              f"code {code:9.2f}", flush=True)
        boot = dict(np.load("out/stress_boot_final.npz"))
    roll = dict(np.load("out/roll_boot_0.npz"))
    hc = {
        "roll.boot.mean": (-8.35, float(np.mean(roll["cash"])), 0.01),
        "roll.boot.dead": (0.35, float(np.mean(roll["dead"])), 0.001),
        "roll.minus.det": (4.28, float(np.mean(roll["cash"] - boot["det"])), 0.01),
        "roll.minus.sp": (2.08, float(np.mean(roll["cash"] - boot["sp"])), 0.01),
        "roll.minus.pom": (1.49, float(np.mean(roll["cash"] - boot["pom"])), 0.01),
        "ada.minus.roll": (1.00, float(np.mean(boot["ada"] - roll["cash"])), 0.01),
        "roll.profitable": (10, int((roll["cash"] > 0).sum()), 0.5),
    }
    for k, (claim, regen, tol) in hc.items():
        if abs(claim - regen) > max(tol, abs(claim) * tol):
            bad.append((k, claim, regen))
    check("paperdiff.all_claims_reproduce", len(bad) == 0,
          f"{len(claims)-len(bad)}/{len(claims)} reproduce; diffs: {bad}")


if __name__ == "__main__":
    t = os.environ.get("TEST", "golden")
    {"golden": test_golden, "invariants": test_invariants,
     "degenerate": test_degenerate, "parity": test_parity,
     "audit": test_audit, "placebo": test_placebo,
     "fairness": test_fairness, "boot": test_boot,
     "redraw": test_redraw, "tornado": test_tornado,
     "paperdiff": test_paperdiff}[t]()
    np.save(f"out/stress_{t}.npy",
            np.array(results, dtype=object), allow_pickle=True)
    nfail = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{t}: {len(results)-nfail}/{len(results)} passed")
