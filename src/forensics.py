"""
Forensic audit of committed stopes against the hidden truth.

REPRODUCTION MODULE. Given the per-approach open/backfill logs, this
scores every committed stope on its TRUE diluted grade against a
full-cost breakeven, and reduces the logs to the named mining
behaviours of Chapter 6 (effective cut-off, losers mined, payers
foregone, fans that changed commitments, method mix, months of
negative trailing cash, development metres). The only input the real
mine lacks is the referee: the hidden truth, which the simulator
supplies. Every quantity here is observable at a producing mine
except that referee.
"""
import numpy as np
import madsen_sim as MS
import madsen_agents as MA
from madsen_world import (make_truth, drill_pierce_points, make_ensemble,
                          interpretation_A, stope_realized, machinery,
                          SX, SZL, REV_REAL, COST_TRUE, GRAMS_PER_OZ)
from madsen_sim import (NarrowVeinSim, run_deterministic, build_plan,
                        FIXED_TRUE, DEBT_SERVICE, MILL_CAP)
from madsen_agents import run_agents
from madsen_sp import run_sp, POMDP_FLAGS

LOG = []
_orig_step = NarrowVeinSim.step

def logged_step(self, action):
    a = action or {}
    served = sum(1 for (lvl, x) in a.get("drill", []) if self.dev[lvl] >= x)
    opens = [(self.t, i, j, self.method[i, j]) for (i, j) in a.get("open", [])]
    milled_before = len(self.milled)
    obs, cash, done = _orig_step(self, action)
    LOG.append(dict(t=self.t - 1, opens=opens, fans=served,
                    dev=float(self.dev.sum()), cash=cash,
                    milled=self.milled[milled_before:]))
    return obs, cash, done


def true_stope_econ(world, premium):
    """Full-cost breakeven classification on TRUE diluted grade, det design
    envelope, per method. Returns g_dil (LHOS basis) and pays matrices."""
    TH_A, G_A = interpretation_A(world["_pierce"])
    env = TH_A > 0
    out = {}
    for m in ("LHOS", "CF"):
        method = np.full((SX, SZL), m, dtype=object)
        mined, metal, ore, over, gap = stope_realized(
            world["TH"], world["G"], env, m)
        g_dil = np.where(mined > 1e-9, metal / np.maximum(mined, 1e-9), 0.0)
        alloc = (FIXED_TRUE + DEBT_SERVICE) / (MILL_CAP * 1000)   # C$/t
        margin = g_dil * REV_REAL - (COST_TRUE[m] + premium) - alloc
        out[m] = (g_dil, margin > 0, mined)
    return out


def run_world(s):
    global LOG
    w = make_truth(200 + s)
    pierce = drill_pierce_points(w)
    w["_pierce"] = pierce
    ens = make_ensemble(pierce, n_real=60, frac_A=0.5, seed=s + 77)
    arms = {}
    NarrowVeinSim.step = logged_step
    try:
        LOG = []; det, _ = run_deterministic(w, seed=s, bulk_sample=False,
                                             pierce=pierce)
        arms["det"] = (det, list(LOG), 55.0)
        LOG = []; sp = run_sp(w, pierce, ens, seed=s)
        arms["sp"] = (sp, list(LOG), 15.0)
        LOG = []; pom, _ = run_agents(w, seed=s, flags=POMDP_FLAGS,
                                      ensemble=ens, pierce=pierce)
        arms["pom"] = (pom, list(LOG), 15.0)
        LOG = []; ada, _ = run_agents(w, seed=s, ensemble=ens, pierce=pierce)
        arms["ada"] = (ada, list(LOG), 15.0)
    finally:
        NarrowVeinSim.step = _orig_step

    rows = {}
    for k, (sim, log, prem) in arms.items():
        econ = true_stope_econ(w, prem)
        opens = [o for e in log for o in e["opens"]]
        n_open = len(opens)
        gd, pays, mined_kt = econ["LHOS"][0], None, econ["LHOS"][2]
        g_of, pay_of, losers, cf_n = [], [], 0, 0
        for (t, i, j, m) in opens:
            g, p, _ = econ[m if m in econ else "LHOS"]
            g_of.append(g[i, j]); pay_of.append(p[i, j])
            losers += 0 if p[i, j] else 1
            cf_n += 1 if m == "CF" else 0
        # paying stopes never opened (opportunity discipline the other way)
        opened_set = {(i, j) for (_, i, j, _) in opens}
        pays_lhos = econ["LHOS"][1]
        missed = sum(1 for i in range(SX) for j in range(SZL)
                     if pays_lhos[i, j] and mined_kt[i, j] > 0.5
                     and (i, j) not in opened_set)
        # bleeding months: milled while trailing-4 op cash negative
        cash = np.array([e["cash"] for e in log])
        op0 = sim.op_start
        bleed = 0
        for t in range(op0 + 4, len(cash)):
            if cash[t-4:t].mean() < -0.25e6 and log[t]["milled"]:
                bleed += 1
        fans = sum(e["fans"] for e in log)
        kt = sum(m[0] for m in sim.milled)
        kg = sum(m[1] for m in sim.milled)
        q1 = sim.milled[:3]
        q1g = (sum(m[1] for m in q1) / max(sum(m[0] for m in q1), 1e-9)
               if q1 else 0.0)
        dev_total = log[-1]["dev"] if log else 0.0
        rows[k] = dict(
            n_open=n_open,
            g_opened=float(np.mean(g_of)) if g_of else 0.0,
            losers=losers,
            missed=missed,
            cf_share=cf_n / max(n_open, 1),
            fans=fans,
            bleed=bleed,
            kt_per_stope=kt / max(n_open, 1),
            q1_grade=q1g,
            head=kg / max(kt, 1e-9),
            dev_m=dev_total,
            months=len(cash),
            opcash=float(np.sum(sim.cash[op0:]) / 1e6),
            dead=sim.dead is not None,
        )
    return rows


def main():
    machinery()
    keys = None
    acc = {a: {} for a in ("det", "sp", "pom", "ada")}
    for s in range(10):
        rows = run_world(s)
        for a, r in rows.items():
            for k, v in r.items():
                acc[a].setdefault(k, []).append(v)
        print(f"world {s} done", flush=True)
    print(f"\n{'metric':22s} {'det':>8s} {'sp':>8s} {'pom':>8s} {'ada':>8s}")
    for k in acc["det"]:
        vals = [np.mean(acc[a][k]) for a in ("det", "sp", "pom", "ada")]
        print(f"{k:22s} " + " ".join(f"{v:8.2f}" for v in vals))
    np.savez("out/forensics.npz",
             **{f"{a}_{k}": np.array(acc[a][k]) for a in acc for k in acc[a]})


if __name__ == "__main__":
    main()
