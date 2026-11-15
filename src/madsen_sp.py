"""
madsen_sp.py - third and fourth arms: stochastic mine planning and a
single-agent POMDP (belief-MDP) planner.

A faithful, charitable SIP-style stochastic planner: it receives the SAME
mixed-interpretation ensemble as the agents and uses the SAME risk-aware
machinery (CVaR-blended scoring, mill-space hurdle, per-stope method choice,
boundary design from ensemble P(vein)) -- but evaluates everything ONCE on
the prior (pierce points only) and executes open-loop: no assimilation from
faces or the mill, no definition fans, no stop-loss, no reconsideration.
It is granted the orderly-operations cost premium (it executes a coherent
fixed plan), making the comparison conservative for the closed-loop thesis.

Decomposition on identical worlds:
    SP - deterministic  = the value of planning on ensembles (the classical
                          stochastic-mine-planning contribution)
    agents - SP         = the value of the closed multi-agent loop
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from madsen_world import (make_truth, drill_pierce_points, make_ensemble,
                          machinery, GRAMS_PER_OZ)
from madsen_sim import run_deterministic, CREDIT_LIMIT
from madsen_agents import (run_agents, GeologyAgent, AgentSim, CostAgent,
                           AGENT_EXEC_PREMIUM, DEV_RUNRATE, LAM, PREPROD)
from madsen_world import interpretation_A, SX, SZL, CX, DX, NX
from madsen_sim import (FIXED_TRUE, DEBT_SERVICE, MILL_CAP, HEADING_CAP,
                        MAX_HEADINGS, N_LEVELS, OPEN)
from madsen_agents import MAX_OPEN, COST_BELIEF


def run_sp(world, pierce, ensemble, seed=0):
    """Risk-aware stochastic plan built ONCE on the prior ensemble, executed
    open-loop, mine closed when the plan completes. Charitable: orderly-ops
    premium, hurdle- and CVaR-aware scoring, method+boundary per stope."""
    TH_A, G_A = interpretation_A(pierce)
    env0 = TH_A > 0
    method0 = np.full((SX, SZL), "LHOS", dtype=object)
    sim = AgentSim(world, env0, method0, seed=seed,
                   exec_premium=AGENT_EXEC_PREMIUM)
    sim.op_start = PREPROD + 3
    geo = GeologyAgent(pierce, ensemble=ensemble)       # prior only
    hurdle = (FIXED_TRUE + DEBT_SERVICE + DEV_RUNRATE) / (MILL_CAP * 1000)
    cpt = {m: COST_BELIEF[m] + hurdle for m in COST_BELIEF}

    plan = []                                           # (score, s, m, post)
    for i in range(SX):
        for j in range(SZL):
            best = None
            for m in ("LHOS", "CF"):
                post = geo.stope_posterior(i, j, m, cpt[m])
                if post and (best is None or post["ev"] > best[1]["ev"]):
                    best = (m, post)
            if best is None:
                continue
            m, post = best
            score = (1 - LAM) * post["ev"] + LAM * post["cvar"]
            if score > 0 and post["ev"] > 0 and post["p_pos"] >= 0.55:
                plan.append((score, (i, j), m, post))
    plan.sort(key=lambda c: (-c[1][1], -c[0]))          # top-down, then score
    for _, s, m, post in plan:
        sim.redesign(*s, post["env"], m)
    q = [s for _, s, _, _ in plan]

    dev_order = list(range(N_LEVELS - 1, -1, -1))
    done = False
    while not done:
        max_h = MAX_HEADINGS if sim.t < PREPROD else 2
        develop = []
        need_lvls = {j for (i, j) in q} | {j + 1 for (i, j) in q}
        for lvl in dev_order:
            if sim.dev[lvl] < NX * DX - 1e-9 and                (sim.t < PREPROD or lvl in need_lvls):
                develop.append((lvl, HEADING_CAP))
            if len(develop) == max_h:
                break
        to_open = []
        if sim.t >= sim.op_start:
            n_open = int((sim.status == OPEN).sum())
            for s in list(q):
                if n_open + len(to_open) >= MAX_OPEN:
                    break
                if sim.can_open(*s):
                    to_open.append(s); q.remove(s)
        active = np.any((sim.status == OPEN) | (sim.status == 3))
        quit_now = (sim.t >= sim.op_start and not q and not to_open
                    and not active)
        _, _, done = sim.step({"develop": develop, "open": to_open,
                               "quit": quit_now})
    return sim

N = 10
SP_FLAGS = dict(assimilate=False, fans=False, stop_loss=False,
                hurdle=True, method_choice=True, redesign=True, orderly=True)
# single-agent POMDP: closed loop on the orebody, open loop on economics.
# Assimilates faces/fans/mill and redesigns boundaries and methods from the
# posterior, but has no cost agent (keeps FS cost beliefs) and no strategic
# portfolio agent (no mill-space hurdle, no stop-loss). Orderly premium.
POMDP_FLAGS = dict(assimilate=True, fans=True, redesign=True,
                   method_choice=True, hurdle=False, stop_loss=False,
                   cost_update=False, orderly=True)


def outcome(sim):
    cash = float(np.sum(sim.cash[sim.op_start:]) / 1e6)
    oz = sum(m[1] for m in sim.milled) * 1000 / GRAMS_PER_OZ
    kt = sum(m[0] for m in sim.milled)
    return dict(cash=cash, dead=sim.dead is not None, oz=oz,
                g=oz * GRAMS_PER_OZ / 1000 / max(kt, 1e-9))


def main():
    machinery()
    worlds = [make_truth(200 + s) for s in range(N)]
    print("Ensembles ...")
    pierces0 = [drill_pierce_points(w) for w in worlds]
    ens = [make_ensemble(p, n_real=60, frac_A=0.5, seed=s + 77)
           for s, p in enumerate(pierces0)]

    rows = []
    curves = []
    pierces = pierces0
    for s, w in enumerate(worlds):
        det, _ = run_deterministic(w, seed=s, bulk_sample=False,
                                   pierce=pierces[s])
        sp = run_sp(w, pierces[s], ens[s], seed=s)
        pom, _ = run_agents(w, seed=s, flags=POMDP_FLAGS, ensemble=ens[s],
                            pierce=pierces[s])
        ada, _ = run_agents(w, seed=s, ensemble=ens[s], pierce=pierces[s])
        o = dict(det=outcome(det), sp=outcome(sp), pom=outcome(pom),
                 ada=outcome(ada))
        rows.append(o)
        curves.append((np.cumsum(det.cash) / 1e6, np.cumsum(sp.cash) / 1e6,
                       np.cumsum(pom.cash) / 1e6, np.cumsum(ada.cash) / 1e6))
        print(f"world {s}: det {o['det']['cash']:6.1f} "
              f"({'D' if o['det']['dead'] else 'a'}) | "
              f"SP {o['sp']['cash']:6.1f} ({'D' if o['sp']['dead'] else 'a'}) "
              f"| POMDP {o['pom']['cash']:6.1f} "
              f"({'D' if o['pom']['dead'] else 'a'}) g={o['pom']['g']:.1f} "
              f"| agents {o['ada']['cash']:6.1f} "
              f"({'D' if o['ada']['dead'] else 'a'}) g={o['ada']['g']:.1f}")

    A = {arm: {k: np.array([r[arm][k] for r in rows], dtype=float)
               for k in rows[0]["det"]}
         for arm in ("det", "sp", "pom", "ada")}
    print("\n=========== FOUR ARMS, 10 HIDDEN MADSENS (zone C$M) ===========")
    print(f"{'':24s} {'determ.':>8s} {'stoch.':>8s} {'POMDP':>8s} "
          f"{'agents':>8s}")
    for lab, key, fmt in [("operating cash, mean", "cash", "{:>8.1f}"),
                          ("operating cash, worst", "cash", None),
                          ("insolvency rate", "dead", "{:>8.0%}"),
                          ("head grade g/t", "g", "{:>8.1f}"),
                          ("ounces", "oz", "{:>8.0f}")]:
        vals = []
        for arm in ("det", "sp", "pom", "ada"):
            v = A[arm][key]
            v = v.min() if lab.endswith("worst") else v.mean()
            f = fmt or "{:>8.1f}"
            vals.append(f.format(v))
        print(f"{lab:24s} " + " ".join(vals))
    print(f"\nensemble planning (SP - det):      "
          f"{(A['sp']['cash']-A['det']['cash']).mean():+.1f} C$M/world")
    print(f"orebody loop only (POMDP - det):   "
          f"{(A['pom']['cash']-A['det']['cash']).mean():+.1f} C$M/world")
    print(f"economic agents (agents - POMDP):  "
          f"{(A['ada']['cash']-A['pom']['cash']).mean():+.1f} C$M/world")
    print(f"closed loop over SP (agents - SP): "
          f"{(A['ada']['cash']-A['sp']['cash']).mean():+.1f} C$M/world")

    # figure: three-arm trajectories + bars
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    ax = axes[0]
    for (cd, cs, cp, ca) in curves:
        ax.plot(cd, color="#8d99ae", alpha=0.5, lw=1.0)
        ax.plot(cs, color="#e9a13b", alpha=0.55, lw=1.0)
        ax.plot(cp, color="#7d5ba6", alpha=0.55, lw=1.0)
        ax.plot(ca, color="#d1495b", alpha=0.6, lw=1.0)
    ax.axhline(CREDIT_LIMIT / 1e6, color="k", ls="--", lw=1)
    for c, l in (("#8d99ae", "Deterministic"),
                 ("#e9a13b", "Stochastic plan (open-loop)"),
                 ("#7d5ba6", "Single-agent POMDP"),
                 ("#d1495b", "Multi-agent (closed-loop)")):
        ax.plot([], [], color=c, lw=2, label=l)
    ax.set_xlabel("Month"); ax.set_ylabel("Cumulative cash (C$M, zone)")
    ax.legend(fontsize=8)
    ax.set_title("Four planning paradigms on ten hidden Madsens", fontsize=10)

    ax = axes[1]
    x = np.arange(N)
    ax.bar(x - 0.3, A["det"]["cash"], 0.2, color="#8d99ae", label="Det.")
    ax.bar(x - 0.1, A["sp"]["cash"], 0.2, color="#e9a13b", label="Stoch.")
    ax.bar(x + 0.1, A["pom"]["cash"], 0.2, color="#7d5ba6", label="POMDP")
    ax.bar(x + 0.3, A["ada"]["cash"], 0.2, color="#d1495b", label="Agents")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("World"); ax.set_ylabel("Operating cash (C$M)")
    ax.set_title("Final operating cash by world", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("Four planning paradigms on ten hidden Madsens",
                 fontsize=12)
    fig.savefig("out/fig18_three_arms.png", dpi=160)
    np.savez_compressed("out/four_arm_results.npz",
                        det=A["det"]["cash"], sp=A["sp"]["cash"],
                        pom=A["pom"]["cash"], ada=A["ada"]["cash"],
                        det_dead=A["det"]["dead"], sp_dead=A["sp"]["dead"],
                        pom_dead=A["pom"]["dead"], ada_dead=A["ada"]["dead"])
    print("Wrote fig18_three_arms.png, four_arm_results.npz")


if __name__ == "__main__":
    main()
