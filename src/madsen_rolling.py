"""
madsen_rolling.py - M3: the rolling-horizon stochastic planner.

The strongest fair competitor not yet tested: a stochastic planner that
re-optimizes on the UPDATED ensemble every quarter. It assimilates faces
and mill reconciliation monthly (the re-conditioning channel), rebuilds
its risk-aware plan every 3 months with the same CVaR blend, mill-space
hurdle, per-stope method choice, and boundary redesign as the plan-once
SP, and executes open-loop between epochs. It purchases no definition
fans, holds feasibility-study cost beliefs (no cost agent), has no
stop-loss, and closes when a freshly rebuilt plan is empty and nothing
is active (plan completion). Orderly execution premium.

REGISTERED PREDICTION (before first run): rolling-horizon lands between
the plan-once SP and the single-agent POMDP, and the multi-agent edge
over it decomposes into information purchase (fans), online cost
learning, and the stop-loss.

Decomposition on identical worlds:
    rolling - SP      = the value of quarterly re-conditioning alone
    agents  - rolling = the value of fans + live costs + stop-loss
"""

import os
import numpy as np

from madsen_world import (make_truth, drill_pierce_points, make_ensemble,
                          machinery, interpretation_A, SX, SZL, DX, NX)
from madsen_sim import (FIXED_TRUE, DEBT_SERVICE, MILL_CAP, HEADING_CAP,
                        MAX_HEADINGS, N_LEVELS, OPEN)
from madsen_agents import (GeologyAgent, AgentSim, AGENT_EXEC_PREMIUM,
                           DEV_RUNRATE, LAM, PREPROD, MAX_OPEN, COST_BELIEF)


def run_sp_rolling(world, pierce, ensemble, seed=0, replan_every=3):
    TH_A, G_A = interpretation_A(pierce)
    env0 = TH_A > 0
    method0 = np.full((SX, SZL), "LHOS", dtype=object)
    sim = AgentSim(world, env0, method0, seed=seed,
                   exec_premium=AGENT_EXEC_PREMIUM)
    sim.op_start = PREPROD + 3
    geo = GeologyAgent(pierce, ensemble=ensemble)
    hurdle = (FIXED_TRUE + DEBT_SERVICE + DEV_RUNRATE) / (MILL_CAP * 1000)
    cpt = {m: COST_BELIEF[m] + hurdle for m in COST_BELIEF}   # frozen FS

    def build_plan():
        plan = []
        for i in range(SX):
            for j in range(SZL):
                if sim.status[i, j] >= OPEN:          # committed already
                    continue
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
        plan.sort(key=lambda c: (-c[1][1], -c[0]))    # top-down, then score
        return [(s, m, post) for _, s, m, post in plan]

    q = build_plan()
    replans = 1
    dev_order = list(range(N_LEVELS - 1, -1, -1))
    done = False
    while not done:
        if sim.t > 0 and sim.t % replan_every == 0:
            q = build_plan()
            replans += 1
        max_h = MAX_HEADINGS if sim.t < PREPROD else 2
        develop = []
        need_lvls = {j for ((i, j), _, _) in q} | \
                    {j + 1 for ((i, j), _, _) in q}
        for lvl in dev_order:
            if sim.dev[lvl] < NX * DX - 1e-9 and \
               (sim.t < PREPROD or lvl in need_lvls):
                develop.append((lvl, HEADING_CAP))
            if len(develop) == max_h:
                break
        to_open = []
        if sim.t >= sim.op_start:
            n_open = int((sim.status == OPEN).sum())
            for item in list(q):
                s, m, post = item
                if n_open + len(to_open) >= MAX_OPEN:
                    break
                if sim.can_open(*s):
                    sim.redesign(*s, post["env"], m)  # current posterior
                    to_open.append(s)
                    q.remove(item)
        active = np.any((sim.status == OPEN) | (sim.status == 3))
        quit_now = (sim.t >= sim.op_start and not q and not to_open
                    and not active)
        if quit_now:
            q = build_plan()                          # final reconsideration
            quit_now = not q
        obs, cash, done = sim.step({"develop": develop, "open": to_open,
                                    "quit": quit_now})
        geo.assimilate(obs)                           # monthly, free channels
    return sim, replans


def outcome(sim):
    from madsen_world import GRAMS_PER_OZ
    cash = float(np.sum(sim.cash[sim.op_start:]) / 1e6)
    oz = sum(m[1] for m in sim.milled) * 1000 / GRAMS_PER_OZ
    kt = sum(m[0] for m in sim.milled)
    n_com = int((sim.status >= OPEN).sum())
    return dict(cash=cash, dead=float(sim.dead is not None), oz=oz,
                g=oz * 31.1035 / 1000 / max(kt, 1e-9), n=n_com)


if __name__ == "__main__":
    machinery()
    mode = os.environ.get("MODE", "canon")
    if mode == "canon":
        res = {k: [] for k in ("cash", "dead", "oz", "g", "n")}
        for s in range(10):
            w = make_truth(200 + s)
            p = drill_pierce_points(w)
            ens = make_ensemble(p, n_real=60, frac_A=0.5, seed=s + 77)
            sim, replans = run_sp_rolling(w, p, ens, seed=s)
            o = outcome(sim)
            for k in res:
                res[k].append(o[k])
            print(f"world {s}: roll {o['cash']:6.1f} "
                  f"({'D' if o['dead'] else 'a'}) g={o['g']:.1f} "
                  f"n={o['n']} replans={replans}", flush=True)
        np.savez("out/rolling_canon.npz",
                 **{k: np.array(v) for k, v in res.items()})
        print("canon mean cash %.2f, insolvency %.0f%%"
              % (np.mean(res['cash']), 100 * np.mean(res['dead'])))
    else:
        W0, W1 = int(os.environ["W0"]), int(os.environ["W1"])
        res = {k: [] for k in ("cash", "dead", "g", "n")}
        for s in range(W0, W1):
            w = make_truth(300 + s)
            p = drill_pierce_points(w)
            ens = make_ensemble(p, n_real=40, frac_A=0.5, seed=s + 77)
            sim, _ = run_sp_rolling(w, p, ens, seed=s)
            o = outcome(sim)
            for k in res:
                res[k].append(o[k])
            if s % 5 == 0:
                print(f"  w{s}: roll {o['cash']:6.1f}", flush=True)
        np.savez(f"out/roll_boot_{W0}.npz",
                 **{k: np.array(v) for k, v in res.items()})
        print(f"chunk {W0}-{W1-1} done")
