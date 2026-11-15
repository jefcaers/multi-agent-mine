"""
madsen_ablations.py - the price of each idea.

Full multi-agent system minus one mechanism at a time, paired by world
against the cached deterministic baseline. Marginal value of mechanism X =
mean over worlds of [swing(full) - swing(without X)].

The 'orderly' ablation is reported separately: it is not a mechanism but the
sensitivity of the execution-premium modeling claim (agents pay the full
+C$55/t chaos premium instead of +C$15/t).
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from madsen_world import make_truth, machinery
from madsen_sim import run_deterministic
from madsen_agents import run_agents
from madsen_world import drill_pierce_points, make_ensemble

N = 10
VARIANTS = [
    ("full", {}),
    ("no belief updating", {"assimilate": False}),
    ("no definition fans", {"fans": False}),
    ("no hurdle rate", {"hurdle": False}),
    ("no stop-loss", {"stop_loss": False}),
    ("no method choice", {"method_choice": False}),
    ("no boundary redesign", {"redesign": False}),
    ("chaos-cost sensitivity", {"orderly": False}),
]


def outcome(sim):
    cash = float(np.sum(sim.cash[sim.op_start:]) / 1e6)
    oz = sum(m[1] for m in sim.milled) * 1000 / 31.1035
    kt = sum(m[0] for m in sim.milled)
    return dict(cash=cash, dead=sim.dead is not None, oz=oz,
                g=oz * 31.1035 / 1000 / max(kt, 1e-9))


def main():
    machinery()
    worlds = [make_truth(200 + s) for s in range(N)]
    print("Shared pierce draws + ensembles ...")
    pierces, ensembles = [], []
    for s, w in enumerate(worlds):
        p = drill_pierce_points(w)
        pierces.append(p)
        ensembles.append(make_ensemble(p, n_real=60, frac_A=0.5, seed=s + 77))
    det = []
    print("Deterministic baseline ...")
    for s, w in enumerate(worlds):
        d, _ = run_deterministic(w, seed=s, bulk_sample=False,
                                 pierce=pierces[s])
        det.append(outcome(d))
        print(f"  world {s}: {det[-1]['cash']:6.1f}M "
              f"{'DEAD' if det[-1]['dead'] else 'alive'}")
    v0 = int(os.environ.get("V0", 0)); v1 = int(os.environ.get("V1", len(VARIANTS)))
    results = {}
    for name, flags in VARIANTS[v0:v1]:
        print(f"\nVariant: {name}", flush=True)
        rows = []
        for s, w in enumerate(worlds):
            t0 = time.time()
            sim, info = run_agents(w, seed=s, flags=flags,
                                   ensemble=ensembles[s],
                                   pierce=pierces[s])
            o = outcome(sim)
            o["swing"] = o["cash"] - det[s]["cash"]
            rows.append(o)
            print(f"  world {s}: {o['cash']:6.1f}M "
                  f"{'DEAD' if o['dead'] else 'alive':5s} "
                  f"g={o['g']:.1f} swing {o['swing']:+.1f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        results[name] = rows
        np.savez_compressed(f"out/abl_{v0}.npz",
                            **{n.replace(' ', '_'): np.array(
                                [[r['cash'], r['dead'], r['g'], r['swing']]
                                 for r in rws]) for n, rws in results.items()})

    # ---------------- summary ----------------
    print("\n" + "=" * 72)
    print(f"{'variant':26s} {'cash C$M':>9s} {'insolv.':>8s} {'g/t':>6s} "
          f"{'swing/wld':>10s} {'mech. value':>12s}")
    full_sw = np.array([r["swing"] for r in results["full"]])
    table = []
    for name, _ in VARIANTS:
        rows = results[name]
        cash = np.mean([r["cash"] for r in rows])
        dead = np.mean([r["dead"] for r in rows])
        g = np.mean([r["g"] for r in rows])
        sw = np.array([r["swing"] for r in rows])
        mech = (full_sw - sw).mean() if name != "full" else 0.0
        table.append((name, cash, dead, g, sw.mean(), mech))
        print(f"{name:26s} {cash:>9.1f} {dead:>8.0%} {g:>6.1f} "
              f"{sw.mean():>+10.1f} "
              f"{('%+12.1f' % mech) if name != 'full' else '        base'}")

    # ---------------- fig17: waterfall of mechanism values ----------------
    mech_names = [t[0] for t in table[1:-1]]           # exclude full + chaos
    mech_vals = [t[5] for t in table[1:-1]]
    order = np.argsort(mech_vals)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(mech_names))
    ax.barh(y, [mech_vals[i] for i in order], color="#d1495b")
    ax.set_yticks(y)
    ax.set_yticklabels([mech_names[i] for i in order], fontsize=9)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel("Marginal value of mechanism (C$M/world, zone share)")
    ax.set_title("The price of each idea: value lost when one mechanism\n"
                 "is removed from the full agent system", fontsize=10)
    for yi, v in zip(y, [mech_vals[i] for i in order]):
        ax.text(v, yi, f" {v:+.1f}", va="center", fontsize=8)

    ax = axes[1]
    names = [t[0] for t in table]
    dead = [t[2] for t in table]
    colors = ["#2a9d8f" if n == "full" else
              ("#e9a13b" if "chaos" in n else "#8d99ae") for n in names]
    ax.barh(np.arange(len(names)), dead, color=colors)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Insolvency rate over 10 worlds")
    ax.axvline(0.8, color="k", ls="--", lw=1, label="Deterministic arm (80%)")
    ax.legend(fontsize=8)
    ax.set_title("Survival under each ablation", fontsize=10)
    fig.suptitle("Madsen ablations: attribution by removal", fontsize=12)
    fig.savefig("out/fig17_ablations.png", dpi=160)
    np.savez_compressed("out/ablation_results.npz",
                        **{n.replace(" ", "_"): np.array(
                            [[r["cash"], r["dead"], r["g"], r["swing"]]
                             for r in rows])
                           for n, rows in results.items()})
    print("\nWrote fig17_ablations.png, ablation_results.npz")


if __name__ == "__main__":
    main()
