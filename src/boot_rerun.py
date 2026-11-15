import os
import numpy as np
from madsen_world import make_truth, drill_pierce_points, make_ensemble, machinery
from madsen_sim import run_deterministic
from madsen_agents import run_agents, GeologyAgent

machinery()
W0, W1 = int(os.environ["W0"]), int(os.environ["W1"])
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

ada, pla, dada, dpla = [], [], [], []
for s in range(W0, W1):
    w = make_truth(300 + s)
    p = drill_pierce_points(w)
    ens = make_ensemble(p, n_real=40, frac_A=0.5, seed=s + 77)
    a, _ = run_agents(w, seed=s, ensemble=ens, pierce=p, n_real=40)
    ada.append(np.sum(a.cash[a.op_start:]) / 1e6); dada.append(a.dead is not None)
    GeologyAgent.assimilate = scrambled
    try:
        pl, _ = run_agents(w, seed=s, ensemble=ens, pierce=p, n_real=40)
    finally:
        GeologyAgent.assimilate = orig
    pla.append(np.sum(pl.cash[pl.op_start:]) / 1e6); dpla.append(pl.dead is not None)
    if s % 10 == 0:
        print(f"  w{s}: ada {ada[-1]:6.1f} placebo {pla[-1]:6.1f}", flush=True)
np.savez(f"out/boot2_{W0}.npz", ada=np.array(ada), placebo=np.array(pla),
         dead_ada=np.array(dada, float), dead_placebo=np.array(dpla, float))
print(f"chunk {W0}-{W1-1} done")
