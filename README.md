# Multi-Agent Mine Planning: a counterfactual study of the Madsen failure

This repository contains the code, data, and reproduction tests for a
study of multi-agent decision-making in underground mine planning,
calibrated to a single documented failure: the Madsen gold mine (Red
Lake, Ontario), planned on one geological model that was never updated
as production disagreed with it.

The study asks a counterfactual question. Given the same hidden
orebodies and the same monthly observations, how would different
decision processes have performed? Five approaches are compared, from
a conventional deterministic plan to a four-agent system in which
geology, cost, planning, and management each hold one part of a shared,
stoppable loop.

**Headline result.** Most of the measured value comes from the loop
itself — evidence re-conditioning a belief that feeds staged,
stoppable commitments. The multi-agent decomposition adds a smaller,
bounded increment on top of that, plus auditability and a clean map
onto how a mine is actually staffed. Two registered predictions failed
and are reported as failures.

Everything here runs on a laptop. Every number in the paper and book
regenerates from fixed random seeds.

## Layout

```
src/       simulator, the five decision approaches, the forensic audit,
           and the stress-test / reconciliation harness
data/      the calibration workbook (every model input mapped to its
           public source) and the calibration notes
results/   pre-computed .npz archives; every figure and table
           regenerates from these (also reachable as ./out)
tests/     pytest reproduction tests (fast subset of the gate)
docs/      stress-test report and Phase-2 results notes
paper/     the manuscript source
```

## Source modules (`src/`)

- `madsen_world.py` — world generator; the two interpretation families
  (wide / narrow), conditional realizations, the machinery() constants.
- `madsen_sim.py` — the underground simulator: development, stope
  extraction, dilution, milling, cash to the credit limit.
- `madsen_agents.py` — the belief, cost, planning, and strategic
  agents. Design constants are named and documented at the top of
  `GeologyAgent` (tempered ESS-floored localized Bayes; CVaR-blended
  score; the 0.55 commitment stage; the full-cost hurdle).
- `madsen_sp.py` — the stochastic plan-once approach.
- `madsen_rolling.py` — the rolling-horizon approach.
- `madsen_ablations.py` — mechanism-removal experiments.
- `forensics.py` — scores committed stopes against the hidden truth
  and reduces the logs to the named mining behaviours of the audit.
- `stress_tests.py` — the reconciliation harness: golden checks,
  invariants, degenerate worlds, bootstraps, ablations, adversarial
  controls, and `paperdiff`.

## Quick start

```bash
pip install -r requirements.txt
pytest -q                      # fast reproduction tests (~20 s)
```

To reproduce a headline comparison directly:

```python
import sys; sys.path.insert(0, "src")
from madsen_world import make_truth, drill_pierce_points, make_ensemble, machinery
from madsen_sim import run_deterministic
from madsen_agents import run_agents
machinery()
w = make_truth(205); p = drill_pierce_points(w)
ens = make_ensemble(p, n_real=60, frac_A=0.5, seed=205 + 77)
det, _ = run_deterministic(w, seed=5, bulk_sample=False, pierce=p)
ada, _ = run_agents(w, seed=5, ensemble=ens, pierce=p, n_real=40)
```

## Full reproduction

The complete suite — bootstraps, ablations, adversarial controls, and
the paper-claim reconciliation — runs from:

```bash
cd src && TEST=paperdiff python3 stress_tests.py
```

`paperdiff` regenerates the claim table and blocks if any regenerated
number disagrees with the written one. The full run takes a few hours;
the `results/` archives let the figures and tables rebuild in seconds.

## Data and provenance

The study uses no proprietary dataset. Its inputs are roughly two
dozen numbers read from public documents (the 2019 feasibility study,
the quarterly filings, the post-failure technical reports), each mapped
to its source in `data/madsen_calibration_data.xlsx`. The workbook's
Machinery sheet marks which values are design choices, frozen before
the held-out worlds were drawn, versus values read from the filings.

## License

Code and data in this repository are released under the MIT License
(see `LICENSE`). The cited source documents remain the property of
their publishers and are not redistributed here.

## Note on `out/`

The code reads and writes result archives under `out/`. Running
`pytest` creates that path automatically (it links to `results/`). To
run scripts directly, first: `ln -s results out` from the repo root.
