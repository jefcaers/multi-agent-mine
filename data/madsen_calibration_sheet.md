# Madsen Calibration Sheet — Multi-Agent Mine Demonstration

**Purpose.** Calibrate the synthetic-world generator (`world.py`) and mine simulator
(`mine_sim.py`) to the Pure Gold Madsen case (Red Lake, Ontario), so that:
1. the DETERMINISTIC planner, given only 2019-feasibility-level information,
   independently reproduces the documented failure trajectory (validation of the
   control arm, untuned), and
2. the multi-agent system runs as the counterfactual on the same hidden truths.

**Document trail (all public):**
- PG-FS-2019 — Pure Gold Feasibility Study, JDS Energy & Mining, dated 2019-02-05
  (SEDAR). Resource by Ginto Consulting (data cut-off 2018-01-16).
- PG-Q — Pure Gold quarterly/annual filings 2020–2022 (SEDAR): realized grades,
  tonnes, costs.
- SRK-2023 — "Independent NI 43-101 Technical Report and Updated MRE for the
  PureGold Mine" SRK Consulting, eff. 2021-12-31, dated 2023-06-16 (amended
  2024-04-24). The post-failure re-model.
- WRLG-PFS-2025 — West Red Lake Gold PFS, dated 2025-02-18, plus WRLG 2025
  operating updates (dense re-drilling; the truth-adjacent model).
- CRUX-2023 — Crux Investor retrospective (quantified failure analysis).

---

## 1. The planning world (what the deterministic planner is given)

These parameterize the SPARSE-INFORMATION model — the equivalent of our kriged
model + drill data. Source: PG-FS-2019 unless noted.

| Report parameter | Value (source) | Simulator variable | Calibrated value |
|---|---|---|---|
| Commodity / units | Gold, g/t | grade field units | switch %Cu → g/t Au (lognormal) |
| Probable reserve | 3.5 Mt @ 9.0 g/t, 1.0 Moz | panel-scale check | synthetic panel reserve grade ≈ 9 g/t at FS cutoffs |
| Resource (Ginto) | Ind. 7.2 Mt @ 8.9 g/t (2.1 Moz); Inf. 1.9 Mt @ 7.7 g/t | prior mean/target | MU_LOG, SIG_LOG set so diluted stope grades ≈ 8–10 g/t above cutoff |
| Cut-off grades | 4.0 g/t (McVeigh); 4.75 g/t (Austin/S.Austin/8 Zone) | BREAKEVEN | 4.75 g/t primary; zone-dependent later |
| Gold price / FX | US$1,275/oz; 0.75 C$/US$ | PRICE | US$1,275/oz → C$52.7/g (×0.9 recovery basis below) |
| Recovery | ~96–97% (historic mill, "well known") | RECOVERY | 0.96 |
| Mining methods | Longhole open stoping + cut-and-fill | sim methods | keep LHOS; add C&F mode (dilution/cost differ) |
| Min / design stope width | 2 m design width | stope geometry | narrow stopes: 2–3 m width, 15–20 m strike, 20–25 m height |
| Planned dilution | 13% total (JDS) | planner's assumed dilution | 13% at planning; TRUE dilution model separate (§3) |
| Mining opex | C$223/t (JDS, all-in unit) | MINE_COST+PROC_COST | planner believes C$223/t |
| AISC / cash cost | US$787/oz AISC; US$607/oz cash | economics sanity check | derived, not input |
| Throughput | 800 tpd (~24 kt/month) | HOIST_CAP | 24 kt/period (monthly) |
| Initial capex | C$95M (FS) | project capex | C$95M planner belief |
| LOM / NPV / IRR | 12.2 yr; pre-tax NPV5 C$353M, IRR 43% | planner's promise | the "guidance" the deterministic arm publishes |
| Wireframes in model | 14 domains; mineable structures drawn up to 30 m wide | interpretation A | SMOOTH interpretation: wide, connected ore envelopes |
| Bulk-sample reconciliation | resource-test stopes: tonnes +46%, grade +8% (+56% oz); reserve-test: +36% t, +12% g (+52% oz) | validation trap | small-sample positive reconciliation must be reproducible in favorable-zone bulk tests — while the full mine still fails. Include a "bulk sample validation" event in the deterministic arm. |

## 2. The truth-adjacent world (what reality turned out to be)

These parameterize the HIDDEN TRUTH generator. Source: SRK-2023, WRLG-PFS-2025,
CRUX-2023.

| Report parameter | Value (source) | Simulator implication |
|---|---|---|
| Re-modeled resource | Ind. 1.65 Moz @ 7.4 g/t; Inf. 0.37 Moz @ 6.3 g/t (cutoff 3.38 g/t, US$1,800/oz) [SRK-2023] | truth contains materially fewer ounces at lower grade than interpretation A above equivalent cutoffs |
| Wireframes | 14 → 50; structures "generally 2 m or less, irregular outlines" [CRUX] | TRUE ore = narrow, discontinuous, sharply bounded structures; interpretation A over-connects them |
| Realistic dilution | 20–25% for these widths/methods [CRUX] | geometric dilution model must yield ~20–25% when mining 2 m structures with the planned methods |
| Realistic opex | C$305/t (Macassa benchmark, comparable narrow C&F) [CRUX] | TRUE unit cost ~C$305/t vs believed C$223/t (+37% cost error) |
| Honest cutoff | ~7.45 g/t at real dilution+costs [CRUX] | large fraction of planned stopes truly uneconomic — the decision field for the agents |
| WRLG mine plan | 800 tpd, 7.25 yr; C&F 59% / LHOS 41%; avg stope width 3.2–7.2 m; hydraulic backfill [WRLG-PFS-2025] | post-truth method mix: C&F share rises when narrowness is acknowledged |
| WRLG operating | mill ~650 tpd @ 95% rec.; head grades 3→6.5 g/t through ramp [WRLG 2025] | ramp-up grade trajectory reference |
| WRLG definition drilling | 38,393 m underground in ~6 months of 2025 | density of "truth-revealing" drilling; VOI agent budget scale |

## 3. Failure trajectory — VALIDATION TARGETS (must be reproduced untuned)

The deterministic arm, planned on interpretation A with FS costs/dilution and
executed open-loop against truths drawn from §2 statistics, must produce:

| # | Documented outcome [source] | Simulation acceptance band |
|---|---|---|
| V1 | First mill feed 13% below reserve grade (7.80 vs 8.97 g/t); ROM stockpile 3.32 g/t [CRUX/PG-Q] | month-1..2 head grade 10–20% below reserve grade; low-grade stockpile emerges |
| V2 | Q1-2021 production 4,000 oz vs 13,700 planned (−70%) [Globe & Mail] | first-quarter ounces 55–80% below plan across world ensemble median |
| V3 | Named causes: excess overbreak/dilution; barren dykes inside stopes [CRUX] | attribution log shows unplanned dilution + internal-waste incidents as dominant loss buckets |
| V4 | Realized dilution ~20–25% vs 13% planned [CRUX] | simulated realized dilution in 18–28% band |
| V5 | Unit costs → C$305/t class vs C$223/t plan [CRUX] | realized C$/t 30–40% above plan |
| V6 | Death spiral: commercial production declared m~8, insolvent by m~26; capex C$95→145.6M (+53%) [CRUX] | cumulative cash never turns sustainably positive; negative FCF through month 24 in majority of worlds |
| V7 | Post-failure re-model cuts ounces sharply (2.1→1.65 Moz Ind., grade 8.9→7.4) [SRK-2023] | re-estimating on dense data inside sim reproduces 20–30% ounce write-down vs interpretation A |
| V8 | Small bulk samples reconciled POSITIVELY pre-failure [PG-FS-2019 §14.1.14] | favorable-zone bulk-sample test in sim can pass while LOM fails — demonstrates why that validation was misleading |

Counterfactual then runs agents on identical truths with identical starting
information. Attribution buckets (engineer-native): dilution avoided via
boundary redesign; barren-dyke stopes deferred after face/fan evidence; cutoff
re-estimated online as true costs reveal (the agents should discover ~7.4 g/t);
sequencing to protect the guidance band; definition-drilling VOI net of cost.

## 4. Geostatistical parameters — extracted vs. to-extract

Known now:
- Structures: sub-vertical, narrow (≤2–3 m true width), strike-continuous
  segments 10s of m; hosted in altered basalt/ultramafic contacts [SRK/WRLG].
- Grade distribution: strongly positive-skewed; reserve grade 9 g/t over cutoff
  4.75 → distribution mean(above-cutoff) ≈ 2× cutoff; capping applied in both
  MREs (values TBD from reports).
- Two-interpretation structure is ESSENTIAL: same drillholes support a
  14-wireframe smooth model AND a 50-wireframe narrow model. This maps directly
  onto our geological-interpretation-uncertain ensemble (condition_ensemble_geo):
  interpretation A = wide/connected trend family; interpretation B = narrow/
  discontinuous family; truth drawn from B-like statistics.

To extract from full PDFs (flagged; not in news/secondary sources):
- [ ] Variogram models per domain: nugget, structures, ranges (PG-FS-2019 §14;
      SRK-2023 §14). Literature prior for Red Lake-style orogenic gold pending
      extraction: relative nugget 40–60%, ranges ~10–30 m along structure,
      3–10 m across (Dominy et al. on nuggety lode gold) — placeholders only.
- [ ] Capping/top-cut values per domain and their effect on mean grade.
- [ ] Drillhole spacing statistics by zone at FS date (surface-only) and block
      model size (SMU) in both models.
- [ ] Bulk-sample locations/tonnages/grades (PG-FS-2019 pp. 213–218) to
      replicate V8 exactly.
- [ ] SRK stope-by-stope or zone-by-zone depletion vs model, if disclosed
      (CRUX notes the reconciliation section is conspicuously absent — its
      absence is itself citable).
- [ ] WRLG-PFS-2025 dilution assumptions, mining recovery, and unit costs by
      method (C&F vs LHOS) for the truth-side economics.

## 5. Simulator upgrades this calibration forces

1. Units/economics: Au g/t, C$/t costs, US$/oz price with FX; stope value =
   f(width, grade, method).
2. Geometry: narrow sub-vertical structures inside a barren host — replace the
   "blob" trend with vein/shoot objects (object-based or high-threshold
   truncated-Gaussian), so ore/waste contacts are sharp and connectivity is the
   uncertain thing.
3. Geometric dilution model: mined grade = (vein t×g + overbreak waste t×~0)/
   total t, with overbreak a function of method (LHOS > C&F), stope width, and
   boundary-knowledge error; planned vs unplanned dilution separately tracked.
4. Internal waste objects (barren dykes) that surface drilling can miss.
5. Cost-truth ≠ cost-belief: true C$/t drawn ~N(305, σ) while planner believes
   223; agents must estimate cost online from realized period costs (a second,
   non-geological belief to update — directly implicates the honest-cutoff
   discovery ~7.45 g/t).
6. Two mining methods with different cost/dilution/rate (C&F vs LHOS) and the
   method choice per stope as an agent decision.
7. Bulk-sample event support (mine 2 test stopes, reconcile, report) for V8.

## 6. Deliberate scope exclusions (state in any write-up)

Not modeled, per the case record: permitting delays (secondary), plant/
metallurgy (explicitly unproblematic at Madsen — recovery held ~95–96%),
construction capex overrun mechanics (we take capex as sunk at t0; the
demonstration concerns OPERATING value destruction and its avoidance).
Financing dynamics (dilutive raises) excluded, though the guidance-band
machinery is the direct counterfactual to the credibility collapse that made
financing impossible.


---

## 7. CALIBRATION STATUS (achieved by `madsen_world.py`, 8-world ensemble)

| Metric | Target (source) | Achieved |
|---|---|---|
| Interpretation-A reserve grade @ 4.75 cutoff, 13% dil | ~9.0 g/t [PG-FS-2019] | 9.34 g/t |
| Q1 head-grade shortfall, first stopes | 45-65% (4,000 vs 13,700 oz, tonnage-adjusted) [Globe & Mail / PG-Q] | 64% |
| LOM head-grade shortfall (context) | large [CRUX] | 39% |
| Overbreak dilution on real structures (V4) | 18-28% [CRUX 20-25%] | 24% |
| Internal waste (gaps/dykes) share of feed (V3) | material [CRUX: dykes in stopes] | 36% |
| Honest diluted cutoff at plan price | ~5.8 g/t [derived] | 5.43 g/t |
| Honest undiluted cutoff | ~7.45 g/t [CRUX] | 7.30 g/t |
| Economic ounce write-down A -> honest truth (V7) | 30-60% [SRK -21% Ind. oz; CRUX economics imply far larger] | 52% |

Metric note: the documented -13% applies to the literal FIRST ORE (Dec-2020
mill feed, best development muck); the first-QUARTER grade ran ~-50 to -60%
(4,000 oz vs 13,700 planned incl. tonnage). The simulator's 'first stopes'
metric corresponds to the quarter, hence the 45-65% band.

World-generator structure locked by this calibration:
- decoupled PRESENCE field (42% coverage, 75x13 m plunge-anisotropic) and
  THICKNESS-where-present (base 0.9 m + lognormal, mean ~2.4 m, cap 6 m);
- in-vein grade lognormal (median 6.5 g/t, sigma_log 0.75, 50% rel. nugget);
- 2-3 plunging shoots and 2-3 barren dyke corridors per world;
- pierce-point drilling on ~32x28 m pattern, 15% thickness error, 35%
  lognormal grade error; grades top-cut at 28 g/t in the planning model;
- interpretation A: 130x45 m ranges, envelope threshold 1.15 m kriged
  thickness, floored to 2.0 m minimum width (the FS-style model);
- geometric dilution: ELOS 0.28 m/wall LHOS, 0.13 m/wall C&F; min width 2 m.


## 8. FAILURE-TRAJECTORY VALIDATION (achieved by `madsen_sim.py`, deterministic arm, 10 worlds)

| Target | Band (source) | Simulated |
|---|---|---|
| V2 Q1 head-grade shortfall vs plan | 45-65% [4,000 vs 13,700 oz] | 42% |
| V5 all-in operating cost overrun | 40-70% [C$223->~C$305/t direct + fixed-cost spreading] | 41% |
| V6 insolvency frequency | majority [PG: certain] | 100% |
| V6 median insolvency month | 18-30 [first ore Dec-20 -> CCAA Oct-22 = 22 mo] | 18 |
| V8 bulk sample reconciles positive | frequent [PG: +36-52% oz] | 40% of worlds (+28-46% when positive) |
| V8 joint: positive bulk AND mine fails | the trap | 40% of worlds |

Accounting conventions locked: 9 months pre-production + 3 months
commissioning capitalized (insolvency clock runs from commercial
production, as at PG); funding envelope C$-14M zone-share = credit +
dilutive emergency raises; development throttled to 2 headings during
operations (as PG did to conserve cash -- the spiral's positive feedback:
saving development cash starves future ore availability).

Illustrative world 203: promised 34.0 koz (zone share), delivered 11% of
plan, op head grades 2.3-3.5 g/t vs 9 g/t reserve, insolvent month 18.

STATUS: control arm fully validated. The simulator now reproduces the
documented Madsen pathology from first principles (geometry + interpretation
gap + cost error), untuned on the trajectory itself. Next: the multi-agent
counterfactual on identical hidden truths with identical 2019-level starting
information.


## 9. FOUR-ARM COUNTERFACTUAL (canonical; `madsen_sp.py`, 10 worlds, zone C$M)

All arms share one pierce-noise draw per world; agents' ensemble is
conditioned on exactly the pierce data all arms observe. Stochastic-planning
arm = same risk-aware machinery (mixed ensemble, CVaR, hurdle, method +
boundary choice) evaluated once on the prior, executed open-loop, closed on
plan completion, granted the orderly-ops premium.

Single-agent POMDP arm added: belief-MDP planner, closed loop on the
orebody (assimilation, fans, redesign, method choice) but a single
planner's economics: FS cost beliefs, no hurdle, no stop-loss. Orderly
premium granted.

POST-STRESS-SUITE CANONICALS (stop-loss commitment-block inversion
fixed in madsen_agents.py; all numbers regenerated):

| Metric | Deterministic | Stochastic plan | Single POMDP | Multi-agent |
|---|---|---|---|---|
| Operating cash, mean | -13.1 | -10.9 | -11.0 | -7.6 |
| Operating cash, worst | -15.5 | -15.4 | -14.5 | -15.3 |
| Insolvency rate | 80% | 50% | 60% | 30% |
| Realized head grade | 5.4 g/t | 5.7 g/t | 5.3 g/t | 6.8 g/t |
| Ounces delivered | 16,573 | 12,844 | 23,091 | 14,855 |
| Best world | -6.4 | +4.1 | +4.4 | +10.3 |

Decomposition: ensembles +2.3, orebody loop +2.2, economic agents +3.4,
total +5.6 C$M/world (~C$33M mine-scale). N=60 controls: det+naive-SL
-7.4/2%/0 profitable; placebo -12.4/70% (n.s. vs det); agents -7.4/30%/
10 profitable/P90 +2.8; ada-detsl +0.09 [-1.58,+1.86] n.s. on mean. POMDP finding: most ounces of
any arm at the lowest closed-loop grade, level with open-loop SP, dies in
world 9 where even the frozen plan survived. Geological learning without
economic agents converts information into more mining, not better
mining. World 5: SP -3.4 vs agents +10.0, C$13M opening after month ~12.

## 10. ABLATIONS -- CANONICAL (`madsen_ablations.py`, same information set)

Full-system swing +5.3. Precision note: mechanisms with |value| <~1 are NOT
resolved at n=10 (they reshuffle, and boundary redesign changes sign, under
pierce-noise redraws); the two bold mechanisms are robust.

| Mechanism removed | Cash | Insolvency | Mechanism value |
|---|---|---|---|
| (full system) | -7.6 | 30% | base |
| Belief updating | -14.5 | 100% | **+6.9** |
| Portfolio stop-loss | -9.9 | 70% | **+2.3** |
| Definition fans | -8.6 | 40% | +1.0 (indicative) |
| Hurdle-rate scoring | -8.3 | 50% | +0.7 (indicative) |
| Method choice | -7.6 | 30% | +0.0 (indicative) |
| Boundary redesign | -6.6 | 40% | -0.9 (indicative) |
| [sensitivity] orderly-premium claim | -9.5 | 40% | +1.9 |

Key readings: (1) discipline-without-loop swings NEGATIVE (-1.4) with 100%
insolvency -- worse than deterministic, and worse than the proper SP arm
(which never pays for information it cannot use); (2) stop-loss is the
second robust mechanism; (3) the premium claim carries +1.8 of +5.3 --
residual swing +3.5 with insolvency 50% vs 80% even at full chaos costs.

STATUS: paper (multi_agent_mine.pdf, 12 pp) incorporates the three-arm
comparison, canonical ablations, and precision statement. Fig18 replaces
fig16 in the paper; fig16 retained in outputs as the two-arm view.
