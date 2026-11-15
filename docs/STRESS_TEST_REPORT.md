# Stress-Test Report: The Multi-Agent Mine

Scope: correctness, fairness, information hygiene, statistical robustness,
and paper-versus-code reconciliation for the Madsen simulation study.
Harness: `stress_tests.py`. Archives: `stress_boot_merged.npz`,
`stress_tornado.npz`, `stress_fairness.npz`, `boot_{0,20,40}.npz`.
No paper or method updates have been made. Findings only.

## 1. Suite and verdicts

| # | Test | Verdict | One-line result |
|---|---|---|---|
| 1 | Golden stope (hand P&L) | 4/4 PASS | sim cash matches hand arithmetic to the dollar over a full extraction |
| 2 | Ledger accountant (live runs) | PASS | max per-period discrepancy C$0.0000 across instrumented agent runs |
| 3 | Mass balance | PASS | milled metal never exceeds world metal, buckets sum exactly |
| 4 | Static truth-access audit (AST) | PASS | GeologyAgent, CostAgent, run_agents never index the hidden truth |
| 5 | Information parity | PASS | all four arms consume one (truth, pierce) hash per world |
| 6 | Degenerate: all-barren | PASS | every arm mines zero ounces and exits with minimal loss |
| 7 | Degenerate: truth-in-ensemble oracle | PASS | truth ranked #1 in 100% of stopes at the ESS-floor ceiling |
| 8 | Degenerate: homogeneous world | FAIL (finding F6) | +4.0M residual agent advantage at equal cost premium |
| 9 | Placebo (noise observations), n=6 | PASS w/ alarm | resolved at N=60, see F3 |
| 10 | Fairness: MAX_OPEN | PASS | 5 vs 8 concurrent stopes moves less than 1M either arm |
| 11 | Fairness: premium gift to det | measured | orderly premium is worth +1.8M to the deterministic arm |
| 12 | Fairness: naive stop-loss | FAIL (finding F2) | headline finding, below |
| 13 | Bootstrap, 6 arms, N=60 | complete | CIs below |
| 14 | Pierce-redraw stability | PASS | mean swing {+3.9, +4.6, +3.2}, sd 0.58 |
| 15 | Tornado, 9 parameter settings | PASS | swing positive in every setting, range [+2.6, +5.2] |
| 16 | Paper-vs-code reconciliation | 29/30 | one rounding inconsistency (F7) |

## 2. Bootstrap results (60 fresh worlds, paired, 20k resamples)

| Arm | Mean C$M | Insolvency | Profitable worlds | P90 | Best |
|---|---|---|---|---|---|
| Deterministic | -12.6 | 75% | 3/60 | -7.3 | +8.2 |
| Det + naive stop-loss | -7.4 | 2% | **0/60** | -3.8 | -1.6 |
| Stochastic plan | -10.4 | 48% | 4/60 | -2.2 | +10.5 |
| Single POMDP | -9.8 | 52% | 6/60 | -1.6 | +21.6 |
| Multi-agent | **-7.8** | **30%** | **8/60** | **+2.7** | +14.9 |
| Placebo agents | -12.1 | 77% | 4/60 | -2.6 | +5.8 |

95% CIs on paired differences (C$M/world, * = excludes zero):
SP-det +2.20 [+1.08, +3.39]*. POMDP-det +2.79 [+1.74, +3.95]*.
Agents-POMDP +2.07 [+0.74, +3.47]*. Agents-SP +2.67 [+1.69, +3.64]*.
Agents-det +4.87 [+3.42, +6.35]*. DetSL-det +5.18 [+4.01, +6.21]*.
Agents-detSL -0.31 [-1.94, +1.40], not significant.
Placebo-det +0.52 [-0.91, +1.99], not significant.
Agents-placebo +4.34 [+3.21, +5.54]*.

The published decomposition ordering and significance survive at 6x the
sample size. Insolvency 75% to 30% survives.

## 3. Findings

**F1 (bug, fixed, zero numerical impact).** `GeologyAgent` set its
realization count from the `n_real` argument instead of the supplied
ensemble. Latent because the sizes always matched. Fixed to derive K from
the ensemble. Full canonical reproduction after the fix confirms no
number in the paper changes.

**F2 (headline finding, method design).** A belief-free trailing-cash
stop-loss bolted onto the plain deterministic plan recovers +5.18M/world,
and the full multi-agent system is statistically indistinguishable from
it on MEAN operating cash (-0.31 [-1.94, +1.40]). The distributions are
opposite products. The naive rule produces zero profitable worlds in
sixty, kills even the three mines the plain deterministic arm finishes
profitably (it cannot tell a ramp-up dip from a death spiral), and caps
every outcome below -1.6M. The agents produce eight profitable worlds, a
positive P90, and a best world of +14.9M. Diagnosis of the mean gap: the
agents' own stop-loss is gated on `remaining_ev < 12 x burn`, so a
hopeful posterior keeps them mining in bad worlds where the dumb rule
exits. Beliefs ride losers. Implication for the paper: the claim "the
multi-agent system improves mean cash" is incomplete without this
comparator. The defensible claim is distributional: it is the only arm
that simultaneously delivers large downside protection AND creates
profitable mines. Recommended method change (not yet applied): tighten or
remove the inventory gate on the stop-loss and re-run.

**F3 (anti-hallucination control, passes at scale).** Agents fed
information-free observations collapse to +0.52 [-0.91, +1.99] over
deterministic, indistinguishable from zero, while real observations are
worth +4.34* over placebo. The measured value is carried by information
content, not by machinery, luck, or leakage. The alarming +5.5 placebo
swing on the 6-world subsample was small-sample noise from two rich
worlds, a caution against every small-N claim in the study.

**F4 (modeling claim quantified again).** Granting the deterministic arm
the orderly premium is worth +1.8M, matching the published sensitivity
ablation from an independent construction.

**F5 (fairness).** The MAX_OPEN asymmetry (8 vs 5) moves less than 1M in
either direction for either arm and slightly favors the agents when
removed. Immaterial.

**F6 (degenerate-world residual).** On a saturated homogeneous world with
exact data, agents at equal premium still beat deterministic by +4.0M.
On realistic worlds the fairness ablations bound the non-informational
residual near +1M. The saturated world amplifies scheduling-speed and
selection effects that are negligible at Madsen-like ore density. Known,
bounded, and worth one sentence in limitations.

**F7 (paper inconsistency, one number).** The four-arm table prints
"total +5.2/world". The regenerated value is 5.27, which rounds to 5.3,
and the ablation caption already says +5.3. One-character fix pending.

**F8 (reproduction).** All 30 quantitative claims checked (four-arm cash,
insolvency, grades, ounces, best worlds, all seven ablation values)
regenerate from code post-bugfix, 29 exact at stated precision, 1 the
rounding above.

**F9 (robustness).** Swing direction is stable under 3 pierce-noise
redraws (sd 0.58M) and under all 9 tornado settings (ELOS +-0.08, cost
truth +-30, credit limit -10M/-18M, agent premium 35, lambda 0/0.8),
range +2.6 to +5.2. Largest sensitivities: funding envelope and risk
weight lambda.

## 4. Recommended updates (none applied yet)

1. Add the naive stop-loss comparator and the placebo control to the
   paper, and reframe the multi-agent claim distributionally (F2, F3).
2. Fix "total +5.2" to "+5.3" (F7).
3. Tighten the agents' stop-loss inventory gate and re-run the suite; the
   mean gap to detSL should close or invert (F2).
4. Report the N=60 CIs alongside the 10-world canonical table (F9).
5. Add one limitations sentence on the saturated-world residual (F6) and
   one on the premium share (F4).


## 5. Post-fix addendum (updates executed)

All five recommended updates have been executed. F2's root cause was
reclassified from a design weakness to a coding bug: the stop-loss
commitment block read `if stop_loss and not active`, so new stopes kept
opening while any stope was active and the rule almost never bit. Fixed
to block all new commitments and paid drilling whenever the stop-loss
fires, with orderly completion of open stopes.

Post-fix canonicals (10 worlds): agents -7.6 mean, 30% insolvency,
6.8 g/t, best +10.3. Decomposition: ensembles +2.3, orebody loop +2.2,
economic agents +3.4, total +5.6 (~C$33M mine-scale). Ablations: belief
+6.9, stop-loss +2.3, chaos sensitivity +1.9 of +5.6.

Post-fix bootstrap (N=60, det/detsl/sp/pom arms unchanged): agents -7.4,
30% insolvency, 10/60 profitable, P90 +2.8, best +14.9. Agents-detsl
+0.09 [-1.58, +1.86]: mean parity with the naive rule stands, and the
distributional contrast strengthens (10 vs 0 profitable worlds).
Placebo remains null (+0.22 [-1.13, +1.51]); agents-placebo +5.05
[+3.67, +6.57]*. Paper, deck, Word document, and calibration sheet all
carry the regenerated numbers, the controls table (Table 5), the
adversarial-controls section, and the limitations additions. The paper
discloses the harness and the fixed inversion.
