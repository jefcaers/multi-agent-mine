# Phase 2 results (M3, M5, M1-lite)

Held-out worlds, seeds 300-359, N=60, paired against
stress_boot_final.npz. Canonical worlds, seeds 200-209, N=10.
Code: madsen_rolling.py. Archives: rolling_canon.npz, roll_boot_0.npz,
m1lite.npz.

## M3. Rolling-horizon stochastic planning

Registered prediction: rolling lands between plan-once SP and the
POMDP. Result: the prediction fails in the informative direction.
Rolling lands above the POMDP and just below the full system.

Held-out, N=60 (mean cash C$M/zone, paired 95% CI, 20k resamples):

| comparison        | diff  | CI              | sig |
|-------------------|-------|-----------------|-----|
| rolling - det     | +4.28 | [+2.93, +5.71]  | *   |
| rolling - sp      | +2.08 | [+1.13, +3.08]  | *   |
| rolling - pomdp   | +1.49 | [+0.26, +2.81]  | *   |
| agents - rolling  | +1.00 | [+0.02, +1.93]  | *   |
| rolling - detsl   | -0.90 | [-2.62, +0.97]  | n.s.|
| rolling - placebo | +4.05 | [+2.55, +5.79]  | *   |

Rolling: mean -8.35, insolvency 35% (21/60), 10 profitable mines,
P90 +3.5, best +14.5. Agents: -7.35, 30% (18/60), 10 profitable,
P90 +2.8, best +14.9. Correlation across worlds 0.88.

Interpretation. Quarterly re-conditioning feeding the same gated
commitment machinery (p >= 0.55 threshold, mill-space hurdle) captures
most of the closed-loop value: +4.3 of the full system's +5.3 over the
deterministic baseline. The residual from the agents-only mechanisms
(definition fans, live cost model, cash stop-loss) is +1.00 with a CI
that touches zero from above; it does not survive family-wise
correction. The plan-completion close on a re-conditioned plan acts as
an implicit stop rule: when the posterior turns against a world, the
plan empties and the mine closes. Per-world extremes show the agents'
edge concentrated in worlds where selective deferral avoided marginal
commitments (up to +12.1), against worlds where rolling's fuller plan
paid (up to -12.4).

Claim revision required in the paper and book: the value lies in the
loop, evidence re-conditioning a belief that feeds gated, stoppable
commitments, and not in the agent decomposition per se. The
decomposition's defensible claims: a small residual (+1.0 mean, sig.
unadjusted), lower insolvency (30% vs 35%, n.s.), and auditability,
which rolling does not provide. The naive stop-loss comparison is
unchanged and remains decisive against detsl: 0 profitable mines
against 10 for rolling and agents.

## M5. Certainty equivalents, exponential utility

CE (C$M/zone), N=60:

| approach        | g=0     | g=0.05  | g=0.15  |
|-----------------|---------|---------|---------|
| deterministic   | -12.63  | -13.11  | -13.65  |
| naive stop-loss | -7.45   | -7.66   | -8.10   |
| sp              | -10.42  | -11.20  | -12.13  |
| pomdp           | -9.83   | -10.86  | -11.97  |
| rolling         | -8.35   | -9.59   | -11.05  |
| agents          | -7.35   | -8.54   | -10.15  |

The agents rank first at every risk aversion; the ordering
agents > detsl > rolling holds throughout, so the headline is not an
artifact of risk preference. Portfolio framing: across 60 zones the
naive rule creates 0 profitable mines; rolling and agents each create
10.

## M1-lite. No-interpretation-diversity probe

Agents run with an ensemble drawn entirely from the wide interpretation
(frac_A=1.0), canonical 10 worlds: mean -8.23, insolvency 40%, against
-7.6 and 30% with the mixed ensemble. Registered prediction (orebody
loop value collapses) also fails, and again informatively: the system
degrades by only 0.6 mean and 10 insolvency points. Mechanism: mill
reconciliation still reweights within the wide family toward its
lower-grade members, and the economic gates act on whatever belief
exists. Interpretation diversity is worth roughly +0.6 mean and -10
insolvency points on this sample; the larger protection comes from the
gates plus any live re-conditioning. For deployment (Chapter 9) this is
a robustness result: the loop self-corrects partially even under a
mis-specified prior family.

## Consequences for documents

1. Paper: Table 5/6 gain a rolling column; abstract and contribution
   claims narrow to the loop-not-architecture statement; decomposition
   section rewritten around rolling - sp (+2.1, re-conditioning) and
   agents - rolling (+1.0, fans + costs + stop-loss, fragile);
   Bonferroni set enlarged; limitations note both failed registered
   predictions, stated as such.
2. Book Chapter 6: the comparison becomes five approaches (or four plus
   rolling as the strongest control in Chapter 8). Chapter 8 gains the
   two falsified predictions as worked examples of the method working.
3. Deployment protocol: unchanged gates; add the M1-lite robustness
   note.
