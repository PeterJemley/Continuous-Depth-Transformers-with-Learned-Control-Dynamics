# The necessity audit: a check before every run

Date: 2026-10-21    Status: method note, adopted from N12 on.

**Source.** This note adapts a method the project's author developed for analysing arguments in legal writing: finding the arguments that turn on a claim that some condition is necessary, which philosophers call transcendental arguments, and answering them move by move. Here the method's structure and its catalogue of replies are applied to the project's own designs.

## Why

Every preregistration rests on claims of the form "this requires that" or "without that, not this":
- the test requires that the manipulation took hold;
- the comparison requires a yardstick the arms cannot move;
- the measure, if it works, must recognise a known case.

A claim of necessity is a universal claim ("never this without that"), so one counterexample refutes it. Three of the project's claims were stated too strongly, two of them claims of necessity. In the first two a run found them, when an audit before the run could have; in the third a check before the run found what the audit had missed:
- **N11's dose check** was true at step 300 and false at step 2,879, a claim fixed to the wrong time.
- **N12's convergence rule** said a converged fit changes by at most 2%. A fit at zero loss converged and changed by 7%.
- **N13's stated mechanism** said the discount tells kept from replaced targets from the start, by frequency. The calibration's gradient check found them equal at the start: both kinds of target have the unigram distribution. The audit had covered the identity's dependence on time, not the mechanism's own claim about time.

The audit lists the necessity claims of a design before it runs, so that the cheapest criticism comes first.

## How to describe each claim

Each claim has three parts:
- a starting point (a test, measure, control or interpretation);
- the condition claimed to be necessary for it;
- the conclusion drawn.

Four features describe each claim:
- **Direction.** Forward: the starting point holds, so the condition holds. Reverse: the condition fails, so the starting point fails.
- **Strict.** The design's own instrument or claim is turned on itself. For example, a measure that claims to detect a shared reading must recognise a known one.
- **Functional.** The claim rests on what a measure or control is for. For example, a yardstick that the arms can move would make the comparison meaningless.
- **Ground.** What makes the condition necessary. There are three grounds:
  1. *Definitional:* a definition, a construction or a proof. In this project this is a claim about what a system can represent at all. Test it by constructing the case, or by fitting a known target.
  2. *A present mechanism with a good explanation:* how the system works now. In this project this is a claim about what training finds, or about how a procedure behaves. Test it with matched runs or a recorded check.
  3. *A prophecy:* that no method will ever find another way. This is never admissible as a claim. Restate it as what is not yet known, as the project's untestable verdicts already do.

## The replies, as questions the audit asks

| Reply | The question asked of each necessity claim | Where the project has met it |
|---|---|---|
| 1. Examine how the starting point is described | Does the description of the test already contain its conclusion? | `sub20` is defined by `n80`'s exact unreplaced positions, so "the same informative targets" holds by construction |
| 2. Show the starting point without the condition | Is there a counterexample? | N7; N12's convergence rule |
| 3. Name another mechanism | Could something other than the conjectured cause produce the same result? | N12's competing hypotheses |
| 4. Hold the argument to its modest conclusion; restore the modal word | Does the result show what the claim says, or only something weaker? Does a summary say more than its source? | The agreement ratio measures how predictions depend on the state, and "what a head reads" is a gloss on it |
| 5. Fix the condition to a time or a level | Is the claim true at one time or level and assumed at another? | N11's dose check; function against geometry (N10, N11) |
| 6. Turn a necessity into a question of fact | Can the claim be measured instead of argued? | N12d's fitted readout |
| 7. Show the rival rule's mirror-image defect | Which error does a decision rule choose to risk, and is that the right one? | "Untestable" rather than "refuted" when a manipulation fails |
| 8. Show that the argument proves too much | Does the test pass where it must fail? | G1's first measure passed untrained models |
| 9. Apply the argument to itself | Does the rule meet its own condition? | Identical heads must score 1; `CRITICAL_REVIEW.md` section 9 |
| 10. Grant the possibility, deny that it happened | Is a claim possible but not shown by this test? | N10a, N11a |

## The procedure

1. **List every necessity claim the design depends on.** Cover:
   - the data (unseen, matched across arms);
   - the instruments (they can fail, and they can recognise a known case);
   - the manipulations (they took hold);
   - the controls (exact by construction);
   - the analysis (fixed yardsticks, and thresholds that still mean something at their boundaries);
   - the interpretation (what each outcome would teach).
2. **For each claim:**
   - state its features and its ground;
   - name the check that would expose it, whether an existing precondition or a new one;
   - say whether that check still works at the boundary: zero, saturation, and the end of the schedule.
3. **Run replies 2, 3, 4 and 5 against each claim, and reply 8 against every test.**
4. **Write the critic's strongest objections, and answer each two moves deep:** the reply, and the critic's best answer to it.
5. **Record the audit.**
   - Before approval, it goes in the preregistration.
   - After approval and before the run, it goes in the deviation log.
   - A finding that changes a claim, rule, arm, window or measure is an amendment, and it goes back to the author for approval.

## Modal words in summaries

A summary that restates a note keeps the note's modal word: "consistent with", "suggests", "exploratory", "secondary", "untestable", "intermediate". A summary never upgrades it.

`PLAIN_SUMMARY.md`, `CRITICAL_REVIEW.md` and the result lines in `CONJECTURE.md` and `PATHS_AND_UNDERSTANDING.md` are checked against the notes by a fresh reader that has not seen the drafting, an adversarial reread that is part of the method. The first such check, on 2026-10-21, found three substantive problems and seventeen minor ones in the two summaries, most of them modal upgrades; all were corrected, and the review's section on open correction records them. A second round, on the N12 passages, found three substantive and ten minor problems, all corrected. A third round, on the N13 passages, found three substantive and twenty-two minor problems, all corrected; most were again modal upgrades, and the largest was an exploratory reading of where head B's cost to head A comes from that had become a flat claim in three summaries. A fourth round, on the N14 passages, found nine substantive and seventeen minor problems, all corrected; the largest were a 95% exploratory interval quoted beside a preregistered 97.5% rule in five places, "the same in five seeds to within 0.02" said of a ladder whose rungs vary by 0.05 across seeds, and an inconclusive verdict described as the forbidden outcome having happened. A fifth round, on the N15 and N16 passages, found eight substantive and twenty minor problems, all corrected; the largest were an error in the algebra of an exploratory derivation, a calibration value presented as an input to a margin it had postdated, and claims beyond what the comparisons separated (a rival "ruled out" where only its prediction failed, and a judge's two differences counted where there were three). A sixth round, on N17's note and then on its summaries, found four substantive problems in the note and ten in the summaries, with minor ones besides, all corrected. The largest were the note's split of N16's result into shares for accuracy and for the loop, which the design does not separate; sharpness called matched when it matched only at the end of training; and a summary sentence crediting accuracy with part of the effect.

## The record so far

| Claim | Features and ground | What happened |
|---|---|---|
| A test of a payload presupposes that the payload took hold (N10, N11) | Forward; functional; rule | The manipulation checks failed, and the claims were untestable rather than refuted |
| A comparison is meaningless if the arms move its yardstick (N10's G) | Functional; present mechanism | It refuted G as a measure, and N11 used a fixed yardstick |
| A measure of shared reading must recognise a known one (N8, N9a) | Strict, self-applied; definitional | The measure passed: identical heads scored 1, and the ideal copy 0.95–1.00 |
| The payload holds, judged from 300 steps (N11's dose check) | A claim fixed to the wrong time; mechanism | Refuted by the full run: 0.27 at step 300, 0.59 at the end |
| A converged fit changes by at most 2% (N12) | Rule, stated too strongly | Refuted by a counterexample at zero loss; replaced by amendment A1 |
| N12's design (audited 2026-10-21, before its real run) | See the N12 deviation log | Three findings about interpretation, and one added secondary measure. The run was read as the audit prescribed: the form's share of the gap was reported as at most 38%, and the mixture head's gain over the fitted single softmax (0.10) was left unattributed, since it could be the cleaner learning signal the audit named or the fitting procedure's shortfall. Separating them is N12's next conjecture 1 |
| The discount tells kept from replaced targets from the start, by frequency (N13's stated mechanism) | A claim fixed to the wrong time; mechanism | Refuted by the calibration's gradient check before any run: equal at the start (0.090 against 0.087), apart by step 300 in the calibration run, and from about step 200 in the real run's record. Corrected in the deviation log; no claim or rule changed |
| N13's design (audited 2026-10-22, at drafting, before approval) | See `N13_PREREGISTRATION.md` | Eight claims listed, each with its check and that check's behaviour at the boundary. The run met every precondition, including the foreseen boundary of a replicate floor of 0, and supported all three claims. The audit had asked after the identity's dependence on time and not after the mechanism's own time claim; see the row above |
| `sub20-rw` isolates the discount's reweighting of the kept targets (N14c's control) | Functional; present mechanism; the audit flagged a sharper weighting | The calibration found a collapse of function (Δ_A 0.81 against 0.17), beyond what a sharper weighting implied: the control sits elsewhere on the function axis, and N14c's direct comparison reads that as the reweighting's trace. The rule stood as approved and was refuted; against the ladder the control sits 0.10 *above* the curve (a secondary measure, read as exploratory). The audit named the wrong consequence |
| The clean-target ladder is a fair null for "function without random targets" (N14's premise) | Reply 3; a present mechanism | The four heads with random targets sat on it within 0.02 on average over seeds (within 0.05 in any one seed); the reweighted control did not. The premise held where the test needed it and failed where the design had not matched function, which is the finding, not a flaw in the premise |
| N14's design (audited 2026-10-22, at drafting, before approval) | See `N14_PREREGISTRATION.md` | Nine claims listed, with each check at its boundary; every precondition held, and the ladder bracketed and ordered every seed. The audit had stated the detectable residual (0.06–0.08) and the result fell below it: inconclusive by rule, below the stated limit, with the point estimates on the rival's side |
| The tempered discounts train (N15's premise for its β rule) | Present mechanism; the audit named the cold start and set a calibration rule for it | Refuted by N15's calibration before any real run, on one seed: neither β = 2 nor β = 4 trained (9.8 and 10.0 nats from head A), so no β run met the rule, and β = 8, not run, would admit less still; N15 was withdrawn by the author |
| `sub20-oracle` against `sub20-rw` isolates the judge (N16c) | Functional; reply 3; the audit flagged sharpness as a second difference | The comparison worked against the rival it was built for (C −0.45, where the rival predicted within 0.10). But head A differs from head B in accuracy as well as in self-reference, and the audit did not list accuracy, so the label "self-reference is the mechanism" claimed more than the comparison separated. The audit had also prescribed reading a support with S1; the note does so (on the targets outside its own ignored set, `sub20-rw` is no worse than `sub20-oracle`, so its excess is concentrated in the 8.5% it ignores, a description rather than a decomposition). Lesson 14 in `CRITICAL_REVIEW.md` |
| N16's margins, set after one calibration seed had been seen (the critic's objection at drafting) | Reply 5 (level): a single value | The margins were set far from the values in hand (F 1.02 and P 0.15 from N15's calibration; for C, the −0.57 implied by N14's control, N16's own calibration later giving −0.48), and the five real runs came out near the calibration values (F 1.08, P 0.10, C −0.45), not near the margins |
| N16's design (audited 2026-10-22, at drafting, before approval) | See `N16_PREREGISTRATION.md` | Eight claims listed; every precondition held; three claims supported with room, and N16d inconclusive at the limit the design stated; the exploratory arms were dropped by the calibration's rule, as designed |
| Head C changes nothing in the trunk or heads A and B, and a copy of head B as judge stays head B (N17's passenger and copied-peer checks) | Definitional; strict, self-applied | Both passed bit for bit on the A100 with 16-bit training on, in the calibration and the real run. The passenger check also failed on a planted bug, before any run |
| N17b's support margin is reachable | Reply 8 at the boundary; found by a fresh reader at drafting, before approval | With the peer's recovery near 0, d is unlikely to fall far below minus the loop-free line's height. The height condition added to precondition 6b made a low line untestable. The line was low (0.31–0.37), so N17b was untestable, as the calibration entry had foreseen. A margin relative to the line's height would have stayed reachable |
| Matching the judges' accuracy matches them in sharpness and share (N17's exploratory reading) | Reply 5 (time): a condition fixed to one time | They matched at the end of training and differed during it, so the third that is not accuracy is not separated from sharpness or the course of training. A fresh reader's check of the note caught the first draft's "matched" |
| N17's design (audited 2026-10-25, at drafting, before approval) | See `N17_PREREGISTRATION.md` | Eleven claims listed. One fresh-reader round on the draft changed the design before approval, adding the judge-source check, the overflow test and the height condition in 6b. While the calibration ran, an independent review of the code found two defects and three gaps in the checks, all fixed before any calibration value was seen. Every precondition but 6b held: N17a was some effect, size unclear (0.334), and N17b untestable, both as the calibration had foreseen |
