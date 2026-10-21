# The necessity audit: a check before every run

Date: 2026-10-21    Status: method note, adopted from N12 on.

**Source.** This note adapts the detection step and the catalogue of replies of *transcendental-counterplay*, a legal-research skill written by the project's author. That skill finds, in legal writing, the arguments that turn on a claim that some condition is necessary, and it answers them move by move. This note borrows its structure and applies it to the project's own designs. The skill's legal examples, and their verification, stay in the skill.

## Why

Every preregistration rests on claims of the form "this requires that" or "without that, not this":
- the test requires that the manipulation took hold;
- the comparison requires a yardstick the arms cannot move;
- the measure, if it works, must recognise a known case.

A claim of necessity is a universal claim ("never this without that"), so one counterexample refutes it. Two of the project's failures were necessity claims stated too strongly, and in both a run found them, when an audit before the run could have:
- **N11's dose check** was true at step 300 and false at step 2,879, a claim fixed to the wrong time.
- **N12's convergence rule** said a converged fit changes by at most 2%. A fit at zero loss converged and changed by 7%.

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

| Reply (from the skill) | The question asked of each necessity claim | Where the project has met it |
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

This follows the skill's checklist addition 6. A summary that restates a note keeps the note's modal word: "consistent with", "suggests", "exploratory", "secondary", "untestable", "intermediate". A summary never upgrades it.

`PLAIN_SUMMARY.md`, `CRITICAL_REVIEW.md` and the result lines in `CONJECTURE.md` and `PATHS_AND_UNDERSTANDING.md` are checked against the notes by a fresh reader that has not seen the drafting, as the author's writing method prescribes for its adversarial reread. The first such check, on 2026-10-21, found three substantive problems and seventeen minor ones in the two summaries, most of them modal upgrades; all were corrected, and the review's section on open correction records them.

## The record so far

| Claim | Features and ground | What happened |
|---|---|---|
| A test of a payload presupposes that the payload took hold (N10, N11) | Forward; functional; rule | The manipulation checks failed, and the claims were untestable rather than refuted |
| A comparison is meaningless if the arms move its yardstick (N10's G) | Functional; present mechanism | It refuted G as a measure, and N11 used a fixed yardstick |
| A measure of shared reading must recognise a known one (N8, N9a) | Strict, self-applied; definitional | The measure passed: identical heads scored 1, and the ideal copy 0.95–1.00 |
| The payload holds, judged from 300 steps (N11's dose check) | A claim fixed to the wrong time; mechanism | Refuted by the full run: 0.27 at step 300, 0.59 at the end |
| A converged fit changes by at most 2% (N12) | Rule, stated too strongly | Refuted by a counterexample at zero loss; replaced by amendment A1 |
| N12's design (audited 2026-10-21, before its real run) | See the N12 deviation log | Three findings about interpretation, and one added secondary measure |
