# What we guessed, how we tried to break it, and what survived

Date: 2026-10-10, updated 2026-10-18 with N10, 2026-10-20 with N11, 2026-10-21 with section 9, lessons 9–10, corrections from a fresh reader's audit against the notes, and N12, and 2026-10-22 with N13, section 11 and lesson 12, and N14, section 12 and lesson 13, and N15's withdrawal and N16, section 13 and lesson 14, with a correction to the count of amendments (N14's A1 had been left out), and 2026-10-26 with N17, section 14 and lessons 15–16. A review of all the work in this repository, from the replication of the paper through N17, in the manner of Karl Popper and David Deutsch. As in `PLAIN_SUMMARY.md`, the main text avoids mathematics, and the details are in numbered notes at the end. Every number comes from a result note in this repository; the notes say which.

## How to read this review

**The view of knowledge behind it.**
- Popper's view: we never prove a general claim true. We guess (conjecture), then try as hard as we can to show the guess is wrong (criticism). A guess that survives a test it could have failed is not proven. It has earned the right to be taken seriously until the next test.[^1]
- Deutsch adds that what we are after is not just predictions that survive, but *good explanations*. An explanation is good when it is hard to vary: its details are fixed by what it explains, so it cannot be bent to fit whatever happens.[^2]

**On this view, a body of work can fall short in three ways.**
1. **A guess that forbids nothing.** If no result could count against it, it says nothing.
2. **A test that cannot fail.** If it would pass whether or not the guess is true, passing it means nothing.
3. **A refuted guess that is quietly saved.** This happens when, after a "no", the claim, the measure or the threshold is changed so that the "no" goes away. Popper called these moves *conventionalist stratagems*.[^1]

**So this review asks four things of every step:**
- Did the guess forbid something specific?
- Could the test have failed, and was that checked?
- When the answer was "no", was the "no" kept?
- What explanation stands now, and how hard is it to vary?

It is meant to criticise the work as hard as the work tried to criticise its guesses. That includes the mistakes made along the way, by me (Claude, who designed and ran the tests and wrote the notes) and by the method.

## The ledger

The main claims put to a test, in order; some rows merge a test's paired claims, and the secondary claims and T1's rivals are in the notes. "Forbade" is what the claim said could not happen. That is the result that would refute it.

| Test | The guess | What it forbade | Verdict |
|---|---|---|---|
| WikiText-2 replication | The paper's claims: steering works, at no cost in accuracy or speed, with a learned scale near 0.065 and distinct flow regions | Unreliable steering, slower models, other values of the scale, no distinct regions | Mostly **refuted**[^3] |
| WikiText-103 | The model matches an ordinary transformer when properly trained | A gap larger than 0.02 nats | **Refuted**: worse by 1.10 nats in 6 of 6 runs[^3] |
| T0 | The training shortcut (the adjoint) gives a learning signal pointing well away from the exact one | Close agreement | **Untestable**: the measurement was too noisy[^4] |
| T1 | The shortcut, not the design, causes most of the gap | Training exactly closing less than a material part of the gap | **Supported**: about 95% of the gap[^4] |
| G1 | The control signal sends the flow through regions of different curvature | A curvature ratio below 1.25 | **Refuted**[^5] |
| G1c | It sets how strongly the flow pulls nearby states together | A contraction ratio below 1.25 | **Supported**, and it replicated[^5] |
| G1d | The model needs that pull | Removing it costing less than 0.05 nats | **Refuted**[^5] |
| N1 | Switching readings needs only a fixed shift, not the dynamics | A loss when the dynamics are bypassed | **Supported**[^5] |
| N2 | Two readings held at once get different paces in the flow | A ratio of paces below 1.25 | **Refuted**[^6] |
| N4 | Readings are kept apart by direction, and the model needs that | Forcing shared directions costing less than 0.05 nats extra | **Supported**[^7] |
| N5 | N4's cost is a competition for directions | A cost remaining when nothing is restricted | **Inconclusive**[^7] |
| N6 | Sharing depends on how different the readings are | Overlap falling by less than 0.10 as the second reading looks further ahead | **Untestable**: its control was set wrongly[^8] |
| N6r | The same, with the control fixed | The same | **Supported**[^8] |
| N7 | Sharing tracks how different readings are, not how hard | A harder copy of the same reading sharing as little as a different one | **Refuted**[^9] |
| N8a, N8b | With mistakes left out of the measure, the harder copy shares and converges | It still not sharing | **Refuted**[^10] |
| N8c | Readings further ahead use different directions, mistakes left out | The ladder flattening | **Supported**[^10] |
| N9a | The measure can recognise an ideal noisy copy of a reading | Low agreement between a reading and its ideal noisy copy | **Supported**[^11] |
| N9b | The trained noisy head is close to that ideal copy | Low agreement between them | **Refuted**[^11] |
| N10a | A small functional payload preserves the reading under heavy noise | The gap staying above 0.20 nats | **Untestable**: the payload did not take hold[^14] |
| N10b | Anchoring function beats pinning geometry | Their gaps differing by less than 0.20 | **Untestable**, as N10a[^14] |
| N10c | Light noise does not change what a head reads | Low agreement at 20% noise | **Intermediate**: 0.79–0.84, against 0.85[^14] |
| N10d | The change grows with the noise | Agreement not falling with noise | **Supported**[^14] |
| N11a | A strong functional payload preserves the reading's function under heavy noise | Less than two thirds of the gap closed | **Untestable**: the payload again failed its check[^15] |
| N11b | The effect grows with the payload's strength | The strongest payload doing no better than the weakest | **Supported**, with sharply diminishing returns; its last step is within the GPU's own effect[^15] |
| N11c | Anchoring function beats pinning geometry, each at full strength | A difference under 0.30 of the gap | **Untestable**, as N11a[^15] |
| N12a | A head whose form can represent the noisy mixture keeps the reading, from the noisy targets alone | Less than two thirds of the gap closed | **Supported**: 71%[^16] |
| N12b | And reads the same directions as its ideal copy | Agreement at most 0.60 | **Intermediate**: 0.79–0.85, under 0.85 in every seed[^16] |
| N12c | Scarcity of informative targets alone does not cause the drift | Less than two thirds closed by the data-matched clean head | **Supported**: 80%[^16] |
| N12d | The single-softmax bottleneck is large | A single softmax fitted directly to the copy closing two thirds or more | **Supported**: 62%[^16] |
| N13a | The pipeline follows the chain rule's identity: a single softmax trained with the discount is the mixture head's inner softmax | A difference above 0.10 nats in 4 of 5 seeds, with the replicate within 0.03 | **Supported**: within 0.005 in 5 of 5[^17] |
| N13b | The mixture form, applied at readout to the standard head, closes none of the gap | A third or more of the gap closed | **Supported**: S = −1.01, the gap doubled[^17] |
| N13c | The mixture head's gain is not from head B pulling less on the shared network | A constant discount giving half or more of the gain | **Supported**: D = +0.05, none of it[^17] |
| N14a | Random targets move a head's directions beyond their cost to its clean-target prediction | A residual within 0.01 of the clean-target ladder | **Inconclusive**: +0.009 [−0.035, +0.052]; on the ladder within 0.02[^18] |
| N14b | And the trace grows with their number | The trace complete at the smallest dose | **Intermediate**: the smallest dose carried about half[^18] |
| N14c | The discount's reweighting of the kept targets is not the trace | The reweighting alone moving the directions by two thirds of the trace | **Refuted**: 0.125, with the control's function collapsed[^18] |
| N16a | The mixture head's shortfall against the oracle mask is in its self-judged discount: a discount judged by head A recovers it | Head A's judgement recovering a quarter of the shortfall or less | **Supported**: 108% [104%, 112%][^19] |
| N16b | The random weight the discount leaves is not the cause | The recovery coming with three quarters or more of the random share removed | **Supported**: a tenth removed[^19] |
| N16c | Self-reference is the mechanism: on clean targets alone, head A's judgement costs far less than head B's own | The two judges' weightings costing alike, within 0.10 nats | **Supported**: 0.45 nats less; the design does not separate self-reference from head A's accuracy[^19] |
| N16d | The recovered head's directions follow its function along the clean-target ladder | A residual beyond 0.04 | **Inconclusive**: +0.011 [−0.022, +0.043], the design's stated limit[^19] |
| N17a | Without the self-confirming loop, a judge as accurate as head B recovers most of the shortfall | Such a judge recovering a quarter of it or less | **Some effect, size unclear**: a third, 0.334 [0.318, 0.349][^20] |
| N17b | A judge that is not head B but has its own loop does far worse than a loop-free judge of its accuracy | The peer falling short by a quarter of the shortfall or less | **Untestable**: the loop-free line at the peer's accuracy was too low for the rule to apply[^20] |

**Out of the 42 verdicts in this ledger:**
- 10 refuted, or mostly refuted;
- 18 supported;
- 7 inconclusive, intermediate, or some effect of unclear size;
- 7 untestable.

N15 has no row: it was withdrawn before any real run, and decided nothing (section 13).

Counted per preregistered claim, including the secondary ones the ledger leaves out (N1a, N2b and N6's N6b, each untestable; N4a, supported; N5b, inconclusive), the untestable verdicts number at least ten.

None of the refuted claims was later restored.

## The thread of explanation

The work is best read as one explanation being replaced by another. Each "no" says what is wrong, and each new guess has to explain both the "no" and what still stands.

1. **"The continuous block steers the model at no cost."** Refuted. The replacement, *the training shortcut is to blame*, was tested directly (T1) and survived. It is a good explanation in Deutsch's sense: it says which part of the system is at fault, by how much, and what would show otherwise. Its mechanism, how the shortcut goes wrong, is untested: T0 could not give a verdict.
2. **"The dynamics carry the readings."** Three different ways were tested, and each refuted:
   - different curvature (G1);
   - a needed contraction (G1d);
   - different paces (N2).

   One way that held, the contraction (G1c), turned out not to be needed. The replacement, *the readings live in the geometry, the arrangement of directions, not in the flow*, came from something N2 noticed but had not predicted. It was then tested in a way that could fail (N4), and it survived.
3. **"How much two readings share depends on how different they are."** The distance ladder (N6, N6r) fitted it. It was then stated as a rule in the plain summary, before it had been tested against the obvious rival: further readings are also harder. When it was tested (N7), it was refuted.
4. **"The measure confuses where a reading's mistakes fall with what it uses."** Offered after N7 and refuted by N8: removing the mistakes changed almost nothing. With hindsight it was a weak explanation. A known result already implied it would fail (see "Where we fell short").
5. **"The measure cannot recognise a noisy copy of a reading."** This was a criticism of the instrument, not of the claim. It was tested independently (N9), with N8's verdict left standing whatever happened, and refuted. The measure recognised an ideal noisy copy as the same reading. The noisy head that training actually produced was not one.
6. **What stands now** is narrower and harder to vary:
   - readings further ahead depend on different directions of the state, under two different measures;
   - forcing two readings through shared directions costs accuracy;
   - a head trained on noisy targets learns a different, weaker reading, not a noisy copy of the right one.

   The last explanation is hard to vary, because N9's design left it no room. The measure was shown to recognise the copy. The trained head was shown not to be the copy. And its predictions were shown to be weaker on the clean targets.

## Where the criticism worked

These are the practices that kept the work honest. Each is a way of making the world's "no" audible.

- **Plans fixed in advance.** Every test had its claim, thresholds and decision rules written and approved by the author before any run. Every change made after approval went back for approval with the reason written down, and was recorded in a deviation log. There have been twelve such amendments, counting N12's and N14's, each made before the real run it concerns; and one preregistration, N15, was withdrawn before any real run, by the author's decision.[^12]
- **Checking that tests can fail, on models that know nothing.** Before every run, the measure was tried on untrained models, where a true claim should fail.
  - Checks before the real run caught problems in six tests: G1, N2, N4, N7, N8 and N9. Three of those catches came from the untrained models; the others came from a loss bound, a reread of the plan, and a caution that turned out not to be needed.[^13]
  - In G1, the original measure would have called an untrained model a success.
  - In N8, the planned control would have passed automatically.[^13]
- **Controls at both ends.** Most tests from N4 on had a control that must match, such as two copies of the same reading, and one that must differ. A result counted only if both behaved.
- **Refutations kept.** No claim refuted here was later reinstated, and no threshold was moved after a result was seen. Some exploratory observations became new claims: the contraction in G1, the directions in N2, the ladder in N6. Each was then tested as a new conjecture, on new seeds and new data, and the old verdict stayed as it was.
- **Mistakes in the notes corrected openly.** When a later test showed that an earlier note had said too much, a dated correction was added beside the original text, which was left in place. This happened four times: in the notes of N4, N6r (twice) and N7. The plain summary's sentence stating the rule was replaced, and the summary now says that the rule did not survive its test. On 2026-10-21 a fresh reader, given only the summaries and the notes, audited both summaries against the notes. It found three substantive problems (N1's result stated as what the model does rather than what suffices; N11's share of the gap presented as a share of the whole cost of scrambling; section 9's statement of the amendment rule contradicting the record) and seventeen minor ones, most of them modal upgrades. All were corrected in one change, and the audit is now part of the method (`NECESSITY_AUDIT.md`). A second round, on the N12 passages written the next day, found three substantive problems (a one-sided result stated as a cap on what any single softmax could do; the stronger wording the audit had withdrawn, reinstated in three places; an unseparated effect presented as identified) and ten minor ones. All were corrected. In both rounds the substantive errors were in the most recently written sentences.
- **Fair comparisons.** In the WikiText-103 study, the ordinary transformer's best learning rate was the largest one tried. Its range was therefore extended before the main runs, so that the comparison would not favour the paper's model by under-tuning its rival.[^3]
- **Numbers from code.** Every reported number was computed by an analysis script from saved records. Each confirmatory script was written and tested on synthetic records, covering every branch of its rules, before the real run. The exploratory scripts (N10's, N11's and the hardware comparison) were written after the results, and they are labelled so.

## Where we fell short

This is the most important part of the review. A method that never finds its own errors is not working.

### 1. Explanations offered faster than they were criticised

Three times, a note went further than its evidence:
- **N4's note** said where two readings end up depends on where they start. It generalised from one width, and N5 showed it was wrong.[^7]
- **N6r's note**, and then the plain summary, stated "the more different the voices, the more separate their directions" as a rule. The rival it had to beat, difficulty, was named in both preregistrations but had not been tested. N7 refuted the rule.[^9]
- **N7's note** offered "the measure picks up mistakes" as the likely reason. N8 refuted it.[^10] It should not have needed a test. The two measures have the same expected value whenever a head's predictions match its targets. That result is known (Kunstner et al., 2019), it was cited in N8's own preregistration, and it was not applied until after N8 ran.

In Deutsch's terms, these explanations were *easy to vary*. Each fitted the result in front of it, but nothing in its details was forced by the evidence. "The more different the voices, the more separate" would have fitted almost any falling ladder.

**The lesson.** An explanation offered after a result should be checked against what is already known before it is written down as "likely". Until a test has had a chance to refute it, it belongs in a note as a conjecture, not in a summary as a finding.

### 2. Controls carried over without re-checking

Twice, a number set for one test was reused in another whose conditions had changed:
- **N6** reused N4's threshold for its must-match control, although it measured on almost half as many passages. With less text, noise kept even identical readings below the threshold, and N6 became untestable.[^8]
- **N5** assumed runs would vary as little as they had in N4. At other widths they varied 3 to 24 times more, and N5 became inconclusive.[^7]

Neither was a rescue, since both were recorded as what they were. But both cost a run.

**The lesson.** A threshold belongs to its measurement conditions. Re-check it whenever they change.

### 3. Checks that could not fail, built first and caught later

On the first attempt, three checks could not have failed:
- the analytic check of N8's estimator: a broken sampler passed it at 0.996;
- the analytic check of N9's estimator: measuring the wrong head passed it at 0.999;
- N8's planned calibration of its must-match control: untrained heads already scored about 1.[^13]

Each was caught before any real run, because the wrong version was also tried. That is the method working. But the pattern repeated, which means it is a habit to guard against, not a one-off slip.

**The lesson.** Build every check together with the wrong answer it must reject, and confirm that it rejects it.

### 4. The measure was trusted before it was validated

From N2 to N7, the overlap of two readings' sensitive directions was read as "how much of what they use is shared".
- Its two ends were anchored: identical readings score about 1, and random directions about 0.06.
- But until N9 it was never tested on a case where the right answer is known and lies in between, or on the same reading in a different form.
- N7's and N8's surprises could have come from this gap. N9 closed it: the measure recognised an ideal noisy copy at 0.98, and so the surprise is in the model. The validation came after three tests that relied on the measure. It did not overturn them, but it should have come first.

**The lesson.** Before a measure is used as evidence about a claim, it should be checked on transformations where the answer is known.

This is the Duhem–Quine point, which Popper answered. A failed prediction refutes the claim *together with* the assumptions needed to test it, here that the measure measures what it is meant to. It is legitimate to test such an assumption on its own afterwards, as N9 does, provided the original verdict is left standing whatever the new test shows. It is a stratagem to change the assumption only so that the verdict changes.[^1]

### 5. The tests drifted from the question that started them

The project began with a conjecture about reading as listening to music: holding characters, arguments and tone at once, as voices in counterpoint. That conjecture's direct tests, V1 to V3 in `CONJECTURE.md`, have not been run. They would use texts with several strands, and models built to hold several strands at once.

**What was tested instead.**
- Every reading tested so far is a future word, one, two, four or eight places ahead. These are very close relatives.
- From N6 on, the questions have increasingly been about the instrument, the overlap measure, rather than about reading.

That was the right order. A measure has to be trusted before it can carry the main question. But it means the project's central guess is untested, not supported. The results so far are about the parts it would be built from.

### 6. Reuse, scale and scope

- **Reused models.** N8 and N9 measure models trained for N7 and N6r. The measures are new and the predictions were fixed first, but every finding on those models shares one set of training luck. N6r's ladder replicated in N7, and N7's in N8, but N8 used the same models again.[^10] The noisy-head finding has since replicated on new seeds, in N10 and N11.
- **Small models and short training.** All the models have about 30 million parameters, on one dataset, at 10% of the full training budget, with one learning rate. Any of the findings could change at scale.
- **One readout width carries most of the claims.** It is 64 of 256. N5 showed that how much the runs vary depends strongly on width.[^7]
- **One kind of GPU until N11.** N1 to N10 ran on a T4, and N11 on an A100. On 14 matched runs, the GPU shifted nothing systematically, but it moved losses by up to 0.024 nats, about as much as a change of seed. Quantities measured within one model moved less. That was enough to erase N11b's last step on the T4, so a claim resting on a difference of about 0.02 or less needs one kind of GPU throughout.[^15]

### 7. Payloads that did not take hold, twice, and a yardstick that moved (N10, N11)

- **The functional payload did not take hold.**
  - I set its strength at λ = 1, on 4 windows per step against 32 in the training batch. That gave it about an eighth of the weight of the objective it was meant to counter.
  - A short calibration run showed it moving in the right direction. I judged it "likely" to meet its check by the end of training, extrapolating from 150 of 2,879 steps. It did not.
  - The preregistered manipulation check caught this and made the claim untestable rather than refuted. That is the rule working, but a dose check before queuing would have avoided spending the run.
- **The functional measure's yardstick moved.**
  - G compared head B with the ideal copy of the same model's head A, and the payloads changed head A.
  - So G flattered the geometric payload, whose pin made head A worse. The fix, a fixed yardstick, was added after the results as exploratory analysis.
  - *The lesson:* a measure meant to compare arms must not be defined relative to something the arms change.
- **N11's payload failed its check too, after a dose check meant to prevent exactly that.**
  - The dose check covered 300 of 2,879 steps. Its ratio rose at each checkpoint (0.15, 0.23, 0.27), and I recorded that the payload "holds at every checkpoint". By the end of training the ratio was 0.59, against at most 0.5 required.
  - I read a trend as a level. The manipulation check again did its job, and the primary claim is again untestable.
  - What the runs did show is not small: the payload closed 47% of the gap on new text at no cost to the other reading, against 19% for the geometric pin, which cost it 0.14 nats. But the claim that would weigh that difference got no verdict, and the review counts it as none.
  - *The lesson, sharpened:* a dose check must run the whole training schedule, and a rising trend in it is a warning, not a pass.

### 8. Smaller things

- **Engineering slips.** One bug, a matrix left on the wrong device, crashed two N8 sessions before any model was measured. It was logged.[^10] Earlier Colab sessions stopped for Drive and resume problems, all logged.
- **The limits of five runs.** The exact sign test can never reach p = 0.05 with five runs. The rules therefore used intervals and counts, and intervals with only four degrees of freedom are wide. This was known in advance and stated, but it limits how small an effect any test here could see.

### 9. Can a preregistration bind its own author?

This criticism turns the method's own instrument against it. It has the shape of an old argument in contract law, Benjamin Cardozo's in *Beatty v. Guggenheim Exploration Co.*, 225 N.Y. 380, 387 (1919): a contract clause forbidding oral changes can itself be changed by the parties, because the power that made it can unmake it. The replies used below are the necessity audit's (`NECESSITY_AUDIT.md`), a method that comes from the analysis of legal arguments, which is why a contract case fits here.

- **The criticism.** A preregistration binds only because its author approves it, and the same power of approval can amend it. So it binds nothing. The record has an example: N12's preregistration said the calibration could change only the number of fit steps, and amendment A1 changed a precondition instead.
- **The reply** fixes the condition to a time and a level:
  - amendments come only before the real run whose claims they touch, and the ones that changed a measure, a threshold or an arm (G1, N2, N7, N11) all came before any result at full size;
  - each records what had been seen when it was made, so a reader can judge whether what was seen could have shaped it;
  - after the real run, a change can only add an exploratory analysis, labelled so.
  - N12's A1 is the narrowest kind: it changed a precondition's convergence test, and no claim, threshold, arm, window or measure.
- **The critic's best answer.** The line between before and after is formal. A1 was approved after one calibration seed had shown a value that would favour N12d (S(`fit-single`) = 0.614, below the 0.67 threshold).
- **The next move.**
  - A1 changes only the convergence test of the fitting procedure's must-match control.
  - `fit-single`'s loss can fall below A1's new floor only by reaching the copy. In that case N12d is refuted, so A1 cannot tilt N12d's verdict towards support.
  - Anyone can recompute the precondition under both rules from the saved records.
- **The weakest point.** The rule A1 replaced was stated too strongly from the start, and that was my error. A convergence test should have been written to work at a loss of zero. A preregistration binds only as far as its rules survive their own boundary cases, so the defence depends on writing those rules carefully in the first place.

### 10. An explanation that survived three of its four tests, and a note that had said too much (N12)

- **The conjecture was worded too strongly, and the audit caught it before the run.** The preregistration said the drift since N7 "was the readout's form, not noisy training". The necessity audit, before the run, showed that the design could bound the form's share of the drift but not attribute the whole of it to the form. The run then put the form's share at most 38% of the gap, with training on noisy targets through that form accounting for the rest. Without the audit, "all four supported" would have been read as confirming the stronger wording. As it was, three of four were supported and the stronger wording had already been withdrawn.
- **The geometric claim fell just short, for the second time.** N12b required agreement of at least 0.85 and found 0.79–0.85. N10c had required the same and found 0.79–0.84. Two near misses at the same bar, from heads functionally close to their copies, are a pattern, and a pattern is not an explanation. N12's note offers one, that the random targets leave a geometric trace that function does not show, labelled as a conjecture, with the test that would decide it.
- **A note had said too much.** N10's note stated that head A was "unaffected by noise in head B alone", from a range of validation losses across arms. N12 found a cost of 0.04 nats, and N10's and N11's own records show the same shift in every seed. The range hid it, and nobody looked at the paired differences. A dated correction has been added to N10's note. *The lesson:* a claim of no effect needs the paired differences and their interval, as a claim of an effect does.
- **A slip in the pipeline.** The session that ran N12 had no GitHub token, so the notebook delivered a zip instead of uploading. The records were committed from it, with the run information, and the deviation log says so. The GPU check stopped three sessions on an L4 before any training, as designed.

### 11. A conjecture that survived all three of its tests, and what that is worth (N13)

- **The preregistration's own critic said the primary claim was a test of arithmetic.** It was, and the preregistration said so before the run: N13a tests whether the pipeline implements the chain rule's identity, and its support is knowledge about the code, not the world. What it bought is what N13c means given it, and a measured fact every later paired test needs: two implementations of the same mathematics differ by up to 0.005 nats in Δ_A, while a replicate of the same code differs by nothing.
- **The replicate floor measured the wrong thing, and the design survived it.** Precondition 3 asked how much a replicate of the same training differs, expecting the pipeline's nondeterminism to set a floor. The pipeline is deterministic, and the floor was exactly zero. The audit at drafting had foreseen that boundary and kept an absolute tolerance, so the rule still worked. But the noise that matters for comparing two implementations was measured by N13a's own arms, not by the replicate, and the per-position divergence between those arms (0.09–0.10 nats) was of the same order as between two trainings with different losses, and at the inner softmax six times larger. *The lesson:* a floor should be measured between two implementations of the same mathematics, not two runs of the same code, and models must be compared on aggregates.
- **The stated mechanism was wrong at one time point, and the audit had not asked.** The preregistration said the discount tells kept from replaced targets from the start, by frequency. The calibration's gradient check found the two equal at the start, as they must be, since both kinds of target have the unigram distribution. The necessity audit had covered the identity's dependence on time (reply 5) and not the mechanism's own claim about time. The correction changed no claim or rule and went in the deviation log before any run. It is N11's dose-check error again, a claim fixed to the wrong time, caught earlier this time.
- **What the survival is worth.** Each claim had a failure mode: a broken identity, a clean estimate hidden in the standard head, a gain from pulling less. None occurred, and together they narrow N12's reading: the mixture head's two "effects" are two roles of one form, the discount in training and the mixture at readout. The result is still one noise rate, one scale and N12's seeds, and it says nothing about a head that must estimate the noise rather than being told it.

### 12. A conjecture that did not survive, a control that measured the wrong thing, and a bar explained (N14)

- **The preregistration recorded what the numbers in hand predicted.** Before the run it recorded that they gave the same drop in agreement per nat of clean-target loss along two different axes, so that a straight clean-target ladder would predict the mixture head's agreement and leave no trace for the conjecture. The ladder was measured: it is smooth in every seed to within 0.02, with rung values varying by up to 0.05 across seeds, and the four heads with random targets sit on it. That is the test working as intended: the point estimates fell where the conjecture forbade, within noise the rule could not resolve.
- **The rule returned "inconclusive", below the design's stated limit.** The residual's spread across seeds (0.028) equals the 0.03 margin, so the interval reaches 0.052 from a point estimate of 0.009. The preregistration stated the detectable residual as about 0.06 and said a smaller one would fall short of support; below both margins the rule's outcome is "inconclusive". It is the design's limit, and the point estimates are reported as what they are: on the rival's side.
- **The control measured the wrong thing, and the calibration said so.** N14c compared agreements directly between the reweighted control and `sub20`. The calibration run showed the control's clean-target loss collapsing to 0.81 nats from head A, beyond the ladder's top rung, so that the rule would read a loss of function as the reweighting's geometric effect. The amendment proposed after calibration added the ladder rung needed to read the control against function; the author kept the rule. It is refuted, and the ladder reading, the control 0.10 *above* the curve, is reported beside it as exploratory. *The lesson:* a control must be matched on the axis the claim is about, or the rule must correct for it; a calibration that shows a rule answering a different question calls for an amendment of the rule.
- **A bar explained, within noise.** The ladder crosses 0.85 at a clean-target loss 0.26 nats from head A's; the mixture head's inner reading is 0.34 away. N12b's near miss needs no geometric trace; N10c's cannot be read from here. Open problem 2 is answered in part.
- **What the control adds.** A head can read head A's directions better than its loss predicts: the reweighted head's function fell to near a 1.25% clean head's while its directions stayed a 5% head's. So the ladder is not a law, and "geometry follows function" is a finding about heads trained with random targets, not a rule about heads.

### 13. A design withdrawn after its calibration, and a successor that survived three of its four tests (N15, N16)

- **N15 was withdrawn, not amended.** Its calibration run, at full length on one seed, found the sharpened discounts it was built on not training at all, and its oracle arm, a discount judged by head A, recovering the whole shortfall while removing too little of the random share for its rule to apply. No amendment could have saved the design without changing what it tested. The author withdrew it before any real run, and N16 was preregistered from what the calibration showed, with the calibration recorded as motivation, not evidence.
- **The critic's objection was the obvious one, and the margins answered it.** N16's claims were written after one seed had been seen. The margins were set far from the values in hand: 0.75 against 1.02 for the recovered fraction and 0.5 against 0.15 for the removed share, both from N15's calibration, and −0.30 against the −0.57 implied by N14's control for the judges' difference (N16's own calibration later gave −0.48). The five real runs, on N12's seeds reused as declared, came out far from the margins and near the calibration values: F 1.08, P 0.10, C −0.45, each just outside its real interval.
- **A label that said more than its comparison separated.** N16c compared the same clean targets weighted by two judges, head B's own and head A's, and was labelled "self-reference is the mechanism". Head A differs from head B in at least three ways: it does not learn from its own weights, it is the better predictor, and it judges more sharply. The audit at drafting asked whether the comparison isolates the judge and flagged sharpness; it did not flag accuracy. The verdict stands by its rule, against the rival it was built for; the note reports the attribution as open and proposes a test of the first two.
- **The fourth claim was inconclusive at the stated limit.** The preregistration said N16d would most likely be inconclusive unless the residual was near zero or beyond 0.04; support needed it within about 0.007 of zero, and it was 0.011, with one seed at 0.039.
- **What the exploratory analysis adds.** The four reweighted heads form a two-by-two of judge against random targets. The random targets help under either judge (by 0.40 nats when the head judges itself, 0.14 when head A judges), the opposite of the premise N15 started from; and with head A as judge the head ends slightly closer to head A than the oracle mask does, in every seed, which the calibration entry had recorded as possible before the run. Both are exploratory; the second has a derivation behind it (the random positions carry part of the judge's prediction) and no test.

### 14. Matching the judge's accuracy, and a claim its margins could not reach (N17)

- **The primary claim landed between its margins, on a narrow interval.** A loop-free judge interpolated to head B's own accuracy recovered 0.334 [0.318, 0.349] of the shortfall. That is not the three quarters the conjecture claimed, and not the quarter or less that accuracy alone, the main rival, predicted. The verdict is "some effect, size unclear".
- **The second claim's margins were out of reach, and a precondition said so.** N17b's margins were set in units of the shortfall, on the conjecture's own expectation that a loop-free judge would recover most of it. A fresh reader's check of the draft showed that the support margin could not be reached if the loop-free line was low at the peer's accuracy, and the height condition added to precondition 6b made that case untestable. The line was low (0.31–0.37), so N17b was untestable. A margin relative to the line's height would have stayed reachable (lesson 15).
- **The calibration foresaw both verdicts,** and the entry saying so was committed before the real run. Nothing was changed after it: the preregistration allowed only the rungs to change, and they fell to the default pair.
- **An independent review of the code found what the tests had not.** While the calibration ran, a fresh reviewer compared the code with the preregistration. It found two defects: the analysis pooled N17b's seeds by part of the quantity it tested, which would have favoured support, and a resumed calibration would have stopped at its last step. It also found three gaps in the checks. All were fixed before any calibration value was seen (lesson 16).
- **What the exploratory analysis adds, and what it does not.**
  - The peer, as accurate as head B, recovered 0.04 (a secondary measure). A second self-judging head does nearly as badly as head B judging itself, though it shares head B's trunk, targets and training.
  - At head B's accuracy, the loop-free judges' sharpness and random share matched head B's own at the end of training. They were sharper during it, so the third is not separated from sharpness or the course of training.
  - A first draft of the note split N16's 1.08 into 0.75 for accuracy and 0.33 for the loop. A fresh reader's check showed that head A's point lies outside the rungs and that the third is not separated from the other candidates. The note now reports the 0.75 as description and no longer credits the 0.33 to the loop.

## What stands now, and what would overturn it

These are the current best explanations. They are conjectures that have survived tests, not established facts.

- **The paper's claims are refuted at the scales tested.** The main cause was the training shortcut, not the design (T1). *Would be overturned by:* exact training failing to close the gap on another dataset or at full budget. How the shortcut goes wrong is untested.
- **At this scale, the continuous dynamics do not carry the readings** (G1, G1d, N1, N2). A fixed shift does the control's work as well as the dynamics do, and the readings do not get their own paces. *Would be overturned by:* a design in which the dynamics are given work only they can do, such as V2's separate continuous strands, outperforming a matched model.
- **Readings held at once occupy partly different directions, and forcing them together costs accuracy** (N4). This is measured in loss, so it does not depend on the overlap measure. *Would be overturned by:* the cost vanishing on new seeds. Or, if the cost is a competition for directions, by a cost that stays when nothing is restricted. That is the question N5 could not settle.
- **Readings further ahead depend on more different directions** (N6r, N7, N8c). The ladder held under two measures, and the target-free measure has now been validated on a case with a known answer (N9). *Would be overturned by:* the ladder flattening on new seeds, or under a measure of what the heads compute rather than how sensitive they are.
- **A head trained on noisy targets reads differently from the clean head** (N7, N8, N9; replicated on new seeds in N10, N11 and N12), **and N12 divides the reason in two.** A single softmax on a linear readout cannot represent the mixture a noisy target calls for, and that accounts for at most 38% of the gap (the best single softmax found by direct fitting reached 62%, N12d); training on the noisy targets through that form, which reaches none of the 62%, accounts for the rest. A head whose form can represent the mixture keeps 71% of the function from the same targets (N12a) and most of the directions (0.83, N12b intermediate). *Would be overturned by:* a single-softmax fitting procedure that reaches the copy, or a mixture head that drifts as far as the standard one at another scale or noise rate.
- **The mixture head's advantage is its learning signal** (N13). In training the mixture form is nothing but a discount on each target by the head's own posterior that it is clean, and the pipeline follows that identity (N13a: within 0.005 nats in every seed). The form applied at readout to the standard head doubles its gap (N13b), and a constant discount with the real discount's mean weight gives head B's reading none of the gain (N13c) and, a secondary finding, leaves head B's cost to head A at zero within the measure's noise. *Would be overturned by:* a discount-trained single softmax that drifts from the mixture head's inner softmax at another scale, or a constant-discount head that recovers the gain at another noise rate.
- **Along the axis of random targets, geometry follows function** (N14). Heads trained with 10 to 80% random targets at a fixed 20% of clean ones read head A's directions as a clean-target head with the same clean-target loss does, within 0.02 (residuals 0.008–0.019, every interval including 0; N14a inconclusive by its rule, with the design's detectable residual stated beforehand as 0.06). The 0.85 bar corresponds to a clean-target loss within 0.26 nats of head A's, and the mixture head's inner reading is 0.34 away, so N12b's near miss needs no geometric trace. The relation is not a law: a head trained with the discount's reweighting and no random targets reads 0.10 better than its loss predicts. *Would be overturned by:* a head with random targets found well below the ladder at another noise rate or scale, or a sharper discount that raises function without raising agreement.
- **What holds the self-judged head back is its judge, not the random weight it leaves** (N16). A discount computed from head A's prediction, with nothing else changed, recovered the whole shortfall against the oracle mask (N16a: 108%) while removing a tenth of the random share (N16b), and on clean targets alone head A's judgement cost 0.45 nats less than head B's own (N16c). Whether head A helps by being independent of head B, by being the better predictor or by judging more sharply is not separated. The random targets, in an exploratory comparison, help the reweighted head under either judge. *Would be overturned by:* a head-A-judged discount failing to recover the shortfall at another noise rate or scale. A judge with head B's accuracy that does not depend on head B's own prediction would not overturn it, but would say which of the two differences matters.
- **A judge as accurate as the self-judged head, but with no loop, recovers a third of the shortfall** (N17a: 0.334 [0.318, 0.349], some effect, size unclear), and a second self-judging head of the same accuracy recovers almost none (secondary). What makes up the third, the loop itself, training on clean targets only, the course of training or sharpness during training, is not separated. *Would be overturned by:* a loop-free judge of matched accuracy recovering nothing, or most of the shortfall, on new seeds.

## Problems now open

On Deutsch's view, problems are what knowledge grows from. These are the ones this work has made sharp:
1. Does anchoring what a head predicts preserve its reading better than pinning the directions it reads? N10 and N11 could not say, because in both the functional payload failed its manipulation check. Both payloads were aimed at the mixture, which N12's fits show a single softmax of this state does not reach; whether that is why the checks failed was not tested. With a mixture head the target is reachable, which removes at least that obstacle.
2. *Answered in part by N14, 2026-10-22:* within the design's noise, heads functionally close to their copies stop under 0.85 at a clean-target loss the ladder says is too high. The bar corresponds to about 0.26 nats from head A, and the mixture head's reading is 0.34 away (section 12). What it leaves open: a residual of up to about 0.05 below the clean-target ladder, the interval's upper end, which five seeds could not resolve; N10c's head, whose inner reading was not measured in this form; and why a head trained with the discount's reweighting and no random targets reads better than its loss predicts.
3. *Answered by N13, 2026-10-22:* the discounting does the work, and the form matters only at readout (section 11). What it leaves open: whether a head that must estimate the noise, rather than being told it, keeps the advantage; and why head B's cost to head A tracked its weight rather than the randomness of its targets, yet `sub20`, with clean targets on a fifth of the positions, cost more than the standard head scaled to a fifth. *N16, 2026-10-22:* the shortfall that remains against the oracle mask lies in the discount's judge (section 13); open: whether independence from head B or accuracy is what a judge needs, and what the random targets carry when head A judges them. *N17, 2026-10-26:* a loop-free judge as accurate as head B would recover a third of the shortfall (N17a, some effect, size unclear). Open: what makes up that third, and whether a judge more accurate than head B can be had without a clean head.
4. Is N4's cost of sharing a competition for directions, or a clash of learning signals? (N5 needed about ten runs at full width to tell.)
5. Does the cost of sharing grow along the distance ladder? That would link the geometric finding to function.
6. Do readings that differ in kind, such as a topic label against the next word, follow the same pattern as readings that differ in distance?
7. The project's own question: can a model built with separate, continuous, interacting strands hold more voices than a matched transformer (V2, V3)?

## What we would do differently

1. **Write every check together with the wrong version it must reject.**
2. **Re-calibrate every threshold when the number of passages, the width or the model changes.**
3. **Validate a measure on known transformations before using it as evidence.**
4. **Before calling an after-the-fact explanation "likely", check it against known results,** and name the test that could refute it.
5. **State a rule in a summary only after it has survived a test against its strongest rival.**
6. **Prefer new seeds to reused models** whenever a test can afford them.
7. **Check that a manipulation is strong enough before queuing a run that depends on it,** with a dose across strengths over the whole training schedule, not an extrapolation from a short run. A trend in a short check is a warning, not a pass.
8. **Define every comparison measure against a fixed yardstick** that the arms being compared cannot change.
9. **Before each run, list the design's claims of necessity and test them against the standard replies** (`NECESSITY_AUDIT.md`). Its first use, on N12, found that a rival hypothesis had been understated and that the conjecture's wording claimed more than the design could show. Both were found before the real run.
10. **Write every rule so that it still works at its boundaries:** zero, saturation, and the end of the schedule.
11. **A claim of no effect needs the paired differences and their interval,** as a claim of an effect does. A range across arms can hide a shift that every seed shows.
12. **Measure a floor between two implementations of the same mathematics, not two runs of the same code, and compare models on aggregates.** A deterministic pipeline gives a replicate floor of zero, and per-position differences between models cannot tell a different learning signal from rounding amplified by training, the conjectured cause.
13. **Match a control on the axis the claim is about, or let the rule correct for it.** N14c compared agreements directly between a control and its reference; the control's function collapsed, and the rule read that as the effect under test. The calibration showed it before the run, and the rule was kept. When a calibration shows that a rule will answer a different question from the one asked, amend the rule, not only the measurement that would allow a better reading.
14. **Name everything a comparison changes before naming the claim after one of the changes.** N16c changed the judge from head B to head A, which changes both whether the head judges itself and how well the judge predicts; its label named only the first. The audit asks whether a comparison isolates a cause; it should list every difference between the arms compared, and the label should claim no more than the list allows.
15. **Set a margin on a difference relative to the quantity it is a difference from.** N17b's margins were in units of the shortfall, set on the conjecture's expectation that the loop-free line would be high. When it was low, the support margin was out of reach whatever the mechanism, and only a precondition added after a fresh reader's check turned that into an untestable verdict.
16. **Have the code reviewed against the preregistration, by a reader who did not write it, before the real run.** N17's tests on tiny models and synthetic records passed, and an independent review still found two defects: a pooling of seeds that would have favoured support, and a step that would have stopped a resumed session.

---

## Notes

[^1]: **Popper.** Falsifiability and the rejection of ad hoc rescues: Popper, K. R. (1959). *The Logic of Scientific Discovery*. Hutchinson, chapter 4, where he calls such rescues "conventionalist stratagems". Conjecture and refutation as the growth of knowledge: Popper, K. R. (1963). *Conjectures and Refutations: The Growth of Scientific Knowledge*. Routledge & Kegan Paul. The Duhem–Quine point, that a test refutes a claim together with its auxiliary assumptions, is discussed in both. Popper's answer is that auxiliary assumptions may be changed only if the change is itself testable, and not only to save the claim.

[^2]: **Deutsch.** Good explanations as hard to vary, the reach of explanations, and problems as inevitable and soluble: Deutsch, D. (2011). *The Beginning of Infinity: Explanations That Transform the World*. Allen Lane, chapters 1 and 13.

[^3]: **The replication.**
    - *WikiText-2, eight runs:* steering worked reliably in only 5 of 8. The model was slower, not as fast. The learned scale settled near 0.17, not 0.065. The geometric claim was not borne out by the paper's own measure. The paper's experiments had limits fixed in writing before any replication result was analysed: one small dataset, less than one pass over it, a single seed, steering trained and tested on one prompt, and results drawn from two different architectures. Sources: `../wikitext2-replication/RESULTS.md`, `../wikitext2-replication/KNOWN_LIMITATIONS.md`.
    - *WikiText-103, six runs:* the paper's model minus the ordinary transformer, +1.10 nats, 95% interval [+0.97, +1.23], worse in all 6. Before the main runs, the transformer's learning-rate grid was extended to 0.0024, because its best rate, 0.0012, had been the largest tried. Source: `../wikitext103-training/PREREGISTRATION.md`, `../wikitext103-training/RESULTS.md`.

[^4]: **T0 and T1.**
    - *T0* was untestable for two reasons. A must-match control (a cosine of at least 0.999) failed in 5 of 6 models, at 0.9982 to 0.9990. Noise between dropout masks was far larger than planned. Its exploratory observation, that the shortcut's gradient for the learned scale has the opposite sign to the exact one, has not been tested as a claim.
    - *T1*, adjoint minus exact training: +0.792 nats, 95% interval [+0.661, +0.923], about 95% of the whole gap of +0.834 at 10% of the budget. Rival P (missing weights): +0.020, refuted. C2 (rigidity of a repeated block): +0.007, refuted at this scale. F (the ODE form): +0.035, some effect, size unclear.

    Sources: `../too-linear-conjecture/T0_NOTE.md`, `../too-linear-conjecture/T1_NOTE.md`.

[^5]: **G1, G1c, G1d, N1.**
    - *G1:* curvature ratios between control values were 1.08 to 1.27, against at least 2. Amendment A1 replaced Cohen's d, which untrained models passed at up to 6.0, with this ratio.
    - *G1c:* contraction ratios were 4.1 to 10.2 in five new models, against 0.997 to 1.031 in shuffled-control models.
    - *G1d:* capping contraction removed over 90% of it. The cost difference was −0.002 nats, 95% interval [−0.007, +0.002].
    - *N1:* a fixed offset after the block matched the full model: differences of +0.001 nats, 97.5% intervals [−0.008, +0.010]. Whether the trained model's own control acts as an offset (N1a) was untestable.

    Sources: `G1_NOTE.md`, `G1C_NOTE.md`, `G1D_NOTE.md`, `N1_NOTE.md`.

[^6]: **N2.**
    - Timescale ratios were 1.14 to 1.25, against at least 2.
    - Unpredicted observation: two readings' sensitive subspaces overlapped 0.57 to 0.63, against 0.98 to 0.99 for two copies of one reading.

    Source: `N2_NOTE.md`.

[^7]: **N4 and N5.**
    - *N4:* the difference-of-differences cost of tying the readouts was D = +0.088 nats, 95% interval [+0.078, +0.097].
    - *N5:* at full width, D = +0.030, 97.5% interval [−0.004, +0.065], inconclusive. The standard deviation across seeds was 0.022 to 0.194 at N5's widths, against 0.008 at N4's width.
    - *N4's note* originally said the balance depends on the starting point. A dated correction from N5 records that it does not: two different readings ended at 0.42 to 0.53 from every start. *[Corrected 2026-10-25: this holds for separate readouts. With one shared readout the two readings ended higher, 0.58 to 0.75, and higher the narrower the readout (`N5_NOTE.md`).]*

    Sources: `N4_NOTE.md`, `N5_NOTE.md`.

[^8]: **N6 and N6r.**
    - *N6:* the same-target control needed an overlap of at least 0.95 in 4 of 5, and got it in 3; one missed by 0.0002. The threshold came from N4, which measured on almost twice as many windows (256 against 139).
    - *N6r:* the control was judged by an agreement ratio corrected for measurement noise, on new seeds and test-split windows. Result: o(2) − o(8) = +0.252, 95% interval [+0.219, +0.286].
    - *N6r's own note* records that its prediction had already been seen in N6's exploratory results. So it carries less weight than a fresh prediction.

    Sources: `N6_NOTE.md`, `N6R_NOTE.md`.

[^9]: **N7.**
    - *The design:* a next-token head with 80% of its targets replaced by words drawn at their normal frequencies.
    - *Its overlap with the clean head* was 0.307, against 0.297 for a head reading four ahead and 0.242 for one reading eight ahead. o(noisy) − o(8) = +0.066, 95% interval [+0.045, +0.087], against at least 0.30 required.
    - *Its agreement ratio* was 0.30 to 0.33, against at least 0.85.
    - *Amendment A1* raised the noise from 0.6 to 0.8, before any run, because calibration suggested 0.6 would not make the reading hard enough.

    Source: `N7_NOTE.md`, with its dated correction from N8.

[^10]: **N8.**
    - *The measure:* each head's targets are drawn from its own predictions, so its expectation is the Fisher information. Amendment A1 added a must-differ control, because untrained heads scored about 1 under the new measure.
    - *N8a:* o_F(noisy) − o_F(8) = +0.127, 97.5% interval [+0.105, +0.150]. *N8b:* the agreement ratio was 0.315 to 0.339. *N8c:* o_F(2) − o_F(8) = +0.248, interval [+0.214, +0.281].
    - *Old against new:* the two measures agreed within 0.045 on average in every arm, and within 0.001 for the noisy head. When a head's predictions match its targets' distribution, the two measures have the same expected value: Kunstner, F., Balles, L., & Hennig, P. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. arXiv. https://doi.org/10.48550/arxiv.1905.12558.
    - *The crash:* two sessions stopped before measuring any model, on a matrix left on the CPU.

    Sources: `N8_NOTE.md`, `N8_PREREGISTRATION.md`.

[^11]: **N9.** From each clean head, the ideal noisy copy is built exactly: its prediction is 20% of the clean prediction plus 80% word frequencies, with no training. Amendment A1 doubled the windows to 512, because calibration put the copy's stability within 0.05 of its threshold.
    - *N9a*, the clean head against its copy: 0.955–0.995, at least 0.85 in 5 of 5. The other 15 clean heads scored 0.954–0.991.
    - *N9b*, the trained noisy head against the copy: 0.305–0.349, at most 0.60 in 5 of 5.
    - *Controls:* the zero-noise copy matched at 0.99999 to 1.00000, and the rotated control scored 0.12–0.13.
    - *In prediction space:* the KL divergence from the noisy head to the copy was 0.07 nats per position. On clean targets the noisy head's loss was 6.62, against 6.01 for the copy.

    Sources: `N9_NOTE.md`, `N9_PREREGISTRATION.md`.

[^12]: **Amendments before any real result.** G1 (the measure), N2 (a floor in the wrong units), N4 (an undefined rule), N7 (the noise rate), N8 (a must-differ control), N9 (the number of windows), N11 (the payload on head B's readout only, after the dose check), N12 (a convergence test that works at zero loss, after the calibration run), N14 (A1, a fifth rung of the clean-target ladder, after the calibration run), T0 (made before any real result), and two in the WikiText-103 study (the learning-rate grid, and a safeguard). Each was approved by the author, and each records what had been seen when it was made. One deviation came during a run: N11's D1, which redid all runs on one kind of GPU after a session ended partway. It too was approved, and it records the partial results that had been seen. N15 was withdrawn by the author after its calibration run and before any real run; N16 recorded two readings of its preconditions at build time as clarifications, not amendments; they were put to the author with the report that queued the calibration, and no objection was raised before the real run. Sources: the deviation logs of each preregistration.

[^13]: **Checks before the real run.** What each caught, and how:
    - *G1:* the effect-size measure passed untrained models.
    - *N2:* a floor in the wrong units, on untrained models.
    - *N4:* an undefined rule, found on rereading the plan; the untrained calibration found the claim working as written.
    - *N7:* a noise rate that a loss bound suggested would very likely be too low; the untrained calibration found the controls working.
    - *N8:* a must-match control that untrained heads passed automatically (ratios 0.98 and 1.02).
    - *N9:* the copy's stability margin. In trained models the copy turned out more stable (at least 0.91), so this caution was not needed. *[Corrected 2026-10-25: at least 0.907. N9 measured only with the extra windows, so whether it could have done without them was not tested (`N9_NOTE.md`).]*

    It missed N6's threshold, which was not re-checked for the smaller number of windows, and the short full-size checks before N10 and N11 missed that their payloads would not take hold (section 7). The analytic checks that could not fail at first were N8's, where a uniform sampler scored 0.996, and N9's, where measuring head A instead of its copy scored 0.999. Both were redesigned before any run, so that the wrong version fails (0.72, and 0.66 to 0.80). Sources: `N8_PREREGISTRATION.md`, `N9_PREREGISTRATION.md`.

[^14]: **N10.** New seeds 3901–3905, 35 runs.
    - *Dose arms:* head B's agreement with its ideal copy was 0.812, 0.636 and 0.305 at 20, 50 and 80% noise, ordered in 5 of 5 seeds. At 20% it was 0.79–0.84, so N10c was intermediate. G there was 0.04 nats.
    - *Functional payload:* it reduced its own windows' divergence only to 0.73–0.77 of the unanchored value, against at most 0.5 required, so N10a and N10b were untestable.
    - *Exploratory, after the results:* against a fixed yardstick, the functional payload closed 20% of the gap, the geometric 16%, and the data payload made it 45% worse. The geometric payload cost head A 0.13 nats, the data payload 0.36.

    Sources: `N10_NOTE.md`, `N10_EXPLORATORY.md`, `N10_PREREGISTRATION.md`.

[^15]: **N11.** New seeds 4001–4005, 30 runs on an A100 (deviation D1: a first session on a T4 ended at a usage limit, and all runs were redone on one kind of GPU).
    - *Functional payload,* on head B's readout only (amendment A1). S, the share of the gap closed against a fixed yardstick, was 0.40, 0.45 and 0.47 at λ = 8, 32 and 128, rising in 5 of 5 seeds, so N11b was supported.
    - *Manipulation check:* at λ = 128 the payload-window divergence was 0.585–0.601 of the unanchored arm's, against at most 0.5 required, so N11a and N11c were untestable.
    - *Secondary:*
      - The geometric pin closed 0.19 and cost head A 0.14 nats; the functional payloads cost head A at most 0.005.
      - Head B's agreement with its copy was 0.73 pinned, 0.35 under the strongest functional payload, and 0.32 unanchored.
    - *Hardware, exploratory, 14 matched pairs:* no mean difference above 0.004; losses within 0.024 nats; G within 0.003. S(f128) − S(f32) was −0.0005 on the T4 in both seeds compared, against +0.019 and +0.022 on the A100. Source: `N11_HARDWARE.md`.

    Sources: `N11_NOTE.md`, `N11_EXPLORATORY.md`, `N11_PREREGISTRATION.md`.

[^16]: **N12.** New seeds 4101–4105; 20 runs and 10 frozen fits in one A100 session; measured on 512 windows held out from the last 1% of the training text. S is the share of the gap closed against the ideal copy of the same seed's `n80` head A.
    - *N12a:* S(`n80-mix`) 0.713, 97.5% interval [0.690, 0.736]. *N12c:* S(`sub20`) 0.796 [0.772, 0.820]. *N12d:* S(`fit-single`) 0.616 [0.605, 0.626]; the must-match fit in the mixture form reached 0.997.
    - *N12b:* `n80-mix`'s agreement with its copy 0.79–0.85, in no seed at least 0.85; `sub20` 0.90–0.93; `n80` 0.30–0.32.
    - *Secondary and exploratory:* the fitted single softmax's agreement with the copy was 0.13–0.15, the rotated control's level. Head A's cost against `clean` was +0.040 [+0.033, +0.047] with the standard noisy head and −0.009 [−0.024, +0.007] with the mixture head. The mixture head beat the fitted single softmax by 0.097 [0.079, 0.115], and `sub20` beat the mixture head by 0.083 [0.059, 0.107].
    - *Before the run:* amendment A1 (a convergence test that works at zero loss) and the necessity audit, which bounded what the design could show.

    Sources: `N12_NOTE.md`, `N12_RESULTS.md`, `N12_EXPLORATORY.md`, `N12_PREREGISTRATION.md`.

[^17]: **N13.** N12's seeds 4101–4105 and saved models; 15 new runs in one A100 session, measured on N12's 512 held-out windows. Δ_A is the inner softmax's clean-target loss minus head A's; S is as in N12.
    - *N13a:* |Δ_A(`n80-rw`) − Δ_A(`n80-mix`)| −0.005 to +0.003, within 0.03 in 5 of 5; the replicate `n80-mix-rep` bit-identical to `n80-mix`. *N13b:* S(`n80`, mixture readout) −1.014, 97.5% interval [−1.045, −0.983]. *N13c:* D = +0.050 [+0.036, +0.064], against margins of −0.27 (a quarter of N12's gain of 1.062 nats) and −0.53.
    - *Secondary and exploratory:* under the constant discount, head A −0.051 [−0.066, −0.036] against `n80` and head B's own softmax −0.001 [−0.011, +0.009]; head A against `clean`: `n80-rw` −0.006 [−0.024, +0.012], `n80-w02` −0.011 [−0.026, +0.004]. The discount in `n80-rw` at the end of training: kept 0.58, replaced 0.10; its batch mean 0.19 over the schedule. `n80-rw` and `n80-mix` differ per position by 0.097 (inner softmax) and 0.095 (head A) nats of symmetrised KL; `n80-w02` and `n80` by 0.016 and 0.138.
    - *Before the run:* no amendment; the calibration corrected the stated mechanism's claim about the start of training.

    Sources: `N13_NOTE.md`, `N13_RESULTS.md`, `N13_EXPLORATORY.md`, `N13_CALIBRATION.md`, `N13_PREREGISTRATION.md`.

[^18]: **N14.** N12's seeds 4101–4105 and saved `sub20` and `n80-mix`; 40 new runs in one A100 session (a 40 GB model), measured on N12's 512 held-out windows. a is the inner softmax's agreement with head A; Δ_A its clean-target loss minus head A's; â the linear interpolation of a at a model's Δ_A on the seed's clean-target ladder (`sub20`, `sub10`, `sub05`, `sub025`, `sub0125`, by amendment A1); b = â − a.
    - *N14a:* b(`n80-mix`) +0.009, 97.5% interval [−0.035, +0.052], standard deviation 0.028 (supported: wholly above 0.03; refuted: wholly below 0.01). *N14b:* D(`r10`) < D(`n80-mix`) / 2 in 2 of 5 seeds; ratios 0.03–0.74. *N14c:* a(`sub20`) − a(`sub20-rw`) +0.125 [+0.080, +0.169] (refuted: wholly above 0.06).
    - *The ladder,* means over seeds, (Δ_A, a): (0.16, 0.90), (0.28, 0.84), (0.43, 0.78), (0.61, 0.71), (0.82, 0.64); leave-one-out residuals 0.004–0.005 on average, 0.021 at most. It crosses 0.85 at Δ_A 0.255 [0.225, 0.284]; `n80-mix` is at 0.338, 0.084 [0.055, 0.112] beyond.
    - *Secondary and exploratory:* residuals by dose (95%) +0.019 [−0.006, +0.044], +0.008 [−0.023, +0.040], +0.016 [−0.019, +0.051], +0.009 [−0.026, +0.043]; pooled +0.013 [−0.011, +0.037]. The trace 0.040, 0.042, 0.069, 0.091 and the functional cost 0.042, 0.069, 0.107, 0.175 at doses 0.1–0.8, against the random targets' share of the gradient weight at the end, 0.14, 0.22, 0.31, 0.42. The control: Δ_A 0.736 [0.711, 0.760], a 0.772, residual −0.104 [−0.138, −0.070]. Head A against `clean`: `sub20` +0.020 [+0.002, +0.039], `n80-mix` −0.009 [−0.024, +0.007], `sub025` −0.018 [−0.034, −0.002], `sub0125` −0.020 [−0.037, −0.003].
    - *Before the run:* the calibration run (seed 3001) showed the control's function collapsing; amendment A1 added the fifth rung; the N14c rule was kept.

    Sources: `N14_NOTE.md`, `N14_RESULTS.md`, `N14_EXPLORATORY.md`, `N14_CALIBRATION.md`, `N14_PREREGISTRATION.md`.

[^19]: **N16.** N12's seeds 4101–4105 and saved models, with N13's `n80-rw` (here `rw-b1`) and N14's `sub20-rw` and clean-target ladder; 10 new runs in one A100 session, measured on N12's 512 held-out windows. `rw-oracle`: `rw-b1` with its weight computed from head A's prediction, detached. `sub20-oracle`: `sub20`'s clean targets weighted by head A's judgement, no random targets. G = Δ_A(`rw-b1`) − Δ_A(`sub20`), 0.165–0.179; F = (Δ_A(`rw-b1`) − Δ_A(`rw-oracle`)) / G; P = 1 − s(`rw-oracle`) / s(`rw-b1`), s the random share of the summed weight at the end of training; C = Δ_A(`sub20-oracle`) − Δ_A(`sub20-rw`).
    - *N16a:* F 1.081, 98.75% interval [1.043, 1.119] (supported: wholly above 0.75). *N16b:* P 0.103 [0.065, 0.140] (supported: wholly below 0.5). *N16c:* C −0.448 [−0.474, −0.423] (supported: wholly below −0.30). *N16d:* b(`rw-oracle`) +0.011 [−0.022, +0.043], standard deviation 0.017 (supported: wholly within ±0.04; refuted: wholly outside).
    - *Exploratory (95%):* the random targets' effect −0.399 [−0.422, −0.375] nats under head B's own judgement, −0.138 [−0.145, −0.131] under head A's, interaction +0.261 [+0.243, +0.278]; `rw-oracle` below `sub20` by 0.014 [0.010, 0.018], in 5 of 5 seeds.
    - *Before the run:* N15 withdrawn after its calibration; two readings of N16's preconditions fixed at build time; the calibration's rule dropped both exploratory warm-up arms.

    Sources: `N16_NOTE.md`, `N16_RESULTS.md`, `N16_EXPLORATORY.md`, `N16_CALIBRATION.md`, `N16_PREREGISTRATION.md`, `N15_PREREGISTRATION.md`, `N15_CALIBRATION.md`.

[^20]: **N17.** N12's seeds 4101–4105 and saved models, with N13's `n80-rw` (here `rw-b1`), N16's `rw-oracle` and N14's clean-target ladder. 15 new runs in one A100 session, measured on N12's 512 held-out windows. Head C reads the trunk detached and judges head B. In `rw-peer` it is trained as `rw-b1`'s head B is; in `rw-j10` and `rw-j05`, at full weight on the clean targets at N14's nested 10% and 5% positions. x is the judge's Δ_A, F the recovered fraction, F* the line through the two rungs' (x, F) at head B's own x, and d = F(`rw-peer`) minus the line at the peer's x.
    - *N17a:* F* 0.334, 97.5% interval [0.318, 0.349], standard deviation 0.010 (support needed wholly above 0.75; refutation wholly below 0.25).
    - *N17b:* untestable. Precondition 6b needed the line at the peer's x to be at least 0.75 in 4 of 5 seeds; it was 0.31–0.37.
    - *By arm, (x, F):* `rw-oracle` (0, 1.081); `rw-j10` (0.288, 0.449); `rw-b1` (0.337, 0); `rw-peer` (0.339, 0.041); `rw-j05` (0.429, 0.122).
    - *Secondary and exploratory (95%):* F(`rw-peer`) 0.041 [0.003, 0.080]; d −0.290 [−0.315, −0.265], description only. At head B's accuracy, sharpness 5.64 and share 0.421 at the last logged step against head B's own 5.64 and 0.422, but sharper by 0.16–0.18 at steps 500–1000.
    - *Before the run:* seven readings fixed at build time and five fixes after an independent code review, none changing a claim, margin or rule; the calibration chose the default rungs, and its entry said N17a would most likely land between its margins and N17b would most likely be untestable.

    Sources: `N17_NOTE.md`, `N17_RESULTS.md`, `N17_CALIBRATION.md`, `N17_PREREGISTRATION.md`.
