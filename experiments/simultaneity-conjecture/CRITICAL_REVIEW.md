# What we guessed, how we tried to break it, and what survived

Date: 2026-10-10, updated 2026-10-18 with N10, 2026-10-20 with N11, and 2026-10-21 with section 9, lessons 9–10 and corrections from a fresh reader's audit against the notes. A review of all the work in this repository, from the replication of the paper through N11, in the manner of Karl Popper and David Deutsch. As in `PLAIN_SUMMARY.md`, the main text avoids mathematics, and the details are in numbered notes at the end. Every number comes from a result note in this repository; the notes say which.

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

**Out of the 26 verdicts in this ledger:**
- 9 refuted, or mostly refuted;
- 9 supported;
- 2 inconclusive or intermediate;
- 6 untestable.

Counted per preregistered claim, including the secondary ones the ledger leaves out (N1a, N2b and N6's N6b, each untestable; N4a, supported; N5b, inconclusive), the untestable verdicts number at least nine.

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

- **Plans fixed in advance.** Every test had its claim, thresholds and decision rules written and approved by the author before any run. Every change made after approval went back for approval with the reason written down, and was recorded in a deviation log. There have been eleven such amendments, counting N12's, each made before the real run it concerns.[^12]
- **Checking that tests can fail, on models that know nothing.** Before every run, the measure was tried on untrained models, where a true claim should fail.
  - Checks before the real run caught problems in six tests: G1, N2, N4, N7, N8 and N9. Three of those catches came from the untrained models; the others came from a loss bound, a reread of the plan, and a caution that turned out not to be needed.[^13]
  - In G1, the original measure would have called an untrained model a success.
  - In N8, the planned control would have passed automatically.[^13]
- **Controls at both ends.** Most tests from N4 on had a control that must match, such as two copies of the same reading, and one that must differ. A result counted only if both behaved.
- **Refutations kept.** No claim refuted here was later reinstated, and no threshold was moved after a result was seen. Some exploratory observations became new claims: the contraction in G1, the directions in N2, the ladder in N6. Each was then tested as a new conjecture, on new seeds and new data, and the old verdict stayed as it was.
- **Mistakes in the notes corrected openly.** When a later test showed that an earlier note had said too much, a dated correction was added beside the original text, which was left in place. This happened four times: in the notes of N4, N6r (twice) and N7. The plain summary's sentence stating the rule was replaced, and the summary now says that the rule did not survive its test. On 2026-10-21 a fresh reader, given only the summaries and the notes, audited both summaries against the notes. It found three substantive problems (N1's result stated as what the model does rather than what suffices; N11's share of the gap presented as a share of the whole cost of scrambling; section 9's statement of the amendment rule contradicting the record) and seventeen minor ones, most of them modal upgrades. All were corrected in one change, and the audit is now part of the method (`NECESSITY_AUDIT.md`).
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

This criticism turns the method's own instrument against it. It has the shape of Benjamin Cardozo's argument in *Beatty v. Guggenheim Exploration Co.*, 225 N.Y. 380, 387 (1919): a contract clause forbidding oral changes can itself be changed by the parties, because the power that made it can unmake it. The author's legal-research skill *transcendental-counterplay* classifies that argument and verifies the passage, and it supplies the replies used below (`NECESSITY_AUDIT.md`).

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

## What stands now, and what would overturn it

These are the current best explanations. They are conjectures that have survived tests, not established facts.

- **The paper's claims are refuted at the scales tested.** The main cause was the training shortcut, not the design (T1). *Would be overturned by:* exact training failing to close the gap on another dataset or at full budget. How the shortcut goes wrong is untested.
- **At this scale, the continuous dynamics do not carry the readings** (G1, G1d, N1, N2). A fixed shift does the control's work as well as the dynamics do, and the readings do not get their own paces. *Would be overturned by:* a design in which the dynamics are given work only they can do, such as V2's separate continuous strands, outperforming a matched model.
- **Readings held at once occupy partly different directions, and forcing them together costs accuracy** (N4). This is measured in loss, so it does not depend on the overlap measure. *Would be overturned by:* the cost vanishing on new seeds. Or, if the cost is a competition for directions, by a cost that stays when nothing is restricted. That is the question N5 could not settle.
- **Readings further ahead depend on more different directions** (N6r, N7, N8c). The ladder held under two measures, and the target-free measure has now been validated on a case with a known answer (N9). *Would be overturned by:* the ladder flattening on new seeds, or under a measure of what the heads compute rather than how sensitive they are.
- **A head trained on noisy targets reads differently from the clean head** (N7, N8, N9; replicated on new seeds in N10 and N11). It is not the ideal noisy copy of the clean reading: it reads different directions, and reads the context less well. The measure can recognise the ideal copy, so this is a property of training, not of the instrument. *Would be overturned by:* noisy heads coming close to their ideal copies at another scale or noise rate, or under a head whose form can represent the copy (N12 tests one).

## Problems now open

On Deutsch's view, problems are what knowledge grows from. These are the ones this work has made sharp:
1. Does anchoring what a head predicts preserve its reading better than pinning the directions it reads? N10 and N11 could not say, because in both the functional payload failed its manipulation check. N11's secondary measures point one way: the functional payload restored function without restoring geometry, and the geometric pin did the reverse. Two things would sharpen the question:
   - finding what limits the payload. One conjecture is the *softmax bottleneck*: the noisy copy is a mixture that a single softmax readout cannot represent;
   - anchoring the clean reading instead of the noisy copy.
2. Is N4's cost of sharing a competition for directions, or a clash of learning signals? (N5 needed about ten runs at full width to tell.)
3. Does the cost of sharing grow along the distance ladder? That would link the geometric finding to function.
4. Do readings that differ in kind, such as a topic label against the next word, follow the same pattern as readings that differ in distance?
5. The project's own question: can a model built with separate, continuous, interacting strands hold more voices than a matched transformer (V2, V3)?

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
    - *N4's note* originally said the balance depends on the starting point. A dated correction from N5 records that it does not: two different readings ended at 0.42 to 0.53 from every start.

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

[^12]: **Amendments before any real result.** G1 (the measure), N2 (a floor in the wrong units), N4 (an undefined rule), N7 (the noise rate), N8 (a must-differ control), N9 (the number of windows), N11 (the payload on head B's readout only, after the dose check), N12 (a convergence test that works at zero loss, after the calibration run), T0 (made before any real result), and two in the WikiText-103 study (the learning-rate grid, and a safeguard). Each was approved by the author, and each records what had been seen when it was made. One deviation came during a run: N11's D1, which redid all runs on one kind of GPU after a session ended partway. It too was approved, and it records the partial results that had been seen. Sources: the deviation logs of each preregistration.

[^13]: **Checks before the real run.** What each caught, and how:
    - *G1:* the effect-size measure passed untrained models.
    - *N2:* a floor in the wrong units, on untrained models.
    - *N4:* an undefined rule, found on rereading the plan; the untrained calibration found the claim working as written.
    - *N7:* a noise rate that a loss bound suggested would very likely be too low; the untrained calibration found the controls working.
    - *N8:* a must-match control that untrained heads passed automatically (ratios 0.98 and 1.02).
    - *N9:* the copy's stability margin. In trained models the copy turned out more stable (at least 0.91), so this caution was not needed.

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
