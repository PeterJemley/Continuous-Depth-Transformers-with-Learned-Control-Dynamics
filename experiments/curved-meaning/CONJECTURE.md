# Conjecture note: meanings as curved sets, criticised by a protected record

**Date:** 2026-11-11.
- Revised the same day after a first fresh reader's check, which made 24 findings.
- Revised on 2026-11-13 with the author's answers on compression and on which data are mission-critical.
- Revised again the same day after a second fresh reader's check, which made 30 findings.
- Revised on 2026-11-15 with the author's answers on lossless compression and on the tension in the design, which add C5.

**Status:** conjecture, untested. Nothing here has been run. The tests are a sketch: the design is not ready for preregistration.

**Authorship.** The ideas are the author's (Peter Jemley). Claude drafted this note from their discussion. Wherever an idea, reading or inference is Claude's, it is marked as Claude's.

## Glossary

### The author's design
- **Record:** an item of original data, explicit (in words), kept unchanged.
- **Mission-critical:** the class of data that is kept as records. Which data belong to it is decided per use case.
- **Micro container:** where the compressed records are held.
- **Payload:** the author's term for the records in the micro container.
  - The repository's N10 and N11 notes use "payload" for something else, an extra training signal. To avoid the clash, this note calls theirs an **anchoring signal**.
- **Payload corruption device:** the author's mechanism for protecting the payload. Its three roles are under "The design as clarified".

### Collapse
- **Generated data:** text written by a model, not by people.
- **Curse of recursion, or collapse:** the damage a model suffers when it is trained, generation after generation, on its predecessors' generated data. The term is Shumailov et al.'s (2023).
- **Generation 0:** the first model, trained on original data only.
- **Sampling at temperature 1 with no truncation:** drawing each next word from the model's own probabilities exactly, without cutting off unlikely words.

### Models, their parts and their measures
- **The project's model:** a transformer whose middle layers are replaced by a continuous-depth block. The block's output, the **state**, is a list of 256 numbers for each token (a word or a piece of a word).
- **Head:** a readout that turns the state into a prediction of the next token. It is not an attention head.
  - **Head A** predicts the genuine next token.
  - **Head B** is trained on answers a share of which were scrambled, meaning replaced at random. In N10 the share was 0, 20, 50 or 80%; from N11 on it was 80%.
- **Readout width:** the number of directions a head's readout uses, 64 in most of these tests.
- **Reading:** what a head extracts from the state.
- **Curved (hyperbolic) head:** a head that scores each candidate token by a distance in hyperbolic space. A **flat head of the same form** scores by a distance in ordinary space, and is the same head at zero curvature.
- **Nats:** the unit of the prediction loss used throughout. Lower is better.
- **Seed noise:** how much a result varies between runs that differ only in their random starting values.
- **Manipulation check:** a check that an intervention did what it was meant to, before its effect is judged.
- **Dose check:** a short trial of an intervention's strength, run before the real test.

### Judges (in N16 to N18)
A judge estimates, for each of head B's answers, the chance that it is genuine. Head B's training weights each answer by that estimate.
- **Shortfall:** how far head B, judging itself, falls short of a head trained on the genuine answers alone.
- **Peer:** a separate judge trained on the scrambled answers as head B is, weighing its own training by its own predictions.
- **Loop-free twin:** a further head trained at full weight on the same genuine answers as a judge, whose estimates weigh that judge's own training. It judges nothing itself.

### Where the earlier results are
- The notes cited as N10 to N18 are in `experiments/simultaneity-conjecture/`, for example `N18_NOTE.md`.
- The review is `CRITICAL_REVIEW.md` in the same folder.
- G1 is an earlier test there, described in `G1_NOTE.md`.

## Guiding principles

These shape what is tested. They are philosophical arguments, held as guides, not as premises the tests rely on.

- **Learning is error correction.** In Popper's and Deutsch's sense, knowledge grows by conjecture and criticism. A conjecture is held, experience criticises it, and it changes when it is shown wrong.
- **Knowledge comes in layers.** This is from a transcript the author supplied; its source is to be added.
  - Explicit knowledge is stored in words ("lions are dangerous").
  - Its meaning is inexplicit: "stored in configurations of neurons, which are changeable", at a level above the physical substrate.
  - Knowledge fixed at the substrate level, like an old light switch's wiring, cannot be corrected, only replaced.
- **Nothing beyond explanation.** Deutsch, quoted as the author supplied it (source to be added): "there can't be anything beyond explanation. Explanatory universality rests on Turing universality, and that rests on physics."
  - *Claude's reading:* in the Turing sense, which assumes unbounded memory and time, a curved or folded design can compute nothing a flat one cannot.
  - At a fixed size, what a model can represent or learn can still differ, and that is all C1 below is about. So the principle does not rule C1 out.
- **Self-improvement is possible but unreliable.** Deutsch, as supplied: an AGI interested in improving itself "could do so — not reliably, any more than humans can."
  - *Claude's inference, not Deutsch's:* where self-criticism is unreliable, criticism from outside the thing being corrected may help.
  - N18, below, is one small case consistent with that inference. It does not establish it.

## The question

### As the author put it
- Model language meanings as curved or folded sets instead of linearly, with equations that parse language as such sets.
- The aim is to go beyond current AI, and to use far less compute, by fixing the relationships between ideas held in memory.
- The approach follows the chain "Compression > Payload/Payload Corruption Device > Micro Container", to mitigate the curse of recursion.
- Shumailov et al. (2023): "To make sure that learning is sustained over a long time period, one needs to make sure that access to the original data source is preserved and that additional data not generated by LLMs remain available over time."

### Claude's restatement, as separate claims each able to fail on its own

1. **Representation.** A model's computations on meanings are better described by curved or folded geometry than by flat subspaces.
   - This is not a claim about where meanings sit: a curved set can sit inside a flat space, as a sphere does in three dimensions.
   - It is a claim about which distances and operations fit how meanings relate.
   - No test proposed here measures it directly.
2. **Efficiency.** A head with curved geometry built in reaches a flat head's performance at a smaller width.
   - *This narrows the author's claim.* The author tied the compute saving to "fixing the relationships between ideas held in memory", not only to curvature.
   - That broader claim is not tested here. Testing it needs a definition of what is fixed, and where.
3. **Robustness.** A small, protected record of mission-critical data, used as a critic, prevents most of the damage collapse does to data of that kind. The author's word was "mitigate"; "prevents most" is Claude's sharpening, so that a test can fail.
4. **Correctability.** Protecting the records does not freeze errors in what lies outside them.

**"Folded" is not yet defined in a form a test could use.** In Palmer's chapter (2022), a folded set is an attractor that a flow stretches and folds back on itself. What the analogue would be for meanings is open, and this note tests only curvature.

## The design as clarified

### Three layers

1. **Mission-critical explicit records.**
   - *The author's:*
     - Data are classified first. Only data classified as mission-critical are compressed and kept, as original records, in the micro container, where they are held unchanged.
     - Which types count is decided per use case, in context, by weighing that case's risks. The design does not fix a list.
     - Changes to mission-critical knowledge are checked against the records, to stop the curse of recursion. This applies to mission-critical data only.
   - *Claude's, which the author accepted:* the records' role is to criticise, not to be the knowledge.
   - *The author's, 2026-11-15:* the compression is lossless, so the exact records can always be recovered. It saves storage and changes nothing the critic sees, so the tests below do not test it.
2. **The inexplicit layer** (the author's).
   - It is the curved or folded geometry of meanings. It stays changeable and learns by being criticised.
   - The records stay fixed; what they are taken to mean, and how they relate to other ideas, can change.
3. **New experience** (the author's).
   - It is data the model did not generate.
   - It criticises and extends the inexplicit layer, including what the model learns about the setting of an encounter.

### The payload corruption device

Claude proposed these three roles, and the author confirmed all three:
- **A detector:** it notices when the model has drifted from what the records say.
- **A deliberate corruption:** known damage introduced on purpose, to show that the detector catches damage known to be there.
- **A marker:** it tags generated data so that it cannot pass as original data.

### A tension in the design (Claude's observation)

The most effective protection known, Son's (2026), does what the design says should not happen.
- Son's method keeps the model's predictions on the record's sequences close to generation 0's. It holds still what the model takes the records to mean, not only the records' words.
- The design keeps the records fixed and lets their meanings change.

The form of protection that fixes only the records' words is to train on them as targets. That fits the design better, but Son found it much weaker (see "The literature", below).

So whether any criticism can protect as well as Son's while leaving the records' meanings correctable is an open question. Interpretation I3 and test 1 are one step towards answering it.

### The author's answer: meaning tied to purpose and time, reviewed for future cases

The author, 2026-11-15: the choice is not either-or. A record keeps "specific meaning(s) for specified purpose(s) at the time they were specified, but review for future cases". That preserves the original mission-critical data and still lets meanings change.

*Claude's reading of how this would work in a model:*
- **Held still between reviews.** For its specified purposes, a record's meaning is held still as of the time it was specified. That is what the critic does, with generation 0 standing for the time of specification. It protects against change that nobody reviewed, which is what collapse is.
- **Changeable through review.** For future cases, the meaning is reviewed. A review that accepts a change makes the reviewed model the critic's new reference from then on, and the earlier reference is kept on record.
  - This repository already works the same way: a dated correction is added beside the original text, which stays in place.
- **The review must not be the model judging itself.** In N18, a judge that weighted its own training by its own predictions did harm. So a review needs criticism from outside the model, either genuine data the model did not generate or a person. This is Claude's inference.
- **Still open:** what triggers a review, what it checks, and how a purpose is specified so that a model can tell one purpose's context from another's.

C5, below, puts this to a test.

## Interpretations

These are four ways the design could be read, each with what it predicts and the test that would decide it.

| Interpretation | Where it would show in a model | What it predicts | Tested by |
|---|---|---|---|
| I1. Curvature as a fixed-size gain | A curved head against the same head at zero curvature, across widths | The curved head reaches the flat head's loss at a smaller width | Test 3 |
| I2. Fixed or learned curvature (the switch against the neurons) | Curvature set in advance against curvature trained | Descriptive only: no claim is made | Test 3, secondary arm |
| I3. Judged through generation 0's predictions (critic) or trained on the record's own words (store) | The same weighted term on the same record, with generation 0's predictions or the record's next tokens as its targets | The critic protects more | Test 1 |
| I4. Mission-critical against other data | An error inside the protected record against one outside it | Errors outside the record stay correctable; errors inside it are corrected only through review | Test 2 |

**I3 is not a comparison at equal information.** The critic also keeps generation 0, or its stored predictions on the record, and those carry far more than the record's words do. Any result compares the two forms, not two records of equal size.

**Without review, that errors inside the record persist is not a test.** The critic holds the model's predictions on the record by construction, so that arm is used below only as a manipulation check. With a review, correction inside the record becomes C5's test.

## What the evidence already says

### This repository

These are small models (about 30 million parameters, or 23 to 43 million with two heads), on one dataset, at 10% of the full training budget, with one learning rate. Each note gives its own details.

**Anchoring signals (N10, N11).**
- *What failed:* the functional anchoring signals failed their manipulation checks, through design errors.
  - N10's was too weak.
  - N11's dose check stopped too early.
  - So N10a, N10b, N11a and N11c are *untestable*, and the question they asked is open.
- *What took hold:* the geometric pins (overlap 0.998 and 0.997, where 1 means the two readouts use the same directions).
- *The other claims:* N10d and N11b were *supported*, and N10c was *intermediate*.
  - N11b's gains flattened: their last step was no larger than the effect of changing the kind of GPU.
- *Secondary:* N11's functional signal closed 40–47% of the gap in head B's predictions, the distance between head B and its ideal noisy copy.
  - It came at no cost to head A, but under N11's amendment A1 it trained only head B's readout. So head A was out of its reach by construction.
- *Exploratory:* N10's data signal trained head B on the genuine answers of 64 windows while head B's own objective was the scrambled answers, and it backfired.
  - Head B memorised those windows: its loss on them fell to 0.003 nats.
  - Its note: "Repeated clean answers on a few passages taught the shared network those passages, not the reading."
- *Claude's reading, for I3:* a few stored passages trained on as targets can be memorised instead of protecting anything. A store arm under collapse would not have N10's conflict with head B's own objective, so the reading may not carry over.

**An outside judge (N16, N17).**
- *N16:* weighting head B's answers by head A's judgement recovered all of head B's shortfall (108%, *supported*).
  - As a judge, head A differs from head B in at least three ways. It does not learn from the weights it sets, though it shares the trunk with head B. It predicts better. And its judgement is sharper.
  - N16 does not separate which of these matters.
- *N17:* a judge with no loop, interpolated to head B's own accuracy, recovered a third of the shortfall (0.334, 97.5% interval [0.318, 0.349], *some effect, size unclear*).
- *Both:* these judges need genuine answers marked as such, so they are diagnostics, not methods.

**A self-weighting loop (N18).**
- *The setting:* a judge was trained on genuine answers that it was told were genuine. Its training weights came from its own predictions.
- *The result:* it cost head B about twice the peer's gap, beyond what the same weighting cost when a loop-free twin supplied the weights. R = 2.07, 95% interval [1.85, 2.29] (*supported*).
  - The peer's gap here is how much further behind head B ends when the peer judges it than when a judge trained at full weight on the genuine answers does.
- *What qualifies it:*
  - About 1.6 of the 2.07 is what the judge's loss of accuracy predicts (exploratory).
  - The weighting cost 0.86 of the peer's gap even without a loop (secondary).
  - Adding the scrambled answers to the self-weighting judge's training helped head B (description). Whether the peer's own gap is the loop's is not shown.
  - Self-reference is not separated from the weights' effect on the weighter.
  - These judges need genuine answers marked as such, so they are diagnostics, not methods.
- *Claude's reading:* this is a loop in which a model's training is steered by its own predictions, which is one ingredient of the curse of recursion. The analogy fails in two places.
  - N18 has no generated data and no generations.
  - Its own note compares it to confirmation bias in self-training (Arazo et al., 2020), not to collapse.

**Lesson 8 of the critical review:** "Define every comparison measure against a fixed yardstick that the arms being compared cannot change."

**The geometric measures summarise each reading as a linear subspace:** the 16 directions of the block's 256-number output that the reading's loss is most sensitive to.
- This is a summary of sensitivity, not an assumption that meanings are flat.
- The loss measures do not use it.
- G1 measured "curvature" in another sense, the bending of the block's trajectories through depth. That is unrelated to the curvature of a head here.

### The literature

**Curved geometry**
- **Curved embeddings suit hierarchy.** Hyperbolic embeddings outperform flat ones on data with latent hierarchies (Nickel & Kiela, 2017).
  - In their taxonomy experiments, against translational embeddings, they did better "while using an embedding that is smaller by an order of magnitude".
- **Curved networks gain most on trees** (Ganea et al., 2018).
  - Their prefix task asks whether one sentence is a prefix of another; added noise makes its structure less tree-like. On its most tree-like version, curved recurrent networks cut the error by a factor of 3.35 against flat ones.
  - On textual entailment (SNLI) they "perform similarly as with their Euclidean variants".
  - "As soon as the underlying structure diverges more and more from a tree, the accuracy gap decreases."
  - Their own conjecture is that gains are larger "when the underlying data structure is closer to a tree".
  - *Claude's reading:* this favours rival 3, below, for plain text.
- **Some curved networks are smaller for the same result.** One family "can outperform Euclidean baselines with fewer parameters" (Chen et al., 2022).

**Collapse, and what prevents it**
- **What collapse loses.**
  - Training on generated content makes "tails of the original content distribution disappear" (Shumailov et al., 2024; the paper carries an author correction, 2025).
  - It has also been described as "a contraction of distributional variance and representational quality" (Gu et al., 2026).
- **The closest prior art to C2 and I3 is Son (2026).** N10's preregistration already cites this preprint and judged it "suggestive at most". It used one 70-million-parameter model and one dataset. What follows is from its abstract and text, retrieved through Scite.
  - *The critic:* a penalty on the divergence from "the frozen generation-0 model's next-token distribution", on 64 real sequences.
    - It prevented 99.6% of the damage in one generation, at the stronger of two penalty weights (97.9% at the weaker).
    - It prevented 95% across six generations.
  - *The store:* "mixing the same 32k tokens into training" prevented 53.5% in one generation.
    - Its residual damage was 0.680 nats, against 0.029 for the critic at the weaker weight.
    - Son reports that ratio as "at least 23-fold".
  - *The geometry:* pinning the representation's geometry prevented about 12% (at most 17% when a whole layer was pinned).
  - *The caveats:*
    - In the space Son measured, "collapse is not primarily a loss of tails". That differs from Shumailov et al.'s finding on outputs.
    - Fine-tuning on fresh real data also moved Son's early signal, a drift in a small subspace of the model's internal state that Son calls J-space. So that signal is "not specific to collapse".
    - Son's paper says its anchor "is a form of self-distillation and is not our contribution".
- **Other mitigations, known from titles only.**
  - One paper's title says accumulating real and synthetic data breaks the curse of recursion (Gerstgrasser et al., 2024).
  - Another's says self-verification provably prevents collapse (Fu et al., 2025).
  - Neither has been read. Whether their methods use criticism from outside the model is not known here.

**Marking generated data**
- Watermarking aims to distinguish model-generated from human text. But "human edits on LLM-generated text dilutes watermark signals" (Li et al., 2025).
- One paper's title reports that watermarks can be stolen (Jovanović et al., 2024).
- *Claude's reading:* the marker role would inherit these weaknesses.

## Conjectures

### C1. At a fixed width, a curved head is cheaper for meanings with hierarchical structure

- *Scope:* heads only. Most of the project's parameters are in its token tables, so a narrower head saves little of the whole model. A whole-model claim would need a curved trunk, and is deferred.
- *Mechanism:* hyperbolic space has room that grows exponentially outward, like a branching tree of meanings. Flat space has room that grows only polynomially.
- *C1a (hierarchy):* across readout widths, the curved head's held-out loss lies below the same head's at zero curvature at small widths. The gap grows with how tree-like the data is.
- *C1b (the author's broader claim, language in general):* the same holds, to a smaller degree, on plain next-token text.
- *They forbid:*
  - C1a forbids no gap at small widths on tree-like data, and forbids a gap that does not grow with tree-likeness.
  - C1b forbids no gap on plain text.

### C2. A small record of one type, used as critic, prevents most of the damage collapse does to that type

- *Mechanism:* a model trained on its predecessors' outputs is criticised only by those outputs. A record it did not generate restores criticism from outside.
- *What "critic" means here:* Claude's way of putting the design into practice, following Son. During training, a weighted penalty on how far the model's predictions on the record's sequences have drifted from generation 0's. The record's sequences are never used as training targets.
- *The measure:* S, the share of the damage prevented on held-out text of the record's type:

  S = (loss under no protection − loss under the critic) / (loss under no protection − loss with no generated data)

- *Consequence:* S is large. Its thresholds, the record size, the number of generations and the penalty's weight are each fixed once in the preregistration. Other sizes and generations are reported as secondary.
- *It forbids:* S's interval lying wholly below the preregistered lower threshold.
- *Secondary (Claude's extension, a replication of Son):* a record drawn at random protects held-out text in general.

### C3. Protecting the records does not freeze errors outside them

- *Mechanism:* the critic constrains the model only on the record's sequences. Elsewhere, meanings stay changeable, and new experience can correct them.
- *The risk:* the constraint acts through shared parameters, so it might spread.
- *Consequence:* a wrong relationship learned by generation 0, lying outside the record, is corrected by later genuine data about as fast under the critic as without it.
- *It forbids:* the critic arm correcting such an error by less than a set fraction of the unprotected arm's correction.

### C4. The detector tells damage from ordinary learning

This is the device's detector role, a claim of its own. It does not gate C2.
- *The detector:* the divergence from generation 0 on the record.
- *The test:* it must flag deliberately damaged models at a hit rate, and undamaged models at a false-alarm rate, both fixed in advance.
  - A damaged model is trained for a fixed number of steps on generated text.
  - An undamaged model is trained for the same number of steps on fresh genuine text.
- *The risk:* Son found that fresh real data also moved an early signal, so false alarms are a real possibility.

### C5. A review judged on outside data lets errors inside the record be corrected, while keeping the protection

This is the author's resolution of the tension, put in testable form by Claude.
- *What a review is here:* at a fixed point in training, the current model is put forward as the critic's new reference.
  - It is accepted only if it does better than the current reference on a review set: held-out genuine text for the record's purpose, which the model is never trained on.
  - If accepted, it becomes the reference from then on.
- *Mechanism:* between reviews, the critic blocks change that nobody reviewed. A review admits change that outside data support.
- *Consequences:*
  - After a review, an error inside the record is corrected by at least a set fraction of the correction seen with no critic.
  - Under collapse, a critic with reviews keeps a share of the damage prevented (S, defined under C2) within a set margin of the critic without reviews, because a review judged on genuine text should refuse a collapsed model.
- *It forbids:*
  - no correction inside the record after an accepted review;
  - reviews letting collapse back in, so that S falls by more than the margin.
- *Mechanism check:* a review judged by the model itself, for example by its own loss on its own samples, should accept collapsed models. If it refuses them as well as the outside review does, outside criticism is not what makes the review safe.

## Rivals

| Rival | What it predicts that differs | Cheapest test that tells them apart |
|---|---|---|
| 1. Only the volume of real data matters | At the record sizes tested, critic and store protect alike, and neither comes near training with no generated data | Test 1, critic against store |
| 2. The critic's accuracy matters, not its being outside the loop (N16 and N17's open question) | A reference model inside the loop protects as well as an outside one of the same accuracy | Test 1, the mechanism check |
| 3. Curvature helps only hierarchy | C1a holds and C1b fails | Test 3, both kinds of data |
| 4. Tuning or optimisation differences (curved heads are harder to optimise) | The ranking changes when each arm gets its own best learning rate | Test 3's learning-rate sweep |
| 5. The critic freezes errors beyond the record | C3 fails | Test 2 |
| 6. Review lets collapse back in | A critic with reviews protects no better than no critic | Test 1, the review arm |

## Prior art

**Searched with Scite**, on 2026-11-11 and earlier in the discussion, for:
- hyperbolic embeddings and networks;
- model collapse;
- watermarking of generated text;
- Son (2026), cited in this repository.

**Not yet searched:**
- compression of training data for continual learning (rehearsal, replay buffers, coresets), and self-distillation as a guard against forgetting. Both are close to the micro container.
- knowledge editing, which changes specific facts in a trained model. It is close to C5's review.

**So neither the critic record nor its role should be called new.** Pending those searches, this note may add:
- the test of whether protecting records freezes errors outside them (C3);
- a detector judged by its hit and false-alarm rates (C4);
- dated review of protected meanings, judged on outside data (C5);
- the pairing with curved geometry.

## Proposed tests, in order

A sketch. Exact weights, sizes, rates, thresholds and decision rules belong in a preregistration, written after the author approves a direction. Tokens and compute are matched across arms.

### Test 1. C2: collapse and the critic

**Set-up.**
- Train generation 0 on original data.
- Each later generation continues from its predecessor and trains on text sampled from it, at temperature 1 with no truncation. Each generation's text replaces the last.
- The record holds k sequences of one type, for example articles on one topic. The secondary record is k sequences drawn at random.
- No sequence of a record is in the held-out text.

**Arms**
- *A, no protection.*
- *B, no generated data:* the same steps on fresh original data instead. This is the yardstick for "damage" (lesson 8).
- *Store:* a weighted term training on the record's own next tokens.
- *Critic:* the same weighted term, with generation 0's predictions on the record as its targets.

**Measures**
- *Primary:* loss on held-out text of the record's type.
- *Secondary:*
  - loss on all other held-out text;
  - for the random record, loss on held-out text in general;
  - loss on held-out positions whose token is rare, with "rare" fixed by generation 0's training data.

**Preconditions.** If any fails, C2 is *untestable*.
1. *The sampler works.* Generation 0's loss on its own samples must match its own predictive entropy within a set tolerance. A model scored on its own exact samples should show this, and truncation or a wrong temperature would break it.
2. *Collapse happens.* Arm A's loss must exceed arm B's by a preregistered amount, so that there is damage to prevent.
3. *The penalty takes hold.* By the last generation, the critic arm's divergence from generation 0 on the record must be below a set fraction of arm A's. The store arm's loss on the record must be below a set fraction of arm A's.
   - The weight is chosen by a calibration over the full schedule, not a short trial. N11's dose check stopped too early (CRITICAL_REVIEW, section 7).

**Mechanism check, for rival 2.** Keep the record fixed and vary only the reference model whose predictions the critic uses:
- generation 0, which is outside the loop;
- the predecessor, which is inside the loop;
- a model trained on original data only, for fewer steps, so that its held-out loss matches the predecessor's. It is outside the loop, at the predecessor's accuracy.

If the predecessor protects as well as the matched model, being outside the loop is not what protects. As in N17, the match can only be approximate.

**Review arms, for C5 and rival 6.**
- The critic with a review every set number of generations, judged on a review set of held-out genuine text of the record's type.
- The same, with the review judged by the model itself (its own loss on its own samples). This is C5's mechanism check.

### Test 2. C3: a seeded error, without collapse

**Set-up.**
- In generation 0's training data, swap a fixed family of relationships consistently. For example, exchange the countries in a set of capital-city sentences.
- Then continue training on genuine data only, including sentences that state the true relationships.
- With no generated data, collapse plays no part.

**Arms**
- *The critic,* with a record chosen to exclude every sequence that mentions the swapped family. This is checked by search, not assumed.
- *No critic.*
- *Manipulation check:* the critic with the swapped family inside its record. It must keep the error. If it does not, the penalty is not acting, and C3 cannot be decided.
- *For C5:* the critic with the swapped family inside its record, and one review at a fixed point. The review set holds held-out sentences stating the true relationships.

**Precondition.** Generation 0 must favour the swapped completion by a set margin on held-out probe sentences. If it never learned the swap, there is nothing to correct.

**Measure.** On the probes, how much more probable the model finds the true completion than the swapped one, after a fixed amount of further training.

### Test 3. C1: curvature in a head

**Arms**
- *The curved head,* at several widths.
- *The same head at zero curvature,* the flat arm.
- *The standard linear head,* for reference only.
- *Learned against fixed curvature,* for I2 (descriptive).

Each arm gets its own learning-rate sweep, at equal budget.

**Two kinds of data**
- *Plain text:* next-token prediction on the project's model and data.
- *Tree-like data:* Ganea et al.'s prefix task at several noise levels, in their own set-up, not the project's model.

**Instrument control (must match).**
- At small curvature, the curved head's loss must agree with the zero-curvature head's within a stated tolerance.
- It is a limit, not an identity, because the formulas recover a scaled flat distance as curvature goes to zero.

### Test 4. C4: the detector

The damaged and undamaged models described under C4, several of each, scored by the divergence on the record. This test is independent of the others.

## What each outcome would teach

| Outcome | What we would learn | Next conjecture |
|---|---|---|
| A precondition of test 1 fails | Nothing about C2, which is untestable; the fault is named | Fix the named fault |
| C2 holds | A small typed record, used as critic, protects its own type at this scale | How small the record can be; whether it protects other data too |
| C2 fails | A record of this size, used this way, does not protect its type here | Whether a larger record, or the store, does |
| Store protects as well as critic | Fixing the records' words is enough, and the design's tension dissolves at this scale | Whether that holds as collapse deepens |
| Critic protects, store does not | The tension stands: protection here comes from holding the records' meanings still | A critic that protects without freezing meanings |
| Predecessor protects as well as the matched outside model | Accuracy, not being outside the loop, is what protects (rival 2) | Which property of the reference does the work |
| C3 holds | Protecting records leaves what lies outside them correctable | Whether errors inside the records need their own correction procedure |
| C3 fails | The critic freezes errors beyond the record, through shared parameters | A critic acting on fewer parameters |
| C5 holds | Meaning tied to purpose and time, with review, keeps both the protection and correctability | What should trigger a review, and how purposes are specified |
| C5 fails: the error stays after review | The review set is too weak to move the reference, or the critic acts beyond its reference | A stronger review set, or a review that also edits the model directly |
| C5 fails: reviews let collapse back in | Outside data on the review set do not catch collapse at this scale | A review that also uses the detector (C4) |
| The self-judged review refuses collapsed models too | Outside criticism is not what makes the review safe | Which property of the review set does the work |
| C4 holds | Drift on the record tells damage from learning | Whether it warns earlier than held-out loss |
| C4 fails | Drift on the record moves with ordinary learning too | Anchoring geometry prevented little in Son's study; whether a geometric detector does better is open |
| C1a holds, C1b fails | A curved head buys width only for hierarchy | A mixed geometry: curved for hierarchy, flat for the rest |
| C1a and C1b hold | A curved head buys width on plain text too, at this scale | A curved trunk |
| C1 fails everywhere | At this scale, a curved head buys nothing a flat head cannot learn | Whether "folded", once defined, behaves differently |

## What this note does not claim

- Nothing here is a finding. Every conjecture is untested.
- The Deutsch arguments are guiding principles, not results.
- The compression is lossless, and is not tested.
- No list of mission-critical data types is proposed: the design leaves that to each use case. The tests use one topic only as a stand-in.
- The marker role is not tested. In these tests, which data are generated is known by construction. The marker matters only where provenance is unknown.
- The author's claim that fixing relationships in memory saves compute is not tested.
- This repository's results come from small models at a tenth of the full training budget, and would not show what happens at scale.
- Every earlier verdict in this repository stands, whatever these tests show.

## Questions for the author

**Answered on 2026-11-13**
1. *What does the compression keep?* The original records. Data must first be classified as mission-critical to be compressed and kept.
2. *Which types of data count as mission-critical?* That decision weighs risks that can only be judged in context, so it is left open for each use case.

**Answered on 2026-11-15**
3. *Is the compression lossless?* Yes.
4. *Should the design hold the records' meanings still, or keep them changeable?* Neither alone: "specific meaning(s) for specified purpose(s) at the time they were specified, but review for future cases". This is C5.
5. *Where should this line of work live?* In its own folder on `main`, `experiments/curved-meaning/`.

**Open**
6. What, concretely, would "fixing the relationships between ideas held in memory" fix, and where?
7. What should trigger a review, and what should it check?
8. How is a purpose specified, so that a model can tell one purpose's context from another's?

## References

Arazo, E., Ortego, D., Albert, P., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

Chen, W., Han, X., Lin, Y., Zhao, H., Liu, Z., Li, P., Sun, M., & Zhou, J. (2022). Fully hyperbolic neural networks. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics* (pp. 5672–5686). https://doi.org/10.18653/v1/2022.acl-long.389

Fu, S., Wang, Y., Chen, Y., Shen, L. C., & Tao, D. (2025). Self-verification provably prevents model collapse in recursive synthetic training. *Advances in Neural Information Processing Systems 38*. https://doi.org/10.52202/085713-1213

Ganea, O. E., Bécigneul, G., & Hofmann, T. (2018). Hyperbolic neural networks. arXiv. https://doi.org/10.48550/arxiv.1805.09112

Gerstgrasser, M., Schaeffer, R., Dey, A., Rafailov, R., Sleight, H., Hughes, J. D., Korbak, T., Agrawal, R., Pai, D., Gromov, A., Roberts, D. A., Yang, D., Donoho, D. L., & Koyejo, S. (2024). Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data. arXiv. https://doi.org/10.48550/arxiv.2404.01413

Gu, Y., Pang, L., Ye, X., Wang, T., Lin, J., Priebe, C. E., & Aue, A. (2026). SIGMA: Scalable spectral insights for LLM model collapse. arXiv. https://doi.org/10.48550/arxiv.2601.03385

Jovanović, N. M., Staab, R., & Vechev, M. (2024). Watermark stealing in large language models. arXiv. https://doi.org/10.48550/arxiv.2402.19361

Li, X., Ruan, F., Wang, H., Long, Q., & Su, W. J. (2025). Robust detection of watermarks for large language models under human edits. *Journal of the Royal Statistical Society Series B: Statistical Methodology*, *88*(2), 491–515. https://doi.org/10.1093/jrsssb/qkaf056

Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. arXiv. https://doi.org/10.48550/arxiv.1705.08039

Palmer, T. (2022). *The primacy of doubt*, chapter 2, "The geometry of chaos", pp. 27–33. Supplied by the author as screen grabs; full publication details to be added.

Shumailov, I., Shumaylov, Z., Zhao, Y., Gal, Y., Papernot, N., & Anderson, R. (2023). The curse of recursion: Training on generated data makes models forget. arXiv. https://doi.org/10.48550/arxiv.2305.17493

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature*, *631*(8022), 755–759. https://doi.org/10.1038/s41586-024-07566-y

Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R., & Gal, Y. (2025). Author correction: AI models collapse when trained on recursively generated data. *Nature*, *640*(8058), E6. https://doi.org/10.1038/s41586-025-08905-3

Son, C. (2026). *J-space forecasts model collapse, but only anchoring the function prevents it* [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-10713541/v1
