# Conjecture note: where this programme can go, and whether understanding works like perception

Date: 2026-10-06    Status: conjecture, untested. Written while G1d runs; no G1d result had been seen.

## The questions

1. **How can these experiments go on, and succeed?** What tests come next, and how does each outcome branch?
   - "Succeed" here means create knowledge. A programme succeeds when its tests keep telling conjectures apart, not when a favourite conjecture survives.
   - So this note maps where each path leads whichever way its test comes out, and tries not to favour any outcome.
2. **The author's conjecture about human intelligence.** Human-level intelligence evolved because the brain became able to *understand* in a way like seeing and hearing: taking in structure as a whole and at once. It is granted that this is not with the same kind of simultaneity sight and hearing achieve.
3. **Could this project become, or contribute to, AGI?**

## Where the programme stands

| Test | Result | What it taught |
|---|---|---|
| WikiText-2 replication and WikiText-103 study | The paper's claims refuted for Model A as built: 1.10 nats worse than the baseline; α grew; solver choice mattered | The original model, trained as it was, did not do what the paper said |
| T0 | Untestable under its own rules | Exploratory: the adjoint gradient for α pointed the opposite way to the exact gradient |
| T1 | The adjoint training method accounts for about 0.79 of the 0.83-nat gap at 10% of the budget | With exact gradients, Model A was within 0.04 nats of the baseline; weight-tied depth matched separate layers |
| G1 | Regimes of different curvature refuted: ratios 1.08 to 1.27 | The control nudges how trajectories turn, nothing more |
| G1c | Regimes of different contraction supported: ratios 4.1 to 10.2, on new seeds and windows | The control sets how much the flow forgets differences between inputs. The difference lies in the field itself and in its shape, not only its gain; u = 0 sits midway |
| G1d | Running | Is that contraction needed, or a by-product? |

**What the pattern says so far.**
- Most of the architecture's original failure was the training method, not the idea.
- The control does divide the dynamics into regimes, but not the ones the paper pictured. They are regimes of *how much to forget*, not of *how to turn*.
- That shift came from criticism, refutations and one confirmed new conjecture, which is what the method is for.

## Making "understanding works like perception" precise

The conjecture has several readings, and each predicts something different.

| Reading | What it says | What it predicts | Prior work bearing on it |
|---|---|---|---|
| **U1. Parallel** | Many structures are taken in at once, not one after another | Understanding a richer text should not take proportionally longer, up to some limit | Vision itself is only partly parallel: simple features are registered in parallel, but binding them into objects takes serial attention (Treisman & Gelade, 1980). Perception is not purely simultaneous either. |
| **U2. Immediate** | Understanding happens automatically, without deliberation, once learned | Experts should *see* structure that novices must work out | Chess masters perceive whole configurations as chunks that novices assemble piece by piece (Chase & Simon, 1973) |
| **U3. Conjectural** | Perception, like science, proposes hypotheses about what is there and corrects them, and understanding does the same over abstract structure | Systematic misunderstandings should behave like illusions: confident, fast, and correctable only by new evidence | Perceptions as hypotheses (Gregory, 1980); the brain as a prediction machine (Clark, 2013; Friston, 2010) |
| **U4. Grounded** | Concepts are reactivated perceptual simulations | Abstract understanding should recruit perceptual machinery | Perceptual symbol systems (Barsalou, 1999) |
| **U5. Holistic** | Understanding grasps a whole that cannot be broken into steps | No step-by-step procedure reproduces it | Hard to test as stated; the most likely to be unfalsifiable, and so the reading to sharpen or drop |

Keeping these apart matters. U1 is the reading closest to the founding inspiration, polyphony. U3 is the one that connects to the epistemology this project works by.

## Criticising the author's conjecture

### What speaks for it
- **Experts perceive.** Expertise turns slow reasoning into something like perception (Chase & Simon, 1973).
- **One account covers both.** Predictive-processing accounts treat perception and higher cognition as one kind of process, which makes "understanding like perception" a natural reading rather than a metaphor (Clark, 2013).
- **Fluent reading is like seeing.** Once a language is fluent, you cannot help understanding a sentence in it, much as you cannot help seeing a face. That is everyday experience, not a cited finding.

### The strongest criticisms
- **Animals perceive superbly without human understanding.** If understanding were *just* perception, animals with excellent sight and hearing should understand more than they do. The conjecture has to say what is different about the human case.
- **The competing explanations of the human difference:**
  - **Recursion.** The capacity for recursion, possibly evolved for reasons other than language, may be the uniquely human component (Hauser, Chomsky & Fitch, 2002).
  - **Social life.** The demands of social life drove brain size (Dunbar, 1998).
  - **Creativity.** David Deutsch's view, which this project's own method draws on: what makes humans general intelligences is *creativity*, the ability to conjecture new explanations. Perception does not do that on its own (Deutsch, 2011, a book, not checked through the citation index).
- **Even vision is not fully simultaneous** (Treisman & Gelade, 1980). So the contrast with "understanding" may be one of degree, not kind. The author grants this already.

### A synthesis that keeps what survives

**A sharper conjecture, A\*.**
- **The claim.** Human understanding is perception turned on one's own conjectures. It is a fast, parallel process that proposes and tests many candidate interpretations at once, as Gregory's perceptual hypotheses do, but over abstract and recursive structures rather than over light and sound. Slow, serial, explicit reasoning is the layer that corrects its errors.
- **The evolutionary claim.** The leap was not perception itself. It was turning the brain's hypothesis-generating perceptual machinery onto internally generated, recursive representations.
- **What it keeps.** It keeps the author's insight, the perceptual character of understanding, and takes in the main rivals instead of ignoring them. Recursion supplies the material, creativity is the hypothesis generation itself, and perception supplies the speed and parallelism.
- **What it forbids:**
  - understanding of a new abstract structure that arrives *only* through step-by-step reasoning, with no fast interpretive stage ever forming with practice;
  - expertise that never becomes perception-like.
- **What would refute it in AI terms.** Systems that hold several interpretations in parallel never beat serial ones, at matched compute, on tasks that need several structures tracked at once.

A\* is a conjecture about human minds, and most of it lies outside what this project can test. Its AI version can be tested, and the paths below include that.

## AGI: could this project contribute?

### What this project is not
- **Nothing here approaches AGI.** These are 10%-budget language models of about 30 million parameters, most of them in the embedding and output layers, on one dataset.
- **None of these tests measures what Deutsch calls the mark of general intelligence:** creating new explanations.

Claiming otherwise would be the kind of story the method exists to prevent.

### What it could contribute, honestly

- **A mechanism-level piece.**
  - G1c shows that one control signal can put a network's computation into distinct dynamical regimes. Those regimes do measurably different things: one forgets detail fast, one keeps it.
  - That is a small, real instance of "reading the same text in different ways" being a property of the dynamics.
  - If G1d shows the regime is needed, it is a mechanism, not a decoration.
- **A precise version of the founding idea.**
  - Transformers already process every *position* in parallel: attention looks across the whole context at once. So "simultaneity over positions" is not new.
  - What current models lack, as far as these experiments show, is explicit simultaneity over *readings*: several interpretations of the same text, each running in its own regime, side by side.
  - That is the gap the author's polyphony idea points at, and a test can target it directly (path D below).
- **A way of doing the science.** Preregistered, refutation-seeking, mechanistic experiments on small models, with a record of what failed, are themselves a contribution to how claims about machine understanding are tested.

### What would count against "contributes to AGI"
- Parallel readings give nothing that a larger standard transformer doesn't.
- The regimes are by-products (G1d refuted, and the whole-stack follow-up refuted too).
- The effects vanish as models grow.

Each of these is testable, which is what makes the ambition scientific rather than hopeful.

## Paths forward

### Branches from G1d

| G1d outcome | Next test | Why |
|---|---|---|
| **Supported**: the contraction is needed | Farther or harder: add a task as near as the next token but harder, such as the next token with the current token hidden | Separates the two properties confounded in every test so far |
| | Specificity: a constraint of matched strength on another property of the u = −1 dynamics, such as speed | Shows whether contraction is special or any constraint would hurt |
| | u as a dial: train with a continuous u that sets a mixture of targets, and predict contraction monotone in u | G1c's midpoint for u = 0 suggests it |
| **Refuted**: no material cost | Cap contraction across the whole stack, from the embedding to the output | Tells "a by-product" from "work other layers take over" |
| | Exploratory and cheap: measure whether the capped models' later blocks contract more, using G1d's saved weights | A first look at substitution, before preregistering |
| **Some effect, or inconclusive** | More seeds, as a new preregistered test | The size is unsettled |
| **Untestable**: the cap would not hold | Make contraction a named part of the field: f = −γ(u)·h + g(h, τ), with γ a learned contraction rate per control value. Then intervene by fixing γ | Turns a soft penalty into an exact intervention, with an exact must-match control: γ free reproduces the trained model. It also leads into G3, multiplicative control |

### Tracks that don't depend on G1d

- **A. Model A trained properly, at full budget.**
  - Exact gradients, the full WikiText-103 budget, against the baseline.
  - T1 suggests the gap may nearly close.
  - This is the result that decides whether the architecture deserves attention at all, and it is worth running whatever the regime experiments show.
- **B. Contraction as forgetting, checked directly.**
  - The claim: if u = −1 contracts more because it discards detail about the current token, a linear probe for the current token's identity, applied after the ODE block, should be less accurate under u = −1 than under u = +1.
  - Cheap: it runs on G1c's saved models. As a first look it is exploratory, and a preregistered version would use new seeds.
  - Prior work: this resembles the information-bottleneck view of deep learning (Tishby & Zaslavsky, 2015). That view's general claims were strongly criticised (Saxe et al., 2019), so the test should stand on its own measurement, not borrow the theory's authority.
- **C. Polyphony on a task built for it.**
  - The task: two unrelated texts interleaved token by token, or a two-speaker dialogue, with the model predicting the next token of each strand.
  - The measures:
    - Do token trajectories group into regimes by strand, measured by contraction and other geometry?
    - Does how distinct a strand's regime is predict how well it is tracked? This is G2.
  - Must-fail control: strand labels shuffled.
  - Then: natural text with several strands, such as long chains of reference to the same person, or nested clauses.
- **D. Several readings at once: a polyphonic ODE.**
  - The architecture: K flows over the same tokens, each under its own control u_k, coupled by attention between flows, with each flow read out for its own target (for example the next token, the token after next, and a topic label).
  - The comparison: one flow with K output heads, at matched parameters and compute.
  - Why it matters: this is the most direct test of the founding idea, simultaneity over readings, and of A\*'s AI version.
  - Must-match control: with the coupling between flows switched off and the controls tied, the K-flow model must reduce exactly to K copies of one flow.
- **E. Control the model sets for itself.**
  - The design: a small network infers u for each token from context, with no task labels.
  - The questions: do regimes emerge without supervision, and do they line up with linguistic structure, such as syntax, meaning or discourse?
  - Must-fail control: a randomly assigned u.
  - Why it matters: this moves from regimes we impose to regimes the model discovers, which is closer to understanding.
- **F. Many horizons at once.**
  - The two-task setup resembles multi-token prediction, which has been reported to improve language models (Gloeckle et al., 2024).
  - The conjecture: a single block whose control selects the horizon, with a regime per horizon, beats separate output heads at matched parameters.
  - This would connect the regimes to a known practical gain.
- **G. Systematic generalisation.**
  - A bold conjecture, and quite possibly a false one: controlled-regime models generalise compositionally better.
  - The test: a benchmark built to probe this, such as SCAN (Lake & Baroni, 2018).
  - Cheap enough to try once the regime machinery is reusable. A refutation would still bound what regimes do.
- **H. Scale.**
  - Do contraction regimes persist, strengthen or vanish with width, depth and training budget?
  - Without this, nothing here generalises beyond small models.

### My recommended order

The recommendation is driven by knowledge gained per hour of compute:

1. **G1d** (running).
2. **B, the probe.** It costs minutes on saved models and tests the *explanation* behind G1c, not just the effect.
3. **Farther or harder.** It removes the one confound in every task so far.
4. **C, then D.** They go straight at the founding idea, polyphony, and at A\*. They are the biggest potential gain for the author's core vision.
5. **A, the full-budget run.** It decides whether the architecture itself is competitive, which matters for anyone else taking the work seriously.

## Red team: how this programme could fool itself

- **Forking paths.** G1c came from an exploratory look. Every such finding must keep going through fresh seeds, fresh windows and fixed thresholds, as G1c did.
- **Too many tests.** Many tests, each at 5%, will produce some false "supported" results by chance. Count them across the programme, and treat isolated supports cautiously until replicated.
- **Wanting regimes.** The polyphony picture is attractive, and that makes it dangerous. The guard is a must-fail control in every test, and outcome tables written before the results.
- **One-number summaries.** Contraction, like curvature before it, is one summary of the Jacobian. A full spectrum, or several summaries, would show what the trace hides.
- **Small-model artefacts.** Everything so far is at 10% of the budget, at one size. Track H is the guard.
- **Metaphor drift.** "Voices", "readings" and "forgetting" are useful names. They are not evidence. Every claim must reduce to a measured quantity.

## What each outcome would teach

| Outcome | What we would learn | Next conjecture |
|---|---|---|
| G1d supported, probe shows less current-token information under u = −1 | Contraction is a functional forgetting mechanism, selected by the control | Regimes as voices: do strands of a text get their own forgetting rates? (C, G2) |
| G1d supported, probe shows no difference | Contraction is needed, but not because it discards the current token | A new conjecture about what is discarded: a different probe target |
| G1d refuted | The regime exists but is not needed by this model at this budget | Whole-stack test; or regimes matter only when several must coexist (D) |
| The polyphonic ODE beats one flow at matched compute on multi-strand tasks | Parallel readings are a real design gain at this scale. This is the AI version of A\*, supported. | Scale it (H); test whether it generalises (G) |
| The polyphonic ODE gives no gain | Parallel readings add nothing a single flow can't do at this scale. A\*'s AI version is refuted at this scale. | Whether the gain needs scale, or whether readings should be serial after all |
| Regimes emerge from self-set control and line up with syntax and meaning | The model discovers ways of reading on its own | The closest this programme can come to a machine that "sees" structure in text |

## What this note does not claim

- **Nothing in it is a finding.** Every path, conjecture and prediction here is untested. The only results are those already in the result notes, summarised in the first table.
- **Earlier refutations stand.** The paper's claims about Model A as built, and G1's curvature regimes, stay refuted whatever later tests show.
- **The evolutionary conjecture is mostly beyond reach.** A\* is a reasoned refinement of the author's conjecture, not support for it. Most of its human part lies outside what this project can test.
- **No AGI claim.** The project could contribute a mechanism and a method; it is not on a path to AGI by itself, and saying otherwise would need evidence it does not have.

## After G1d: what three regime tests add up to, and where next (2026-10-06)

Written after G1d's result. Everything in this section is conjecture.

### What the evidence now says

- **G1:** the trained control barely changes how trajectories turn.
- **G1c:** it changes how strongly they contract, by a factor of 4 to 10. This replicated twice.
- **G1d:** removing over 90% of that contraction difference costs nothing differential. Meanwhile the models keep using the control fully: the wrong-control cost was unchanged, and the curvature-profile classifier still told the readings apart.

So the model switches between readings *without* needing the large dynamical difference. The reading is implemented somewhere, or somehow, else.

**A methodological lesson.** A classifier will find *some* geometric difference between two control values almost whenever one exists, and existence is cheap. The bar G1d set, necessity tested by intervention, should be required of every future regime claim.

### New conjectures

**N1. The reading is a direction, not a regime.** This is the deflationary conjecture, and the most likely mundane rival to everything above.
- **The explanation.** The control enters the vector field by addition. The easiest solution is for each control value to shift the state along its own direction, which later layers read. Then the "regime" differences are side effects of where the shift puts the state.
- **Predictions:**
  - The difference in the block's output between u = −1 and u = +1 is nearly the same vector for every window: one direction carries most of its variance.
  - A model in which u enters *only* as a learned bias added after the ODE block, with no u inside the dynamics, matches the main models on both tasks within 0.05 nats.
- **Refuted if** the bias-only model is materially worse on either task.
- **Cost:** cheap. The first prediction runs on saved weights; the second is 5 short training runs.
- **Why it matters either way:**
  - **If supported,** the ODE's dynamics are not where the reading happens. The polyphony programme should then either move to representation subspaces, or use tasks where an offset cannot do the job.
  - **If refuted,** the dynamics do work an offset cannot, and the next task is to find it.

**N2. Regimes become load-bearing only when readings must coexist.**
- **The explanation.** Switching lets one state serve one reading at a time, so nothing forces the readings apart. When two readings must be held in the same state *at once*, they interfere unless kept separate. One way to keep them separate is for different subspaces of the state to run at different contraction rates: different timescales, like registers in music.
- **Prior art.** Networks with neurons at multiple timescales develop functional hierarchy (Yamashita & Tani, 2008). State-space sequence models run many decay rates side by side, one per eigenvalue (Gu, Goel & Ré, 2021; Gu & Dao, 2023), though over *time*, where N2 is about *depth*.
- **The test:**
  - **The task.** A dual-reading model with two output heads, the next token and the token after next, at every position, in one forward pass and with no control.
  - **The measures.** How far apart the two heads' readout directions lie, and the contraction restricted to each readout subspace (Hutchinson's estimator with a projection).
  - **The prediction.** Different contraction per subspace.
  - **The intervention, G1d-style.** Force equal contraction across the two subspaces; the prediction is that this costs the joint loss.
- **Must-fail control:** a single-head model shows no such split.
- **Why it matters.** This is the most direct test yet of the founding idea: polyphony needs simultaneity, and switching was never polyphony.

**N3. Multiplicative control makes regimes cheap, and therefore used (G3, sharpened by N1).**
- **The explanation.** If additive control naturally produces offsets (N1), control that multiplies parts of the vector field can change its Jacobian directly.
- **The prediction.** Under multiplicative control, contraction differences are larger *and* pass a G1d-style necessity test.
- **Refuted if** they are no larger, or not needed.

**N4.** Self-set control (path E) and the probe (path B) are unaffected by G1d and stay on the list.

**N1 result, 2026-10-06: supported.** An offset after the block matched the full model on both tasks within 0.001 nats. See `N1_NOTE.md`. N2 is next in the order below.

**N2 result, 2026-10-06: refuted.** Coexisting readings did not get distinct timescales (ratios 1.14 to 1.25). By the table above, this is the "N1 supported, N2 refuted" row: readings live in representation geometry, not depth dynamics, at this scale. Exploratory: the two readings' directions overlap only about 0.6. See `N2_NOTE.md`.

### Recommended order now

1. **N1.** It is the cheapest and settles where the reading lives, which changes what every later test should look at.
2. **N2.** The core polyphony question, in its sharpest testable form.
3. **N3 / G3.**
4. **The probe (B).**
5. **The full-budget run (A).**

### What each outcome would teach

| N1 | N2 | What we would learn | Next |
|---|---|---|---|
| Supported: an offset is enough | Supported | Switching needs only a direction; simultaneity needs dynamics. Regimes matter exactly when readings coexist, which is the founding idea, in testable form. | Polyphonic ODE (D), then scale (H) |
| Supported | Refuted | Neither switching nor simultaneity needs depth dynamics at this scale. Readings live in representation geometry. | Move the programme to subspaces and attention; test N3 before leaving dynamics |
| Refuted: the dynamics matter | Supported | The dynamics carry readings in both cases | Find the mechanism, starting from the Jacobian spectrum |
| Refuted | Refuted | The dynamics do something for switching, but not by separating timescales | A new conjecture from N1's diagnostics |

### References for this section

Gu, A., & Dao, T. (2023). *Mamba: Linear-time sequence modeling with selective state spaces*. arXiv. https://doi.org/10.48550/arxiv.2312.00752

Gu, A., Goel, K., & Ré, C. (2021). *Efficiently modeling long sequences with structured state spaces*. arXiv. https://doi.org/10.48550/arxiv.2111.00396

Yamashita, Y., & Tani, J. (2008). Emergence of functional hierarchy in a multiple timescale neural network model: A humanoid robot experiment. *PLoS Computational Biology, 4*(11), e1000220. https://doi.org/10.1371/journal.pcbi.1000220 (A correction was published in 2010: https://doi.org/10.1371/annotation/c580e39c-00bc-43a2-9b15-af71350f9d43)

## References

Barsalou, L. W. (1999). Perceptual symbol systems. *Behavioral and Brain Sciences, 22*(4), 577–660. https://doi.org/10.1017/s0140525x99002149

Chase, W. G., & Simon, H. A. (1973). Perception in chess. *Cognitive Psychology, 4*(1), 55–81. https://doi.org/10.1016/0010-0285(73)90004-2

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences, 36*(3), 181–204. https://doi.org/10.1017/s0140525x12000477

Deutsch, D. (2011). *The beginning of infinity: Explanations that transform the world*. Allen Lane. (Book; not retrieved through the citation index.)

Dunbar, R. I. M. (1998). The social brain hypothesis. *Evolutionary Anthropology, 6*(5), 178–190. https://doi.org/10.1002/(sici)1520-6505(1998)6:5<178::aid-evan5>3.0.co;2-8

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience, 11*(2), 127–138. https://doi.org/10.1038/nrn2787

Gloeckle, F., Youbi Idrissi, B., Rozière, B., Lopez-Paz, D., & Synnaeve, G. (2024). *Better & faster large language models via multi-token prediction*. arXiv. https://doi.org/10.48550/arxiv.2404.19737

Gregory, R. L. (1980). Perceptions as hypotheses. *Philosophical Transactions of the Royal Society of London B, 290*(1038), 181–197. https://doi.org/10.1098/rstb.1980.0090

Hauser, M. D., Chomsky, N., & Fitch, W. T. (2002). The faculty of language: What is it, who has it, and how did it evolve? *Science, 298*(5598), 1569–1579. https://doi.org/10.1126/science.298.5598.1569

Lake, B. M., & Baroni, M. (2018). *Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks*. arXiv. https://doi.org/10.48550/arxiv.1711.00350

Lake, B. M., Ullman, T. D., Tenenbaum, J. B., & Gershman, S. J. (2017). Building machines that learn and think like people. *Behavioral and Brain Sciences, 40*, e253. https://doi.org/10.1017/s0140525x16001837

Saxe, A. M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey, B. D., & Cox, D. D. (2019). On the information bottleneck theory of deep learning. *Journal of Statistical Mechanics: Theory and Experiment, 2019*(12), 124020. https://doi.org/10.1088/1742-5468/ab3985

Tishby, N., & Zaslavsky, N. (2015). Deep learning and the information bottleneck principle. *2015 IEEE Information Theory Workshop*, 1–5. https://doi.org/10.1109/itw.2015.7133169

Treisman, A. M., & Gelade, G. (1980). A feature-integration theory of attention. *Cognitive Psychology, 12*(1), 97–136. https://doi.org/10.1016/0010-0285(80)90005-5
