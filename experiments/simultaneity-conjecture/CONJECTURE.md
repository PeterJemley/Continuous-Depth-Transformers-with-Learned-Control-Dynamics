# Conjecture note: reading language polyphonically

Date: 2026-10-05    Status: conjecture, partly tested (updated 2026-10-22). G1 and the tests that followed from it (G1c, G1d, N1, N2, N4, N5, N6, N6r, N7, N8, N9, N10, N11, N12, N13, N14) have run. Their results are in the dated result lines at the end of "Dynamical regimes as voices" below, and in each test's note. V1–V3, G2 and G3 remain untested.

Written with the scientific-experimentation skill (conjecture-note template). Nothing here is a finding. Revised the same day after the author corrected the first version, which had misread the aim as setting word order aside.

## The question

**As the author put it.** Sight and hearing let us perceive more than one thing at a time: many notes sound and are heard together, and the elements of an image are taken in together. Language reaches us one word after another, in grammatical order. That order is right and stays. The aim is an AI that traces and follows meaning throughout a text *simultaneously*, perhaps using ODEs: its dimensions, ideas, and how they interact and interconnect, as naturally as a listener follows several voices in music or a viewer takes in a scene. This is why the project began.

**Restated against the systems at hand.** A text, like music, unfolds in order. A listener to a fugue does not hear one note at a time. They hear several voices at once, each a continuous line with its own course, and hear how the lines relate. The question is whether an AI reading a text holds its many strands the same way:
- the characters and their changing states;
- the arguments and themes;
- who knows or wants what;
- the tone;

each strand followed continuously, all at once, distinct from one another, and in interaction. And would a design built for this, with each strand a continuous trajectory along the text, governed by coupled differential equations, do it better than today's models?

## Readings

"Following meaning simultaneously" has several parts. Each is a property a model can have or lack, and each predicts something different.

| Reading | What it means for a model | What it predicts |
|---|---|---|
| P1. How many voices at once | The number of strands whose current state can be read from the model at a given point in the text | Readout degrades as the number of concurrent strands rises, as listeners' tracking degrades with more voices |
| P2. Voices stay distinct | Strands don't blend into each other, even when they share words or alternate closely | More errors when strands are similar or interleave tightly, the analogue of voices crossing in music |
| P3. Voices interact | A change in one strand updates the strands it affects: "A gave B the key" changes both A and B | Errors concentrate where strands interact, if interaction is handled poorly |
| P4. Each voice is continuous | A strand's state is carried and updated through the text, not reconstructed only when it is mentioned | A strand's state stays readable at points where it is not mentioned |

## What the evidence already says

- **In music, tracking voices has limits, and distinctness helps.** Listeners followed a cued voice in Bach's *The Art of the Fugue* less well as the voices increased from 2 to 4. For young listeners, the decline was smaller when the voices were played on different instruments: "performance degraded less drastically with an increase in the number of voices for timbrally heterogeneous mixtures" (Siedenburg et al., 2021). Bach himself kept voices from crossing in pitch, apparently to limit "perceptual confusion as the density of auditory images increases" (Huron, 1991). A model of polyphonic reading should expect the same pressures: number, distinctness and crossing.
- **Today's language models already track several entities.** In natural narratives, entity tracking "is already present at 410 million parameters" and improves with scale. In humans, it "degrades specifically with narrative complexity, not narrative length" (Drożdż & Heilbron, 2026). That preprint also reports earlier probing evidence that most of the measured accuracy came from trivial cases. So the measure has to exclude cases a model can answer without tracking.
- **Explicit voices have been tried.** EntityNLM gives each entity its own representation and updates it through the text (Ji et al., 2017). Slot attention gives each object in an image its own slot (Locatello et al., 2020). Recurrent Independent Mechanisms run several modules with their own dynamics that interact sparingly (Goyal et al., 2019). Neural controlled differential equations let a continuous state evolve driven by an input path (Kidger et al., 2020). Taken together, these are the parts a polyphonic design would be built from.
- **This project's results don't test the idea.** The continuous-depth block evolved one representation through the model's depth, not several strands along the text. So the ODE this idea calls for, many voices each running continuously along the text, has not been tried here, and the refuted claims say nothing against it. One finding carries over: any continuous design should be trained with exact gradients, since T1 found the adjoint method cost Model A about 0.8 nats.

## Conjectures

**V1, descriptive: models have a limit on how many voices they can hold, set by voices, not length.** For a language model reading texts with k interleaved strands, the accuracy of reading each strand's current state falls as k rises, more steeply than when the text simply gets longer with k fixed. It falls further when strands are similar or cross. This is also a measurement of how many voices today's models hold.
- Forbids: accuracy flat in k at matched length, in every model tested.
- A crossing effect absent where strands are similar would criticise the music analogy.

**V2, design: continuous, separate voices hold more at once.** A model with k explicit strand states, each evolving continuously along the text and driven by it (a controlled differential equation), is compared with a transformer of matched size, compute and training data. The strands are coupled through interactions that the strands' content modulates. The polyphonic model's accuracy falls less as k rises.
- What kind of claim: learnability, not capacity. A transformer can in principle hold the same information.
- Forbids: the same slope of decline at matched budgets, or a lower slope only at k = 1, where polyphony isn't needed.

**V3, interaction: voices that inform each other.** On texts where strands affect each other ("A gave B the key"), V2's model with modulated coupling beats the same model with the coupling switched off, by more than on texts where strands are independent. Without this, the voices are parallel but not polyphonic in the musical sense, where lines answer each other. This draws on the earlier "too linear" conjecture C3.

## Dynamical regimes as voices

Added 2026-10-05, from the author's clarification.

**The idea.** The paper reported that "adaptive ODE solvers reveal geometric structure in the learned dynamics: the control signal partitions the vector field into distinct dynamical regimes with different curvature characteristics." The author's aim links that claim to polyphony. If a control signal can place trajectories in geometrically distinct regions of a vector field, each with its own curvature, then several strands of meaning could be carried at once as continuous trajectories through distinct regimes. Each strand would be a voice with its own character, guided by the ODE, and the voices would be told apart by their geometry.

The analogy has a precise counterpart in music. Listeners track a voice better when the voices differ in timbre (Siedenburg et al., 2021). Here, a regime's curvature plays the part of a voice's timbre.

**The ODE block already has many simultaneous trajectories.** Each token's state traces its own path through the block's depth, all of them at the same time, coupled through attention. So the block is already a set of simultaneous, interacting trajectories. What has not been shown is that the control signal, or a strand's identity, sends them through distinct regimes.

**What the evidence says so far.** The paper's measure was solver effort: dopri5 needed more function evaluations at negative control. The WikiText-2 replication tested this as H6 and refuted it. Negative control needed more effort in only 2 of 8 seeds, and in most seeds the count was the same at every control value. That refutation stands for the model as built.

But that model was trained with adjoint gradients that T1 found costly. Its control signal was also attached after training, on one prompt. And solver effort is only a coarse measure of geometry. Kelly et al. (2020) show that an adaptive solver's cost is governed by the higher-order derivatives of the trajectories, so it reflects curvature only indirectly and in coarse steps of 6 evaluations. So whether a properly trained model forms distinct regimes is untested, not refuted. Testing it is a new conjecture, not a rescue of H6.

**Measures, more direct than solver effort.**
- The curvature of each trajectory: how sharply it turns, computed from its velocity and acceleration along the path.
- The local Jacobian spectrum of the vector field along the path: whether it contracts, expands or rotates the state, and how strongly.
- Solver effort, kept as a secondary measure for comparison with the paper.

Prior work shows trajectory geometry can also be shaped deliberately by regularisation (Finlay et al., 2020; Kelly et al., 2020).

**G1, regimes exist when trained for.** In a model trained with exact gradients, with the control signal used throughout training, different control values send trajectories through geometrically distinct regimes. The differences in curvature and Jacobian spectrum between control values exceed the spread within each control value.
- Forbids: no separation beyond the within-control spread.
- Must fail: an untrained model, and a model trained with shuffled control labels, show no separation.
- Must match: two runs at the same control value give the same geometry.

**G2, regimes as voices.** In texts with several strands, token trajectories group into regimes by strand. Strands whose regimes are more distinct are tracked more accurately, the analogue of timbre.
- Forbids: no relation between how distinct a strand's regime is and how well it is tracked, measured with TV1's tracking measure.

**G3, a control signal per voice.** Giving each strand its own control signal, entering the vector field multiplicatively as in conjecture C1 of the "too linear" note, makes the regimes more distinct and tracking more accurate than one shared additive control.
- Forbids: no gain from per-strand multiplicative control over shared additive control, at matched size.

**G1 result, 2026-10-05: refuted at this scale.** With exact gradients, additive control and two tasks, the trained control changed curvature by factors of only 1.08 to 1.27, against the factor of 2 required. The models clearly used u, and the shuffled-control models showed no difference. Exploratory: the two control values differed about fivefold in how fast they contract the state. See `G1_NOTE.md`.

**G1c result, 2026-10-06: supported.** That contraction difference was preregistered as a new conjecture and tested on new seeds and windows. The farther-ahead control value contracted the state 4.1 to 10.2 times faster in all five models; shuffled-control and untrained models showed none of it. Regimes exist here in contraction, not curvature. See `G1C_NOTE.md`.

**G1d result, 2026-10-06: refuted.** Capping that contraction during training, which removed over 90% of it, cost the farther task no more than the next-token task: D = −0.002, 95% interval [−0.007, +0.002], against a 0.05 margin. The regime is real but, by this test, not needed: a by-product, or work other layers take over. See `G1D_NOTE.md`.

**N1 result, 2026-10-06: an offset is enough.** A model whose control is only a learned vector added after the ODE block, with u never entering the dynamics, matched the full model on both tasks: differences of 0.001 nats, 97.5% intervals within ±0.01. Switching between readings needs nothing from the dynamics. Whether dynamics matter when readings must coexist is the open question (N2). See `N1_NOTE.md`.

**N2 result, 2026-10-06: refuted.** When one state must serve two readings at once, the farther reading's directions contracted only 14 to 25% faster, against the factor of 2 predicted. N2b (necessity) is therefore untestable. Exploratory: the two readings occupy partly separate directions (overlap about 0.6, against 0.99 for two heads reading the same target). The voices may separate in direction rather than in timescale. See `N2_NOTE.md`.

**N4 result, 2026-10-06: supported.** Forcing two coexisting readings through one shared set of 64 readout directions cost 0.088 nats more than the same restriction cost two heads reading the same target (95% interval [+0.078, +0.097]). The readings' directions stayed partly separate (overlap 0.46–0.49), while same-target heads converged (0.95–0.97). Coexisting voices need separate directions; they do not get separate timescales (N2). Exploratory: training did not push the readings apart, it failed to bring them together. See `N4_NOTE.md`.

**N5 result, 2026-10-07: inconclusive.** The readout-width sweep could not settle whether N4's cost is a competition for directions. At full width the cost was positive in all five seeds (mean +0.030, about a third of N4's) but its interval crossed both zero and the margin. Narrow widths made training itself erratic. Exploratory, and striking: two different readings settled at about half-overlap (0.42–0.53) at every width, whether they started above or below it, while same-target heads converged. See `N5_NOTE.md`.

**N6 result, 2026-10-08: untestable.** The must-fail control (same-target heads at an overlap of at least 0.95) held in 3 of 5 models, not 4; seed 3601 missed by 0.0002. The threshold had not been re-checked for N6's smaller number of windows. Exploratory, and consistent with the conjecture: the two readings' overlap fell from 0.48 at k = 2 to 0.31 at k = 4 and 0.25 at k = 8, in that order in all 5 seeds; o(2) − o(8) = +0.227, 95% interval [+0.184, +0.269]. A new test with the control set for its measurement (N6r) would settle it. See `N6_NOTE.md`.

**N6r result, 2026-10-08: supported, both claims.** With the must-fail control judged like with like and 256 fresh test-split windows, the two readings' overlap fell from 0.49 (next token and the token after) to 0.31 (four apart) and 0.23 (eight apart), in that order in all 5 models: o(2) − o(8) = +0.252, 95% interval [+0.219, +0.286]. Same-target heads converged (0.96). How much of the state two readings share depends on how different they are; half-overlap is what adjacent tokens need, not a fixed property of the model. The confound with difficulty remains. See `N6R_NOTE.md`.

**N7 result, 2026-10-09: refuted, both claims.** A head reading exactly the next token, made harder by randomising 80% of its targets, shared no more with the clean next-token head (0.31) than a head reading four words ahead (0.30), and did not converge with it (agreement ratio 0.30–0.33, against 0.85 required). So N6r's ladder, which stands and replicated, does not measure how different readings are, not on its own. The overlap measure likely reflects where each reading's errors fall as well as which features it uses. N4's cost of sharing, measured in loss, is unaffected. See `N7_NOTE.md`.

**N8 result, 2026-10-10: N8a and N8b refuted, N8c supported.** Measured again with a target-free version of the overlap, on the same models, N7's noisy head still shared about a third with the clean head (0.31; agreement ratio 0.32–0.34, against 0.85 required), while the ladder persisted (0.95, 0.44, 0.25, 0.19). The two measures agree to within 0.045 in every arm, so N7's explanation, that the old measure mixed in where errors fall, is refuted. The ladder reflects which directions each reading's predictions depend on. Why a noisy copy of the next-token reading reads different directions is open. The design did not check whether the measure could recognise an ideal noisy copy, and that is the next test. See `N8_NOTE.md`.

**N9 result, 2026-10-10: N9a supported, N9b refuted.** The target-free measure recognises an ideal noisy copy of the clean next-token head as the same reading (agreement 0.955–0.995 at 80% noise). The trained noisy head is not close to that copy (0.305–0.349). So N7's anomaly is in the model, not the measure. Training a head on noisy targets made it read different directions of the state, and read them less well. In what it predicts, the noisy head is close to the copy (0.07 nats of KL divergence per position), because both are dominated by the same fixed word frequencies. The ladder's measure is now validated on a case with a known answer. See `N9_NOTE.md`.

**N10 result, 2026-10-18: N10a and N10b untestable, N10c intermediate, N10d supported.** On new seeds, a head trained on noisier answers read directions further from its ideal noisy copy, in the order 20% > 50% > 80% noise in every seed (ratios 0.81, 0.64, 0.31). At 20% noise it was functionally almost the ideal copy (0.04 nats), but its directions had already drifted (0.81, against 0.85 required). The payload test gave no verdict: the functional payload was too weak to take hold (a design error, caught by its preregistered manipulation check). Exploratory: pinning the geometry raised the geometric agreement to 0.70 but improved the head's function by only 16% of the gap and cost the other reading 0.13 nats, N4's cost of sharing again. Whether anchoring function beats pinning geometry is still open. See `N10_NOTE.md`.

**N11 result, 2026-10-20: N11a and N11c untestable, N11b supported.** A functional payload at three strengths, on the second head's readout only, closed 40%, 45% and 47% of the gap against a fixed yardstick on new text, rising with the strength in every seed but flattening. It cost the clean reading nothing. It still failed its manipulation check (0.59 of the unanchored divergence, against at most 0.5): the dose check had covered only the first tenth of training, and its rising trend was misread as a level. Secondary measures: the geometric pin restored the directions (agreement 0.73) but only 19% of the function, and cost the clean reading 0.14 nats; the functional payload restored function without restoring the directions (0.35, against 0.32 unanchored). Function and geometry came apart. Whether anchoring function beats pinning geometry is still formally open. A comparison with the interrupted T4 runs (2026-10-21) showed that the last step of the rise, from λ = 32 to 128, is within the GPU's own effect. See `N11_NOTE.md` and `N11_HARDWARE.md`.

**N12 result, 2026-10-21: N12a, N12c and N12d supported, N12b intermediate.** The head trained on 80% noise reads differently from the clean head for two reasons, each now bounded: its readout cannot represent what it is trained to predict, which accounts for at most 38% of the gap, and training on noisy targets through that readout accounts for the rest. A head whose output is the noisy mixture itself, trained on the same noisy targets, closed 71% of the gap to the ideal copy (N12a); a clean head given the same informative targets and no random ones closed 80% (N12c); a single softmax fitted directly to the copy stopped at 62% (N12d), so the form's share of the drift is at most 38%, and training on noisy targets through that form reaches none of the 62% a single softmax can. The mixture head's directions moved from 0.31 to 0.83 agreement with the copy, under the 0.85 bar in every seed (N12b intermediate). Secondary: the standard noisy head cost the clean head 0.04 nats [0.03, 0.05]; the mixture head's cost was −0.01 [−0.02, +0.01]. See `N12_NOTE.md`.

**N13 result, 2026-10-22: N13a, N13b and N13c supported.** The mixture head's advantage is its learning signal. By the chain rule, training through the mixture form is a discount on each target by the head's own posterior that the target is clean, and the pipeline follows that identity: a single softmax trained with the discount matched the mixture head's inner softmax within 0.005 nats in every seed (N13a). The form applied at readout to the standard head, trained without the discount, doubled its gap to the copy (S = −1.01, N13b). A constant discount of 0.2, the real discount's mean weight (0.19), gave head B's reading none of the gain (D = +0.05 [+0.04, +0.06], N13c) and, an exploratory reading, removed head B's cost to head A within the measure's noise (−0.05 [−0.07, −0.04], 95%). On this evidence the cost to head A tracks how hard head B pulls, though `sub20`, with clean targets on a fifth of the positions, cost more than the standard head scaled to a fifth, which is not understood; head B's gain is a matter of which targets it trusts: by the end of training the discount weighted a kept target 5.6 times a replaced one. N12's two effects are two roles of one form: the discount makes the inner reading clean, and the form at readout expresses the noisy-target prediction from it. See `N13_NOTE.md`.

**N14 result, 2026-10-22: N14a inconclusive, N14b intermediate, N14c refuted; the conjecture did not survive.** Heads trained with 10, 20, 40 and 80% random targets at a fixed 20% of clean ones read head A's directions as a clean-target head with the same clean-target loss does, within 0.02: against a ladder of heads trained on 20, 10, 5, 2.5 and 1.25% of clean targets and no random ones, their residuals were 0.019, 0.008, 0.016 and 0.009, every interval including 0 (N14a: +0.009 [−0.035, +0.052], inconclusive by its rule; the design could resolve 0.06). The ladder crosses the 0.85 bar at a clean-target loss 0.26 [0.23, 0.28] nats from head A's; the mixture head's inner reading is 0.34 away, so N12b's near miss needs no geometric trace. The trace at the smallest dose was about half of the trace at the largest (N14b intermediate), and the trace and the functional cost both grew with the random targets' share of the gradient weight at the end of training (0.14 to 0.42). A head given the discount's reweighting of the kept targets and no random targets lost 0.125 [0.080, 0.169] of agreement (N14c refuted as stated), with its clean-target loss collapsed to near a 1.25% clean head's while its directions stayed a 5% head's: 0.10 above the ladder, not below it (a secondary measure, read as exploratory). Exploratory: head A's cost against the clean arm fell with the dose (+0.020 to −0.009) and with fewer clean targets (+0.020 to −0.020). See `N14_NOTE.md`.

G1 comes first, because it establishes that regimes can be produced and measured at all. G2 and G3 need TV1's tracking measure too.

| Outcome | What we would learn | Next |
|---|---|---|
| G1: distinct regimes appear when the control is trained in, and not in the controls | The paper's geometric picture holds for a properly trained model, with a direct measure of geometry. This is a new result; H6's refutation of the original model stands. | G2 |
| G1: no distinct regimes even when trained for | Control in this architecture shifts the dynamics without partitioning them, at least by these measures. | Test multiplicative control (G3) before giving up the geometric picture. |
| G2: regime distinctness predicts tracking | Geometry is how the model keeps its voices apart, and gives a design target. | G3 |
| G2: no relation | Voices are kept apart some other way, if at all. The timbre analogy does not carry over. | Return to V2's explicit-voice design. |

## Rivals

| Rival | What it predicts that differs | How the test tells them apart |
|---|---|---|
| Length, not voices | Accuracy falls with the number of tokens, whatever k is | A length-matched control: the same length with fewer strands, padded with neutral sentences |
| Shortcuts: the answer is the last mention | Accuracy high whenever the queried strand was mentioned recently | Query points chosen so the last mention does not give the answer; report accuracy by distance since the last mention |
| Too few items, or chance | Apparent slopes from noise | Preregistered item counts per k, with the chance range checked in advance |
| Matched budgets not matched | V2's model wins by having more parameters or compute | Match parameters and compute, with equal tuning, as in T1 |
| Training defects | A continuous design loses because of its gradients | Exact gradients, after T1 |

## Proposed tests, in order

**TV1. Measure V1 on existing models, with no training.**
- **Texts:** generated from templates, with k = 1 to 8 interleaved strands, such as characters whose locations, possessions or moods change. Each text is checked by code to have one correct state per strand at every query point.
- **Queries:** at chosen points, ask each strand's current state, by prompting and by probing the hidden states, with the shortcut controls above.
- **Models:** public pretrained models at a few sizes, which tests the measure against scale, plus our own small WikiText-103 models.
- **Primary outcome:** the slope of accuracy against k at matched length.
- **Controls:** must fail, shuffled answers and an untrained model at chance; must match, near-perfect accuracy at k = 1 for the larger models.
- **Cost:** runs on a CPU here, or briefly on Colab for larger models.

**TV2. Test V2 by training.** Small transformer and polyphonic models, matched, trained on the same pinned data with strand-tracking texts mixed in, using TV1's measure.

**TV3. Test V3 by switching the coupling off,** on texts with and without interacting strands.

## What each outcome would teach

| Outcome | What we would learn | Next |
|---|---|---|
| TV1: a clear limit set by voices, worse with crossing | Models have a polyphony limit like listeners'. The measure works, and the target is concrete. | TV2: does a continuous, separate-voice design raise the limit? |
| TV1: the limit grows with scale and vanishes in large models | Polyphonic reading emerges with scale in standard models. | Ask what small models lack, and whether a design gives it to them cheaply, which is TV2 at small scale. |
| TV1: no effect of voices, only of length | Models are limited by length, not by holding several strands at once, at least by this measure. The analogy to listening may not carry over. | Revise V1, for example toward interacting strands (V3), before any design work. |
| TV2: the polyphonic model's decline is smaller | Separate, continuous voices make holding many strands easier to learn. | TV3, and then natural texts. |
| TV2: no difference | The design adds nothing at this scale. | Look at where errors fall (P2, P3) before trying other designs. |
| Controls fail | The measure isn't trustworthy. | Repair the measure first. |

## What this note does not claim

- That word order is a defect or should change. Order stays; the aim is to follow many strands at once within it.
- That the continuous-depth model addressed this aim. It did not, and its results stand as they are.
- Novelty. Each component has prior art above; only their combination for this purpose appears untried in what was searched, and the search should be widened before any claim.

## References

Drożdż, K., & Heilbron, M. (2026). *Entity tracking emerges in sub-billion parameter language models and exceeds human performance in naturalistic narratives*. arXiv. https://doi.org/10.48550/arxiv.2608.18083 (preprint)

Finlay, C., Jacobsen, J.-H., Nurbekyan, L., et al. (2020). *How to train your neural ODE: The world of Jacobian and kinetic regularization*. arXiv. https://doi.org/10.48550/arxiv.2002.02798

Goyal, A., Lamb, A., Hoffmann, J., et al. (2019). *Recurrent independent mechanisms*. arXiv. https://doi.org/10.48550/arxiv.1909.10893

Huron, D. (1991). The avoidance of part-crossing in polyphonic music: Perceptual evidence and musical practice. *Music Perception, 9*(1), 93–103. https://doi.org/10.2307/40286160

Ji, Y., Tan, C., Martschat, S., et al. (2017). Dynamic entity representations in neural language models. In *Proceedings of EMNLP 2017* (pp. 1830–1839). https://doi.org/10.18653/v1/d17-1195

Kelly, J., Bettencourt, J., Johnson, M. J., et al. (2020). *Learning differential equations that are easy to solve*. arXiv. https://doi.org/10.48550/arxiv.2007.04504

Kidger, P., Morrill, J., Foster, J., et al. (2020). *Neural controlled differential equations for irregular time series*. arXiv. https://doi.org/10.48550/arxiv.2005.08926

Locatello, F., Weissenborn, D., Unterthiner, T., et al. (2020). *Object-centric learning with slot attention*. arXiv. https://doi.org/10.48550/arxiv.2006.15055

Siedenburg, K., Goldmann, K., & van de Par, S. (2021). Tracking musical voices in Bach's *The Art of the Fugue*: Timbral heterogeneity differentially affects younger normal-hearing listeners and older hearing-aid users. *Frontiers in Psychology, 12*, 608684. https://doi.org/10.3389/fpsyg.2021.608684

Author lists after the third author were not verified.
