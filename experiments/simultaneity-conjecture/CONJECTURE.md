# Conjecture note: reading language polyphonically

Date: 2026-10-05    Status: conjecture, untested

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

Goyal, A., Lamb, A., Hoffmann, J., et al. (2019). *Recurrent independent mechanisms*. arXiv. https://doi.org/10.48550/arxiv.1909.10893

Huron, D. (1991). The avoidance of part-crossing in polyphonic music: Perceptual evidence and musical practice. *Music Perception, 9*(1), 93–103. https://doi.org/10.2307/40286160

Ji, Y., Tan, C., Martschat, S., et al. (2017). Dynamic entity representations in neural language models. In *Proceedings of EMNLP 2017* (pp. 1830–1839). https://doi.org/10.18653/v1/d17-1195

Kidger, P., Morrill, J., Foster, J., et al. (2020). *Neural controlled differential equations for irregular time series*. arXiv. https://doi.org/10.48550/arxiv.2005.08926

Locatello, F., Weissenborn, D., Unterthiner, T., et al. (2020). *Object-centric learning with slot attention*. arXiv. https://doi.org/10.48550/arxiv.2006.15055

Siedenburg, K., Goldmann, K., & van de Par, S. (2021). Tracking musical voices in Bach's *The Art of the Fugue*: Timbral heterogeneity differentially affects younger normal-hearing listeners and older hearing-aid users. *Frontiers in Psychology, 12*, 608684. https://doi.org/10.3389/fpsyg.2021.608684

Author lists after the third author were not verified.
