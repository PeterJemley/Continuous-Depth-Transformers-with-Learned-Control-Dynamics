# Conjecture note: perceiving language at once, not as an extrusion

Date: 2026-10-05    Status: conjecture, untested

Written with the scientific-experimentation skill (conjecture-note template). Nothing here is a finding.

## The question

**As the author put it.** Sight and hearing let us perceive more than one thing at a time. Language reaches us "extruded": one word after another, in an order grammar imposes, such as subject, verb, object. That order is correct, but it limits how meaning is perceived. The aim is not to fix language. It is to let an AI perceive a text's dimensions, ideas and interconnections as smoothly and naturally as sight and hearing take in a scene or a chord. This is why the project began.

**Restated against the systems at hand.** A text is a one-dimensional sequence of tokens. Is the meaning a model builds tied to the order the text arrived in, or does the model recover the order-independent structure of what is said: who did what to whom, and how the ideas relate? And would a different design let it do so more completely, earlier in its depth, or from less data?

Linguistics has a name for the speaker's side of this: the linearization problem. Thought can be multidimensional, but speech must put it in a single order. Levelt (1981) put the open question this way: "An important question is whether the linearization requirement is enforced by the oral modality of speech or whether it is a deeper modality-independent property of language use." This project asks the listener's side of the same question, for machines.

## Readings

"Extruded" has several readings in today's language models. They make different predictions, so they are kept apart.

| Reading | Where it shows | What it predicts |
|---|---|---|
| E1. Reading is serial | **Not true of transformers.** Self-attention takes in the whole context at once and compares every position with every other in one step. The models here do not read word by word. | Nothing. This reading is set aside. The extrusion, if any, lies elsewhere. |
| E2. Seeing is one-way | Language models use a causal mask: each position sees only what came before it, as hearing receives sound in time. A word cannot be understood in the light of what follows it until a later position. | Meaning that depends on later words is assembled late, or only at the end of the sentence. |
| E3. Position is a line | Each token's position is a single coordinate on a line. Structure such as a tree, or a graph of relations, has to be recovered from that line. | Paraphrases that put the same meaning in different orders give different representations, at least in early layers. |
| E4. Output is serial | Generation produces one token at a time. | Outside this note's scope, which is about perception, not production. |

## What the evidence already says

- **Models already recover some structure the line hides.** Hewitt and Manning (2019) found that whole syntax trees are embedded in the geometry of BERT's and ELMo's word vectors: "such transformations exist for both ELMo and BERT but not in baselines, providing evidence that entire syntax trees are embedded implicitly in deep models' vector geometry." So the sequence is not simply what the model keeps. Some of it is undone. Whether meaning beyond syntax is recovered independently of order is the open part.
- **Human languages use simultaneity where the medium allows it.** Sign languages layer meaning across both hands, the face and space at once (Loos et al., 2022). That is natural evidence that some of language's linearity comes from the vocal channel, as Levelt asked. They are also "unexpectedly sequential" in places, and fully simultaneous production of two predicates is rare (Dushkina, 2019). So simultaneity has limits in cognition too, not only in speech. Any conjecture that assumes unlimited simultaneity is criticised by this.
- **Meaning can be written down without order.** Abstract Meaning Representation writes a sentence's meaning as a graph that "abstracts away from syntactic features such as word order" (Azin & Eryiğit, 2019). It gives one concrete target for order-independent meaning.
- **This project's own results don't test the inspiration.** The continuous-depth block changed how a representation evolves through the model's depth, not how the model takes in the sequence: it saw the same tokens, in the same order, under the same causal mask. Its studies refuted the paper's claims and, through T1, traced most of its loss gap to a training defect. Neither says anything for or against perceiving language at once. The project's first design addressed a different axis from the one that inspired it. That is worth knowing.

## Conjectures

**S1, descriptive: trained models already partly perceive meaning independently of order.** At some depth, a trained language model's representation of a sentence is similar for paraphrases that say the same thing in different orders, and different for sentences with the same words in roles that change the meaning.

- Paraphrases: "The dog chased the cat" and "The cat was chased by the dog"; "She gave the book to him" and "She gave him the book".
- Role swaps: "The dog chased the cat" and "The cat chased the dog".
- Consequence: a linearization-invariance score, paraphrase similarity minus role-swap similarity, rises above zero at some layer and grows with depth.
- Forbids: no layer, in any model tested, where paraphrases are reliably more similar than role swaps.
- What kind of claim: a measurement of existing models, the foundation every later test needs.

**S2, design: the one-way view is part of what extrudes meaning.** A model allowed to see in both directions builds order-independent meaning earlier in its depth, and from less training data, than a causal model of the same size.

- This is a claim about learnability. A causal model can represent the same meaning by the sentence's last word.
- Consequence: matched bidirectional and causal models, trained on the same data, differ in how early and how strongly S1's score appears.
- Forbids: matched models with the same invariance profile.

**S3, the inspiration as a design.** A model that holds a text as a set of units whose interactions are modulated by what they mean perceives the text's relations more completely than one that holds it as a sequence. This draws on the earlier "too linear" conjecture C3. Its test depends on S1's measure and on knowing what S2 shows, so it comes last.

## Rivals

| Rival | What it predicts that differs | How the test tells them apart |
|---|---|---|
| Word overlap | Paraphrases share more words than unrelated sentences, so similarity rises for that reason alone | Role swaps share every word. The score subtracts them, so overlap alone gives a score near zero. |
| Length and position | Pairs of different length or with words moved differ for surface reasons | Match lengths in the item set, and compare against a bag-of-words baseline and an untrained model. |
| The probe or similarity measure creates the effect | Any model, even an untrained one, shows a positive score | An untrained model of the same architecture must score near zero. That is the must-fail control. |
| Too few items | Chance produces a positive score | Preregister at least 200 pairs of each kind, and check the chance range in advance. |

## Prior art

- Levelt (1981), the speaker's linearization problem.
- Hewitt and Manning (2019), syntax trees in the geometry of word vectors.
- Loos et al. (2022) and Dushkina (2019), simultaneity in sign languages and its limits.
- Abstract Meaning Representation, described by Azin and Eryiğit (2019).

None of these measures order-independent meaning directly, in the sense S1 does, as far as this search found. That is a statement about the search, not a novelty claim. Paraphrase and sentence-similarity work exists and should be searched further before any claim of novelty.

## Proposed tests, in order

**TS1. Measure S1 on existing models.**
- Models: our WikiText-103 baseline and Model A checkpoints, and public pretrained models of a few sizes, so the measure is checked across scales.
- Items: a preregistered set of at least 200 paraphrase pairs and 200 role-swap pairs, built from templates and checked by hand, with lengths matched.
- Primary outcome: the invariance score by layer, with an interval across items.
- Must fail: the same architecture untrained, and a bag-of-words representation, both scoring near zero.
- Must match: a sentence compared with itself gives the maximum.
- Cost: no training. It runs on a CPU in this environment once the models' weights are downloaded.

**TS2. S2, if TS1 finds a usable measure.** Matched small causal and bidirectional models trained on the same pinned data, with S1's score as the primary outcome.

**TS3. S3, only after TS1 and TS2.**

## What each outcome would teach

| Outcome | What we would learn | Next |
|---|---|---|
| TS1: invariance appears and grows with depth | Models already partly perceive meaning independently of order. The measure works and can steer design. | TS2: does the one-way view limit it? |
| TS1: invariance only in large models | Order-independent meaning emerges with scale; small models stay closer to the sequence. | Design tests at a scale where it appears, or ask what small models lack. |
| TS1: no invariance in any model | Today's models keep meaning tied to order, at least by this measure. The inspiration names a real gap. | Check the measure against AMR-annotated sentences, then go straight to design tests such as S3. |
| TS1: the controls fail | The measure is not trustworthy. | Redesign the measure before any conclusion. |

## What this note does not claim

- That language should be changed, or that serial order is a defect. Order carries meaning: "she married and became pregnant" and "she became pregnant and married" differ, as Levelt notes.
- That the continuous-depth model addressed this aim. It addressed another axis, and its results stand as they are.
- Novelty, beyond what the search shows.

## References

Azin, Z., & Eryiğit, G. (2019). Towards Turkish Abstract Meaning Representation. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop* (pp. 43–47). https://doi.org/10.18653/v1/p19-2006

Dushkina, V. A. (2019). Simultaneity in polypredicative constructions in Russian Sign Language. *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.3499645

Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. In *Proceedings of NAACL-HLT 2019* (pp. 4129–4138). https://doi.org/10.18653/v1/n19-1419

Levelt, W. J. M. (1981). The speaker's linearization problem. *Philosophical Transactions of the Royal Society of London. B, Biological Sciences, 295*(1077), 305–315. https://doi.org/10.1098/rstb.1981.0142

Loos, C., German, A., & Meier, R. P. (2022). Simultaneous structures in sign languages: Acquisition and emergence. *Frontiers in Psychology, 13*, 992589. https://doi.org/10.3389/fpsyg.2022.992589
