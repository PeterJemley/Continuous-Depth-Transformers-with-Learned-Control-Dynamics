# The simultaneity conjecture: the tests

This folder holds the tests of the project's own conjecture, the author's: that a model could hold several readings of a text at once, as a listener follows several voices in music, and that continuous dynamics like the paper's block might help it do so. The tests start small. They ask how one model holds two readings, such as predicting the next word and the word after it.

Each test's plan, its decision rules and the result that would count against it were written down and approved by the author before it ran. The tests were designed, run and written up by Claude, an AI model made by Anthropic; where these documents say "I", the writer is Claude.

**Start with [`PLAIN_SUMMARY.md`](PLAIN_SUMMARY.md).** It tells the whole story in plain language. [`CRITICAL_REVIEW.md`](CRITICAL_REVIEW.md) lists every claim tested, what it forbade and its verdict, and where the method fell short.

## How the tests are set up

- **The model** is the paper's, about 30 million parameters, trained with exact gradients on WikiText-103 at a tenth of the full training budget. Most arms are trained on five seeds, and each verdict comes from intervals or counts across them.
- **In G1 to N1** one model performs one reading at a time, and the control signal switches it between them. **From N2 on** the model has two heads that perform both readings at once from the same internal state.

## Words used throughout

- **Reading:** a task the model performs from its internal state, such as predicting the next word.
- **Head:** the part of the model that turns the shared internal state into one reading's prediction.
- **Readout:** the last step of a head, which maps the state to a probability for every word.
- **Directions:** the internal state is a list of 256 numbers. A reading depends on some directions in that space more than others, and the tests ask how much two readings rely on the same ones. Two measures are used: the overlap (N4–N7) and the agreement ratio (N8 on). Both are 1 when two readings rely on the same directions; unrelated directions score about 0.06 on the first and 0.1–0.16 on the second.
- **Scrambled answers:** training answers replaced by words drawn at random in proportion to how common they are. "80% scrambling" means four answers in five were replaced. A head trained this way is also called the noisy head.
- **The ideal noisy copy:** the prediction a perfect head trained on scrambled answers would make: one fifth the clean head's prediction plus four fifths word frequencies, at 80% scrambling.
- **Nats:** the unit of prediction loss. A difference of 0.05 nats is about a 5% change in how uncertain the model is about the next word.
- **Arm:** one condition of a test, such as one kind of head. **Seed:** one independent training run.
- **Verdicts:**
  - *supported* or *refuted*: the result met the preregistered rule for support, or the one for refutation, such as an interval lying wholly past a margin;
  - *some effect, size unclear*: an interval excluded zero but met neither margin;
  - *inconclusive*: neither;
  - *intermediate*: a count fell between the support and refutation thresholds;
  - *untestable*: a precondition failed, such as a treatment that did not take hold, so the claim could not be judged.

## The tests

Each row links to the test's note, the account to read first.

| Test | The question | Verdicts | Note |
|---|---|---|---|
| G1 | Does the control signal send the flow through regions of different curvature, as the paper suggested? | Refuted | [G1](G1_NOTE.md) |
| G1c | Does it set how strongly the flow pulls nearby states together? | Supported, and replicated | [G1c](G1C_NOTE.md) |
| G1d | Does the model need that pull? | Refuted | [G1d](G1D_NOTE.md) |
| N1 | Does switching between readings need the dynamics, or only a fixed shift of the state? | Supported: a fixed shift suffices | [N1](N1_NOTE.md) |
| N2 | When two readings are held at once, does the flow give each its own pace? | Refuted | [N2](N2_NOTE.md) |
| N4 | Are two readings held at once kept apart by direction, and does the model need that? | Supported | [N4](N4_NOTE.md) |
| N5 | Is N4's cost a competition for directions? | Inconclusive | [N5](N5_NOTE.md) |
| N6 | Does how much two readings share depend on how different they are? | Untestable: its control was set wrongly | [N6](N6_NOTE.md) |
| N6r | N6 again, with the control fixed | Supported | [N6r](N6R_NOTE.md) |
| N7 | Is it how different the readings are, or how hard? | Refuted, both claims | [N7](N7_NOTE.md) |
| N8 | Does the pattern survive a measure that leaves the right answers out? | Two claims refuted; the ladder across distances supported | [N8](N8_NOTE.md) |
| N9 | Can the measure recognise an ideal noisy copy, and is the scrambled head one? | The measure supported; the head is not a copy (refuted) | [N9](N9_NOTE.md) |
| N10 | How much scrambling does it take, and can a small payload protect a reading? | Two untestable, one intermediate, one supported | [N10](N10_NOTE.md) |
| N11 | Does a stronger payload work, judged against a yardstick it cannot move? | Two untestable, one supported | [N11](N11_NOTE.md) |
| N12 | Is the scrambled head held back by the shape of its readout? | Three supported, one intermediate | [N12](N12_NOTE.md) |
| N13 | Does the blended readout help by representing the blend, or by how it learns? | Three supported | [N13](N13_NOTE.md) |
| N14 | Do scrambled answers leave a trace in the directions that accuracy does not show? | Inconclusive, intermediate, refuted | [N14](N14_NOTE.md) |
| N15 | Would a sharper discount remove the remaining shortfall? | Withdrawn before any real run | [plan](N15_PREREGISTRATION.md) |
| N16 | Is the remaining shortfall in who judges the answers? | Three supported, one inconclusive | [N16](N16_NOTE.md) |

**Gaps and oddities in the numbering.**
- **There is no N3.** It was planned in [`PATHS_AND_UNDERSTANDING.md`](PATHS_AND_UNDERSTANDING.md) as control that multiplies parts of the flow instead of adding to it, and N2's note named it as the test to run before leaving the dynamics. It was not run.
- **N6r** is N6 run again with its control corrected.
- **N15** was approved and calibrated, then withdrawn by the author before any real run, when its calibration showed its rules could not be applied. It has a plan and a calibration report but no note.
- **The plan's numbers differ from what ran:** in `PATHS_AND_UNDERSTANDING.md`, "N4" names a different planned test from the N4 that ran.
- **T0 and T1**, on why the paper's model fell short, are in [`../too-linear-conjecture/`](../too-linear-conjecture/).

## What each file is

For a test named, say, N16:

| File | What it is |
|---|---|
| `N16_PREREGISTRATION.md` | The plan, fixed before any run: the claims, what would refute them, the decision rules. It ends with a deviation log of everything that happened after approval. |
| `N16_CALIBRATION.md` | From N12 on: a report on one full-length run, on a seed and windows the real run does not use, made before the real run to check that the design works. |
| `N16_RESULTS.md` | The verdicts, generated by code from the records under the preregistered rules. It is not edited by hand. |
| `N16_NOTE.md` | The account to read: what was expected, what happened, what kind of result it is, and what it teaches. |
| `N16_EXPLORATORY.md` | From N10 on: analyses written after the results were seen. They decide nothing. |
| `n16_judge.py` | The job that trains and measures the models, run on a Colab GPU (see [`../../colab/README.md`](../../colab/README.md)). |
| `n16_analyze.py` | Applies the decision rules to the records and writes `N16_RESULTS.md`. Running it again regenerates the report. |
| `results/n16/` | One JSON record per training run and measurement, as uploaded from the Colab runs. |

Other documents:
- [`CONJECTURE.md`](CONJECTURE.md): the conjecture as the author put it, with a dated result line for each test.
- [`PATHS_AND_UNDERSTANDING.md`](PATHS_AND_UNDERSTANDING.md): the plan of tests written after G1d, with result lines.
- [`NECESSITY_AUDIT.md`](NECESSITY_AUDIT.md): from N12 on, a check of what each design relies on, made before approval.
- `N11_HARDWARE.md`: N11's runs on a T4 GPU against its runs on an A100, exploratory.
