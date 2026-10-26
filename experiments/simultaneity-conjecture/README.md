# The simultaneity conjecture: the tests

This folder holds the tests of the project's own conjecture, the author's: that a model could hold several readings of a text at once, as a listener follows several voices in music, and that continuous dynamics like the paper's block might help it do so. The tests start small. They ask how one model holds two readings, such as predicting the next word and the word after it.

Each test's plan, its decision rules and the result that would count against it were written down and approved by the author before it ran. The tests were designed, run and written up by Claude, an AI model made by Anthropic; where these documents say "I", the writer is Claude.

**Start with [`PLAIN_SUMMARY.md`](PLAIN_SUMMARY.md).** It tells the whole story in plain language. [`CRITICAL_REVIEW.md`](CRITICAL_REVIEW.md) lists every main claim tested, what it forbade and its verdict, and where the method fell short.

## How the tests are set up

- **The model** is the paper's Model A, trained with exact gradients on WikiText-103 at a tenth of the full training budget, with one learning rate: about 30 million parameters, or 23 to 43 million with two reading heads, most of them in the tables that map words to vectors and back. The readings tested are close relatives, words one to eight places ahead. Most arms are trained on five seeds, and each verdict comes from intervals or counts across them.
- **In G1 to N1** one model performs one reading at a time, and the control signal switches it between them. **From N2 on** the model has two heads that perform both readings at once from the same internal state.

## Words used throughout

- **Reading:** a task the model performs from its internal state, such as predicting the next word.
- **Head:** the part of the model that turns the shared internal state into one reading's prediction.
- **Readout:** the last step of a head, which maps the state to a probability for every word.
- **Directions:** the internal state is a list of 256 numbers. A reading depends on some directions in that space more than others, and the tests ask how much two readings rely on the same ones.
  - **Overlap** (N2 on): 1 for identical directions, about 0.06 for random ones. Sampling noise keeps even two copies of one reading a little below 1 (about 0.95–0.99).
  - **Agreement ratio** (N6r on): the overlap divided by each reading's own stability between two halves of the data, so that the same directions score about 1. Unrelated directions, a rotated control, score about 0.1 to 0.2.
  - **From N8 on,** both are computed without the right answers, from what each head predicts.
- **Scrambled answers:** training answers replaced by words drawn at random in proportion to how common they are. "80% scrambling" means four answers in five were replaced. A head trained this way is also called the noisy head.
- **The ideal noisy copy:** the prediction a perfect head trained on scrambled answers would make: one fifth the clean head's prediction plus four fifths word frequencies, at 80% scrambling.
- **Nats:** the unit of prediction loss. A difference of 0.05 nats is about a 5% change in how uncertain the model is about the next word.
- **Payload** (N10, N11): a small extra training signal meant to protect a reading under scrambling, such as matching the ideal noisy copy on a few passages.
- **Discount** (N13 on): each training answer weighted by an estimate that it is genuine. **The judge** (N16) is whichever head makes that estimate. **The shortfall** is how far the discounted head falls short of a head given only the genuine answers.
- **Ladder:** in N6r–N8, the series of readings one, two, four and eight words ahead; in N14 and N16, a series of heads given fewer and fewer genuine answers and no scrambled ones.
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
| N1 | Does switching between readings need the dynamics, or only a fixed shift of the state? | Supported: a fixed shift suffices (a secondary claim untestable) | [N1](N1_NOTE.md) |
| N2 | When two readings are held at once, does the flow move at a different speed for each? | Refuted (a secondary claim untestable) | [N2](N2_NOTE.md) |
| N4 | Are two readings held at once kept apart by direction, and does the model need that? | Supported | [N4](N4_NOTE.md) |
| N5 | Is N4's cost a competition for directions? | Inconclusive | [N5](N5_NOTE.md) |
| N6 | Does how much two readings share depend on how different they are? | Untestable: a control that had to score two copies of one reading as the same reached 0.9498, just under its 0.95 threshold | [N6](N6_NOTE.md) |
| N6r | N6 again, with the control fixed | Supported | [N6r](N6R_NOTE.md) |
| N7 | Does sharing track how different two readings are, rather than how hard? | Refuted, both claims: a scrambled copy of the next-word reading shared barely more than a reading four words ahead | [N7](N7_NOTE.md) |
| N8 | Does the pattern survive a measure that leaves the right answers out? | Yes: the scrambled head still shared little and did not converge (two claims that it would, refuted), and the ladder across distances held (supported) | [N8](N8_NOTE.md) |
| N9 | Can the measure recognise an ideal noisy copy, and is the scrambled head one? | The measure supported; the head is not a copy (refuted) | [N9](N9_NOTE.md) |
| N10 | How much scrambling does it take, and can a small payload protect a reading? | Drift grows with scrambling (supported); at 20% it is intermediate; the payload claims untestable, as the payloads did not take hold | [N10](N10_NOTE.md) |
| N11 | Does a stronger payload work, judged against a fixed reference? | Two untestable, as the payload again failed its check; one supported, with sharply diminishing returns | [N11](N11_NOTE.md) |
| N12 | How much of the scrambled head's drift is the shape of its readout? | Three supported, one intermediate: the shape accounts for at most two fifths of the gap | [N12](N12_NOTE.md) |
| N13 | Does the blended readout help by representing the blend, or by how it learns? | Three supported: by how it learns, through the discount | [N13](N13_NOTE.md) |
| N14 | Do scrambled answers leave a trace in the directions that accuracy does not show? | Inconclusive, intermediate, refuted; the numbers favoured no trace | [N14](N14_NOTE.md) |
| N15 | Would a sharper discount remove the remaining shortfall? | Withdrawn before any real run | [plan](N15_PREREGISTRATION.md) |
| N16 | Is the remaining shortfall in who judges the answers? | Three supported, one inconclusive: judged by the clean head, the shortfall vanished; whether independence, accuracy or sharpness matters is open | [N16](N16_NOTE.md) |
| N17 | Was it the judge's accuracy, or the loop of a head judging answers by its own predictions? | Some effect, size unclear; one untestable: a judge with no loop, as accurate as the head itself, recovered a third of the shortfall | [N17](N17_NOTE.md) |

**Gaps and oddities in the numbering.**
- **There is no N3.** It was planned in [`PATHS_AND_UNDERSTANDING.md`](PATHS_AND_UNDERSTANDING.md) as control that multiplies parts of the flow instead of adding to it (G3 in `CONJECTURE.md`), and N2's note named it as the test to run before leaving the dynamics. It was not run.
- **N6r** is N6 run again with its control corrected.
- **N15** was approved and calibrated, then withdrawn by the author before any real run, when its calibration showed its rules could not be applied. It has a plan and a calibration report but no note.
- **The plan's numbers differ from what ran:** in `PATHS_AND_UNDERSTANDING.md`, "N4" names a different planned test from the N4 that ran.
- **T0 and T1**, on why the paper's model fell short, are in [`../too-linear-conjecture/`](../too-linear-conjecture/).

## What each file is

For a test named, say, N16:

| File | What it is |
|---|---|
| `N16_PREREGISTRATION.md` | The plan, fixed before any run: the claims, what would refute them, the decision rules. Its deviation log records everything that happened after approval. |
| `N16_CALIBRATION.md` | From N12 on: full-length runs of each new arm on one seed, measured on text the real run does not use, made before the real run to check that the design works. Written by the analysis script. |
| `N16_RESULTS.md` | The verdicts, generated by code from the records under the preregistered rules. It is not edited by hand. |
| `N16_NOTE.md` | The account to read: what was expected, what happened, what kind of result it is, and what it teaches. |
| `N16_EXPLORATORY.md` | From N10 on: analyses written after the results were seen, generated by `n16_exploratory.py`. They decide nothing. |
| `n16_judge.py` | The job that trains and measures the models, run on a Colab GPU (see [`../../colab/README.md`](../../colab/README.md)). Each test's job has its own name, such as `g1_regimes.py` or `n4_directions.py`. |
| `n16_analyze.py` | Applies the decision rules to the records and writes `N16_RESULTS.md`. Running it again regenerates the report. |
| `results/n16/` | One JSON record per training run and measurement, from the Colab runs. Calibration records are in `results/n16_calib/`, and earlier tests' run logs sit beside their folders. |

Other documents:
- [`CONJECTURE.md`](CONJECTURE.md): the conjecture as the author put it, with a dated result line for each test.
- [`PATHS_AND_UNDERSTANDING.md`](PATHS_AND_UNDERSTANDING.md): a note written while G1d ran; its section "After G1d" plans N1–N4. With result lines.
- [`NECESSITY_AUDIT.md`](NECESSITY_AUDIT.md): from N12 on, a check of what each design relies on, made before the real run (for N12 after approval, from N13 at drafting).
- `N11_HARDWARE.md`: N11's runs on a T4 GPU against its runs on an A100, exploratory.
- [`make_figures.py`](make_figures.py) and [`figures/`](figures/): the plain summary's figures, drawn from the records, with every plotted value in [`figures/FIGURE_DATA.md`](figures/FIGURE_DATA.md).
