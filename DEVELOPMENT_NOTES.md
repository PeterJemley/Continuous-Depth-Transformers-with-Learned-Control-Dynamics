# Development Notes: A Falsificationist Approach to Continuous-Depth Transformers

This document outlines the **problem situations**, **tentative theories**, and **critical tests** that shaped the final paper. It moves away from inductive assertion ("this works because we say so") toward a deductive approach: proposing a model and attempting to break it.

---

## 1. The Problem Situation and The Conjecture

### The Shift from Inductivism to Critical Rationalism

**Original Flaw:** The initial proposal suffered from verificationism. It asserted that "language is nonlinear → neural ODEs are appropriate" without specifying conditions under which this would be false. It lacked **potential falsifiers**—concrete failure modes that would refute the theory.

**The Popperian Correction (Key Revisions):**
We redefined the project around a **bold conjecture**: *Control dynamics can be learned end-to-end within the training process itself, rather than discovered post-hoc.*

* **Demarcation:** We distinguished our theory from existing work (Neural ODEs, Activation Steering) by identifying a specific new capacity: the ability to learn a control manifold during training.
* **Risky Prediction:** We predicted that a hybrid model could achieve control *without* the massive compute overhead usually associated with ODEs. This was a risky prediction because standard ODE theory suggests a high computational cost.

---

## 2. Severe Testing (Experimental Validation)

We subjected the "Continuous-Depth" conjecture to four **crucial experiments**. A crucial experiment is one designed to yield a result that is inconsistent with the theory if the theory is false.

### Test 1: Stability and Efficiency (Attempting to Falsify Viability)

**The Null Hypothesis:** Integrating an ODE solver into a transformer will lead to vanishing gradients or exploding computational costs due to the recursive nature of the solver.

**The Test:** We replaced layers 2-3 with an ODE flow (4 Euler steps) and trained on WikiText-2.

**Corroboration:**
The theory survived the test. Not only did gradients remain stable (Norm: 0.509 vs Baseline 0.521), but the model also achieved *better* loss (6.449 vs 6.471) with *fewer* parameters.
* *Note:* Popper would argue this result is highly corroborative because it was "improbable" given the background knowledge of ODE computational heaviness.

### Test 2: The "Hybrid Unfreeze" (Isolating the Mechanism)

**The Risk:** The model might appear to steer effectively by "cheating"—learning to associate the control vector $u$ with specific output tokens directly, bypassing the continuous dynamics. This would be an *ad hoc* adjustment.

**The Severe Test:** We froze the embeddings and output head, forcing the model to learn steering *solely* through the ODE block vector field. If the ODE block could not carry the semantic load, the experiment would fail.

**Result:** The model successfully steered sentiment (98% Positive vs 88% Negative accuracy). The theory that "dynamics alone can carry semantics" was corroborated.

### Test 3: The Manifold Hypothesis (Testing for Continuity)

**The Conjecture:** The control signal $u$ operates on a continuous manifold, not as a binary switch.
**Potential Falsifier:** If sweeping $u$ produced a step-function (abrupt jump from "Good" to "Bad"), the "continuous depth" theory would be falsified in favor of a discrete switching model.

**Result:** The sweep from -2 to +2 produced smooth sigmoid curves. The theory of a continuous manifold stands.

---

## 3. The Logic of the Model (Explanations)

### The Governing Equation as a Universal Law

$$
\frac{dH}{d\tau} = F_\theta(H, \tau, u)
$$

Popper emphasized that good theories have high **empirical content**—they forbid many things. This equation forbids "teleportation" in the state space. It asserts that the hidden state $H$ must evolve continuously through depth $\tau$.

**Visualizing the Conjecture:**
* **The Vector Field:** We conjecture that the model learns a vector field. "Integration" is simply following the arrows.
* **The Control Signal ($u$):** This is not a discrete injection but a modification of the law of motion itself. It warps the entire field, changing the trajectory of *all* points.

---

## 4. Objective Knowledge from Subjective Tools (Solver Probing)

Popper argued that "our knowledge grows through the correction of our mistakes." The use of adaptive solvers provided an unintended instrument to probe the objective structure of the learned representation.

### The Unexpected Discovery: NFE Asymmetry

We treated the solver's Number of Function Evaluations (NFE) as a measurement instrument.

* **Observation:** The solver required significantly more steps (NFE=20) to resolve "Negative" sentiment trajectories than "Positive" ones (NFE=14).
* **Interpretation:** This suggests the "Negative" region of the vector field has higher curvature or complexity.

**Why this is Popperian:**
We did not set out to prove this. The "solver as instrument" revealed a geometric reality (an asymmetry) that our initial theory did not predict but must now account for. It generated a *new problem situation*: Why is negativity geometrically more complex than positivity in WikiText-2?

**Intersubjective Testing:**
To ensure this wasn't an artifact of the instrument (the specific solver), we cross-checked with `dopri5` and `adaptive_heun`. Their agreement (within 15%) validates the objectivity of the finding.

---

## 5. Error Correction (The Debugging Phase)

Science is a process of error elimination. The initial failure of the steering experiment (where all probabilities were identical) provided a falsification of our training protocol.

**The Refutation:** The model failed to steer when trained end-to-end without warmup.
**The Diagnosis:** We identified that the *auxiliary hypothesis* (that "simultaneous training is sufficient") was false. The model cannot steer a concept it hasn't yet learned to recognize.

**The Correction:** We adopted a phased training approach (Warmup -> Control Embed -> Hybrid Unfreeze). This was not an *ad hoc* save to rescue the theory, but a procedural correction that led to reproducible, high-accuracy results (94.5% success).

---

## 6. Future Conjectures (Next Steps)

Our theory is not "true" in an absolute sense; it is merely the best current explanation that has survived these specific tests. It remains open to falsification by:

1.  **Scaling:** Will the efficiency gains hold at 124M parameters?
2.  **Complex Controls:** Can it handle multi-dimensional controls (e.g., Formality + Sentiment) without chaotic interference?

We invite future work to attempt to refute these extensions.
