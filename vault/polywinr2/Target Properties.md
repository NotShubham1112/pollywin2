# Target Properties

Part of: [[AISEHack 2.0 - Round 2 - MOC]] · Context: [[Competition Overview]]

---

The training data spans **7 distinct polymer properties**. Each training sample is a ([[SMILES]], `target`, `target_type`) triple — `target_type` says *which* of these properties the value belongs to.

## 1. Chain Bandgap ([[Egc]])
Electronic bandgap of an **isolated polymer chain**. Related: [[Egb]] (bulk counterpart — typically smaller due to intermolecular interactions).

## 2. Bulk Bandgap ([[Egb]])
Electronic bandgap of the polymer in the **bulk phase**. Related: [[Egc]].

## 3. Ionisation Energy ([[Ei]])
Energy required to **remove an electron** from the polymer. Related concept: [[Eea]] (its energetic opposite).

## 4. Electron Affinity ([[Eea]])
Energy released when the polymer **accepts an electron**. Related concept: [[Ei]].

## 5. Dielectric Constant ([[EPS]])
Ability of the polymer to **store electrical energy** in an electric field.

## 6. Refractive Index ([[Nc]])
**Optical property** describing the interaction of light with the polymer. Often correlated with [[EPS]] (Maxwell relation: n² ≈ ε for non-magnetic materials at optical frequencies).

## 7. Glass Transition Temperature ([[Tg]])
Temperature at which the polymer transitions from a **glassy to a rubbery state**. Most distinct from the others — thermal/mechanical rather than electronic/optical.

---

## Modeling Implication

Because `target_type` is given at test time, options include:
- One multi-task model with property-specific heads
- Seven separate single-task models
- One model with `target_type` as an input feature

Correlations to exploit: [[Egc]]↔[[Egb]], [[Ei]]↔[[Eea]], [[EPS]]↔[[Nc]].

See [[Dataset Files]] for data format and [[Baseline Model]] for the reference approach.

#properties #targets #machine-learning
