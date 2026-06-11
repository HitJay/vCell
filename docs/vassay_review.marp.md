---
marp: true
theme: default
paginate: true
size: 16:9
header: 'vAssay Review · Imaging → Seahorse Model'
footer: 'NNRCC · RIC · Qiuye (Jay) · 2026-06-11'
style: |
  section { font-size: 21px; padding: 38px 56px; }
  section.title { font-size: 28px; text-align: center; }
  section.title h1 { font-size: 40px; }
  section.section { background: #003a70; color: #fff; text-align: center; }
  section.section h1 { color: #fff; font-size: 38px; border: none; }
  section.section h3 { color: #cfe0f0; }
  h1 { font-size: 29px; color: #003a70; }
  h2 { font-size: 24px; color: #003a70; border-bottom: 2px solid #003a70; padding-bottom: 4px; }
  h3 { font-size: 20px; color: #00518a; }
  table { font-size: 15px; border-collapse: collapse; }
  th { background: #003a70; color: #fff; padding: 4px 9px; }
  td { padding: 3px 9px; border: 1px solid #ccc; }
  blockquote { font-size: 18px; color: #444; border-left: 4px solid #003a70; padding-left: 12px; background: #f4f7fa; }
  code { background: #f0f4f8; color: #003a70; padding: 1px 4px; border-radius: 3px; }
  pre { background: #f0f4f8; font-size: 14px; padding: 8px; border-radius: 4px; }
  .small table { font-size: 13px; }
  .red { color: #b00020; font-weight: bold; }
  .green { color: #1a7d3a; font-weight: bold; }
  .amber { color: #b8860b; font-weight: bold; }
---

<!-- _class: title -->

# vAssay Review: Imaging → Seahorse

### How good is the model, really?

A leakage-aware re-evaluation of the HepG2 DINOv2 → TabPFN pipeline

**Qiuye (Jay)** · NNRCC · RIC · 2026-06-11

---

<!-- _class: section -->

# 1. What vAssay is

---

## The vAssay idea

**Goal**: use cheap high-content imaging to predict expensive Seahorse respiration — a "virtual assay" for energy expenditure (EE).

```
HepG2 wells → 4-channel imaging → DINOv2 features (384-d)
            → channel aggregation (C1/C12/C14/C24)
            → TabPFN regressor → predict Seahorse (AUC / Max-Basal)
```

| Channel tag | Imaging input | Meaning |
| --- | --- | --- |
| **C1** | ch1 | brightfield (label-free morphology) |
| C12 | ch1 + ch2 | + TMRM (ΔΨm) |
| C14 | ch1 + ch4 | + MitoTracker (mito mass) |
| **C24** | ch2 + ch4 | TMRM + MitoTracker (mitochondrial) |

> `C` number = imaging channel combination (verified from data, not a model version).

---

## The legacy result — and the doubt

- Legacy notebooks reported **R² ≈ 0.77** (random 5-fold CV) → looked great.
- Training set: **264 image rows**, 9 plates, 15 treatments (compounds + siRNA).
- <span class="red">Doubt: is 0.77 real, or an artifact of how it was validated?</span>

> This review re-evaluates the same data/models with **leakage-aware cross-validation**, and asks whether the model is usable for the real deployment domain (siRNA knockdown screens).

---

<!-- _class: section -->

# 2. Why random R² is inflated
### Label leakage

---

## Root cause: shared Seahorse labels

The 264 image rows do **not** carry 264 independent answers:

| Fact | Value |
| --- | --- |
| Image rows (samples) | 264 |
| **Unique Seahorse y-values** | **88** |
| (Plate, Treatment) groups | 36 |
| Avg. rows sharing one y-value | **~3** |
| Within-group y std (median) | **0.00** |

**Seahorse is a well-level measurement**, but each well has multiple imaging fields/replicates → they all inherit the **same** y-value.

---

## How random CV "cheats"

```
One Seahorse value = 3 image rows (different fields, same answer)

Random KFold  →  some rows in TRAIN, some in TEST
              →  model sees the answer in training,
                 then is asked the SAME answer at test time
              =  open-book exam  →  inflated R²
```

Second, weaker layer — **plate effect**:
- Between-plate y std = **38.7**  vs  within-plate = **16.8**
- Random folds share the same plate → model can "recognize the plate".

> Random CV measures **interpolation within shared answers**, not generalization.

---

<!-- _class: section -->

# 3. The leakage-aware framework
### How we test it honestly

---

## Three cross-validation schemes

| Scheme | Split by | Tests | Strictness |
| --- | --- | --- | --- |
| `random` | shuffle wells | interpolation (leaky) | loosest |
| **`group_plate`** | whole plate held out | generalize to a **new plate** | medium |
| **`group_treatment` / LOTO** | whole perturbation held out | generalize to a **new target** | strictest (= real use) |

Plus two data treatments:
- **aggregate** — collapse (plate, treatment) field-replicates into 1 independent unit (264 → 36) → removes field-level label leakage.
- **siRNA-only** — drop compounds, keep knockdowns → match the deployment domain (→ 18 units).

> Metrics are computed on **pooled out-of-fold predictions** (all test folds concatenated, scored once) — the correct estimator for small / leave-one-out folds.

---

## A note on R² vs Spearman

- **R²** is sensitive to *absolute scale*. When a held-out plate sits at a different overall Seahorse level, per-fold R² can go strongly negative even if the *ranking* is fine.
- **Spearman** measures *ranking* — which is exactly what EE hit-calling needs.

> <span class="amber">Lesson:</span> for cross-plate / cross-target generalization, **judge by Spearman (ranking), not by R²**. An earlier "R² = −1.3 → generalization ≈ 0" read was a per-fold artifact; pooled-OOF Spearman tells the real story.

---

<!-- _class: section -->

# 4. Systematic benchmark
### The honest performance picture

---

## Leakage decomposition (AUC, Spearman)

Out-of-fold Spearman by setting × channel:

| Setting | C1 | C12 | C14 | C24 |
| --- | --- | --- | --- | --- |
| RAW (264, field leakage) — `group_treatment` | 0.67 | 0.69 | 0.70 | **0.75** |
| AGGREGATED (36, de-leaked) — `group_treatment` | 0.81 | 0.85 | 0.79 | **0.82** |
| siRNA-domain (18, LOTO) | −0.10 | 0.23 | −0.14 | **0.33** |

- **Aggregated, generalize-to-new-treatment Spearman ≈ 0.8** → ranking signal is **real and strong**; leakage did not create it.
- **siRNA domain collapses** to ρ ≈ 0.33 (C24 best) — the true challenge.

---

## What this means

- ✅ **Random R² 0.77 was inflated** by field-replicate label leakage — confirmed.
- ✅ **But the model is not worthless**: for ranking perturbations within the assayed domain, honest Spearman is **~0.8** (C24/C12 best).
- <span class="red">⚠ The deployment domain (pure siRNA) is the real bottleneck</span>: leave-one-target-out on 18 units gives ρ ≈ 0.33.
- **R² across plates is misleading** — scale drift, not lack of signal.

> Mean-prediction baseline under the same honest CV: Spearman **−0.4 to −0.9** → the model is clearly learning, just data-limited in the siRNA domain.

---

## Cross-domain check (legacy → drug-seq)

Legacy vAssay predictions vs **real Seahorse** on drug-seq siRNA targets (n = 12):

| Subset | n | Pearson | Spearman |
| --- | --- | --- | --- |
| all | 12 | 0.76 | 0.35 |
| excl. needs-repeat | 9 | 0.73 | **0.18** |

> Pearson 0.76 is propped up by one low outlier (PSMC3); the **ranking (Spearman) is weak across domains** → domain shift (compound-trained → siRNA-applied) is real.

---

<!-- _class: section -->

# 5. Conclusions & next steps

---

## Verdict

| Use case | Trust? |
| --- | --- |
| Rank perturbations **within the assayed domain** (Spearman) | ✅ yes (~0.8) |
| Absolute Seahorse value / legacy R² 0.77 | <span class="red">✗ no (leakage)</span> |
| Cross to **pure siRNA** domain | <span class="amber">weak (ρ 0.2–0.35)</span> |

**Best channel under honest CV**: C24 (TMRM + MitoTracker) ≈ C12, then C1/C14.

---

## Route 1: siRNA-domain retrain — status

- **De-leakage pipeline is built & tested** (aggregate → independent units, grouped/LOTO CV, pooled OOF).
- **Bottleneck is sample size, not method**: only **18 independent siRNA units** with paired Seahorse → LOTO ρ 0.33.
- **Next options**:
  1. Collect more siRNA + Seahorse pairs (more independent plates).
  2. Use drug-seq full screen (~170 targets w/ imaging+TMRM; 16 w/ real Seahorse) for semi-supervised / transfer.

---

## Deliverables (in `vCell` repo)

- `src/vcell/vassay/` — leakage-aware framework (grouped CV, aggregation, pooled OOF)
- `scripts/vassay_benchmark.py`, `vassay_systematic_benchmark.py`, `vassay_crossdomain.py`, `vassay_summary.py`
- `tests/test_vassay.py` — 7 tests (31 total passing)
- `output/2026-06-11/vassay_systematic/` — results CSV + leakage-decomposition figure
- `docs/research_plan_EE_drugseq.md` §13 — full written record

---

<!-- _class: section -->

# Thank you
### Questions?

**Key message**: the 0.77 was leakage; the *ranking* signal (~0.8) is real;
the real work now is **more siRNA-domain data**, not a new model.
