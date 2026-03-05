# Balansis Formal Verification (Lean 4 + Mathlib)

Machine-checked proofs of the 12 axioms of **Absolute Compensation Theory (ACT)** using Lean 4 and Mathlib.

## Toolchain

- **Lean**: `leanprover/lean4:v4.28.0`
- **Mathlib**: `v4.28.0`

## Building

```bash
cd formal
lake build
```

A successful build with zero errors certifies that all 12 axioms and supporting lemmas are fully proven — no `sorry`, no `axiom` declarations, no unfinished proofs.

## Proof Structure

| File | Content | Theorems |
|------|---------|----------|
| `BalansisFormal/Direction.lean` | Direction type `{pos, neg}` with multiplication | 12 lemmas: `mul_comm`, `mul_assoc`, `toReal_injective`, ... |
| `BalansisFormal/AbsoluteValue.lean` | Core ACT type over `NNReal x Direction` | Axioms **A1-A5** + `add_toReal`, `mul_toReal`, `toReal_injective` |
| `BalansisFormal/EternalRatio.lean` | Ratio type with non-zero denominator proof | Axioms **E1-E4** + bridge lemmas |
| `BalansisFormal/Algebra.lean` | Algebraic structure verification | Axioms **S1-S3**: additive group, multiplicative group, field |
| `BalansisFormal.lean` | Root import (entry point) | Imports all modules |

## ACT Axioms Proven

### AbsoluteValue (A1-A5)

| Axiom | Theorem name | Statement |
|-------|-------------|-----------|
| **A1** (Existence) | `a1_fromReal_toReal` | `fromReal(toReal(a)) = a` for all `a` |
| **A2** (Non-negativity) | `a2_nonneg` | `0 <= toReal(a)` when `a` is non-negative magnitude |
| **A3** (Compensation) | `a3_compensation` | `a + neg(a) = absolute` (perfect cancellation) |
| **A4** (Identity) | `a4_identity_right`, `a4_identity_left` | `a + absolute = a` |
| **A5** (Direction) | `a5_direction_preservation` | Positive scalar preserves direction |

### EternalRatio (E1-E4)

| Axiom | Theorem name | Statement |
|-------|-------------|-----------|
| **E1** (Well-definedness) | `e1_well_defined` | Denominator is structurally non-absolute |
| **E2** (Stability) | `e2_stability` | Bounded numerator/denominator implies bounded ratio |
| **E3** (Identity) | `e3_identity_right`, `e3_identity_left` | `r * unity = r` |
| **E4** (Inverse) | `e4_inverse` | `r * inv(r) = unity` for non-zero `r` |

### Algebraic Structures (S1-S3)

| Axiom | Theorem names | Statement |
|-------|--------------|-----------|
| **S1** (Additive Group) | `s1_associativity`, `s1_commutativity`, `s1_identity`, `s1_inverse` | `(AbsoluteValue, +, absolute, neg)` is an abelian group |
| **S2** (Multiplicative Group) | `s2_mul_associativity`, `s2_mul_commutativity`, `s2_mul_identity`, `s2_mul_inverse` | `(AbsoluteValue \ {absolute}, *, one, inv)` is an abelian group |
| **S3** (Field) | `s3_add_*`, `s3_mul_*`, `s3_distributivity` | `(EternalRatio, +, *)` is a field |

## Proof Strategy

The core technique is the **toReal bridge**: each structured ACT operation corresponds to a standard real-number operation via a `toReal` homomorphism. Properties are first proven on `R` (using Lean's `ring` and `nlinarith` tactics), then lifted to structural equality via `toReal_injective`. Non-negativity of magnitudes is enforced structurally via Mathlib's `NNReal` type.
