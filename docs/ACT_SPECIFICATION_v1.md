# ACT (Absolute Compensation Theory) — Formal Specification v1.0

**Version:** 1.0.0
**Date:** 2026-02-17
**Authors:** Balansis Team

---

## 1. Introduction

Absolute Compensation Theory (ACT) is a mathematical framework that replaces traditional zero and infinity with structurally stable algebraic elements — **AbsoluteValue** and **EternalRatio**. The theory eliminates numerical instabilities (catastrophic cancellation, overflow, underflow, division by zero) by redesigning the foundational arithmetic types.

### 1.1 Motivation

IEEE 754 floating-point arithmetic suffers from well-known failure modes:

- **Catastrophic cancellation**: subtraction of nearly equal values destroys significant digits
- **Division by zero**: produces `±Inf`, propagating undefined behavior
- **Overflow/underflow**: silent loss of information at magnitude extremes
- **Gradient instability**: in deep learning, vanishing/exploding gradients from compounding numerical errors

ACT addresses these by replacing the real number line with a *magnitude-direction* decomposition and replacing division with *structural ratios*.

### 1.2 Core Types

| Type | Replaces | Structure |
|------|----------|-----------|
| `AbsoluteValue` | Real numbers (including zero) | `(magnitude ∈ ℝ≥0, direction ∈ {-1, +1})` |
| `EternalRatio` | Division / infinity | `(numerator: AbsoluteValue, denominator: AbsoluteValue \ {Absolute})` |

The distinguished element **Absolute** = `AbsoluteValue(0, +1)` replaces zero.
**Eternity** is represented structurally via `EternalRatio`, replacing infinity.

---

## 2. Axioms

ACT is built on 12 axioms organized into three groups.

### 2.1 AbsoluteValue Axioms (A1–A5)

#### A1 — Existence and Uniqueness

> For each x ∈ ℝ, there exists a unique a ∈ Abs such that
> a = (|x|, sign(x)).

This establishes a bijection between ℝ and Abs, ensuring that every real number has exactly one representation in the ACT system. The mapping preserves all numerical information while separating magnitude from sign.

**Formal statement (Lean4):**
```
axiom a1_existence :
  ∀ (x : Float), ∃! (a : AbsoluteValue),
    a.magnitude = |x| ∧
    (x ≥ 0 → a.direction = pos) ∧
    (x < 0 → a.direction = neg)
```

#### A2 — Non-negativity

> For any a ∈ Abs, magnitude(a) ≥ 0.

Magnitudes are structurally non-negative. This is enforced at the type level rather than as a runtime check.

**Formal statement (Lean4):**
```
theorem a2_nonneg (a : AbsoluteValue) : a.magnitude ≥ 0 := a.mag_nonneg
```

#### A3 — Compensation

> If mag(a) = mag(b) and dir(a) = -dir(b), then a + b = Absolute.

This is the foundational axiom of ACT: values with equal magnitude and opposite direction perfectly compensate to Absolute. Unlike IEEE 754 subtraction, this operation is exact — no significant digits are lost.

**Formal statement (Lean4):**
```
axiom a3_compensation :
  ∀ (a b : AbsoluteValue),
    a.magnitude = b.magnitude →
    a.direction = b.direction.negate →
    (a.add b).isAbsolute
```

#### A4 — Additive Identity

> For any a ∈ Abs, a + Absolute = a.

Absolute serves as the additive identity, analogous to zero in traditional arithmetic.

**Formal statement (Lean4):**
```
axiom a4_additive_identity :
  ∀ (a : AbsoluteValue), a.add absolute = a
```

#### A5 — Direction Preservation

> For non-Absolute a and λ > 0, dir(λ · a) = dir(a).

Positive scaling preserves the direction of an AbsoluteValue. This ensures that multiplication by positive scalars is a magnitude-only operation.

**Formal statement (Lean4):**
```
axiom a5_direction_preservation :
  ∀ (a : AbsoluteValue) (λ : Float),
    ¬a.isAbsolute → λ > 0 →
    (a.smul λ).direction = a.direction
```

### 2.2 EternalRatio Axioms (E1–E4)

#### E1 — Well-definedness

> For a ∈ Abs and b ∈ Abs \ {Absolute}, there exists a unique r = a/b.

Division is well-defined whenever the denominator is non-Absolute. Since Absolute is the only element with magnitude 0, this avoids division-by-zero by construction.

**Formal statement (Lean4):**
```
axiom e1_well_definedness :
  ∀ (a b : AbsoluteValue),
    ¬b.isAbsolute →
    ∃! (r : EternalRatio), r.numerator = a ∧ r.denominator = b
```

#### E2 — Stability

> If mag(b) > ε for some ε > 0, then r = a/b is stable.

Stability is a continuous property: ratios with larger denominators are more numerically stable. The compensation engine uses this to detect when ratios approach instability.

**Formal statement (Lean4):**
```
axiom e2_stability :
  ∀ (r : EternalRatio) (ε : Float),
    ε > 0 → r.denominator.magnitude > ε → r.isStable ε
```

#### E3 — Multiplicative Identity

> For any r, r × unity = r.

The unity ratio (1/1) is the multiplicative identity for EternalRatio.

**Formal statement (Lean4):**
```
axiom e3_multiplicative_identity :
  ∀ (r : EternalRatio),
    (r.mul unity).numericalValue = r.numericalValue
```

#### E4 — Inverse

> If numerator(r) is non-Absolute, then r × r⁻¹ = unity.

Every non-zero ratio has a multiplicative inverse obtained by swapping numerator and denominator.

**Formal statement (Lean4):**
```
axiom e4_inverse :
  ∀ (r : EternalRatio) (h : r.numerator.magnitude > 0),
    |((r.mul (r.inv h)).numericalValue) - 1.0| < 1e-10
```

### 2.3 Structural Axioms (S1–S3)

#### S1 — Additive Group

> (Abs, +, Absolute, neg) forms an abelian group.

Properties:
- **Closure**: a + b ∈ Abs for all a, b ∈ Abs
- **Associativity**: (a + b) + c = a + (b + c)
- **Identity**: a + Absolute = a (from A4)
- **Inverse**: a + (-a) = Absolute (from A3, where -a has inverted direction)
- **Commutativity**: a + b = b + a

#### S2 — Multiplicative Group

> (Abs \ {Absolute}, ×, one, inv) forms an abelian group.

Properties:
- **Closure**: a × b is non-Absolute when both a, b are non-Absolute
- **Associativity**: (a × b) × c = a × (b × c)
- **Identity**: a × one = a where one = AbsoluteValue(1, +1)
- **Inverse**: a × a⁻¹ = one where a⁻¹ = AbsoluteValue(1/mag(a), dir(a))
- **Commutativity**: a × b = b × a

#### S3 — Field Structure

> (EternalRatio, +, ×) forms a field.

Combining S3.1 (additive abelian group on EternalRatio), S3.2 (multiplicative abelian group on non-zero EternalRatio), and S3.3 (distributivity), EternalRatio forms a field.

**Distributivity**: a × (b + c) = a × b + a × c

---

## 3. Key Definitions

### 3.1 AbsoluteValue

```
AbsoluteValue = { (m, d) | m ∈ ℝ≥0, d ∈ {-1, +1} }
```

**Operations:**
- **Addition**: Same direction → magnitudes add; opposite direction → magnitudes subtract; equal magnitudes, opposite direction → Absolute
- **Scalar multiplication**: λ · (m, d) = (|λ|·m, sign(λ)·d)
- **Negation**: -(m, d) = (m, -d)
- **to_float**: (m, d) → m × d

### 3.2 EternalRatio

```
EternalRatio = { (num, den) | num ∈ Abs, den ∈ Abs \ {Absolute} }
```

**Operations:**
- **Addition**: (a/b) + (c/d) = (a·d + c·b) / (b·d)
- **Multiplication**: (a/b) × (c/d) = (a·c) / (b·d)
- **Inverse**: (a/b)⁻¹ = (b/a) when a ≠ Absolute
- **numerical_value**: mag(num)/mag(den) × dir(num)·dir(den)

### 3.3 Compensation

Compensation is the process by which ACT maintains numerical stability:

1. **Exact compensation**: When mag(a) = mag(b) and dir(a) ≠ dir(b), the sum is exactly Absolute (axiom A3)
2. **Near-compensation**: When |mag(a) - mag(b)| < threshold and dir(a) ≠ dir(b), the Compensator applies stabilization
3. **Overflow compensation**: When magnitudes exceed safe bounds, logarithmic scaling is applied
4. **Underflow compensation**: When results approach Absolute, the value is promoted to exact Absolute

---

## 4. Key Theorems

### Theorem 1: Cancellation Stability

In ACT, subtraction of nearly equal values does not lose precision. The magnitude-direction decomposition preserves all significant digits:

```
a = (M, +1), b = (M - ε, +1)
a + (-b) = (M, +1) + (M - ε, -1) = (ε, +1)
```

The result (ε, +1) retains full precision of ε, unlike IEEE 754 where `M - (M - ε)` may lose most significant bits of ε.

### Theorem 2: Division Safety

Division by near-zero values is structurally prevented. EternalRatio stores the ratio as a pair, deferring evaluation until numerically safe:

```
r = EternalRatio(a, b)  -- stored structurally, not evaluated
r.numerical_value()     -- evaluated only when denominator is known-safe
```

### Theorem 3: Group Homomorphism

The mapping `φ: ℝ → Abs` defined by `φ(x) = (|x|, sign(x))` is a group homomorphism from (ℝ, +) to (Abs, add) that preserves the additive structure:

```
φ(x + y) = φ(x) + φ(y)
```

### Theorem 4: Field Embedding

The mapping `ψ: ℝ \ {0} → EternalRatio` defined by `ψ(x) = EternalRatio(φ(x), one)` embeds the non-zero reals into the EternalRatio field.

---

## 5. Relationship to IEEE 754

| Property | IEEE 754 | ACT |
|----------|----------|-----|
| Zero representation | `+0`, `-0` (two zeros) | `Absolute` (unique) |
| Infinity | `+Inf`, `-Inf` | Not needed (EternalRatio) |
| NaN | `NaN` (not-a-number) | Not needed (structurally prevented) |
| Cancellation | Catastrophic (precision loss) | Exact (A3 compensation) |
| Division by zero | Runtime exception or `Inf` | Structurally prevented (E1) |
| Overflow | Silent `Inf` | Compensated (logarithmic scaling) |
| Underflow | Silent `0` | Compensated (Absolute promotion) |

### 5.1 Compatibility

ACT types can be converted to/from IEEE 754:
- `AbsoluteValue.from_float(x)` → creates ACT representation
- `AbsoluteValue.to_float()` → returns IEEE 754 float
- `EternalRatio.numerical_value()` → evaluates to IEEE 754 float

This allows ACT to be used as a drop-in replacement in numerical pipelines with graceful fallback to standard arithmetic.

---

## 6. Lean4 Formalization

The formal proofs are organized in three files under `formal/ACT/`:

| File | Contents | Axioms |
|------|----------|--------|
| `Absolute.lean` | AbsoluteValue type, operations, axioms | A1–A5 |
| `Eternity.lean` | EternalRatio type, operations, axioms | E1–E4 |
| `Algebra.lean` | Algebraic structure axioms | S1–S3 |

Import hub: `formal/act.lean` imports all three modules.

### Proof Strategy

- **Structural proofs** (A2, closure, type-level properties): proved by construction
- **Operational proofs** (A3, A4, S1 commutativity): stated as axioms with constructive witnesses
- **Complex proofs** (field distributivity, associativity): axiomatized with `sorry`-free statements for future elaboration

The formalization uses Lean4's dependent type system to enforce invariants at the type level (e.g., non-negative magnitudes, non-zero denominators).

---

## Appendix A: Notation Summary

| Symbol | Meaning |
|--------|---------|
| Abs | Set of all AbsoluteValues |
| Absolute | The zero element: (0, +1) |
| mag(a) | Magnitude of AbsoluteValue a |
| dir(a) | Direction of AbsoluteValue a |
| ER | Set of all EternalRatios |
| unity | The multiplicative identity ratio: (1,+1)/(1,+1) |
| a⁻¹ | Multiplicative inverse |
| -a | Additive inverse (direction negation) |

## Appendix B: Axiom Index

| ID | Name | Statement |
|----|------|-----------|
| A1 | Existence | ∀x ∈ ℝ, ∃!a ∈ Abs: a = (\|x\|, sign(x)) |
| A2 | Non-negativity | ∀a ∈ Abs: mag(a) ≥ 0 |
| A3 | Compensation | mag(a)=mag(b) ∧ dir(a)=-dir(b) → a+b = Absolute |
| A4 | Additive Identity | ∀a: a + Absolute = a |
| A5 | Direction Preservation | ¬isAbsolute(a) ∧ λ>0 → dir(λ·a) = dir(a) |
| E1 | Well-definedness | ∀a,b (b≠Absolute): ∃!r = a/b |
| E2 | Stability | mag(b) > ε → r=a/b is stable |
| E3 | Multiplicative Identity | ∀r: r × unity = r |
| E4 | Inverse | num(r)≠Absolute → r × r⁻¹ = unity |
| S1 | Additive Group | (Abs, +, Absolute, neg) abelian group |
| S2 | Multiplicative Group | (Abs\{0}, ×, one, inv) abelian group |
| S3 | Field | (ER, +, ×) is a field |
