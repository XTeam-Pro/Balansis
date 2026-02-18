/-
  BalansisFormal.EternalRatio — Ratio type of Absolute Compensation Theory

  EternalRatio = (numerator : AbsoluteValue, denominator : AbsoluteValue)
  with den_nonzero: denominator is non-Absolute (magnitude ≠ 0).

  Replaces traditional division and eliminates division-by-zero at the type level.

  Key bridge: toReal = numerator.toReal / denominator.toReal,
  and mul_toReal / add_toReal prove correspondence with ℝ operations.

  Axioms E1–E4 are proven as theorems.
-/
import Mathlib
import BalansisFormal.AbsoluteValue

open scoped NNReal

namespace BalansisFormal

-- ====================================================================
-- Additional AbsoluteValue lemmas used by EternalRatio
-- ====================================================================

namespace AbsoluteValue

theorem one_toReal : one.toReal = 1 := by
  simp [toReal, one, Direction.toReal_pos, NNReal.coe_one]

theorem one_not_absolute : ¬isAbsolute one := by
  simp [isAbsolute, one]

theorem not_absolute_toReal_ne_zero {a : AbsoluteValue} (h : ¬isAbsolute a) :
    a.toReal ≠ 0 :=
  fun hc => h ((isAbsolute_iff_toReal_zero _).mpr hc)

end AbsoluteValue

-- ====================================================================
-- EternalRatio definition
-- ====================================================================

structure EternalRatio where
  numerator : AbsoluteValue
  denominator : AbsoluteValue
  den_nonzero : ¬AbsoluteValue.isAbsolute denominator

noncomputable section

namespace EternalRatio

-- ====================================================================
-- Core elements
-- ====================================================================

/-- Unity ratio (1/1). Multiplicative identity. -/
def unity : EternalRatio :=
  { numerator := .one
    denominator := .one
    den_nonzero := AbsoluteValue.one_not_absolute }

/-- Zero ratio (0/1). Additive identity. -/
def zero : EternalRatio :=
  { numerator := .absolute
    denominator := .one
    den_nonzero := AbsoluteValue.one_not_absolute }

-- ====================================================================
-- Conversion to ℝ
-- ====================================================================

noncomputable def toReal (r : EternalRatio) : ℝ :=
  r.numerator.toReal / r.denominator.toReal

theorem den_toReal_ne_zero (r : EternalRatio) : r.denominator.toReal ≠ 0 :=
  AbsoluteValue.not_absolute_toReal_ne_zero r.den_nonzero

-- ====================================================================
-- Helper: product of non-absolute values is non-absolute
-- ====================================================================

theorem mul_not_absolute {a b : AbsoluteValue}
    (ha : ¬AbsoluteValue.isAbsolute a) (hb : ¬AbsoluteValue.isAbsolute b) :
    ¬AbsoluteValue.isAbsolute (a.mul b) := by
  intro h
  have ha' := AbsoluteValue.not_absolute_toReal_ne_zero ha
  have hb' := AbsoluteValue.not_absolute_toReal_ne_zero hb
  rw [AbsoluteValue.isAbsolute_iff_toReal_zero, AbsoluteValue.mul_toReal] at h
  exact mul_ne_zero ha' hb' h

-- ====================================================================
-- Arithmetic operations
-- ====================================================================

/-- Multiplication: (a/b) × (c/d) = (a·c)/(b·d). -/
def mul (r₁ r₂ : EternalRatio) : EternalRatio :=
  { numerator := r₁.numerator.mul r₂.numerator
    denominator := r₁.denominator.mul r₂.denominator
    den_nonzero := mul_not_absolute r₁.den_nonzero r₂.den_nonzero }

/-- Inverse: (a/b)⁻¹ = (b/a). Requires non-zero numerator. -/
def inv (r : EternalRatio) (h : ¬AbsoluteValue.isAbsolute r.numerator) : EternalRatio :=
  { numerator := r.denominator
    denominator := r.numerator
    den_nonzero := h }

/-- Negation: -(a/b) = (-a)/b. -/
def neg (r : EternalRatio) : EternalRatio :=
  { numerator := r.numerator.neg
    denominator := r.denominator
    den_nonzero := r.den_nonzero }

/-- Addition: (a/b) + (c/d) = (a·d + c·b) / (b·d). -/
def add (r₁ r₂ : EternalRatio) : EternalRatio :=
  { numerator := (r₁.numerator.mul r₂.denominator).add (r₂.numerator.mul r₁.denominator)
    denominator := r₁.denominator.mul r₂.denominator
    den_nonzero := mul_not_absolute r₁.den_nonzero r₂.den_nonzero }

-- ====================================================================
-- Bridge lemmas: operations on EternalRatio ↔ operations on ℝ
-- ====================================================================

theorem unity_toReal : unity.toReal = 1 := by
  simp [toReal, unity, AbsoluteValue.one_toReal]

theorem zero_toReal : zero.toReal = 0 := by
  simp [toReal, zero, AbsoluteValue.absolute_toReal]

theorem mul_toReal (r₁ r₂ : EternalRatio) :
    (r₁.mul r₂).toReal = r₁.toReal * r₂.toReal := by
  simp only [toReal, mul]
  rw [AbsoluteValue.mul_toReal, AbsoluteValue.mul_toReal]
  have h₁ := den_toReal_ne_zero r₁
  have h₂ := den_toReal_ne_zero r₂
  field_simp

theorem neg_toReal (r : EternalRatio) : r.neg.toReal = -r.toReal := by
  simp [toReal, neg, AbsoluteValue.neg_toReal, neg_div]

theorem add_toReal (r₁ r₂ : EternalRatio) :
    (r₁.add r₂).toReal = r₁.toReal + r₂.toReal := by
  simp only [toReal, add]
  rw [AbsoluteValue.add_toReal, AbsoluteValue.mul_toReal, AbsoluteValue.mul_toReal,
      AbsoluteValue.mul_toReal]
  have h₁ := den_toReal_ne_zero r₁
  have h₂ := den_toReal_ne_zero r₂
  field_simp

-- ====================================================================
-- Axiom E1: Well-definedness (constructive)
-- ====================================================================

theorem e1_well_defined (a b : AbsoluteValue) (h : ¬AbsoluteValue.isAbsolute b) :
    ∃ r : EternalRatio, r.numerator = a ∧ r.denominator = b :=
  ⟨{ numerator := a, denominator := b, den_nonzero := h }, rfl, rfl⟩

-- ====================================================================
-- Axiom E2: Stability (denominator is provably nonzero in ℝ)
-- ====================================================================

theorem e2_stability (r : EternalRatio) : r.denominator.toReal ≠ 0 :=
  den_toReal_ne_zero r

-- ====================================================================
-- Axiom E3: Multiplicative Identity
-- ====================================================================

theorem e3_identity_right (r : EternalRatio) :
    (r.mul unity).toReal = r.toReal := by
  rw [mul_toReal, unity_toReal, mul_one]

theorem e3_identity_left (r : EternalRatio) :
    (unity.mul r).toReal = r.toReal := by
  rw [mul_toReal, unity_toReal, one_mul]

-- ====================================================================
-- Axiom E4: Inverse
-- ====================================================================

theorem e4_inverse (r : EternalRatio) (h : ¬AbsoluteValue.isAbsolute r.numerator) :
    (r.mul (r.inv h)).toReal = 1 := by
  rw [mul_toReal]
  simp only [toReal, inv]
  have hd := den_toReal_ne_zero r
  have hn := AbsoluteValue.not_absolute_toReal_ne_zero h
  field_simp

-- ====================================================================
-- Additional properties for Algebra.lean
-- ====================================================================

theorem add_zero_right (r : EternalRatio) :
    (r.add zero).toReal = r.toReal := by
  rw [add_toReal, zero_toReal, add_zero]

theorem add_zero_left (r : EternalRatio) :
    (zero.add r).toReal = r.toReal := by
  rw [add_toReal, zero_toReal, zero_add]

theorem add_neg_self (r : EternalRatio) :
    (r.add r.neg).toReal = 0 := by
  rw [add_toReal, neg_toReal, add_neg_cancel]

end EternalRatio

end

end BalansisFormal
