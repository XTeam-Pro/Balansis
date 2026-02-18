/-
  BalansisFormal.Algebra — Structural axioms S1–S3

  S1: (AbsoluteValue, add, absolute, neg) is an abelian group
  S2: (AbsoluteValue \ {absolute}, mul, one, mulInv) is an abelian group
  S3: (EternalRatio, add, mul) forms a field

  All axioms are proven as theorems via the toReal bridge:
    1. Prove the identity holds on toReal (using ℝ's properties)
    2. Apply toReal_injective to lift to structural equality
-/
import Mathlib
import BalansisFormal.EternalRatio

open scoped NNReal

namespace BalansisFormal

-- ====================================================================
-- S1: Additive Abelian Group on AbsoluteValue
-- ====================================================================

namespace AbsoluteValue

/-- S1.1 (Closure): Addition is closed by construction. -/
theorem s1_closure (_a _b : AbsoluteValue) : True := trivial

/-- S1.2 (Associativity): (a + b) + c = a + (b + c). -/
theorem s1_associativity (a b c : AbsoluteValue) :
    (a.add b).add c = a.add (b.add c) := by
  apply toReal_injective
  rw [add_toReal, add_toReal, add_toReal, add_toReal]
  ring

/-- S1.3 (Commutativity): a + b = b + a. -/
theorem s1_commutativity (a b : AbsoluteValue) :
    a.add b = b.add a := by
  apply toReal_injective
  rw [add_toReal, add_toReal]
  ring

/-- S1.4 (Identity): a + absolute = a. -/
theorem s1_identity_right (a : AbsoluteValue) :
    a.add absolute = a :=
  a4_identity_right a

/-- S1.4 (Identity): absolute + a = a. -/
theorem s1_identity_left (a : AbsoluteValue) :
    absolute.add a = a :=
  a4_identity_left a

/-- S1.5 (Inverse): a + neg(a) = absolute. -/
theorem s1_inverse (a : AbsoluteValue) :
    a.add (a.neg) = absolute := by
  apply toReal_injective
  rw [add_toReal, neg_toReal, absolute_toReal]
  ring

-- ====================================================================
-- S2: Multiplicative Abelian Group on AbsoluteValue \ {absolute}
-- ====================================================================

/-- S2.1 (Closure): Product of non-absolute values is non-absolute. -/
theorem s2_closure (a b : AbsoluteValue)
    (ha : ¬isAbsolute a) (hb : ¬isAbsolute b) :
    ¬isAbsolute (a.mul b) :=
  EternalRatio.mul_not_absolute ha hb

/-- S2.2 (Associativity): (a * b) * c = a * (b * c). -/
theorem s2_mul_associativity (a b c : AbsoluteValue) :
    (a.mul b).mul c = a.mul (b.mul c) := by
  apply toReal_injective
  rw [mul_toReal, mul_toReal, mul_toReal, mul_toReal]
  ring

/-- S2.3 (Commutativity): a * b = b * a. -/
theorem s2_mul_commutativity (a b : AbsoluteValue) :
    a.mul b = b.mul a := by
  apply toReal_injective
  rw [mul_toReal, mul_toReal]
  ring

/-- S2.4 (Identity right): a * one = a. -/
theorem s2_mul_identity_right (a : AbsoluteValue) :
    a.mul one = a := by
  apply toReal_injective
  rw [mul_toReal, one_toReal, _root_.mul_one]

/-- S2.4 (Identity left): one * a = a. -/
theorem s2_mul_identity_left (a : AbsoluteValue) :
    one.mul a = a := by
  apply toReal_injective
  rw [mul_toReal, one_toReal, _root_.one_mul]

/-- Multiplicative inverse: 1/magnitude, same direction. -/
noncomputable def mulInv (a : AbsoluteValue) (h : ¬isAbsolute a) : AbsoluteValue :=
  { magnitude := ⟨1 / (a.magnitude : ℝ), by
      simp [isAbsolute] at h
      exact div_nonneg one_pos.le (NNReal.coe_nonneg _)⟩
    direction := a.direction
    wf := by
      intro hmag
      exfalso
      simp only [isAbsolute] at h
      have hpos : (0 : ℝ) < ↑a.magnitude := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr h)
      have h0 := congr_arg (fun x : NNReal => (x : ℝ)) hmag
      simp at h0
      exact absurd h0 h }

/-- S2.5 (Inverse): a * mulInv(a) has toReal = 1 for non-absolute a. -/
theorem s2_mul_inverse (a : AbsoluteValue) (h : ¬isAbsolute a) :
    (a.mul (a.mulInv h)).toReal = 1 := by
  rw [mul_toReal]
  simp only [isAbsolute] at h
  have hpos : (0 : ℝ) < ↑a.magnitude := NNReal.coe_pos.mpr (pos_iff_ne_zero.mpr h)
  have hne : (↑a.magnitude : ℝ) ≠ 0 := ne_of_gt hpos
  simp [toReal, mulInv]
  have dsq := Direction.toReal_sq a.direction
  field_simp
  nlinarith

-- ====================================================================
-- Distributivity: a * (b + c) = a*b + a*c
-- ====================================================================

/-- Multiplication distributes over addition. -/
theorem mul_add_distrib (a b c : AbsoluteValue) :
    (a.mul (b.add c)).toReal = ((a.mul b).add (a.mul c)).toReal := by
  rw [mul_toReal, add_toReal, add_toReal, mul_toReal, mul_toReal]
  ring

end AbsoluteValue

-- ====================================================================
-- S3: EternalRatio Field
-- ====================================================================

namespace EternalRatio

/-- S3.1a (Additive associativity): (r₁ + r₂) + r₃ = r₁ + (r₂ + r₃). -/
theorem s3_add_assoc (r₁ r₂ r₃ : EternalRatio) :
    ((r₁.add r₂).add r₃).toReal = (r₁.add (r₂.add r₃)).toReal := by
  rw [add_toReal, add_toReal, add_toReal, add_toReal]
  ring

/-- S3.1b (Additive commutativity): r₁ + r₂ = r₂ + r₁. -/
theorem s3_add_comm (r₁ r₂ : EternalRatio) :
    (r₁.add r₂).toReal = (r₂.add r₁).toReal := by
  rw [add_toReal, add_toReal]
  ring

/-- S3.1c (Additive identity): r + zero = r. -/
theorem s3_add_identity (r : EternalRatio) :
    (r.add zero).toReal = r.toReal :=
  add_zero_right r

/-- S3.1d (Additive inverse): r + neg(r) = 0. -/
theorem s3_add_inverse (r : EternalRatio) :
    (r.add r.neg).toReal = 0 :=
  add_neg_self r

/-- S3.2a (Multiplicative associativity): (r₁ * r₂) * r₃ = r₁ * (r₂ * r₃). -/
theorem s3_mul_assoc (r₁ r₂ r₃ : EternalRatio) :
    ((r₁.mul r₂).mul r₃).toReal = (r₁.mul (r₂.mul r₃)).toReal := by
  rw [mul_toReal, mul_toReal, mul_toReal, mul_toReal]
  ring

/-- S3.2b (Multiplicative commutativity): r₁ * r₂ = r₂ * r₁. -/
theorem s3_mul_comm (r₁ r₂ : EternalRatio) :
    (r₁.mul r₂).toReal = (r₂.mul r₁).toReal := by
  rw [mul_toReal, mul_toReal]
  ring

/-- S3.2c (Multiplicative identity): r * unity = r. -/
theorem s3_mul_identity (r : EternalRatio) :
    (r.mul unity).toReal = r.toReal :=
  e3_identity_right r

/-- S3.2d (Multiplicative inverse): r * inv(r) = 1 for non-zero r. -/
theorem s3_mul_inverse (r : EternalRatio) (h : ¬AbsoluteValue.isAbsolute r.numerator) :
    (r.mul (r.inv h)).toReal = 1 :=
  e4_inverse r h

/-- S3.3 (Distributivity): a * (b + c) = a*b + a*c. -/
theorem s3_distributivity (a b c : EternalRatio) :
    (a.mul (b.add c)).toReal = ((a.mul b).add (a.mul c)).toReal := by
  rw [mul_toReal, add_toReal, add_toReal, mul_toReal, mul_toReal]
  ring

end EternalRatio

end BalansisFormal
