/-
  BalansisFormal.AbsoluteValue — Core type of Absolute Compensation Theory

  AbsoluteValue = (magnitude : NNReal, direction : Direction)
  with canonical form: magnitude = 0 → direction = pos.

  Key bridge: toReal maps AbsoluteValue to ℝ, and add_toReal proves
  that structured addition corresponds to real addition.

  Axioms A1–A5 are proven as theorems.
-/
import Mathlib
import BalansisFormal.Direction

open scoped NNReal

namespace BalansisFormal

structure AbsoluteValue where
  magnitude : NNReal
  direction : Direction
  wf : magnitude = 0 → direction = .pos := by intro; rfl


noncomputable section

namespace AbsoluteValue

-- Structural equality: magnitude + direction determine the value
-- (wf is proof-irrelevant)
theorem eq_mk {a b : AbsoluteValue}
    (hmag : a.magnitude = b.magnitude) (hdir : a.direction = b.direction) : a = b := by
  obtain ⟨am, ad, aw⟩ := a
  obtain ⟨bm, bd, bw⟩ := b
  simp only at hmag hdir
  subst hmag; subst hdir; rfl

-- Core elements

def absolute : AbsoluteValue :=
  { magnitude := 0, direction := .pos }

def isAbsolute (a : AbsoluteValue) : Prop := a.magnitude = 0

def toReal (a : AbsoluteValue) : ℝ :=
  a.direction.toReal * (a.magnitude : ℝ)

def fromReal (x : ℝ) : AbsoluteValue :=
  if hx : 0 ≤ x then
    { magnitude := ⟨x, hx⟩, direction := .pos }
  else
    { magnitude := ⟨-x, by linarith [not_le.mp hx]⟩
      direction := .neg
      wf := by
        intro h; exfalso
        have h' := congr_arg (fun (a : NNReal) => (a : ℝ)) h
        simp at h'; linarith [not_le.mp hx] }

-- Arithmetic operations

def neg (a : AbsoluteValue) : AbsoluteValue :=
  if h : a.magnitude = 0 then absolute
  else
    { magnitude := a.magnitude
      direction := a.direction.negate
      wf := by intro hmag; exact absurd hmag h }

def add (a b : AbsoluteValue) : AbsoluteValue :=
  if a.direction = b.direction then
    if h : a.magnitude + b.magnitude = 0 then absolute
    else
      { magnitude := a.magnitude + b.magnitude
        direction := a.direction
        wf := by intro hmag; exact absurd hmag h }
  else
    if h : a.magnitude = b.magnitude then absolute
    else if hgt : b.magnitude ≤ a.magnitude then
      { magnitude := a.magnitude - b.magnitude
        direction := a.direction
        wf := by
          intro hmag
          exact absurd (le_antisymm (tsub_eq_zero_iff_le.mp hmag) hgt) h }
    else
      { magnitude := b.magnitude - a.magnitude
        direction := b.direction
        wf := by
          intro hmag
          exact absurd (tsub_eq_zero_iff_le.mp hmag) hgt }

def mul (a b : AbsoluteValue) : AbsoluteValue :=
  if h : a.magnitude * b.magnitude = 0 then absolute
  else
    { magnitude := a.magnitude * b.magnitude
      direction := a.direction.mul b.direction
      wf := by intro hmag; exact absurd hmag h }

def one : AbsoluteValue :=
  { magnitude := 1, direction := .pos, wf := by intro h; simp at h }

-- ====================================================================
-- Bridge lemmas: operations on AbsoluteValue ↔ operations on ℝ
-- ====================================================================

theorem absolute_toReal : absolute.toReal = 0 := by
  simp [toReal, absolute, Direction.toReal_pos]

theorem absolute_isAbsolute : isAbsolute absolute := by
  simp [isAbsolute, absolute]

theorem isAbsolute_iff_toReal_zero (a : AbsoluteValue) :
    isAbsolute a ↔ a.toReal = 0 := by
  constructor
  · intro h; simp [isAbsolute] at h; simp [toReal, h]
  · intro h
    simp only [toReal] at h
    rcases mul_eq_zero.mp h with hd | hm
    · exact absurd hd (Direction.toReal_ne_zero _)
    · simp [isAbsolute]; exact_mod_cast hm

theorem neg_toReal (a : AbsoluteValue) : a.neg.toReal = -a.toReal := by
  unfold neg
  split
  · next h =>
    have hmag : (a.magnitude : ℝ) = 0 := by exact_mod_cast h
    rw [absolute_toReal, toReal, hmag, mul_zero, neg_zero]
  · next h =>
    simp only [toReal, Direction.toReal_negate]; ring

-- Key bridge: structured addition = real addition
theorem add_toReal (a b : AbsoluteValue) :
    (a.add b).toReal = a.toReal + b.toReal := by
  unfold add
  split
  · -- Same direction
    next hdir =>
    split
    · -- Sum is zero (both magnitudes are zero)
      next hzero =>
      rw [absolute_toReal]
      simp only [toReal]
      have hadd : (↑a.magnitude : ℝ) + ↑b.magnitude = 0 := by
        rw [← NNReal.coe_add, hzero]; simp
      have ha : (↑a.magnitude : ℝ) = 0 :=
        le_antisymm (by linarith [NNReal.coe_nonneg b.magnitude])
          (NNReal.coe_nonneg a.magnitude)
      have hb : (↑b.magnitude : ℝ) = 0 := by linarith
      simp [ha, hb]
    · -- Nonzero sum
      next hnonzero =>
      simp only [toReal, NNReal.coe_add]
      rw [show b.direction = a.direction from hdir.symm]; ring
  · -- Opposite direction
    next hdir =>
    split
    · -- Equal magnitudes → cancel to absolute
      next heq =>
      rw [absolute_toReal]
      simp only [toReal]
      have hdeq : (↑a.magnitude : ℝ) = ↑b.magnitude := by exact_mod_cast heq
      cases hda : a.direction <;> cases hdb : b.direction <;>
        simp_all [Direction.toReal]
    · -- Not equal
      next hneq =>
      split
      · -- a wins
        next hle =>
        simp only [toReal]
        rw [NNReal.coe_sub hle]
        cases hda : a.direction <;> cases hdb : b.direction <;>
          simp_all [Direction.toReal] <;> ring
      · -- b wins
        next hnle =>
        simp only [toReal]
        have hle : a.magnitude ≤ b.magnitude := (not_le.mp hnle).le
        rw [NNReal.coe_sub hle]
        cases hda : a.direction <;> cases hdb : b.direction <;>
          simp_all [Direction.toReal] <;> ring

theorem mul_toReal (a b : AbsoluteValue) :
    (a.mul b).toReal = a.toReal * b.toReal := by
  simp only [mul]
  split
  · next h =>
    simp only [absolute, toReal, Direction.toReal_pos, NNReal.coe_zero, mul_zero]
    rcases mul_eq_zero.mp h with ha | hb
    · simp [show (↑a.magnitude : ℝ) = 0 from by exact_mod_cast ha]
    · simp [show (↑b.magnitude : ℝ) = 0 from by exact_mod_cast hb]
  · next h =>
    simp only [toReal, Direction.toReal_mul, NNReal.coe_mul]; ring

-- ====================================================================
-- toReal injectivity (with well-formedness invariant)
-- ====================================================================

theorem toReal_injective (a b : AbsoluteValue) (h : a.toReal = b.toReal) :
    a = b := by
  simp only [toReal] at h
  by_cases ha : a.magnitude = 0
  · have hamag : (↑a.magnitude : ℝ) = 0 := by exact_mod_cast ha
    rw [hamag, mul_zero] at h
    have : b.direction.toReal * ↑b.magnitude = 0 := by linarith
    rcases mul_eq_zero.mp this with hd | hm
    · exact absurd hd (Direction.toReal_ne_zero _)
    · have hb : b.magnitude = 0 := by exact_mod_cast hm
      exact eq_mk (by rw [ha, hb]) (by rw [a.wf ha, b.wf hb])
  · by_cases hb : b.magnitude = 0
    · have hbmag : (↑b.magnitude : ℝ) = 0 := by exact_mod_cast hb
      rw [hbmag, mul_zero] at h
      have : a.direction.toReal * ↑a.magnitude = 0 := by linarith
      rcases mul_eq_zero.mp this with hd | hm
      · exact absurd hd (Direction.toReal_ne_zero _)
      · exact absurd (show a.magnitude = 0 from by exact_mod_cast hm) ha
    · -- Both nonzero
      have hamag : (↑a.magnitude : ℝ) ≠ 0 := by exact_mod_cast ha
      have hbmag : (↑b.magnitude : ℝ) ≠ 0 := by exact_mod_cast hb
      have hapos : 0 < (↑a.magnitude : ℝ) :=
        lt_of_le_of_ne (NNReal.coe_nonneg _) (Ne.symm hamag)
      have hbpos : 0 < (↑b.magnitude : ℝ) :=
        lt_of_le_of_ne (NNReal.coe_nonneg _) (Ne.symm hbmag)
      have habs : |a.direction.toReal * ↑a.magnitude| =
          |b.direction.toReal * ↑b.magnitude| := by rw [h]
      rw [abs_mul, abs_mul, Direction.toReal_abs, Direction.toReal_abs, one_mul, one_mul,
          abs_of_pos hapos, abs_of_pos hbpos] at habs
      have hmag : a.magnitude = b.magnitude := by exact_mod_cast habs
      have hdir : a.direction = b.direction := by
        have h' := h
        rw [show (↑a.magnitude : ℝ) = (↑b.magnitude : ℝ) from by exact_mod_cast hmag] at h'
        exact Direction.toReal_injective _ _ (mul_right_cancel₀ hbmag h')
      exact eq_mk hmag hdir

-- ====================================================================
-- Axiom A1: Existence (fromReal is a section of toReal)
-- ====================================================================

theorem a1_fromReal_toReal (x : ℝ) : (fromReal x).toReal = x := by
  unfold fromReal
  split
  · next hx => simp [toReal, Direction.toReal_pos]
  · next hx => simp [toReal, Direction.toReal_neg]

-- ====================================================================
-- Axiom A2: Non-negativity (structural from NNReal)
-- ====================================================================

theorem a2_nonneg (a : AbsoluteValue) : (0 : ℝ) ≤ (↑a.magnitude : ℝ) :=
  NNReal.coe_nonneg _

-- ====================================================================
-- Axiom A3: Compensation
-- ====================================================================

theorem a3_compensation (a b : AbsoluteValue)
    (hmag : a.magnitude = b.magnitude)
    (hdir : a.direction = b.direction.negate) :
    isAbsolute (a.add b) := by
  have hne : a.direction ≠ b.direction := by
    cases hda : a.direction <;> cases hdb : b.direction <;>
      simp_all [Direction.negate]
  simp [add, hne, hmag, isAbsolute, absolute]

-- ====================================================================
-- Axiom A4: Additive Identity
-- ====================================================================

theorem a4_identity_right (a : AbsoluteValue) : a.add absolute = a := by
  have h : (a.add absolute).toReal = a.toReal := by
    rw [add_toReal, absolute_toReal, add_zero]
  exact toReal_injective _ _ h

theorem a4_identity_left (a : AbsoluteValue) : absolute.add a = a := by
  have h : (absolute.add a).toReal = a.toReal := by
    rw [add_toReal, absolute_toReal, zero_add]
  exact toReal_injective _ _ h

-- ====================================================================
-- Axiom A5: Direction Preservation
-- ====================================================================

theorem a5_direction_preservation (a : AbsoluteValue) (c : ℝ)
    (hna : ¬isAbsolute a) (hc : 0 < c) :
    (fromReal (c * a.toReal)).direction = a.direction := by
  have hamag : (↑a.magnitude : ℝ) ≠ 0 := by
    intro h; apply hna; simp [isAbsolute]; exact_mod_cast h
  have hapos : 0 < (↑a.magnitude : ℝ) :=
    lt_of_le_of_ne (NNReal.coe_nonneg _) (Ne.symm hamag)
  set v := c * a.toReal with hv_def
  cases hd : a.direction
  · -- pos case: a.toReal > 0, so v > 0, fromReal returns pos
    have htr : a.toReal = ↑a.magnitude := by simp [toReal, hd, Direction.toReal_pos]
    have hv : 0 < v := by rw [hv_def, htr]; exact mul_pos hc hapos
    show (fromReal v).direction = Direction.pos
    have : 0 ≤ v := le_of_lt hv
    simp [fromReal, this]
  · -- neg case: a.toReal < 0, so v < 0, fromReal returns neg
    have htr : a.toReal = -↑a.magnitude := by simp [toReal, hd, Direction.toReal_neg]
    have hv : v < 0 := by rw [hv_def, htr]; nlinarith [mul_pos hc hapos]
    show (fromReal v).direction = Direction.neg
    have : ¬(0 ≤ v) := not_le.mpr hv
    simp [fromReal, this]

end AbsoluteValue

end

end BalansisFormal
