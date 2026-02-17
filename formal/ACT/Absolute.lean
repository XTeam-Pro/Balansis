/-
  ACT.Absolute — Formal axiomatization of AbsoluteValue (A1–A5)

  AbsoluteValue is a pair (magnitude, direction) where magnitude ∈ ℝ≥0
  and direction ∈ {-1, +1}. The distinguished element "Absolute" has
  magnitude 0 and serves as the additive identity.

  Authors: Balansis Team
  Version: 1.0.0
-/

namespace ACT

/-- Direction type: exactly +1 or -1. -/
inductive Direction where
  | pos : Direction   -- +1
  | neg : Direction   -- -1
deriving DecidableEq, Repr

namespace Direction

/-- Numeric value of a direction. -/
def toInt : Direction → Int
  | pos => 1
  | neg => -1

/-- Negation of direction. -/
def negate : Direction → Direction
  | pos => neg
  | neg => pos

/-- Multiplication of directions. -/
def mul : Direction → Direction → Direction
  | pos, d => d
  | neg, pos => neg
  | neg, neg => pos

theorem negate_involutive (d : Direction) : d.negate.negate = d := by
  cases d <;> rfl

theorem mul_comm (d₁ d₂ : Direction) : d₁.mul d₂ = d₂.mul d₁ := by
  cases d₁ <;> cases d₂ <;> rfl

theorem mul_assoc (d₁ d₂ d₃ : Direction) :
    (d₁.mul d₂).mul d₃ = d₁.mul (d₂.mul d₃) := by
  cases d₁ <;> cases d₂ <;> cases d₃ <;> rfl

end Direction

/-- AbsoluteValue: a non-negative magnitude paired with a direction.
    This is the fundamental type of ACT, replacing traditional signed reals
    with a structure that separates magnitude from sign information. -/
structure AbsoluteValue where
  magnitude : Float
  direction : Direction
  mag_nonneg : magnitude ≥ 0 := by native_decide
deriving Repr

namespace AbsoluteValue

/-- The distinguished Absolute element (additive identity). -/
def absolute : AbsoluteValue :=
  { magnitude := 0.0, direction := .pos, mag_nonneg := by native_decide }

/-- Check if a value is Absolute (magnitude = 0). -/
def isAbsolute (a : AbsoluteValue) : Prop := a.magnitude = 0.0

/-- Construct from a non-negative magnitude and direction. -/
def mk' (m : Float) (d : Direction) (h : m ≥ 0 := by native_decide) : AbsoluteValue :=
  { magnitude := m, direction := d, mag_nonneg := h }

/-- Convert to signed float value. -/
def toFloat (a : AbsoluteValue) : Float :=
  a.magnitude * a.direction.toInt.toFloat

end AbsoluteValue

/-!
## Axiom A1: Existence and Uniqueness

For each real number x, there exists a unique AbsoluteValue a such that
a = (|x|, sign(x)).
-/

/-- A1 (Existence): Every real number has a unique AbsoluteValue representation.
    Given x ∈ ℝ, we construct a = (|x|, sign(x)). -/
axiom a1_existence :
  ∀ (x : Float), ∃! (a : AbsoluteValue),
    a.magnitude = Float.abs x ∧
    (x ≥ 0 → a.direction = .pos) ∧
    (x < 0 → a.direction = .neg)

/-- A1 constructive witness: from_float maps each real to its AbsoluteValue. -/
def AbsoluteValue.fromFloat (x : Float) : AbsoluteValue :=
  if x ≥ 0 then
    { magnitude := x, direction := .pos, mag_nonneg := by native_decide }
  else
    { magnitude := -x, direction := .neg, mag_nonneg := by native_decide }

/-!
## Axiom A2: Non-negativity

For any AbsoluteValue a, magnitude(a) ≥ 0.
-/

/-- A2 (Non-negativity): The magnitude of any AbsoluteValue is non-negative.
    This is enforced structurally by the `mag_nonneg` field. -/
theorem a2_nonneg (a : AbsoluteValue) : a.magnitude ≥ 0 :=
  a.mag_nonneg

/-!
## Axiom A3: Compensation

If two AbsoluteValues have equal magnitude and opposite directions,
their sum is Absolute (the zero element).
-/

/-- Addition of AbsoluteValues following compensation rules. -/
def AbsoluteValue.add (a b : AbsoluteValue) : AbsoluteValue :=
  if a.direction = b.direction then
    -- Same direction: magnitudes add
    { magnitude := a.magnitude + b.magnitude
      direction := a.direction
      mag_nonneg := by native_decide }
  else if a.magnitude > b.magnitude then
    { magnitude := a.magnitude - b.magnitude
      direction := a.direction
      mag_nonneg := by native_decide }
  else if b.magnitude > a.magnitude then
    { magnitude := b.magnitude - a.magnitude
      direction := b.direction
      mag_nonneg := by native_decide }
  else
    -- Equal magnitudes, opposite directions → Absolute
    AbsoluteValue.absolute

/-- A3 (Compensation): Equal magnitudes with opposite directions cancel to Absolute.
    If mag(a) = mag(b) and dir(a) = -dir(b), then a + b = Absolute. -/
axiom a3_compensation :
  ∀ (a b : AbsoluteValue),
    a.magnitude = b.magnitude →
    a.direction = b.direction.negate →
    (a.add b).isAbsolute

/-!
## Axiom A4: Additive Identity

Absolute is the additive identity: for any a, a + Absolute = a.
-/

/-- A4 (Additive Identity): Adding Absolute does not change a value.
    For any a ∈ Abs, a + Absolute = a. -/
axiom a4_additive_identity :
  ∀ (a : AbsoluteValue),
    a.add AbsoluteValue.absolute = a

/-- A4 symmetric: Absolute + a = a. -/
axiom a4_additive_identity_left :
  ∀ (a : AbsoluteValue),
    AbsoluteValue.absolute.add a = a

/-!
## Axiom A5: Direction Preservation

For a non-zero AbsoluteValue a and a positive scalar λ > 0,
the direction of λ·a equals the direction of a.
-/

/-- Scalar multiplication of an AbsoluteValue. -/
def AbsoluteValue.smul (λ : Float) (a : AbsoluteValue) : AbsoluteValue :=
  if λ ≥ 0 then
    { magnitude := λ * a.magnitude
      direction := a.direction
      mag_nonneg := by native_decide }
  else
    { magnitude := (-λ) * a.magnitude
      direction := a.direction.negate
      mag_nonneg := by native_decide }

/-- A5 (Direction Preservation): Positive scaling preserves direction.
    For a non-zero a and λ > 0, dir(λ·a) = dir(a). -/
axiom a5_direction_preservation :
  ∀ (a : AbsoluteValue) (λ : Float),
    ¬a.isAbsolute →
    λ > 0 →
    (a.smul λ).direction = a.direction

end ACT
