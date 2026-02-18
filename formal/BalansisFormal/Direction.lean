/-
  BalansisFormal.Direction — Sign type for ACT

  Direction is a two-element type {pos, neg} representing the sign
  component of an AbsoluteValue. It forms a group under multiplication.
-/
import Mathlib

namespace BalansisFormal

inductive Direction where
  | pos : Direction
  | neg : Direction
deriving DecidableEq, Repr

namespace Direction

noncomputable def toReal : Direction → ℝ
  | pos => 1
  | neg => -1

def negate : Direction → Direction
  | pos => neg
  | neg => pos

def mul : Direction → Direction → Direction
  | pos, d => d
  | neg, pos => neg
  | neg, neg => pos

-- Basic structural properties (by exhaustive case analysis)

theorem negate_involutive (d : Direction) : d.negate.negate = d := by
  cases d <;> rfl

theorem mul_comm (d₁ d₂ : Direction) : d₁.mul d₂ = d₂.mul d₁ := by
  cases d₁ <;> cases d₂ <;> rfl

theorem mul_assoc (d₁ d₂ d₃ : Direction) :
    (d₁.mul d₂).mul d₃ = d₁.mul (d₂.mul d₃) := by
  cases d₁ <;> cases d₂ <;> cases d₃ <;> rfl

theorem mul_pos_left (d : Direction) : pos.mul d = d := by
  cases d <;> rfl

theorem mul_pos_right (d : Direction) : d.mul pos = d := by
  cases d <;> rfl

-- toReal lemmas

theorem toReal_pos : pos.toReal = 1 := rfl

theorem toReal_neg : neg.toReal = -1 := rfl

theorem toReal_ne_zero (d : Direction) : d.toReal ≠ 0 := by
  cases d <;> simp [toReal]

theorem toReal_negate (d : Direction) : d.negate.toReal = -d.toReal := by
  cases d <;> simp [negate, toReal]

theorem toReal_mul (d₁ d₂ : Direction) :
    (d₁.mul d₂).toReal = d₁.toReal * d₂.toReal := by
  cases d₁ <;> cases d₂ <;> simp [mul, toReal]

theorem toReal_sq (d : Direction) : d.toReal * d.toReal = 1 := by
  cases d <;> simp [toReal]

theorem toReal_abs (d : Direction) : |d.toReal| = 1 := by
  cases d <;> simp [toReal]

theorem toReal_injective (d₁ d₂ : Direction) (h : d₁.toReal = d₂.toReal) :
    d₁ = d₂ := by
  cases d₁ <;> cases d₂ <;> simp [toReal] at h <;> first | rfl | (exfalso; linarith)

end Direction

end BalansisFormal
