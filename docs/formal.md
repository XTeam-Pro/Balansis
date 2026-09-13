# Lean formalization

The Lean project uses the toolchain in `formal/lean-toolchain` and the mathlib
revision in `formal/lakefile.lean`. `BalansisFormal` defines signed-magnitude
values over nonnegative reals, canonical zero, a quotient representation for
ratios and a three-state extended ratio. `ACT` re-exports theorem statements.

`AbsoluteValue.toReal` and `fromReal` connect the signed-magnitude representation
to `ℝ`. The field, order, metric, completeness and continuity constructions use
these maps. The extended-ratio definitions give case-based arithmetic and
singular policies.

```bash
cd formal
lake build BalansisFormal ACT
lake env lean FormalAudit.lean
```

`FormalAudit.lean` checks the exported statements and typeclass instances.
The Python semantic-parity tests exercise the corresponding extended-ratio
cases with concrete runtime values. [Lean source](../formal).
