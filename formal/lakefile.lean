import Lake
open Lake DSL

package balansis_formal where
  moreLeanArgs := #[]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.28.0"

@[default_target]
lean_lib BalansisFormal
