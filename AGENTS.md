# Repository Instructions

These instructions apply to every human or automated contributor working in
the public Balansis repository. All repository, branch, and CI work must follow
[the GitHub repository standard](docs/GITHUB_REPOSITORY_STANDARD.md).

## Sources of Truth

- `README.md` and `docs/` describe the supported public surface.
- `pyproject.toml` defines the Python package and quality gates.
- `formal/lakefile.lean` and `formal/README.md` define the Lean environment.
- `LICENSING.md`, `LICENSE`, and `CONTRIBUTING.md` govern distribution and
  contributions; do not reinterpret their terms in code or documentation.

## Engineering Rules

- Work on a focused branch and use a pull request. Never push directly to
  `master` or another protected branch.
- Run relevant local checks before marking a pull request ready. Do not use
  GitHub Actions as an iterative development or debugging shell.
- Use Draft PRs for active work. Ready PRs run the single cheap `PR Gate`;
  heavy validation is manual and requires a stated risk-based reason.
- Do not add feature-push or scheduled workflows. New triggers, permissions,
  runners, secrets, and publication steps require explicit maintainer review.
- Preserve Python 3.10+ compatibility and the public API unless a breaking
  change is explicitly requested and documented.
- Separate proved Lean theorems, tested runtime behaviour, and research
  hypotheses. Do not strengthen mathematical or accuracy claims beyond the
  evidence committed in the same change.
- Add or update tests for behavioural changes. Run `python -m pytest tests/`
  for the Python package; when `formal/` changes, also run `lake build` there.
- Do not commit credentials, private datasets, generated environments,
  confidential legal material, unpublished mechanisms, or internal project
  documentation.

## Change Hygiene

Keep changes coherent, preserve unrelated work, and update public
documentation when a supported command, API, proof status, or compatibility
contract changes. Report local checks, any GitHub Actions intentionally
triggered, and whether manual Heavy Validation was run.
