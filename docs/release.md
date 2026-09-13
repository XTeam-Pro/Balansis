# Preparing Balansis 1.2.0

The release consists of the Python library and its optional native kernels.
Package versions are declared in `pyproject.toml`, `balansis/__init__.py` and
`native/pyproject.toml`. TNSIM has separate metadata in `tnsim/setup.py`.

1. Run the checks in [Development](development.md).
2. Build the Python wheel and sdist with `python -m build`.
3. Build the native wheel and sdist with `python -m build native`.
4. Inspect package contents with `python scripts/validate_distributions.py`,
   then install the resulting wheels in a fresh environment outside the source tree.
5. Run `balansis doctor`, numerical examples and native capability checks there.
6. Record artifact hashes and generate notes with
   `python scripts/generate_release_notes.py 1.2.0`.
7. Submit the candidate for review through the repository release workflow.

The existing Release workflow validates the version, builds distributions,
tests installation, and promotes the same Python distributions through its
configured publication jobs. Native distributions are built separately.
