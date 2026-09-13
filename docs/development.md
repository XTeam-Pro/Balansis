# Development and verification

```bash
python -m pip install -e . pytest pytest-cov mypy build twine
python -m pip install ./native
python -m pytest tests/
python -m mypy balansis --follow-imports=silent
python scripts/validate_version.py v1.2.0
python scripts/check_changelog.py
python scripts/validate_api_docs.py
python scripts/validate_examples.py
python .github/scripts/check_markdown_links.py
python -m build
python -m twine check dist/*
python scripts/validate_distributions.py
```

TNSIM tests run with `python -m pytest tnsim/tests`. Set
`TNSIM_TEST_DATABASE_URL` to an isolated PostgreSQL database for API integration
checks. The tests use synthetic series and the schema in `tnsim/database/init.sql`.

Native tests under `native/tests` can be compiled with AddressSanitizer and
UndefinedBehaviorSanitizer. Keep strict floating-point compiler flags. Use the
pure Python backend and native backend against the same rational oracle.

Regenerate API signatures with `python scripts/validate_api_docs.py --write`.
Examples are executable Python scripts and notebooks. Keep notebook outputs
empty in source control; validation executes their code cells.

[Repository workflow](GITHUB_REPOSITORY_STANDARD.md).
