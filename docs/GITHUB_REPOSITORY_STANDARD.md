# GitHub Repository Standard

**Audience:** maintainers, contributors, and coding agents

**Status:** canonical
**Source of truth:** this document

## Purpose

This public policy defines the minimum safe and cost-conscious GitHub workflow
for Balansis. It governs repository operations only; mathematical definitions,
runtime behaviour, licensing terms, and release contents remain governed by
their existing source documents.

## Branch and Pull Request Model

`master` is the default branch, pull-request target, and release-validation
boundary. Work must be performed on a focused branch and integrated by pull
request. Direct pushes, force pushes, and branch deletion should be blocked by
repository rules.

The expected lifecycle is:

```text
work branch -> local checks -> Draft PR -> Ready PR -> PR Gate -> review -> master
```

- Intermediate feature-branch pushes do not run GitHub Actions.
- Draft PRs do not consume runner compute.
- A ready, non-draft PR runs one stable `PR Gate` job.
- New commits cancel stale PR Gate runs.
- One PR should contain one coherent change.

Existing branches other than `master` have no role assigned by this policy.
Do not merge, delete, or reinterpret them without explicit maintainer approval.

## CI Architecture

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| PR Gate | Ready, non-draft PR events targeting `master` | One Ubuntu runner; whitespace, affected configuration, targeted formal audit, and Python tests only when relevant |
| Release Validation | Push to `master` | Validate the integrated state, affected package build/tests, version metadata, licensing files, and affected Lean sources |
| Heavy Validation | Manual `workflow_dispatch` only | Compatibility matrices, dependency/security audit, documentation audit, formal build, or benchmarks |
| Release | Version tag or explicit manual dispatch | Existing protected package-publication flow |

There are no scheduled heavy workflows. A schedule may be introduced only with
explicit maintainer approval and a documented safety requirement.

## Change Scope

The stable PR Gate always reports a result so branch protection is not left
waiting on docs-only changes. Inside that job:

- Markdown, policy, license, and documentation-only changes receive a local
  link and diff-integrity check without dependency installation.
- Workflow and CI-helper changes receive YAML and Python syntax checks.
- Python package, test, script, benchmark, example, dependency, and unknown
  executable changes receive the supported-runtime lint and test gate.
- Lean source or toolchain changes receive the full cached Lean build and
  source audit in the PR Gate, after merge, and by manual Heavy Validation.

The post-merge workflow applies the same scope model. Unrecognized non-doc
paths fail safe by receiving the Python validation path.

## Local Validation

Use the repository-native commands that match the affected scope:

```bash
python -m pytest tests/ --no-cov
python -m flake8 balansis/ tests/ --count --select=E9,F63,F7,F82
python .github/scripts/check_markdown_links.py
```

For Lean changes:

```bash
cd formal
lake build BalansisFormal
lake build ACT
lake env lean FormalAudit.lean
```

The repository also configures Black, isort, mypy, coverage, and pre-commit.
Run those checks when the affected area supports them, but report any
pre-existing failure separately from failures introduced by the change.

## Workflow Security and Cost Rules

- Default `GITHUB_TOKEN` permissions are read-only. Grant writes only to the
  specific release job that requires them.
- External Actions must use immutable full commit SHAs.
- Every job must set `timeout-minutes`.
- PR workflows must use stale-run cancellation; release and publication
  workflows must not cancel an in-progress mutation.
- Matrices use `fail-fast: true` and belong in manual Heavy Validation unless
  a supported release boundary requires them.
- Pull requests from forks run only on GitHub-hosted runners and receive no
  secrets or privileged credentials.
- Dependency caches must be keyed by the committed manifests and must never
  contain credentials.
- Do not use empty commits or repeated pushes to debug CI. Reproduce failures
  locally first.

## Releases

The `Release` workflow is the only package-publication workflow. Publication
retains its protected environments and does not run from feature branches or
pull requests. Build distributions once, test the same artifact, and promote
that artifact through TestPyPI, PyPI, and the GitHub Release step.

Running a publication workflow, changing a release trigger, or changing a
publication credential model requires explicit maintainer authorization.

## Heavy Validation

Heavy Validation is opt-in and manual. The initiator must choose the narrowest
scope that addresses a concrete risk: compatibility, security, documentation,
benchmarks, formal verification, or full validation. Duplicate manual runs for
the same ref and scope are serialized rather than cancelled.

GitHub-hosted runners are used because no trusted public-repository
self-hosted-runner boundary is documented. Never execute untrusted fork code on
a privileged self-hosted runner.

## Public Disclosure

Only information already suitable for the public repository may be added.
Never publish secrets, private datasets, internal infrastructure or repository
topology, confidential legal advice, unpublished patent material, or
proprietary research mechanisms. Escalate uncertain disclosure before it is
committed or published.

## Ownership and Remote Rules

`CODEOWNERS` identifies accountable reviewers but does not itself enable branch
protection. Required reviews and the stable `PR Gate` must be configured in
GitHub repository rules only after their event and no-op paths are verified.
In a single-maintainer repository, do not require a non-author approval that no
verified collaborator can provide.

## Exceptions

An exception must be narrow, public-safe, approved by the maintainer, and
recorded with its reason, scope, safety controls, and removal condition. Cost
alone never justifies removing a release, security, or correctness control.
