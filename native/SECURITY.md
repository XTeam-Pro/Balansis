# Security Policy

## Supported Versions

Balansis is on the stable `1.x` line and follows [Semantic Versioning](https://semver.org/).
Security fixes are prioritized for the newest maintained release.

| Version line | Security support |
|--------------|------------------|
| `1.1.x` (current) | Supported |
| `1.0.x` | Supported (best effort) |
| Older than `1.0.0` | Not supported |
| Unreleased development branches | Best effort only |

`tnsim` is maintained inside the same repository and follows the same policy
unless a specific release note states otherwise.

## Reporting a Vulnerability

Please do **not** open a public GitHub issue for a suspected security problem.

Report vulnerabilities privately to:

- Email: `andrew@xteam.pro`

When possible, include:

- affected package/module and version
- impact summary
- reproduction steps or proof of concept
- conditions required for exploitation
- suggested mitigation, if known

## Response Process

Best-effort response targets:

- initial acknowledgement: within 5 business days
- triage decision: within 10 business days
- coordinated fix or mitigation timeline: as soon as reasonably possible based
  on severity and available maintainer capacity

These targets are operational goals, not contractual commitments, unless a
separate commercial support agreement states otherwise.

## Disclosure Expectations

Please allow reasonable time for triage and remediation before public
disclosure. If a fix requires a coordinated release, maintainers may ask for an
embargo window while a patch is prepared.

## Scope

This policy applies to:

- the `balansis` Python package
- the `formal/` Lean verification project where a defect could affect trust or
  integrity claims
- the `tnsim` code shipped in this repository
- release artifacts built from this repository

This policy does not create any bug bounty program, guaranteed SLA, or
commercial support obligation by itself.
