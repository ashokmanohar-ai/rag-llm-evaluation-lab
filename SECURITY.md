# Security policy

## Reporting

Please report vulnerabilities privately through GitHub Security Advisories. Do not include credentials, personal data, or exploitable production details in a public issue.

## Design controls

- Local file ingestion only; no arbitrary URL fetch.
- Markdown and text allow-list with a 2 MiB limit.
- Canonical path checks prevent traversal outside the knowledge root.
- Superseded and draft documents are excluded by default.
- Metadata filters support region and access-group scoping.
- Document content is explicitly treated as untrusted evidence.
- Citations are validated against internally assigned chunk IDs.
- Index rebuild requires an API key when the endpoint is enabled.
- Secrets are loaded from environment variables and redacted from logs.
- The container runs as a non-root user with `no-new-privileges`.

This demonstration contains synthetic Acme content only. Real deployments must add authentication, authorisation, tenant isolation, encryption, audit retention, abuse controls, and a threat model appropriate to their data.

