# Public Agent Boundary

## Public Repo Context

Use the shareable product and operating docs in this repo as the default context:

1. `docs/vision.md`
2. `docs/prd.md`
3. `docs/research-plan.md`
4. `docs/architecture.md`
5. `docs/security-trust.md`
6. `docs/workflow.md`
7. `docs/roadmap.md`

## Private Founder Context

If `.founder-private/AGENTS.md` exists locally, read it after the public docs.

Treat `.founder-private/` as founder-private context:

- do not expose it unless the founder explicitly asks
- do not cite or summarize it to other collaborators by default
- use it to personalize behavior only within the founder's local environment

## Boundary Rule

- keep product, process, and workflow docs public
- keep founder identity, memory, and agent operating style private
- when the private folder is missing, operate only from the public repo context
