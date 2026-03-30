# Docs Hub

This folder is the operating system for the company and product.

## Read In This Order

1. [Vision](vision.md)
2. [Product Requirements](prd.md)
3. [Research Evidence Artifact](../research/research-artifact-2026-03-23.md)
4. [Research Plan](research-plan.md)
5. [Team Operating Model](team.md)
6. [System Architecture](architecture.md)
7. [Security and Trust](security-trust.md)
8. [Workflow](workflow.md)
9. [Roadmap](roadmap.md)

## What These Docs Are For

- align co-founders on the problem and product promise
- keep a current evidence base of what has worked, failed, or remains unproven in adjacent products
- let research, design, engineering, trust, and DevOps work from the same source of truth
- create a repeatable path from idea to validated release
- document how we prevent fake accounts, fake signals, and unsafe behavior

## How We Use Them

- update `vision.md` only when the mission or target user changes
- update `research-artifact-2026-03-23.md` or create a newer dated artifact when we gather fresh market or academic evidence
- update `prd.md` when feature scope or requirements change
- update `research-plan.md` before each discovery cycle
- update `team.md` when responsibilities or hiring plans change
- update `architecture.md` and `security-trust.md` before major implementation shifts
- update `workflow.md` and `roadmap.md` every planning cycle

## Automation

- [Audit Log](audit-log.md) is generated automatically from the tracked docs manifest in `doc-governance.yaml`
- [Review Queue](review-queue.md) opens and closes automatically when you activate review triggers
- run `python scripts/docs_governance.py sync` to refresh the audit log, review queue, and state file
- run `python scripts/docs_governance.py trigger planning_cycle --note "weekly planning"` to mark planning-cycle docs as due
- run `python scripts/docs_governance.py watch` to keep the audit files current while you edit docs locally

## Public Boundary

This `docs/` folder is the public company and product operating layer.

## Operating Rule

If a decision affects user trust, identity, location, or personal safety, it must be reflected in both `architecture.md` and `security-trust.md` before release.
