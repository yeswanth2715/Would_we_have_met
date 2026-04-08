# Would We Have Met

Would We Have Met is evolving from a meeting-detection prototype into a trust-first discovery platform for socially selective people who prefer intentional connection over random social exposure.

## Product Direction

The new product direction is:

- help people discover high-potential connections based on temporal, geographic, behavioral, and mood/context signals
- reduce forced social randomness for users who do not want to spend energy on noisy events or low-signal interactions
- protect users with strong anti-fake-account controls, privacy-aware location handling, and safety-first design

In simple terms, we are building a system that helps someone say:

"I am in this kind of mood, in this kind of neighborhood, open to this kind of connection. Show me the safest and most meaningful nearby or online possibilities."

## What Already Exists In This Repo

The current codebase already gives us a useful intelligence foundation:

- encounter detection based on place and time
- feature engineering around novelty, unexpectedness, and usefulness
- a serendipity scoring pipeline
- model training scripts for experimentation

These pieces can become the first layer of our future match-intelligence service.

## Alpha App Surface

The repo now also includes a working alpha app surface built on FastAPI:

- SQLite-backed persistence for mood check-ins and mediator sessions
- a browser UI served from `/` for mood check-in, candidate discovery, and mediated intros
- a mediator v0 that opens the first exchange and suggests a handoff once both people have shared enough context

Run it locally with:

```bash
uvicorn api.main:app --reload
```

Then open `http://127.0.0.1:8000/`.

## Founder Docs

The repo now includes a documentation backbone for turning this prototype into a real product company:

- [Docs Hub](docs/README.md)
- [Research Workspace](research/README.md)
- [Vision](docs/vision.md)
- [Product Requirements](docs/prd.md)
- [Research Evidence Artifact](research/research-artifact-2026-03-23.md)
- [Research Plan](docs/research-plan.md)
- [Team Operating Model](docs/team.md)
- [System Architecture](docs/architecture.md)
- [Security and Trust](docs/security-trust.md)
- [Workflow From Research To Deployment](docs/workflow.md)
- [Roadmap](docs/roadmap.md)
- [Docs Audit Log](docs/audit-log.md)
- [Docs Review Queue](docs/review-queue.md)

## Public Boundary

This repository is the shareable product, process, and workflow workspace.

## Working Principles

- intentional over random
- safety over growth hacks
- privacy by design
- mutual consent before deeper reveal
- real people only
- small, high-quality launches before scale

## Suggested Next Build Path

1. validate the problem with user research
2. define the MVP trust and identity model
3. stand up a real API and data model around profiles, intent, and candidate matching
4. build the first mobile or web client for onboarding, mood capture, and safe discovery
5. harden security, moderation, observability, and deployment

## Current Status

This repository is now positioned as an early-stage product and research workspace. The core prototype remains available, while the new docs define the team, process, and product direction needed to take it from concept to launch.
