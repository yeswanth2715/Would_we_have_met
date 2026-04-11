# System Architecture

## Architecture Goal

Build a privacy-aware, trust-first matching system that can recommend and explain high-quality local or online connections without exposing sensitive personal data too early.

## High-Level Components

### 1. Client App

- iOS, Android, or responsive web client
- onboarding, profile, mood check-in, discovery, mutual opt-in, messaging

### 2. API Gateway

- authenticated entry point
- rate limiting
- request validation
- device and session metadata capture

### 3. Identity And Trust Service

- email and phone verification
- liveness and identity verification hooks
- device fingerprinting
- trust score aggregation
- suspicious behavior flags

### 4. Profile And Intent Service

- public profile
- private preferences
- intent, boundaries, and availability
- mood and social battery state

### 5. Candidate Retrieval Service

- coarse geographic filtering
- time-window filtering
- availability and intent pre-filtering
- retrieval of candidates with meaningful historical co-presence or likely future overlap

### 6. Encounter And Path Insight Service

- historical overlap detection using safe time and place buckets
- future overlap estimation from recurring routines, declared availability, and neighborhood movement patterns
- narrative generation for user-facing explanations such as same street, same building zone, or same venue area
- privacy policy checks for whether an insight should be shown at neighborhood, street, venue, or building granularity
- suppression of insights that could feel unsafe, overly specific, or stalking-adjacent

### 7. Match Intelligence Service

- compatibility scoring
- mood and context weighting
- weighting of past overlap strength and future overlap likelihood
- quality ranking
- feedback-driven learning

This is where the existing prototype can evolve. The current encounter detection and scoring code can become early feature-generation logic for a more robust ranking system.

### 8. Safety And Moderation Service

- report and block actions
- abuse queue
- manual review console
- enforcement actions

### 9. Messaging Or Connection Service

- mutual opt-in state machine
- chat unlock rules
- safety nudges and conversation controls

### 10. AI Mediation And Voice Bridge Service

- voice conversation with the assistant for users who are not ready for direct exposure
- extraction of shared intent, mood, communication cues, and compatibility patterns from consented interactions
- relay of sanitized summaries or mediated audio between matched users
- identity masking until both sides explicitly opt in to reveal more
- abuse logging, moderation hooks, and policy enforcement for mediated sessions

### 11. Analytics And Feature Store

- event tracking
- offline model features
- experimentation metrics
- audit-friendly change history
- insight-generation and insight-reveal events

## Recommended Data Boundaries

- separate identity data from public profile data
- separate exact location telemetry from coarse discovery location
- separate raw path history from derived overlap insight summaries
- separate raw voice artifacts from derived compatibility signals
- store moderation artifacts in restricted access systems
- treat trust scores as internal only

## Recommended Initial Stack

- Next.js for the responsive web client once the alpha UX is ready to harden
- Tailwind for fast client-side UI iteration and design system consistency
- FastAPI for backend APIs
- PostgreSQL for transactional data
- Redis for caching and short-lived session or rate-limit data
- object storage for media and verification artifacts
- background workers for scoring, moderation queues, and verification callbacks
- analytics warehouse for experiments and model evaluation
- MCP-compatible internal tooling for agent-assisted research, moderation, and operations workflows when that layer is introduced

## Environment Strategy

- local development
- staging with anonymized or synthetic data only
- production with strict access control and audit logs

## Observability

- structured logging
- request tracing
- security event logging
- ranking latency monitoring
- moderation queue health
- insight-reveal audit logging

## Mapping From Current Repo

Current modules that remain useful:

- `meetings.py`: event or encounter detection ideas
- `features/*.py`: early feature engineering references
- `train_models.py`: experimentation path for ranking research
- `main.py`: prototype pipeline orchestration

These should eventually move behind service boundaries instead of remaining as a single script pipeline.

## Current Alpha Implementation

The current shipped alpha uses:

- FastAPI for the API and routing layer
- SQLite for local persistence of check-ins, sessions, and mood history
- a simple browser UI served from the API app

This is intentionally lighter than the target production stack so the team can validate the workflow before moving the client to Next.js and the database layer fully to PostgreSQL.
