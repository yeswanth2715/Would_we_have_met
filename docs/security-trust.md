# Security And Trust

## Security Principle

If users are trusting us with identity, mood, and location-adjacent data, then fake-account prevention and privacy protection are part of the product, not a later add-on.

## Threat Model Focus

- fake account creation
- catfishing and identity misrepresentation
- scripted signup farms
- stalking or location abuse
- location inference through past or future path insights
- spam and unsolicited contact
- model manipulation through fake behavior or synthetic feedback
- internal overexposure of sensitive personal data

## Multi-Layer Fake Account Defense

### Signup Layer

- email verification
- phone verification
- CAPTCHA or equivalent bot resistance
- IP and device reputation checks
- signup rate limiting by device, network, and fingerprint cluster

### Identity Layer

- optional or mandatory selfie plus liveness verification depending launch market
- name consistency checks where legally appropriate
- duplicate face, phone, and device detection
- manual review path for ambiguous cases

### Profile Layer

- completeness checks
- content moderation on text and images
- suspicious claim detection
- profile age and activity consistency scoring

### Behavior Layer

- anomaly detection on swipe, like, message, and session patterns
- trust penalties for scripted or repetitive behavior
- automatic isolation of suspicious accounts before they reach real users

### Connection Layer

- mutual opt-in before direct contact
- graduated permissions for newer or lower-trust accounts
- block, report, and pause tools in every conversation state
- anonymous AI-mediated voice exchange before identity reveal for users who prefer a safer first step
- path-based insight reveals only after consent, trust, and policy checks

## Privacy Controls

- never expose exact live location by default
- use coarse location zones or neighborhood buckets for discovery
- store exact coordinates separately with short retention where possible
- treat historical and predicted path insights as sensitive derived location data
- keep path insights delayed, revocable, and coarse until mutual interest and explicit consent permit more context
- street-, road-, venue-, or building-level labels appear only when both sides have allowed that level and the system determines the reveal is safe
- future overlap hints must be probabilistic and non-live; never present exact real-time route prediction to another user
- let users disable path-based insights and remove retained path-history signals where policy allows
- keep names, faces, and direct identifiers hidden during mediated voice mode unless both users explicitly opt in
- encrypt data in transit and at rest
- limit internal access with role-based controls
- keep audit logs for access to moderation and identity systems

## Security Engineering Requirements

- secrets managed outside source control
- environment isolation between dev, staging, and prod
- dependency scanning and patch cadence
- authentication token rotation and revocation support
- rate limiting on all sensitive endpoints
- immutable audit trail for moderation actions
- immutable audit trail for path-insight generation, reveal decisions, and manual overrides
- clear retention, consent, and review rules for voice artifacts and AI-generated summaries

## Trust Scoring Inputs

- verification completion
- device reputation
- behavior consistency
- report history
- content quality signals
- account age
- successful mutual interactions without abuse reports

Trust score should affect:

- onboarding progression
- candidate visibility
- messaging permissions
- need for manual review

## Safety Operations

- documented moderation policy
- severity-based incident queue
- emergency user protection workflow
- escalation path for threats, stalking, or impersonation
- recurring abuse review meeting
- review queue for path-based insight complaints, false positives, and creepy-feeling suggestions
- special review policy for mediated voice misuse, coercion, impersonation, or prompt-based leakage of identity

## Compliance Track

Before public launch, confirm legal and privacy requirements for the target geography, especially around:

- identity verification
- biometric processing if used
- location data retention
- predictive mobility inference and derived location profiling
- user data deletion
- age gating and consent
