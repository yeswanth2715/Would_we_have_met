# Product Requirements

## Product Summary

Would We Have Met becomes a trust-first connection platform that recommends nearby or online connection opportunities based on:

- temporal signals
- geographic proximity or neighborhood overlap
- past co-presence and likely future path overlap
- behavioral patterns
- declared mood and social energy
- mutual preferences and intent

## Primary Use Cases

1. A user wants low-pressure friendship opportunities nearby this week/this day/this time.
2. A user is open to dating, but only in a calm environment and within a small radius.
3. A user wants online-first connection when their energy for in-person interaction is low.
4. A user wants the system to filter out low-trust profiles automatically.
5. A user wants to understand whether they may have crossed paths with someone in the past, or are likely to overlap again soon, without exposing either person's exact live route.

## Core User Inputs

- profile basics
- connection intent: friendship, dating, activity partner, conversation
- mood: quiet, curious, open, cautious, energized, reflective
- social battery: low, medium, high
- availability window
- neighborhood radius
- online-only or in-person preference
- hard boundaries and safety preferences
- consent level for path-based insight sharing

## MVP Features

### 1. Trust-Gated Onboarding

- email verification
- phone verification
- selfie plus liveness check or equivalent trust step
- device reputation scoring
- suspicious signup throttling
- profile completeness scoring before discovery access

### 2. Intent And Mood Capture

- quick daily or session check-in
- preferred interaction mode
- time horizon: now, today, this week, later
- "not open right now" state

### 3. Candidate Discovery

- surface a limited number of high-confidence candidates
- rank by compatibility, trust score, and timing fit
- support neighborhood-level discovery without revealing exact address
- support online candidates when in-person fit is low

### 4. Path And Timing Insights

- show privacy-safe insight cards that explain why a match feels meaningful
- support historical overlap narratives such as same street, same venue area, or same building zone when policy and consent allow
- support future overlap narratives such as likely to pass through the same area again based on recurring routines and declared availability
- explain timing shifts when relevant, for example that users may have been near each other before but mutual openness or intent aligns only now
- avoid exact route playback and avoid revealing precise live movement

### 5. Safe Reveal

- no direct messaging until both sides opt in
- staged reveal of identity and location details
- report, block, and pause actions in every interaction state

### 6. Feedback Loop

- ask whether a suggestion felt right
- learn from declines without punishing boundaries
- feed trust and ranking systems with explicit user feedback

## Ranking Inputs

- temporal overlap
- neighborhood overlap
- historical co-presence strength
- likely future overlap windows
- behavioral compatibility
- intent match
- mood match
- change in readiness or intent over time
- conversation style match
- trust score
- historical quality feedback

## Functional Requirements

- users can create profiles and set intent safely
- users can update mood and availability in under 30 seconds
- system can generate candidate recommendations within acceptable latency
- system can hide low-trust or low-quality candidates automatically
- users can opt for online-only discovery
- system can generate human-readable past or future overlap insights in a privacy-safe format
- users can control whether path-based insights are used for ranking or reveal
- admins can review flagged users and suspicious clusters

## Safety And Privacy Requirements

- exact live location is never shown by default
- path-based insights must use delayed and policy-safe time and place buckets until mutual interest and trust thresholds are met
- street-, venue-, or building-level labels only unlock when both users have consented and the system marks that reveal as safe
- sensitive trust signals are stored separately from public profile data
- every user-facing interaction supports block and report
- users can disable path-based insights and delete related history where policy allows
- moderation tools must include audit logs
- fake-account resistance must exist at signup, ranking, messaging, and payout or premium layers if introduced later

## Release Slices

### Alpha

- verified onboarding
- mood capture
- basic candidate ranking
- internal-only overlap scoring and narrative testing
- mediated intro v0 with the system opening the first exchange and guiding a calm handoff
- no unrestricted direct chat yet

### Beta

- safe messaging after mutual opt-in
- moderation console
- trust score refinement
- neighborhood suggestions with privacy buffers
- user-facing path and timing insight cards with strict safety gating

### Public V1

- in-person and online modes
- feedback-driven ranking loop
- past and future overlap insights that explain why a match is timely
- event or venue integration if it improves trust and quality

## Metrics

- verified signup rate
- daily mood update rate
- candidate acceptance rate
- insight-to-mutual-interest conversion
- match-to-conversation conversion
- conversation-to-meeting conversion where applicable
- fake-account capture rate before first interaction
- moderation turnaround time

## Out Of Scope For MVP

- fully open public feeds
- exact map tracking
- exact route replay or continuous path sharing
- anonymous mass messaging
- growth loops that bypass trust review
