# Research Artifact: Evidence, Contradictions, and Comparable Products

Date: 2026-03-23

## Purpose

This artifact is a founder-facing evidence brief for the product direction we are pursuing:

- trust-first discovery
- socially selective users
- temporal, geographic, behavioral, and mood-aware matching
- nearby or online connection
- aggressive fake-account resistance

It is a curated evidence base, not a claim that every relevant paper or company has already been captured.

## Research Question

Can a product like ours work, and if so, under what conditions does it have a real chance of succeeding instead of becoming:

- another swipe app
- another fake-account magnet
- another privacy-risky location product
- another algorithmic promise that cannot actually predict chemistry

## Executive Takeaways

- The broader category clearly works. People do use digital systems to discover relationships, friendships, and communities. Hinge, Bumble, Meetup, and Slowly all validate different parts of the problem space.
- Geography matters a lot. The best evidence in online dating research says proximity is one of the strongest drivers of interaction.
- Trust is not optional. Users already think dating platforms do a weak job removing fake accounts, romance scams remain costly, and failure cases like IRL show how destructive fake growth can be.
- Slower and lower-noise formats can work. Products such as Slowly and Meetup show there is demand for alternatives to pure swipe velocity.
- The biggest contradiction is real: algorithms are weak at predicting which two specific strangers will have chemistry before they meet.
- Therefore, our strongest positioning is not "we predict your soulmate." It is "we help you discover safer, better-timed, higher-signal opportunities for connection."

## Comparable Products Still Running

| Product | Current signal | What it proves | What it does not prove |
| --- | --- | --- | --- |
| Hinge | On May 8, 2025, Match Group said Hinge's new AI-powered recommendation algorithm drove a 15%+ increase in matches and contact exchanges. On February 3, 2026, Match Group said Hinge direct revenue grew 26% year over year in Q4 2025 and monthly active users in European expansion markets grew by nearly 50% in FY25. Hinge also uses selfie verification with liveness check and 3D face authentication. | Curated recommendations plus trust tooling can drive engagement and business growth. | Growth does not mean the system can truly predict pair-specific chemistry before two people meet. |
| Bumble | As of March 3, 2026, Bumble publicly lists photo verification, ID verification, in-app calls, report tools, and AI-based Deception Detector. On March 11, 2026, Bumble also reported that 2025 was a "quality reset" year, but total revenue fell 10% for full-year 2025 and Q4 paying users declined. | Safety and authenticity are now central product expectations, not optional extras. | Trust features alone do not guarantee product growth, retention, or monetization. |
| Bumble BFF / BFF | Bumble said on March 11, 2026 that the relaunched BFF app had not generated revenue as of December 31, 2025. | Friendship use cases are real enough to justify product investment. | Friendship discovery is not automatically a durable business model. |
| Meetup | Meetup still operates as a local community platform and explicitly centers "find your people," local groups, online events, and shared-interest discovery. | People will use interest-based and geographically relevant systems to meet others, especially when the context is clear. | Group discovery is not the same as high-trust one-to-one matching. |
| Slowly | Slowly is still active and explicitly markets slower, more meaningful communication. Letters take time to be delivered based on distance, there is no swiping, and matching is based on language and common interests. | A slower, more reflective interaction model can appeal to people who dislike fast, noisy discovery. | Online pen-pal success does not automatically translate into neighborhood or in-person success. |

## What The Research Supports

### 1. Proximity is a real signal

The strongest driver of romantic interaction in large-scale online dating data is simple geographic proximity. This supports a neighborhood-aware product direction, especially if we use coarse location rather than exact live coordinates.

Why it matters for us:

- neighborhood relevance is a feature, not a detail
- distance should be a first-pass filter
- exact location sharing should remain hidden

### 2. Fewer, higher-confidence options may be better than endless browsing

Psychological research and later review work suggest that large pools can create choice overload, commoditize people, and reduce commitment. That supports a low-volume discovery system for socially selective users.

Why it matters for us:

- we should not optimize for infinite scroll
- we should rank for quality and fit, then show fewer candidates
- the product should feel calm, not addictive

### 3. Anti-fraud detection is technically feasible

Research on online dating fraud shows that romance scam detection can be built with a combination of structured, unstructured, and learned signals, and one published system reported high holdout accuracy. This supports building a real trust stack, not relying on manual reporting alone.

Why it matters for us:

- identity and behavior signals can be modeled
- fake-account resistance should be layered at signup and after signup
- human review still matters for edge cases

### 4. Users care deeply about location control

Location-sharing research shows people are especially worried about stalking, home exposure, and lack of control over who can see their location. More recent work also shows that location data can be re-identified even when privacy protections exist.

Why it matters for us:

- never expose exact live location by default
- use neighborhood buckets or privacy-preserving zones
- separate precise telemetry from discovery-grade location data

### 5. Perception and framing affect user trust in matching

One line of research suggests that belief in the legitimacy of a matching process can affect how users experience first dates. This does not prove the algorithm is objectively better, but it does mean explainability and user trust matter.

Why it matters for us:

- users need to understand why a suggestion appears
- the system should communicate signals in human terms
- product trust is partly technical and partly experiential

This is an inference from the literature, not proof that better explanations alone create better matches.

## Contradictions And Reasons This May Not Work

### 1. Compatibility algorithms are weak at predicting spark

This is the most important contradiction in the entire space. Multiple research sources argue that pre-meeting trait and preference data do not reliably predict which two people will uniquely click.

Implication:

- we should not promise deterministic compatibility
- our model should be framed as opportunity ranking, not soulmate prediction

### 2. Popularity bias can distort recommendations

Research in platform operations suggests that dating systems often over-recommend already popular users because that is easier for the platform. But popular users are also less likely to respond, which can make outcomes worse for everyone else.

Implication:

- the ranking system must optimize for realistic mutual outcomes, not vanity exposure
- we should actively avoid rich-get-richer recommendation dynamics

### 3. Mainstream dating app design can drift into problematic use

Systematic review work and later empirical work point to risks around boredom-driven use, reward loops, impulsivity, geolocation-enabled immediacy, and coping-oriented usage.

Implication:

- no casino-style growth loops
- avoid endless consumption mechanics
- give users ways to pause, narrow, or slow the system

### 4. Users already distrust platform handling of abuse and fake accounts

Pew's 2023 U.S. research shows users are more negative than positive about companies' ability to remove bots and fake accounts, and women under 50 report especially high levels of unwanted sexual content and contact.

Implication:

- fake-account prevention is a go-to-market requirement
- moderation speed and visibility matter
- women and vulnerable groups will judge us heavily on protection quality

### 5. Fraud and regulatory exposure are real, not theoretical

FTC and SEC materials show that romance scams remain materially costly and that platforms can face serious enforcement and trust consequences when fraud, fake engagement, or deceptive growth tactics are present.

Implication:

- trust metrics must be auditable
- growth claims must be real and explainable
- we should avoid any dark pattern that simulates demand or desirability

## Failure And Warning Cases

### IRL

IRL is the clearest cautionary example in the adjacent social-discovery space. In June 2023, reporting on the shutdown stated that an internal investigation found 95% of users were fake or automated. On July 31, 2024, the SEC charged founder Abraham Shafi with a $170 million fraud case tied to misleading statements about growth and advertising practices. On August 27, 2025, the U.S. Department of Justice announced criminal fraud and obstruction charges.

Why it matters for us:

- authenticity metrics must be measurable and independently reviewable
- user counts and growth claims can become existential risk if they are inflated
- bot resistance is not only a product issue, it is a governance issue

### Match Group regulatory risk

The FTC's online-dating enforcement page shows that on August 12, 2025, Match Group agreed to pay $14 million and stop certain deceptive advertising, cancellation, and billing practices. The same FTC page also describes prior allegations involving fake love-interest advertising and fraud exposure.

Why it matters for us:

- do not fake demand
- do not rely on manipulative monetization
- user trust and consumer protection discipline have to be built into growth strategy

### Bumble friend-finding monetization risk

Bumble's BFF direction validates user interest in friendship discovery, but the company's March 11, 2026 disclosure that BFF had not generated revenue as of December 31, 2025 is a warning about business-model difficulty.

Why it matters for us:

- friendship may be strategically important
- but it should be tested as a user-value wedge, not assumed to be the default revenue engine

## Strategic Conclusion For Our Product

The evidence says this idea can work if we build it as:

- a trust-first opportunity engine
- a low-volume, high-signal recommendation system
- a neighborhood-aware but privacy-preserving product
- a mood, intent, and social-energy layer on top of discovery
- a mutual-consent system before deeper reveal or messaging

The evidence says this idea will struggle if we build it as:

- a magic chemistry predictor
- an exact live-location radar
- an engagement-maximizing swipe casino
- a growth-first product that treats trust and verification as a later phase

## Product Design Implications

- Promise better timing and safer discovery, not guaranteed compatibility.
- Ask for mood, energy, and intent because static profiles are not enough.
- Use location coarsely and privately.
- Show fewer candidates with clearer reasons.
- Gate messaging behind mutual interest and trust thresholds.
- Invest early in identity, device, behavior, and moderation systems.
- Track "felt worth my energy" and "felt safe" as first-class product metrics.

## Sources

### Running products and company disclosures

1. Match Group, "Match Group Announces First Quarter Results," May 8, 2025. https://ir.mtch.com/investor-relations/news-events/news-events/news-details/2025/Match-Group-Announces-First-Quarter-Results/
2. Match Group, "Match Group Announces Fourth Quarter and Full-Year Results," February 3, 2026. https://ir.mtch.com/investor-relations/news-events/news-events/news-details/2026/Match-Group-Announces-Fourth-Quarter-and-Full-Year-Results/
3. Bumble Support, "Our safety features," last updated March 3, 2026. https://support.bumble.com/hc/en-us/articles/28537051467293-Our-safety-features
4. Bumble Inc., "Bumble Inc. Announces Fourth Quarter and Full Year 2025 Results," March 11, 2026. https://ir.bumble.com/news/news-details/2026/Bumble-Inc--Announces-Fourth-Quarter-and-Full-Year-2025-Results/default.aspx
5. Hinge Help, "What is Selfie Verification?," last updated December 17, 2025. https://help.hinge.co/hc/en-us/articles/10303221435539-What-is-Selfie-Verification
6. Meetup, "About." https://www.meetup.com/about/
7. Slowly, "Penpal Reimagined." https://slowly.app/

### Academic and research literature

8. Eli J. Finkel, Paul W. Eastwick, Benjamin R. Karney, Harry T. Reis, and Susan Sprecher, "Online Dating: A Critical Analysis From the Perspective of Psychological Science," Psychological Science in the Public Interest, 2012. APS summary: https://www.psychologicalscience.org/publications/journals/pspi/online-dating.html
9. Liesel L. Sharabi, "Finding Love on a First Data: Matching Algorithms in Online Dating," Harvard Data Science Review, January 27, 2022. https://hdsr.mitpress.mit.edu/pub/i4eb4e8b
10. Samantha Joel, Paul W. Eastwick, and Eli J. Finkel, "Is Romantic Desire Predictable? Machine Learning Applied to Initial Romantic Attraction," Psychological Science, 2017. PubMed: https://pubmed.ncbi.nlm.nih.gov/28853645/
11. Elizabeth E. Bruch and M. E. J. Newman, "Structure of Online Dating Markets in U.S. Cities," Sociological Science, 2019. PubMed: https://pubmed.ncbi.nlm.nih.gov/31363485/
12. Musa Eren Celdir, Soo-Haeng Cho, and Elina H. Hwang, "Popularity Bias in Online Dating Platforms: Theory and Empirical Evidence," Manufacturing and Service Operations Management, 2024. https://ideas.repec.org/a/inm/ormsom/v26y2024i2p537-553.html
13. Gabriel Bonilla-Zorita, Mark D. Griffiths, and Daria J. Kuss, "Online Dating and Problematic Use: A Systematic Review," International Journal of Mental Health and Addiction, 2021. https://link.springer.com/article/10.1007/s11469-020-00318-9
14. Germano Vera Cruz et al., "Online dating: predictors of problematic Tinder use," BMC Psychology, February 29, 2024. https://link.springer.com/article/10.1186/s40359-024-01566-3
15. Guillermo Suarez-Tangil et al., "Automatically dismantling online dating fraud," IEEE Transactions on Information Forensics and Security, 2019. Open-access summary: https://open.bu.edu/handle/2144/40282
16. Janice Tsai, Patrick Gage Kelley, Lorrie Faith Cranor, and Norman Sadeh, "Location-Sharing Technologies: Privacy Risks and Controls," TPRC 2009. SSRN: https://ssrn.com/abstract=1997782
17. Yi Zhang, Yaobang Gong, Madeline Shi, and Xianfeng Yang, "Privacy risk of transportation location-based data: Re-identification and de-anonymization," Transportation Research Part C: Emerging Technologies, February 2026. https://doi.org/10.1016/j.trc.2025.105495

### Public trust, scam, and enforcement signals

18. Pew Research Center, "The experiences of U.S. online daters," February 2, 2023. https://www.pewresearch.org/internet/2023/02/02/the-experiences-of-u-s-online-daters/
19. Federal Trade Commission, "New FTC Data Reveals Top Lies Told by Romance Scammers," February 2023. https://www.ftc.gov/news-events/news/press-releases/2023/02/new-ftc-data-reveals-top-lies-told-romance-scammers
20. Federal Trade Commission, "Online Dating" tag and enforcement archive, accessed March 23, 2026. https://www.ftc.gov/online-dating
21. U.S. Securities and Exchange Commission, "SEC Charges Founder of Social Media Company 'IRL' with $170 Million Fraud," July 31, 2024. https://www.sec.gov/newsroom/press-releases/2024-92
22. U.S. Department of Justice, "Former Silicon Valley CEO Charged with Fraud and Obstruction of Justice," August 27, 2025. https://www.justice.gov/opa/pr/former-silicon-valley-ceo-charged-fraud-and-obstruction-justice
23. TechCrunch, "Unicorn social app IRL to shut down after admitting 95% of its users were fake," June 26, 2023. https://techcrunch.com/2023/06/26/irl-shut-down-fake-users/
