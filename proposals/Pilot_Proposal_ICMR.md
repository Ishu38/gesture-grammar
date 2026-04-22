# Research Pilot Proposal
## Efficacy of a Gesture-Based Multimodal Intervention for English Language Acquisition in Children with Motor Disabilities
### Submitted to: Indian Council of Medical Research (ICMR) — Extramural Research Division

**Principal Investigator:** Neil Shankar Ray, MLAF Technologies
**Co-PI:** [Collaborating clinician / institution — to be confirmed]
**Date:** March 2026
**Provisional Patent Filed**

---

## 1. Background and Rationale

Cerebral palsy (CP) affects approximately 3 per 1,000 live births in India (Gulati & Sondhi, 2018). A substantial proportion of children with CP experience dysarthria or anarthria, limiting their access to conventional language instruction that relies on verbal or written output.

Current rehabilitation approaches for language development in CP include:
- Speech-language therapy (limited availability in rural India; typical ratio of 1 SLP per 2.5 lakh population)
- Augmentative and Alternative Communication (AAC) devices (cost-prohibitive; typically Rs. 50,000+)
- Picture Exchange Communication Systems (limited to single-word requests; no grammar instruction)

**Gap:** No existing tool addresses *English grammatical structure* (syntax, morphological agreement) through a motor-accessible modality at low cost.

**MLAF** (Multimodal Language Acquisition Framework) addresses this gap by using computer-vision-based hand gesture recognition to teach English sentence construction, with real-time motor adaptation for tremor and limited range of motion.

## 2. Objectives

**Primary:** To evaluate whether 8 weeks of MLAF-based gesture intervention improves English sentence construction accuracy in children with CP compared to standard care.

**Secondary:**
- To measure changes in gesture recognition speed (automaticity) as a proxy for motor-linguistic integration
- To assess the motor calibration system's effectiveness across GMFCS Levels I-III
- To evaluate learner engagement and dropout rates
- To document ISL-to-English word order transfer patterns in deaf participants

## 3. Methodology

### 3.1 Study Design
Prospective, controlled, pre-test/post-test study with two arms.

### 3.2 Participants
- **Sample size:** 30 children (15 intervention, 15 control)
- **Age range:** 6-14 years
- **Inclusion criteria:** Diagnosis of CP (GMFCS Levels I-III), non-verbal autism, or bilateral hearing impairment; functional use of at least one hand; no prior structured English grammar instruction
- **Exclusion criteria:** GMFCS Levels IV-V (insufficient hand function for gesture input); uncorrected visual impairment
- **Recruitment:** Through collaborating rehabilitation center / special school

### 3.3 Intervention
- 3 sessions per week, 20 minutes per session, for 8 weeks (48 sessions total)
- MLAF application running on a tablet with front-facing camera
- Curriculum: 5 levels progressing from single gestures to full SVO sentences with verb agreement
- Motor calibration enabled (system adapts to each child's tremor profile and range of motion)
- Control group receives standard classroom instruction

### 3.4 Outcome Measures

| Measure | Instrument | Timepoint |
|---------|-----------|-----------|
| Sentence construction accuracy | MLAF Assessment Module (20-item SVO ordering test) | Pre, Mid (Week 4), Post (Week 8) |
| Gesture automaticity | Response onset time logged by application (ms) | Every session |
| Motor adaptation | Calibration tolerance values recorded per session | Every session |
| Engagement | Session duration, completion rate, voluntary continuation | Every session |
| Qualitative feedback | Semi-structured interview with teachers and parents | Post |

### 3.5 Data Management
- All data stored locally on device (no cloud transmission)
- De-identified participant codes
- Session logs exported as JSON for analysis
- Data retained for 5 years per ICMR guidelines

### 3.6 Statistical Analysis
- Primary outcome: paired t-test / Wilcoxon signed-rank test (pre vs. post), between-group comparison (Mann-Whitney U)
- Secondary outcomes: linear mixed-effects models for automaticity trajectories
- Significance threshold: p < 0.05, Bonferroni-corrected for multiple comparisons

### 3.7 Ethics
- Protocol to be submitted to Institutional Ethics Committee of collaborating institution
- Written informed parental consent; child assent where appropriate
- Participants free to withdraw at any time
- No personal health data transmitted off-device

## 4. Innovation

MLAF integrates several techniques not previously combined in assistive EdTech:

1. **Neural gesture recognition** — 1D CNN (41K parameters, 99.5% test accuracy) running entirely in-browser via ONNX Runtime. No cloud dependency.
2. **Formal grammar validation** — Earley parser enforcing context-free grammar rules, not pattern matching. Catches verb agreement errors, word order violations, and incomplete sentences with linguistically precise feedback.
3. **Bayesian multimodal fusion** — Combines visual (gesture), acoustic (vocalization), and gaze modalities using weighted log-likelihood fusion. Adapts modality weights based on signal quality.
4. **Motor-adaptive calibration** — Resting boundary calibrator learns each child's neutral hand position; cognitive load adapter monitors jitter and adjusts recognition thresholds dynamically.

Provisional patent filed covering the multimodal fusion architecture.

## 5. Budget

| Item | Cost (INR) |
|------|-----------|
| 8 tablets (Android, with camera) | 1,60,000 |
| Travel and field visits (8 weeks, 2 sites) | 60,000 |
| Research assistant (part-time, 4 months) | 1,20,000 |
| Data analysis software and statistician consultation | 40,000 |
| Ethics committee application fee | 15,000 |
| Printed materials (consent, assessment, teacher guides) | 15,000 |
| Report preparation and manuscript writing | 30,000 |
| Contingency (10%) | 44,000 |
| **Total** | **4,84,000** |

## 6. Timeline

| Month | Activity |
|-------|----------|
| 1 | Ethics approval, site agreements, participant recruitment |
| 2 | Pre-test assessment, device setup, teacher orientation |
| 2-4 | Intervention period (8 weeks) |
| 4 | Post-test assessment, data export |
| 5 | Data analysis, draft manuscript |
| 6 | Final report submission to ICMR, manuscript submission |

## 7. Expected Outcomes and Significance

A positive result would provide the first controlled evidence that gesture-based grammar instruction is effective for motor-impaired children — opening a pathway to scalable, low-cost language rehabilitation that requires only a camera-equipped device. Findings would be submitted for peer-reviewed publication and shared with the National Trust and state-level disability commissions for potential curriculum integration.

## 8. Principal Investigator

**Neil Shankar Ray** — Sole architect and developer of MLAF. Designed and implemented the complete system (19,000+ lines of code, 30+ modules) over 3 months, covering computer vision, formal linguistics, Bayesian machine learning, and adaptive HCI. Provisional patent holder.

A collaborating clinical co-PI from a recognized rehabilitation institution will be confirmed prior to ethics submission.

---

*This proposal requests extramural research funding. All intellectual property rights remain with the applicant.*
