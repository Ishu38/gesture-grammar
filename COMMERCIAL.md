# Commercial Licensing — MLAF

## When you need a commercial licence

The Business Source License 1.1 (see `LICENSE`) lets you use MLAF **for
free** if you are:

- an individual learner, family, or caregiver
- an accredited school or college using MLAF in classroom instruction with
  no separate fee charged to the student
- a registered non-profit serving persons with disabilities
- a university or hospital using MLAF for non-commercial research

If you fall **outside** all of these — for example you are a private clinic
that bills patients, a startup building a paid product on top of MLAF, a
hardware vendor bundling MLAF with a tablet you sell, a hospital with paid
services, or an EdTech company integrating MLAF into a paid platform — you
need a commercial licence.

## What a commercial licence buys you

| Item                                                  | Free tier (BSL) | Commercial tier |
|-------------------------------------------------------|:---------------:|:---------------:|
| Source code access                                    | ✓               | ✓               |
| Self-host the web app                                 | ✓               | ✓               |
| Modify the code                                       | ✓               | ✓               |
| Patent grant under app. 202631020540                  | Permitted-purposes only | Unrestricted within agreed scope |
| Use in a fee-charging service or product              |   —             | ✓               |
| Multi-user cohort dashboards & analytics              |   —             | ✓               |
| Branded child / student reports (PDF export)          |   —             | ✓               |
| Priority email support                                |   —             | ✓               |
| Custom motor-profile calibration for your population  |   —             | optional add-on |
| Integration help (Teachmint / Fedena / hospital EMR)  |   —             | optional add-on |
| OEM redistribution rights (hardware bundles, resale)  |   —             | optional add-on |
| Indemnity against patent claims                       |   —             | ✓               |

## Indicative pricing (India, FY 2026-27)

These are starting points — final price depends on user count, support
level, and customisation. Quoted in INR; international pricing on request.

| Buyer profile                              | Typical structure                | Indicative range            |
|--------------------------------------------|----------------------------------|-----------------------------|
| Single private clinic / SLP practice       | Annual per-clinician seat        | ₹6,000 – ₹18,000 / seat / yr |
| Single special-needs school (≤ 200 students) | Annual institutional licence     | ₹40,000 – ₹1,20,000 / yr     |
| Multi-school chain or NGO network          | Tiered by site count             | ₹3 L – ₹15 L / yr            |
| Hospital department                        | Annual departmental licence       | ₹1.5 L – ₹6 L / yr           |
| EdTech / SaaS integration partner          | Annual platform licence + per-active-user fee | ₹6 L – ₹40 L / yr + ₹50–₹200 PAU |
| Hardware OEM bundle                        | Royalty per device sold          | ₹400 – ₹1,500 / device       |
| Source-code modification & redistribution  | Annual OEM licence               | ₹10 L – ₹35 L / yr           |

**What is *not* included by default** — and is a separately scoped paid
engagement: bespoke gesture-recognition model training on your population,
on-site SLP / teacher training, integrations with proprietary EMR systems,
or any work requiring NDA-protected access to your data.

## What ships in the free tier today (honest scope, May 2026)

- The web app at `multi-modal-gesture-grammar.vercel.app` (browser, camera-based)
- The 10 gestures defined in `src/data/GestureLexicon.json`
- The Earley parser in `src/grammar/GestureGrammar.js`
- 8 accessibility profiles (Motor Impaired, CP sub-types, ASD variants, Eye-Gaze AAC, Standard)
- Browser TTS output
- LocalStorage progress tracking (single-user)
- PWA install for offline-capable shell

**Not in the free tier today:** multi-user dashboards, server-side
persistence, branded PDF reports, cohort analytics, EMR integrations.
These are the artefacts a commercial licence pays for the engineering of —
they are deliverables, not flag flips.

## Process

1. Email **roychinu45@gmail.com** with:
   - Organisation name, country, registration number.
   - Use case in 3-5 sentences.
   - Estimated number of end-users and seats.
   - Required support level (best-effort vs. SLA).
2. Indicative quote returned within 5 working days.
3. If terms work, a short Master Services Agreement (English law of contracts
   under Indian Contract Act 1872) is signed; payment is upfront for the
   first year of any subscription.
4. You receive a signed commercial-licence letter referencing this
   repository's licence grant, with your name as a permitted commercial
   user, plus any contracted deliverables.

## Refund and termination

- Subscriptions are non-refundable after delivery of the licence letter and
  any contracted dashboards / customisations.
- Either party may terminate for cause on 30 days' written notice.
- On termination, your patent licence under `PATENTS.md` reverts to the
  free-tier Permitted-Purposes scope.

---

*Questions about which tier you fall into? Just email — easier to ask than
to guess wrong.*
