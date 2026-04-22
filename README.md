# MLAF — Multimodal Language Acquisition Framework

In-browser, real-time English grammar acquisition through hand gestures — designed for learners with motor impairment and deaf/hard-of-hearing users. Combines symbolic AI (formal grammars, type theory, knowledge graphs) with neural perception (MediaPipe vision, ONNX classification) so the system adapts to each learner's motor abilities in real time.

**Live Demo:** https://multi-modal-gesture-grammar.vercel.app

**Patent Pending** — Provisional application TEMP/E-1/22951/2026-KOL, Indian Patent Office (2026). 10 claims.

---

## What's novel

1. **Bidirectional abductive feedback** — symbolic reasoning modifies neural perception thresholds, Bayesian priors, and temporal parameters in real time
2. **In-browser 4-layer Graph RAG** — deductive (Montague), compositional, abductive (Peirce), and contextual reasoning over a knowledge graph
3. **Bayesian trimodal gesture fusion** — visual + acoustic + gaze, with visual sub-channel decomposition
4. **Compositional-generalization proof certificates** via typed lambda calculus — 100% novel-composition acceptance by construction (internal benchmark; reported baselines: DreamCoder 89%, COGS 81.6%, SCAN 14.2%)
5. **Per-constraint dichotic error vectors** for closed-loop gesture correction
6. **Gesture lifecycle DFA** with adaptive temporal velocity thresholds
7. **Indian Sign Language (ISL) interference detection and remediation** — SOV-to-SVO reordering, topic fronting, pro-drop / object-drop correction
8. **Accessibility-adaptive tolerance bands** — recognition parameters auto-adjust per learner profile (motor, visual, dyslexia, cognitive)
9. **Client-side Random Forest gesture inference** — runs entirely in-browser, zero network calls, zero data leakage
10. **Spatial grammar mapping** — camera-space hand position mapped to syntactic zones (Subject, Verb, Object)

## Architecture

```
Browser
├── React 19 + Vite UI
├── MediaPipe HandLandmarker (gesture perception)
├── ONNX Runtime Web (Random Forest gesture classifier)
├── Nearley/Earley CFG parser (compiled grammar)
└── graphology in-browser knowledge graph

   ↕ /grammar/* (HTTPS)

FastAPI Service (Python 3.11)
├── SWI-Prolog via pyswip (X-bar syntax + binding theory)
├── ISL interference detection
└── Sentence well-formedness validation
```

Frontend deploys to Vercel as a PWA. Backend deploys to Hugging Face Spaces (Docker) — see `services/grammar_engine/Dockerfile`.

## Stack

- **Frontend:** React 19, Vite 7, MediaPipe, ONNX Runtime Web, Nearley, graphology, Tailwind CSS, vite-plugin-pwa
- **Backend:** Python 3.11, FastAPI, Uvicorn, pyswip, SWI-Prolog, Pydantic v2
- **Infra:** Docker (multi-stage), Vercel (frontend), Hugging Face Spaces (backend)
- **Code:** ~12K LOC of core, 214 tests passing

## Run locally

Requires Node 20+, Python 3.11+, and [SWI-Prolog](https://www.swi-prolog.org/Download.html) installed system-wide.

```bash
# 1. Install
npm install
pip install -r services/grammar_engine/requirements.txt

# 2. Compile the gesture grammar
npm run compile:grammar

# 3. Run frontend + grammar engine concurrently
npm run dev
```

Then open http://localhost:5173 and grant camera permission.

Backend-only: `npm run dev:grammar` (port 8300).
Frontend-only: `npm run dev:frontend` (port 5173, Vite proxy forwards `/grammar/*` to 8300).

## Project structure

```
.
├── src/                          # React frontend
│   ├── components/               # UI: SandboxMode, GestureRecorder, ParseTreeVisualizer, etc.
│   ├── core/                     # Gesture classifiers, accessibility profiles, OBF export
│   ├── grammar/                  # Nearley grammar source + compiled artifacts
│   └── __tests__/                # Vitest test suite (214 passing)
├── services/grammar_engine/      # FastAPI + Prolog backend
│   ├── app/                      # FastAPI app, Pydantic schemas, Prolog engine wrapper
│   ├── prolog/                   # X-bar syntax + binding theory rules
│   └── Dockerfile                # Multi-stage build with apt-installed swi-prolog
├── public/                       # Static assets, MediaPipe model task files
├── vercel.json                   # Frontend deploy config + /grammar/* rewrite
└── vite.config.js                # PWA config, dev proxy, build chunking
```

## Status

| Component | State |
|---|---|
| Frontend (React/MediaPipe/ONNX) | Live, deployed |
| Grammar engine (FastAPI + Prolog) | Live, deployed |
| Random Forest classifier | Operational; participatory training with target-user population in progress |
| ISL interference detection | Operational baseline; expanding rule coverage |
| Compositional-generalization proof certificates | Implemented; formal external benchmark publication pending |

Built in participatory consultation with motor-impaired and deaf/hard-of-hearing learners. Empirical user-study results forthcoming.

## License

[PolyForm Noncommercial 1.0.0](LICENSE) — free to read, run locally, study, evaluate, and use for any noncommercial purpose (research, education, personal study, hobby). Commercial use, redistribution, and derivative works for commercial purposes require a separate license from the author.

For commercial licensing inquiries, contact the author directly.

## Author

**Neil Shankar Ray** — NLP & Speech AI Engineer · Applied Linguist (MA, 14 yrs) · IIT Patna AI/ML
LinkedIn: https://www.linkedin.com/in/neilsray
Email: roychinu45@gmail.com

---

*Patent Pending — Provisional application TEMP/E-1/22951/2026-KOL filed with the Indian Patent Office (2026). All rights reserved.*
