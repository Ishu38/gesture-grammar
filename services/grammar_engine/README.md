---
title: MLAF Grammar Engine
sdk: docker
app_port: 8300
pinned: false
short_description: FastAPI + SWI-Prolog backend for MLAF demo
---

# MLAF Grammar Engine

Backend service for **MLAF** — Multimodal Language Acquisition Framework. An in-browser, real-time English grammar acquisition system for learners with motor impairment and deaf/hard-of-hearing users.

This Space hosts the FastAPI + SWI-Prolog grammar engine (X-bar syntax, binding theory, Indian Sign Language interference detection). The frontend lives at https://multi-modal-gesture-grammar.vercel.app.

## Stack

- FastAPI (Python 3.11) + Uvicorn
- SWI-Prolog via pyswip
- Pydantic v2

## Endpoints

- `GET  /health` — liveness probe
- `POST /predict-next` — grammatically valid next-token continuations
- `POST /validate` — sentence well-formedness check
- `POST /interference` — L1-interference (ISL → English) pattern detection

---

**Patent-Pending** — Provisional application TEMP/E-1/22951/2026-KOL, Indian Patent Office (2026). All rights reserved.

Designed and built by Neil Shankar Ray.
