"""MLAF Grammar Engine — FastAPI application (port 8300).

Prolog-based X-bar syntactic analysis service. Replaces imperative JS grammar
logic with declarative Prolog queries grounded in Chomskyan generative syntax.

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8300
"""

from __future__ import annotations

import csv
import fcntl
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

from .engine import PrologEngine
from .schemas import (
    GestureSequence,
    InterferencePattern,
    InterferenceResponse,
    MovementTrace,
    ParseTreeNode,
    PredictNextResponse,
    SemanticInterpretation,
    TransformResponse,
    TreeWellFormedness,
    ValidNextEntry,
    ValidationResponse,
    AgreementInfo,
    ThetaInfo,
    BindingViolation,
)

logger = logging.getLogger("grammar_engine")

# ---------------------------------------------------------------------------
# Gesture recording — sharded storage for 25GB+ datasets
# ---------------------------------------------------------------------------
# Layout:
#   data/custom/landmarks/{gesture_id}.csv   ← one file per gesture (append-only)
#   data/custom/recording_stats.json         ← atomic O(1) stats cache
#
# Each shard has the same header: lm_0_x, lm_0_y, lm_0_z, ..., lm_20_z
# (gesture_id is implicit from the filename, but also stored as the first column
#  for backward-compat with preprocess.py's _load_webcam)

_CUSTOM_DIR = Path(__file__).resolve().parent.parent / "data" / "custom"
_LANDMARKS_DIR = _CUSTOM_DIR / "landmarks"
_LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
_STATS_JSON = _CUSTOM_DIR / "recording_stats.json"

# CSV header: gesture_id + 63 landmark columns
_CSV_HEADER = ["gesture_id"] + [
    f"lm_{i}_{c}" for i in range(21) for c in ("x", "y", "z")
]


def _read_stats() -> dict:
    """Read the cached stats JSON (O(1))."""
    if _STATS_JSON.exists():
        try:
            return json.loads(_STATS_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"total": 0, "per_gesture": {}}


def _write_stats(stats: dict) -> None:
    """Atomically write stats JSON (write-to-tmp then rename)."""
    tmp = _STATS_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(stats))
    tmp.rename(_STATS_JSON)


class GestureRecordingRequest(BaseModel):
    """Batch of recorded landmark frames for a single gesture."""
    gesture_id: str
    frames: list[list[float]] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Gesture ID mapping: frontend UPPERCASE → Prolog lowercase
# ---------------------------------------------------------------------------

GESTURE_ID_MAP: dict[str, str] = {
    # Pronouns
    "SUBJECT_I":    "subject_i",
    "SUBJECT_YOU":  "subject_you",
    "SUBJECT_HE":   "subject_he",
    "SUBJECT_SHE":  "subject_she",
    "SUBJECT_WE":   "subject_we",
    "SUBJECT_THEY": "subject_they",
    # Verbs
    "WANT":  "verb_want",
    "EAT":   "verb_eat",
    "SEE":   "verb_see",
    "GRAB":  "verb_grab",
    "DRINK": "verb_drink",
    "GO":    "verb_go",
    "STOP":  "verb_stop",
    # Objects
    "FOOD":  "object_food",
    "WATER": "object_water",
    "BOOK":  "object_book",
    "APPLE": "object_apple",
    "BALL":  "object_ball",
    "HOUSE": "object_house",
}

GESTURE_ID_MAP_REVERSE: dict[str, str] = {v: k for k, v in GESTURE_ID_MAP.items()}


def _resolve_gesture_ids(raw: list[str]) -> list[str]:
    """Map frontend gesture IDs to Prolog IDs, with fallback to .lower()."""
    resolved = []
    for g in raw:
        if g in GESTURE_ID_MAP:
            resolved.append(GESTURE_ID_MAP[g])
        else:
            resolved.append(g.lower())
    return resolved


# ---------------------------------------------------------------------------
# Lifespan: instantiate PrologEngine singleton
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        engine = PrologEngine()
        app.state.engine = engine
        logger.info("Grammar engine ready — Prolog modules loaded")
    except Exception as exc:
        logger.error("Failed to initialize PrologEngine: %s", exc)
        app.state.engine = None
    yield


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MLAF Grammar Engine",
    version="1.0.0",
    description=(
        "Prolog-based X-bar syntactic analysis service. "
        "Provides grammaticality judgments, ISL interference detection, "
        "and parse tree generation grounded in Chomskyan generative syntax."
    ),
    lifespan=_lifespan,
    default_response_class=ORJSONResponse,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://multi-modal-gesture-grammar.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: latency header
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _latency_header(request: Request, call_next):
    t0 = time.perf_counter_ns()
    response = await call_next(request)
    elapsed_us = (time.perf_counter_ns() - t0) / 1_000
    response.headers["X-Grammar-Latency-Us"] = str(int(elapsed_us))
    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    engine = getattr(app.state, "engine", None)
    return {
        "status": "ok",
        "engine_loaded": engine is not None,
    }


@app.post("/validate", response_model=ValidationResponse)
async def validate(req: GestureSequence) -> ValidationResponse:
    """Grammaticality judgment + X-bar parse tree + agreement info."""
    engine: PrologEngine = app.state.engine
    gesture_ids = _resolve_gesture_ids(req.gestures)
    result = await engine.validate(gesture_ids)

    # Extract tree well-formedness from parse tree metadata
    tree_wf = None
    pt = result.get("parse_tree")
    if pt and "_well_formed" in pt:
        wf_data = pt["_well_formed"]
        tree_wf = TreeWellFormedness(
            well_formed=wf_data.get("well_formed", False),
            node_count=wf_data.get("node_count"),
            depth=wf_data.get("depth"),
        )

    # Extract compositional semantics
    sem = None
    sem_data = result.get("semantics")
    if sem_data:
        sem = SemanticInterpretation(
            semantic_form=sem_data.get("semantic_form", "unknown"),
            result_type=sem_data.get("result_type", "unknown"),
            complete=sem_data.get("complete", False),
            gesture_types=sem_data.get("gesture_types", []),
        )

    return ValidationResponse(
        grammatical=result["grammatical"],
        parse_tree=_dict_to_tree_node(result.get("parse_tree")),
        agreement=_dict_to_agreement(result.get("agreement")),
        theta=_dict_to_theta(result.get("theta")),
        binding_violations=[
            BindingViolation(**v) for v in result.get("binding_violations", [])
        ],
        tense_resolution=result.get("tense_resolution", "present"),
        grammaticality_score=result.get("grammaticality_score", 0.0),
        semantics=sem,
        tree_well_formedness=tree_wf,
    )


@app.post("/predict-next", response_model=PredictNextResponse)
async def predict_next(req: GestureSequence) -> PredictNextResponse:
    """Valid next gestures with feature constraints."""
    engine: PrologEngine = app.state.engine
    gesture_ids = _resolve_gesture_ids(req.gestures)
    valid_next = await engine.predict_next(gesture_ids)

    entries = []
    for entry in valid_next:
        entries.append(ValidNextEntry(
            grammar_id=entry["grammar_id"],
            category=entry["category"],
            phonological_form=entry["phonological_form"],
            features=None,
            theta_role=entry.get("theta_role"),
        ))

    # Determine parse progress
    if not gesture_ids:
        progress = "incomplete"
        state = "START"
    elif not valid_next:
        progress = "complete"
        state = "END"
    else:
        progress = "incomplete"
        categories = []
        for gid in gesture_ids:
            cat_map = {"d": "NP", "v": "VP", "n": "NP"}
            for e in valid_next:
                pass
            categories.append("NP" if gid.startswith("subject") else "VP" if gid.startswith("verb") else "NP")
        state = categories[-1] if categories else "START"

    return PredictNextResponse(
        valid_next=entries,
        current_state=state,
        parse_progress=progress,
    )


@app.post("/interference", response_model=InterferenceResponse)
async def interference(req: GestureSequence) -> InterferenceResponse:
    """ISL transfer error detection + transform suggestion."""
    engine: PrologEngine = app.state.engine
    gesture_ids = _resolve_gesture_ids(req.gestures)
    interferences = await engine.detect_interference(gesture_ids)

    patterns = [
        InterferencePattern(
            type=i["type"],
            severity=i["severity"],
            description=i["description"],
        )
        for i in interferences
    ]

    has_interference = len(patterns) > 0
    severity = "none"
    if any(p.severity == "error" for p in patterns):
        severity = "error"
    elif any(p.severity == "warning" for p in patterns):
        severity = "warning"

    # Get transform suggestion
    transform_suggestion = None
    if has_interference:
        transform = await engine.transform_isl_to_english(gesture_ids)
        if transform["transform"] != "none":
            eng = transform["english_order"]
            transform_suggestion = f"Suggested order: {' '.join(eng)}"

    return InterferenceResponse(
        has_interference=has_interference,
        patterns=patterns,
        severity=severity,
        transform_suggestion=transform_suggestion,
    )


@app.post("/parse-tree")
async def parse_tree(req: GestureSequence) -> dict:
    """Full X-bar tree + theta satisfaction + binding violations."""
    engine: PrologEngine = app.state.engine
    gesture_ids = _resolve_gesture_ids(req.gestures)
    result = await engine.validate(gesture_ids)
    return result


@app.post("/compose-semantics")
async def compose_semantics(req: GestureSequence) -> dict:
    """Compositional semantic interpretation via lambda calculus (Partee Ch 13)."""
    engine: PrologEngine = app.state.engine
    gesture_ids = _resolve_gesture_ids(req.gestures)
    return await engine.compose_semantics(gesture_ids)


@app.get("/grammar-capabilities")
async def grammar_capabilities() -> dict:
    """Chomsky Hierarchy classification of MLAF's grammar components (Partee Ch 16)."""
    engine: PrologEngine = app.state.engine
    return await engine.get_grammar_capabilities()


@app.post("/transform/isl-to-english", response_model=TransformResponse)
async def transform_isl_to_english(req: GestureSequence) -> TransformResponse:
    """Reordered sequence + movement traces."""
    engine: PrologEngine = app.state.engine
    gesture_ids = _resolve_gesture_ids(req.gestures)
    result = await engine.transform_isl_to_english(gesture_ids)

    return TransformResponse(
        isl_order=result["isl_order"],
        english_order=result["english_order"],
        transform=result["transform"],
        movement_traces=[
            MovementTrace(
                operation=mt["operation"],
                description=mt["description"],
            )
            for mt in result.get("movement_traces", [])
        ],
    )


# ---------------------------------------------------------------------------
# Gesture Recording endpoints
# ---------------------------------------------------------------------------

@app.post("/gesture-recordings/save")
async def save_gesture_recording(req: GestureRecordingRequest) -> dict:
    """Append recorded landmark frames to a per-gesture shard CSV.

    Sharding strategy: one CSV per gesture_id under data/custom/landmarks/.
    This scales to 25GB+ without single-file bottlenecks. File-level locking
    (fcntl.LOCK_EX) ensures safe concurrent writes.
    """
    gesture_id = req.gesture_id
    frames = req.frames

    # Validate frame dimensions (21 landmarks × 3 coords = 63 floats)
    for i, frame in enumerate(frames):
        if len(frame) != 63:
            return {"saved": 0, "error": f"Frame {i} has {len(frame)} values, expected 63"}

    shard_path = _LANDMARKS_DIR / f"{gesture_id}.csv"
    write_header = not shard_path.exists() or shard_path.stat().st_size == 0

    # Append with exclusive file lock — safe for concurrent requests
    with open(shard_path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(_CSV_HEADER)
            for frame in frames:
                writer.writerow([gesture_id] + [f"{v:.6f}" for v in frame])
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # Update cached stats atomically
    stats = _read_stats()
    prev = stats["per_gesture"].get(gesture_id, 0)
    stats["per_gesture"][gesture_id] = prev + len(frames)
    stats["total"] = stats.get("total", 0) + len(frames)
    _write_stats(stats)

    logger.info(
        "Saved %d frames for '%s' (shard: %s, total: %d)",
        len(frames), gesture_id, shard_path.name, stats["total"],
    )
    return {"saved": len(frames), "gesture_id": gesture_id, "shard": shard_path.name}


@app.get("/gesture-recordings/stats")
async def gesture_recording_stats() -> dict:
    """Return per-gesture frame counts — O(1) from cached JSON."""
    return _read_stats()


@app.post("/gesture-recordings/rebuild-stats")
async def rebuild_recording_stats() -> dict:
    """Rebuild stats cache by scanning all shard CSVs. Use if stats drift."""
    per_gesture: dict[str, int] = {}
    total = 0
    for shard in _LANDMARKS_DIR.glob("*.csv"):
        gid = shard.stem
        count = 0
        with open(shard, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for _ in reader:
                count += 1
        per_gesture[gid] = count
        total += count

    stats = {"total": total, "per_gesture": per_gesture}
    _write_stats(stats)
    logger.info("Rebuilt recording stats: %d total frames across %d gestures", total, len(per_gesture))
    return stats


# ---------------------------------------------------------------------------
# Response converters
# ---------------------------------------------------------------------------

def _dict_to_tree_node(d: dict | None) -> ParseTreeNode | None:
    if not d:
        return None
    children = None
    if "children" in d and d["children"]:
        children = [_dict_to_tree_node(c) for c in d["children"] if c]
    return ParseTreeNode(
        label=d.get("label", ""),
        children=children,
        features=d.get("features"),
        trace=d.get("trace"),
        phonological_form=d.get("phonological_form"),
    )


def _dict_to_agreement(d: dict | None) -> AgreementInfo | None:
    if not d:
        return None
    return AgreementInfo(
        agrees=d.get("agrees", False),
        inflected_form=d.get("inflected_form"),
        reason=d.get("reason"),
    )


def _dict_to_theta(d: dict | None) -> ThetaInfo | None:
    if not d:
        return None
    return ThetaInfo(
        satisfied=d.get("satisfied", False),
        roles=d.get("roles"),
        violation_type=d.get("violation_type"),
        missing_count=d.get("missing_count", 0),
    )
