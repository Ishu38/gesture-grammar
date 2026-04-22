"""Pydantic request/response models for the Prolog-based grammar engine."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class FeatureBundle(BaseModel):
    """Morphosyntactic feature bundle from Prolog unification."""
    person: int | None = None
    number: str | None = None          # "sg" | "pl"
    case: str | None = None            # "nom" | "acc"
    tense: str | None = None           # "pres" | "past" | "fut"
    transitive: str | None = None      # "yes" | "no"
    vform: str | None = None           # "base" | "inflected"
    anaphor: str | None = None         # "yes" | "no"
    pronominal: str | None = None      # "yes" | "no"
    countable: str | None = None       # "yes" | "no"


class ParseTreeNode(BaseModel):
    """Recursive X-bar parse tree node."""
    label: str
    children: list[ParseTreeNode] | None = None
    features: dict[str, str | int | bool] | None = None
    trace: str | None = None
    phonological_form: str | None = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class GestureSequence(BaseModel):
    """Input: list of gesture IDs from the frontend token buffer."""
    gestures: list[str] = Field(..., min_length=1, max_length=20)
    spatial_modifiers: list[str] | None = None


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ValidNextEntry(BaseModel):
    """A single valid-next gesture with feature constraints."""
    grammar_id: str
    category: str                        # "d" | "v" | "n"
    phonological_form: str
    features: FeatureBundle | None = None
    theta_role: str | None = None


class PredictNextResponse(BaseModel):
    """Valid next gestures from the grammar engine."""
    valid_next: list[ValidNextEntry]
    current_state: str
    parse_progress: str                  # "incomplete" | "complete" | "error"


class AgreementInfo(BaseModel):
    """Subject-verb agreement result."""
    agrees: bool
    inflected_form: str | None = None
    subject_features: dict[str, str | int] | None = None
    verb_features: dict[str, str | int] | None = None
    reason: str | None = None


class ThetaInfo(BaseModel):
    """Theta criterion satisfaction."""
    satisfied: bool
    roles: list[str] | None = None
    violation_type: str | None = None
    missing_count: int = 0


class BindingViolation(BaseModel):
    """A single binding theory violation."""
    principle: str                       # "principle_a" | "principle_b" | "principle_c"
    gesture_id: str
    message: str


class SemanticInterpretation(BaseModel):
    """Compositional semantic representation (Frege's Principle, Partee Ch 13)."""
    semantic_form: str = "unknown"
    result_type: str = "unknown"
    complete: bool = False
    gesture_types: list[dict[str, str]] = Field(default_factory=list)


class TreeWellFormedness(BaseModel):
    """Formal tree well-formedness report (Partee Ch 16)."""
    well_formed: bool = False
    node_count: int | str | None = None
    depth: int | str | None = None


class ValidationResponse(BaseModel):
    """Full grammaticality judgment from the engine."""
    grammatical: bool
    parse_tree: ParseTreeNode | None = None
    agreement: AgreementInfo | None = None
    theta: ThetaInfo | None = None
    binding_violations: list[BindingViolation] = Field(default_factory=list)
    tense_resolution: str | None = None  # "present" | "past" | "future"
    grammaticality_score: float = 0.0    # 0.0 - 1.0
    semantics: SemanticInterpretation | None = None
    tree_well_formedness: TreeWellFormedness | None = None


class InterferencePattern(BaseModel):
    """A single ISL interference pattern."""
    type: str                            # "sov_order" | "topic_fronting" | "object_drop"
    severity: str                        # "error" | "warning"
    description: str


class InterferenceResponse(BaseModel):
    """ISL transfer error detection result."""
    has_interference: bool
    patterns: list[InterferencePattern] = Field(default_factory=list)
    severity: str = "none"               # "none" | "warning" | "error"
    transform_suggestion: str | None = None


class MovementTrace(BaseModel):
    """Record of a syntactic movement operation."""
    operation: str
    description: str


class TransformResponse(BaseModel):
    """ISL-to-English word order transformation."""
    isl_order: list[str]
    english_order: list[str]
    transform: str                       # "sov_to_svo" | "osv_to_svo" | "object_drop_repair" | "none"
    movement_traces: list[MovementTrace] = Field(default_factory=list)
