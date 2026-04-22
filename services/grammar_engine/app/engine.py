"""PrologEngine — Singleton wrapper around pyswip for X-bar syntactic analysis.

Thread safety: asyncio.Lock around all Prolog queries (pyswip is not thread-safe).
Loads all Prolog modules at init time. Uses serialize.pl for clean data marshalling
(pyswip returns compound terms as opaque strings; serialize.pl converts them to
nested lists of atoms that pyswip marshals cleanly to Python).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from pyswip import Prolog

logger = logging.getLogger("grammar_engine")

_PROLOG_DIR = Path(__file__).resolve().parent.parent / "prolog"

_MODULES = [
    "lexicon",
    "agreement",
    "subcategorization",
    "binding",
    "movement",
    "isl_grammar",
    "contrastive",
    "xbar",
    "tree_validation",
    "compositional",
    "chomsky_hierarchy",
    "serialize",
]


def _flatten_cons_cells(node: Any) -> list | Any:
    """Recursively flatten pyswip cons cell representation.

    pyswip marshals Prolog lists as nested ['[|]', head, tail] structures.
    This converts them back to flat Python lists.
    """
    if isinstance(node, list):
        if len(node) == 3 and node[0] == '[|]':
            head = _flatten_cons_cells(node[1])
            tail = _flatten_cons_cells(node[2])
            if isinstance(tail, list):
                return [head] + tail
            return [head]
        return [_flatten_cons_cells(item) for item in node]
    return node


def _extract_features(node: Any) -> dict[str, str] | None:
    """Extract feature dict from a Prolog feature list.

    After flattening, feature lists look like:
    [['=', 'person', 1], ['=', 'number', 'sg'], ...]
    """
    if not isinstance(node, list):
        return None
    features: dict[str, str] = {}
    for item in node:
        if isinstance(item, list) and len(item) == 3 and item[0] == '=':
            features[str(item[1])] = str(item[2])
    if features:
        return features
    return None


def _nested_list_to_tree(node: Any) -> dict[str, Any] | None:
    """Convert serialize.pl nested list tree to dict.

    Format from Prolog: [Label, Child1, Child2, ...]
    Handles cons cell representation from pyswip.
    """
    node = _flatten_cons_cells(node)

    if node is None:
        return None

    if isinstance(node, str):
        return {"label": node}

    if isinstance(node, (int, float)):
        return {"label": str(node)}

    if isinstance(node, list) and len(node) >= 1:
        label = str(node[0])
        children: list[dict[str, Any]] = []
        features: dict[str, str] | None = None

        for arg in node[1:]:
            if isinstance(arg, list):
                # Check if it's a feature list [['=', k, v], ...]
                feat = _extract_features(arg)
                if feat:
                    features = feat
                else:
                    child = _nested_list_to_tree(arg)
                    if child:
                        children.append(child)
            elif isinstance(arg, str):
                children.append({"label": arg})
            elif isinstance(arg, (int, float)):
                children.append({"label": str(arg)})

        result: dict[str, Any] = {"label": label}
        if children:
            result["children"] = children
        if features:
            result["features"] = features
        return result

    return None


class PrologEngine:
    """Singleton wrapper around SWI-Prolog via pyswip."""

    _instance: PrologEngine | None = None
    _lock: asyncio.Lock

    def __new__(cls) -> PrologEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._prolog = Prolog()
        self._lock = asyncio.Lock()
        self._load_modules()
        self._initialized = True
        logger.info("PrologEngine initialized with %d modules", len(_MODULES))

    def _load_modules(self) -> None:
        for mod in _MODULES:
            path = _PROLOG_DIR / f"{mod}.pl"
            if not path.exists():
                raise FileNotFoundError(f"Prolog module not found: {path}")
            self._prolog.consult(str(path))
            logger.debug("Loaded Prolog module: %s", mod)

    # ------------------------------------------------------------------
    # Public async methods
    # ------------------------------------------------------------------

    async def validate(self, gesture_ids: list[str]) -> dict[str, Any]:
        async with self._lock:
            return self._validate_sync(gesture_ids)

    async def predict_next(self, gesture_ids: list[str]) -> list[dict[str, Any]]:
        async with self._lock:
            return self._predict_next_sync(gesture_ids)

    async def detect_interference(self, gesture_ids: list[str]) -> list[dict[str, Any]]:
        async with self._lock:
            return self._detect_interference_sync(gesture_ids)

    async def transform_isl_to_english(self, gesture_ids: list[str]) -> dict[str, Any]:
        async with self._lock:
            return self._transform_sync(gesture_ids)

    async def get_parse_tree(self, gesture_ids: list[str]) -> dict[str, Any] | None:
        async with self._lock:
            return self._get_parse_tree_sync(gesture_ids)

    async def compose_semantics(self, gesture_ids: list[str]) -> dict[str, Any]:
        async with self._lock:
            return self._compose_semantics_sync(gesture_ids)

    async def get_grammar_capabilities(self) -> dict[str, Any]:
        async with self._lock:
            return self._get_grammar_capabilities_sync()

    # ------------------------------------------------------------------
    # Synchronous implementations (called under lock)
    # ------------------------------------------------------------------

    def _validate_sync(self, gesture_ids: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "grammatical": False,
            "parse_tree": None,
            "agreement": None,
            "theta": None,
            "binding_violations": [],
            "tense_resolution": "present",
            "grammaticality_score": 0.0,
        }

        # 1. Parse tree
        parse_tree = self._get_parse_tree_sync(gesture_ids)
        if parse_tree:
            result["parse_tree"] = parse_tree
            result["grammatical"] = True
            score = 1.0
        else:
            score = 0.3

        # 2. Agreement
        result["agreement"] = self._check_agreement_sync(gesture_ids)
        if result["agreement"] and not result["agreement"].get("agrees", True):
            score -= 0.2

        # 3. Theta criterion
        result["theta"] = self._check_theta_sync(gesture_ids)
        if result["theta"] and not result["theta"].get("satisfied", True):
            score -= 0.2
            result["grammatical"] = False

        # 4. Binding violations
        result["binding_violations"] = self._check_binding_sync(gesture_ids)
        if result["binding_violations"]:
            score -= 0.1 * len(result["binding_violations"])
            result["grammatical"] = False

        # 5. Tense
        result["tense_resolution"] = self._resolve_tense(gesture_ids)
        result["grammaticality_score"] = max(0.0, min(1.0, score))

        # 6. Compositional semantics (Frege's Principle)
        result["semantics"] = self._compose_semantics_sync(gesture_ids)

        return result

    def _predict_next_sync(self, gesture_ids: list[str]) -> list[dict[str, Any]]:
        valid_next: list[dict[str, Any]] = []
        categories = self._categorize_ids(gesture_ids)

        if not gesture_ids:
            expected_cat = "d"
        elif all(c == "subj" for c in categories):
            expected_cat = "v"
        elif "verb" in categories and "obj" not in categories:
            expected_cat = "n"
        else:
            return []

        if expected_cat == "d":
            results = list(self._prolog.query(
                "lexicon:lex(ID, Form, d, Feats), member(case=nom, Feats)"
            ))
            # Deduplicate — pyswip may return multiple bindings
            seen = set()
            for r in results:
                gid = str(r.get("ID", ""))
                if gid not in seen:
                    seen.add(gid)
                    valid_next.append(self._build_lex_entry(gid, "d"))
        elif expected_cat == "v":
            results = list(self._prolog.query("lexicon:lex(ID, Form, v, _)"))
            seen = set()
            for r in results:
                gid = str(r.get("ID", ""))
                if gid not in seen:
                    seen.add(gid)
                    valid_next.append(self._build_lex_entry(gid, "v"))
        elif expected_cat == "n":
            results = list(self._prolog.query("lexicon:lex(ID, Form, n, _)"))
            verb_id = self._find_verb(gesture_ids)
            seen = set()
            for r in results:
                gid = str(r.get("ID", ""))
                if gid not in seen:
                    seen.add(gid)
                    entry = self._build_lex_entry(gid, "n")
                    if verb_id:
                        role_results = list(self._prolog.query(
                            f"subcategorization:role_assignment({verb_id}, 2, Role)"
                        ))
                        if role_results:
                            entry["theta_role"] = str(role_results[0].get("Role", ""))
                    valid_next.append(entry)

        return valid_next

    def _detect_interference_sync(self, gesture_ids: list[str]) -> list[dict[str, Any]]:
        ids = self._to_prolog_list(gesture_ids)
        query = f"serialize:serialize_interference({ids}, Patterns)"
        results = list(self._prolog.query(query))

        interferences: list[dict[str, Any]] = []
        if results:
            patterns = results[0].get("Patterns", [])
            for p in patterns:
                if isinstance(p, list) and len(p) >= 3:
                    interferences.append({
                        "type": str(p[0]),
                        "severity": str(p[1]),
                        "description": str(p[2]),
                    })
        return interferences

    def _transform_sync(self, gesture_ids: list[str]) -> dict[str, Any]:
        ids = self._to_prolog_list(gesture_ids)
        query = f"serialize:serialize_transform({ids}, Eng, TType, Ops)"
        results = list(self._prolog.query(query))

        if results:
            r = results[0]
            eng_order = r.get("Eng", gesture_ids)
            if isinstance(eng_order, list):
                eng_list = [str(x) for x in eng_order]
            else:
                eng_list = gesture_ids

            transform_type = str(r.get("TType", "unknown"))
            raw_ops = r.get("Ops", [])
            operations: list[dict[str, str]] = []
            if isinstance(raw_ops, list):
                for op in raw_ops:
                    if isinstance(op, list) and len(op) >= 2:
                        operations.append({
                            "operation": str(op[0]),
                            "description": str(op[1]),
                        })

            return {
                "isl_order": gesture_ids,
                "english_order": eng_list,
                "transform": transform_type,
                "movement_traces": operations,
            }

        return {
            "isl_order": gesture_ids,
            "english_order": gesture_ids,
            "transform": "none",
            "movement_traces": [],
        }

    def _get_parse_tree_sync(self, gesture_ids: list[str]) -> dict[str, Any] | None:
        ids = self._to_prolog_list(gesture_ids)
        query = f"serialize:serialize_tree({ids}, TreeList)"
        results = list(self._prolog.query(query))

        if results:
            tree_list = results[0].get("TreeList")
            if tree_list:
                tree = _nested_list_to_tree(tree_list)
                # Validate tree well-formedness (Partee et al., Ch 16)
                tree_valid = self._validate_tree_sync(gesture_ids)
                if tree and tree_valid is not None:
                    tree["_well_formed"] = tree_valid
                return tree
        return None

    def _validate_tree_sync(self, gesture_ids: list[str]) -> dict[str, Any] | None:
        """Validate parse tree well-formedness using formal conditions.

        Uses the raw Prolog parse tree (not serialized) for validation,
        since well_formed_tree/1 operates on compound terms.
        """
        ids = self._to_prolog_list(gesture_ids)
        try:
            # Query the raw parse tree and validate it directly
            wf_query = (
                f"xbar:parse_sentence({ids}, T, []), "
                f"tree_validation:well_formed_tree(T)"
            )
            wf_results = list(self._prolog.query(wf_query))
            is_well_formed = len(wf_results) > 0

            # Get detailed report from raw tree
            report_query = (
                f"xbar:parse_sentence({ids}, T, []), "
                f"tree_validation:validate_tree_structure(T, Report)"
            )
            report_results = list(self._prolog.query(report_query))

            report: dict[str, Any] = {"well_formed": is_well_formed}
            if report_results:
                raw_report = report_results[0].get("Report", [])
                raw_report = _flatten_cons_cells(raw_report)
                if isinstance(raw_report, list):
                    for item in raw_report:
                        if isinstance(item, list) and len(item) == 2:
                            report[str(item[0])] = str(item[1]) if not isinstance(item[1], (int, float)) else item[1]
            return report
        except Exception as exc:
            logger.warning("Tree validation failed: %s", exc)
            return {"well_formed": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Agreement, Theta, Binding checks
    # ------------------------------------------------------------------

    def _check_agreement_sync(self, gesture_ids: list[str]) -> dict[str, Any] | None:
        subj_id = None
        verb_id = None
        for gid in gesture_ids:
            cat = self._get_category(gid)
            if cat == "d" and subj_id is None:
                subj_id = gid
            elif cat == "v" and verb_id is None:
                verb_id = gid

        if not subj_id or not verb_id:
            return None

        query = (
            f"serialize:serialize_agreement({subj_id}, {verb_id}, "
            f"Agrees, InflForm, _)"
        )
        results = list(self._prolog.query(query))
        if results:
            r = results[0]
            agrees = str(r.get("Agrees", "no")) == "yes"
            inflected_form = str(r.get("InflForm", "unknown"))
            return {
                "agrees": agrees,
                "inflected_form": inflected_form,
                "subject_id": subj_id,
                "verb_id": verb_id,
            }
        return None

    def _check_theta_sync(self, gesture_ids: list[str]) -> dict[str, Any] | None:
        verb_id = self._find_verb(gesture_ids)
        if not verb_id:
            return None

        args = [gid for gid in gesture_ids if gid != verb_id]
        args_list = self._to_prolog_list(args)
        query = f"subcategorization:check_theta_criterion({verb_id}, {args_list}, Result)"
        results = list(self._prolog.query(query))

        if results:
            result_str = str(results[0].get("Result", ""))
            if result_str == "satisfied":
                role_results = list(self._prolog.query(
                    f"lexicon:theta_grid({verb_id}, Roles)"
                ))
                roles = []
                if role_results:
                    roles_term = role_results[0].get("Roles", [])
                    if isinstance(roles_term, list):
                        roles = [str(r) for r in roles_term]
                return {"satisfied": True, "roles": roles}
            else:
                # Parse violation(type, count) string
                if "missing_args" in result_str:
                    return {"satisfied": False, "violation_type": "missing_args", "missing_count": 1}
                elif "extra_args" in result_str:
                    return {"satisfied": False, "violation_type": "extra_args", "missing_count": 1}
                return {"satisfied": False, "violation_type": "unknown", "missing_count": 0}

        return None

    def _check_binding_sync(self, gesture_ids: list[str]) -> list[dict[str, Any]]:
        ids = self._to_prolog_list(gesture_ids)
        query = f"binding:check_binding({ids}, local, Violations)"
        results = list(self._prolog.query(query))

        violations: list[dict[str, Any]] = []
        if results:
            raw = results[0].get("Violations", [])
            if isinstance(raw, list):
                for v in raw:
                    v_str = str(v)
                    if "principle_" in v_str:
                        # Parse violation(principle_x, id, message) string
                        violations.append({
                            "principle": "principle_b" if "principle_b" in v_str else
                                         "principle_c" if "principle_c" in v_str else
                                         "principle_a",
                            "gesture_id": "",
                            "message": v_str,
                        })
        return violations

    # ------------------------------------------------------------------
    # Compositional Semantics (Partee et al., Ch 13)
    # ------------------------------------------------------------------

    def _compose_semantics_sync(self, gesture_ids: list[str]) -> dict[str, Any]:
        """Compute compositional semantic representation via lambda calculus."""
        ids = self._to_prolog_list(gesture_ids)
        try:
            query = f"compositional:compose_sentence({ids}, Sem, Type)"
            results = list(self._prolog.query(query))

            if results:
                r = results[0]
                sem_raw = r.get("Sem", "unknown")
                type_raw = r.get("Type", "unknown")

                return {
                    "semantic_form": self._term_to_string(sem_raw),
                    "result_type": str(type_raw),
                    "complete": str(type_raw) == "t",
                    "gesture_ids": gesture_ids,
                    "gesture_types": self._get_semantic_types(gesture_ids),
                }

            return {
                "semantic_form": "unknown",
                "result_type": "unknown",
                "complete": False,
                "gesture_ids": gesture_ids,
                "gesture_types": self._get_semantic_types(gesture_ids),
            }
        except Exception as exc:
            logger.warning("Compositional semantics failed: %s", exc)
            return {
                "semantic_form": "error",
                "result_type": "error",
                "complete": False,
                "error": str(exc),
            }

    def _get_semantic_types(self, gesture_ids: list[str]) -> list[dict[str, str]]:
        """Get semantic type for each gesture ID."""
        types: list[dict[str, str]] = []
        for gid in gesture_ids:
            try:
                results = list(self._prolog.query(
                    f"compositional:semantic_type({gid}, Type)"
                ))
                if results:
                    types.append({
                        "gesture_id": gid,
                        "type": str(results[0].get("Type", "unknown")),
                    })
                else:
                    types.append({"gesture_id": gid, "type": "unknown"})
            except Exception:
                types.append({"gesture_id": gid, "type": "error"})
        return types

    def _term_to_string(self, term: Any) -> str:
        """Convert a pyswip term to a readable string.

        Cleans up pyswip's Functor representation and unwraps entity() wrappers.
        """
        import re

        if isinstance(term, (int, float)):
            return str(term)
        if isinstance(term, list):
            flat = _flatten_cons_cells(term)
            if isinstance(flat, list):
                parts = [self._term_to_string(t) for t in flat]
                return f"[{', '.join(parts)}]"
            return self._term_to_string(flat)
        # Handle pyswip Functor objects
        if hasattr(term, 'name') and hasattr(term, 'args'):
            name = str(term.name)
            if name == 'entity' and term.args and len(list(term.args)) == 1:
                return str(list(term.args)[0])
            if term.args:
                args_str = ", ".join(self._term_to_string(a) for a in term.args)
                return f"{name}({args_str})"
            return name
        # String cleanup: pyswip sometimes returns stringified Functor refs
        s = str(term)
        # Clean up Functor(id,arity,name) → name
        s = re.sub(r'Functor\(\d+,\d+,(\w+)\)', r'\1', s)
        return s

    # ------------------------------------------------------------------
    # Chomsky Hierarchy (Partee et al., Ch 16)
    # ------------------------------------------------------------------

    def _get_grammar_capabilities_sync(self) -> dict[str, Any]:
        """Get MLAF's formal grammar classification report."""
        try:
            components: list[dict[str, Any]] = []
            results = list(self._prolog.query(
                "chomsky_hierarchy:grammar_class(Component, Type)"
            ))
            for r in results:
                comp = str(r.get("Component", ""))
                typ = str(r.get("Type", ""))
                level_results = list(self._prolog.query(
                    f"chomsky_hierarchy:chomsky_level({typ}, Level)"
                ))
                level = int(level_results[0].get("Level", -1)) if level_results else -1
                components.append({
                    "component": comp,
                    "chomsky_type": typ,
                    "level": level,
                })

            # Classify by level
            by_level: dict[str, list[str]] = {
                "type_3_regular": [],
                "type_2_context_free": [],
                "type_1_context_sensitive": [],
            }
            for c in components:
                ct = c["chomsky_type"]
                if ct in by_level:
                    by_level[ct].append(c["component"])

            return {
                "components": components,
                "summary": {
                    "total": len(components),
                    "regular_count": len(by_level["type_3_regular"]),
                    "context_free_count": len(by_level["type_2_context_free"]),
                    "context_sensitive_count": len(by_level["type_1_context_sensitive"]),
                },
                "by_level": by_level,
                "overall_power": "mildly_context_sensitive",
                "note": "Natural languages are mildly context-sensitive (Joshi 1985). "
                        "MLAF uses CFG base + context-sensitive feature checking.",
            }
        except Exception as exc:
            logger.warning("Grammar capabilities query failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_prolog_list(self, ids: list[str]) -> str:
        items = ", ".join(ids)
        return f"[{items}]"

    def _get_category(self, gid: str) -> str | None:
        results = list(self._prolog.query(f"lexicon:lex({gid}, _, Cat, _)"))
        if results:
            return str(results[0].get("Cat", ""))
        return None

    def _find_verb(self, gesture_ids: list[str]) -> str | None:
        for gid in gesture_ids:
            if self._get_category(gid) == "v":
                return gid
        return None

    def _build_lex_entry(self, gid: str, cat: str) -> dict[str, Any]:
        results = list(self._prolog.query(f"lexicon:lex({gid}, Form, _, _)"))
        form = str(results[0].get("Form", "")) if results else gid
        return {
            "grammar_id": gid,
            "category": cat,
            "phonological_form": form,
        }

    def _categorize_ids(self, gesture_ids: list[str]) -> list[str]:
        categories = []
        for gid in gesture_ids:
            cat = self._get_category(gid)
            if cat == "d":
                categories.append("subj")
            elif cat == "v":
                categories.append("verb")
            elif cat == "n":
                categories.append("obj")
            else:
                categories.append("unknown")
        return categories

    def _resolve_tense(self, gesture_ids: list[str]) -> str:
        for gid in gesture_ids:
            results = list(self._prolog.query(
                f"lexicon:lex({gid}, _, v, Feats), member(tense=T, Feats)"
            ))
            if results:
                return str(results[0].get("T", "present"))
        return "present"
