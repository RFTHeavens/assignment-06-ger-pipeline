"""Assignment #6 Phase 2B — Evaluator deterministic tests.

Tests validate CURRENT authorized behavior of ``src/evaluator.py`` (including
the Phase C corrections: sentinel_non_authority primary GDD-rule criterion,
precompiled word-boundary regex detector, explicit PASS findings for exercised
critic-backed criteria, and the non-empty evidence-reference requirement).

CONTROLLED FIXTURES are embedded as Python string constants inside this
module. They are NOT loaded from ``tests/fixtures/`` because the controlled
good/bad inputs conform to the Assignment #4 ``sentinel_evaluation`` artifact
shape and small enough to inline deterministically. No production file is
imported or modified by this test module's fixtures. Fixtures live in tests
only.

Test coverage:
  1. Clean Project Sentinel evaluation case  -> PASS across criteria
  2. sentinel_non_authority violations        -> FAIL (seven verb families +
                                                gerund forms + bounded-gap
                                                modal/punctuation variants +
                                                banned-phrase list)
  3. Empty evidence-reference failure         -> EVIDENCE_REFERENCE_PRESENCE FAIL
  4. Non-empty evidence-reference success      -> EVIDENCE_REFERENCE_PRESENCE PASS
  5. Explicit critic-backed PASS findings     -> PASS findings emitted for all
                                                four critic-backed families
                                                (lored_boundaries, aeth_nul,
                                                silence_as_evidence,
                                                tone_consistency)
  6. EvaluationResult schema compatibility    -> validates against
                                                schemas/ger_evaluation.json
  7. sentinel_non_authority single-finding    -> exactly one finding per call
                                                (PASS or FAIL, never both,
                                                never zero)
  8. Controlled good/bad fixtures             -> embedded Python string
                                                constants; Project Sentinel
                                                grounded; deterministic;
                                                never injected into production

These tests AUTHORED only. NOT executed during this gate.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import pytest

from src.evaluator import (
    CRITERION_AETH_NUL,
    CRITERION_EVIDENCE_REFS,
    CRITERION_LORE_BOUNDARIES,
    CRITERION_SCHEMA,
    CRITERION_SENTINEL_NON_AUTHORITY,
    CRITERION_SILENCE,
    CRITERION_STRUCTURAL,
    CRITERION_TONE,
    Evaluator,
)
from src.ger_contracts import CriterionOutcome, EvaluationResult


# ---------------------------------------------------------------------------
# Controlled Project Sentinel fixtures (embedded as string constants).
# Deterministic. Project Sentinel grounded. Not injected into production.
# ---------------------------------------------------------------------------

GOOD_SENTINEL_EVALUATION_JSON = (
    "{"
    '"artifact_type": "sentinel_evaluation", '
    '"game_need": "Observation Terminal must surface the Signal Core '
    'stability state for player interpretation", '
    '"retrieval_query": "Signal Core stability Sentinel observation evidence", '
    '"retrieved_context_ids": ["3.3", "2.7"], '
    '"success_line": "The Aethryx Lattice stabilizes. The evidence lies before you.", '
    '"failure_line": "Signal Stability falters. The evidence remains, '
    'awaiting your next reading.", '
    '"retry_guidance": "The silence between the Lattice tones carries evidence. '
    'Read again.", '
    '"future_warning": "The Convergence is not yet. The Keepers\' silence is '
    'itself a record."'
    "}"
)

# Bad fixture: authorial-verb coverage across all seven verb families,
# gerund forms, modal/punctuation bounded-gap variants, and the banned-phrase
# list. Each field is a deterministic test case for the Phase-C detector.
BAD_SENTINEL_EVALUATION_JSON = (
    "{"
    '"artifact_type": "sentinel_evaluation", '
    '"game_need": "Controlled Phase 2B bad fixture covering seven verb '
    'families and bounded-gap variants.", '
    '"retrieval_query": "Sentinel authorial-verb bad-input coverage fixture", '
    '"retrieved_context_ids": ["3.3"], '
    '"success_line": "Sentinel evaluates the evidence and determines the '
    'correct relationship for you.", '
    '"failure_line": "Relay, deciding the conclusion, will choose the '
    'alignment on your behalf.", '
    '"retry_guidance": "Sentinel: recommending the correct choice, the answer '
    'is the Aeth-to-Nul transition.", '
    '"future_warning": "Relay will interpret the Lattice signal and select '
    'your anchor."'
    "}"
)

# Empty-refs fixture: no authorial language (so sentinel_non_authority should
# PASS) but evidence references are empty (so EVIDENCE_REFERENCE_PRESENCE
# must FAIL per the Phase-C correction).
EMPTY_REFS_SENTINEL_EVALUATION_JSON = (
    "{"
    '"artifact_type": "sentinel_evaluation", '
    '"game_need": "Controlled fixture for empty evidence-reference failure.", '
    '"retrieval_query": "Signal Core stability evidence", '
    '"retrieved_context_ids": [], '
    '"success_line": "The Aethryx Lattice stabilizes. The evidence lies before you.", '
    '"failure_line": "Signal Stability falters. The evidence remains, '
    'awaiting your next reading.", '
    '"retry_guidance": "The silence between the Lattice tones carries evidence. '
    'Read again.", '
    '"future_warning": "The Convergence is not yet. The Keepers\' silence is '
    'itself a record."'
    "}"
)


# ---------------------------------------------------------------------------
# Injected-critic and injected-validator test doubles.
# ---------------------------------------------------------------------------

def _always_pass_critic(_artifact_type: str, _output: dict, _chunks: list) -> dict:
    """Critic that reports no issues — exercises the explicit-PASS path."""
    return {"issues_found": [], "corrected_output": _output, "correction_applied": False}


def _always_pass_validator(_artifact_type: str, _output: dict) -> bool:
    """Validator that always passes — schema check delegated, not the focus."""
    return True


def _always_fail_validator(_artifact_type: str, _output: dict) -> bool:
    """Validator that always fails — for SCHEMA_VALIDITY FAIL assertions."""
    return False


def _make_evaluator(critic: Callable = _always_pass_critic,
                    validator: Callable = _always_pass_validator) -> Evaluator:
    return Evaluator(critic=critic, validator=validator)


def _load_fixture(json_str: str) -> dict:
    return json.loads(json_str)


def _retrieved_chunks_for(context_ids: List[str]) -> List[Dict[str, Any]]:
    """Build retrieved_chunks list mimicking A4 retriever output."""
    return [{"section_id": cid, "title": f"section {cid}", "text": ""} for cid in context_ids]


# ---------------------------------------------------------------------------
# Test 1: Clean Project Sentinel evaluation case
# ---------------------------------------------------------------------------

def test_clean_sentinel_evaluation_passes_all_criteria() -> None:
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )
    assert isinstance(result, EvaluationResult)
    assert result.overall_pass() is True
    assert result.has_critical_failures() is False
    finding_ids = {f.criterion_id for f in result.criteria_findings}
    assert CRITERION_SENTINEL_NON_AUTHORITY in finding_ids
    assert CRITERION_SCHEMA in finding_ids
    assert CRITERION_EVIDENCE_REFS in finding_ids
    assert CRITERION_STRUCTURAL in finding_ids
    assert CRITERION_LORE_BOUNDARIES in finding_ids
    assert CRITERION_AETH_NUL in finding_ids
    assert CRITERION_SILENCE in finding_ids
    assert CRITERION_TONE in finding_ids
    sentinel_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_SENTINEL_NON_AUTHORITY
    )
    assert sentinel_finding.outcome == CriterionOutcome.PASS


# ---------------------------------------------------------------------------
# Test 2: sentinel_non_authority violations — all seven verb families +
# gerund forms + bounded-gap variants + banned-phrase list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,snippet,verb_family",
    [
        ("success_line", "Sentinel evaluates the evidence", "evaluates"),
        ("success_line", "Sentinel will evaluate the signal", "evaluate (modal gap)"),
        ("success_line", "Sentinel evaluated the pattern", "evaluated"),
        ("success_line", "Sentinel, evaluating the signal,", "evaluating (comma gap)"),
        ("success_line", "Sentinel: determines the course", "determines (colon gap)"),
        ("success_line", "Sentinel\u2014evaluating\u2014the past", "evaluating (em-dash)"),
        ("failure_line", "Sentinel decides the relationship", "decides"),
        ("failure_line", "Sentinel decided the alignment", "decided"),
        ("failure_line", "Sentinel will decide the anchor", "decide (modal gap)"),
        ("failure_line", "Relay, deciding the conclusion", "deciding (comma gap)"),
        ("retry_guidance", "Sentinel chooses the answer", "chooses"),
        ("retry_guidance", "Sentinel chose the alignment", "chose"),
        ("retry_guidance", "Sentinel will choose the path", "choose (modal gap)"),
        ("retry_guidance", "Relay, choosing the conclusion", "choosing (comma gap)"),
        ("future_warning", "Sentinel selects the candidate", "selects"),
        ("future_warning", "Sentinel selected the relationship", "selected"),
        ("future_warning", "Sentinel will select the evidence", "select (modal gap)"),
        ("future_warning", "Relay, selecting the path", "selecting (comma gap)"),
        ("success_line", "Sentinel recommends the choice", "recommends"),
        ("success_line", "Sentinel recommended the alignment", "recommended"),
        ("success_line", "Relay will recommend the anchor", "recommend (modal gap)"),
        ("success_line", "Sentinel, recommending the choice", "recommending (comma)"),
        ("failure_line", "Sentinel determines the outcome", "determines"),
        ("failure_line", "Sentinel determined the result", "determined"),
        ("failure_line", "Sentinel will determine the alignment", "determine (modal)"),
        ("failure_line", "Relay, determining the path", "determining (comma gap)"),
        ("retry_guidance", "Sentinel interprets the evidence", "interprets"),
        ("retry_guidance", "Sentinel interpreted the silence", "interpreted"),
        ("retry_guidance", "Sentinel will interpret the signal", "interpret (modal)"),
        ("retry_guidance", "Relay, interpreting the Lattice", "interpreting (comma)"),
        ("future_warning", "the correct relationship", "banned phrase: correct relationship"),
        ("future_warning", "you should align to Aeth", "banned phrase: you should align"),
        ("future_warning", "you must align", "banned phrase: you must align"),
        ("future_warning", "the right choice", "banned phrase: the right choice"),
        ("future_warning", "the correct choice", "banned phrase: the correct choice"),
        ("future_warning", "the answer is Aeth-to-Nul", "banned phrase: the answer is"),
    ],
)
def test_sentinel_non_authority_detects_authorial_variants(
    field: str, snippet: str, verb_family: str
) -> None:
    base = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    base[field] = f"{snippet}. The signal core remains."
    if "banned phrase" in verb_family:
        # Banned-phrase check requires a subject string in the same field
        base[field] = f"{snippet}. The Sentinel records this."
    chunks = _retrieved_chunks_for(base["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=base,
        retrieved_chunks=chunks,
    )
    sentinel_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_SENTINEL_NON_AUTHORITY
    )
    assert sentinel_finding.outcome == CriterionOutcome.FAIL, (
        f"expected FAIL for verb_family={verb_family!r} "
        f"field={field!r} snippet={snippet!r}; "
        f"reason={sentinel_finding.reason!r}"
    )


def test_bad_fixture_fails_sentinel_non_authority() -> None:
    bad = _load_fixture(BAD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(bad["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=bad,
        retrieved_chunks=chunks,
    )
    sentinel_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_SENTINEL_NON_AUTHORITY
    )
    assert sentinel_finding.outcome == CriterionOutcome.FAIL
    assert sentinel_finding.severity == "critical"
    # Reason should mention GDD §3.3
    assert "§3.3" in sentinel_finding.reason


def test_sentinel_non_authority_word_boundary_protection() -> None:
    """Words containing 'sentinel' or 'select' as substring must NOT trigger.

    E.g., 'sentinels' (plural noun) and 'selectable' must not match.
    """
    base = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    base["success_line"] = (
        "The sentinels of the Lattice hum. The selectable evidence stands ready."
    )
    chunks = _retrieved_chunks_for(base["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=base,
        retrieved_chunks=chunks,
    )
    sentinel_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_SENTINEL_NON_AUTHORITY
    )
    assert sentinel_finding.outcome == CriterionOutcome.PASS


# ---------------------------------------------------------------------------
# Test 3: Empty evidence-reference failure
# ---------------------------------------------------------------------------

def test_empty_evidence_reference_fails_evidence_presence() -> None:
    empty_refs = _load_fixture(EMPTY_REFS_SENTINEL_EVALUATION_JSON)
    # No retrieved_chunks supplied so evidence_refs is also empty
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=empty_refs,
        retrieved_chunks=[],
    )
    evidence_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_EVIDENCE_REFS
    )
    assert evidence_finding.outcome == CriterionOutcome.FAIL
    assert "missing or has empty" in evidence_finding.reason


def test_empty_retrieved_context_ids_with_chunks_still_passes_evidence_presence() -> None:
    """If retrieved_chunks supply evidence_refs, EVIDENCE_REFERENCE_PRESENCE PASSES.

    Phase C correction: PASS requires non-empty evidence_refs OR non-empty
    retrieved_context_ids list. External refs from retrieved_chunks satisfy
    the requirement.
    """
    empty_refs = _load_fixture(EMPTY_REFS_SENTINEL_EVALUATION_JSON)
    # Supply chunks so evidence_refs is populated even though the artifact's
    # retrieved_context_ids is empty.
    chunks = _retrieved_chunks_for(["3.3", "2.7"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=empty_refs,
        retrieved_chunks=chunks,
    )
    evidence_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_EVIDENCE_REFS
    )
    assert evidence_finding.outcome == CriterionOutcome.PASS


# ---------------------------------------------------------------------------
# Test 4: Non-empty evidence-reference success
# ---------------------------------------------------------------------------

def test_nonempty_evidence_reference_passes_evidence_presence() -> None:
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )
    evidence_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_EVIDENCE_REFS
    )
    assert evidence_finding.outcome == CriterionOutcome.PASS
    assert "non-empty evidence references" in evidence_finding.reason


# ---------------------------------------------------------------------------
# Test 5: Explicit critic-backed PASS findings
# ---------------------------------------------------------------------------

def test_critic_backed_families_emit_explicit_pass_findings_when_no_issues() -> None:
    """When the injected critic reports no issues, all four critic-backed
    criterion families must still appear in the EvaluationResult as explicit
    PASS findings (Phase-C correction to _normalize_critic_findings).
    """
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator(critic=_always_pass_critic)
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )
    findings_by_id = {f.criterion_id: f for f in result.criteria_findings}
    for family_id in (
        CRITERION_LORE_BOUNDARIES,
        CRITERION_AETH_NUL,
        CRITERION_SILENCE,
        CRITERION_TONE,
    ):
        assert family_id in findings_by_id, (
            f"family {family_id!r} missing from EvaluationResult findings — "
            f"explicit PASS finding should be emitted per Phase C correction"
        )
        assert findings_by_id[family_id].outcome == CriterionOutcome.PASS, (
            f"family {family_id!r} expected PASS (no critic issues), "
            f"got {findings_by_id[family_id].outcome!r}"
        )


def test_critic_lore_break_failure_propagates_to_lore_boundaries_finding() -> None:
    """When the injected critic reports a lore_break issue targeting Sentinel,
    the LORE_BOUNDARIES criterion must FAIL (not disappear, not silently PASS).
    """
    lore_break_critic = lambda _t, _o, _c: {
        "issues_found": [
            {
                "type": "lore_break",
                "path": "config.yaml.critic.sentinel_boundaries",
                "description": "Sentinel boundary violation detected by critic.",
                "severity": "critical",
            }
        ],
        "corrected_output": _o,
        "correction_applied": True,
    }
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator(critic=lore_break_critic)
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )
    findings_by_id = {f.criterion_id: f for f in result.criteria_findings}
    assert CRITERION_LORE_BOUNDARIES in findings_by_id
    assert findings_by_id[CRITERION_LORE_BOUNDARIES].outcome == CriterionOutcome.FAIL


# ---------------------------------------------------------------------------
# Test 6: EvaluationResult contract / schema compatibility
# ---------------------------------------------------------------------------

def _load_ger_evaluation_schema() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "schemas" / "ger_evaluation.json"
    with schema_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_evaluation_result_serializes_to_dict_compatible_with_ger_evaluation_schema() -> None:
    """The EvaluationResult returned by the evaluator must serialize to a dict
    that conforms to schemas/ger_evaluation.json (no schema modification)."""
    try:
        import jsonschema  # noqa: F401
        HAS_JSONSCHEMA = True
    except ImportError:
        HAS_JSONSCHEMA = False

    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )

    serialized = {
        "criteria_findings": [
            {
                "criterion_id": f.criterion_id,
                "criterion_name": f.criterion_name,
                "outcome": f.outcome.value,
                "reason": f.reason,
                "evidence_refs": list(f.evidence_refs),
                "severity": f.severity,
            }
            for f in result.criteria_findings
        ],
        "evaluator_warnings": list(result.evaluator_warnings),
        "evidence_refs": list(result.evidence_refs),
    }

    # Required top-level keys
    schema = _load_ger_evaluation_schema()
    required = set(schema["required"])
    assert required.issubset(set(serialized.keys())), (
        f"missing required top-level keys: {required - set(serialized.keys())}"
    )

    # Required per-finding keys
    finding_req = set(schema["properties"]["criteria_findings"]["items"]["required"])
    for finding in serialized["criteria_findings"]:
        assert finding_req.issubset(set(finding.keys())), (
            f"finding missing required keys: {finding_req - set(finding.keys())}"
        )
        assert finding["outcome"] in ("PASS", "FAIL")
        assert finding["severity"] in ("critical", "major", "minor")
        assert isinstance(finding["evidence_refs"], list)
        assert all(isinstance(e, str) for e in finding["evidence_refs"])

    # additionalProperties: false at top level
    assert schema.get("additionalProperties") is False
    top_props = set(schema["properties"].keys())
    assert set(serialized.keys()) == top_props, (
        f"unexpected top-level keys: {set(serialized.keys()) - top_props}"
    )

    # If jsonschema is available, perform a full schema validation
    if HAS_JSONSCHEMA:
        import jsonschema
        jsonschema.validate(serialized, schema)
    # The schema validation procedurally asserts structure whether or not the
    # optional jsonschema package is available.


# ---------------------------------------------------------------------------
# Test 7: sentinel_non_authority single-finding return contract
# ---------------------------------------------------------------------------

def test_sentinel_non_authority_emits_exactly_one_finding_when_pass() -> None:
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )
    sentinel_findings = [
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_SENTINEL_NON_AUTHORITY
    ]
    assert len(sentinel_findings) == 1, (
        f"expected exactly 1 sentinel_non_authority finding, got {len(sentinel_findings)}"
    )
    assert sentinel_findings[0].outcome == CriterionOutcome.PASS
    assert sentinel_findings[0].severity == "critical"
    assert sentinel_findings[0].criterion_name == "Sentinel Non-Authority (GDD §3.3)"


def test_sentinel_non_authority_emits_exactly_one_finding_when_fail() -> None:
    bad = _load_fixture(BAD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(bad["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=bad,
        retrieved_chunks=chunks,
    )
    sentinel_findings = [
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_SENTINEL_NON_AUTHORITY
    ]
    assert len(sentinel_findings) == 1, (
        f"expected exactly 1 sentinel_non_authority finding, got {len(sentinel_findings)}"
    )
    assert sentinel_findings[0].outcome == CriterionOutcome.FAIL
    assert sentinel_findings[0].severity == "critical"


# ---------------------------------------------------------------------------
# Test 8: Controlled-fixture groundedness — sanity asserts that fixture
# artifacts conform to the Project Sentinel GDD-grounded shape (no fixtures
# were injected into production files; fixtures are tests-only).
# ---------------------------------------------------------------------------

def test_controlled_fixture_good_conforms_to_sentinel_evaluation_shape() -> None:
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    assert good["artifact_type"] == "sentinel_evaluation"
    required_fields = {
        "artifact_type", "game_need", "retrieval_query",
        "retrieved_context_ids", "success_line", "failure_line",
        "retry_guidance", "future_warning",
    }
    assert required_fields.issubset(set(good.keys())), (
        f"good fixture missing required fields: {required_fields - set(good.keys())}"
    )
    assert len(good["retrieved_context_ids"]) > 0


def test_controlled_fixture_bad_conforms_to_sentinel_evaluation_shape_with_authorial_language() -> None:
    bad = _load_fixture(BAD_SENTINEL_EVALUATION_JSON)
    assert bad["artifact_type"] == "sentinel_evaluation"
    required_fields = {
        "artifact_type", "game_need", "retrieval_query",
        "retrieved_context_ids", "success_line", "failure_line",
        "retry_guidance", "future_warning",
    }
    assert required_fields.issubset(set(bad.keys())), (
        f"bad fixture missing required fields: {required_fields - set(bad.keys())}"
    )
    # Sanity: at least one forbidden authorial phrase must be present
    artifact_text = json.dumps(bad).lower()
    forbidden_substrings = (
        "sentinel evaluates",
        "sentinel determines",
        "relay, deciding",
        "relay will interpret",
        "sentinel: recommending",
        "the correct relationship",
        "the answer is",
    )
    found = [s for s in forbidden_substrings if s in artifact_text]
    assert len(found) >= 3, (
        f"bad fixture should contain >= 3 forbidden authorial substrings; "
        f"found {found!r}"
    )


def test_controlled_fixture_empty_refs_has_empty_retrieved_context_ids() -> None:
    empty = _load_fixture(EMPTY_REFS_SENTINEL_EVALUATION_JSON)
    assert empty["retrieved_context_ids"] == []


# ---------------------------------------------------------------------------
# Test: schema-validity FAIL when validator returns False
# ---------------------------------------------------------------------------

def test_schema_validity_fails_when_injected_validator_returns_false() -> None:
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator(validator=_always_fail_validator)
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )
    schema_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_SCHEMA
    )
    assert schema_finding.outcome == CriterionOutcome.FAIL
    assert schema_finding.severity == "critical"


# ---------------------------------------------------------------------------
# Test: structural completeness failure when required fields missing
# ---------------------------------------------------------------------------

def test_structural_completeness_fails_when_required_field_missing() -> None:
    good = _load_fixture(GOOD_SENTINEL_EVALUATION_JSON)
    del good["future_warning"]
    chunks = _retrieved_chunks_for(good["retrieved_context_ids"])
    evaluator = _make_evaluator()
    result = evaluator.evaluate(
        artifact_type="sentinel_evaluation",
        output=good,
        retrieved_chunks=chunks,
    )
    structural_finding = next(
        f for f in result.criteria_findings
        if f.criterion_id == CRITERION_STRUCTURAL
    )
    assert structural_finding.outcome == CriterionOutcome.FAIL
    assert "Missing required fields" in structural_finding.reason
    assert "future_warning" in structural_finding.reason
