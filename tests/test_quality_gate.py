"""Assignment #6 Phase 4 — Quality Gate deterministic tests.

Tests validate the CURRENT QualityGate implementation.
"""

import pytest

from src.quality_gate.gate import QualityGate, QualityGateDecision
from src.ger_contracts import (
    CriterionFinding,
    CriterionOutcome,
    EvaluationResult,
    TerminalState,
)


# ---------------------------------------------------------------------------
# Controlled fixtures and helpers
# ---------------------------------------------------------------------------


class _FakeException(Exception):
    """Controlled exception for testing."""
    pass


def _make_criterion_finding(
    criterion_id: str,
    outcome: CriterionOutcome,
    severity: str = "major",
    reason: str = "test reason",
) -> CriterionFinding:
    """Create a controlled CriterionFinding."""
    return CriterionFinding(
        criterion_id=criterion_id,
        criterion_name=criterion_id.replace("_", " ").title(),
        outcome=outcome,
        reason=reason,
        evidence_refs=["section_test"],
        severity=severity,
    )


def _make_evaluation_result(
    findings: list[CriterionFinding],
    evaluator_warnings: list[str] = None,
    evidence_refs: list[str] = None,
) -> EvaluationResult:
    """Create a controlled EvaluationResult."""
    if evaluator_warnings is None:
        evaluator_warnings = []
    if evidence_refs is None:
        evidence_refs = []
    return EvaluationResult(
        criteria_findings=findings,
        evaluator_warnings=evaluator_warnings,
        evidence_refs=evidence_refs,
    )


def _make_artifact(artifact_type: str = "sentinel_evaluation", retrieved_context_ids=None) -> dict:
    """Create a minimal controlled artifact dict."""
    if retrieved_context_ids is None:
        retrieved_context_ids = ["section_test"]
    return {
        "artifact_type": artifact_type,
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": retrieved_context_ids,
    }


# ---------------------------------------------------------------------------
# 1. ALL PASS → ACCEPT
# ---------------------------------------------------------------------------

def test_all_pass_accepts() -> None:
    findings = [
        _make_criterion_finding("schema_validity", CriterionOutcome.PASS),
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.PASS),
    ]
    result = _make_evaluation_result(findings)

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=1,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
    )

    assert decision.action == "ACCEPT"
    assert decision.terminal_state == TerminalState.ACCEPT
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 2. ALL PASS + WARNINGS → ACCEPT
# ---------------------------------------------------------------------------

def test_all_pass_with_warnings_accepts() -> None:
    findings = [
        _make_criterion_finding("schema_validity", CriterionOutcome.PASS),
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.PASS),
    ]
    result = _make_evaluation_result(
        findings=findings,
        evaluator_warnings=["Schema validation warning"],
    )

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=1,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
    )

    assert decision.action == "ACCEPT"
    assert decision.terminal_state == TerminalState.ACCEPT
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 3. FAIL + RETRY BUDGET AVAILABLE → RETRY
# ---------------------------------------------------------------------------

def test_fail_with_retry_budget_returns_retry() -> None:
    findings = [
        _make_criterion_finding("schema_validity", CriterionOutcome.PASS),
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    result = _make_evaluation_result(findings)

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=1,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 4. FAIL AT MAX ATTEMPTS → ESCALATE
# ---------------------------------------------------------------------------

def test_fail_at_max_attempts_escalates() -> None:
    findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    result = _make_evaluation_result(findings)

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=3,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
    )

    assert decision.action == "ESCALATE"
    assert decision.terminal_state == TerminalState.ESCALATE
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 5. ATTEMPT ABOVE MAX → ESCALATE
# ---------------------------------------------------------------------------

def test_attempt_above_max_escalates() -> None:
    findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL),
    ]
    result = _make_evaluation_result(findings)

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=4,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
    )

    assert decision.action == "ESCALATE"
    assert decision.terminal_state == TerminalState.ESCALATE


# ---------------------------------------------------------------------------
# 6. CASE A — UNCHANGED ARTIFACT + UNCHANGED FAILURES → ESCALATE
# ---------------------------------------------------------------------------

def test_case_a_unchanged_artifact_unchanged_failures_escalates() -> None:
    findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    result = _make_evaluation_result(findings)

    artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3", "section_2.7"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=artifact,
        previous_evaluation=result,
        previous_artifact=artifact,
    )

    assert decision.action == "ESCALATE"
    assert decision.terminal_state == TerminalState.ESCALATE
    assert decision.no_progress is True


# ---------------------------------------------------------------------------
# 7. CASE B — CHANGED ARTIFACT + SAME FAILURES → RETRY
# ---------------------------------------------------------------------------

def test_case_b_changed_artifact_same_failures_retries() -> None:
    findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    result = _make_evaluation_result(findings)

    prev_artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]}
    curr_artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3", "section_2.7"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=curr_artifact,
        previous_evaluation=result,
        previous_artifact=prev_artifact,
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 8. CASE C — CHANGED ARTIFACT + FEWER FAILURES → RETRY
# ---------------------------------------------------------------------------

def test_case_c_changed_artifact_fewer_failures_retries() -> None:
    prev_findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
        _make_criterion_finding("schema_validity", CriterionOutcome.FAIL, reason="schema"),
    ]
    prev_result = _make_evaluation_result(prev_findings)

    curr_findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    curr_result = _make_evaluation_result(curr_findings)

    prev_artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]}
    curr_artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3", "section_2.7"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=curr_result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=curr_artifact,
        previous_evaluation=prev_result,
        previous_artifact=prev_artifact,
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 9. CASE D — CHANGED ARTIFACT + DIFFERENT FAILURES (same count) → RETRY
# ---------------------------------------------------------------------------

def test_case_d_changed_artifact_different_failures_retries() -> None:
    prev_findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    prev_result = _make_evaluation_result(prev_findings)

    curr_findings = [
        _make_criterion_finding("schema_validity", CriterionOutcome.FAIL, reason="schema violation"),
    ]
    curr_result = _make_evaluation_result(curr_findings)

    prev_artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]}
    curr_artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3", "section_2.7"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=curr_result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=curr_artifact,
        previous_evaluation=prev_result,
        previous_artifact=prev_artifact,
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 10. CASE E — UNCHANGED ARTIFACT + CHANGED FAILURES → RETRY
# ---------------------------------------------------------------------------

def test_case_e_unchanged_artifact_changed_failures_retries() -> None:
    prev_findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    prev_result = _make_evaluation_result(prev_findings)

    curr_findings = [
        _make_criterion_finding("schema_validity", CriterionOutcome.FAIL, reason="schema"),
    ]
    curr_result = _make_evaluation_result(curr_findings)

    artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3", "section_2.7"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=curr_result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=artifact,
        previous_evaluation=prev_result,
        previous_artifact=artifact,
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 11. FINDING ORDER INDEPENDENCE
# ---------------------------------------------------------------------------

def test_failed_findings_order_independence() -> None:
    finding_a = _make_criterion_finding("a_sentinel_non_authority", CriterionOutcome.FAIL, reason="first")
    finding_b = _make_criterion_finding("b_schema_validity", CriterionOutcome.FAIL, reason="second")

    prev_result = _make_evaluation_result([finding_a, finding_b])
    curr_result = _make_evaluation_result([finding_b, finding_a])

    artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=_make_evaluation_result([finding_b, finding_a]),
        attempt_number=2,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]},
        previous_evaluation=prev_result,
        previous_artifact={"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]},
    )

    # Order-independent comparison should detect no-progress
    assert decision.action == "ESCALATE"
    assert decision.no_progress is True


# ---------------------------------------------------------------------------
# 12. PASS FINDINGS DO NOT AFFECT FAILURE SIGNATURE
# ---------------------------------------------------------------------------

def test_pass_findings_ignored_for_failure_signature() -> None:
    fail_finding = _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)
    pass_finding_1 = _make_criterion_finding("schema_validity", CriterionOutcome.PASS, reason="old")
    pass_finding_2 = _make_criterion_finding("tone_consistency", CriterionOutcome.PASS, reason="new")

    prev_result = _make_evaluation_result([fail_finding, pass_finding_1])
    curr_result = _make_evaluation_result([fail_finding, pass_finding_2])

    artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=curr_result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=artifact,
        previous_evaluation=prev_result,
        previous_artifact=artifact,
    )

    # PASS findings differ but FAIL findings are identical → no-progress
    assert decision.action == "ESCALATE"
    assert decision.no_progress is True


# ---------------------------------------------------------------------------
# 13. FAILURE REASON CHANGE COUNTS AS DIFFERENT
# ---------------------------------------------------------------------------

def test_failure_reason_change_not_identical() -> None:
    prev_finding = _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="old reason")
    curr_finding = _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="new reason")

    prev_result = _make_evaluation_result([prev_finding])
    curr_result = _make_evaluation_result([curr_finding])

    artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=curr_result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=artifact,
        previous_evaluation=prev_result,
        previous_artifact=artifact,
    )

    # Reason changed → not identical → RETRY
    assert decision.action == "RETRY"
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 14. FAILURE SEVERITY CHANGE COUNTS AS DIFFERENT
# ---------------------------------------------------------------------------

def test_failure_severity_change_not_identical() -> None:
    prev_finding = _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, severity="critical")
    curr_finding = _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, severity="major")

    prev_result = _make_evaluation_result([prev_finding])
    curr_result = _make_evaluation_result([curr_finding])

    artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]}

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=curr_result,
        attempt_number=2,
        max_attempts=3,
        current_artifact=artifact,
        previous_evaluation=prev_result,
        previous_artifact=artifact,
    )

    assert decision.action == "RETRY"
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 15. NO PREVIOUS EVALUATION
# ---------------------------------------------------------------------------

def test_no_previous_evaluation_returns_retry() -> None:
    findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    result = _make_evaluation_result(findings)

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=2,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
        previous_evaluation=None,
        previous_artifact=None,
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 16. NO PREVIOUS ARTIFACT
# ---------------------------------------------------------------------------

def test_no_previous_artifact_returns_retry() -> None:
    findings = [
        _make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL, reason="violation"),
    ]
    result = _make_evaluation_result(findings)

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=2,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
        previous_evaluation=result,
        previous_artifact=None,
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None
    assert decision.no_progress is False


# ---------------------------------------------------------------------------
# 17. INVALID ATTEMPT NUMBER → ValueError
# ---------------------------------------------------------------------------

def test_invalid_attempt_number_raises() -> None:
    gate = QualityGate()

    with pytest.raises(ValueError) as exc_info:
        gate.decide(
            evaluation_result=_make_evaluation_result([]),
            attempt_number=0,
            max_attempts=3,
            current_artifact={},
        )
    assert "attempt_number must be >= 1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 18. INVALID MAX_ATTEMPTS → ValueError
# ---------------------------------------------------------------------------

def test_invalid_max_attempts_raises() -> None:
    gate = QualityGate()

    with pytest.raises(ValueError) as exc_info:
        gate.decide(
            evaluation_result=_make_evaluation_result([]),
            attempt_number=1,
            max_attempts=0,
            current_artifact={},
        )
    assert "max_attempts must be >= 1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 19. RETRY IS NOT TERMINALSTATE
# ---------------------------------------------------------------------------

def test_retry_is_not_terminal_state() -> None:
    findings = [_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    result = _make_evaluation_result(findings)

    gate = QualityGate()
    decision = gate.decide(
        evaluation_result=result,
        attempt_number=1,
        max_attempts=3,
        current_artifact={"artifact_type": "sentinel_evaluation"},
    )

    assert decision.action == "RETRY"
    assert decision.terminal_state is None


# ---------------------------------------------------------------------------
# 20. REJECT IS NEVER EMITTED
# ---------------------------------------------------------------------------

def test_reject_never_emitted() -> None:
    # Test multiple scenarios that could potentially emit REJECT
    scenarios = [
        # Max attempts exhausted
        {
            "findings": [_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)],
            "attempt": 3,
            "max": 3,
        },
        # No progress detected
        {
            "findings": [_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)],
            "attempt": 2,
            "max": 3,
            "prev_eval": True,
            "prev_artifact": {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_3.3"]},
        },
    ]

    gate = QualityGate()

    for scenario in scenarios:
        findings = scenario["findings"]
        result = _make_evaluation_result(findings)

        kwargs = {
            "evaluation_result": result,
            "attempt_number": scenario["attempt"],
            "max_attempts": scenario["max"],
            "current_artifact": {"artifact_type": "sentinel_evaluation"},
        }

        if "prev_eval" in scenario:
            prev_findings = scenario.get("prev_findings", findings)
            kwargs["previous_evaluation"] = _make_evaluation_result(prev_findings)
            kwargs["previous_artifact"] = scenario.get("prev_artifact", {"artifact_type": "sentinel_evaluation"})

        decision = gate.decide(**kwargs)

        # REJECT should never be emitted
        assert decision.terminal_state != TerminalState.REJECT


# ---------------------------------------------------------------------------
# 21. DETERMINISTIC REPEATABILITY
# ---------------------------------------------------------------------------

def test_deterministic_repeatability() -> None:
    findings = [_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    result = _make_evaluation_result(findings)
    artifact = {"artifact_type": "sentinel_evaluation", "retrieved_context_ids": ["section_test"]}

    gate = QualityGate()

    decision_1 = gate.decide(
        evaluation_result=result,
        attempt_number=1,
        max_attempts=3,
        current_artifact=artifact,
    )

    decision_2 = gate.decide(
        evaluation_result=result,
        attempt_number=1,
        max_attempts=3,
        current_artifact=artifact,
    )

    # Same inputs → identical decisions
    assert decision_1.action == decision_2.action
    assert decision_1.terminal_state == decision_2.terminal_state
    assert decision_1.reason == decision_2.reason
    assert decision_1.no_progress == decision_2.no_progress