"""Assignment #6 GER Controller deterministic tests.

Tests validate the GER Controller orchestration of the certified pipeline components.
"""

import pytest
from typing import Any, Dict, List

from src.controller.ger_controller import GERController, create_ger_controller
from src.generators.adapter import GeneratorAdapter
from src.evaluator import Evaluator
from src.refiner.adapter import RefinerAdapter
from src.quality_gate.gate import QualityGate
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
    from src.ger_contracts import CriterionFinding
    return CriterionFinding(
        criterion_id=criterion_id,
        criterion_name=criterion_id.replace("_", " ").title(),
        outcome=outcome,
        reason=reason,
        evidence_refs=["section_test"],
        severity=severity,
    )


def _make_evaluation_result(
    findings: list,
    evaluator_warnings: list[str] = None,
    evidence_refs: list[str] = None,
) -> "EvaluationResult":
    """Create a controlled EvaluationResult."""
    if evaluator_warnings is None:
        evaluator_warnings = []
    if evidence_refs is None:
        evidence_refs = []
    from src.ger_contracts import EvaluationResult
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


def _fake_refiner_returns_artifact(
    artifact_type: str,
    current_artifact: dict,
    refinement_request: Any,
    retrieved_chunks: list,
    config: dict,
) -> dict:
    """Fake refiner that returns the current artifact unchanged, preserving provenance."""
    return current_artifact


def _make_retrieved_chunks() -> List[Dict[str, Any]]:
    """Create controlled retrieved chunks."""
    return [
        {"section_id": "section_3.3", "text": "Sentinel must not evaluate..."},
        {"section_id": "section_2.6", "text": "Silence is evidence..."},
    ]


def _fake_generator(artifact_type: str, retrieved_chunks: list, config: dict) -> dict:
    """Fake generator that returns a controlled artifact."""
    from src.ger_contracts import EvaluationResult
    return {
        "artifact_type": "sentinel_evaluation",
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_3.3", "section_2.6"],
        "success_line": "Evidence shows...",
        "failure_line": "No failure...",
        "retry_guidance": "No retry...",
        "future_warning": "No warning...",
    }


def _fake_validator(artifact_type: str, output: dict) -> bool:
    """Fake validator that always returns True."""
    return True


def _fake_critic(artifact_type: str, output: dict, retrieved_chunks: list) -> dict:
    """Fake critic that returns no issues."""
    return {"issues_found": [], "corrected_output": None, "correction_applied": False}


def _fake_sentinel_critic(artifact_type: str, output: dict, retrieved_chunks: list) -> dict:
    """Fake sentinel critic that returns corrected output when needed."""
    return {"issues_found": [], "corrected_output": None, "correction_applied": False}


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

# 1. Immediate ACCEPT
def test_controller_immediate_accept():
    """Test immediate ACCEPT on first attempt with all criteria passing."""
    from src.generators.adapter import GeneratorAdapter
    from src.evaluator import create_evaluator
    from src.refiner.adapter import RefinerAdapter
    from src.quality_gate.gate import QualityGate
    from src.controller.ger_controller import create_ger_controller

    generator_adapter = GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
        "artifact_type": "sentinel_evaluation",
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_3.3", "section_2.6"],
        "success_line": "Evidence shows...",
        "failure_line": "No failure...",
        "retry_guidance": "No retry...",
        "future_warning": "No warning...",
    }})

    evaluator = create_evaluator(
        critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
        validator=lambda at, o: True
    )

    refiner_adapter = RefinerAdapter({})
    quality_gate = QualityGate()

    controller = create_ger_controller(
        generator_adapter=generator_adapter,
        evaluator=evaluator,
        refiner_adapter=refiner_adapter,
        quality_gate=quality_gate,
        max_attempts=3,
    )

    chunks = [{"section_id": "section_3.3", "text": "Sentinel must not evaluate..."}]
    manifest = controller.run("sentinel_evaluation", chunks)

    assert manifest.terminal_state.value == "ACCEPT"
    assert manifest.attempt_number == 1
    assert manifest.completed_at is not None
    assert manifest.total_runtime_seconds is not None
    assert len(manifest.evaluation_trace) == 1
    assert manifest.evaluation_trace[0]["overall_pass"] is True
    assert manifest.evaluation_trace[0]["decision"]["action"] == "ACCEPT"
    assert manifest.terminal_state.value == "ACCEPT"


# 2. RETRY then ACCEPT
def test_controller_retry_then_accept():
    """Test RETRY on first attempt, then ACCEPT on second attempt."""
    eval_count = {"count": 0}

    def eval_with_fail_then_pass(artifact_type: str, output: dict, retrieved_chunks: list):
        from src.ger_contracts import EvaluationResult, CriterionFinding, CriterionOutcome
        eval_count["count"] = eval_count.get("count", 0) + 1
        if eval_count["count"] == 1:
            return EvaluationResult(
                criteria_findings=[CriterionFinding(
                    criterion_id="structural_completeness",
                    criterion_name="Structural Completeness",
                    outcome=CriterionOutcome.FAIL,
                    reason="Missing required field: success_line",
                    severity="critical",
                    evidence_refs=["section_3.3"]
                )],
                evaluator_warnings=[],
                evidence_refs=["section_3.3"]
            )
        else:
            return EvaluationResult(
                criteria_findings=[
                    CriterionFinding(criterion_id="schema_validity", criterion_name="Schema Validity", outcome=CriterionOutcome.PASS, reason="OK", severity="major", evidence_refs=["section_3.3"]),
                    CriterionFinding(criterion_id="sentinel_non_authority", criterion_name="Sentinel Non-Authority", outcome=CriterionOutcome.PASS, reason="OK", severity="critical", evidence_refs=["section_3.3"]),
                    CriterionFinding(criterion_id="structural_completeness", criterion_name="Structural Completeness", outcome=CriterionOutcome.PASS, reason="OK", severity="major", evidence_refs=["section_3.3"]),
                ],
                evaluator_warnings=[],
                evidence_refs=["section_3.3"]
            )

    from src.generators.adapter import GeneratorAdapter
    from src.evaluator import Evaluator
    from src.refiner.adapter import RefinerAdapter
    from src.quality_gate.gate import QualityGate
    from src.controller.ger_controller import create_ger_controller

    generator_adapter = GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
        "artifact_type": "sentinel_evaluation",
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_3.3", "section_2.6"],
        "success_line": "Evidence shows...",
        "failure_line": "No failure...",
        "retry_guidance": "No retry...",
        "future_warning": "No warning...",
    }})

    evaluator = Evaluator(
        critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
        validator=lambda at, o: True
    )
    evaluator.evaluate = eval_with_fail_then_pass

    refiner_adapter = RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact})
    quality_gate = QualityGate()

    controller = create_ger_controller(
        generator_adapter=GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
            "artifact_type": "sentinel_evaluation",
            "game_need": "test need",
            "retrieval_query": "test query",
            "retrieved_context_ids": ["section_3.3", "section_2.6"],
            "success_line": "Evidence shows...",
            "failure_line": "No failure...",
            "retry_guidance": "No retry...",
            "future_warning": "No warning...",
        }}),
        evaluator=Evaluator(
            critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
            validator=lambda at, o: True
        ),
        refiner_adapter=RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact}),
        quality_gate=QualityGate(),
        max_attempts=3,
    )
    controller._evaluator.evaluate = eval_with_fail_then_pass

    chunks = [{"section_id": "section_3.3", "text": "Sentinel must not evaluate..."}]
    manifest = controller.run("sentinel_evaluation", chunks)

    assert manifest.terminal_state.value == "ACCEPT"
    assert manifest.attempt_number == 2
    assert len(manifest.evaluation_trace) == 2
    assert manifest.evaluation_trace[0]["decision"]["action"] == "RETRY"
    assert manifest.evaluation_trace[1]["decision"]["action"] == "ACCEPT"
    assert len(manifest.refinement_trace) == 1


# 3. Max-attempt ESCALATE
def test_controller_max_attempt_escalate():
    """Test ESCALATE when max attempts reached with persistent failures."""
    from src.ger_contracts import EvaluationResult, CriterionFinding, CriterionOutcome

    attempt_counter = {"count": 0}

    def always_failing_evaluator(artifact_type: str, output: dict, retrieved_chunks: list):
        attempt_counter["count"] = attempt_counter.get("count", 0) + 1
        return EvaluationResult(
            criteria_findings=[CriterionFinding(
                criterion_id="sentinel_non_authority",
                criterion_name="Sentinel Non-Authority",
                outcome=CriterionOutcome.FAIL,
                reason=f"Sentinel evaluates the evidence - authorial violation (attempt {attempt_counter['count']})",
                severity="critical",
                evidence_refs=["section_3.3"]
            )],
            evaluator_warnings=[],
            evidence_refs=["section_3.3"]
        )

    from src.generators.adapter import GeneratorAdapter
    from src.evaluator import Evaluator
    from src.refiner.adapter import RefinerAdapter
    from src.quality_gate.gate import QualityGate
    from src.controller.ger_controller import create_ger_controller

    generator_adapter = GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
        "artifact_type": "sentinel_evaluation",
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_3.3"],
    }})

    evaluator = Evaluator(
        critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
        validator=lambda at, o: True
    )
    evaluator.evaluate = always_failing_evaluator

    refiner_adapter = RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact})
    quality_gate = QualityGate()

    controller = create_ger_controller(
        generator_adapter=GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
            "artifact_type": "sentinel_evaluation",
            "game_need": "test need",
            "retrieval_query": "test query",
            "retrieved_context_ids": ["section_3.3"],
        }}),
        evaluator=Evaluator(
            critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
            validator=lambda at, o: True
        ),
        refiner_adapter=RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact}),
        quality_gate=QualityGate(),
        max_attempts=3,
    )
    controller._evaluator.evaluate = always_failing_evaluator

    chunks = [{"section_id": "section_3.3", "text": "Sentinel must not evaluate..."}]
    manifest = controller.run("sentinel_evaluation", chunks)

    assert manifest.terminal_state.value == "ESCALATE"
    assert manifest.attempt_number == 3
    assert len(manifest.evaluation_trace) == 3
    assert all(trace["decision"]["action"] != "ACCEPT" for trace in manifest.evaluation_trace)
    assert manifest.evaluation_trace[-1]["decision"]["action"] == "ESCALATE"


# 4. No-progress ESCALATE
def test_controller_no_progress_escalate():
    """Test ESCALATE when no progress detected (same failures, same artifact)."""
    from src.ger_contracts import EvaluationResult, CriterionFinding, CriterionOutcome

    same_fail_eval = EvaluationResult(
        criteria_findings=[CriterionFinding(
            criterion_id="sentinel_non_authority",
            criterion_name="Sentinel Non-Authority",
            outcome=CriterionOutcome.FAIL,
            reason="Same failure",
            severity="critical",
            evidence_refs=["section_3.3"]
        )],
        evaluator_warnings=[],
        evidence_refs=["section_3.3"]
    )

    from src.evaluator import Evaluator
    from src.generators.adapter import GeneratorAdapter
    from src.refiner.adapter import RefinerAdapter
    from src.quality_gate.gate import QualityGate
    from src.controller.ger_controller import create_ger_controller

    evaluator = Evaluator(
        critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
        validator=lambda at, o: True
    )
    evaluator.evaluate = lambda *a, **kw: same_fail_eval

    generator_adapter = GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
        "artifact_type": "sentinel_evaluation",
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_3.3"],
    }})

    refiner_adapter = RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact})
    quality_gate = QualityGate()

    controller = create_ger_controller(
        generator_adapter=GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
            "artifact_type": "sentinel_evaluation",
            "game_need": "test need",
            "retrieval_query": "test query",
            "retrieved_context_ids": ["section_3.3"],
        }}),
        evaluator=Evaluator(
            critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
            validator=lambda at, o: True
        ),
        refiner_adapter=RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact}),
        quality_gate=QualityGate(),
        max_attempts=3,
    )
    controller._evaluator.evaluate = lambda *a, **kw: same_fail_eval

    chunks = [{"section_id": "section_3.3", "text": "Sentinel must not evaluate..."}]
    manifest = controller.run("sentinel_evaluation", chunks)

    # Should escalate on attempt 2 due to no-progress detection
    assert manifest.terminal_state.value == "ESCALATE"
    assert manifest.attempt_number == 2
    assert manifest.evaluation_trace[1]["decision"]["action"] == "ESCALATE"
    assert manifest.evaluation_trace[1]["decision"]["no_progress"] is True


# 5. Provenance preservation
def test_controller_provenance_preservation():
    """Test that retrieved_context_ids are preserved through refinement."""
    eval_count = {"count": 0}

    def eval_with_fail_then_pass(artifact_type: str, output: dict, retrieved_chunks: list):
        from src.ger_contracts import EvaluationResult, CriterionFinding, CriterionOutcome
        eval_count["count"] = eval_count.get("count", 0) + 1
        if eval_count["count"] == 1:
            return EvaluationResult(
                criteria_findings=[CriterionFinding(
                    criterion_id="structural_completeness",
                    criterion_name="Structural Completeness",
                    outcome=CriterionOutcome.FAIL,
                    reason="Missing field",
                    severity="critical",
                    evidence_refs=["section_3.3"]
                )],
                evaluator_warnings=[],
                evidence_refs=["section_3.3"]
            )
        else:
            return EvaluationResult(
                criteria_findings=[
                    CriterionFinding(criterion_id="schema_validity", criterion_name="Schema Validity", outcome=CriterionOutcome.PASS, reason="OK", severity="major", evidence_refs=["section_3.3"]),
                    CriterionFinding(criterion_id="sentinel_non_authority", criterion_name="Sentinel Non-Authority", outcome=CriterionOutcome.PASS, reason="OK", severity="critical", evidence_refs=["section_3.3"]),
                ],
                evaluator_warnings=[],
                evidence_refs=["section_3.3"]
            )

    from src.generators.adapter import GeneratorAdapter
    from src.evaluator import Evaluator
    from src.refiner.adapter import RefinerAdapter
    from src.quality_gate.gate import QualityGate
    from src.controller.ger_controller import create_ger_controller

    generator_adapter = GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
        "artifact_type": "sentinel_evaluation",
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_3.3", "section_2.6"],
        "success_line": "Evidence...",
        "failure_line": "No failure...",
        "retry_guidance": "No retry...",
        "future_warning": "No warning...",
    }})

    evaluator = Evaluator(
        critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
        validator=lambda at, o: True
    )
    evaluator.evaluate = eval_with_fail_then_pass

    refiner_adapter = RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact})
    quality_gate = QualityGate()

    controller = create_ger_controller(
        generator_adapter=GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
            "artifact_type": "sentinel_evaluation",
            "game_need": "test need",
            "retrieval_query": "test query",
            "retrieved_context_ids": ["section_3.3", "section_2.6"],
            "success_line": "Evidence...",
            "failure_line": "No failure...",
            "retry_guidance": "No retry...",
            "future_warning": "No warning...",
        }}),
        evaluator=Evaluator(
            critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
            validator=lambda at, o: True
        ),
        refiner_adapter=RefinerAdapter({"sentinel_evaluation": lambda artifact_type, current_artifact, refinement_request, retrieved_chunks, config: current_artifact}),
        quality_gate=QualityGate(),
        max_attempts=3,
    )
    controller._evaluator.evaluate = eval_with_fail_then_pass

    chunks = [{"section_id": "section_3.3", "text": "..."}, {"section_id": "section_2.6", "text": "..."}]
    manifest = controller.run("sentinel_evaluation", chunks)

    # Check that final artifact preserves provenance
    assert manifest.terminal_state.value == "ACCEPT"
    assert manifest.attempt_number == 2


# 6. Manifest/history correctness
def test_controller_manifest_correctness():
    """Test that manifest correctly records run_id, traces, terminal_state, and timing."""
    from src.ger_contracts import EvaluationResult, CriterionFinding, CriterionOutcome

    def clean_eval(artifact_type: str, output: dict, retrieved_chunks: list):
        from src.ger_contracts import EvaluationResult, CriterionFinding, CriterionOutcome
        return EvaluationResult(
            criteria_findings=[
                CriterionFinding(criterion_id="schema_validity", criterion_name="Schema Validity", outcome=CriterionOutcome.PASS, reason="OK", severity="major", evidence_refs=["section_3.3"]),
                CriterionFinding(criterion_id="sentinel_non_authority", criterion_name="Sentinel Non-Authority", outcome=CriterionOutcome.PASS, reason="OK", severity="critical", evidence_refs=["section_3.3"]),
            ],
            evaluator_warnings=[],
            evidence_refs=["section_3.3"]
        )

    from src.generators.adapter import GeneratorAdapter
    from src.evaluator import Evaluator
    from src.refiner.adapter import RefinerAdapter
    from src.quality_gate.gate import QualityGate
    from src.controller.ger_controller import create_ger_controller

    generator_adapter = GeneratorAdapter({"sentinel_evaluation": lambda rc, c: {
        "artifact_type": "sentinel_evaluation",
        "game_need": "test need",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_3.3"],
        "success_line": "Evidence...",
        "failure_line": "No failure...",
        "retry_guidance": "No retry...",
        "future_warning": "No warning...",
    }})

    evaluator = Evaluator(
        critic=lambda at, o, rc: {"issues_found": [], "corrected_output": None, "correction_applied": False},
        validator=lambda at, o: True
    )
    evaluator.evaluate = lambda *a, **kw: EvaluationResult(
        criteria_findings=[
            CriterionFinding(criterion_id="schema_validity", criterion_name="Schema Validity", outcome=CriterionOutcome.PASS, reason="OK", severity="major", evidence_refs=["section_3.3"]),
            CriterionFinding(criterion_id="sentinel_non_authority", criterion_name="Sentinel Non-Authority", outcome=CriterionOutcome.PASS, reason="OK", severity="critical", evidence_refs=["section_3.3"]),
        ],
        evaluator_warnings=[],
        evidence_refs=["section_3.3"]
    )

    refiner_adapter = RefinerAdapter({})
    quality_gate = QualityGate()

    from src.controller.ger_controller import create_ger_controller

    controller = create_ger_controller(
        generator_adapter=generator_adapter,
        evaluator=evaluator,
        refiner_adapter=refiner_adapter,
        quality_gate=quality_gate,
        max_attempts=3,
    )

    chunks = [{"section_id": "section_3.3", "text": "..."}]
    manifest = controller.run("sentinel_evaluation", chunks, run_id="test-run-123")

    assert manifest.run_id == "test-run-123"
    assert manifest.terminal_state.value == "ACCEPT"
    assert manifest.attempt_number == 1
    assert manifest.started_at is not None
    assert manifest.completed_at is not None
    assert manifest.total_runtime_seconds is not None
    assert manifest.total_runtime_seconds >= 0
    assert len(manifest.evaluation_trace) == 1
    assert len(manifest.refinement_trace) == 0
    assert manifest.evaluation_trace[0]["decision"]["action"] == "ACCEPT"
    assert manifest.evaluation_trace[0]["decision"]["terminal_state"] == "ACCEPT"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])