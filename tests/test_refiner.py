"""Assignment #6 Phase 3 — Refiner deterministic tests.

Tests validate the CURRENT instance-based RefinerAdapter and sentinel
refinement implementation.
"""

import pytest
from typing import Any

from src.generators.adapter import GeneratorCallable
from src.refiner.adapter import RefinerAdapter, RefinementCallable
from src.refiner.sentinel import create_sentinel_refiner
from src.ger_contracts import CriterionFinding, CriterionOutcome, RefinementRequest


# ---------------------------------------------------------------------------
# Controlled fixtures and fake callables
# ---------------------------------------------------------------------------

class _FakeException(Exception):
    """Controlled exception for propagation tests."""
    pass


def _make_fake_artifact(artifact_type: str) -> dict:
    """Create a minimal controlled artifact dict for the given type."""
    return {
        "artifact_type": artifact_type,
        "game_need": f"test {artifact_type}",
        "retrieval_query": "test query",
        "retrieved_context_ids": ["section_test"],
    }


def _make_refinement_request(failed_criteria=None, actionable_feedback=None, original_artifact_ref="test_ref"):
    """Create a RefinementRequest with controlled findings."""
    if failed_criteria is None:
        failed_criteria = []
    if actionable_feedback is None:
        actionable_feedback = []
    return RefinementRequest(
        failed_criteria=failed_criteria,
        actionable_feedback=actionable_feedback,
        original_artifact_ref=original_artifact_ref,
    )


# ---------------------------------------------------------------------------
# Fake refiner callables
# ---------------------------------------------------------------------------

def _fake_refiner_returns_artifact(
    artifact_type: str,
    current_artifact: dict,
    refinement_request: Any,
    retrieved_chunks: list,
    config: dict,
) -> dict:
    return _make_fake_artifact(artifact_type)


def _fake_refiner_captures_args(captured: dict):
    def _inner(
        artifact_type: str,
        current_artifact: dict,
        refinement_request: Any,
        retrieved_chunks: list,
        config: dict,
    ) -> dict:
        captured["artifact_type"] = artifact_type
        captured["current_artifact"] = current_artifact
        captured["refinement_request"] = refinement_request
        captured["retrieved_chunks"] = retrieved_chunks
        captured["config"] = config
        return _make_fake_artifact("sentinel_evaluation")
    return _inner


def _fake_refiner_mutates_context_ids(
    artifact_type: str,
    current_artifact: dict,
    refinement_request: Any,
    retrieved_chunks: list,
    config: dict,
) -> dict:
    # Mutate the original artifact's retrieved_context_ids IN PLACE
    current_artifact["retrieved_context_ids"] = ["mutated"]
    return _make_fake_artifact("sentinel_evaluation")


def _fake_refiner_changes_context_ids(
    artifact_type: str,
    current_artifact: dict,
    refinement_request: Any,
    retrieved_chunks: list,
    config: dict,
) -> dict:
    result = _make_fake_artifact(artifact_type)
    result["retrieved_context_ids"] = ["changed"]
    return result


def _fake_refiner_raises(
    artifact_type: str,
    current_artifact: dict,
    refinement_request: Any,
    retrieved_chunks: list,
    config: dict,
) -> dict:
    raise _FakeException("controlled refiner failure")


# ---------------------------------------------------------------------------
# Sentinel-specific fake critics
# ---------------------------------------------------------------------------

def _fake_critic_correction_applied(
    artifact_type: str,
    output: dict,
    retrieved_chunks: list,
) -> dict:
    corrected = dict(output)
    corrected["success_line"] = "Corrected by A4 critic"
    return {
        "issues_found": [{"type": "lore_break", "description": "test"}],
        "corrected_output": corrected,
        "correction_applied": True,
    }


def _fake_critic_no_correction(
    artifact_type: str,
    output: dict,
    retrieved_chunks: list,
) -> dict:
    return {
        "issues_found": [{"type": "lore_break", "description": "test"}],
        "corrected_output": output,
        "correction_applied": False,
    }


def _fake_critic_bad_corrected_output(
    artifact_type: str,
    output: dict,
    retrieved_chunks: list,
) -> dict:
    return {
        "issues_found": [{"type": "lore_break", "description": "test"}],
        "corrected_output": "not a dict",
        "correction_applied": True,
    }


def _make_criterion_finding(criterion_id: str, outcome: CriterionOutcome) -> CriterionFinding:
    return CriterionFinding(
        criterion_id=criterion_id,
        criterion_name=criterion_id.replace("_", " ").title(),
        outcome=outcome,
        reason="test reason",
        evidence_refs=["section_test"],
        severity="critical",
    )


# ---------------------------------------------------------------------------
# 1. REFINEMENTREQUEST CANONICAL HANDOFF
# ---------------------------------------------------------------------------

def test_refinement_request_passed_to_callable() -> None:
    captured = {}
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_captures_args(captured),
    })
    artifact = _make_fake_artifact("sentinel_evaluation")
    chunks = [{"section_id": "test"}]
    config = {"test": "config"}
    refinement_request = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    )

    adapter.refine("sentinel_evaluation", artifact, refinement_request, chunks, config)

    assert captured["refinement_request"] is refinement_request


# ---------------------------------------------------------------------------
# 2. REFINED ARTIFACT RETURN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("artifact_type", [
    "sentinel_evaluation",
    "keeper_trace",
    "resonance_alignment_trial",
])
def test_refined_artifact_returned_unchanged(artifact_type: str) -> None:
    adapter = RefinerAdapter({
        artifact_type: _fake_refiner_returns_artifact,
    })
    artifact = _make_fake_artifact(artifact_type)
    refinement_request = _make_refinement_request()

    result = adapter.refine(artifact_type, artifact, _make_refinement_request(), [], {})

    assert result == _make_fake_artifact(artifact_type)
    assert result["artifact_type"] == artifact_type


# ---------------------------------------------------------------------------
# 3. RETRIEVED CHUNKS PASS-THROUGH
# ---------------------------------------------------------------------------

def test_retrieved_chunks_passed_unchanged() -> None:
    captured = {}
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_captures_args(captured),
    })
    chunks = [
        {"section_id": "1", "text": "chunk one"},
        {"section_id": "2", "text": "chunk two"},
    ]
    artifact = _make_fake_artifact("sentinel_evaluation")

    adapter.refine("sentinel_evaluation", artifact, _make_refinement_request(), chunks, {})

    assert captured["retrieved_chunks"] is chunks
    assert captured["retrieved_chunks"] == chunks


# ---------------------------------------------------------------------------
# 4. CONFIG PASS-THROUGH
# ---------------------------------------------------------------------------

def test_config_passed_unchanged() -> None:
    captured = {}
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_captures_args(captured),
    })
    config = {"example": "value", "nested": {"key": 42}}
    artifact = _make_fake_artifact("sentinel_evaluation")

    adapter.refine("sentinel_evaluation", artifact, _make_refinement_request(), [], config)

    assert captured["config"] is config
    assert captured["config"] == config


def test_none_config_becomes_empty_dict() -> None:
    captured = {}
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_captures_args(captured),
    })
    artifact = _make_fake_artifact("sentinel_evaluation")

    adapter.refine("sentinel_evaluation", artifact, _make_refinement_request(), [], None)

    assert captured["config"] == {}


# ---------------------------------------------------------------------------
# 5. UNSUPPORTED ARTIFACT TYPE
# ---------------------------------------------------------------------------

def test_unsupported_artifact_type_raises_valueerror() -> None:
    adapter = RefinerAdapter({})

    with pytest.raises(ValueError) as exc_info:
        adapter.refine("unsupported_type", {}, _make_refinement_request(), [], {})

    assert "unsupported_type" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 6. REFINER EXCEPTION PROPAGATION
# ---------------------------------------------------------------------------

def test_refiner_exception_propagates_unchanged() -> None:
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_raises,
    })

    with pytest.raises(_FakeException) as exc_info:
        adapter.refine("sentinel_evaluation", {}, _make_refinement_request(), [], {})

    assert str(exc_info.value) == "controlled refiner failure"


# ---------------------------------------------------------------------------
# 7. INSTANCE ISOLATION
# ---------------------------------------------------------------------------

def test_instance_isolation() -> None:
    adapter_a = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_returns_artifact,
    })
    adapter_b = RefinerAdapter({
        "keeper_trace": _fake_refiner_returns_artifact,
    })

    result_a = adapter_a.refine("sentinel_evaluation", _make_fake_artifact("sentinel_evaluation"), _make_refinement_request(), [], {})
    assert result_a["artifact_type"] == "sentinel_evaluation"

    with pytest.raises(ValueError):
        adapter_a.refine("keeper_trace", _make_fake_artifact("keeper_trace"), _make_refinement_request(), [], {})

    result_b = adapter_b.refine("keeper_trace", _make_fake_artifact("keeper_trace"), _make_refinement_request(), [], {})
    assert result_b["artifact_type"] == "keeper_trace"

    with pytest.raises(ValueError):
        adapter_b.refine("sentinel_evaluation", _make_fake_artifact("sentinel_evaluation"), _make_refinement_request(), [], {})


# ---------------------------------------------------------------------------
# 8. CONSTRUCTOR COPY ISOLATION
# ---------------------------------------------------------------------------

def test_constructor_copies_mapping() -> None:
    original_mapping = {"sentinel_evaluation": _fake_refiner_returns_artifact}
    adapter = RefinerAdapter(original_mapping)

    # Mutate the original mapping after adapter construction
    original_mapping["keeper_trace"] = _fake_refiner_returns_artifact

    # Adapter should still only have its original callable
    result = adapter.refine("sentinel_evaluation", _make_fake_artifact("sentinel_evaluation"), _make_refinement_request(), [], {})
    assert result["artifact_type"] == "sentinel_evaluation"

    with pytest.raises(ValueError):
        adapter.refine("keeper_trace", _make_fake_artifact("keeper_trace"), _make_refinement_request(), [], {})


# ---------------------------------------------------------------------------
# 9. UNCHANGED RETRIEVED_CONTEXT_IDS PASS
# ---------------------------------------------------------------------------

def _fake_refiner_preserves_context_ids(
    artifact_type: str,
    current_artifact: dict,
    refinement_request: Any,
    retrieved_chunks: list,
    config: dict,
) -> dict:
    # Local fixture for provenance test: return artifact with same retrieved_context_ids
    return {**current_artifact}


def test_provenance_invariant_unchanged_passes() -> None:
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_preserves_context_ids,
    })
    artifact = _make_fake_artifact("sentinel_evaluation")
    artifact["retrieved_context_ids"] = ["section_3.3", "section_2.7"]

    result = adapter.refine("sentinel_evaluation", artifact, _make_refinement_request(), [], {})

    assert result["retrieved_context_ids"] == ["section_3.3", "section_2.7"]


# ---------------------------------------------------------------------------
# 10. CHANGED RETRIEVED_CONTEXT_IDS FAIL
# ---------------------------------------------------------------------------

def test_provenance_invariant_changed_fails() -> None:
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_changes_context_ids,
    })
    artifact = _make_fake_artifact("sentinel_evaluation")
    artifact["retrieved_context_ids"] = ["section_3.3"]

    with pytest.raises(ValueError) as exc_info:
        adapter.refine("sentinel_evaluation", artifact, _make_refinement_request(), [], {})

    assert "provenance invariant" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 11. IN-PLACE MUTATION DEFENSE
# ---------------------------------------------------------------------------

def test_provenance_invariant_defends_against_in_place_mutation() -> None:
    adapter = RefinerAdapter({
        "sentinel_evaluation": _fake_refiner_mutates_context_ids,
    })
    artifact = _make_fake_artifact("sentinel_evaluation")
    artifact["retrieved_context_ids"] = ["section_3.3", "section_2.7"]

    with pytest.raises(ValueError) as exc_info:
        adapter.refine("sentinel_evaluation", artifact, _make_refinement_request(), [], {})

    assert "provenance invariant" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 12. SENTINEL_NON_AUTHORITY FAIL TRIGGERS CRITIC
# ---------------------------------------------------------------------------

def test_sentinel_non_authority_fail_triggers_critic() -> None:
    call_count = {"count": 0}

    def counting_critic(artifact_type, output, retrieved_chunks):
        call_count["count"] += 1
        return _fake_critic_correction_applied(artifact_type, output, retrieved_chunks)

    sentinel_refiner = create_sentinel_refiner(counting_critic)
    adapter = RefinerAdapter({
        "sentinel_evaluation": sentinel_refiner,
    })

    artifact = _make_fake_artifact("sentinel_evaluation")
    refinement_request = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    )
    chunks = [{"section_id": "test"}]

    result = adapter.refine("sentinel_evaluation", artifact, refinement_request, chunks, {})

    assert call_count["count"] == 1
    assert result["success_line"] == "Corrected by A4 critic"


# ---------------------------------------------------------------------------
# 13. NO RELEVANT FAIL DOES NOT TRIGGER CRITIC
# ---------------------------------------------------------------------------

def test_no_relevant_fail_does_not_trigger_critic() -> None:
    call_count = {"count": 0}

    def counting_critic(artifact_type, output, retrieved_chunks):
        call_count["count"] += 1
        return _fake_critic_correction_applied(artifact_type, output, retrieved_chunks)

    sentinel_refiner = create_sentinel_refiner(counting_critic)
    adapter = RefinerAdapter({
        "sentinel_evaluation": sentinel_refiner,
    })

    artifact = _make_fake_artifact("sentinel_evaluation")
    refinement_request = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("structural_completeness", CriterionOutcome.FAIL)]
    )

    result = adapter.refine("sentinel_evaluation", artifact, refinement_request, [], {})

    assert call_count["count"] == 0
    assert result == artifact


# ---------------------------------------------------------------------------
# 14. PASS FINDING DOES NOT TRIGGER CRITIC
# ---------------------------------------------------------------------------

def test_pass_finding_does_not_trigger_critic() -> None:
    call_count = {"count": 0}

    def counting_critic(artifact_type, output, retrieved_chunks):
        call_count["count"] += 1
        return _fake_critic_correction_applied(artifact_type, output, retrieved_chunks)

    sentinel_refiner = create_sentinel_refiner(counting_critic)
    adapter = RefinerAdapter({
        "sentinel_evaluation": sentinel_refiner,
    })

    artifact = _make_fake_artifact("sentinel_evaluation")
    refinement_request = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.PASS)]
    )

    result = adapter.refine("sentinel_evaluation", artifact, refinement_request, [], {})

    assert call_count["count"] == 0
    assert result == artifact


# ---------------------------------------------------------------------------
# 15. CRITIC CORRECTION_APPLIED TRUE
# ---------------------------------------------------------------------------

def test_critic_correction_applied_true_returns_corrected() -> None:
    sentinel_refiner = create_sentinel_refiner(_fake_critic_correction_applied)
    adapter = RefinerAdapter({
        "sentinel_evaluation": sentinel_refiner,
    })

    artifact = _make_fake_artifact("sentinel_evaluation")
    refinement_request = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    )

    result = adapter.refine("sentinel_evaluation", artifact, _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    ), [], {})

    assert result["success_line"] == "Corrected by A4 critic"


# ---------------------------------------------------------------------------
# 16. CRITIC CORRECTION_APPLIED FALSE
# ---------------------------------------------------------------------------

def test_critic_correction_applied_false_returns_unchanged() -> None:
    sentinel_refiner = create_sentinel_refiner(_fake_critic_no_correction)
    adapter = RefinerAdapter({
        "sentinel_evaluation": sentinel_refiner,
    })

    artifact = _make_fake_artifact("sentinel_evaluation")
    refinement_request = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    )

    result = adapter.refine("sentinel_evaluation", artifact, refinement_request, [], {})

    assert result == artifact


# ---------------------------------------------------------------------------
# 17. MISSING / INVALID CORRECTED_OUTPUT
# ---------------------------------------------------------------------------

def test_critic_bad_corrected_output_fallbacks_to_unchanged() -> None:
    sentinel_refiner = create_sentinel_refiner(_fake_critic_bad_corrected_output)
    adapter = RefinerAdapter({
        "sentinel_evaluation": sentinel_refiner,
    })

    artifact = _make_fake_artifact("sentinel_evaluation")
    refinement_request = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    )

    result = adapter.refine("sentinel_evaluation", artifact, refinement_request, [], {})

    assert result == artifact


# ---------------------------------------------------------------------------
# 18. A6 FINDING CONTROLS REFINEMENT
# ---------------------------------------------------------------------------

def test_a6_finding_controls_refinement() -> None:
    call_log = {"count_a": 0, "count_b": 0}

    def critic_a(*, artifact_type, output, retrieved_chunks):
        call_log["count_a"] += 1
        return _fake_critic_correction_applied(artifact_type, output, retrieved_chunks)

    def critic_b(*, artifact_type, output, retrieved_chunks):
        call_log["count_b"] += 1
        return _fake_critic_correction_applied(artifact_type, output, retrieved_chunks)

    sentinel_refiner = create_sentinel_refiner(critic_a)
    adapter = RefinerAdapter({
        "sentinel_evaluation": sentinel_refiner,
    })
    artifact = _make_fake_artifact("sentinel_evaluation")
    chunks = [{"section_id": "test"}]

    # Case A: sentinel_non_authority FAIL -> critic invoked
    refinement_request_a = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("sentinel_non_authority", CriterionOutcome.FAIL)]
    )
    adapter.refine("sentinel_evaluation", artifact, refinement_request_a, chunks, {})

    # Case B: no relevant FAIL -> critic NOT invoked
    refinement_request_b = _make_refinement_request(
        failed_criteria=[_make_criterion_finding("structural_completeness", CriterionOutcome.FAIL)]
    )
    adapter.refine("sentinel_evaluation", artifact, refinement_request_b, chunks, {})

    assert call_log["count_a"] == 1
    assert call_log["count_b"] == 0