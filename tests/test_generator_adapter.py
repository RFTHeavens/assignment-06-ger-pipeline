"""Assignment #6 Phase E — GeneratorAdapter deterministic tests.

Tests validate the CURRENT instance-based GeneratorAdapter behavior.
"""

import pytest

from src.generators.adapter import GeneratorAdapter, GeneratorCallable


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


def _fake_generator_sentinel_eval(
    retrieved_chunks: list, config: dict
) -> dict:
    return _make_fake_artifact("sentinel_evaluation")


def _fake_generator_keeper_trace(
    retrieved_chunks: list, config: dict
) -> dict:
    return _make_fake_artifact("keeper_trace")


def _fake_generator_resonance_trial(
    retrieved_chunks: list, config: dict
) -> dict:
    return _make_fake_artifact("resonance_alignment_trial")


def _fake_generator_raises(
    retrieved_chunks: list, config: dict
) -> dict:
    raise _FakeException("controlled generator failure")


# ---------------------------------------------------------------------------
# 1. Unchanged return contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "artifact_type,generator",
    [
        ("sentinel_evaluation", _fake_generator_sentinel_eval),
        ("keeper_trace", _fake_generator_keeper_trace),
        ("resonance_alignment_trial", _fake_generator_resonance_trial),
    ],
)
def test_generate_returns_injected_artifact_unchanged(
    artifact_type: str, generator: GeneratorCallable
) -> None:
    adapter = GeneratorAdapter({artifact_type: generator})
    chunks = [{"section_id": "test", "text": "test text"}]
    config = {"test_key": "test_value"}

    result = adapter.generate(artifact_type, chunks, config)

    expected = _make_fake_artifact(artifact_type)
    assert result == expected


# ---------------------------------------------------------------------------
# 2. Retrieved chunks pass-through
# ---------------------------------------------------------------------------

def test_retrieved_chunks_passed_unchanged() -> None:
    captured_chunks = {}

    def capture_generator(retrieved_chunks: list, config: dict) -> dict:
        captured_chunks["chunks"] = retrieved_chunks
        captured_chunks["config"] = config
        return _make_fake_artifact("sentinel_evaluation")

    adapter = GeneratorAdapter({"sentinel_evaluation": capture_generator})
    chunks = [
        {"section_id": "1", "text": "chunk one"},
        {"section_id": "2", "text": "chunk two"},
    ]

    adapter.generate("sentinel_evaluation", chunks, {})

    assert captured_chunks["chunks"] is chunks
    assert captured_chunks["chunks"] == chunks


# ---------------------------------------------------------------------------
# 3. Config pass-through
# ---------------------------------------------------------------------------

def test_config_passed_unchanged() -> None:
    captured_config = {}

    def capture_generator(retrieved_chunks: list, config: dict) -> dict:
        captured_config["config"] = config
        return _make_fake_artifact("sentinel_evaluation")

    adapter = GeneratorAdapter({"sentinel_evaluation": capture_generator})
    config = {"example": "value", "nested": {"key": 42}}

    adapter.generate("sentinel_evaluation", [], config)

    assert captured_config["config"] is config
    assert captured_config["config"] == config


def test_none_config_becomes_empty_dict() -> None:
    captured_config = {}

    def capture_generator(retrieved_chunks: list, config: dict) -> dict:
        captured_config["config"] = config
        return _make_fake_artifact("sentinel_evaluation")

    adapter = GeneratorAdapter({"sentinel_evaluation": capture_generator})

    adapter.generate("sentinel_evaluation", [], None)

    assert captured_config["config"] == {}


# ---------------------------------------------------------------------------
# 4. Unsupported artifact type
# ---------------------------------------------------------------------------

def test_unsupported_artifact_type_raises_valueerror() -> None:
    adapter = GeneratorAdapter({})

    with pytest.raises(ValueError) as exc_info:
        adapter.generate("unsupported_type", [], {})

    assert "unsupported_type" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 5. Generator exception propagation
# ---------------------------------------------------------------------------

def test_generator_exception_propagates_unchanged() -> None:
    adapter = GeneratorAdapter(
        {"sentinel_evaluation": _fake_generator_raises}
    )

    with pytest.raises(_FakeException) as exc_info:
        adapter.generate("sentinel_evaluation", [], {})

    assert str(exc_info.value) == "controlled generator failure"


# ---------------------------------------------------------------------------
# 6. Instance isolation
# ---------------------------------------------------------------------------

def test_instance_isolation() -> None:
    adapter_a = GeneratorAdapter(
        {"sentinel_evaluation": _fake_generator_sentinel_eval}
    )
    adapter_b = GeneratorAdapter(
        {"keeper_trace": _fake_generator_keeper_trace}
    )

    # Each instance has only its own registered generator
    result_a = adapter_a.generate("sentinel_evaluation", [], {})
    assert result_a["artifact_type"] == "sentinel_evaluation"

    with pytest.raises(ValueError):
        adapter_a.generate("keeper_trace", [], {})

    result_b = adapter_b.generate("keeper_trace", [], {})
    assert result_b["artifact_type"] == "keeper_trace"

    with pytest.raises(ValueError):
        adapter_b.generate("sentinel_evaluation", [], {})

    # Neither instance can access the other's generator
    with pytest.raises(ValueError):
        adapter_a.generate("resonance_alignment_trial", [], {})

    with pytest.raises(ValueError):
        adapter_b.generate("resonance_alignment_trial", [], {})


# ---------------------------------------------------------------------------
# 7. Constructor copy isolation
# ---------------------------------------------------------------------------

def test_constructor_copies_mapping() -> None:
    original_mapping = {"sentinel_evaluation": _fake_generator_sentinel_eval}
    adapter = GeneratorAdapter(original_mapping)

    # Mutate the original mapping after adapter construction
    original_mapping["keeper_trace"] = _fake_generator_keeper_trace

    # Adapter should still only have its original generator
    result = adapter.generate("sentinel_evaluation", [], {})
    assert result["artifact_type"] == "sentinel_evaluation"

    with pytest.raises(ValueError):
        adapter.generate("keeper_trace", [], {})


# ---------------------------------------------------------------------------
# Sanity: adapters with full generator sets work
# ---------------------------------------------------------------------------

def test_full_generator_set() -> None:
    adapter = GeneratorAdapter({
        "sentinel_evaluation": _fake_generator_sentinel_eval,
        "keeper_trace": _fake_generator_keeper_trace,
        "resonance_alignment_trial": _fake_generator_resonance_trial,
    })

    assert adapter.generate("sentinel_evaluation", [], {})["artifact_type"] == "sentinel_evaluation"
    assert adapter.generate("keeper_trace", [], {})["artifact_type"] == "keeper_trace"
    assert adapter.generate("resonance_alignment_trial", [], {})["artifact_type"] == "resonance_alignment_trial"