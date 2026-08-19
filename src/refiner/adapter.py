"""Assignment #6 Refiner Adapter.

Instance-based injected-callable refiner with constructor injection.
Uses the Phase 1 RefinementRequest as the canonical handoff contract.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

from src.ger_contracts import RefinementRequest

# Type alias for injected refinement callable
RefinementCallable = Callable[
    [
        str,                    # artifact_type
        Dict[str, Any],         # current_artifact
        Any,                    # RefinementRequest (imported locally to avoid circular)
        List[Dict[str, Any]],   # retrieved_chunks
        Optional[Dict[str, Any]],  # config
    ],
    Dict[str, Any],             # revised_artifact
]


class RefinerAdapter:
    """Instance-based refiner adapter with constructor injection.

    Eliminates module-global mutable registry. Each instance holds its own
    refiner mapping, providing natural test isolation and explicit
    dependency injection.
    """

    def __init__(
        self,
        refiners: Mapping[str, Callable],
    ) -> None:
        """Create a RefinerAdapter with the given refiner callables.

        Args:
            refiners: Mapping from artifact_type to refinement callable.
                Supported artifact types:
                - "sentinel_evaluation"
                - "keeper_trace"
                - "resonance_alignment_trial"
        """
        self._refiners = dict(refiners)

    def refine(
        self,
        artifact_type: str,
        current_artifact: Dict[str, Any],
        refinement_request: Any,  # RefinementRequest (avoid circular import at module level)
        retrieved_chunks: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Refine an artifact using the injected refinement callable.

        Args:
            artifact_type: One of the registered artifact types.
            current_artifact: The artifact dict to refine.
            refinement_request: The RefinementRequest contract containing
                failed_criteria, actionable_feedback, and original_artifact_ref.
            retrieved_chunks: List of retrieved GDD chunks with section_id, text, excerpt, etc.
            config: Optional config dict passed through for A4 signature compatibility.

        Returns:
            The refined artifact dict exactly as produced by the injected refinement callable.

        Raises:
            ValueError: If artifact_type is not registered, or if provenance invariant violated.
        """
        refiner = self._refiners.get(artifact_type)
        if refiner is None:
            raise ValueError(f"Unsupported artifact_type: {artifact_type!r}")

        # Provenance invariant: snapshot retrieved_context_ids before refinement
        original_ids = list(current_artifact.get("retrieved_context_ids", []))

        refined = refiner(
            artifact_type=artifact_type,
            current_artifact=current_artifact,
            refinement_request=refinement_request,
            retrieved_chunks=retrieved_chunks,
            config=config or {},
        )

        # Provenance invariant enforcement
        refined_ids = refined.get("retrieved_context_ids", [])
        if refined_ids != original_ids:
            raise ValueError(
                "Refiner violated provenance invariant: retrieved_context_ids changed"
            )

        return refined