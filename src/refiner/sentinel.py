"""Assignment #6 Sentinel Refinement.

Concrete deterministic refinement for sentinel_evaluation artifacts.
Wraps the injected Assignment #4 sentinel critic backend.
Only invoked when A6 evaluation finds sentinel_non_authority FAIL.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.ger_contracts import CriterionOutcome, RefinementRequest


# Type alias for injected A4 sentinel critic callable signature
# A4 critic: (artifact_type: str, output: dict, retrieved_chunks: list) -> dict
A4SentinelCritic = Callable[[str, Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]]


def _has_sentinel_non_authority_fail(refinement_request: Any) -> bool:
    """Check if refinement_request contains sentinel_non_authority FAIL finding."""
    for finding in getattr(refinement_request, "failed_criteria", []):
        if (
            getattr(finding, "criterion_id", "") == "sentinel_non_authority"
            and getattr(finding, "outcome", "") == CriterionOutcome.FAIL
        ):
            return True
    return False


def create_sentinel_refiner(
    a4_sentinel_critic: Callable[[str, Dict[str, Any], List[Dict[str, Any]]], Dict[str, Any]],
) -> Callable:
    """Factory that creates a sentinel_evaluation refinement callable.

    The returned callable conforms to the RefinementCallable signature and:
    - Only invokes the A4 critic when refinement_request contains
      sentinel_non_authority FAIL finding
    - Returns corrected_output from A4 critic when correction_applied
    - Returns current_artifact unchanged when no relevant FAIL or no correction

    Args:
        a4_sentinel_critic: Injected A4 critic callable with signature
            (artifact_type: str, output: dict, retrieved_chunks: list) -> dict
            Returns: {"issues_found": [...], "corrected_output": dict, "correction_applied": bool}

    Returns:
        A RefinementCallable for sentinel_evaluation artifact type.
    """

    def sentinel_evaluation_refiner(
        artifact_type: str,
        current_artifact: Dict[str, Any],
        refinement_request: Any,  # RefinementRequest
        retrieved_chunks: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # A6 evaluation controls refinement: only run if sentinel_non_authority FAIL
        if not _has_sentinel_non_authority_fail(refinement_request):
            return current_artifact

        # Invoke injected A4 sentinel critic
        critic_result = a4_sentinel_critic(
            artifact_type=artifact_type,
            output=current_artifact,
            retrieved_chunks=retrieved_chunks,
        )

        # Use corrected_output if critic applied a correction
        if critic_result.get("correction_applied", False):
            corrected = critic_result.get("corrected_output", current_artifact)
            if isinstance(corrected, dict):
                return corrected

        # No meaningful correction from A4 critic
        return current_artifact

    return sentinel_evaluation_refiner