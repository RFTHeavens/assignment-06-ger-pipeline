"""Assignment #6 Quality Gate.

Deterministic decision logic for GER pipeline termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

from src.ger_contracts import (
    EvaluationResult,
    CriterionFinding,
    CriterionOutcome,
    TerminalState,
)


@dataclass(frozen=True)
class QualityGateDecision:
    """Result of a quality gate decision.

    action: Loop-control action ("ACCEPT", "RETRY", "ESCALATE").
    terminal_state: Terminal state for manifest (None for RETRY).
    reason: Human-readable deterministic reason.
    no_progress: True if no-progress stall was detected.
    """

    action: Literal["ACCEPT", "RETRY", "ESCALATE"]
    terminal_state: Optional[TerminalState]
    reason: str
    no_progress: bool = False


class QualityGate:
    """Deterministic quality gate for GER pipeline.

    Stateless decision logic that consumes evaluation results and attempt
    state from the GER Controller and produces a deterministic decision.
    """

    def __init__(self) -> None:
        """Create a QualityGate with no internal retry budget state.

        The retry budget (max_attempts) is supplied explicitly by the
        GER Controller on each decision call.
        """
        pass

    def decide(
        self,
        evaluation_result: EvaluationResult,
        attempt_number: int,
        max_attempts: int,
        current_artifact: Dict[str, Any],
        previous_evaluation: Optional[EvaluationResult] = None,
        previous_artifact: Optional[Dict[str, Any]] = None,
    ) -> QualityGateDecision:
        """Make a quality gate decision.

        Args:
            evaluation_result: Result from Evaluator for current artifact.
            attempt_number: Current attempt number (1 = initial, 2+ = refinements).
            max_attempts: Maximum attempts allowed (supplied by GER Controller).
            current_artifact: The artifact dict that was evaluated.
            previous_evaluation: EvaluationResult from previous attempt (for no-progress).
            previous_artifact: Artifact dict from previous attempt (for no-progress).

        Returns:
            QualityGateDecision with action, terminal_state, reason, no_progress.

        Raises:
            ValueError: If attempt_number < 1 or max_attempts < 1.
        """
        if attempt_number < 1:
            raise ValueError(f"attempt_number must be >= 1, got {attempt_number}")
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        # 1. ACCEPT - all criteria PASS
        if evaluation_result.overall_pass():
            return QualityGateDecision(
                action="ACCEPT",
                terminal_state=TerminalState.ACCEPT,
                reason="All evaluation criteria passed.",
                no_progress=False,
            )

        # 2. MAX ATTEMPTS EXHAUSTED
        if attempt_number >= max_attempts:
            return QualityGateDecision(
                action="ESCALATE",
                terminal_state=TerminalState.ESCALATE,
                reason=f"Max attempts ({max_attempts}) reached with evaluation failures.",
                no_progress=False,
            )

        # 3. NO PROGRESS DETECTION
        if self._is_no_progress(
            evaluation_result=evaluation_result,
            previous_evaluation=previous_evaluation,
            current_artifact=current_artifact,
            previous_artifact=previous_artifact,
        ):
            return QualityGateDecision(
                action="ESCALATE",
                terminal_state=TerminalState.ESCALATE,
                reason="No progress detected: evaluation failures unchanged and artifact unchanged.",
                no_progress=True,
            )

        # 4. RETRY - failure present, budget remains, progress possible
        fail_count = sum(
            1 for f in evaluation_result.criteria_findings
            if f.outcome == CriterionOutcome.FAIL
        )
        return QualityGateDecision(
            action="RETRY",
            terminal_state=None,
            reason=f"{fail_count} evaluation criterion/criteria failed; retrying (attempt {attempt_number}/{max_attempts}).",
            no_progress=False,
        )

    def _is_no_progress(
        self,
        evaluation_result: EvaluationResult,
        previous_evaluation: Optional[EvaluationResult],
        current_artifact: Dict[str, Any],
        previous_artifact: Optional[Dict[str, Any]],
    ) -> bool:
        """Detect no-progress stall.

        No-progress when BOTH:
        - failed findings structurally unchanged (by criterion_id, outcome, severity, reason)
        - artifact structurally unchanged (deep equality)

        Args:
            evaluation_result: Current evaluation result.
            previous_evaluation: Previous evaluation result (if available).
            current_artifact: Current artifact dict.
            previous_artifact: Previous artifact dict (if available).

        Returns:
            True if no-progress detected, False otherwise.
        """
        if previous_evaluation is None or previous_artifact is None:
            return False

        # Compare failed findings structurally (order-independent)
        current_failures = self._get_failed_findings(evaluation_result)
        previous_failures = self._get_failed_findings(previous_evaluation)

        if not self._failed_findings_equal(current_failures, previous_failures):
            return False

        # Compare artifact structural equality
        if not self._artifacts_equal(current_artifact, previous_artifact):
            return False

        return True

    def _get_failed_findings(self, evaluation_result: EvaluationResult) -> List[CriterionFinding]:
        """Extract failed findings from EvaluationResult."""
        return [
            f for f in evaluation_result.criteria_findings
            if f.outcome == CriterionOutcome.FAIL
        ]

    def _failed_findings_equal(
        self,
        a: List[CriterionFinding],
        b: List[CriterionFinding],
    ) -> bool:
        """Compare two lists of failed findings for structural equality.

        Order-independent comparison by sorting structural tuples.
        Compares: criterion_id, outcome, severity, reason.
        """
        if len(a) != len(b):
            return False

        def finding_key(f: CriterionFinding) -> tuple:
            return (f.criterion_id, f.outcome, f.severity, f.reason)

        return sorted([finding_key(f) for f in a]) == sorted([finding_key(f) for f in b])

    def _artifacts_equal(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        """Deep structural equality of artifact dictionaries."""
        if a.keys() != b.keys():
            return False

        for key in a:
            if not self._values_equal(a[key], b[key]):
                return False
        return True

    def _values_equal(self, a: Any, b: Any) -> bool:
        """Recursive structural equality for JSON-serializable values."""
        if type(a) != type(b):
            return False

        if isinstance(a, dict):
            if a.keys() != b.keys():
                return False
            return all(self._values_equal(a[k], b[k]) for k in a)

        if isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(self._values_equal(x, y) for x, y in zip(a, b))

        return a == b