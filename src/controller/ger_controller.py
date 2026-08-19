"""Assignment #6 GER Pipeline Controller.

Orchestrates the Generate → Evaluate → Quality Gate → Refine loop using
injected certified components. Does not own evaluation, refinement, or
retry policies.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.generators.adapter import GeneratorAdapter
from src.evaluator import Evaluator
from src.refiner.adapter import RefinerAdapter
from src.quality_gate.gate import QualityGate, QualityGateDecision
from src.ger_contracts import (
    CriterionFinding,
    CriterionOutcome,
    EvaluationResult,
    GERManifest,
    RefinementRequest,
    TerminalState,
)


class GERController:
    """Orchestrates the GER pipeline: Generate → Evaluate → QualityGate → Refine."""

    def __init__(
        self,
        generator_adapter: GeneratorAdapter,
        evaluator: "Evaluator",
        refiner_adapter: RefinerAdapter,
        quality_gate: QualityGate,
        max_attempts: int = 3,
    ) -> None:
        """Create a GERController with injected certified components.

        Args:
            generator_adapter: Injected generator adapter for artifact generation.
            evaluator: Injected evaluator for artifact evaluation.
            refiner_adapter: Injected refiner adapter for artifact refinement.
            quality_gate: Injected quality gate for decision making.
            max_attempts: Maximum number of generation/refinement attempts (supplied by caller).
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

        self._generator_adapter = generator_adapter
        self._evaluator = evaluator
        self._refiner_adapter = refiner_adapter
        self._quality_gate = quality_gate
        self._max_attempts = max_attempts

    def run(
        self,
        artifact_type: str,
        retrieved_chunks: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> GERManifest:
        """Execute the full GER pipeline for a single artifact type.

        Args:
            artifact_type: One of "sentinel_evaluation", "keeper_trace", "resonance_alignment_trial"
            retrieved_chunks: List of retrieved GDD chunks with section_id, text, excerpt, etc.
            config: Optional config dict passed to generator/refiner.
            run_id: Optional run identifier (generated if not provided).

        Returns:
            GERManifest with complete execution trace and terminal state.
        """
        run_id = run_id or str(uuid.uuid4())
        started_at = datetime.utcnow()

        # Initialize manifest
        manifest = GERManifest(
            run_id=run_id or str(uuid.uuid4()),
            attempt_number=0,
            started_at=datetime.utcnow(),
        )

        # Initialize
        current_artifact: Optional[Dict[str, Any]] = None
        previous_evaluation: Optional["EvaluationResult"] = None
        previous_artifact: Optional[Dict[str, Any]] = None

        start_time = time.time()

        for attempt_number in range(1, self._max_attempts + 1):
            # Create new manifest with updated attempt number (frozen dataclass)
            manifest = replace(manifest, attempt_number=attempt_number)

            # 1. GENERATE
            if current_artifact is None:
                current_artifact = self._generator_adapter.generate(
                    artifact_type=artifact_type,
                    retrieved_chunks=retrieved_chunks,
                    config=config,
                )

            # 2. EVALUATE
            evaluation_result = self._evaluator.evaluate(
                artifact_type=artifact_type,
                output=current_artifact,
                retrieved_chunks=retrieved_chunks,
            )

            # Record evaluation trace
            evaluation_trace_entry = {
                "attempt_number": attempt_number,
                "overall_pass": evaluation_result.overall_pass(),
                "findings": [
                    {
                        "criterion_id": f.criterion_id,
                        "criterion_name": f.criterion_name,
                        "outcome": f.outcome.value,
                        "reason": f.reason,
                        "severity": f.severity,
                    }
                    for f in evaluation_result.criteria_findings
                ],
                "warnings": evaluation_result.evaluator_warnings,
                "evidence_refs": evaluation_result.evidence_refs,
            }
            manifest = replace(manifest, evaluation_trace=manifest.evaluation_trace + [evaluation_trace_entry])

            # 3. QUALITY GATE
            decision = self._quality_gate.decide(
                evaluation_result=evaluation_result,
                attempt_number=attempt_number,
                max_attempts=self._max_attempts,
                current_artifact=current_artifact,
                previous_evaluation=previous_evaluation,
                previous_artifact=previous_artifact,
            )

            # Record decision
            manifest = replace(manifest, evaluation_trace=[
                *manifest.evaluation_trace[:-1],
                {**manifest.evaluation_trace[-1], "decision": {
                    "action": decision.action,
                    "terminal_state": decision.terminal_state.value if decision.terminal_state else None,
                    "reason": decision.reason,
                    "no_progress": decision.no_progress,
                }}
            ])

            # 4. DECIDE
            if decision.action == "ACCEPT":
                return replace(
                    manifest,
                    terminal_state=TerminalState.ACCEPT,
                    completed_at=datetime.utcnow(),
                    total_runtime_seconds=time.time() - start_time,
                )

            if decision.action == "ESCALATE":
                return replace(
                    manifest,
                    terminal_state=decision.terminal_state or TerminalState.ESCALATE,
                    completed_at=datetime.utcnow(),
                    total_runtime_seconds=time.time() - start_time,
                )

            # 3. RETRY - refine and continue
            # Build refinement request from failed findings
            failed_findings = [
                f for f in evaluation_result.criteria_findings
                if f.outcome == CriterionOutcome.FAIL
            ]
            actionable_feedback = [f.reason for f in failed_findings]

            refinement_request = RefinementRequest(
                failed_criteria=failed_findings,
                actionable_feedback=[f.reason for f in failed_findings],
                original_artifact_ref=f"attempt_{attempt_number}",
            )

            # Refine
            refined_artifact = self._refiner_adapter.refine(
                artifact_type=artifact_type,
                current_artifact=current_artifact,
                refinement_request=refinement_request,
                retrieved_chunks=retrieved_chunks,
                config=None,
            )

            # Record refinement trace
            manifest = replace(manifest, refinement_trace=manifest.refinement_trace + [{
                "attempt_number": attempt_number,
                "refinement_request": {
                    "failed_criteria": [
                        {"criterion_id": f.criterion_id, "reason": f.reason}
                        for f in failed_findings
                    ],
                    "actionable_feedback": actionable_feedback,
                },
                "refined_artifact_type": artifact_type,
            }])

            # Prepare for next iteration
            previous_evaluation = evaluation_result
            previous_artifact = current_artifact
            current_artifact = refined_artifact

        # Loop completed without ACCEPT - max attempts reached
        return replace(
            manifest,
            terminal_state=TerminalState.ESCALATE,
            completed_at=datetime.utcnow(),
            total_runtime_seconds=time.time() - start_time,
        )


def create_ger_controller(
    generator_adapter: GeneratorAdapter,
    evaluator: "Evaluator",
    refiner_adapter: RefinerAdapter,
    quality_gate: QualityGate,
    max_attempts: int = 3,
) -> GERController:
    """Factory function to create a GERController instance."""
    return GERController(
        generator_adapter=generator_adapter,
        evaluator=evaluator,
        refiner_adapter=refiner_adapter,
        quality_gate=quality_gate,
        max_attempts=max_attempts,
    )