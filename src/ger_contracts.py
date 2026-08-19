"""Assignment #6 GER Pipeline Contracts.

Defines the structured data contracts exchanged between GER components.
Implementation-neutral; no behavior logic.

SOURCE STATE = DERIVED-SOURCE-ONLY
RAW TRANSCRIPT = NOT AVAILABLE
RAW CLASS CHAT = NOT AVAILABLE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Literal


class CriterionOutcome(str, Enum):
    """Pass/fail outcome for a single evaluation criterion."""

    PASS = "PASS"
    FAIL = "FAIL"


class TerminalState(str, Enum):
    """Terminal state of a GER run. Defined here as contract vocabulary only.

    Quality Gate (future phase) owns the transition logic.
    """

    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class CriterionFinding:
    """Result of evaluating a single criterion."""

    criterion_id: str
    criterion_name: str
    outcome: CriterionOutcome
    reason: str
    evidence_refs: List[str] = field(default_factory=list)
    severity: Literal["critical", "major", "minor"] = "major"


@dataclass(frozen=True)
class EvaluationResult:
    """Result of evaluating a generated artifact against all criteria.

    Does NOT own final disposition. Quality Gate (future phase) decides
    ACCEPT / REJECT / ESCALATE based on this evaluation.
    """

    criteria_findings: List[CriterionFinding]
    evaluator_warnings: List[str] = field(default_factory=list)
    evidence_refs: List[str] = field(default_factory=list)

    def overall_pass(self) -> bool:
        """True if all findings are PASS."""
        return all(f.outcome == CriterionOutcome.PASS for f in self.criteria_findings)

    def has_critical_failures(self) -> bool:
        """True if any finding is FAIL with critical severity."""
        return any(
            f.outcome == CriterionOutcome.FAIL and f.severity == "critical"
            for f in self.criteria_findings
        )


@dataclass(frozen=True)
class RefinementRequest:
    """Request to refine an artifact based on evaluation findings."""

    failed_criteria: List[CriterionFinding]
    actionable_feedback: List[str]
    original_artifact_ref: str  # reference/identifier for the artifact


@dataclass(frozen=True)
class RefinementResult:
    """Result of attempting a refinement."""

    refined_artifact_ref: str  # reference to the refined artifact
    criteria_addressed: List[str]  # criterion_ids that were addressed
    unresolved_or_new_issues: List[CriterionFinding] = field(default_factory=list)


@dataclass(frozen=True)
class GERManifest:
    """Run manifest recording the complete GER execution trace."""

    run_id: str
    attempt_number: int
    evaluation_trace: List[Dict[str, Any]] = field(default_factory=list)
    refinement_trace: List[Dict[str, Any]] = field(default_factory=list)
    terminal_state: Optional[TerminalState] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_runtime_seconds: Optional[float] = None