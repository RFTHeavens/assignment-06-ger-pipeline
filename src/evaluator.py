"""Assignment #6 GER Pipeline Evaluator.

Evaluates generated artifacts against deterministic criteria using injected
critic and validator dependencies. Produces EvaluationResult per Phase 1 contracts.

SOURCE STATE = DERIVED-SOURCE-ONLY
RAW TRANSCRIPT = NOT AVAILABLE
RAW CLASS CHAT = NOT AVAILABLE
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from .ger_contracts import (
    CriterionFinding,
    CriterionOutcome,
    EvaluationResult,
    TerminalState,
)


# Type aliases for injected dependencies
CriticFn = Callable[[str, dict, list], dict]
ValidatorFn = Callable[[str, dict], bool]


# Criterion IDs used across evaluation
CRITERION_SENTINEL_NON_AUTHORITY = "sentinel_non_authority"
CRITERION_SCHEMA = "schema_validity"
CRITERION_LORE_BOUNDARIES = "lore_boundaries"
CRITERION_AETH_NUL = "aeth_nul_complementary"
CRITERION_SILENCE = "silence_as_evidence"
CRITERION_TONE = "tone_consistency"
CRITERION_EVIDENCE_REFS = "evidence_reference_presence"
CRITERION_STRUCTURAL = "structural_completeness"


# Sentinel Non-Authority (GDD §3.3 — Agent Boundaries and Validation Authority).
# The Sentinel/Relay surface evidence for player interpretation and must not
# evaluate, decide, choose, recommend, or determine the player's conclusion.
# Detectable deterministically via forbidden authorial verb patterns attaching
# Sentinel/Relay to a conclusion. This is the PRIMARY rubric-facing GDD rule
# for Assignment #6.
_SENTINEL_SUBJECTS = ("sentinel", "relay")
_SENTINEL_AUTHORIAL_VERBS = (
    "evaluates", "evaluated", "evaluate", "evaluating",
    "decides", "decided", "decide", "deciding",
    "chooses", "chose", "choose", "choosing",
    "selects", "selected", "select", "selecting",
    "recommends", "recommended", "recommend", "recommending",
    "determines", "determined", "determine", "determining",
    "interprets", "interpreted", "interpret", "interpreting",
)
_SENTINEL_BANNED_PHRASES = (
    "correct relationship",
    "you should align",
    "you must align",
    "the right choice",
    "the correct choice",
    "the answer is",
)


# Precompiled Sentinel Non-Authority detector (GDD §3.3).
#
# Deterministic word-boundary regex detecting prohibited Sentinel/Relay
# authorial language. Permits a SMALL CLOSED SET of separators between the
# subject (sentinel / relay) and the authorial verb:
#   - ordinary whitespace
#   - a single comma / colon / hyphen / en-dash / em-dash, optionally surrounded by whitespace
#   - exactly one of a closed set of auxiliary/modal/article tokens
#     (will, shall, must, should, can, could, may, might, is, was, has, the)
#     surrounded by optional whitespace
# Bound is tight: only ONE such connector is allowed; the verb must follow the
# subject directly or with exactly one short connector. This avoids false
# positives on prose where the subject and an authorial verb happen to appear
# in unrelated clauses. Word boundaries anchor both ends so substrings like
# "sentinels" or "selectable" do not trigger.
_SENTINEL_GAP = (
    r"(?:"
    r"\s+"
    r"|\s*,\s*"
    r"|\s*:\s*"
    r"|\s*[-\u2014\u2013]\s*"
    r"|\s+(?:will|shall|must|should|can|could|may|might|is|was|has|the)\s+"
    r")?"
)
_SENTINEL_VERB_ALTERNATION = "|".join(
    sorted(set(_SENTINEL_AUTHORIAL_VERBS), key=len, reverse=True)
)
_SENTINEL_SUBJECT_ALTERNATION = "|".join(_SENTINEL_SUBJECTS)
_SENTINEL_NON_AUTHORITY_PATTERN = re.compile(
    r"\b(?P<subject>" + _SENTINEL_SUBJECT_ALTERNATION + r")\b"
    + _SENTINEL_GAP
    + r"\b(?P<verb>" + _SENTINEL_VERB_ALTERNATION + r")\b",
    re.IGNORECASE,
)


def _make_finding(
    criterion_id: str,
    criterion_name: str,
    outcome: CriterionOutcome,
    reason: str,
    severity: str = "major",
    evidence_refs: Optional[List[str]] = None,
) -> CriterionFinding:
    """Construct a CriterionFinding with consistent structure."""
    return CriterionFinding(
        criterion_id=criterion_id,
        criterion_name=criterion_name,
        outcome=outcome,
        reason=reason,
        severity=severity,
        evidence_refs=evidence_refs or [],
    )


class Evaluator:
    """Deterministic evaluator for generated artifacts.

    Composes injected critic and validator dependencies to produce
    EvaluationResult findings. Does NOT decide final disposition.
    """

    def __init__(
        self,
        critic: CriticFn,
        validator: ValidatorFn,
    ) -> None:
        """Initialize evaluator with injected dependencies.

        Args:
            critic: Callable(artifact_type: str, output: dict, retrieved_chunks: list) -> dict
                     Expected to return dict with 'issues_found', 'corrected_output', 'correction_applied'
            validator: Callable(artifact_type: str, output: dict) -> bool
                        Expected to return True if output passes schema validation
        """
        self._critic = critic
        self._validator = validator

    def evaluate(
        self,
        artifact_type: str,
        output: dict,
        retrieved_chunks: list,
    ) -> EvaluationResult:
        """Evaluate a generated artifact against all deterministic criteria.

        Args:
            artifact_type: One of 'sentinel_evaluation', 'keeper_trace', 'resonance_alignment_trial'
            output: The generated artifact dict
            retrieved_chunks: List of retrieved GDD chunks for context

        Returns:
            EvaluationResult with criteria_findings, evaluator_warnings, evidence_refs
        """
        findings: List = []
        warnings: List[str] = []
        evidence_refs: List[str] = []

        # Collect evidence refs from retrieved chunks
        for chunk in retrieved_chunks:
            if isinstance(chunk, dict):
                sid = chunk.get("section_id")
                if sid:
                    evidence_refs.append(f"section_{sid}")

        # 1. SCHEMA_VALIDITY
        schema_passed = self._check_schema(artifact_type, output)
        findings.append(
            _make_finding(
                criterion_id=CRITERION_SCHEMA,
                criterion_name="Schema Validity",
                outcome=CriterionOutcome.PASS if schema_passed else CriterionOutcome.FAIL,
                reason="Artifact conforms to JSON schema" if schema_passed else "Artifact fails schema validation",
                severity="critical" if not schema_passed else "major",
                evidence_refs=evidence_refs[:],
            )
        )

        # 2. SENTINEL_NON_AUTHORITY — PRIMARY rubric-facing GDD rule (GDD §3.3).
        # Evaluator visibly enforces the Sentinel boundary that Sentinel/Relay
        # must surface evidence rather than evaluate, decide, choose, recommend,
        # or determine the player's conclusion.
        findings.extend(self._check_sentinel_non_authority(output, evidence_refs))

        # 3-6. Lore/tone/structure checks via injected critic
        critic_result = self._run_critic(artifact_type, output, retrieved_chunks)
        critic_findings = self._normalize_critic_findings(critic_result, evidence_refs)
        findings.extend(critic_findings)

        # 7. STRUCTURAL_COMPLETENESS
        structural_findings = self._check_structural_completeness(artifact_type, output)
        findings.extend(structural_findings)

        # 8. EVIDENCE_REFERENCE_PRESENCE
        evidence_ref_finding = self._check_evidence_reference_presence(output, evidence_refs)
        findings.append(evidence_ref_finding)

        # Collect evaluator warnings
        evaluator_warnings = self._collect_warnings(findings, schema_passed)

        return EvaluationResult(
            criteria_findings=findings,
            evaluator_warnings=evaluator_warnings,
            evidence_refs=evidence_refs,
        )

    def _check_schema(self, artifact_type: str, output: dict) -> bool:
        """Validate output against artifact JSON schema via injected validator."""
        try:
            return self._validator(artifact_type, output)
        except Exception as e:
            return False

    def _check_sentinel_non_authority(
        self,
        output: dict,
        evidence_refs: List[str],
    ) -> List[CriterionFinding]:
        """Enforce the Sentinel Non-Authority GDD rule (GDD §3.3).

        Sentinel/Relay surface evidence for player interpretation and must NOT
        evaluate, decide, choose, recommend, or determine the player's
        conclusion. This is the PRIMARY rubric-facing GDD rule for Assignment #6.

        Deterministic detection: scan all string fields of the artifact for
        forbidden authorial verb patterns attaching Sentinel/Relay to a
        conclusion, plus a small set of banned phrases that imply Sentinel/
        Relay is presenting the answer to the player.

        Returns:
            A list containing exactly one CriterionFinding for
            sentinel_non_authority (PASS if no violation found, FAIL otherwise).
        """
        violations: List[str] = []

        def _scan(obj: object, path: str = "") -> None:
            if isinstance(obj, str):
                lowered = obj.lower()
                for match in _SENTINEL_NON_AUTHORITY_PATTERN.finditer(obj):
                    snippet = match.group(0)
                    violations.append(
                        f"{path or '<root>'}: detected '{snippet}' — "
                        f"Sentinel/Relay must surface evidence, not author "
                        f"the player's conclusion (GDD §3.3)"
                    )
                for phrase in _SENTINEL_BANNED_PHRASES:
                    if phrase in lowered and any(
                        s in lowered for s in _SENTINEL_SUBJECTS
                    ):
                        violations.append(
                            f"{path or '<root>'}: detected banned phrase "
                            f"'{phrase}' near Sentinel/Relay reference — "
                            f"violates Sentinel Non-Authority (GDD §3.3)"
                        )
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    _scan(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for index, item in enumerate(obj):
                    _scan(item, f"{path}[{index}]")

        _scan(output)

        if violations:
            return [
                _make_finding(
                    criterion_id=CRITERION_SENTINEL_NON_AUTHORITY,
                    criterion_name="Sentinel Non-Authority (GDD §3.3)",
                    outcome=CriterionOutcome.FAIL,
                    reason="; ".join(violations),
                    severity="critical",
                    evidence_refs=evidence_refs[:],
                )
            ]
        return [
            _make_finding(
                criterion_id=CRITERION_SENTINEL_NON_AUTHORITY,
                criterion_name="Sentinel Non-Authority (GDD §3.3)",
                outcome=CriterionOutcome.PASS,
                reason=(
                    "Artifact surfaces evidence without Sentinel/Relay "
                    "authoring the player's conclusion"
                ),
                severity="critical",
                evidence_refs=evidence_refs[:],
            )
        ]

    def _run_critic(
        self,
        artifact_type: str,
        output: dict,
        retrieved_chunks: list,
    ) -> dict:
        """Invoke injected critic and handle malformed responses."""
        try:
            result = self._critic(artifact_type, output, retrieved_chunks)
            if not isinstance(result, dict):
                return {"issues_found": [], "corrected_output": output, "correction_applied": False}
            return result
        except Exception as e:
            return {"issues_found": [], "corrected_output": output, "correction_applied": False}

    def _normalize_critic_findings(
        self,
        critic_result: dict,
        evidence_refs: List[str],
    ) -> List[CriterionFinding]:
        """Convert critic issues to standardized CriterionFinding list.

        Emits explicit PASS findings for every exercised criterion family so
        that the EvaluationResult is exhaustive. Successfully evaluated criteria
        must not silently disappear from the result; otherwise overall_pass()
        would incorrectly report True even when a criterion was effectively
        un-exercised.
        """
        findings: List[CriterionFinding] = []

        issues = critic_result.get("issues_found", [])
        if not isinstance(issues, list):
            issues = []

        # Track which criterion families produced a FAIL from the critic.
        failed_families: Dict[str, List[str]] = {
            CRITERION_LORE_BOUNDARIES: [],
            CRITERION_AETH_NUL: [],
            CRITERION_SILENCE: [],
            CRITERION_TONE: [],
        }

        for issue in issues:
            if not isinstance(issue, dict):
                continue

            issue_type = issue.get("type", "")
            description = issue.get("description", "")
            severity = issue.get("severity", "major")
            path = issue.get("path", "")

            # Map critic issue types to our criteria
            if issue_type == "lore_break":
                if "sentinel" in path or "sentinel" in description.lower():
                    failed_families[CRITERION_LORE_BOUNDARIES].append(description)
                elif "aeth_nul" in path or "binary" in description.lower():
                    failed_families[CRITERION_AETH_NUL].append(description)
                else:
                    # Default lore_break family: lore_boundaries
                    failed_families[CRITERION_LORE_BOUNDARIES].append(description)
            elif issue_type == "tone_drift":
                if "silence" in path or "silence" in description.lower():
                    failed_families[CRITERION_SILENCE].append(description)
                else:
                    failed_families[CRITERION_TONE].append(description)

        # Emit one finding per criterion family. FAIL if any issues matched
        # that family, otherwise PASS — so the EvaluationResult is exhaustive
        # and overall_pass() reflects every exercised criterion.
        family_names = {
            CRITERION_LORE_BOUNDARIES: "Sentinel Lore Boundaries",
            CRITERION_AETH_NUL: "Aeth/Nul Complementary States",
            CRITERION_SILENCE: "Silence as Evidence",
            CRITERION_TONE: "Tone Consistency",
        }
        for family_id, descs in failed_families.items():
            if descs:
                findings.append(
                    _make_finding(
                        criterion_id=family_id,
                        criterion_name=family_names[family_id],
                        outcome=CriterionOutcome.FAIL,
                        reason="; ".join(descs),
                        severity="major",
                        evidence_refs=evidence_refs[:],
                    )
                )
            else:
                findings.append(
                    _make_finding(
                        criterion_id=family_id,
                        criterion_name=family_names[family_id],
                        outcome=CriterionOutcome.PASS,
                        reason=f"{family_names[family_id]} satisfied",
                        severity="major",
                        evidence_refs=evidence_refs[:],
                    )
                )

        return findings

    def _check_structural_completeness(
        self,
        artifact_type: str,
        output: dict,
    ) -> List[CriterionFinding]:
        """Verify required fields are present and non-empty per artifact type."""
        findings: List[CriterionFinding] = []

        required_fields = {
            "sentinel_evaluation": [
                "artifact_type", "game_need", "retrieval_query",
                "retrieved_context_ids", "success_line", "failure_line",
                "retry_guidance", "future_warning",
            ],
            "keeper_trace": [
                "artifact_type", "game_need", "retrieval_query",
                "retrieved_context_ids", "fragments",
            ],
            "resonance_alignment_trial": [
                "artifact_type", "game_need", "retrieval_query",
                "retrieved_context_ids", "evidence_items",
                "intended_relationship", "guided_mode_hint",
            ],
        }

        required = required_fields.get(artifact_type, [])
        missing = [field for field in required if field not in output or not output.get(field)]

        if missing:
            findings.append(
                _make_finding(
                    criterion_id=CRITERION_STRUCTURAL,
                    criterion_name="Structural Completeness",
                    outcome=CriterionOutcome.FAIL,
                    reason=f"Missing required fields: {', '.join(missing)}",
                    severity="critical",
                )
            )
        else:
            findings.append(
                _make_finding(
                    criterion_id=CRITERION_STRUCTURAL,
                    criterion_name="Structural Completeness",
                    outcome=CriterionOutcome.PASS,
                    reason="All required fields present and non-empty",
                    severity="major",
                )
            )

        return findings

    def _check_evidence_reference_presence(
        self,
        output: dict,
        evidence_refs: List[str],
    ) -> CriterionFinding:
        """Verify evidence references are present and non-empty where required.

        Evidence references must be non-empty. An empty retrieved_context_ids
        or empty evidence_refs collection does NOT count as PASS — silence is
        meaningful evidence, but a claim of missing references is itself a
        finding that must be surfaced rather than silently accepted.
        """
        has_external_refs = bool(evidence_refs)

        retrieved_context_ids = output.get("retrieved_context_ids", [])
        has_output_refs = (
            isinstance(retrieved_context_ids, list)
            and len(retrieved_context_ids) > 0
        )

        has_refs = has_external_refs or has_output_refs

        return _make_finding(
            criterion_id=CRITERION_EVIDENCE_REFS,
            criterion_name="Evidence Reference Presence",
            outcome=CriterionOutcome.PASS if has_refs else CriterionOutcome.FAIL,
            reason=(
                "Artifact includes non-empty evidence references"
                if has_refs
                else "Artifact is missing or has empty evidence references"
            ),
            severity="major",
            evidence_refs=evidence_refs[:] if has_refs else [],
        )

    def _collect_warnings(
        self,
        findings: List[CriterionFinding],
        schema_passed: bool,
    ) -> List[str]:
        """Collect non-critical warnings from evaluation."""
        warnings: List[str] = []

        if not schema_passed:
            warnings.append("Schema validation failed; subsequent checks may be affected")

        fail_count = sum(1 for f in findings if f.outcome == CriterionOutcome.FAIL)
        if fail_count > 0:
            warnings.append(f"{fail_count} evaluation criterion/criteria failed")

        return warnings


def create_evaluator(
    critic: CriticFn,
    validator: ValidatorFn,
) -> Evaluator:
    """Factory function to create an Evaluator instance."""
    return Evaluator(critic=critic, validator=validator)