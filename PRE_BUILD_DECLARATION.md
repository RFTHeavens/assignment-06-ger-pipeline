# PRE_BUILD_DECLARATION.md
## Assignment
**Assignment #6 — Build a GER Pipeline**

## Content Type
The GER pipeline generates **sentinel_evaluation** artifacts for Project Sentinel — structured JSON descriptions of game needs, success/failure states, retry guidance, and future warnings.

## GDD Rule
Every piece must satisfy **GDD §3.3 — Sentinel Non-Authority**: the Sentinel/Relay must surface evidence for player interpretation and must not evaluate, decide, choose, recommend, or determine the player's conclusion.

## Concrete Failure
A failure occurs when the generated artifact contains forbidden authorial verb patterns attaching Sentinel/Relay to a conclusion, e.g. "Sentinel evaluates the evidence and determines the correct relationship for you." The evaluator flags these as critical failures.

## Pipeline Connection
The GER pipeline's evaluator enforces this rule via a precompiled regex scanning all artifact string fields. 111/111 tests pass with the sentinel_non_authority criterion active.