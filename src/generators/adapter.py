"""Assignment #6 Generator Adapter.

Instance-based injected-callable adapter for Assignment #4 generator functions.
No Assignment #4 imports; callables are injected via constructor.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional

# Type alias for Assignment #4 generator callable signature
GeneratorCallable = Callable[[List[Dict[str, Any]], Dict], Dict]


class GeneratorAdapter:
    """Instance-based generator adapter with constructor injection.

    Eliminates module-global mutable registry. Each instance holds its own
    generator mapping, providing natural test isolation and explicit
    dependency injection.
    """

    def __init__(
        self,
        generators: Mapping[str, GeneratorCallable],
    ) -> None:
        """Create a GeneratorAdapter with the given generator callables.

        Args:
            generators: Mapping from artifact_type to generator callable.
                Supported artifact types:
                - "sentinel_evaluation"
                - "keeper_trace"
                - "resonance_alignment_trial"
        """
        self._generators = dict(generators)

    def generate(
        self,
        artifact_type: str,
        retrieved_chunks: List[Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate an artifact using the injected generator callable.

        Args:
            artifact_type: One of the registered artifact types.
            retrieved_chunks: List of retrieved GDD chunks with section_id, text, excerpt, etc.
            config: Optional config dict passed through for Assignment #4 signature compatibility.

        Returns:
            The artifact dict exactly as produced by the injected generator callable.

        Raises:
            ValueError: If artifact_type is not registered.
        """
        generator = self._generators.get(artifact_type)
        if generator is None:
            raise ValueError(f"Unsupported artifact_type: {artifact_type!r}")

        return generator(retrieved_chunks, config or {})