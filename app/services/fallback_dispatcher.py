import logging
from typing import Any, Dict, List, Optional, Tuple
from app.core.circuit_breaker import CircuitBreaker, CircuitState
from app.core.vector_store import VectorStoreManager
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class UpstreamFallbackDispatcher:
    def __init__(
        self,
        primary_client: Optional[LLMClient] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ):
        self.primary_client = primary_client or LLMClient()
        self.store = vector_store or VectorStoreManager()
        self.breakers: Dict[str, CircuitBreaker] = {}

    def get_circuit_breaker(self, provider_id: str) -> CircuitBreaker:
        if provider_id not in self.breakers:
            self.breakers[provider_id] = CircuitBreaker(
                provider_name=provider_id,
                vector_store=self.store,
                failure_threshold=3,
                recovery_timeout_seconds=20.0,
            )
        return self.breakers[provider_id]

    async def execute_with_fallback(
        self,
        payload: Dict[str, Any],
        fallback_models: Optional[List[str]] = None,
        traceparent: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Attempts execution across a degradation ladder:
        1. Primary requested model.
        2. Sequence of secondary/fallback models.
        Returns: (response_data, serving_model)
        """
        requested_model = payload.get("model", "default")
        candidates = [requested_model]
        if fallback_models:
            candidates.extend([m for m in fallback_models if m != requested_model])

        last_error = None

        for model_name in candidates:
            provider_id = f"provider_{model_name}"
            breaker = self.get_circuit_breaker(provider_id)

            if not breaker.can_execute():
                logger.warning("Skipping model '%s' - Circuit breaker is OPEN", model_name)
                continue

            current_payload = dict(payload)
            current_payload["model"] = model_name

            try:
                response = await self.primary_client.forward_chat_completion(
                    current_payload, traceparent=traceparent
                )
                breaker.record_success()
                return response, model_name
            except Exception as exc:
                logger.error("Upstream invocation failed for model '%s': %s", model_name, exc)
                breaker.record_failure()
                last_error = exc

        # All candidates exhausted or circuit-broken
        raise UpstreamServiceUnavailableError(
            f"All model providers in degradation ladder failed. Last error: {last_error}"
        )


class UpstreamServiceUnavailableError(Exception):
    """Raised when all candidate providers in the degradation ladder fail."""
    pass