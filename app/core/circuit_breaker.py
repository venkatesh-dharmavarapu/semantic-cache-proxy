from enum import Enum
import logging
import time
from typing import Optional, Tuple
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operations: traffic flows through
    OPEN = "OPEN"            # Tripped: requests fail fast or route to fallback
    HALF_OPEN = "HALF_OPEN"  # Testing recovery: trial requests permitted


class CircuitBreaker:
    PREFIX = "circuit:provider:"

    def __init__(
        self,
        provider_name: str,
        vector_store: Optional[VectorStoreManager] = None,
        failure_threshold: int = 4,
        recovery_timeout_seconds: float = 30.0,
    ):
        self.provider_name = provider_name
        self.store = vector_store or VectorStoreManager()
        self.store.connect()
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds

    def _state_key(self) -> str:
        return f"{self.PREFIX}{self.provider_name}:state"

    def _failures_key(self) -> str:
        return f"{self.PREFIX}{self.provider_name}:failures"

    def _last_tripped_key(self) -> str:
        return f"{self.PREFIX}{self.provider_name}:last_tripped"

    def get_state(self) -> CircuitState:
        client = self.store.client
        if not client:
            return CircuitState.CLOSED

        raw_state = client.get(self._state_key())
        state = raw_state.decode("utf-8") if raw_state else CircuitState.CLOSED

        if state == CircuitState.OPEN:
            last_tripped_raw = client.get(self._last_tripped_key())
            last_tripped = float(last_tripped_raw) if last_tripped_raw else 0.0
            if time.time() - last_tripped >= self.recovery_timeout_seconds:
                # Transition to HALF_OPEN to trial upstream health
                client.set(self._state_key(), CircuitState.HALF_OPEN)
                return CircuitState.HALF_OPEN

        return CircuitState(state)

    def can_execute(self) -> bool:
        """Determines if the provider should be attempted."""
        state = self.get_state()
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self):
        """Resets failure counts and closes the circuit on recovery."""
        client = self.store.client
        if not client:
            return

        pipe = client.pipeline()
        pipe.set(self._state_key(), CircuitState.CLOSED)
        pipe.delete(self._failures_key())
        pipe.delete(self._last_tripped_key())
        pipe.execute()

    def record_failure(self) -> CircuitState:
        """Increments error counts; trips circuit to OPEN if threshold breached."""
        client = self.store.client
        if not client:
            return CircuitState.CLOSED

        state = self.get_state()
        if state == CircuitState.HALF_OPEN:
            # Immediate return to OPEN if probe request fails
            client.set(self._state_key(), CircuitState.OPEN)
            client.set(self._last_tripped_key(), str(time.time()))
            return CircuitState.OPEN

        failures = client.incr(self._failures_key())
        # Set expiry window on failure bucket
        client.expire(self._failures_key(), int(self.recovery_timeout_seconds))

        if failures >= self.failure_threshold:
            pipe = client.pipeline()
            pipe.set(self._state_key(), CircuitState.OPEN)
            pipe.set(self._last_tripped_key(), str(time.time()))
            pipe.execute()
            logger.warning(
                "Circuit breaker tripped to OPEN for provider '%s' after %d failures",
                self.provider_name,
                failures,
            )
            return CircuitState.OPEN

        return CircuitState.CLOSED

    def reset(self):
        """Manually forces the circuit back to CLOSED state."""
        self.record_success()