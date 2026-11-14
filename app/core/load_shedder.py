import asyncio
import logging
import time
from typing import Tuple

logger = logging.getLogger(__name__)


class AdaptiveLoadShedder:
    """
    Monitors concurrency and loop lag to drop or downgrade requests
    before CPU or connection queues saturate under sudden traffic spikes.
    """

    def __init__(
        self,
        max_concurrent_requests: int = 500,
        loop_lag_threshold_ms: float = 75.0,
    ):
        self.max_concurrent_requests = max_concurrent_requests
        self.loop_lag_threshold_ms = loop_lag_threshold_ms
        self.current_in_flight: int = 0
        self._last_tick: float = time.perf_counter()
        self._loop_lag_ms: float = 0.0
        self._monitor_task: asyncio.Task | None = None
        self._is_running: bool = False

    async def start_monitor(self):
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._measure_loop_lag())

    async def stop_monitor(self):
        self._is_running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _measure_loop_lag(self):
        """Measures event-loop lag by scheduling a 50ms sleep and checking overshoot."""
        while self._is_running:
            start = time.perf_counter()
            await asyncio.sleep(0.05)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._loop_lag_ms = max(0.0, elapsed_ms - 50.0)

    def acquire(self) -> Tuple[bool, str]:
        """
        Evaluates whether an incoming request can be accepted.
        Returns: (accepted: bool, reason: str)
        """
        if self.current_in_flight >= self.max_concurrent_requests:
            logger.warning(
                "Load shedding triggered: concurrency limit breached (%d/%d)",
                self.current_in_flight,
                self.max_concurrent_requests,
            )
            return False, "CONCURRENCY_SATURATED"

        if self._loop_lag_ms > self.loop_lag_threshold_ms:
            logger.warning(
                "Load shedding triggered: event loop lag high (%.2fms > %.2fms)",
                self._loop_lag_ms,
                self.loop_lag_threshold_ms,
            )
            return False, "EVENT_LOOP_DEGRADED"

        self.current_in_flight += 1
        return True, "OK"

    def release(self):
        if self.current_in_flight > 0:
            self.current_in_flight -= 1