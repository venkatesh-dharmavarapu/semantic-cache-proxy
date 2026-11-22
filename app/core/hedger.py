import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RequestHedger:
    """
    Coordinates speculative parallel hedging across upstream model requests
    to eliminate long-tail latency spikes (P95/P99).
    """

    def __init__(self, default_hedge_delay_ms: float = 650.0):
        self.default_hedge_delay_ms = default_hedge_delay_ms

    async def execute_hedged_race(
        self,
        primary_coro_fn: Callable[[], Coroutine[Any, Any, Dict[str, Any]]],
        secondary_coro_fn: Optional[Callable[[], Coroutine[Any, Any, Dict[str, Any]]]] = None,
        hedge_delay_ms: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], str, bool]:
        """
        Executes primary coroutine. If incomplete after hedge_delay_ms, launches
        secondary coroutine in parallel and returns whichever completes first.
        
        Returns:
            (result_data, winner_tag, was_hedged)
        """
        delay_sec = (hedge_delay_ms or self.default_hedge_delay_ms) / 1000.0
        primary_task = asyncio.create_task(primary_coro_fn())

        if not secondary_coro_fn or delay_sec <= 0:
            # No secondary available: await primary directly
            res = await primary_task
            return res, "primary", False

        # Wait for primary up to the hedge delay window
        done, _ = await asyncio.wait([primary_task], timeout=delay_sec)

        if primary_task in done:
            # Primary returned before hedge threshold
            return primary_task.result(), "primary", False

        # Primary exceeded latency threshold -> trigger speculative hedge
        logger.info("Primary request latency exceeded %.2fms; dispatching speculative hedge", delay_sec * 1000)
        secondary_task = asyncio.create_task(secondary_coro_fn())

        # Race primary and secondary to first completion
        done, pending = await asyncio.wait(
            [primary_task, secondary_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        winning_task = next(iter(done))
        winner_tag = "primary" if winning_task is primary_task else "secondary"

        # Cancel trailing task immediately to free downstream compute
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        try:
            result = winning_task.result()
            return result, winner_tag, True
        except Exception as exc:
            # If the winning task failed with an exception, check if trailing task succeeded
            logger.warning("Winning task (%s) raised exception: %s. Awaiting trailing task...", winner_tag, exc)
            for t in pending:
                if not t.cancelled():
                    return await t, "fallback", True
            raise exc