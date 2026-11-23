import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class FlightChannel:
    def __init__(self):
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.waiter_count: int = 0


class SingleFlightCoordinator:
    """
    Coalesces concurrent duplicate requests in-memory so that only one
    leader executes the work, broadcasting results instantly to all waiters.
    """

    def __init__(self):
        self._flights: Dict[str, FlightChannel] = {}
        self._lock = asyncio.Lock()

    async def execute_or_join(
        self,
        key: str,
        coro_fn: Callable[[], Coroutine[Any, Any, Tuple[Dict[str, Any], Dict[str, str]]]],
    ) -> Tuple[Dict[str, Any], Dict[str, str], bool, int]:
        """
        Executes work or waits on an active in-flight channel.

        Returns:
            (payload, extra_headers, is_leader, waiter_count)
        """
        async with self._lock:
            if key in self._flights:
                flight = self._flights[key]
                flight.waiter_count += 1
                logger.debug("Coalescing request into active in-flight channel for key '%s'", key)
                is_leader = False
            else:
                flight = FlightChannel()
                self._flights[key] = flight
                is_leader = True

        if not is_leader:
            # Wait for leader to complete and broadcast result
            try:
                res, headers = await flight.future
                follower_headers = dict(headers)
                follower_headers["X-Coalesced-Request"] = "true"
                return res, follower_headers, False, flight.waiter_count
            except Exception as exc:
                logger.warning("Leader task failed for key '%s': %s", key, exc)
                raise exc

        # Leader path
        try:
            res, headers = await coro_fn()
            leader_headers = dict(headers)
            if flight.waiter_count > 0:
                leader_headers["X-Coalesced-Followers"] = str(flight.waiter_count)

            # Broadcast success to all waiting followers
            if not flight.future.done():
                flight.future.set_result((res, headers))
            return res, leader_headers, True, flight.waiter_count
        except Exception as exc:
            # Broadcast exception to all waiting followers
            if not flight.future.done():
                flight.future.set_exception(exc)
            raise exc
        finally:
            async with self._lock:
                self._flights.pop(key, None)