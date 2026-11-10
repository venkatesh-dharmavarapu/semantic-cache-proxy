import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional
import redis.asyncio as aioredis
from app.config import settings
from app.schemas.event_models import (
    NODE_INSTANCE_ID,
    CacheInvalidationEvent,
    InvalidationScope,
)

logger = logging.getLogger(__name__)


class DistributedEventMesh:
    CHANNEL_NAME = "cache:events:invalidation"

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._pub_client: Optional[aioredis.Redis] = None
        self._sub_client: Optional[aioredis.Redis] = None
        self._sub_task: Optional[asyncio.Task] = None
        self._handlers: Dict[InvalidationScope, List[Callable[[CacheInvalidationEvent], Any]]] = {
            scope: [] for scope in InvalidationScope
        }
        self.is_running = False

    async def connect(self):
        """Initializes async Redis clients for publishing and subscribing."""
        if not self._pub_client:
            self._pub_client = aioredis.from_url(self.redis_url, decode_responses=True)
        if not self._sub_client:
            self._sub_client = aioredis.from_url(self.redis_url, decode_responses=True)

    async def close(self):
        """Gracefully shuts down subscriber loop and client connections."""
        self.is_running = False
        if self._sub_task and not self._sub_task.done():
            self._sub_task.cancel()
            try:
                await self._sub_task
            except asyncio.CancelledError:
                pass

        if self._sub_client:
            await self._sub_client.close()
            self._sub_client = None

        if self._pub_client:
            await self._pub_client.close()
            self._pub_client = None

    def register_handler(
        self,
        scope: InvalidationScope,
        handler: Callable[[CacheInvalidationEvent], Any],
    ):
        """Registers a callback invoked when an event matching scope is received."""
        self._handlers[scope].append(handler)

    async def publish_event(
        self,
        scope: InvalidationScope,
        target: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Publishes an invalidation event to the Redis mesh."""
        await self.connect()
        event = CacheInvalidationEvent(
            origin_node=NODE_INSTANCE_ID,
            scope=scope,
            target=target,
            metadata=metadata or {},
        )
        try:
            payload = event.model_dump_json()
            if self._pub_client:
                await self._pub_client.publish(self.CHANNEL_NAME, payload)
                return True
        except Exception as exc:
            logger.error("Failed to publish cache invalidation event: %s", exc)
        return False

    async def start_listener(self):
        """Starts background task consuming events from the mesh."""
        await self.connect()
        self.is_running = True
        self._sub_task = asyncio.create_task(self._listener_loop())

    async def _listener_loop(self):
        try:
            pubsub = self._sub_client.pubsub()
            await pubsub.subscribe(self.CHANNEL_NAME)
            logger.info("Subscribed to Redis event mesh channel: %s", self.CHANNEL_NAME)

            while self.is_running:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if not message or message.get("type") != "message":
                    await asyncio.sleep(0.01)
                    continue

                raw_data = message.get("data")
                if not raw_data:
                    continue

                try:
                    event_dict = json.loads(raw_data)
                    event = CacheInvalidationEvent(**event_dict)

                    # Suppress self-echo
                    if event.origin_node == NODE_INSTANCE_ID:
                        continue

                    # Dispatch to registered local handlers
                    handlers = self._handlers.get(event.scope, [])
                    for handler in handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                handler(event)
                        except Exception as h_err:
                            logger.error(
                                "Handler error processing event %s: %s",
                                event.event_id,
                                h_err,
                            )
                except Exception as parse_err:
                    logger.warning("Invalid event payload received on mesh: %s", parse_err)

        except asyncio.CancelledError:
            logger.info("Mesh listener task received cancellation.")
        except Exception as e:
            logger.error("Fatal error in mesh listener loop: %s", e)
        finally:
            self.is_running = False