import asyncio
import pytest
from app.core.event_mesh import DistributedEventMesh
from app.schemas.event_models import (
    NODE_INSTANCE_ID,
    CacheInvalidationEvent,
    InvalidationScope,
)


@pytest.mark.asyncio
async def test_event_schema_serialization():
    event = CacheInvalidationEvent(
        origin_node="node-test-123",
        scope=InvalidationScope.KEY_EVICTED,
        target="cache:exact:default:abc1234",
        metadata={"reason": "negative_feedback"},
    )
    dumped = event.model_dump_json()
    assert "node-test-123" in dumped
    assert "KEY_EVICTED" in dumped
    assert "cache:exact:default:abc1234" in dumped


@pytest.mark.asyncio
async def test_pubsub_broadcast_and_receipt():
    mesh_receiver = DistributedEventMesh()
    mesh_sender = DistributedEventMesh()

    received_events = []

    def on_threshold_update(event: CacheInvalidationEvent):
        received_events.append(event)

    mesh_receiver.register_handler(
        InvalidationScope.THRESHOLD_UPDATED, on_threshold_update
    )

    await mesh_receiver.start_listener()
    # Allow subscriber setup in Redis
    await asyncio.sleep(0.1)

    try:
        # Publish event from another node identity
        external_event = CacheInvalidationEvent(
            origin_node="remote-node-999",
            scope=InvalidationScope.THRESHOLD_UPDATED,
            target="global",
            metadata={"new_threshold": 0.82},
        )
        await mesh_sender.connect()
        await mesh_sender._pub_client.publish(
            DistributedEventMesh.CHANNEL_NAME, external_event.model_dump_json()
        )

        # Allow delivery
        await asyncio.sleep(0.2)

        assert len(received_events) == 1
        assert received_events[0].origin_node == "remote-node-999"
        assert received_events[0].metadata["new_threshold"] == 0.82
    finally:
        await mesh_receiver.close()
        await mesh_sender.close()


@pytest.mark.asyncio
async def test_self_origin_echo_suppression():
    mesh = DistributedEventMesh()
    received_events = []

    def on_event(event: CacheInvalidationEvent):
        received_events.append(event)

    mesh.register_handler(InvalidationScope.TENANT_PURGED, on_event)
    await mesh.start_listener()
    await asyncio.sleep(0.1)

    try:
        # Publishing from the same mesh sends NODE_INSTANCE_ID as origin
        await mesh.publish_event(
            scope=InvalidationScope.TENANT_PURGED,
            target="tenant_test_echo",
        )
        await asyncio.sleep(0.2)

        # Self-originated event must be suppressed
        assert len(received_events) == 0
    finally:
        await mesh.close()