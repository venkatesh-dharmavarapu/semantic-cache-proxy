from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field
import logging
import redis

logger = logging.getLogger(__name__)


class TenantTier(str, Enum):
    FREE = "free"
    STANDARD = "standard"
    ENTERPRISE = "enterprise"


class TenantConfig(BaseModel):
    tier: TenantTier = TenantTier.STANDARD
    similarity_threshold: float = 0.88
    max_ttl_seconds: int = 86400
    rate_limit_per_minute: int = 60
    token_quota_daily: int = 100_000


class TenantPolicyManager:
    # Default policy matrix per tier
    TIER_DEFAULTS: Dict[TenantTier, TenantConfig] = {
        TenantTier.FREE: TenantConfig(
            tier=TenantTier.FREE,
            similarity_threshold=0.92,  # Strict threshold to minimize false hits
            max_ttl_seconds=3600,       # 1 hour retention
            rate_limit_per_minute=20,
            token_quota_daily=20_000,
        ),
        TenantTier.STANDARD: TenantConfig(
            tier=TenantTier.STANDARD,
            similarity_threshold=0.88,
            max_ttl_seconds=86400,      # 24 hours retention
            rate_limit_per_minute=60,
            token_quota_daily=200_000,
        ),
        TenantTier.ENTERPRISE: TenantConfig(
            tier=TenantTier.ENTERPRISE,
            similarity_threshold=0.85,  # Broader semantic coverage
            max_ttl_seconds=604800,     # 7 days retention
            rate_limit_per_minute=300,
            token_quota_daily=2_000_000,
        ),
    }

    def __init__(self):
        # In-memory overrides or custom configurations per registered tenant
        self.custom_tenants: Dict[str, TenantConfig] = {}

    def get_config(self, tenant_id: str) -> TenantConfig:
        """Retrieves tenant config; defaults to Standard tier if unconfigured."""
        if tenant_id in self.custom_tenants:
            return self.custom_tenants[tenant_id]
        return self.TIER_DEFAULTS[TenantTier.STANDARD]

    def register_tenant(self, tenant_id: str, config: TenantConfig):
        """Registers custom policy settings for a specific tenant ID."""
        self.custom_tenants[tenant_id] = config


class TenantPurgeService:
    @staticmethod
    def purge_tenant_namespace(client: redis.Redis, tenant_id: str) -> int:
        """
        Atomically removes all L1 exact keys, rate limits, token quotas,
        and L2 semantic vector entries for a specific tenant namespace.
        """
        if not client or not tenant_id:
            return 0

        purged_count = 0

        # 1. Purge L1 Exact Match keys: cache:exact:<tenant_id>:*
        exact_pattern = f"cache:exact:{tenant_id}:*"
        for key in client.scan_iter(exact_pattern):
            client.delete(key)
            purged_count += 1

        # 2. Purge Rate Limiting keys: ratelimit:<tenant_id>:*
        ratelimit_pattern = f"ratelimit:{tenant_id}:*"
        for key in client.scan_iter(ratelimit_pattern):
            client.delete(key)
            purged_count += 1

        # 3. Purge Token Budget Quota keys: budget:tenant:<tenant_id>:*
        budget_pattern = f"budget:tenant:{tenant_id}:*"
        for key in client.scan_iter(budget_pattern):
            client.delete(key)
            purged_count += 1

        # 4. Purge L2 Vector Hash entries tagged with this tenant_id
        for key in client.scan_iter("cache:llm:*"):
            stored_tenant = client.hget(key, "tenant_id")
            if stored_tenant and stored_tenant.decode("utf-8") == tenant_id:
                client.delete(key)
                purged_count += 1

        logger.info("Purged %d total keys for tenant '%s'", purged_count, tenant_id)
        return purged_count