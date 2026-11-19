import gzip
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from app.core.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class SnapshotCheckpointer:
    SNAPSHOT_DIR = "snapshots"

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.store = vector_store or VectorStoreManager()
        self.store.connect()
        os.makedirs(self.SNAPSHOT_DIR, exist_ok=True)

    def export_snapshot(
        self,
        min_hits: int = 1,
        filename: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> Tuple[str, int]:
        """
        Exports active high-value cache entries into a compressed JSONL file.
        Returns: (snapshot_filepath, exported_count)
        """
        client = self.store.client
        if not client:
            return "", 0

        timestamp = int(time.time())
        target_file = filename or f"cache_snapshot_{timestamp}.jsonl.gz"
        filepath = os.path.join(self.SNAPSHOT_DIR, target_file)

        exported_count = 0

        with gzip.open(filepath, "wt", encoding="utf-8") as gz_out:
            for key in client.scan_iter("cache:llm:*"):
                entry = client.hgetall(key)
                if not entry:
                    continue

                # Parse hash fields
                hit_count = int(entry.get(b"hit_count", 0))
                if hit_count < min_hits:
                    continue

                stored_tenant = entry.get(b"tenant_id", b"default").decode("utf-8")
                if tenant_id and stored_tenant != tenant_id:
                    continue

                # Decode raw float32 vector bytes into float list for export
                vector_bytes = entry.get(b"prompt_vector", b"")
                if vector_bytes:
                    vector_arr = np.frombuffer(vector_bytes, dtype=np.float32).tolist()
                else:
                    vector_arr = []

                record = {
                    "key": key.decode("utf-8") if isinstance(key, bytes) else key,
                    "prompt": entry.get(b"prompt", b"").decode("utf-8"),
                    "response_json": entry.get(b"response_json", b"").decode("utf-8"),
                    "model": entry.get(b"model", b"").decode("utf-8"),
                    "system_prompt_hash": entry.get(b"system_prompt_hash", b"").decode("utf-8"),
                    "params_hash": entry.get(b"params_hash", b"").decode("utf-8"),
                    "context_hash": entry.get(b"context_hash", b"stateless").decode("utf-8"),
                    "tenant_id": stored_tenant,
                    "hit_count": hit_count,
                    "ttl": int(entry.get(b"ttl", 86400)),
                    "prompt_vector": vector_arr,
                }

                gz_out.write(json.dumps(record) + "\n")
                exported_count += 1

        logger.info("Exported %d cache records to snapshot %s", exported_count, filepath)
        return filepath, exported_count