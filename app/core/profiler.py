import time
from typing import Dict, List


class RequestProfiler:
    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}
        self.start_all = time.perf_counter()

    def start(self, stage_name: str):
        self.timers[stage_name] = time.perf_counter()

    def stop(self, stage_name: str) -> float:
        if stage_name in self.timers:
            elapsed_ms = (time.perf_counter() - self.timers[stage_name]) * 1000.0
            self.durations[stage_name] = round(elapsed_ms, 2)
            return self.durations[stage_name]
        return 0.0

    def total_elapsed_ms(self) -> float:
        return round((time.perf_counter() - self.start_all) * 1000.0, 2)

    def to_server_timing_header(self) -> str:
        """
        Formats durations into W3C Server-Timing header format:
        e.g., 'hash;dur=0.4, l1;dur=1.1, embed;dur=14.2, hnsw;dur=3.1'
        """
        parts: List[str] = []
        for name, dur in self.durations.items():
            parts.append(f"{name};dur={dur:.2f}")
        return ", ".join(parts)