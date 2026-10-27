from typing import List
import numpy as np


class VectorCompressor:
    @staticmethod
    def to_float16_bytes(vector: List[float]) -> bytes:
        """Converts float32 vector list into 16-bit half-precision byte buffer."""
        return np.array(vector, dtype=np.float16).tobytes()

    @staticmethod
    def to_float32_bytes(vector: List[float]) -> bytes:
        """Standard 32-bit float vector byte buffer for HNSW indexing."""
        return np.array(vector, dtype=np.float32).tobytes()

    @staticmethod
    def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculates exact cosine similarity between two float vectors."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))