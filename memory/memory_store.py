import math
import uuid
from typing import List, Dict


class MemoryStore:
    def __init__(self):
        self.store = []

    def add(self, text: str):
        item = {"id": str(uuid.uuid4()), "text": text, "vector": self._embed(text)}
        self.store.append(item)

    def query(self, text: str, top_k: int = 5) -> List[Dict]:
        if not self.store:
            return []

        q_vec = self._embed(text)
        scored = []
        for item in self.store:
            score = self._cosine(q_vec, item["vector"])
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored[:top_k]]

    def all(self) -> List[Dict]:
        return self.store

    def _embed(self, text: str) -> List[float]:
        words = text.lower().split()
        vector = [0] * 128
        for w in words:
            idx = hash(w) % 128
            vector[idx] += 1
        return vector

    def _cosine(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
