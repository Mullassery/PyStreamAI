"""Advanced Caching Strategies - Semantic, embedding-level, result caching"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import hashlib
import json

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entry in cache"""
    key: str
    value: Any
    embedding: Optional[List[float]] = None
    similarity_score: float = 0.0
    hit_count: int = 0
    size_bytes: int = 0


class SemanticCache:
    """Cache based on semantic similarity (same meaning = cache hit)"""

    def __init__(self, embedding_model, similarity_threshold: float = 0.95):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.cache: Dict[str, CacheEntry] = {}
        self.embeddings: List[Tuple[str, List[float]]] = []

    def embed(self, text: str) -> List[float]:
        """Get embedding for text"""
        # Simplified - would call actual embedding model
        return [float(ord(c)) / 255.0 for c in text[:10]]

    def similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate embedding similarity (cosine)"""
        if not emb1 or not emb2:
            return 0.0

        dot_product = sum(a * b for a, b in zip(emb1, emb2))
        norm1 = sum(a * a for a in emb1) ** 0.5
        norm2 = sum(b * b for b in emb2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get(self, query: str) -> Optional[Tuple[Any, float]]:
        """Get cached value for similar query"""
        query_embedding = self.embed(query)

        best_match = None
        best_score = 0.0

        for key, embedding in self.embeddings:
            score = self.similarity(query_embedding, embedding)

            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = key

        if best_match:
            entry = self.cache[best_match]
            entry.hit_count += 1
            logger.info(f"Semantic cache hit: similarity={best_score:.3f}")
            return entry.value, best_score

        return None

    def set(self, query: str, result: Any) -> None:
        """Cache result for query"""
        embedding = self.embed(query)
        key = hashlib.md5(query.encode()).hexdigest()

        size = len(json.dumps(result, default=str).encode())

        entry = CacheEntry(
            key=key,
            value=result,
            embedding=embedding,
            size_bytes=size,
        )

        self.cache[key] = entry
        self.embeddings.append((key, embedding))

        logger.info(f"Cached result: key={key}, size={size}B")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {"entries": 0, "hit_rate": 0}

        total_hits = sum(e.hit_count for e in self.cache.values())
        total_size = sum(e.size_bytes for e in self.cache.values())

        return {
            "entries": len(self.cache),
            "total_hits": total_hits,
            "total_size_mb": total_size / 1024 / 1024,
            "avg_hits_per_entry": total_hits / len(self.cache),
        }


class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""

    def __init__(self, embedding_model, max_cache_size_mb: int = 500):
        self.embedding_model = embedding_model
        self.max_cache_size_mb = max_cache_size_mb
        self.cache: Dict[str, List[float]] = {}
        self.total_size_bytes = 0

    def get_key(self, text: str) -> str:
        """Get cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self.get_key(text)
        return self.cache.get(key)

    def set(self, text: str, embedding: List[float]) -> None:
        """Cache embedding"""
        key = self.get_key(text)

        # Check size
        embedding_size_bytes = len(embedding) * 8  # 8 bytes per float
        if self.total_size_bytes + embedding_size_bytes > self.max_cache_size_mb * 1024 * 1024:
            # Evict old entries
            self.cache = dict(list(self.cache.items())[-100:])
            self.total_size_bytes = sum(len(e) * 8 for e in self.cache.values())

        self.cache[key] = embedding
        self.total_size_bytes += embedding_size_bytes

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_embeddings": len(self.cache),
            "total_size_mb": self.total_size_bytes / 1024 / 1024,
            "max_size_mb": self.max_cache_size_mb,
        }


class ResultCache:
    """Cache inference results with versioning"""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}

    def get_key(self, model_id: str, input_data: Dict[str, Any], version: str = "v1") -> str:
        """Generate cache key"""
        key_str = f"{model_id}:{json.dumps(input_data, sort_keys=True)}:{version}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, model_id: str, input_data: Dict[str, Any], version: str = "v1") -> Optional[Any]:
        """Get cached result"""
        key = self.get_key(model_id, input_data, version)

        if key in self.cache:
            entry = self.cache[key]
            entry.hit_count += 1
            return entry.value

        return None

    def set(self, model_id: str, input_data: Dict[str, Any], result: Any, version: str = "v1") -> None:
        """Cache result"""
        key = self.get_key(model_id, input_data, version)

        size = len(json.dumps(result, default=str).encode())

        entry = CacheEntry(
            key=key,
            value=result,
            size_bytes=size,
        )

        self.cache[key] = entry

    def invalidate(self, model_id: str) -> None:
        """Invalidate all results for model (e.g., after retraining)"""
        to_remove = [k for k, v in self.cache.items() if model_id in k]
        for k in to_remove:
            del self.cache[k]

        logger.info(f"Invalidated {len(to_remove)} cache entries for {model_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache:
            return {"entries": 0, "hit_rate": 0, "total_size_mb": 0}

        total_hits = sum(e.hit_count for e in self.cache.values())
        total_size = sum(e.size_bytes for e in self.cache.values())
        total_requests = total_hits + len(self.cache)  # Rough estimate

        return {
            "entries": len(self.cache),
            "total_hits": total_hits,
            "hit_rate": (total_hits / total_requests * 100) if total_requests > 0 else 0,
            "total_size_mb": total_size / 1024 / 1024,
        }


class HybridCache:
    """Combine semantic, embedding, and result caching"""

    def __init__(self, embedding_model):
        self.semantic_cache = SemanticCache(embedding_model)
        self.embedding_cache = EmbeddingCache(embedding_model)
        self.result_cache = ResultCache()

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding with caching"""
        # Check embedding cache first
        cached = self.embedding_cache.get(text)
        if cached:
            return cached

        # Compute embedding
        embedding = self.semantic_cache.embed(text)

        # Cache it
        self.embedding_cache.set(text, embedding)

        return embedding

    def get_result(self, model_id: str, input_data: Dict[str, Any]) -> Optional[Any]:
        """Get result with semantic cache fallback"""
        # Exact match
        result = self.result_cache.get(model_id, input_data)
        if result:
            return result

        # Semantic match (convert input to text)
        query = json.dumps(input_data)
        result, similarity = self.semantic_cache.get(query)
        if result:
            logger.info(f"Semantic cache hit: similarity={similarity:.3f}")
            return result

        return None

    def set_result(self, model_id: str, input_data: Dict[str, Any], result: Any) -> None:
        """Cache result"""
        self.result_cache.set(model_id, input_data, result)

        query = json.dumps(input_data)
        self.semantic_cache.set(query, result)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        return {
            "semantic": self.semantic_cache.get_stats(),
            "embedding": self.embedding_cache.get_stats(),
            "result": self.result_cache.get_stats(),
        }

    def invalidate_model(self, model_id: str) -> None:
        """Invalidate all caches for model"""
        self.result_cache.invalidate(model_id)
        logger.info(f"Invalidated all caches for {model_id}")
