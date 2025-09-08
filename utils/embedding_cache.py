"""
Embedding Cache Manager

Provides intelligent caching for OpenAI embeddings with hybrid storage,
persistence, and advanced features for production use.
"""

import json
import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict
import gzip
import shutil

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """
    Production-ready embedding cache with hybrid storage and advanced features.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache",
        max_memory_entries: int = 10000,
        max_disk_entries: int = 50000,
        ttl_days: int = 30,
        auto_save_interval: int = 300,  # 5 minutes
        compression: bool = True
    ):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_memory_entries: Maximum entries in memory cache
            max_disk_entries: Maximum entries in disk cache
            ttl_days: Time-to-live for cache entries in days
            auto_save_interval: Auto-save interval in seconds
            compression: Whether to compress cache files
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.max_disk_entries = max_disk_entries
        self.ttl_days = ttl_days
        self.auto_save_interval = auto_save_interval
        self.compression = compression
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Memory cache (LRU)
        self._memory_cache: OrderedDict[str, Dict] = OrderedDict()
        
        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "api_calls_saved": 0,
            "total_queries": 0,
            "cache_size": 0,
            "last_save": None,
            "created_at": datetime.now().isoformat()
        }
        
        # Auto-save thread
        self._auto_save_thread = None
        self._shutdown_event = threading.Event()
        
        # Initialize cache
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize the cache directory and load existing cache."""
        try:
            # Create cache directory
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing cache
            self._load_cache()
            
            # Start auto-save thread
            self._start_auto_save()
            
            logger.info(f"✅ Embedding cache initialized: {len(self._memory_cache)} entries loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding cache: {e}")
            # Continue with empty cache
    
    def _get_cache_file_path(self, filename: str) -> Path:
        """Get the full path for a cache file."""
        return self.cache_dir / filename
    
    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        # Normalize text for better cache hits
        normalized_text = text.strip().lower()
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    def _is_expired(self, entry: Dict) -> bool:
        """Check if a cache entry is expired."""
        if "expires_at" not in entry:
            return True
        
        try:
            expires_at = datetime.fromisoformat(entry["expires_at"])
            return datetime.now() > expires_at
        except (ValueError, TypeError):
            return True
    
    def _create_cache_entry(self, text: str, embedding: List[float]) -> Dict:
        """Create a new cache entry."""
        expires_at = datetime.now() + timedelta(days=self.ttl_days)
        
        return {
            "text": text,
            "embedding": embedding,
            "created_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "usage_count": 1,
            "last_used": datetime.now().isoformat()
        }
    
    def _update_entry_usage(self, entry: Dict):
        """Update usage statistics for a cache entry."""
        entry["usage_count"] = entry.get("usage_count", 0) + 1
        entry["last_used"] = datetime.now().isoformat()
    
    def _evict_old_entries(self):
        """Evict old entries if cache is full."""
        while len(self._memory_cache) >= self.max_memory_entries:
            # Remove least recently used entry
            if self._memory_cache:
                self._memory_cache.popitem(last=False)
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            cache_file = self._get_cache_file_path("embeddings.json.gz" if self.compression else "embeddings.json")
            
            if not cache_file.exists():
                logger.info("No existing cache file found, starting with empty cache")
                return
            
            # Load cache data
            if self.compression:
                with gzip.open(cache_file, 'rt', encoding='utf-8') as f:
                    cache_data = json.load(f)
            else:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # Load entries into memory cache
            loaded_count = 0
            expired_count = 0
            
            for key, entry in cache_data.get("entries", {}).items():
                if not self._is_expired(entry):
                    self._memory_cache[key] = entry
                    loaded_count += 1
                else:
                    expired_count += 1
            
            # Load statistics
            if "stats" in cache_data:
                self._stats.update(cache_data["stats"])
            
            logger.info(f"✅ Cache loaded: {loaded_count} entries, {expired_count} expired")
            
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")
            # Continue with empty cache
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with self._lock:
                # Prepare cache data
                cache_data = {
                    "entries": dict(self._memory_cache),
                    "stats": self._stats.copy(),
                    "saved_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
                
                # Save to temporary file first
                temp_file = self._get_cache_file_path("embeddings.json.tmp")
                cache_file = self._get_cache_file_path("embeddings.json.gz" if self.compression else "embeddings.json")
                
                if self.compression:
                    with gzip.open(temp_file, 'wt', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2)
                else:
                    with open(temp_file, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, indent=2)
                
                # Atomic move
                shutil.move(str(temp_file), str(cache_file))
                
                # Update stats
                self._stats["last_save"] = datetime.now().isoformat()
                self._stats["cache_size"] = len(self._memory_cache)
                
                logger.debug(f"✅ Cache saved: {len(self._memory_cache)} entries")
                
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def _start_auto_save(self):
        """Start the auto-save thread."""
        def auto_save_worker():
            while not self._shutdown_event.wait(self.auto_save_interval):
                try:
                    self._save_cache()
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
        
        self._auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        self._auto_save_thread.start()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: The text to get embedding for
            
        Returns:
            Embedding vector if found in cache, None otherwise
        """
        if not text or not text.strip():
            return None
        
        with self._lock:
            self._stats["total_queries"] += 1
            cache_key = self._get_cache_key(text)
            
            # Check memory cache
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                
                # Check if expired
                if self._is_expired(entry):
                    del self._memory_cache[cache_key]
                    self._stats["misses"] += 1
                    return None
                
                # Update usage and move to end (LRU)
                self._update_entry_usage(entry)
                self._memory_cache.move_to_end(cache_key)
                
                # Update stats
                self._stats["hits"] += 1
                self._stats["api_calls_saved"] += 1
                
                logger.debug(f"Cache hit for: {text[:50]}...")
                return entry["embedding"]
            
            # Cache miss
            self._stats["misses"] += 1
            logger.debug(f"Cache miss for: {text[:50]}...")
            return None
    
    def put(self, text: str, embedding: List[float]):
        """
        Store embedding in cache.
        
        Args:
            text: The text that was embedded
            embedding: The embedding vector
        """
        if not text or not text.strip() or not embedding:
            return
        
        with self._lock:
            cache_key = self._get_cache_key(text)
            
            # Create new entry
            entry = self._create_cache_entry(text, embedding)
            
            # Evict old entries if needed
            self._evict_old_entries()
            
            # Store in memory cache
            self._memory_cache[cache_key] = entry
            
            logger.debug(f"Cached embedding for: {text[:50]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            hit_rate = 0
            if self._stats["total_queries"] > 0:
                hit_rate = (self._stats["hits"] / self._stats["total_queries"]) * 100
            
            return {
                **self._stats,
                "hit_rate_percent": round(hit_rate, 2),
                "memory_entries": len(self._memory_cache),
                "cache_size_mb": round(len(self._memory_cache) * 6 / 1024 / 1024, 2)  # ~6KB per entry
            }
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._memory_cache.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "api_calls_saved": 0,
                "total_queries": 0,
                "cache_size": 0,
                "last_save": None,
                "created_at": datetime.now().isoformat()
            }
            logger.info("✅ Cache cleared")
    
    def cleanup_expired(self):
        """Remove expired entries from cache."""
        with self._lock:
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
            
            if expired_keys:
                logger.info(f"✅ Cleaned up {len(expired_keys)} expired entries")
    
    def shutdown(self):
        """Shutdown the cache and save to disk."""
        logger.info("Shutting down embedding cache...")
        
        # Stop auto-save thread
        if self._auto_save_thread:
            self._shutdown_event.set()
            self._auto_save_thread.join(timeout=5)
        
        # Final save
        self._save_cache()
        
        # Log final stats
        stats = self.get_stats()
        logger.info(f"✅ Cache shutdown complete: {stats['memory_entries']} entries, "
                   f"{stats['hit_rate_percent']}% hit rate, "
                   f"{stats['api_calls_saved']} API calls saved")

# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None
_cache_lock = threading.Lock()

def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _embedding_cache
    
    if _embedding_cache is None:
        with _cache_lock:
            if _embedding_cache is None:
                _embedding_cache = EmbeddingCache()
    
    return _embedding_cache

def shutdown_embedding_cache():
    """Shutdown the global embedding cache."""
    global _embedding_cache
    
    if _embedding_cache:
        with _cache_lock:
            if _embedding_cache:
                _embedding_cache.shutdown()
                _embedding_cache = None
