from diskcache import Cache
from functools import wraps
import hashlib
import json
from config import settings

# Initialize cache
cache = Cache(settings.CACHE_DIR, size_limit=1000000000)  # 1GB limit

def cache_llm_response(expire=86400):
    """Cache decorator for LLM responses"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not settings.CACHE_ENABLED:
                return func(*args, **kwargs)
            
            # Create cache key
            key_data = {
                "func": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            }
            key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Check cache
            if key in cache:
                return cache[key]
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, expire=expire)
            return result
        return wrapper
    return decorator

def clear_cache():
    """Clear all cached responses"""
    cache.clear()

def get_cache_stats():
    """Get cache statistics"""
    return {
        "size": cache.volume(),
        "count": len(cache),
        "enabled": settings.CACHE_ENABLED
    }