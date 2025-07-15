"""
Redis client for caching and session management.
"""
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
from redis.exceptions import RedisError, ConnectionError

from app.core.config import settings
from app.core.exceptions import ExternalServiceException
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Redis client wrapper for caching and data storage."""
    
    def __init__(self):
        self.redis_url = settings.REDIS_URL
        self.default_ttl = settings.REDIS_CACHE_TTL
        self._client: Optional[redis.Redis] = None
    
    async def get_client(self) -> redis.Redis:
        """Get Redis client instance."""
        if not self._client:
            self._client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                retry_on_timeout=True,
                health_check_interval=30,
            )
        return self._client
    
    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set a value in Redis.
        
        Args:
            key: Redis key
            value: Value to store
            ttl: Time to live in seconds
            serialize: Whether to serialize the value
            
        Returns:
            True if successful
        """
        try:
            client = await self.get_client()
            
            # Serialize value if needed
            if serialize:
                if isinstance(value, (dict, list)):
                    serialized_value = json.dumps(value, default=str)
                else:
                    serialized_value = str(value)
            else:
                serialized_value = value
            
            # Set TTL
            if ttl is None:
                ttl = self.default_ttl
            
            # Store in Redis
            if ttl > 0:
                result = await client.setex(key, ttl, serialized_value)
            else:
                result = await client.set(key, serialized_value)
            
            logger.debug(f"Set Redis key: {key} (TTL: {ttl})")
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Redis error setting key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting Redis key {key}: {e}")
            return False
    
    async def get(
        self,
        key: str,
        deserialize: bool = True,
        default: Any = None
    ) -> Any:
        """
        Get a value from Redis.
        
        Args:
            key: Redis key
            deserialize: Whether to deserialize the value
            default: Default value if key not found
            
        Returns:
            Retrieved value or default
        """
        try:
            client = await self.get_client()
            value = await client.get(key)
            
            if value is None:
                return default
            
            # Deserialize value if needed
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except RedisError as e:
            logger.error(f"Redis error getting key {key}: {e}")
            return default
        except Exception as e:
            logger.error(f"Error getting Redis key {key}: {e}")
            return default
    
    async def delete(self, key: str) -> bool:
        """
        Delete a key from Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if deleted
        """
        try:
            client = await self.get_client()
            result = await client.delete(key)
            
            logger.debug(f"Deleted Redis key: {key}")
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Redis error deleting key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error deleting Redis key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if exists
        """
        try:
            client = await self.get_client()
            result = await client.exists(key)
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Redis error checking key existence {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking Redis key existence {key}: {e}")
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for a key.
        
        Args:
            key: Redis key
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        try:
            client = await self.get_client()
            result = await client.expire(key, ttl)
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Redis error setting expiration for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting expiration for Redis key {key}: {e}")
            return False
    
    async def incr(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Increment a key value.
        
        Args:
            key: Redis key
            amount: Increment amount
            
        Returns:
            New value or None if error
        """
        try:
            client = await self.get_client()
            result = await client.incrby(key, amount)
            return result
            
        except RedisError as e:
            logger.error(f"Redis error incrementing key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error incrementing Redis key {key}: {e}")
            return None
    
    async def lpush(self, key: str, *values: Any) -> Optional[int]:
        """
        Push values to the left of a list.
        
        Args:
            key: Redis key
            values: Values to push
            
        Returns:
            New list length or None if error
        """
        try:
            client = await self.get_client()
            
            # Serialize values
            serialized_values = [
                json.dumps(value, default=str) if isinstance(value, (dict, list))
                else str(value)
                for value in values
            ]
            
            result = await client.lpush(key, *serialized_values)
            return result
            
        except RedisError as e:
            logger.error(f"Redis error pushing to list {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error pushing to Redis list {key}: {e}")
            return None
    
    async def lrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        deserialize: bool = True
    ) -> List[Any]:
        """
        Get range of values from a list.
        
        Args:
            key: Redis key
            start: Start index
            end: End index
            deserialize: Whether to deserialize values
            
        Returns:
            List of values
        """
        try:
            client = await self.get_client()
            values = await client.lrange(key, start, end)
            
            if not deserialize:
                return values
            
            # Deserialize values
            deserialized = []
            for value in values:
                try:
                    deserialized.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    deserialized.append(value)
            
            return deserialized
            
        except RedisError as e:
            logger.error(f"Redis error getting list range {key}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error getting Redis list range {key}: {e}")
            return []
    
    async def sadd(self, key: str, *members: Any) -> Optional[int]:
        """
        Add members to a set.
        
        Args:
            key: Redis key
            members: Set members
            
        Returns:
            Number of added members or None if error
        """
        try:
            client = await self.get_client()
            
            # Serialize members
            serialized_members = [
                json.dumps(member, default=str) if isinstance(member, (dict, list))
                else str(member)
                for member in members
            ]
            
            result = await client.sadd(key, *serialized_members)
            return result
            
        except RedisError as e:
            logger.error(f"Redis error adding to set {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error adding to Redis set {key}: {e}")
            return None
    
    async def smembers(self, key: str, deserialize: bool = True) -> set:
        """
        Get all members of a set.
        
        Args:
            key: Redis key
            deserialize: Whether to deserialize members
            
        Returns:
            Set of members
        """
        try:
            client = await self.get_client()
            members = await client.smembers(key)
            
            if not deserialize:
                return members
            
            # Deserialize members
            deserialized = set()
            for member in members:
                try:
                    deserialized.add(json.loads(member))
                except (json.JSONDecodeError, TypeError):
                    deserialized.add(member)
            
            return deserialized
            
        except RedisError as e:
            logger.error(f"Redis error getting set members {key}: {e}")
            return set()
        except Exception as e:
            logger.error(f"Error getting Redis set members {key}: {e}")
            return set()
    
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """
        Set field in a hash.
        
        Args:
            key: Redis key
            field: Hash field
            value: Field value
            
        Returns:
            True if successful
        """
        try:
            client = await self.get_client()
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            result = await client.hset(key, field, serialized_value)
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Redis error setting hash field {key}:{field}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting Redis hash field {key}:{field}: {e}")
            return False
    
    async def hget(
        self,
        key: str,
        field: str,
        deserialize: bool = True,
        default: Any = None
    ) -> Any:
        """
        Get field from a hash.
        
        Args:
            key: Redis key
            field: Hash field
            deserialize: Whether to deserialize value
            default: Default value if not found
            
        Returns:
            Field value or default
        """
        try:
            client = await self.get_client()
            value = await client.hget(key, field)
            
            if value is None:
                return default
            
            if deserialize:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            
            return value
            
        except RedisError as e:
            logger.error(f"Redis error getting hash field {key}:{field}: {e}")
            return default
        except Exception as e:
            logger.error(f"Error getting Redis hash field {key}:{field}: {e}")
            return default
    
    async def hgetall(self, key: str, deserialize: bool = True) -> Dict[str, Any]:
        """
        Get all fields from a hash.
        
        Args:
            key: Redis key
            deserialize: Whether to deserialize values
            
        Returns:
            Hash dictionary
        """
        try:
            client = await self.get_client()
            hash_dict = await client.hgetall(key)
            
            if not deserialize:
                return hash_dict
            
            # Deserialize values
            deserialized = {}
            for field, value in hash_dict.items():
                try:
                    deserialized[field] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    deserialized[field] = value
            
            return deserialized
            
        except RedisError as e:
            logger.error(f"Redis error getting hash {key}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error getting Redis hash {key}: {e}")
            return {}
    
    # Specialized methods for application use cases
    
    async def cache_search_results(
        self,
        query_hash: str,
        results: List[Dict[str, Any]],
        ttl: int = 300
    ) -> bool:
        """Cache search results."""
        key = f"search_results:{query_hash}"
        return await self.set(key, results, ttl=ttl)
    
    async def get_cached_search_results(
        self,
        query_hash: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results."""
        key = f"search_results:{query_hash}"
        return await self.get(key, default=None)
    
    async def cache_embeddings(
        self,
        content_hash: str,
        embedding: List[float],
        ttl: int = 86400
    ) -> bool:
        """Cache embeddings."""
        key = f"embedding:{content_hash}"
        # Use pickle for efficient float array storage
        try:
            client = await self.get_client()
            serialized = pickle.dumps(embedding)
            
            if ttl > 0:
                result = await client.setex(key, ttl, serialized)
            else:
                result = await client.set(key, serialized)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
            return False
    
    async def get_cached_embeddings(self, content_hash: str) -> Optional[List[float]]:
        """Get cached embeddings."""
        key = f"embedding:{content_hash}"
        try:
            client = await self.get_client()
            serialized = await client.get(key)
            
            if serialized:
                return pickle.loads(serialized)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached embedding: {e}")
            return None
    
    async def store_session_data(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """Store session data."""
        key = f"session:{session_id}"
        return await self.set(key, data, ttl=ttl)
    
    async def get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        key = f"session:{session_id}"
        return await self.get(key, default=None)
    
    async def store_rate_limit_data(
        self,
        client_id: str,
        endpoint: str,
        window: int,
        ttl: int = 60
    ) -> bool:
        """Store rate limit data."""
        key = f"rate_limit:{client_id}:{endpoint}:{window}"
        return await self.incr(key) and await self.expire(key, ttl)
    
    async def get_rate_limit_count(
        self,
        client_id: str,
        endpoint: str,
        window: int
    ) -> int:
        """Get rate limit count."""
        key = f"rate_limit:{client_id}:{endpoint}:{window}"
        count = await self.get(key, deserialize=False, default="0")
        return int(count)
    
    async def health_check(self) -> bool:
        """Check Redis health."""
        try:
            client = await self.get_client()
            result = await client.ping()
            return result
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """Get Redis server info."""
        try:
            client = await self.get_client()
            info = await client.info()
            return info
            
        except Exception as e:
            logger.error(f"Error getting Redis info: {e}")
            return {}
    
    async def flushdb(self) -> bool:
        """Flush current database (use with caution)."""
        try:
            client = await self.get_client()
            result = await client.flushdb()
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error flushing Redis database: {e}")
            return False
