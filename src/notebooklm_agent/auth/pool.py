"""AuthPool - Cached auth tokens with context-managed client pool.

Design:
- Auth tokens cached in memory with TTL (avoids disk reads every call)
- Client pool manages NotebookLMClient lifecycle (auto-reconnect, refresh)
- All async-native (no sync/async bridge)
- Thread-safe for use across multiple gateways
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Cache TTL: 30 minutes (NotebookLM sessions last a while)
AUTH_CACHE_TTL = 1800

# Default auth storage paths
DEFAULT_STORAGE = Path.home() / ".nlm-agent" / "auth.json"
FALLBACK_STORAGE = None  # Will use notebooklm-py's default


class AuthPool:
    """Singleton auth pool with cached tokens and client management.

    Usage:
        pool = AuthPool()
        auth = await pool.get_auth()
        client = await pool.get_client()

        # Or as context manager:
        async with pool.client() as client:
            notebooks = await client.notebooks.list()
    """

    _instance: "AuthPool | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "AuthPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._auth = None
        self._auth_timestamp: float = 0
        self._clients: list = []  # Active clients for cleanup
        self._client: object | None = None  # Primary long-lived client

    async def get_auth(self, force_refresh: bool = False):
        """Get cached auth tokens.

        Reads from disk only on first call or after TTL expires.
        """
        now = time.time()

        if not force_refresh and self._auth and (now - self._auth_timestamp) < AUTH_CACHE_TTL:
            return self._auth

        # Try SmallClawLM's own auth file
        if DEFAULT_STORAGE.exists():
            try:
                from notebooklm.auth import AuthTokens
                self._auth = await AuthTokens.from_storage(DEFAULT_STORAGE)
                self._auth_timestamp = now
                logger.info("Loaded auth from nlm-agent storage")
                return self._auth
            except Exception as e:
                logger.warning(f"Failed to load nlm-agent auth: {e}")

        # Fallback: notebooklm-py's default storage
        try:
            from notebooklm.auth import AuthTokens
            self._auth = await AuthTokens.from_storage()
            self._auth_timestamp = now
            logger.info("Loaded auth from notebooklm-py storage")
            return self._auth
        except Exception as e:
            raise RuntimeError(
                "No NotebookLM authentication found.\n"
                "Run: nlm-agent login  (or: notebooklm login)"
            ) from e

    async def get_client(self):
        """Get a long-lived NotebookLM client.

        The client is created once and reused across all operations.
        Auto-refreshes auth if the existing client is stale.
        """
        if self._client is not None:
            # Check if client is still usable
            if hasattr(self._client, "is_connected") and await self._client.is_connected():
                return self._client
            # Client is stale, create new one
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None

        auth = await self.get_auth()
        from notebooklm import NotebookLMClient
        self._client = NotebookLMClient(auth)
        await self._client.__aenter__()
        logger.info("Created new NotebookLM client")
        return self._client

    @asynccontextmanager
    async def client(self):
        """Context manager for getting a client. Use for short-lived operations."""
        client = await self.get_client()
        try:
            yield client
        except Exception as e:
            # On auth errors, force refresh and retry once
            if "auth" in str(e).lower() or "unauthorized" in str(e).lower():
                self._auth = None
                self._auth_timestamp = 0
                self._client = None
                client = await self.get_client()
                yield client
            else:
                raise

    async def close(self):
        """Close all managed clients."""
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None

        for client in self._clients:
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        self._clients.clear()

    def clear_cache(self):
        """Force auth refresh on next call."""
        self._auth = None
        self._auth_timestamp = 0


# ─── Module-level convenience functions ───

_pool = AuthPool()


async def get_auth(force_refresh: bool = False):
    """Get cached auth tokens. Module-level shortcut."""
    return await _pool.get_auth(force_refresh)


async def get_client():
    """Get a managed NotebookLM client. Module-level shortcut."""
    return await _pool.get_client()


async def close_pool():
    """Close the global auth pool. Call on shutdown."""
    await _pool.close()
