"""Authentication module - Cached auth with client pool.

Key problems solved from SmallClawLM:
- No disk reads on every API call (singleton cache with TTL)
- No manual __aenter__/__aexit__ management (client pool)
- No sync/async bridge hacks (everything is async)
- Auth auto-refresh when tokens expire
"""

from notebooklm_agent.auth.pool import AuthPool, get_auth, get_client, close_pool

__all__ = ["AuthPool", "get_auth", "get_client", "close_pool"]
