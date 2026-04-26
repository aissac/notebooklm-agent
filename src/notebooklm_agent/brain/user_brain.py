"""UserBrain - One notebook per user, forever.

The core architectural shift: a user's brain is a single NotebookLM notebook
that grows smarter over time. No routing, no notebook selection, no scattering.

Key design:
- user_id -> notebook_id mapping persists in ~/.nlm-agent/brains.json
- Bootstrap runs ONCE on first message, never again
- Research sources are ephemeral: add -> answer -> prune
- Bootstrap source is PERMANENT: never auto-deleted
- Source cap at 40 (NotebookLM limit ~50, keep headroom)
"""

import json
import logging
import time
from pathlib import Path

from notebooklm_agent.brain.constants import MAX_SOURCES, BOOTSTRAP_TITLE, USER_SOURCE_PREFIX

logger = logging.getLogger(__name__)

BRAIN_STORE = Path.home() / ".nlm-agent" / "brains.json"
PRUNE_AFTER_USE = True  # Auto-prune research sources after answering


class UserBrain:
    """One brain per user. The notebook IS the memory.

    Manages the user_id -> notebook_id mapping, bootstrap lifecycle,
    and source pruning. Gateways should use this as their primary interface.
    """

    def __init__(self, client, store_path: Path = BRAIN_STORE):
        self.client = client
        self.store_path = store_path
        self._cache: dict[str, dict] = {}  # user_id -> brain info
        self._notebook_ids: dict[str, str] = {}  # user_id -> notebook_id
        self._load_store()

    def _load_store(self):
        """Load brain mapping from disk."""
        if self.store_path.exists():
            try:
                data = json.loads(self.store_path.read_text())
                self._cache = data.get("brains", {})
                # Build notebook_id lookup
                for uid, info in self._cache.items():
                    self._notebook_ids[uid] = info["notebook_id"]
                logger.info(f"Loaded {len(self._cache)} brain mappings from disk")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Corrupt brains.json, starting fresh: {e}")
                self._cache = {}
        else:
            self._cache = {}

    def _save_store(self):
        """Persist brain mapping to disk."""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": 1, "brains": self._cache}
        self.store_path.write_text(json.dumps(data, indent=2))

    async def get_or_create(self, user_id: str | int, title: str | None = None) -> dict:
        """Get existing brain info for user, or create + bootstrap a new one.

        Returns dict with notebook_id, title, ready status.
        Use brain.core.Brain for actual operations on the notebook.
        """
        uid = str(user_id)

        # Return cached info if we have it
        if uid in self._cache:
            return self._cache[uid]

        # Create new notebook
        nb_title = title or f"Agent: {uid}"
        logger.info(f"Creating new brain for user {uid}: '{nb_title}'")

        notebook = await self.client.notebooks.create(title=nb_title)
        notebook_id = notebook.id if hasattr(notebook, 'id') else str(notebook)

        # Bootstrap
        from notebooklm_agent.brain.bootstrap import BrainBootstrapper
        bootstrapper = BrainBootstrapper(self.client)
        await bootstrapper.bootstrap(notebook_id)

        # Persist mapping
        info = {
            "notebook_id": notebook_id,
            "title": nb_title,
            "created_at": time.time(),
        }
        self._cache[uid] = info
        self._notebook_ids[uid] = notebook_id
        self._save_store()

        logger.info(f"Brain created for user {uid}: {notebook_id}")
        return info

    def get_notebook_id(self, user_id: str | int) -> str | None:
        """Get the notebook_id for a user, or None if not created yet."""
        return self._notebook_ids.get(str(user_id))

    async def prune_research_sources(self, user_id: str | int) -> int:
        """Remove ephemeral research sources, keeping bootstrap and user-added ones.

        Returns the number of sources pruned.
        """
        uid = str(user_id)
        nb_id = self._notebook_ids.get(uid)
        if not nb_id:
            return 0

        sources = await self.client.sources.list(nb_id)
        pruned = 0
        for s in sources:
            title = getattr(s, "title", "")
            # NEVER delete the bootstrap source
            if title == BOOTSTRAP_TITLE:
                continue
            # NEVER delete sources starting with [USER] (manually added)
            if title.startswith(USER_SOURCE_PREFIX):
                continue
            if PRUNE_AFTER_USE:
                await self.client.sources.delete(nb_id, s.id)
                pruned += 1
                logger.debug(f"Pruned research source: {title}")

        logger.info(f"Pruned {pruned} research sources for user {uid}")
        return pruned

    async def enforce_source_cap(self, user_id: str | int) -> int:
        """If source count exceeds MAX_SOURCES, prune oldest research sources.

        Returns number of sources pruned.
        """
        uid = str(user_id)
        nb_id = self._notebook_ids.get(uid)
        if not nb_id:
            return 0

        sources = await self.client.sources.list(nb_id)
        if len(sources) <= MAX_SOURCES:
            return 0

        excess = len(sources) - MAX_SOURCES
        pruned = 0

        for s in sources:
            if pruned >= excess:
                break
            title = getattr(s, "title", "")
            # Protect bootstrap and user-added sources
            if title == BOOTSTRAP_TITLE or title.startswith(USER_SOURCE_PREFIX):
                continue
            try:
                await self.client.sources.delete(nb_id, s.id)
                pruned += 1
            except Exception as e:
                logger.warning(f"Failed to prune {title}: {e}")

        logger.info(f"Source cap enforced for {uid}: pruned {pruned} sources")
        return pruned

    async def delete_brain(self, user_id: str | int) -> bool:
        """Delete a user's brain (notebook) and remove from mapping.

        Used by /reset command to start fresh.
        """
        uid = str(user_id)

        if uid in self._cache:
            nb_id = self._cache[uid]["notebook_id"]
            try:
                await self.client.notebooks.delete(nb_id)
                logger.info(f"Deleted notebook {nb_id} for user {uid}")
            except Exception as e:
                logger.error(f"Failed to delete notebook {nb_id}: {e}")
                return False

            self._cache.pop(uid, None)
            self._notebook_ids.pop(uid, None)
            self._save_store()
            return True

        return False

    def list_brains(self) -> dict[str, dict]:
        """Return all brain mappings (user_id -> info)."""
        return dict(self._cache)
