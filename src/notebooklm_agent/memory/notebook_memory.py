"""NotebookMemory - Persistent memory backed by NotebookLM sources.

Key insight from SmallClawLM: AgentMemory was a sliding window that forgot
everything. NLMMemory stored everything in a notebook but had complex sync
logic. NotebookMemory simplifies: it stores facts as text sources in the
notebook AND keeps a local cache for fast prefix injection.

The notebook IS the memory. Facts become sources. Research becomes context.
The brain grows smarter over time without any external database.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR = Path.home() / ".nlm-agent" / "memory"
DEFAULT_MAX_FACTS = 30
DEFAULT_MAX_CHARS = 2000


class NotebookMemory:
    """Persistent memory backed by NotebookLM sources.

    Two-tier storage:
    1. Local cache: list of recent facts (fast prefix injection for chat)
    2. Notebook sources: all facts ever recorded (persistent, searchable)

    Usage:
        mem = NotebookMemory(client, notebook_id)
        mem.add("User prefers concise answers")
        mem.add("Research on fusion energy: 5 sources added")

        # Get prefix for chat queries
        prefix = mem.render_prefix()

        # Facts survive across sessions because they are notebook sources
    """

    def __init__(
        self,
        client=None,
        notebook_id: str | None = None,
        max_facts: int = DEFAULT_MAX_FACTS,
        max_chars: int = DEFAULT_MAX_CHARS,
        persist_path: Path | None = None,
    ):
        self.client = client
        self.notebook_id = notebook_id
        self.max_facts = max_facts
        self.max_chars = max_chars
        self._facts: list[str] = []
        self._persist_path = persist_path or MEMORY_DIR / "local_cache.json"
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # Load local cache from disk
        self._load_local()

    def add(self, fact: str) -> None:
        """Add a fact to memory.

        Stored in local cache and optionally synced to notebook.
        """
        timestamp = time.strftime("%H:%M")
        entry = f"[{timestamp}] {fact}"
        self._facts.append(entry)
        self._prune()
        self._persist_local()

        # Non-blocking sync to notebook
        if self.client and self.notebook_id:
            try:
                self._sync_to_notebook(entry)
            except Exception as e:
                logger.debug(f"Notebook sync failed (non-fatal): {e}")

    def add_observation(self, tool: str, result: str, max_len: int = 200) -> None:
        """Add a tool observation (auto-truncated)."""
        truncated = result[:max_len] + "..." if len(result) > max_len else result
        self.add(f"{tool} -> {truncated}")

    def add_decision(self, thought: str, action: str) -> None:
        """Add an agent decision."""
        short = thought[:80] + "..." if len(thought) > 80 else thought
        self.add(f"Decided: {short} -> {action}")

    def render_prefix(self) -> str:
        """Render memory as prefix for chat queries.

        Returns empty string if no facts. Otherwise returns
        formatted prefix with recent facts.
        """
        if not self._facts:
            return ""
        header = "[AGENT MEMORY]\n"
        body = "\n".join(self._facts)
        return f"{header}{body}\n[END MEMORY]\n\n"

    def clear(self) -> None:
        """Clear local cache (does not delete notebook sources)."""
        self._facts.clear()
        self._persist_local()

    @property
    def facts(self) -> list[str]:
        return self._facts.copy()

    @property
    def fact_count(self) -> int:
        return len(self._facts)

    async def sync_to_notebook(self) -> None:
        """Sync all local facts to notebook as a text source.

        This makes facts available to future sessions even
        without local cache.
        """
        if not self.client or not self.notebook_id or not self._facts:
            return

        content = "\n".join(f"- {f}" for f in self._facts)
        await self.client.sources.add_text(
            self.notebook_id,
            title="Agent Memory Snapshot",
            content=content,
        )

    async def query(self, question: str) -> str:
        """Query the notebook for a memory-based answer."""
        if not self.client or not self.notebook_id:
            return "Memory not available (no notebook connection)"

        result = await self.client.chat.ask(self.notebook_id, question)
        return result.answer if hasattr(result, "answer") else str(result)

    # ─── Private ───

    def _sync_to_notebook(self, entry: str):
        """Fire-and-forget sync. Creates a task on the running event loop."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._add_source_async(entry))
        except RuntimeError:
            # No running loop, skip sync
            logger.debug("No event loop for notebook sync")

    async def _add_source_async(self, entry: str) -> None:
        try:
            await self.client.sources.add_text(
                self.notebook_id,
                title=f"Memory {time.strftime('%H:%M:%S')}",
                content=entry,
            )
        except Exception as e:
            logger.debug(f"Source sync failed: {e}")

    def _prune(self) -> None:
        """Keep memory within budget."""
        while len(self._facts) > self.max_facts:
            self._facts.pop(0)
        while len(self.render_prefix()) > self.max_chars and len(self._facts) > 5:
            self._facts.pop(0)

    def _persist_local(self) -> None:
        """Save local cache to disk."""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps({"facts": self._facts}, indent=2))
        except Exception as e:
            logger.debug(f"Local persist failed: {e}")

    def _load_local(self) -> None:
        """Load local cache from disk."""
        try:
            if self._persist_path.exists():
                data = json.loads(self._persist_path.read_text())
                self._facts = data.get("facts", [])
        except Exception as e:
            logger.debug(f"Local load failed: {e}")
            self._facts = []

    def __repr__(self) -> str:
        return f"NotebookMemory(facts={len(self._facts)}, chars={len(self.render_prefix())})"
