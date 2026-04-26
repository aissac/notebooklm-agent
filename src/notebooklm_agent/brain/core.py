"""Brain - The core orchestrator. One Brain = One NotebookLM notebook.

The Brain manages:
- Bootstrap: Auto-seeds new notebooks with agent instructions
- Chat: Direct Q&A via chat.ask() (fast path, 90% of queries)
- Research: Atomic start->poll->import->wait pipeline with source cap
- Artifacts: Podcast, report, quiz, mindmap generation
- Memory: Persistent facts stored as notebook sources

This is the primary interface for gateways (Telegram, CLI, etc.).
One user = one brain = one notebook. Forever.

Usage:
    brain = Brain(client, notebook_id="abc123")
    await brain.ensure_ready()  # Bootstrap if needed
    answer = await brain.ask("What is quantum computing?")
"""

import logging
from typing import Any

from notebooklm_agent.brain.bootstrap import BrainBootstrapper, MIN_SOURCES_FOR_READY
from notebooklm_agent.brain.constants import BOOTSTRAP_TITLE, MAX_SOURCES
from notebooklm_agent.brain.chat import ChatSession
from notebooklm_agent.brain.research import ResearchPipeline, ResearchMode, ResearchResult
from notebooklm_agent.brain.artifacts import ArtifactGenerator, ArtifactResult, ArtifactType

logger = logging.getLogger(__name__)


class BrainError(Exception):
    """Base exception for Brain operations."""
    pass


class BrainNotReadyError(BrainError):
    """Raised when trying to chat with an empty brain."""
    pass


class Brain:
    """One Brain = One NotebookLM notebook.

    The Brain is the central orchestrator. It manages the full lifecycle:
    - Auto-bootstrap: Seeds agent instructions before first use
    - Auto-research: If brain is empty, researches the user's topic before answering
    - Chat: Direct Q&A with conversation context
    - Artifacts: Generate podcasts, reports, etc.
    - Source lifecycle: Enforce cap, protect bootstrap, prune stale research

    Usage:
        # Create a new brain (auto-creates notebook)
        brain = await Brain.create(client, title="Physics Research")

        # Or attach to existing notebook
        brain = Brain(client, notebook_id="abc123")

        # Ensure it's bootstrapped and ready
        await brain.ensure_ready()

        # Ask questions
        answer = await brain.ask("Explain quantum entanglement")

        # Generate artifacts
        result = await brain.podcast()
    """

    def __init__(self, client, notebook_id: str | None = None, title: str = "NotebookLM Agent"):
        self.client = client
        self._notebook_id = notebook_id
        self._title = title
        self._ready = False

        # Sub-components (lazy-initialized)
        self._bootstrapper: BrainBootstrapper | None = None
        self._chat: ChatSession | None = None
        self._research: ResearchPipeline | None = None
        self._artifacts: ArtifactGenerator | None = None

        # Memory (simple fact list, synced to notebook)
        self._facts: list[str] = []

    @classmethod
    async def create(cls, client, title: str = "NotebookLM Agent", **kwargs) -> "Brain":
        """Create a new notebook and return a Brain attached to it.

        This is the preferred way to create a Brain from scratch.
        """
        notebook = await client.notebooks.create(title=title)
        notebook_id = notebook.id if hasattr(notebook, 'id') else str(notebook)
        logger.info(f"Created notebook: {title} ({notebook_id})")

        brain = cls(client, notebook_id=notebook_id, title=title)
        await brain.bootstrap()
        return brain

    @property
    def notebook_id(self) -> str | None:
        """The notebook this brain is attached to."""
        return self._notebook_id

    @property
    def is_ready(self) -> bool:
        """Whether bootstrap has completed and brain is usable."""
        return self._ready

    # ─── Lifecycle ───

    async def ensure_ready(self) -> None:
        """Ensure brain is bootstrapped and ready for chat.

        Idempotent — safe to call multiple times.
        """
        if self._ready:
            return

        # Check if bootstrap source exists
        try:
            sources = await self.client.sources.list(self._notebook_id)
            bootstrap_found = any(
                getattr(s, 'title', '') == BOOTSTRAP_TITLE
                for s in sources
            )
            if bootstrap_found:
                logger.info(f"Brain already bootstrapped ({len(sources)} sources)")
                self._ready = True
                return
        except Exception as e:
            logger.warning(f"Failed to check sources: {e}")

        # Need to bootstrap
        await self.bootstrap()

    async def bootstrap(self) -> None:
        """Bootstrap the brain with agent instructions.

        Uploads the identity, behavior, and capabilities sources.
        If already bootstrapped, does nothing.
        """
        if not self._bootstrapper:
            self._bootstrapper = BrainBootstrapper(self.client)

        await self._bootstrapper.bootstrap(self._notebook_id)
        self._ready = True
        logger.info(f"Brain bootstrapped: {self._notebook_id}")

    # ─── Chat ───

    async def ask(self, question: str, context: str | None = None) -> str:
        """Ask a question using the brain's sources.

        This is the FAST PATH — direct chat.ask() for 90% of queries.
        No research, no tools, just Gemini reasoning over sources.

        Args:
            question: The user's question
            context: Optional additional context to prepend

        Returns:
            The answer text with citations
        """
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")

        if not self._chat:
            self._chat = ChatSession(self.client, self._notebook_id)
            self._chat.set_memory(self._facts)

        prompt = question
        if context:
            prompt = f"{context}\n\n{question}"

        return await self._chat.ask(prompt)

    # ─── Research ───

    async def research(self, query: str, mode: str = "fast") -> ResearchResult:
        """Research a topic and import sources into the brain.

        Automatically enforces source cap after import.

        Args:
            query: Research question
            mode: "fast" (90s) or "deep" (15min)

        Returns:
            ResearchResult with source count
        """
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")

        if not self._research:
            self._research = ResearchPipeline(self.client)

        result = await self._research.brain_research(
            self._notebook_id, query, mode=mode
        )

        # Enforce source cap after research
        if result.success:
            await self._enforce_cap()

        return result

    async def add_source(self, url: str, title: str | None = None) -> str:
        """Add a URL source to the brain.

        User-added sources get [USER] prefix for protection from auto-pruning.
        """
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")

        source = await self.client.sources.add_url(
            self._notebook_id, url
        )

        # Rename with [USER] prefix for protection from pruning
        if title:
            try:
                await self.client.sources.rename(
                    self._notebook_id, source.id,
                    f"[USER] {title}"
                )
            except Exception:
                pass  # rename is best-effort

        # Wait for source to be ready
        await self.client.sources.wait_until_ready(self._notebook_id, source.id)

        # Enforce source cap
        await self._enforce_cap()

        return f"Added: {title or url}"

    async def add_text(self, title: str, content: str) -> str:
        """Add a text source to the brain.

        Text sources get [USER] prefix for protection.
        """
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")

        protected_title = f"[USER] {title}" if not title.startswith("[USER]") else title
        source = await self.client.sources.add_text(
            self._notebook_id, content, title=protected_title
        )

        return f"Added: {protected_title}"

    # ─── Source Management ───

    async def list_sources(self) -> list[dict]:
        """List all sources in the brain."""
        if not self._notebook_id:
            return []
        sources = await self.client.sources.list(self._notebook_id)
        return [
            {
                "id": s.id,
                "title": getattr(s, 'title', 'Untitled'),
                "protected": (
                    getattr(s, 'title', '') == BOOTSTRAP_TITLE
                    or getattr(s, 'title', '').startswith("[USER]")
                )
            }
            for s in sources
        ]

    async def source_count(self) -> int:
        """Count sources in the brain."""
        if not self._notebook_id:
            return 0
        sources = await self.client.sources.list(self._notebook_id)
        return len(sources)

    async def _enforce_cap(self) -> int:
        """Enforce source cap by pruning research sources.

        Protected sources (bootstrap + [USER] prefix) are never pruned.
        """
        if not self._notebook_id:
            return 0

        sources = await self.client.sources.list(self._notebook_id)
        if len(sources) <= MAX_SOURCES:
            return 0

        excess = len(sources) - MAX_SOURCES
        pruned = 0

        for s in sources:
            if pruned >= excess:
                break
            title = getattr(s, 'title', '')
            # Protect bootstrap and user-added sources
            if title == BOOTSTRAP_TITLE or title.startswith("[USER]"):
                continue
            try:
                await self.client.sources.delete(self._notebook_id, s.id)
                pruned += 1
                logger.debug(f"Pruned: {title}")
            except Exception as e:
                logger.warning(f"Failed to prune {title}: {e}")

        if pruned:
            logger.info(f"Source cap enforced: pruned {pruned} sources (was {len(sources)})")
        return pruned

    # ─── Artifacts ───

    async def podcast(self, instructions: str | None = None) -> ArtifactResult:
        """Generate a podcast from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")
        if not self._artifacts:
            self._artifacts = ArtifactGenerator(self.client)
        return await self._artifacts.generate_podcast(self._notebook_id, instructions)

    async def report(self, prompt: str | None = None) -> ArtifactResult:
        """Generate a structured report from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")
        if not self._artifacts:
            self._artifacts = ArtifactGenerator(self.client)
        return await self._artifacts.generate_report(self._notebook_id, prompt)

    async def quiz(self, topic: str | None = None) -> ArtifactResult:
        """Generate a quiz from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")
        if not self._artifacts:
            self._artifacts = ArtifactGenerator(self.client)
        return await self._artifacts.generate_quiz(self._notebook_id, topic)

    async def mindmap(self) -> ArtifactResult:
        """Generate a mind map from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")
        if not self._artifacts:
            self._artifacts = ArtifactGenerator(self.client)
        return await self._artifacts.generate_mindmap(self._notebook_id)

    async def video(self, instructions: str | None = None) -> ArtifactResult:
        """Generate an explainer video from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("Brain has no notebook_id")
        if not self._artifacts:
            self._artifacts = ArtifactGenerator(self.client)
        return await self._artifacts.generate_video(self._notebook_id, instructions)
