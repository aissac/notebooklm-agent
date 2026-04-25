"""Brain - The core orchestrator that ties everything together.

One Brain = One NotebookLM notebook. The Brain manages:
- Bootstrap: Auto-seeds new notebooks with agent instructions
- Chat: Direct Q&A via chat.ask() (fast path)
- Research: Atomic start->poll->import->wait pipeline
- Artifacts: Podcast, report, quiz, mindmap generation
- Memory: Persistent facts stored as notebook sources

This is the primary interface for gateways (Telegram, CLI, etc.).

Usage:
    brain = Brain(client, notebook_id="abc123")
    await brain.ensure_ready()  # Bootstrap + auto-research if needed
    answer = await brain.ask("What is quantum computing?")
"""

import logging
from typing import Any

from notebooklm_agent.brain.bootstrap import BrainBootstrapper, MIN_SOURCES_FOR_READY
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
    - Memory: Persistent facts that survive across sessions

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
        nb = await client.notebooks.create(title)
        brain = cls(client, notebook_id=nb.id, title=title, **kwargs)
        return brain

    # ─── Properties ───

    @property
    def notebook_id(self) -> str | None:
        return self._notebook_id

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def chat(self) -> ChatSession:
        """Get or create the chat session."""
        if self._chat is None:
            self._chat = ChatSession(self.client, self._notebook_id)
        return self._chat

    @property
    def research(self) -> ResearchPipeline:
        """Get or create the research pipeline."""
        if self._research is None:
            self._research = ResearchPipeline(self.client)
        return self._research

    @property
    def artifact_generator(self) -> ArtifactGenerator:
        """Get or create the artifact generator."""
        if self._artifacts is None:
            self._artifacts = ArtifactGenerator(self.client)
        return self._artifacts

    # ─── Lifecycle ───

    async def ensure_ready(self, auto_topic: str | None = None) -> dict:
        """Ensure the brain is bootstrapped and has sources.

        If the notebook is empty (only has bootstrap source), this will:
        1. Upload bootstrap source if not present
        2. Auto-research the given topic if no sources exist
        3. Wait for sources to be processed

        Args:
            auto_topic: Topic to research if brain is empty.
                        If None and brain is empty, raises BrainNotReadyError.

        Returns:
            Dict with bootstrap status and source count.
        """
        if self._ready:
            return {"status": "already_ready", "source_count": "??"}

        # 1. Create notebook if needed
        if not self._notebook_id:
            nb = await self.client.notebooks.create(self._title)
            self._notebook_id = nb.id
            logger.info(f"Created notebook: {self._notebook_id}")

        # 2. Bootstrap
        bootstrapper = BrainBootstrapper(self.client)
        await bootstrapper.bootstrap(self._notebook_id)
        logger.info(f"Bootstrapped notebook: {self._notebook_id}")

        # 3. Check source count
        source_count = await bootstrapper.source_count(self._notebook_id)
        if source_count >= MIN_SOURCES_FOR_READY - 1:  # -1 because bootstrap counts as 1
            self._ready = True
            return {"status": "ready", "source_count": source_count}

        # 4. Auto-research if topic provided
        if auto_topic:
            logger.info(f"Auto-researching: {auto_topic}")
            result = await self.research.brain_research(
                self._notebook_id, auto_topic, mode="fast"
            )

            if result.success and result.source_count > 0:
                self._ready = True
                return {
                    "status": "ready",
                    "source_count": result.source_count,
                    "research_task_id": result.task_id,
                }
            elif result.success and result.source_count == 0:
                self._ready = True  # Brain has bootstrap, just no web sources
                return {"status": "ready_no_sources", "source_count": 0}
            else:
                return {"status": "research_failed", "error": result.error}

        # No topic provided and brain is empty
        self._ready = True  # Bootstrap exists, but brain will give limited answers
        return {"status": "bootstrapped_no_sources", "source_count": source_count}

    # ─── Core Operations ───

    async def ask(self, question: str, auto_research: bool = True) -> str:
        """Ask a question to the brain.

        Args:
            question: The question to ask.
            auto_research: If True and brain lacks sources, auto-research the topic.

        Returns:
            The answer text with citations.
        """
        # Auto-ensure ready
        if not self._ready:
            await self.ensure_ready(auto_topic=question if auto_research else None)

        # Set memory prefix
        self.chat.set_memory(self._facts)
        return await self.chat.ask(question)

    async def research_topic(self, query: str, mode: str = "fast") -> ResearchResult:
        """Research a topic and add sources to the brain.

        Args:
            query: Research question.
            mode: "fast" (2 min) or "deep" (15 min).

        Returns:
            ResearchResult with success status and source count.
        """
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID. Call ensure_ready() or create() first.")

        return await self.research.brain_research(self._notebook_id, query, mode=mode)

    async def podcast(self, instructions: str | None = None) -> ArtifactResult:
        """Generate a podcast from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        return await self.artifact_generator.generate_podcast(self._notebook_id, instructions)

    async def report(self, custom_prompt: str | None = None) -> ArtifactResult:
        """Generate a structured report from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        return await self.artifact_generator.generate_report(self._notebook_id, custom_prompt)

    async def quiz(self, instructions: str | None = None) -> ArtifactResult:
        """Generate a quiz from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        return await self.artifact_generator.generate_quiz(self._notebook_id, instructions)

    async def mindmap(self) -> ArtifactResult:
        """Generate a mind map from the brain's sources."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        return await self.artifact_generator.generate_mindmap(self._notebook_id)

    async def add_source(self, url: str, title: str | None = None) -> str:
        """Add a URL as a source to the brain."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        source = await self.client.sources.add_url(
            self._notebook_id, url, title=title if title else None
        )
        # Wait for source to be processed
        await self.client.sources.wait_until_ready(
            self._notebook_id, source.id, timeout=60.0
        )
        name = title or url
        return f"Added source: {name} (id: {source.id})"

    async def add_text_source(self, title: str, content: str) -> str:
        """Add a text source to the brain."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        source = await self.client.sources.add_text(
            self._notebook_id, title=title, content=content
        )
        await self.client.sources.wait_until_ready(
            self._notebook_id, source.id, timeout=60.0
        )
        return f"Added text source: {title} (id: {source.id})"

    async def list_sources(self) -> str:
        """List all sources in the brain's notebook."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        sources = await self.client.sources.list(self._notebook_id)
        if not sources:
            return "No sources in this notebook yet."
        lines = [f"Sources in notebook ({len(sources)} total):"]
        for s in sources:
            name = getattr(s, "title", None) or getattr(s, "name", str(s.id))
            lines.append(f"  - {name}")
        return "\n".join(lines)

    # ─── Memory ───

    def add_fact(self, fact: str) -> None:
        """Add a fact to the brain's local memory.

        Facts are prepended to chat queries for context persistence.
        They can also be synced to the notebook as a source.
        """
        self._facts.append(fact)
        # Keep only last 20 facts
        if len(self._facts) > 20:
            self._facts = self._facts[-20:]

    async def sync_memory(self) -> None:
        """Sync local facts to the notebook as a text source.

        This makes facts available to future conversations even
        after the memory prefix is exceeded.
        """
        if not self._notebook_id or not self._facts:
            return
        content = "\n".join(f"- {f}" for f in self._facts)
        await self.client.sources.add_text(
            self._notebook_id,
            title="Agent Memory",
            content=content,
        )

    # ─── Notebook Management ───

    async def delete_notebook(self) -> bool:
        """Delete the brain's notebook. Irreversible."""
        if not self._notebook_id:
            return False
        result = await self.client.notebooks.delete(self._notebook_id)
        self._notebook_id = None
        self._ready = False
        return result

    async def rename(self, new_title: str) -> None:
        """Rename the brain's notebook."""
        if not self._notebook_id:
            raise BrainNotReadyError("No notebook ID.")
        await self.client.notebooks.rename(self._notebook_id, new_title)
        self._title = new_title

    def __repr__(self) -> str:
        state = "ready" if self._ready else "not_ready"
        nb = self._notebook_id[:8] if self._notebook_id else "none"
        return f"Brain(nb={nb}..., state={state}, facts={len(self._facts)})"
