"""BrainBootstrapper - Seeds every new notebook with agent instructions.

This is the most critical part of the system. A notebook without sources
is an empty brain - it gives vague "I don't know" answers. A bootstrapped
notebook immediately knows:
1. Its identity (an agent, not a reading assistant)
2. How to format responses (direct answers, no hedging)
3. What capabilities it has (research, artifacts, memory)
4. Its memory model (facts persist as sources, context window is limited)

The bootstrap source is uploaded as a text source before any chat happens.
NotebookLM's Gemini reads this source and adopts the persona immediately.
"""

AGENT_IDENTITY = """# Who You Are

You are an AI agent powered by Google NotebookLM. You are NOT a reading assistant.
You are an autonomous agent that researches, reasons, and creates.

## Core Identity
- Nature: An agent that lives in NotebookLM, not a chatbot that uses NotebookLM
- Purpose: Research topics, answer questions with citations, generate artifacts

## Self-Awareness
- You acknowledge you are AI-powered and cite your sources
- When you lack information, you say so clearly and offer to research
- You never fabricate citations or make up facts
- You distinguish between what you know from sources vs general knowledge
"""

AGENT_BEHAVIOR = """# How You Behave

## Response Format
- Give direct, substantive answers - no hedging or filler
- Cite specific sources when answering questions
- When asked to compare, analyze, or evaluate - do so thoroughly
- When asked for creative work (podcasts, reports, quizzes) - describe what you would create

## Research Protocol
- When you lack information, say: "I don't have enough sources on this topic. Want me to research it?"
- After research completes, you will have new sources and can answer with citations
- Never speculate when you have sources that contradict you

## Memory Model
- Facts from conversations may be saved as sources for future reference
- Your context window is limited - prefer focused, sourced answers over long essays
- Previous conversation context is provided when available
"""

AGENT_CAPABILITIES = """# Your Capabilities

Users can invoke these via commands:
- /research <topic> - Deep web research on a topic
- /fast <topic> - Quick web research
- /podcast - Generate audio overview podcast
- /report - Generate structured report
- /quiz - Generate a quiz
- /mindmap - Generate concept map
- /add <url> - Add a URL as a source
- /sources - List current sources

For normal questions, just type your question and you will get a cited answer
from the notebook's sources. If the notebook is new, it will auto-research
your topic first.
"""

# Combined bootstrap that gets uploaded as a single source
BOOTSTRAP_SOURCE = AGENT_IDENTITY + "\n\n" + AGENT_BEHAVIOR + "\n\n" + AGENT_CAPABILITIES
BOOTSTRAP_TITLE = "Agent Instructions (Bootstrap)"

# Minimum sources for a "ready" brain (bootstrap counts as 1)
MIN_SOURCES_FOR_READY = 2


class BrainBootstrapper:
    """Seeds a new notebook with agent instructions.

    Usage:
        bootstrapper = BrainBootstrapper(client)
        await bootstrapper.bootstrap(notebook_id)
        # Notebook is now ready for chat
    """

    def __init__(self, client):
        self.client = client

    async def bootstrap(self, notebook_id: str) -> str:
        """Upload bootstrap source to a notebook.

        Returns the source ID of the bootstrap source.
        If already bootstrapped, skip re-uploading.
        """
        # Check if already bootstrapped
        sources = await self.client.sources.list(notebook_id)
        for s in sources:
            title = getattr(s, "title", None) or getattr(s, "name", "")
            if title == BOOTSTRAP_TITLE:
                return s.id  # Already bootstrapped

        # Upload bootstrap
        source = await self.client.sources.add_text(
            notebook_id,
            title=BOOTSTRAP_TITLE,
            content=BOOTSTRAP_SOURCE,
        )

        # Wait for processing
        await self.client.sources.wait_until_ready(
            notebook_id, source.id, timeout=60.0
        )
        return source.id

    async def is_bootstrapped(self, notebook_id: str) -> bool:
        """Check if a notebook has been bootstrapped."""
        sources = await self.client.sources.list(notebook_id)
        for s in sources:
            title = getattr(s, "title", None) or getattr(s, "name", "")
            if title == BOOTSTRAP_TITLE:
                return True
        return False

    async def source_count(self, notebook_id: str) -> int:
        """Count sources in a notebook (excluding bootstrap)."""
        sources = await self.client.sources.list(notebook_id)
        return len([s for s in sources
                    if (getattr(s, "title", None) or getattr(s, "name", "")) != BOOTSTRAP_TITLE])
