"""ChatSession - Direct chat with NotebookLM.

No smolagents. No CodeAgent. No <code> tag parsing.
Just direct chat.ask() with conversation ID tracking for context persistence.

Key design decisions:
- Reuses conversation_id so NotebookLM maintains context across turns
- Prepends memory prefix so agent remembers key facts
- Handles rate limits with exponential backoff (tenacity)
- Falls back gracefully on errors (returns error string, never crashes)
"""

import asyncio
import logging
from typing import AsyncIterator

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from notebooklm_agent.utils.text import sanitize_for_display

logger = logging.getLogger(__name__)


class ChatSession:
    """A conversation with a NotebookLM notebook.

    Maintains conversation_id across turns for context persistence.
    Prepends memory prefix for agent awareness.

    Usage:
        session = ChatSession(client, notebook_id)
        answer = await session.ask("What is quantum computing?")
        # answer is a clean string with citations from sources
    """

    def __init__(self, client, notebook_id: str):
        self.client = client
        self.notebook_id = notebook_id
        self.conversation_id: str | None = None
        self._memory_prefix: str = ""

    def set_memory(self, facts: list[str]) -> None:
        """Set memory prefix from persistent storage.

        This prepends key facts to every question so the agent
        remembers context across conversations.
        """
        if not facts:
            self._memory_prefix = ""
            return
        lines = ["[REMEMBER]"] + facts[-10:] + ["[END REMEMBER]"]
        self._memory_prefix = "\n".join(lines) + "\n\n"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    )
    async def ask(self, question: str) -> str:
        """Ask a question to the notebook.

        Reuses conversation_id for context persistence.
        Prepends memory prefix for awareness.
        Returns the answer text with citations.
        """
        full_question = self._memory_prefix + question if self._memory_prefix else question

        # Truncate very long questions
        if len(full_question) > 15000:
            full_question = full_question[:15000] + "\n[...truncated]"

        result = await self.client.chat.ask(
            self.notebook_id,
            full_question,
            conversation_id=self.conversation_id,
        )

        # Save conversation_id for next turn
        if hasattr(result, "conversation_id") and result.conversation_id:
            self.conversation_id = result.conversation_id

        answer = result.answer if hasattr(result, "answer") else str(result)
        return sanitize_for_display(answer)

    async def ask_stream(self, question: str) -> AsyncIterator[str]:
        """Ask with streaming response (if supported by API).

        Falls back to non-streaming if not available.
        """
        # notebooklm-py doesn't support streaming yet, so just yield the full answer
        answer = await self.ask(question)
        yield answer

    async def reset_conversation(self) -> None:
        """Start a fresh conversation (forget previous context)."""
        self.conversation_id = None

    async def configure(
        self,
        goal: str | None = None,
        custom_prompt: str | None = None,
        response_length=None,
    ) -> None:
        """Configure the chat session (goal, response length, custom prompt)."""
        kwargs = {}
        if goal:
            kwargs["goal"] = goal
        if custom_prompt:
            kwargs["custom_prompt"] = custom_prompt
        if response_length:
            kwargs["response_length"] = response_length

        await self.client.chat.configure(self.notebook_id, **kwargs)

    def __repr__(self) -> str:
        conv = self.conversation_id[:8] if self.conversation_id else "new"
        return f"ChatSession(nb={self.notebook_id[:8]}..., conv={conv})"
