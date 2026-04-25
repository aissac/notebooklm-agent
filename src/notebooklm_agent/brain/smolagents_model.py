"""NLMModel - smolagents Model backed by NotebookLM chat.

This is the bridge that lets smolagents CodeAgent use NotebookLM's
Gemini as its reasoning engine. It wraps chat.ask() in the Model interface.

Only loaded when the user explicitly requests smolagents mode.
Not a core dependency.

Design from SmallClawLM lessons:
- Stateless: no conversation_id reuse (avoids sync drift)
- Daemon thread event loop for async/sync bridge (smolagents is sync)
- Error -> string conversion so agent can self-correct
- Handles list content in smolagents messages
- Truncates long prompts to 10K chars for NotebookLM limits
"""

import asyncio
import logging
import threading
from typing import Any

try:
    from smolagents.models import ChatMessage, MessageRole, Model
    from smolagents.monitoring import TokenUsage
    HAS_SMOLAGENTS = True
except ImportError:
    HAS_SMOLAGENTS = False
    # Create stub classes so the file can still be imported
    ChatMessage = None
    MessageRole = None
    Model = object
    TokenUsage = None

from notebooklm_agent.auth import get_auth

logger = logging.getLogger(__name__)


if HAS_SMOLAGENTS:

    class NLMModel(Model):
        """smolagents Model backed by NotebookLM chat API.

        Usage (with smolagents):
            model = NLMModel(notebook_id="abc123")
            agent = CodeAgent(model=model, tools=[...])
            result = agent.run("Research fusion energy")
        """

        def __init__(
            self,
            notebook_id: str | None = None,
            notebook_title: str | None = None,
            auto_create: bool = True,
            model_id: str = "notebooklm-chat",
            **kwargs,
        ):
            super().__init__(model_id=model_id, **kwargs)
            self._notebook_id = notebook_id
            self._notebook_title = notebook_title or "NotebookLM Agent"
            self._auto_create = auto_create
            self._client = None
            self._loop = None
            self._thread = None

        def _ensure_loop(self):
            """Daemon event loop for async/sync bridge."""
            if self._loop is None or not self._loop.is_running():
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
                self._thread.start()
            return self._loop

        async def _ensure_notebook(self):
            if self._notebook_id:
                return self._notebook_id
            if not self._auto_create:
                raise RuntimeError("No notebook ID and auto_create=False")
            client = await self._get_client()
            nb = await client.notebooks.create(self._notebook_title)
            self._notebook_id = nb.id
            return self._notebook_id

        async def _get_client(self):
            if self._client is None:
                auth = await get_auth()
                from notebooklm import NotebookLMClient
                self._client = NotebookLMClient(auth)
                await self._client.__aenter__()
            return self._client

        async def _chat(self, prompt: str) -> str:
            nb_id = await self._ensure_notebook()
            client = await self._get_client()
            try:
                result = await client.chat.ask(nb_id, prompt)
                return result.answer if hasattr(result, "answer") else str(result)
            except Exception as e:
                logger.error(f"Chat error: {e}")
                return f"[ERROR] {e}"

        def generate(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            tools_to_call_from: list | None = None,
            **kwargs,
        ) -> ChatMessage:
            """Generate response using NotebookLM chat.

            Flattens smolagents message history into a single prompt,
            sends to NotebookLM, returns response for CodeAgent parsing.
            """
            prompt_parts = []
            for msg in messages:
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(str(item) for item in content)
                content = str(content) if content else ""
                if not content:
                    continue
                if msg.role == MessageRole.SYSTEM:
                    prompt_parts.append(f"[System] {content}")
                elif msg.role == MessageRole.USER:
                    prompt_parts.append(f"[User] {content}")
                elif msg.role == MessageRole.ASSISTANT:
                    truncated = content[:500] if len(content) > 500 else content
                    prompt_parts.append(f"[Assistant] {truncated}")
                else:
                    prompt_parts.append(content)

            full_prompt = "\n\n".join(prompt_parts) if prompt_parts else "[User] Please respond."
            if len(full_prompt) > 10000:
                full_prompt = full_prompt[:10000] + "\n[...truncated]"

            loop = self._ensure_loop()
            future = asyncio.run_coroutine_threadsafe(self._chat(full_prompt), loop)
            try:
                response = future.result(timeout=60)
            except Exception as e:
                response = 'final_answer("Error: ' + str(e) + '")'

            if not response or response.startswith("[ERROR]"):
                response = 'final_answer("Could not get a response. Please try again.")'

            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=response,
                token_usage=TokenUsage(input_tokens=0, output_tokens=0),
            )

else:
    # Stub when smolagents is not installed
    class NLMModel:
        """Stub - install smolagents to use this."""
        def __init__(self, *args, **kwargs):
            raise ImportError("smolagents is required: pip install smolagents")
