"""Agent - Thin wrapper around Brain for programmatic use.

The gateway (Telegram, CLI, etc.) handles command routing directly.
This agent class provides a simple run() method that dispatches
to the appropriate Brain operation based on the task string.

One brain per user. The notebook IS the memory.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

from notebooklm_agent.brain.core import Brain
from notebooklm_agent.brain.research import ResearchResult

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    DIRECT = "direct"    # Single chat.ask() call
    RESEARCH = "research"  # Research + answer
    ARTIFACT = "artifact"  # Generate artifact


@dataclass
class AgentResult:
    """Result of an agent run."""
    success: bool
    answer: str
    mode: AgentMode
    error: str | None = None


class Agent:
    """Simple agent wrapper around Brain.

    Usage:
        brain = Brain(client, notebook_id="abc123")
        await brain.ensure_ready()

        agent = Agent(brain)
        result = await agent.run("What is quantum computing?")
        print(result.answer)
    """

    def __init__(self, brain: Brain):
        self.brain = brain

    async def run(self, task: str, mode: str | None = None) -> AgentResult:
        """Run the agent on a task.

        Args:
            task: The question or command
            mode: Force "direct", "research", "fast", "deep", or None (auto)

        Returns:
            AgentResult with the answer
        """
        try:
            # Direct answer mode
            if mode == "direct" or mode is None:
                answer = await self.brain.ask(task)
                return AgentResult(success=True, answer=answer, mode=AgentMode.DIRECT)

            # Research mode
            if mode in ("research", "fast"):
                result = await self.brain.research(task, mode="fast")
                if result.success:
                    answer = await self.brain.ask(task)
                    return AgentResult(success=True, answer=answer, mode=AgentMode.RESEARCH)
                else:
                    # Research failed, try to answer anyway
                    answer = await self.brain.ask(task)
                    return AgentResult(success=True, answer=answer, mode=AgentMode.DIRECT)

            if mode == "deep":
                result = await self.brain.research(task, mode="deep")
                if result.success:
                    answer = await self.brain.ask(task)
                    return AgentResult(success=True, answer=answer, mode=AgentMode.RESEARCH)
                else:
                    answer = await self.brain.ask(task)
                    return AgentResult(success=True, answer=answer, mode=AgentMode.DIRECT)

            # Default: just ask
            answer = await self.brain.ask(task)
            return AgentResult(success=True, answer=answer, mode=AgentMode.DIRECT)

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return AgentResult(success=False, answer=str(e), mode=AgentMode.DIRECT, error=str(e))
