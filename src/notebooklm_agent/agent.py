"""Agent - The flexible reasoning engine.

Two modes:
1. Direct mode (default): chat.ask() -> instant answer
2. ReAct mode (complex tasks): think -> act -> observe loop via chat.ask()

ReAct loop WITHOUT smolagents:
- Agent sends "Think about X. What tool should you call?" to chat.ask()
- NotebookLM's Gemini responds with reasoning and a tool suggestion
- Agent parses the tool call and executes it via Brain
- Agent feeds the result back as context
- Repeat until Gemini gives a final answer

ReAct loop WITH smolagents (optional):
- Uses NLMModel (chat.ask wrapper) as the smolagents Model
- CodeAgent handles the tool calling loop
- Better for multi-step programming/math tasks

Key lesson from SmallClawLM: smolagents CodeAgent works but adds
complexity (code tag parsing, InterpreterError, sync/async bridge).
The custom ReAct loop is simpler and works with NotebookLM's natural
response format (plain text reasoning + explicit tool calls).

For most use cases, Direct mode is sufficient. ReAct mode activates
automatically for complex multi-step tasks, or when smolagents is installed.
"""

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from notebooklm_agent.brain.core import Brain
from notebooklm_agent.brain.research import ResearchResult
from notebooklm_agent.memory.notebook_memory import NotebookMemory
from notebooklm_agent.router.intent_router import route, RouteResult, Path

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    DIRECT = "direct"    # Single chat.ask() call
    REACT = "react"      # Custom think-act-observe loop
    SMOLAGENTS = "smolagents"  # smolagents CodeAgent (optional)


@dataclass
class AgentStep:
    """One step in the ReAct loop."""
    step_num: int
    thought: str
    action: str
    action_input: dict
    observation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Result of an agent run."""
    success: bool
    answer: str
    mode: AgentMode
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None


# Tool registry for the custom ReAct loop
REACT_TOOLS = {
    "ask_brain": "Ask a question to the notebook's sources",
    "research": "Research a topic on the web (mode: fast or deep)",
    "add_source": "Add a URL as a source to the notebook",
    "list_sources": "List sources in the notebook",
    "generate_podcast": "Generate a podcast from sources",
    "generate_report": "Generate a structured report",
    "generate_quiz": "Generate a quiz from sources",
    "generate_mindmap": "Generate a mind map from sources",
}

REACT_SYSTEM_PROMPT = f"""You are an AI agent powered by NotebookLM. You have these tools:

{chr(10).join(f'- {k}: {v}' for k, v in REACT_TOOLS.items())}

To use a tool, output EXACTLY this format:
ACTION: tool_name
INPUT: {{"key": "value"}}

For example:
ACTION: research
INPUT: {{"query": "quantum computing", "mode": "fast"}}

When you have the final answer, output:
FINAL ANSWER: your complete answer here

Always reason step by step. Use tools when you need information.
If you already know the answer from sources, give FINAL ANSWER directly."""


class Agent:
    """Flexible agent with direct, ReAct, and optional smolagents modes.

    Usage:
        # Direct mode (90% of use cases)
        agent = Agent(client)
        result = await agent.run("What is quantum computing?")

        # Explicit ReAct mode for complex tasks
        result = await agent.run("Compare React vs Vue, research both, then give me a detailed analysis", mode="react")

        # The router automatically picks the right mode
        result = await agent.run("podcast")          # -> fast path
        result = await agent.run("explain quantum")  # -> slow path (direct)
        result = await agent.run("research fusion")   # -> fast path
    """

    def __init__(self, client, default_notebook_id: str | None = None, max_steps: int = 5):
        self.client = client
        self._default_notebook_id = default_notebook_id
        self._max_steps = max_steps
        self._brains: dict[str, Brain] = {}  # Per-chat brains
        self._memory = NotebookMemory(client, default_notebook_id)

    async def get_brain(self, chat_id: str = "default", title: str | None = None) -> Brain:
        """Get or create a Brain for a specific chat/user.

        Each user gets their own Brain (notebook) for isolation.
        """
        if chat_id in self._brains:
            return self._brains[chat_id]

        if self._default_notebook_id:
            brain = Brain(self.client, notebook_id=self._default_notebook_id, title=title or "Agent")
        else:
            brain = await Brain.create(self.client, title=title or f"Agent: {chat_id}")

        self._brains[chat_id] = brain
        return brain

    async def run(
        self,
        task: str,
        chat_id: str = "default",
        mode: str | None = None,
    ) -> AgentResult:
        """Run a task through the agent.

        Args:
            task: User's input text.
            chat_id: Chat/user identifier for brain isolation.
            mode: Force "direct", "react", or None (auto-routed).

        Returns:
            AgentResult with answer and step history.
        """
        brain = await self.get_brain(chat_id)

        # Route the task
        routed = route(task)

        # Determine mode
        if mode:
            agent_mode = AgentMode(mode)
        elif routed.path == Path.FAST:
            agent_mode = AgentMode.DIRECT
        else:
            agent_mode = AgentMode.DIRECT  # Default to direct for slow path too
            # Only switch to react for explicitly multi-step tasks
            if any(kw in task.lower() for kw in ["compare", "then", "step by step", "multi-step"]):
                agent_mode = AgentMode.REACT

        # Ensure brain is ready
        auto_topic = task if routed.path == Path.SLOW else routed.params.get("topic")
        await brain.ensure_ready(auto_topic=auto_topic)

        # Execute based on mode
        if agent_mode == AgentMode.DIRECT:
            return await self._run_direct(brain, task, routed)
        elif agent_mode == AgentMode.REACT:
            return await self._run_react(brain, task)
        elif agent_mode == AgentMode.SMOLAGENTS:
            return await self._run_smolagents(brain, task)
        else:
            return await self._run_direct(brain, task, routed)

    async def _run_direct(self, brain: Brain, task: str, routed: RouteResult) -> AgentResult:
        """Direct mode: route to the appropriate Brain operation.

        Fast path intents go directly to Brain methods.
        Slow path intents go to brain.ask().
        """
        try:
            intent = routed.intent
            params = routed.params

            if intent == "research" or intent == "fast_research":
                topic = params.get("topic", task)
                result = await brain.research_topic(topic, mode="fast")
                if result.success:
                    answer = f"Research complete! Added {result.source_count} sources on '{topic}'.\\n\\nNow you can ask questions about this topic."
                    # Auto-answer the research topic
                    chat_answer = await brain.ask(f"Summarize what we know about {topic}")
                    answer += f"\\n\\n{chat_answer}"
                else:
                    answer = f"Research on '{topic}' failed: {result.error}"
                return AgentResult(success=result.success, answer=answer, mode=AgentMode.DIRECT)

            elif intent == "deep_research":
                topic = params.get("topic", task)
                result = await brain.research_topic(topic, mode="deep")
                if result.success:
                    answer = f"Deep research complete! Added {result.source_count} sources on '{topic}'."
                else:
                    answer = f"Deep research on '{topic}' failed: {result.error}"
                return AgentResult(success=result.success, answer=answer, mode=AgentMode.DIRECT)

            elif intent == "generate_podcast":
                result = await brain.podcast()
                answer = "Podcast generated!" if result.success else f"Podcast failed: {result.error}"
                return AgentResult(success=result.success, answer=answer, mode=AgentMode.DIRECT)

            elif intent == "generate_report":
                result = await brain.report()
                answer = "Report generated!" if result.success else f"Report failed: {result.error}"
                return AgentResult(success=result.success, answer=answer, mode=AgentMode.DIRECT)

            elif intent == "generate_quiz":
                result = await brain.quiz()
                answer = "Quiz generated!" if result.success else f"Quiz failed: {result.error}"
                return AgentResult(success=result.success, answer=answer, mode=AgentMode.DIRECT)

            elif intent == "generate_mindmap":
                result = await brain.mindmap()
                answer = "Mind map generated!" if result.success else f"Mind map failed: {result.error}"
                return AgentResult(success=result.success, answer=answer, mode=AgentMode.DIRECT)

            elif intent == "add_source":
                url = params.get("url", task)
                answer = await brain.add_source(url)
                return AgentResult(success=True, answer=answer, mode=AgentMode.DIRECT)

            elif intent == "list_sources":
                answer = await brain.list_sources()
                return AgentResult(success=True, answer=answer, mode=AgentMode.DIRECT)

            else:
                # Slow path: ask the brain directly
                answer = await brain.ask(task)
                self._memory.add_observation("ask", answer)
                return AgentResult(success=True, answer=answer, mode=AgentMode.DIRECT)

        except Exception as e:
            logger.error(f"Direct mode error: {e}")
            return AgentResult(success=False, answer=str(e), mode=AgentMode.DIRECT, error=str(e))

    async def _run_react(self, brain: Brain, task: str) -> AgentResult:
        """Custom ReAct loop using chat.ask() for reasoning.

        Each step:
        1. Send prompt to NotebookLM with ReAct instructions + context
        2. Parse response for ACTION/FINAL ANSWER
        3. Execute action via Brain
        4. Feed observation back as context
        5. Repeat until FINAL ANSWER or max steps
        """
        steps: list[AgentStep] = []
        context = f"Task: {task}\n\n"

        for step_num in range(1, self._max_steps + 1):
            # Ask the brain to reason
            prompt = f"{REACT_SYSTEM_PROMPT}\n\n{context}\n\nWhat should you do next? Think step by step."
            response = await brain.ask(prompt)

            # Check for FINAL ANSWER
            final_match = re.search(r"FINAL ANSWER:\s*(.+)", response, re.DOTALL)
            if final_match:
                answer = final_match.group(1).strip()
                self._memory.add_decision(f"Completed in {step_num} steps", task)
                return AgentResult(
                    success=True,
                    answer=answer,
                    mode=AgentMode.REACT,
                    steps=steps,
                )

            # Parse ACTION and INPUT
            action_match = re.search(r"ACTION:\s*(\w+)", response)
            input_match = re.search(r"INPUT:\s*(\{[^}]+\})", response)

            if not action_match:
                # No action found, treat as final answer
                self._memory.add_observation("react_no_action", response[:200])
                return AgentResult(
                    success=True,
                    answer=response,
                    mode=AgentMode.REACT,
                    steps=steps,
                )

            action = action_match.group(1).lower()
            action_input = {}
            if input_match:
                try:
                    action_input = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    action_input = {"raw": input_match.group(1)}

            # Execute the action
            observation = await self._execute_react_action(brain, action, action_input)

            step = AgentStep(
                step_num=step_num,
                thought=response[:300],
                action=action,
                action_input=action_input,
                observation=observation[:500],
            )
            steps.append(step)

            # Add observation to context
            context += f"Step {step_num}: Called {action}({action_input})\nObservation: {observation[:500]}\n\n"

            self._memory.add_observation(action, observation[:200])

        # Max steps reached
        # Ask for a summary
        summary_prompt = f"{context}\n\nYou have used all your steps. Give your best FINAL ANSWER based on what you found."
        final = await brain.ask(summary_prompt)
        return AgentResult(
            success=True,
            answer=final,
            mode=AgentMode.REACT,
            steps=steps,
        )

    async def _execute_react_action(self, brain: Brain, action: str, params: dict) -> str:
        """Execute a ReAct tool call via Brain."""
        try:
            if action == "ask_brain":
                question = params.get("question", params.get("query", ""))
                return await brain.ask(question)

            elif action == "research":
                query = params.get("query", params.get("topic", ""))
                mode = params.get("mode", "fast")
                result = await brain.research_topic(query, mode=mode)
                if result.success:
                    return f"Research complete: {result.source_count} sources added on '{query}'"
                return f"Research failed: {result.error}"

            elif action == "add_source":
                url = params.get("url", "")
                return await brain.add_source(url)

            elif action == "list_sources":
                return await brain.list_sources()

            elif action == "generate_podcast":
                result = await brain.podcast(params.get("instructions"))
                return "Podcast generated!" if result.success else f"Failed: {result.error}"

            elif action == "generate_report":
                result = await brain.report(params.get("custom_prompt"))
                return "Report generated!" if result.success else f"Failed: {result.error}"

            elif action == "generate_quiz":
                result = await brain.quiz(params.get("instructions"))
                return "Quiz generated!" if result.success else f"Failed: {result.error}"

            elif action == "generate_mindmap":
                result = await brain.mindmap()
                return "Mind map generated!" if result.success else f"Failed: {result.error}"

            else:
                return f"Unknown action: {action}. Available: {', '.join(REACT_TOOLS.keys())}"

        except Exception as e:
            return f"Error executing {action}: {e}"

    async def _run_smolagents(self, brain: Brain, task: str) -> AgentResult:
        """Optional smolagents CodeAgent mode.

        Requires: pip install smolagents
        Uses NLMModel as the smolagents Model, which wraps chat.ask().
        """
        try:
            from smolagents import CodeAgent
            from notebooklm_agent.brain.smolagents_model import NLMModel
            from notebooklm_agent.brain.smolagents_tools import create_tools
        except ImportError:
            return AgentResult(
                success=False,
                answer="smolagents is not installed. Run: pip install smolagents",
                mode=AgentMode.SMOLAGENTS,
                error="ImportError: smolagents",
            )

        try:
            model = NLMModel(notebook_id=brain.notebook_id)
            tools = create_tools(brain)
            agent = CodeAgent(model=model, tools=tools, max_steps=self._max_steps)
            result = agent.run(task)

            self._memory.add_observation("smolagents", str(result)[:200])
            return AgentResult(
                success=True,
                answer=str(result),
                mode=AgentMode.SMOLAGENTS,
            )
        except Exception as e:
            logger.error(f"smolagents mode error: {e}")
            return AgentResult(
                success=False,
                answer=f"smolagents error: {e}",
                mode=AgentMode.SMOLAGENTS,
                error=str(e),
            )

    # ─── Convenience Methods ───

    async def ask(self, question: str, chat_id: str = "default") -> str:
        """Quick ask - returns just the answer string."""
        result = await self.run(question, chat_id=chat_id)
        return result.answer

    async def research(self, topic: str, chat_id: str = "default", mode: str = "fast") -> str:
        """Quick research - returns result message."""
        result = await self.run(f"research {topic}", chat_id=chat_id)
        return result.answer

    async def reset(self, chat_id: str = "default") -> None:
        """Reset a chat's brain (creates a new notebook)."""
        if chat_id in self._brains:
            brain = self._brains[chat_id]
            await brain.delete_notebook()
            del self._brains[chat_id]

    def __repr__(self) -> str:
        return f"Agent(brains={len(self._brains)}, mode=auto)"
