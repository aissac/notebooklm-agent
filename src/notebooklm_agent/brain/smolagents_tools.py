"""smolagents Tool wrappers for NotebookLM operations.

Only loaded when smolagents mode is requested.
These tools let CodeAgent call Brain operations via smolagents' tool interface.
"""

try:
    from smolagents import Tool
    HAS_SMOLAGENTS = True
except ImportError:
    Tool = object
    HAS_SMOLAGENTS = False

import logging

logger = logging.getLogger(__name__)


if HAS_SMOLAGENTS:

    def create_tools(brain):
        """Create smolagents Tool instances bound to a Brain.

        Args:
            brain: A Brain instance to bind tools to.

        Returns:
            List of smolagents Tool instances.
        """
        import asyncio

        class AskBrainTool(Tool):
            name = "ask_brain"
            description = "Ask a question about the notebook's sources. Returns a cited answer."
            inputs = {"question": {"type": "string", "description": "Question to ask"}}
            output_type = "string"

            def forward(self, question: str) -> str:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(brain.ask(question))
                finally:
                    loop.close()

        class ResearchTool(Tool):
            name = "research"
            description = "Research a topic on the web. Sources are added to the notebook."
            inputs = {
                "query": {"type": "string", "description": "Research question"},
                "mode": {"type": "string", "description": "fast or deep", "nullable": True},
            }
            output_type = "string"

            def forward(self, query: str, mode: str = "fast") -> str:
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(brain.research_topic(query, mode=mode))
                    if result.success:
                        return f"Research complete: {result.source_count} sources added on '{query}'"
                    return f"Research failed: {result.error}"
                finally:
                    loop.close()

        class AddSourceTool(Tool):
            name = "add_source"
            description = "Add a URL as a source to the notebook."
            inputs = {"url": {"type": "string", "description": "URL to add"}}
            output_type = "string"

            def forward(self, url: str) -> str:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(brain.add_source(url))
                finally:
                    loop.close()

        class ListSourcesTool(Tool):
            name = "list_sources"
            description = "List all sources in the notebook."
            inputs = {}
            output_type = "string"

            def forward(self) -> str:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(brain.list_sources())
                finally:
                    loop.close()

        return [AskBrainTool(), ResearchTool(), AddSourceTool(), ListSourcesTool()]

else:
    def create_tools(brain):
        """Stub when smolagents is not installed."""
        raise ImportError("smolagents is required: pip install smolagents")
