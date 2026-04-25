"""Brain module - NotebookLM as the reasoning engine.

The Brain is the core of notebooklm-agent. It wraps NotebookLM's chat API
with auto-bootstrap, research pipeline, and artifact generation.
No smolagents. No local LLM. Gemini through NotebookLM does all reasoning.
"""

from notebooklm_agent.brain.bootstrap import BrainBootstrapper, BOOTSTRAP_SOURCE, BOOTSTRAP_TITLE
from notebooklm_agent.brain.chat import ChatSession
from notebooklm_agent.brain.research import ResearchPipeline
from notebooklm_agent.brain.artifacts import ArtifactGenerator
from notebooklm_agent.brain.core import Brain

__all__ = [
    "Brain",
    "BrainBootstrapper",
    "BOOTSTRAP_SOURCE",
    "BOOTSTRAP_TITLE",
    "ChatSession",
    "ResearchPipeline",
    "ArtifactGenerator",
]
