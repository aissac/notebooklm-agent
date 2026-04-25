"""notebooklm-agent — Zero-token AI agent powered by Google NotebookLM."""

__version__ = "0.1.0"

from notebooklm_agent.agent import Agent
from notebooklm_agent.brain import Brain, BrainBootstrapper
from notebooklm_agent.auth import get_auth, get_client
from notebooklm_agent.memory import NotebookMemory

__all__ = [
    "Agent",
    "Brain",
    "BrainBootstrapper",
    "get_auth",
    "get_client",
    "NotebookMemory",
]
