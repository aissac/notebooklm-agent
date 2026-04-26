"""notebooklm-agent — Zero-token AI agent powered by Google NotebookLM.

One brain per user, forever. The notebook IS the memory.
"""

__version__ = "0.2.0"

from notebooklm_agent.brain import Brain, BrainError, BrainNotReadyError, UserBrain
from notebooklm_agent.brain.bootstrap import BrainBootstrapper, BOOTSTRAP_TITLE
from notebooklm_agent.auth import get_auth, get_client, close_pool
from notebooklm_agent.memory import NotebookMemory

__all__ = [
    "Brain",
    "BrainError",
    "BrainNotReadyError",
    "UserBrain",
    "BrainBootstrapper",
    "BOOTSTRAP_TITLE",
    "get_auth",
    "get_client",
    "close_pool",
    "NotebookMemory",
]
