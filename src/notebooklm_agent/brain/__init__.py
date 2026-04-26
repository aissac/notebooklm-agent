"""Brain module - One notebook per user, forever."""

from notebooklm_agent.brain.constants import MAX_SOURCES, BOOTSTRAP_TITLE, USER_SOURCE_PREFIX
from notebooklm_agent.brain.core import Brain, BrainError, BrainNotReadyError
from notebooklm_agent.brain.bootstrap import BrainBootstrapper, MIN_SOURCES_FOR_READY
from notebooklm_agent.brain.user_brain import UserBrain
from notebooklm_agent.brain.chat import ChatSession
from notebooklm_agent.brain.research import ResearchPipeline, ResearchResult, ResearchMode
from notebooklm_agent.brain.artifacts import ArtifactGenerator, ArtifactResult, ArtifactType

__all__ = [
    # Constants
    "MAX_SOURCES",
    "BOOTSTRAP_TITLE",
    "USER_SOURCE_PREFIX",
    # Core
    "Brain",
    "BrainError",
    "BrainNotReadyError",
    # Bootstrap
    "BrainBootstrapper",
    "MIN_SOURCES_FOR_READY",
    # User brain
    "UserBrain",
    # Components
    "ChatSession",
    "ResearchPipeline",
    "ResearchResult",
    "ResearchMode",
    "ArtifactGenerator",
    "ArtifactResult",
    "ArtifactType",
]
