"""Intent router - Classify user input for fast vs slow path.

Fast path: Direct Brain operation (research, podcast, add source).
Slow path: Agent ReAct loop with chat.ask() + tool calling.

The router uses keyword patterns. No LLM needed for routing.
This saves NotebookLM API calls for obvious commands.

Design from SmallClawLM lessons:
- Research/podcast/quiz = fast path (no thinking needed, just execute)
- Questions/analysis/comparisons = slow path (needs reasoning)
- Unknown = slow path (safer default)
"""

import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Path(Enum):
    FAST = "fast"  # Direct Brain operation
    SLOW = "slow"  # Agent ReAct loop
    COMMAND = "command"  # Gateway-level command (/start, /help)


@dataclass
class RouteResult:
    path: Path
    intent: str  # Operation name
    confidence: float
    params: dict  # Extracted parameters


# Command patterns (gateway-level, never reach the brain)
COMMAND_PATTERNS = {
    "/start", "/help", "/notebooks", "/sources", "/reset",
}

# Intent patterns: (regex, Path, intent_name)
_INTENTS = [
    # Fast: Artifact generation (no reasoning needed)
    (r"\bgenerate\s+podcast\b", Path.FAST, "generate_podcast"),
    (r"\bpodcast\b", Path.FAST, "generate_podcast"),
    (r"\baudio\s+overview\b", Path.FAST, "generate_podcast"),
    (r"\bgenerate\s+video\b", Path.FAST, "generate_video"),
    (r"\bexplainer\s+video\b", Path.FAST, "generate_video"),
    (r"\bgenerate\s+report\b", Path.FAST, "generate_report"),
    (r"\bsummary\s+report\b", Path.FAST, "generate_report"),
    (r"\bgenerate\s+quiz\b", Path.FAST, "generate_quiz"),
    (r"\bquiz\b", Path.FAST, "generate_quiz"),
    (r"\btest\s+me\b", Path.FAST, "generate_quiz"),
    (r"\bmind\s*map\b", Path.FAST, "generate_mindmap"),
    (r"\bconcept\s+map\b", Path.FAST, "generate_mindmap"),
    (r"\bstudy\s+guide\b", Path.FAST, "generate_study_guide"),
    (r"\bflashcards?\b", Path.FAST, "generate_flashcards"),

    # Fast: Source management
    (r"\badd\s+(?:a\s+)?source\b", Path.FAST, "add_source"),
    (r"\bload\s+(?:a\s+)?url\b", Path.FAST, "add_source"),
    (r"\blist\s+(?:the\s+)?sources\b", Path.FAST, "list_sources"),
    (r"\bshow\s+(?:the\s+)?sources\b", Path.FAST, "list_sources"),

    # Fast: Research (atomic pipeline)
    (r"\bresearch\s+on\b", Path.FAST, "research"),
    (r"\bresearch\s+about\b", Path.FAST, "research"),
    (r"\bdeep\s+research\b", Path.FAST, "deep_research"),
    (r"\bresearch\b", Path.FAST, "research"),
    (r"\bfast\s+research\b", Path.FAST, "fast_research"),

    # Slow: Questions and analysis (needs reasoning)
    (r"\bwhy\b", Path.SLOW, "ask"),
    (r"\bhow\b", Path.SLOW, "ask"),
    (r"\bexplain\b", Path.SLOW, "ask"),
    (r"\bcompare\b", Path.SLOW, "ask"),
    (r"\banalyze\b", Path.SLOW, "ask"),
    (r"\bevaluate\b", Path.SLOW, "ask"),
    (r"\bsummarize\b", Path.SLOW, "ask"),
    (r"\btell\s+me\b", Path.SLOW, "ask"),
    (r"\bdescribe\b", Path.SLOW, "ask"),
    (r"\bwhat\s+is\b", Path.SLOW, "ask"),
    (r"\bwhat\s+are\b", Path.SLOW, "ask"),
    (r"\bcan\s+you\b", Path.SLOW, "ask"),
    (r"\bhelp\s+me\b", Path.SLOW, "ask"),
]


def route(user_input: str) -> RouteResult:
    """Classify user input into fast/slow path.

    Fast path: Direct Brain operation (no LLM reasoning).
    Slow path: Agent ReAct loop (chat.ask() + tool calling).
    Command: Gateway-level command (handled before reaching brain).
    """
    text = user_input.strip()

    # Check for gateway commands
    if text.startswith("/"):
        cmd = text.split()[0].lower()
        if cmd in COMMAND_PATTERNS:
            return RouteResult(
                path=Path.COMMAND,
                intent=cmd.lstrip("/"),
                confidence=1.0,
                params={"raw": text},
            )

    # Pattern matching for intents
    lower = text.lower()
    for pattern, path, intent in _INTENTS:
        if re.search(pattern, lower):
            confidence = 0.9 if path == Path.FAST else 0.7
            params = _extract_params(intent, text, lower)
            logger.info(f"Routed: {path.value}/{intent} (conf={confidence})")
            return RouteResult(path=path, intent=intent, confidence=confidence, params=params)

    # Default: slow path (ask the agent)
    logger.info("Routed: slow/ask (default)")
    return RouteResult(
        path=Path.SLOW,
        intent="ask",
        confidence=0.5,
        params={"query": text},
    )


def _extract_params(intent: str, raw: str, lower: str) -> dict:
    """Extract parameters from user input based on intent."""
    params = {"query": raw}

    if intent in ("research", "deep_research", "fast_research"):
        for prefix in ("research on ", "research about ", "deep research ",
                       "fast research ", "research "):
            idx = lower.find(prefix)
            if idx >= 0:
                params["topic"] = raw[idx + len(prefix):].strip()
                break
        if "topic" not in params:
            params["topic"] = raw

    elif intent == "add_source":
        # Extract URL
        import re as re2
        url_match = re2.search(r'https?://\S+', raw)
        if url_match:
            params["url"] = url_match.group(0)

    return params
