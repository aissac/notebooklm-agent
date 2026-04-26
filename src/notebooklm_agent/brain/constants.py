"""Constants shared across brain modules."""

# NotebookLM soft source cap. Keep headroom for research imports.
MAX_SOURCES = 40

# Source title markers
BOOTSTRAP_TITLE = "Agent Instructions (Bootstrap)"
USER_SOURCE_PREFIX = "[USER]"

# Research timeouts
RESEARCH_TIMEOUT_FAST = 120.0   # 2 min
RESEARCH_TIMEOUT_DEEP = 900.0   # 15 min
