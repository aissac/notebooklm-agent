"""ResearchPipeline - Atomic research flow (start -> poll -> import -> wait).

This is the fix for SmallClawLM's biggest pain point: the 3-step manual
research flow where you had to start, then poll, then import separately,
and if you forgot any step, you got empty results.

Now it's one call: await pipeline.brain_research(nb_id, "fusion energy")
That's it. It handles everything.

Key design decisions:
- start -> poll -> import -> wait_until_ready in one atomic call
- Exponential backoff on poll (starts at 2s, max 15s)
- Auto-detects research mode (fast=90s, deep=15min timeout)
- Returns count of imported sources
- Never crashes - returns error messages as strings
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ResearchMode(Enum):
    FAST = "fast"
    DEEP = "deep"


@dataclass
class ResearchResult:
    """Result of a research operation."""
    success: bool
    source_count: int
    task_id: str | None = None
    error: str | None = None


class ResearchPipeline:
    """Atomic research pipeline - one call does it all.

    Usage:
        pipeline = ResearchPipeline(client)
        result = await pipeline.brain_research(notebook_id, "quantum computing")
        if result.success:
            print(f"Added {result.source_count} sources")
    """

    def __init__(self, client, timeout_fast: float = 120.0, timeout_deep: float = 900.0):
        self.client = client
        self.timeout_fast = timeout_fast      # 2 min for fast research
        self.timeout_deep = timeout_deep      # 15 min for deep research

    async def brain_research(
        self,
        notebook_id: str,
        query: str,
        mode: str = "fast",
    ) -> ResearchResult:
        """Full brain research: start -> poll -> import -> wait for processing.

        This is the one-call API. Use this instead of calling start/poll/import
        separately.

        Args:
            notebook_id: Target notebook
            query: Research question
            mode: "fast" (90s) or "deep" (15min)

        Returns:
            ResearchResult with success status and source count
        """
        try:
            # 1. Start research
            logger.info(f"Starting {mode} research on: {query}")
            start_result = await self.client.research.start(
                notebook_id=notebook_id,
                query=query,
                source="web",
                mode=mode,
            )

            if not start_result:
                return ResearchResult(success=False, source_count=0, error="Research start returned None")

            task_id = start_result.get("task_id") if isinstance(start_result, dict) else None

            # 2. Poll until complete
            timeout = self.timeout_fast if mode == "fast" else self.timeout_deep
            sources_data = await self._poll_and_import(notebook_id, task_id, timeout)

            if not sources_data:
                # Research completed but no sources returned
                # This can happen if the query was too narrow
                return ResearchResult(success=True, source_count=0, task_id=task_id)

            # 3. Import sources
            imported = await self.client.research.import_sources(
                notebook_id, task_id, sources_data
            )

            # 4. Wait for sources to be processed
            source_ids = [s.get("source_id") for s in imported if s.get("source_id")]
            if source_ids:
                await self._wait_for_sources(notebook_id, source_ids)

            count = len(source_ids) if source_ids else len(imported)
            logger.info(f"Research complete: {count} sources imported for '{query}'")
            return ResearchResult(success=True, source_count=count, task_id=task_id)

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return ResearchResult(success=False, source_count=0, error=str(e))

    async def _poll_and_import(
        self,
        notebook_id: str,
        task_id: str | None,
        timeout: float,
    ) -> list[dict] | None:
        """Poll research status and return source data for import."""
        max_wait = timeout
        interval = 2.0  # Start polling every 2s
        max_interval = 15.0
        elapsed = 0.0

        while elapsed < max_wait:
            await asyncio.sleep(interval)
            elapsed += interval

            try:
                status = await self.client.research.poll(notebook_id)

                if status and isinstance(status, dict):
                    # Check for completion indicators
                    if status.get("done") or status.get("status") == "complete":
                        return status.get("sources", [])

                    # Check for errors
                    if status.get("error") or status.get("status") == "error":
                        logger.error(f"Research error: {status}")
                        return None

            except Exception as e:
                logger.warning(f"Poll error (will retry): {e}")

            # Exponential backoff
            interval = min(interval * 1.5, max_interval)

        logger.warning(f"Research polling timed out after {elapsed:.0f}s")
        return None

    async def _wait_for_sources(self, notebook_id: str, source_ids: list[str]) -> None:
        """Wait for newly imported sources to be processed by NotebookLM."""
        try:
            await self.client.sources.wait_for_sources(
                notebook_id, source_ids, timeout=120.0
            )
        except Exception as e:
            # Non-fatal - sources may still be processing, but we can proceed
            logger.warning(f"Source wait timed out (non-fatal): {e}")
            # Give it a few more seconds anyway
            await asyncio.sleep(5)

    async def start_only(
        self,
        notebook_id: str,
        query: str,
        mode: str = "fast",
    ) -> dict | None:
        """Start research without waiting. For background/async usage.

        Returns the start result dict (contains task_id for later polling).
        """
        return await self.client.research.start(
            notebook_id=notebook_id,
            query=query,
            source="web",
            mode=mode,
        )

    async def check_status(self, notebook_id: str) -> dict | None:
        """Check research status without importing."""
        return await self.client.research.poll(notebook_id)

    async def quick_research(
        self,
        notebook_id: str,
        query: str,
    ) -> ResearchResult:
        """Fast research with shorter timeout (2 min). Good for Telegram."""
        return await self.brain_research(notebook_id, query, mode="fast")

    async def deep_research(
        self,
        notebook_id: str,
        query: str,
    ) -> ResearchResult:
        """Deep research with full timeout (15 min). Best for complex topics."""
        return await self.brain_research(notebook_id, query, mode="deep")
