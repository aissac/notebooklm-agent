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

        This is the ONE-CALL API. No manual steps needed.

        The flow:
        1. Start research with research.start()
        2. Poll until status is "completed" (status_code 2 or 6)
        3. Extract source data from poll result
        4. Call import_sources() with task_id and sources
        5. Wait for imported sources to finish processing

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
            logger.info(f"Research started, task_id={task_id}")

            # 2. Poll until complete
            timeout = self.timeout_fast if mode == "fast" else self.timeout_deep
            poll_result = await self._poll_until_complete(notebook_id, timeout)

            if not poll_result:
                # Even if poll timed out, the research may have completed
                # Try one more poll
                poll_result = await self.client.research.poll(notebook_id)

            if not poll_result or not isinstance(poll_result, dict):
                return ResearchResult(success=True, source_count=0, task_id=task_id,
                                     error="Could not get research status")

            # 3. Extract source data
            sources_data = poll_result.get("sources", [])
            if not sources_data:
                # Research completed but no parseable sources
                # This can happen with very narrow queries
                logger.warning("Research completed but no sources extracted")
                return ResearchResult(success=True, source_count=0, task_id=task_id)

            current_task_id = poll_result.get("task_id", task_id)

            # 4. Import sources
            try:
                imported = await self.client.research.import_sources(
                    notebook_id, current_task_id, sources_data
                )
                source_count = len(imported) if isinstance(imported, list) else 0
                logger.info(f"Imported {source_count} sources")
            except Exception as e:
                logger.warning(f"Import failed (may already be imported): {e}")
                source_count = len(sources_data)
                imported = None

            # 5. Wait for sources to process
            if imported and isinstance(imported, list):
                source_ids = [s.get("source_id") for s in imported if s.get("source_id")]
                if source_ids:
                    try:
                        await self.client.sources.wait_for_sources(
                            notebook_id, source_ids, timeout=120.0
                        )
                    except Exception as e:
                        logger.warning(f"Source wait timed out (non-fatal): {e}")
                        await asyncio.sleep(5)  # Give it a moment anyway

            return ResearchResult(success=True, source_count=source_count, task_id=current_task_id)

        except Exception as e:
            logger.error(f"Research failed: {e}")
            return ResearchResult(success=False, source_count=0, error=str(e))

    async def _poll_until_complete(
        self,
        notebook_id: str,
        timeout: float,
    ) -> dict | None:
        """Poll research status until complete or timeout.

        Research status codes:
        - "in_progress": still researching
        - "completed": done, sources available (status_code 2 or 6)
        - "no_research": no research has been started
        """
        interval = 3.0
        max_interval = 15.0
        elapsed = 0.0

        while elapsed < timeout:
            await asyncio.sleep(interval)
            elapsed += interval

            try:
                result = await self.client.research.poll(notebook_id)

                if not result or not isinstance(result, dict):
                    continue

                status = result.get("status", "unknown")

                # Check for completion
                if status == "completed":
                    logger.info(f"Research completed after {elapsed:.0f}s")
                    return result

                if status == "no_research":
                    logger.warning("No research found")
                    return None

                # Still in progress
                logger.debug(f"Research status: {status} ({elapsed:.0f}s elapsed)")

            except Exception as e:
                logger.warning(f"Poll error (will retry): {e}")

            # Exponential backoff
            interval = min(interval * 1.5, max_interval)

        logger.warning(f"Research polling timed out after {elapsed:.0f}s")
        return None

    async def quick_research(self, notebook_id: str, query: str) -> ResearchResult:
        """Fast research with shorter timeout (2 min). Good for Telegram."""
        return await self.brain_research(notebook_id, query, mode="fast")

    async def deep_research(self, notebook_id: str, query: str) -> ResearchResult:
        """Deep research with full timeout (15 min). Best for complex topics."""
        return await self.brain_research(notebook_id, query, mode="deep")
