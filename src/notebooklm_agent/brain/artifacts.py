"""ArtifactGenerator - Generate podcasts, reports, quizzes, mind maps, etc.

Wraps notebooklm-py's ArtifactsAPI with proper polling and download support.
"""

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    PODCAST = "podcast"
    REPORT = "report"
    QUIZ = "quiz"
    MINDMAP = "mindmap"
    VIDEO = "video"
    FLASHCARDS = "flashcards"
    STUDY_GUIDE = "study_guide"
    SLIDE_DECK = "slide_deck"
    INFOGRAPHIC = "infographic"
    DATA_TABLE = "data_table"


@dataclass
class ArtifactResult:
    """Result of an artifact generation operation."""
    success: bool
    artifact_type: ArtifactType
    artifact_id: str | None = None
    download_path: str | None = None
    error: str | None = None


class ArtifactGenerator:
    """Generate NotebookLM artifacts with automatic polling.

    Usage:
        gen = ArtifactGenerator(client)
        result = await gen.generate_podcast(notebook_id)
        if result.success:
            print(f"Podcast ready: {result.download_path}")
    """

    def __init__(self, client, default_download_dir: str = "~/.nlm-agent/downloads"):
        self.client = client
        self.download_dir = Path(default_download_dir).expanduser()
        self.download_dir.mkdir(parents=True, exist_ok=True)

    async def generate_podcast(
        self,
        notebook_id: str,
        instructions: str | None = None,
    ) -> ArtifactResult:
        """Generate an audio overview podcast."""
        try:
            kwargs = {"notebook_id": notebook_id}
            if instructions:
                kwargs["instructions"] = instructions

            status = await self.client.artifacts.generate_audio(**kwargs)
            task_id = status.task_id if hasattr(status, "task_id") else str(status)

            # Wait for completion (podcasts take 2-5 min)
            await self._wait_for_artifact(notebook_id, task_id, timeout=600)

            return ArtifactResult(
                success=True,
                artifact_type=ArtifactType.PODCAST,
                artifact_id=task_id,
            )
        except Exception as e:
            logger.error(f"Podcast generation failed: {e}")
            return ArtifactResult(
                success=False,
                artifact_type=ArtifactType.PODCAST,
                error=str(e),
            )

    async def generate_report(
        self,
        notebook_id: str,
        custom_prompt: str | None = None,
    ) -> ArtifactResult:
        """Generate a structured report."""
        try:
            kwargs = {"notebook_id": notebook_id}
            if custom_prompt:
                kwargs["custom_prompt"] = custom_prompt

            status = await self.client.artifacts.generate_report(**kwargs)
            task_id = status.task_id if hasattr(status, "task_id") else str(status)

            await self._wait_for_artifact(notebook_id, task_id, timeout=300)

            return ArtifactResult(
                success=True,
                artifact_type=ArtifactType.REPORT,
                artifact_id=task_id,
            )
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return ArtifactResult(
                success=False,
                artifact_type=ArtifactType.REPORT,
                error=str(e),
            )

    async def generate_quiz(
        self,
        notebook_id: str,
        instructions: str | None = None,
    ) -> ArtifactResult:
        """Generate a quiz."""
        try:
            status = await self.client.artifacts.generate_quiz(
                notebook_id=notebook_id,
                instructions=instructions,
            )
            task_id = status.task_id if hasattr(status, "task_id") else str(status)

            await self._wait_for_artifact(notebook_id, task_id, timeout=300)

            return ArtifactResult(
                success=True,
                artifact_type=ArtifactType.QUIZ,
                artifact_id=task_id,
            )
        except Exception as e:
            logger.error(f"Quiz generation failed: {e}")
            return ArtifactResult(
                success=False,
                artifact_type=ArtifactType.QUIZ,
                error=str(e),
            )

    async def generate_mindmap(self, notebook_id: str) -> ArtifactResult:
        """Generate a mind map."""
        try:
            result = await self.client.artifacts.generate_mind_map(notebook_id=notebook_id)
            return ArtifactResult(
                success=True,
                artifact_type=ArtifactType.MINDMAP,
                artifact_id=str(result) if result else None,
            )
        except Exception as e:
            logger.error(f"Mind map generation failed: {e}")
            return ArtifactResult(
                success=False,
                artifact_type=ArtifactType.MINDMAP,
                error=str(e),
            )

    async def generate_video(
        self,
        notebook_id: str,
        instructions: str | None = None,
    ) -> ArtifactResult:
        """Generate an explainer video."""
        try:
            kwargs = {"notebook_id": notebook_id}
            if instructions:
                kwargs["instructions"] = instructions

            status = await self.client.artifacts.generate_video(**kwargs)
            task_id = status.task_id if hasattr(status, "task_id") else str(status)

            await self._wait_for_artifact(notebook_id, task_id, timeout=900)

            return ArtifactResult(
                success=True,
                artifact_type=ArtifactType.VIDEO,
                artifact_id=task_id,
            )
        except Exception as e:
            logger.error(f"Video generation failed: {e}")
            return ArtifactResult(
                success=False,
                artifact_type=ArtifactType.VIDEO,
                error=str(e),
            )

    async def download_artifact(
        self,
        notebook_id: str,
        artifact_id: str,
        artifact_type: ArtifactType,
        output_path: str | None = None,
    ) -> str:
        """Download a generated artifact to disk.

        Returns the path to the downloaded file.
        """
        if output_path is None:
            output_path = str(self.download_dir / f"{artifact_type.value}_{artifact_id}")

        download_map = {
            ArtifactType.PODCAST: self.client.artifacts.download_audio,
            ArtifactType.REPORT: self.client.artifacts.download_report,
            ArtifactType.QUIZ: self.client.artifacts.download_quiz,
            ArtifactType.MINDMAP: self.client.artifacts.download_mind_map,
            ArtifactType.VIDEO: self.client.artifacts.download_video,
        }

        download_fn = download_map.get(artifact_type)
        if download_fn is None:
            raise ValueError(f"Cannot download artifact type: {artifact_type}")

        return await download_fn(notebook_id, output_path, artifact_id=artifact_id)

    async def _wait_for_artifact(
        self,
        notebook_id: str,
        task_id: str,
        timeout: float = 300.0,
    ) -> None:
        """Wait for an artifact generation task to complete."""
        await self.client.artifacts.wait_for_completion(
            notebook_id, task_id, timeout=timeout
        )
