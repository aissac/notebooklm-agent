"""Telegram Gateway - Async-native, brain-first, plain text.

No Markdown crashes. No sync/async bridge. No empty notebooks.
"""

import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from notebooklm_agent.agent import Agent
from notebooklm_agent.auth import get_client, close_pool
from notebooklm_agent.utils.text import sanitize_for_display, chunk_for_telegram

logger = logging.getLogger(__name__)
MAX_MSG_LEN = 4096

WELCOME = """NotebookLM Agent - Zero-token AI

I research, reason, and create using Google NotebookLM.
No API keys needed - Gemini does the thinking for free.

Commands:
/research <topic> - Quick web research (90s)
/deep <topic> - Deep web research (15min)
/add <url> - Add a URL source
/sources - List loaded sources
/podcast - Generate audio podcast
/report - Generate structured report
/quiz - Generate a quiz
/mindmap - Generate mind map
/reset - Start a fresh brain

Or just ask me anything!"""


class TelegramGateway:
    """Async-native Telegram gateway with brain-first architecture."""

    def __init__(self, token: str, max_steps: int = 5):
        self.token = token
        self.max_steps = max_steps
        self._agent = None
        self._app = None

    async def _init_agent(self) -> Agent:
        if self._agent is None:
            client = await get_client()
            self._agent = Agent(client, max_steps=self.max_steps)
        return self._agent

    def _build_app(self) -> Application:
        app = Application.builder().token(self.token).build()
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("reset", self._cmd_reset))
        app.add_handler(CommandHandler("research", self._cmd_research))
        app.add_handler(CommandHandler("deep", self._cmd_deep))
        app.add_handler(CommandHandler("add", self._cmd_add))
        app.add_handler(CommandHandler("sources", self._cmd_sources))
        app.add_handler(CommandHandler("podcast", self._cmd_podcast))
        app.add_handler(CommandHandler("report", self._cmd_report))
        app.add_handler(CommandHandler("quiz", self._cmd_quiz))
        app.add_handler(CommandHandler("mindmap", self._cmd_mindmap))
        app.add_handler(CommandHandler("notebooks", self._cmd_notebooks))
        # Bitwise AND for filter composition (python-telegram-bot v22)
        text_filter = filters.TEXT & ~filters.COMMAND
        app.add_handler(MessageHandler(text_filter, self._handle_message))
        return app

    async def _cmd_start(self, update, ctx):
        await update.message.reply_text(WELCOME)

    async def _cmd_help(self, update, ctx):
        await update.message.reply_text(WELCOME)

    async def _cmd_reset(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        await agent.reset(chat_id)
        await update.message.reply_text("Brain reset! Next message creates a fresh notebook.")

    async def _cmd_research(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        topic = " ".join(ctx.args) if ctx.args else ""
        if not topic:
            await update.message.reply_text("Usage: /research <topic>")
            return
        await update.message.reply_text(f"Researching {topic!r}... (~90s)")
        result = await agent.run(f"research {topic}", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_deep(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        topic = " ".join(ctx.args) if ctx.args else ""
        if not topic:
            await update.message.reply_text("Usage: /deep <topic>")
            return
        await update.message.reply_text(f"Deep research on {topic!r}... (5-15 min)")
        result = await agent.run(f"deep research {topic}", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_add(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        url = " ".join(ctx.args) if ctx.args else ""
        if not url:
            await update.message.reply_text("Usage: /add <url>")
            return
        await update.message.reply_text(f"Adding source: {url}")
        result = await agent.run(f"add source {url}", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_sources(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        result = await agent.run("list sources", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_podcast(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        await update.message.reply_text("Generating podcast... (2-5 min)")
        result = await agent.run("podcast", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_report(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        await update.message.reply_text("Generating report... (1-3 min)")
        result = await agent.run("report", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_quiz(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        await update.message.reply_text("Generating quiz... (30-90s)")
        result = await agent.run("quiz", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_mindmap(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        await update.message.reply_text("Generating mind map... (30-90s)")
        result = await agent.run("mindmap", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_notebooks(self, update, ctx):
        client = await get_client()
        notebooks = await client.notebooks.list()
        if not notebooks:
            await update.message.reply_text("No notebooks yet.")
            return
        lines = ["Your notebooks:"]
        for nb in notebooks[:20]:
            title = getattr(nb, "title", "Untitled")
            lines.append(f"  - {title}")
        await update.message.reply_text("\n".join(lines))

    async def _handle_message(self, update, ctx):
        text = update.message.text
        if not text:
            return
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        brain = await agent.get_brain(chat_id)
        if not brain.is_ready or not brain.notebook_id:
            await update.message.reply_text("Building brain... (auto-research, ~90s)")
        else:
            await update.message.reply_text("Thinking...")
        try:
            result = await agent.run(text, chat_id=chat_id)
            await self._send_chunks(update, result.answer)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _send_chunks(self, update, text):
        text = sanitize_for_display(text)
        chunks = chunk_for_telegram(text, MAX_MSG_LEN)
        for chunk in chunks:
            try:
                await update.message.reply_text(chunk)
            except Exception as e:
                logger.error(f"Send error: {e}")
                try:
                    safe = chunk.replace("&", "and").replace("<", "").replace(">", "")
                    await update.message.reply_text(safe)
                except Exception:
                    pass

    async def start(self):
        self._app = self._build_app()
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling()
        logger.info("Telegram gateway started")

    async def stop(self):
        if self._app:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        await close_pool()
        logger.info("Telegram gateway stopped")

    async def run(self):
        await self.start()
        try:
            await asyncio.Event().wait()
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            await self.stop()
