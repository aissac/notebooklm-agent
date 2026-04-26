"""Telegram Gateway - Async-native, brain-first, plain text.

No Markdown crashes. No sync/async bridge. No empty notebooks.
Full command menu with inline keyboards for easy interaction.
"""

import asyncio
import logging
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from notebooklm_agent.agent import Agent
from notebooklm_agent.auth import get_client, close_pool
from notebooklm_agent.utils.text import sanitize_for_display, chunk_for_telegram

logger = logging.getLogger(__name__)
MAX_MSG_LEN = 4096

# ─── Command Descriptions (registered with Telegram) ───

BOT_COMMANDS = [
    BotCommand("start", "Start the bot and see welcome message"),
    BotCommand("help", "Show full command reference"),
    BotCommand("ask", "Ask a question about your sources"),
    BotCommand("research", "Quick web research on a topic (~90s)"),
    BotCommand("deep", "Deep web research on a topic (5-15 min)"),
    BotCommand("add", "Add a URL as a source to your notebook"),
    BotCommand("addtext", "Add text content as a source"),
    BotCommand("sources", "List all sources in your notebook"),
    BotCommand("podcast", "Generate an audio podcast from sources"),
    BotCommand("report", "Generate a structured report"),
    BotCommand("quiz", "Generate a quiz from sources"),
    BotCommand("mindmap", "Generate a mind map from sources"),
    BotCommand("video", "Generate an explainer video"),
    BotCommand("rename", "Rename your current notebook"),
    BotCommand("notebooks", "List all your notebooks"),
    BotCommand("status", "Show brain status and source count"),
    BotCommand("reset", "Delete current notebook and start fresh"),
]

WELCOME = """NotebookLM Agent - Zero-token AI

I research, reason, and create using Google NotebookLM.
No API keys needed - Gemini does the thinking for free.

Just type a question and I'll answer from my sources.
If I don't have sources yet, I'll auto-research your topic!

Use /help for the full command menu."""


HELP_TEXT = """NotebookLM Agent Commands

RESEARCH & SOURCES:
/research <topic> - Quick web research (~90s)
/deep <topic> - Deep web research (5-15 min)
/add <url> - Add a URL as a source
/addtext <title> | <content> - Add text as a source
/sources - List your notebook's sources

ASK & CHAT:
/ask <question> - Ask about your sources
Or just type your question directly!

GENERATE ARTIFACTS:
/podcast - Generate audio podcast
/report [prompt] - Generate structured report
/quiz [topic] - Generate a quiz
/mindmap - Generate a mind map
/video [instructions] - Generate explainer video

MANAGE:
/rename <name> - Rename your notebook
/notebooks - List all notebooks
/status - Show brain status & source count
/reset - Delete notebook & start fresh

TIPS:
- First message auto-creates a brain with research
- /add more URLs to expand knowledge
- Artifacts take 1-5 min to generate
- Everything is powered by NotebookLM (free, no API keys)"""


class TelegramGateway:
    """Async-native Telegram gateway with brain-first architecture and command menus."""

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

        # Core commands
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("ask", self._cmd_ask))
        app.add_handler(CommandHandler("research", self._cmd_research))
        app.add_handler(CommandHandler("deep", self._cmd_deep))
        app.add_handler(CommandHandler("add", self._cmd_add))
        app.add_handler(CommandHandler("addtext", self._cmd_addtext))
        app.add_handler(CommandHandler("sources", self._cmd_sources))
        app.add_handler(CommandHandler("podcast", self._cmd_podcast))
        app.add_handler(CommandHandler("report", self._cmd_report))
        app.add_handler(CommandHandler("quiz", self._cmd_quiz))
        app.add_handler(CommandHandler("mindmap", self._cmd_mindmap))
        app.add_handler(CommandHandler("video", self._cmd_video))
        app.add_handler(CommandHandler("rename", self._cmd_rename))
        app.add_handler(CommandHandler("notebooks", self._cmd_notebooks))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("reset", self._cmd_reset))

        # Inline button callbacks
        app.add_handler(CallbackQueryHandler(self._callback_handler))

        # Catch-all text handler (must be last)
        text_filter = filters.TEXT & ~filters.COMMAND
        app.add_handler(MessageHandler(text_filter, self._handle_message))
        return app

    # ─── Inline Keyboards ───

    @staticmethod
    def _main_menu_keyboard() -> InlineKeyboardMarkup:
        """Main action menu shown on /start."""
        keyboard = [
            [
                InlineKeyboardButton("🔍 Research", callback_data="cmd_research"),
                InlineKeyboardButton("💬 Ask", callback_data="cmd_ask"),
            ],
            [
                InlineKeyboardButton("➕ Add Source", callback_data="cmd_add"),
                InlineKeyboardButton("📚 Sources", callback_data="cmd_sources"),
            ],
            [
                InlineKeyboardButton("🎙 Podcast", callback_data="cmd_podcast"),
                InlineKeyboardButton("📊 Report", callback_data="cmd_report"),
            ],
            [
                InlineKeyboardButton("❓ Quiz", callback_data="cmd_quiz"),
                InlineKeyboardButton("🧠 Mind Map", callback_data="cmd_mindmap"),
            ],
            [
                InlineKeyboardButton("📈 Status", callback_data="cmd_status"),
                InlineKeyboardButton("❓ Help", callback_data="cmd_help"),
            ],
        ]
        return InlineKeyboardMarkup(keyboard)

    # ─── Command Handlers ───

    async def _cmd_start(self, update, ctx):
        await self._set_commands(ctx)
        await update.message.reply_text(
            WELCOME,
            reply_markup=self._main_menu_keyboard(),
        )

    async def _cmd_help(self, update, ctx):
        await update.message.reply_text(HELP_TEXT)

    async def _cmd_ask(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        question = " ".join(ctx.args) if ctx.args else ""
        if not question:
            await update.message.reply_text(
                "Usage: /ask <question>\n\nExample: /ask What are the key findings?"
            )
            return
        await update.message.reply_text("Thinking...")
        try:
            result = await agent.run(question, chat_id=chat_id)
            await self._send_chunks(update, result.answer)
        except Exception as e:
            logger.error(f"Ask error: {e}")
            await update.message.reply_text(f"Error: {e}")

    async def _cmd_research(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        topic = " ".join(ctx.args) if ctx.args else ""
        if not topic:
            await update.message.reply_text(
                "Usage: /research <topic>\n\nExample: /research quantum computing"
            )
            return
        await update.message.reply_text(f"Researching '{topic}'... (~90s)")
        result = await agent.run(f"research {topic}", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_deep(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        topic = " ".join(ctx.args) if ctx.args else ""
        if not topic:
            await update.message.reply_text(
                "Usage: /deep <topic>\n\nExample: /deep artificial intelligence ethics"
            )
            return
        await update.message.reply_text(f"Deep research on '{topic}'... (5-15 min)")
        result = await agent.run(f"deep research {topic}", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_add(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        url = " ".join(ctx.args) if ctx.args else ""
        if not url:
            await update.message.reply_text(
                "Usage: /add <url>\n\nExample: /add https://en.wikipedia.org/wiki/Python_(programming_language)"
            )
            return
        await update.message.reply_text(f"Adding source: {url}")
        result = await agent.run(f"add source {url}", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_addtext(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        text = " ".join(ctx.args) if ctx.args else ""
        if not text or "|" not in text:
            await update.message.reply_text(
                "Usage: /addtext <title> | <content>\n\nExample: /addtext Meeting Notes | The team decided to pursue option B"
            )
            return
        parts = text.split("|", 1)
        title = parts[0].strip()
        content = parts[1].strip()
        await update.message.reply_text(f"Adding text source: {title}")
        brain = await agent.get_brain(chat_id)
        if not brain.notebook_id:
            await brain.ensure_ready(auto_topic=title)
        result = await brain.add_text_source(title, content)
        await update.message.reply_text(result)

    async def _cmd_sources(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        result = await agent.run("list sources", chat_id=chat_id)
        await self._send_chunks(update, result.answer)

    async def _cmd_podcast(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        instructions = " ".join(ctx.args) if ctx.args else None
        await update.message.reply_text("Generating podcast... (2-5 min)")
        brain = await agent.get_brain(chat_id)
        if not brain.is_ready:
            await brain.ensure_ready(auto_topic="podcast")
        result = await brain.podcast(instructions=instructions)
        if result.success:
            await update.message.reply_text("Podcast generated! Use NotebookLM to listen.")
        else:
            await update.message.reply_text(f"Podcast failed: {result.error}")

    async def _cmd_report(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        custom_prompt = " ".join(ctx.args) if ctx.args else None
        await update.message.reply_text("Generating report... (1-3 min)")
        brain = await agent.get_brain(chat_id)
        if not brain.is_ready:
            await brain.ensure_ready(auto_topic="report")
        result = await brain.report(custom_prompt=custom_prompt)
        if result.success:
            await update.message.reply_text("Report generated! Check NotebookLM for the full report.")
        else:
            await update.message.reply_text(f"Report failed: {result.error}")

    async def _cmd_quiz(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        instructions = " ".join(ctx.args) if ctx.args else None
        await update.message.reply_text("Generating quiz... (30-90s)")
        brain = await agent.get_brain(chat_id)
        if not brain.is_ready:
            await brain.ensure_ready(auto_topic="quiz")
        result = await brain.quiz(instructions=instructions)
        if result.success:
            # Quiz content comes from NotebookLM chat
            answer = await brain.ask("Show me the quiz questions")
            await self._send_chunks(update, answer)
        else:
            await update.message.reply_text(f"Quiz failed: {result.error}")

    async def _cmd_mindmap(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        await update.message.reply_text("Generating mind map... (30-90s)")
        brain = await agent.get_brain(chat_id)
        if not brain.is_ready:
            await brain.ensure_ready(auto_topic="mindmap")
        result = await brain.mindmap()
        if result.success:
            await update.message.reply_text("Mind map generated! Check NotebookLM for the visual.")
        else:
            await update.message.reply_text(f"Mind map failed: {result.error}")

    async def _cmd_video(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        instructions = " ".join(ctx.args) if ctx.args else None
        await update.message.reply_text("Generating video... (5-15 min)")
        brain = await agent.get_brain(chat_id)
        if not brain.is_ready:
            await brain.ensure_ready(auto_topic="video")
        try:
            result = await brain.artifact_generator.generate_video(
                brain.notebook_id, instructions=instructions
            )
            if result.success:
                await update.message.reply_text("Video generated! Check NotebookLM to watch.")
            else:
                await update.message.reply_text(f"Video failed: {result.error}")
        except Exception as e:
            await update.message.reply_text(f"Video failed: {e}")

    async def _cmd_rename(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        new_name = " ".join(ctx.args) if ctx.args else ""
        if not new_name:
            await update.message.reply_text("Usage: /rename <new name>\n\nExample: /rename My Research Notebook")
            return
        brain = await agent.get_brain(chat_id)
        if not brain.notebook_id:
            await update.message.reply_text("No active notebook. Send a message first to create one.")
            return
        await brain.rename(new_name)
        await update.message.reply_text(f"Notebook renamed to: {new_name}")

    async def _cmd_notebooks(self, update, ctx):
        client = await get_client()
        notebooks = await client.notebooks.list()
        if not notebooks:
            await update.message.reply_text("No notebooks yet. Send a message to create one!")
            return
        lines = ["Your notebooks:"]
        for nb in notebooks[:20]:
            title = getattr(nb, "title", "Untitled")
            nb_id = getattr(nb, "id", "?")
            lines.append(f"  - {title} ({nb_id[:8]}...)")
        await update.message.reply_text("\n".join(lines))

    async def _cmd_status(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        brain = await agent.get_brain(chat_id)

        status_lines = ["Brain Status"]
        status_lines.append(f"  Ready: {'Yes' if brain.is_ready else 'No'}")
        status_lines.append(f"  Notebook: {brain.notebook_id[:12] + '...' if brain.notebook_id else 'None'}")

        if brain.notebook_id:
            try:
                sources_text = await brain.list_sources()
                lines = sources_text.split("\n")
                count_line = lines[0] if lines else "Sources: ?"
                status_lines.append(f"  {count_line}")
            except Exception:
                status_lines.append("  Sources: (unable to count)")

        await update.message.reply_text("\n".join(status_lines))

    async def _cmd_reset(self, update, ctx):
        agent = await self._init_agent()
        chat_id = str(update.effective_chat.id)
        await agent.reset(chat_id)
        await update.message.reply_text("Brain reset! Next message will create a fresh notebook.")

    # ─── Callback Handler (Inline Buttons) ───

    async def _callback_handler(self, update, ctx):
        """Handle inline keyboard button presses."""
        query = update.callback_query
        await query.answer()

        data = query.data
        if data == "cmd_start":
            await query.edit_message_text(
                WELCOME,
                reply_markup=self._main_menu_keyboard(),
            )
        elif data == "cmd_help":
            await query.edit_message_text(HELP_TEXT)
        elif data == "cmd_research":
            await query.edit_message_text("Type: /research <topic>\n\nExample: /research quantum computing")
        elif data == "cmd_ask":
            await query.edit_message_text("Type: /ask <question>\n\nOr just type your question directly!")
        elif data == "cmd_add":
            await query.edit_message_text("Type: /add <url>\n\nExample: /add https://en.wikipedia.org/wiki/Python")
        elif data == "cmd_sources":
            agent = await self._init_agent()
            chat_id = str(update.effective_chat.id)
            result = await agent.run("list sources", chat_id=chat_id)
            await query.edit_message_text(result.answer[:4096])
        elif data == "cmd_podcast":
            await query.edit_message_text("Type: /podcast\nOr: /podcast <custom instructions>")
        elif data == "cmd_report":
            await query.edit_message_text("Type: /report\nOr: /report <custom prompt>")
        elif data == "cmd_quiz":
            await query.edit_message_text("Type: /quiz\nOr: /quiz <topic>")
        elif data == "cmd_mindmap":
            await query.edit_message_text("Type: /mindmap")
        elif data == "cmd_video":
            await query.edit_message_text("Type: /video\nOr: /video <instructions>")
        elif data == "cmd_status":
            agent = await self._init_agent()
            chat_id = str(update.effective_chat.id)
            brain = await agent.get_brain(chat_id)
            status = f"Brain Status\n  Ready: {'Yes' if brain.is_ready else 'No'}\n  Notebook: {brain.notebook_id[:12] + '...' if brain.notebook_id else 'None'}"
            await query.edit_message_text(status)

    # ─── Text Message Handler ───

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

    # ─── Utility ───

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

    async def _set_commands(self, ctx):
        """Register bot commands with Telegram so they appear in the menu."""
        try:
            await ctx.bot.set_my_commands(BOT_COMMANDS)
            logger.info("Registered %d bot commands", len(BOT_COMMANDS))
        except Exception as e:
            logger.warning(f"Could not set bot commands: {e}")

    async def start(self):
        self._app = self._build_app()
        await self._app.initialize()
        await self._app.start()
        # Register commands on startup
        await self._set_commands(self._app)
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