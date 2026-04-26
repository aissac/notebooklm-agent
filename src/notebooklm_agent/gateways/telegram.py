"""Telegram Gateway - One brain per user, forever.

No notebook routing. No multi-notebook management. Each user gets one notebook
that grows smarter over time. The notebook IS the memory.
"""

import asyncio
import logging
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from notebooklm_agent.brain.core import Brain
from notebooklm_agent.brain.user_brain import UserBrain
from notebooklm_agent.auth import get_client, close_pool
from notebooklm_agent.utils.text import sanitize_for_display, chunk_for_telegram

logger = logging.getLogger(__name__)
MAX_MSG_LEN = 4096
BOT_COMMANDS = [
    BotCommand("start", "Start the bot"),
    BotCommand("help", "Show command menu"),
    BotCommand("ask", "Ask a question"),
    BotCommand("research", "Quick web research"),
    BotCommand("deep", "Deep web research"),
    BotCommand("add", "Add URL source"),
    BotCommand("addtext", "Add text source"),
    BotCommand("sources", "List sources"),
    BotCommand("podcast", "Generate podcast"),
    BotCommand("report", "Generate report"),
    BotCommand("quiz", "Generate quiz"),
    BotCommand("mindmap", "Generate mind map"),
    BotCommand("video", "Generate video"),
    BotCommand("status", "Brain status"),
    BotCommand("prune", "Prune research sources"),
    BotCommand("reset", "Delete your brain"),
]

WELCOME = "NLM Agent - Zero-token AI\n\nYour brain grows smarter over time.\nEvery question adds knowledge.\n\nJust type a question or /help for commands."

HELP_TEXT = """Commands:
/research <topic> - Quick web research (~90s)
/deep <topic> - Deep research (5-15 min)
/add <url> - Add URL source
/addtext <title> | <content> - Add text source
/sources - List your sources
/ask <question> - Ask about sources
/prune - Remove research sources
/podcast - Generate podcast
/report [prompt] - Generate report
/quiz [topic] - Generate quiz
/mindmap - Generate mind map
/video [instructions] - Generate video
/status - Brain status
/reset CONFIRM - Delete your brain"""


class TelegramGateway:
    """Async-native Telegram gateway with one-brain-per-user architecture."""

    def __init__(self, token: str, max_steps: int = 5):
        self.token = token
        self.max_steps = max_steps
        self._user_brain = None
        self._app = None

    async def _get_brain(self, user_id: int) -> Brain:
        """Get or create the user brain. One brain per user, forever."""
        if not self._user_brain:
            client = await get_client()
            self._user_brain = UserBrain(client)
        info = await self._user_brain.get_or_create(user_id)
        brain = Brain(self._user_brain.client, notebook_id=info["notebook_id"])
        await brain.ensure_ready()
        return brain

    async def _send(self, update, text, **kw):
        for chunk in chunk_for_telegram(text, MAX_MSG_LEN):
            await update.message.reply_text(chunk, **kw)

    async def _send_typing(self, update):
        try:
            await update.message.chat.send_action("typing")
        except Exception:
            pass

    async def cmd_start(self, update, context):
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Research", callback_data="research"),
             InlineKeyboardButton("Ask", callback_data="ask")],
            [InlineKeyboardButton("Sources", callback_data="sources"),
             InlineKeyboardButton("Status", callback_data="status")],
        ])
        await update.message.reply_text(WELCOME, reply_markup=keyboard)

    async def cmd_help(self, update, context):
        await self._send(update, HELP_TEXT)

    async def cmd_status(self, update, context):
        user_id = update.effective_user.id
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            sources = await brain.list_sources()
            count = len(sources)
            protected = sum(1 for s in sources if s.get("protected"))
            lines = [
                "Brain Status", "",
                f"Notebook: {brain.notebook_id}",
                f"Total sources: {count}",
                f"Protected: {protected}",
                f"Research: {count - protected}",
                f"Ready: {brain.is_ready}",
            ]
            if count > 35:
                lines.append(f"Warning: {count}/50 sources. Use /prune.")
            await self._send(update, chr(10).join(lines))
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_ask(self, update, context):
        query = " ".join(context.args) if context.args else None
        if not query:
            await self._send(update, "Usage: /ask <question>")
            return
        user_id = update.effective_user.id
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            answer = await brain.ask(query)
            await self._send(update, answer)
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_research(self, update, context):
        query = " ".join(context.args) if context.args else None
        if not query:
            await self._send(update, "Usage: /research <topic>")
            return
        user_id = update.effective_user.id
        await self._send(update, f"Researching: {query}...")
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            result = await brain.research(query, mode="fast")
            if result.success:
                await self._send(update, f"Done! Added {result.source_count} sources. Ask about {query}!")
            else:
                await self._send(update, f"Failed: {result.error}")
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_deep(self, update, context):
        query = " ".join(context.args) if context.args else None
        if not query:
            await self._send(update, "Usage: /deep <topic>")
            return
        user_id = update.effective_user.id
        await self._send(update, f"Deep research: {query} (5-15 min)...")
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            result = await brain.research(query, mode="deep")
            if result.success:
                await self._send(update, f"Done! Added {result.source_count} sources.")
            else:
                await self._send(update, f"Failed: {result.error}")
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_add(self, update, context):
        url = " ".join(context.args) if context.args else None
        if not url:
            await self._send(update, "Usage: /add <url>")
            return
        user_id = update.effective_user.id
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            await brain.add_source(url)
            await self._send(update, f"Added: {url}")
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_addtext(self, update, context):
        text = " ".join(context.args) if context.args else None
        if not text or "|" not in text:
            await self._send(update, "Usage: /addtext <title> | <content>")
            return
        title, content_text = text.split("|", 1)
        user_id = update.effective_user.id
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            await brain.add_text(title.strip(), content_text.strip())
            await self._send(update, f"Added: {title.strip()}")
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_sources(self, update, context):
        user_id = update.effective_user.id
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            sources = await brain.list_sources()
            if not sources:
                await self._send(update, "No sources yet. Use /research or /add!")
                return
            lines = [f"Sources ({len(sources)}):"]
            for i, s in enumerate(sources, 1):
                marker = " [!]" if s["protected"] else ""
                lines.append(f"{i}. {s['title']}{marker}")
            await self._send(update, chr(10).join(lines))
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def _artifact(self, update, context, name, method, has_instr=False):
        user_id = update.effective_user.id
        args = " ".join(context.args) if context.args else None
        await self._send(update, f"Generating {name}...")
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            kw = {}
            if has_instr and args:
                kw["instructions"] = args
            elif name == "report" and args:
                kw["prompt"] = args
            elif name == "quiz" and args:
                kw["topic"] = args
            result = await getattr(brain, method)(**kw)
            if result.success:
                await self._send(update, f"{name.title()} ready! ID: {result.artifact_id}")
            else:
                await self._send(update, f"Failed: {result.error}")
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_podcast(self, update, context):
        await self._artifact(update, context, "podcast", "podcast", has_instr=True)

    async def cmd_report(self, update, context):
        await self._artifact(update, context, "report", "report")

    async def cmd_quiz(self, update, context):
        await self._artifact(update, context, "quiz", "quiz")

    async def cmd_mindmap(self, update, context):
        await self._artifact(update, context, "mindmap", "mindmap")

    async def cmd_video(self, update, context):
        await self._artifact(update, context, "video", "video", has_instr=True)

    async def cmd_prune(self, update, context):
        user_id = update.effective_user.id
        await self._send_typing(update)
        try:
            if not self._user_brain:
                client = await get_client()
                self._user_brain = UserBrain(client)
            pruned = await self._user_brain.prune_research_sources(user_id)
            await self._send(update, f"Pruned {pruned} research sources.")
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def cmd_reset(self, update, context):
        user_id = update.effective_user.id
        if not context.args or context.args[0] != "CONFIRM":
            await self._send(update, "WARNING: Deletes your brain! Send /reset CONFIRM.")
            return
        await self._send_typing(update)
        try:
            if not self._user_brain:
                client = await get_client()
                self._user_brain = UserBrain(client)
            deleted = await self._user_brain.delete_brain(user_id)
            msg = "Brain deleted. Next message creates fresh one." if deleted else "No brain found."
            await self._send(update, msg)
        except Exception as e:
            await self._send(update, f"Error: {e}")

    async def handle_message(self, update, context):
        """Handle non-command messages with auto-research."""
        text = update.message.text
        if not text:
            return
        user_id = update.effective_user.id
        await self._send_typing(update)
        try:
            brain = await self._get_brain(user_id)
            source_count = await brain.source_count()
            if source_count <= 1:
                await self._send(update, f"Learning: {text[:50]}... (auto-researching)")
                result = await brain.research(text[:200], mode="fast")
                if not result.success:
                    await self._send(update, "Auto-research failed. Answering anyway...")
            answer = await brain.ask(text)
            await self._send(update, answer)
            if self._user_brain:
                await self._user_brain.enforce_source_cap(user_id)
        except Exception as e:
            logger.error(f"Message handler error: {e}")
            await self._send(update, f"Error: {e}")

    async def handle_callback(self, update, context):
        query = update.callback_query
        await query.answer()
        if query.data == "research":
            await query.message.reply_text("Use /research <topic>")
        elif query.data == "ask":
            await query.message.reply_text("Just type your question!")
        elif query.data == "sources":
            await self.cmd_sources(update, context)
        elif query.data == "status":
            await self.cmd_status(update, context)

    def _register_handlers(self):
        for cmd, handler in [
            ("start", self.cmd_start),
            ("help", self.cmd_help),
            ("ask", self.cmd_ask),
            ("research", self.cmd_research),
            ("deep", self.cmd_deep),
            ("add", self.cmd_add),
            ("addtext", self.cmd_addtext),
            ("sources", self.cmd_sources),
            ("podcast", self.cmd_podcast),
            ("report", self.cmd_report),
            ("quiz", self.cmd_quiz),
            ("mindmap", self.cmd_mindmap),
            ("video", self.cmd_video),
            ("status", self.cmd_status),
            ("prune", self.cmd_prune),
            ("reset", self.cmd_reset),
        ]:
            self._app.add_handler(CommandHandler(cmd, handler))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        self._app.add_handler(CallbackQueryHandler(self.handle_callback))

    async def _set_commands(self):
        await self._app.bot.set_my_commands(BOT_COMMANDS)

    async def run(self):
        logger.info("Starting Telegram gateway...")
        self._app = Application.builder().token(self.token).build()
        self._register_handlers()
        await self._set_commands()
        logger.info(f"Registered {len(BOT_COMMANDS)} commands")
        async with self._app:
            await self._app.initialize()
            await self._app.start()
            logger.info("Telegram gateway started")
            await self._app.updater.start_polling(drop_pending_updates=False)
            try:
                import signal
                ev = asyncio.Event()

                def _shutdown(sig, frame):
                    ev.set()

                signal.signal(signal.SIGINT, _shutdown)
                signal.signal(signal.SIGTERM, _shutdown)
                await ev.wait()
            except (KeyboardInterrupt, SystemExit):
                pass
            finally:
                logger.info("Shutting down...")
                await self._app.updater.stop()
                await self._app.stop()
                await close_pool()
