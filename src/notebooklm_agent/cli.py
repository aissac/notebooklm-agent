"""CLI entry point for notebooklm-agent."""

import asyncio
import click
import logging

from notebooklm_agent.agent import Agent, AgentMode
from notebooklm_agent.auth import get_client, close_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def main(debug):
    """NotebookLM Agent - Zero-token AI powered by Google NotebookLM."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.option("--notebook", "-n", help="Existing notebook ID")
@click.option("--title", "-t", default="NotebookLM Agent", help="Notebook title")
@click.argument("query")
def run(query, notebook, title):
    """Ask the agent a question."""
    async def _run():
        try:
            client = await get_client()
            agent = Agent(client, default_notebook_id=notebook)
            result = await agent.run(query, chat_id="cli")
            click.echo(result.answer)
        finally:
            await close_pool()
    asyncio.run(_run())


@main.command()
@click.option("--mode", type=click.Choice(["fast", "deep"]), default="fast")
@click.argument("topic")
def research(topic, mode):
    """Research a topic using NotebookLM."""
    async def _run():
        try:
            client = await get_client()
            agent = Agent(client)
            result = await agent.run(f"research {topic}", chat_id="cli")
            click.echo(result.answer)
        finally:
            await close_pool()
    asyncio.run(_run())


@main.command(name="serve")
@click.option("--gateway", type=click.Choice(["telegram", "cli", "web"]), default="telegram")
@click.option("--token", envvar="TELEGRAM_BOT_TOKEN", help="Bot token for Telegram gateway")
@click.option("--max-steps", default=5, help="Max agent steps")
def serve(gateway, token, max_steps):
    """Start a gateway server."""
    async def _serve():
        try:
            if gateway == "telegram":
                if not token:
                    click.echo("Error: TELEGRAM_BOT_TOKEN required for Telegram gateway")
                    click.echo("Set it via --token or TELEGRAM_BOT_TOKEN env var")
                    return
                from notebooklm_agent.gateways.telegram import TelegramGateway
                gw = TelegramGateway(token=token, max_steps=max_steps)
                click.echo(f"Starting Telegram gateway...")
                await gw.run()
            elif gateway == "web":
                click.echo("Web gateway coming soon!")
            else:
                click.echo("CLI gateway: just use "nlm-agent run <query>"")
        finally:
            await close_pool()
    asyncio.run(_serve())


@main.command()
def login():
    """Login to Google NotebookLM (opens browser)."""
    click.echo("Running notebooklm login...")
    import subprocess
    subprocess.run(["notebooklm", "login"], check=True)


@main.command()
def notebooks():
    """List your NotebookLM notebooks."""
    async def _list():
        try:
            client = await get_client()
            nbs = await client.notebooks.list()
            if not nbs:
                click.echo("No notebooks found.")
                return
            for nb in nbs:
                title = getattr(nb, "title", "Untitled")
                nb_id = getattr(nb, "id", "?")
                click.echo(f"  {title} ({nb_id})")
        finally:
            await close_pool()
    asyncio.run(_list())


if __name__ == "__main__":
    main()
