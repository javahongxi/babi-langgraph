"""CLI entry point for the Babi Agent.

Uses Click for argument parsing and provides an interactive REPL
for chatting with the agent from the terminal.

Usage:
    export DASHSCOPE_API_KEY=your_key
    babi                          # default workspace ~/babi-langgraph-workspace
    babi --workspace ~/my-project
"""

from __future__ import annotations

import asyncio
import logging

import click

from babi.config import get_settings

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--workspace",
    default=None,
    help="Workspace directory (default: ~/babi-langgraph-workspace)",
)
@click.option(
    "--model",
    default=None,
    help="Model name override (default: from config/env)",
)
@click.option(
    "--host",
    default=None,
    help="Web server host (default: 127.0.0.1)",
)
@click.option(
    "--port",
    default=None,
    type=int,
    help="Web server port (default: 8900)",
)
@click.option(
    "--web",
    is_flag=True,
    help="Start web server instead of CLI mode",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def main(
    workspace: str | None,
    model: str | None,
    host: str | None,
    port: int | None,
    web: bool,
    verbose: bool,
) -> None:
    """Babi Agent — AI-powered coding assistant.

    Run in CLI mode (default) or start a web server with --web.
    """
    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    babi_log_level = log_level if web else logging.WARNING
    logging.getLogger("babi").setLevel(babi_log_level)

    settings = get_settings()

    # Override settings from CLI args
    if workspace:
        settings.workspace = workspace
    if model:
        settings.model_name = model
    if host:
        settings.host = host
    if port:
        settings.port = port

    # Validate API key early
    _ = settings.dashscope_api_key

    if web:
        _start_web(settings)
    else:
        asyncio.run(_cli_repl(settings))


def _start_web(settings) -> None:
    """Start the web server via uvicorn."""
    import uvicorn

    app = _create_web_app(settings)
    print()
    print("=" * 50)
    print("  Babi Web Server (LangGraph)")
    print(f"  http://{settings.host}:{settings.port}")
    print(f"  API docs: http://{settings.host}:{settings.port}/docs")
    print("=" * 50)
    print()
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


def _create_web_app(settings):
    """Create the FastAPI web application."""
    from babi.web.app import create_app

    return create_app(settings)


async def _cli_repl(settings) -> None:
    """Interactive CLI read-eval-print loop."""

    from babi.agent.builder import build_agent_async, get_workspace_path

    workspace_path = get_workspace_path(settings)

    print()
    print("=" * 60)
    print("  Babi Agent — Powered by LangGraph")
    print("=" * 60)
    print(f"  Workspace: {workspace_path}")
    print("  Built-in tools: read_file, write_file, edit_file, grep_files, glob_files, shell_execute")
    print(
        "  Custom tools: fetch_url, http_request, web_search, github_api_request, github_pinned_repos, list_skills, use_skill"
    )
    print("  Type 'exit' to quit.")
    print()

    # Build agent with async checkpointer (PostgreSQL persistence if configured)
    checkpointer_cm, _checkpointer, agent = await build_agent_async(settings, workspace_path)

    if checkpointer_cm:
        print("  ✓ Session persistence enabled (PostgreSQL)")
    else:
        print("  ⚠ Session persistence disabled (in-memory only)")
    print()

    # Session config for checkpointer and recursion limit
    config = {
        "configurable": {"thread_id": "cli-session"},
        "recursion_limit": settings.max_iters * 2,  # each iter = agent + tool node
    }

    # REPL loop
    loop = asyncio.get_event_loop()
    try:
        while True:
            try:
                user_input = await loop.run_in_executor(None, lambda: input("You: "))
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            user_input = user_input.strip()
            if not user_input:
                continue
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            print("\nBabiAgent: ", end="", flush=True)

            try:
                # Stream the response token by token
                async for event in agent.astream_events(
                    {"messages": [("user", user_input)]},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_chat_model_stream":
                        # Token-level streaming from the LLM
                        content = event.get("data", {}).get("chunk")
                        if content and hasattr(content, "content") and content.content:
                            # Skip tool call chunks
                            if isinstance(content.content, str):
                                print(content.content, end="", flush=True)

                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        print(f"\n  [Tool: {tool_name}]", end="", flush=True)

                print("\n")

            except Exception as e:
                print(f"\nError: {e}\n")
                logger.exception("Agent error")
    finally:
        # Clean up checkpointer connection by exiting the context manager
        if checkpointer_cm:
            try:
                await checkpointer_cm.__aexit__(None, None, None)
            except Exception as e:
                logger.warning("Error closing checkpointer: %s", e)


if __name__ == "__main__":
    main()
