"""CLI entry point for the Babi Agent.

Uses Click for argument parsing and provides an interactive REPL
for chatting with the agent from the terminal.

Usage:
    export DASHSCOPE_API_KEY=your_key
    babi                          # workspace = current directory
    babi --workspace ~/my-project
"""

from __future__ import annotations

import asyncio
import logging
import os

from contextlib import asynccontextmanager

import click

from babi.config import get_settings

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--workspace",
    default=None,
    help="Workspace directory (default: current directory)",
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
    elif not web:
        settings.workspace = os.getcwd()
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


@asynccontextmanager
async def _agent_context(checkpointer_cm, checkpointer, agent, settings, workspace_path):
    """Async context manager for checkpointer lifecycle.

    If a checkpointer CM is provided, enters it and builds the agent.
    Otherwise yields the pre-built agent directly.
    """
    from babi.agent.builder import (
        _get_all_tools,
        _get_llm,
        build_agent_with_checkpointer,
    )
    from babi.agent.prompt import build_system_prompt
    from babi.tools.skill_tool import SkillTool

    if checkpointer_cm is not None:
        async with checkpointer_cm as checkpointer:
            await checkpointer.setup()
            llm = _get_llm(settings)
            skill_tool = SkillTool(workspace_path)
            sys_prompt = build_system_prompt(list(skill_tool.skills.values()), workspace_path=workspace_path)
            tools = _get_all_tools(skill_tool)
            agent = build_agent_with_checkpointer(llm, tools, sys_prompt, checkpointer)
            yield agent
    else:
        yield agent


async def _cli_repl(settings) -> None:
    """Interactive CLI read-eval-print loop."""

    from babi.agent.builder import (
        _get_all_tools,
        _get_llm,
        build_agent_async,
        build_agent_with_checkpointer,
        get_workspace_path,
    )
    from babi.agent.prompt import build_system_prompt
    from babi.tools.skill_tool import SkillTool

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
    checkpointer_cm, checkpointer, agent = await build_agent_async(settings, workspace_path)

    if checkpointer_cm:
        print("  ✓ Session persistence enabled (PostgreSQL)")
    else:
        print("  ⚠ Session persistence disabled (in-memory only)")
    print()

    # Session config for checkpointer and recursion limit
    config = {
        "configurable": {"thread_id": "cli-session"},
        "recursion_limit": settings.max_iters * 2,
    }

    # Use `async with` to manage the checkpointer lifecycle
    async with _agent_context(checkpointer_cm, checkpointer, agent, settings, workspace_path) as ctx_agent:
        agent = ctx_agent

        # REPL loop
        loop = asyncio.get_running_loop()
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
                async for event in agent.astream_events(
                    {"messages": [("user", user_input)]},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_chat_model_stream":
                        content = event.get("data", {}).get("chunk")
                        if content and hasattr(content, "content") and content.content:
                            if isinstance(content.content, str):
                                print(content.content, end="", flush=True)

                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        print(f"\n  [Tool: {tool_name}]", end="", flush=True)

                print("\n")

            except Exception as e:
                print(f"\nError: {e}\n")
                logger.exception("Agent error")


if __name__ == "__main__":
    main()
