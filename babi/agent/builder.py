"""Agent builder: assembles the BabiAgent with LangGraph.

Uses ChatOpenAI with DashScope compatible API for Qwen models,
and LangGraph's create_react_agent for the ReAct loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

from babi.agent.prompt import build_system_prompt
from babi.config import Settings
from babi.tools.fetch_url import fetch_url
from babi.tools.filesystem import (
    edit_file,
    glob_files,
    grep_files,
    read_file,
    shell_execute,
    write_file,
)
from babi.tools.github_api import github_api_request, github_pinned_repos
from babi.tools.http_request import http_request
from babi.tools.skill_tool import SkillTool
from babi.tools.web_search import web_search
from babi.utils.helpers import init_agents_md, resolve_workspace

logger = logging.getLogger(__name__)


def _get_llm(settings: Settings):
    """Create the LLM instance using ChatOpenAI + DashScope compatible API.

    Args:
        settings: Application settings

    Returns:
        ChatOpenAI instance configured for DashScope
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.dashscope_api_key,
        openai_api_base=settings.dashscope_base_url,
        streaming=True,
        max_retries=settings.max_retries,
        temperature=0,
    )
    return llm


def _get_all_tools(skill_tool: SkillTool) -> list:
    """Collect all tools for the agent.

    Args:
        skill_tool: SkillTool instance for skill-related tools

    Returns:
        List of LangChain tool instances
    """
    from langchain_core.tools import tool as tool_decorator

    # Wrap SkillTool methods as @tool functions
    @tool_decorator
    def list_skills() -> str:
        """List all available skills. Returns skill names and descriptions.

        Call this before use_skill to discover what skills are available.
        """
        return skill_tool.list_skills()

    @tool_decorator
    def use_skill(skill_name: str) -> str:
        """Activate a skill by name and get its full instructions.

        The instructions will guide you through the workflow.
        Call list_skills first to see available skills.

        Args:
            skill_name: Name of the skill to activate.
        """
        return skill_tool.use_skill(skill_name)

    return [
        # Filesystem tools
        read_file,
        write_file,
        edit_file,
        grep_files,
        glob_files,
        shell_execute,
        # Network tools
        fetch_url,
        http_request,
        web_search,
        # GitHub tools
        github_api_request,
        github_pinned_repos,
        # Skill tools
        list_skills,
        use_skill,
    ]


def build_agent(settings: Settings, workspace_path: Path | None = None):
    """Build and configure the BabiAgent using LangGraph.

    Note: For PostgreSQL persistence, use build_agent_async() instead.
    This sync version uses MemorySaver only.

    Args:
        settings: Application settings
        workspace_path: Optional workspace path override

    Returns:
        Compiled LangGraph agent (CompiledGraph) ready for use
    """
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    # Resolve workspace
    if workspace_path is None:
        workspace_path = resolve_workspace(settings.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Initialize AGENTS.md
    init_agents_md(workspace_path)
    logger.info("Agent workspace: %s", workspace_path)

    # Build system prompt with skills
    skill_tool = SkillTool()
    skills_list = list(skill_tool.skills.values())
    sys_prompt = build_system_prompt(skills_list, workspace_path=workspace_path)

    # Get LLM and tools
    llm = _get_llm(settings)
    tools = _get_all_tools(skill_tool)

    logger.info(
        "Registered tools: read_file, write_file, edit_file, grep_files, glob_files, "
        "shell_execute, fetch_url, http_request, web_search, "
        "github_api_request, github_pinned_repos, list_skills, use_skill"
    )

    # Sync version uses MemorySaver only
    checkpointer = MemorySaver()
    logger.info("Using MemorySaver checkpointer (sync mode)")

    # Create the ReAct agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=sys_prompt,
    )

    logger.info("BabiAgent built successfully with model: %s", settings.model_name)
    return agent


async def build_agent_async(settings: Settings, workspace_path: Path | None = None):
    """Build and configure the BabiAgent with async PostgreSQL persistence.

    This is the recommended way to build the agent when PostgreSQL is configured.
    Uses AsyncPostgresSaver for session persistence across restarts.

    Args:
        settings: Application settings
        workspace_path: Optional workspace path override

    Returns:
        Tuple of (checkpointer_cm_or_none, checkpointer, agent)
        - checkpointer_cm_or_none: The async context manager (for lifecycle management) or None if using MemorySaver
        - checkpointer: The actual checkpointer instance (for operations like clearing sessions)
        - agent: The compiled LangGraph agent
        """
    from langgraph.prebuilt import create_react_agent

    # Resolve workspace
    if workspace_path is None:
        workspace_path = resolve_workspace(settings.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    # Initialize AGENTS.md
    init_agents_md(workspace_path)
    logger.info("Agent workspace: %s", workspace_path)

    # Build system prompt with skills
    skill_tool = SkillTool()
    skills_list = list(skill_tool.skills.values())
    sys_prompt = build_system_prompt(skills_list, workspace_path=workspace_path)

    # Get LLM and tools
    llm = _get_llm(settings)
    tools = _get_all_tools(skill_tool)

    logger.info(
        "Registered tools: read_file, write_file, edit_file, grep_files, glob_files, "
        "shell_execute, fetch_url, http_request, web_search, "
        "github_api_request, github_pinned_repos, list_skills, use_skill"
    )

    # Build checkpointer
    pg_dsn = settings.pg_dsn
    if pg_dsn:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            # Create the context manager - caller must enter and exit it
            checkpointer_cm = AsyncPostgresSaver.from_conn_string(pg_dsn)
            # Return the CM itself, caller will enter it
            logger.info("Using AsyncPostgresSaver checkpointer - sessions will persist!")
            # We need to enter the context manager here and return both the CM and the entered checkpointer
            checkpointer = await checkpointer_cm.__aenter__()
            await checkpointer.setup()
            agent = _build_agent_with_checkpointer(llm, tools, sys_prompt, checkpointer)
            return checkpointer_cm, checkpointer, agent
        except Exception as e:
            logger.warning("Failed to create AsyncPostgresSaver, falling back to MemorySaver: %s", e)

    # Fallback to in-memory
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    logger.info("Using MemorySaver checkpointer (in-memory)")
    return None, checkpointer, _build_agent_with_checkpointer(llm, tools, sys_prompt, checkpointer)


def _build_agent_with_checkpointer(llm, tools, sys_prompt, checkpointer):
    """Internal helper to create the agent with a given checkpointer."""
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        model=llm,
        tools=tools,
        checkpointer=checkpointer,
        prompt=sys_prompt,
    )


def get_workspace_path(settings: Settings) -> Path:
    """Resolve and prepare the workspace directory.

    Args:
        settings: Application settings

    Returns:
        Resolved workspace Path
    """
    workspace_path = resolve_workspace(settings.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    init_agents_md(workspace_path)
    return workspace_path
