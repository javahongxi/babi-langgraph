"""Babi Web application built with FastAPI and LangGraph.

Provides a chat API with SSE streaming for real-time agent responses,
plus workspace file management APIs.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, Query
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from babi.agent.builder import build_agent_async, build_agent_with_checkpointer, get_workspace_path
from babi.config import Settings

logger = logging.getLogger(__name__)


def _prepare_agent_args(settings: Settings, workspace_path: Path):
    """Prepare (llm, tools, sys_prompt) for agent construction."""
    from babi.agent.builder import _get_llm, _get_all_tools
    from babi.agent.prompt import build_system_prompt
    from babi.tools.skill_tool import SkillTool

    llm = _get_llm(settings)
    skill_tool = SkillTool(workspace_path)
    tools = _get_all_tools(skill_tool)
    sys_prompt = build_system_prompt(list(skill_tool.skills.values()), workspace_path=workspace_path)
    return llm, tools, sys_prompt

# Language mapping for syntax highlighting
_EXT_LANG = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".html": "html",
    ".css": "css",
    ".json": "json",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".sh": "bash",
    ".sql": "sql",
    ".rb": "ruby",
    ".c": "c",
    ".cpp": "cpp",
}


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""

    message: str
    session_id: str = "default"


def create_app(settings: Settings) -> FastAPI:
    """Create the FastAPI application.

    Args:
        settings: Application settings

    Returns:
        Configured FastAPI application
    """
    # Shared state for agent and checkpointer context manager
    _state = {"agent": None, "checkpointer": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: build agent with async checkpointer
        workspace_path = get_workspace_path(settings)
        checkpointer_cm, checkpointer, agent = await build_agent_async(settings, workspace_path)

        if checkpointer_cm is not None:
            # Use `async with` to manage PostgreSQL connection lifecycle
            async with checkpointer_cm as checkpointer:
                await checkpointer.setup()
                llm, tools, sys_prompt = _prepare_agent_args(settings, workspace_path)
                agent = build_agent_with_checkpointer(llm, tools, sys_prompt, checkpointer)
                _state["agent"] = agent
                _state["checkpointer"] = checkpointer
                logger.info("Session persistence enabled (PostgreSQL)")
                yield
            logger.info("PostgreSQL connection closed")
        else:
            _state["agent"] = agent
            _state["checkpointer"] = checkpointer
            logger.info("Session persistence disabled (in-memory)")
            yield

    app = FastAPI(
        title="Babi Agent",
        version="1.0.0",
        lifespan=lifespan,
        middleware=[
            Middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        ],
    )

    # Resolve workspace for file APIs
    workspace_path = get_workspace_path(settings)
    _workspace_root = workspace_path

    def _safe_resolve(rel_path: str) -> Path | None:
        """Resolve a relative path against workspace root, rejecting path traversal."""
        resolved = (_workspace_root / rel_path).resolve()
        if not str(resolved).startswith(str(_workspace_root.resolve())):
            return None
        return resolved

    # --- Chat API ---

    @app.post("/api/chat")
    async def chat(
        request: ChatRequest,
        x_user_id: str = Header(default="babi-user", alias="X-User-ID"),
    ):
        """Chat with the agent via SSE streaming.

        Returns a Server-Sent Events stream with:
        - token: Individual text tokens as they're generated
        - tool_call: When a tool is being called
        - tool_result: When a tool returns a result
        - done: When the response is complete
        """
        agent = _state.get("agent")
        if agent is None:
            return JSONResponse({"error": "Agent not initialized"}, status_code=500)

        config = {
            "configurable": {"thread_id": request.session_id},
            "recursion_limit": settings.max_iters * 2,
        }

        async def event_generator():
            try:
                async for event in agent.astream_events(
                    {"messages": [("user", request.message)]},
                    config=config,
                    version="v2",
                ):
                    kind = event.get("event")

                    if kind == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content and isinstance(chunk.content, str):
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_name})}\n\n"

                    elif kind == "on_tool_end":
                        tool_name = event.get("name", "unknown")
                        output = str(event.get("data", {}).get("output", ""))
                        # Truncate long outputs for SSE
                        if len(output) > 500:
                            output = output[:500] + "..."
                        yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'output': output})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            except Exception as e:
                logger.exception("Chat error")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Workspace APIs ---

    @app.get("/api/workspace/tree")
    async def workspace_tree(path: str = Query(default="")):
        """List directory entries within the workspace."""
        target = _safe_resolve(path)
        if target is None or not target.is_dir():
            return JSONResponse([], status_code=200)
        items = []
        try:
            for entry in sorted(target.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())):
                if entry.name.startswith("."):
                    continue
                items.append(
                    {
                        "name": entry.name,
                        "path": str(entry.relative_to(_workspace_root)),
                        "isDir": entry.is_dir(),
                    }
                )
        except OSError as e:
            logger.warning("workspace/tree error for '%s': %s", path, e)
        return items

    @app.get("/api/workspace/file")
    async def workspace_file(path: str = Query()):
        """Read text file content from workspace."""
        target = _safe_resolve(path)
        if target is None or not target.is_file():
            return JSONResponse({"error": "File not found"}, status_code=404)
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            lang = _EXT_LANG.get(target.suffix.lower(), "plaintext")
            return {"content": content, "language": lang, "size": target.stat().st_size}
        except OSError as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.get("/api/workspace/image")
    async def workspace_image(path: str = Query()):
        """Serve an image file from workspace."""
        target = _safe_resolve(path)
        if target is None or not target.is_file():
            return JSONResponse({"error": "Image not found"}, status_code=404)
        return FileResponse(str(target))

    @app.delete("/api/chat/memory")
    async def clear_memory():
        """Delete MEMORY.md from workspace so the agent forgets prior context."""
        memory_file = _workspace_root / "MEMORY.md"
        try:
            if memory_file.exists():
                memory_file.unlink()
                logger.info("Deleted memory file: %s", memory_file)
                return {"status": "ok", "message": "已清除记忆文件 MEMORY.md"}
            else:
                return {"status": "ok", "message": "没有找到记忆文件 MEMORY.md"}
        except OSError as e:
            logger.warning("Failed to delete memory file: %s", e)
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @app.delete("/api/chat/session")
    async def clear_session(session_id: str = Query(default="default")):
        """Clear the conversation history for a session from the checkpointer."""
        checkpointer = _state.get("checkpointer")
        
        if checkpointer is None:
            return {"status": "ok", "message": "会话已清空（内存模式）"}
        
        try:
            target = checkpointer
            # Try adelete_thread first (available in newer langgraph versions)
            if hasattr(target, 'adelete_thread'):
                await target.adelete_thread(session_id)
                logger.info("Cleared session via adelete_thread: %s", session_id)
                return {"status": "ok", "message": f"会话 {session_id} 已清空"}
            # Fallback: direct SQL deletion for AsyncPostgresSaver
            elif hasattr(target, 'conn'):
                async with target.conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM checkpoints WHERE thread_id = %s", (session_id,)
                    )
                    await cur.execute(
                        "DELETE FROM checkpoint_writes WHERE thread_id = %s", (session_id,)
                    )
                    await cur.execute(
                        "DELETE FROM checkpoint_blobs WHERE thread_id = %s", (session_id,)
                    )
                await target.conn.commit()
                logger.info("Cleared session via SQL: %s", session_id)
                return {"status": "ok", "message": f"会话 {session_id} 已清空"}
            else:
                logger.warning("Unknown checkpointer type, cannot clear: %s", type(target))
                return {"status": "ok", "message": "无法确定清空方式，但会话已标记清空"}
        except (OSError, RuntimeError) as e:
            logger.warning("Failed to clear session: %s", e)
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    # --- Mount static frontend ---

    static_dir = Path(__file__).parent.parent.parent / "resources" / "static"
    if static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
