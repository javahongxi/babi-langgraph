"""Filesystem tools for the Babi Agent.

Provides read_file, write_file, edit_file, grep_files, glob_files, and shell_execute
as LangChain @tool decorated functions for use with LangGraph agents.
"""

from __future__ import annotations

import os
import subprocess

from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file. Returns the file content as text.

    Args:
        file_path: Absolute or relative path to the file to read.
    """
    try:
        path = os.path.expanduser(file_path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        if len(content) > 50000:
            return content[:50000] + "\n... (truncated, file too large)"
        return content
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file, creating it if it doesn't exist. Overwrites existing files.

    Args:
        file_path: Absolute or relative path to the file to write.
        content: The content to write to the file.
    """
    try:
        path = os.path.expanduser(file_path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} bytes to {file_path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def edit_file(file_path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing exact text matches. The old_text must match exactly.

    Args:
        file_path: Path to the file to edit.
        old_text: The exact text to find and replace.
        new_text: The replacement text.
    """
    try:
        path = os.path.expanduser(file_path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_text)
        if count == 0:
            return f"Error: Text not found in {file_path}. Make sure old_text matches exactly."
        if count > 1:
            return f"Error: Text found {count} times in {file_path}. old_text must be unique. Add more context."

        new_content = content.replace(old_text, new_text, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return f"Successfully edited {file_path} (1 replacement)"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error editing file: {e}"


@tool
def grep_files(pattern: str, path: str = ".", include: str = "") -> str:
    """Search for a pattern in files using ripgrep (rg) or grep.

    Args:
        pattern: The regex pattern to search for.
        path: Directory or file to search in (default: current directory).
        include: File glob pattern to filter (e.g. '*.py', '*.java'). Optional.
    """
    try:
        search_path = os.path.expanduser(path)
        cmd = ["rg", "--no-heading", "--line-number", "--color", "never"]
        if include:
            cmd.extend(["-g", include])
        cmd.extend([pattern, search_path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if not output:
            return f"No matches found for pattern '{pattern}' in {path}"
        if len(output) > 30000:
            return output[:30000] + "\n... (truncated)"
        return output
    except FileNotFoundError:
        # Fallback to grep
        try:
            cmd = ["grep", "-rn", "--color=never"]
            if include:
                cmd.extend([f"--include={include}"])
            cmd.extend([pattern, search_path])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout
            if not output:
                return f"No matches found for pattern '{pattern}' in {path}"
            if len(output) > 30000:
                return output[:30000] + "\n... (truncated)"
            return output
        except Exception as e:
            return f"Error searching: {e}"
    except subprocess.TimeoutExpired:
        return "Error: Search timed out. Try narrowing the search path."
    except Exception as e:
        return f"Error searching: {e}"


@tool
def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern.

    Args:
        pattern: Glob pattern to match (e.g. '**/*.py', '*.md').
        path: Directory to search in (default: current directory).
    """
    try:
        import glob as glob_module

        search_path = os.path.expanduser(path)
        full_pattern = os.path.join(search_path, pattern)
        matches = sorted(glob_module.glob(full_pattern, recursive=True))
        if not matches:
            return f"No files matching pattern '{pattern}' in {path}"
        # Limit to 200 results
        if len(matches) > 200:
            return "\n".join(matches[:200]) + f"\n... and {len(matches) - 200} more"
        return "\n".join(matches)
    except Exception as e:
        return f"Error: {e}"


@tool
def shell_execute(command: str) -> str:
    """Execute a shell command and return its output.

    Use for build, test, deployment, and other terminal commands.
    Be cautious with destructive commands (rm, etc.).

    Args:
        command: The shell command to execute.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += "\n" + result.stderr if output else result.stderr
        if not output.strip():
            output = f"(Command completed with exit code {result.returncode}, no output)"
        if len(output) > 30000:
            output = output[:30000] + "\n... (truncated)"
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 120 seconds."
    except Exception as e:
        return f"Error executing command: {e}"
