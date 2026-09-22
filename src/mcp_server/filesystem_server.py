import sys
from pathlib import Path

from mcp.server import MCPServer

MAX_READ_CHARS = 8000

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()

mcp = MCPServer(
    "FilesystemTools", "0.1.0",
    f"Read and write access to files under {ROOT}",
)


def _resolve_safe(relative_path: str) -> Path:
    target = (ROOT / relative_path).resolve()
    if target != ROOT and ROOT not in target.parents:
        raise ValueError(f"'{relative_path}' is outside the allowed root")
    return target


@mcp.tool()
def list_directory(path: str = ".") -> list[str]:
    """List files and subdirectories at a path relative to the allowed root (e.g. 'src' or 'src/mcp_server', not an absolute path). Use '.' for the root itself. Directories are suffixed with '/'."""
    target = _resolve_safe(path)
    if not target.is_dir():
        raise ValueError(f"'{path}' is not a directory")
    return sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())


@mcp.tool()
def read_file(path: str) -> str:
    """Read the text contents of a file at a path relative to the allowed root (e.g. 'src/main.py', not just 'main.py' if it's in a subdirectory — call list_directory first if unsure of the exact path). Truncated if very large."""
    target = _resolve_safe(path)
    if not target.is_file():
        raise ValueError(f"'{path}' is not a file")
    text = target.read_text(encoding="utf-8", errors="replace")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n...[truncated, {len(text) - MAX_READ_CHARS} more characters]"
    return text


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Create or overwrite a text file at a path relative to the allowed root (e.g. 'notes.txt' or 'src/notes.txt'). The parent directory must already exist -- this will not create new directories. Overwrites without warning if the file already exists."""
    target = _resolve_safe(path)
    if not target.parent.is_dir():
        raise ValueError(f"Parent directory of '{path}' does not exist")
    if target.is_dir():
        raise ValueError(f"'{path}' is a directory, not a file")
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


@mcp.tool()
def delete_file(path: str) -> str:
    """Delete a file at a path relative to the allowed root. Cannot delete directories."""
    target = _resolve_safe(path)
    if not target.is_file():
        raise ValueError(f"'{path}' is not a file")
    target.unlink()
    return f"Deleted {path}"


if __name__ == "__main__":
    mcp.run()
