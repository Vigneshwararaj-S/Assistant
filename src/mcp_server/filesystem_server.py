import sys
from pathlib import Path

from mcp.server import MCPServer

MAX_READ_CHARS = 8000

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()

mcp = MCPServer(
    "FilesystemTools", "0.1.0",
    f"Read-only access to files under {ROOT}",
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


if __name__ == "__main__":
    mcp.run()
