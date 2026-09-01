from mcp.server import MCPServer
from datetime import datetime

mcp = MCPServer("AssistantTools", "0.1.0", "A set of tools for the Assistant")

@mcp.tool()
def get_current_datetime() -> str:
    """Get the current date and time in the ISO 8601 format"""
    return datetime.now().isoformat(timespec='seconds')

if __name__ == "__main__":
    mcp.run()
