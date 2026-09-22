import ipaddress
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from mcp.server import MCPServer

MAX_FETCH_CHARS = 8000
TIMEOUT_SECONDS = 10

mcp = MCPServer("WebTools", "0.1.0", "Read-only web page fetching")


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Only allow http/https URLs that resolve to a public address -- blocks
    localhost/private-network targets, as a cheap defense against the model
    being tricked (e.g. by instructions embedded in a fetched page) into
    probing this machine's own local network."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False, f"unsupported scheme '{parsed.scheme}' -- only http/https allowed"
    hostname = parsed.hostname
    if not hostname:
        return False, "URL has no hostname"
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        return False, f"could not resolve host: {e}"
    for family, _, _, _, sockaddr in addr_info:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, f"resolves to a private/local address ({ip}) -- blocked"
    return True, ""


@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch a public web page and return its readable text content (HTML stripped). Only http/https URLs to public addresses are allowed -- local/private network addresses are blocked. Truncated if very large."""
    safe, reason = _is_safe_url(url)
    if not safe:
        raise ValueError(f"Refusing to fetch '{url}': {reason}")

    try:
        response = requests.get(
            url, timeout=TIMEOUT_SECONDS, headers={"User-Agent": "Mozilla/5.0 (personal-assistant-bot)"}
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch '{url}': {e}")

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(separator=" ").split())

    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS] + f"\n...[truncated, {len(text) - MAX_FETCH_CHARS} more characters]"
    return text


if __name__ == "__main__":
    mcp.run()
