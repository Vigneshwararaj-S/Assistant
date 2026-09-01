import asyncio
import threading

from mcp import Client, StdioServerParameters


class ToolClient:
    def __init__(self, command: str, args: list[str]):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        self._client = None
        self._shutdown_event = None
        ready = threading.Event()

        server = StdioServerParameters(command=command, args=args)
        self._session_future = asyncio.run_coroutine_threadsafe(
            self._session_task(server, ready), self._loop
        )
        ready.wait()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _session_task(self, server, ready):
        self._shutdown_event = asyncio.Event()
        async with Client(server) as client:
            self._client = client
            ready.set()
            await self._shutdown_event.wait()

    def _run_coro(self, coro):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def list_tools(self):
        result = self._run_coro(self._client.list_tools())
        return result.tools

    def call_tool(self, name: str, arguments: dict):
        result = self._run_coro(self._client.call_tool(name, arguments))
        return result.structured_content

    def close(self):
        self._loop.call_soon_threadsafe(self._shutdown_event.set)
        self._session_future.result()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
