"""
base_query.py

Agent tool base class:

Since I'm implementing 2 MCP Clients (freq_query and yfin_query), this will be a common
base class with all the core MCP logic.  The 2 clients with then extend this class for
their specific tool queries to their respective MCP Servers.


The client will be implemented as a class, so I can maintain session state over
multiple agent calls.

This class will handle all the "plumbing".  The extensions (fred_query and yfin_query) will
then provide a wrapper which forwards the agent's request to that server and returns
the structured result.
"""
from __future__ import annotations

# src/sector_rotation_agent/fred_query.py
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import sys
import time
import logging
from typing import Optional
from contextlib import AsyncExitStack
from pathlib import Path

class MCPClientBase:
    def __init__(
            self,
            server_path: str | Path
            ) -> None:

        self._server_path = server_path
        self._server_params = StdioServerParameters(command=sys.executable, args=[str(server_path)])
        self._session: Optional[ClientSession] = None
        self._stack: AsyncExitStack = AsyncExitStack()
        self._logger = logging.getLogger(__name__)
        self._logger.info("MCP client created for server: %s", self._server_path)

    async def connect(self):
        self._logger.info("Connecting to MCP server: %s", self._server_path)
        t0 = time.perf_counter()
        try:
            stdio_transport = await self._stack.enter_async_context(
                stdio_client(self._server_params)
            )

            _stdio, _write = stdio_transport
            self._session = await self._stack.enter_async_context(
                ClientSession(_stdio, _write)
            )

            await self._session.initialize()
        except Exception:
            self._logger.exception("Failed to connect to MCP server: %s", self._server_path)
            raise
        self._logger.info("MCP session initialized: %s (%.0f ms)",
                          self._server_path, (time.perf_counter() - t0) * 1000.0)

    def session(self):
        if self._session is None:
            raise ConnectionError("Client session is not initialized or cached.")
        return self._session
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    # -----  Tools   ------------------------------------------------------------
    # Here's where we publish and call the Tools that agents will be able to call
    async def list_tools(self) -> list[types.Tool]:
        result = await self.session().list_tools()
        tools = result.tools
        self._logger.debug("MCP server %s exposes %d tool(s): %s",
                           self._server_path, len(tools), [t.name for t in tools])
        return tools
    
    async def call_tool(
            self,
            tool_name: str,
            tool_input: dict,
        ) -> types.CallToolResult:

        self._logger.info("MCP tool call -> %s", tool_name)
        self._logger.debug("MCP tool input for %s: %s", tool_name, tool_input)
        t0 = time.perf_counter()
        try:
            result = await self.session().call_tool(tool_name, tool_input)
        except Exception:
            self._logger.exception(
                "MCP tool call FAILED: %s after %.0f ms",
                tool_name, (time.perf_counter() - t0) * 1000.0,
            )
            raise
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if getattr(result, "isError", False):
            self._logger.warning("MCP tool %s returned an error result (%.0f ms)",
                                 tool_name, latency_ms)
        else:
            self._logger.info("MCP tool call <- %s (%.0f ms)", tool_name, latency_ms)
        return result
    
    # -----  Prompts   ------------------------------------------------------------
    # This Client talks to an MCP Server that only exposes Tools

    # -----  Resources   ------------------------------------------------------------
    # This Client talks to an MCP Server that only exposes Tools

    # Cleanup functions
    async def cleanup(self):
        self._logger.debug("Closing MCP session: %s", self._server_path)
        await self._stack.aclose()
        self._session = None

    async def __aexit__(self, exc_type, exc, tb):
        await self.cleanup()

