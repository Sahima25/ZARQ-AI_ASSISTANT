"""Notion integration for ZARQ via Notion's official MCP server.

Spawns @notionhq/notion-mcp-server as a local (stdio) process and exposes a
focused subset of its tools to the agent. Enabled only when NOTION_TOKEN is
set to an internal integration secret (create one at
https://www.notion.so/profile/integrations and share pages with it); without
the token the agent runs with no Notion tools at all.
"""

import logging
import os

from livekit.agents import mcp
from livekit.agents.llm.tool_context import (
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from mcp import types as mcp_types

logger = logging.getLogger("notion")

# The server exposes 24 API-derived tools. Voice conversations only need this
# core set: find things, read them, write them, and query databases. Block-level
# edits, deletions, comments, users, and data-source schema management stay out
# so the model's tool list (and every turn's context) stays small.
_ALLOWED_TOOLS = frozenset(
    {
        "API-post-search",  # find pages and databases by title
        "API-retrieve-page-markdown",  # read a page's content as Markdown
        "API-update-page-markdown",  # edit a page's content
        "API-post-page",  # create a page or note
        "API-retrieve-a-page",  # page properties (title, URL, parent)
        "API-patch-page",  # update page properties (rename, status)
        "API-query-data-source",  # read entries from a database
        "API-retrieve-a-database",  # database -> data source IDs for queries
    }
)

# Notion can return very long pages; cap what reaches the voice model, keeping
# the head (page start) and tail (most recently edited content).
_MAX_TOOL_RESULT_CHARS = 12000


def _resolve_tool_result(ctx: mcp.MCPToolResultContext) -> str:
    """Serialize MCP results for the LLM and cap length."""
    parts: list[str] = []
    for item in ctx.result.content:
        if isinstance(item, mcp_types.TextContent):
            parts.append(item.text)
        else:
            parts.append(str(item.model_dump_json(by_alias=True)))

    text = "\n".join(parts)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        head_len = _MAX_TOOL_RESULT_CHARS * 2 // 3
        tail_len = _MAX_TOOL_RESULT_CHARS // 3
        text = (
            f"{text[:head_len]}\n"
            "... [page truncated to fit voice context] ...\n"
            f"{text[-tail_len:]}"
        )
    return text


def _tool_name(tool: object) -> str:
    if is_function_tool(tool):
        return get_function_info(tool).name
    if is_raw_function_tool(tool):
        return get_raw_function_info(tool).name
    return ""


class NotionToolset(mcp.MCPToolset):
    """MCPToolset that keeps only the allowed Notion tools after fetching them."""

    def __init__(self, *, allowed: frozenset[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._allowed = allowed

    async def setup(self, *, reload: bool = False) -> "NotionToolset":
        await super().setup(reload=reload)
        fetched = len(self._tools)
        self.filter_tools(lambda tool: _tool_name(tool) in self._allowed)
        logger.info(
            "Notion MCP ready: keeping %d of %d tools",
            len(self._allowed),
            fetched,
        )
        return self


def create_notion_toolset() -> NotionToolset | None:
    """Build the Notion MCP toolset, or None when NOTION_TOKEN is unset."""
    token = os.environ.get("NOTION_TOKEN", "").strip()
    if not token:
        logger.info("NOTION_TOKEN not set; Notion tools are disabled")
        return None

    return NotionToolset(
        allowed=_ALLOWED_TOOLS,
        id="notion",
        mcp_server=mcp.MCPServerStdio(
            # On Windows npx is a .cmd/.ps1 shim, so it must be run through cmd.
            "cmd",
            ["/c", "npx", "-y", "@notionhq/notion-mcp-server"],
            env={**os.environ, "NOTION_TOKEN": token},
            client_session_timeout_seconds=60,  # cold npx start can exceed the 5s default
            tool_result_resolver=_resolve_tool_result,
        ),
    )
