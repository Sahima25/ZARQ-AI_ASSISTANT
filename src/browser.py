"""Browser control for ZARQ via the Playwright MCP server.

Spawns Microsoft's @playwright/mcp as a local (stdio) MCP server and exposes
its browser automation tools to the agent through an MCPToolset. The server
drives a visible Chrome window with a persistent profile, so site logins made
once by Sahima stay available to the agent.
"""

import logging
import os

from livekit.agents import mcp
from livekit.agents.llm.tool_context import (
    ToolFlag,
    get_function_info,
    get_raw_function_info,
    is_function_tool,
    is_raw_function_tool,
)
from mcp import types as mcp_types

logger = logging.getLogger("browser")

# Tools that routinely take longer than a few seconds. Listed tools on an
# MCPToolset run as async (background) tools, so the conversation keeps
# flowing while a page loads. CANCELLABLE lets "stop" interrupt a hung
# operation; on_duplicate="replace" means a corrected repeat call (e.g.
# "no, go to example.com instead") replaces the running one.
_SLOW_TOOLS = [
    "browser_navigate",
    "browser_wait_for",
    "browser_take_screenshot",
    "browser_evaluate",
    "browser_fill_form",
    "browser_file_upload",
]

# browser_run_code_unsafe executes arbitrary code in the MCP server process
# (RCE-equivalent per the Playwright MCP docs). Every other tool the server
# exposes stays enabled.
_DISALLOWED_TOOLS = frozenset({"browser_run_code_unsafe"})

# Accessibility snapshots of large pages can be very long. Cap what reaches
# the model, keeping the head (page structure) and tail (latest state).
_MAX_TOOL_RESULT_CHARS = 9000


def _resolve_tool_result(ctx: mcp.MCPToolResultContext) -> str:
    """Serialize MCP results for the LLM, dropping images and capping length."""
    parts: list[str] = []
    for item in ctx.result.content:
        if isinstance(item, mcp_types.TextContent):
            parts.append(item.text)
        elif isinstance(item, mcp_types.ImageContent):
            parts.append("[screenshot taken; image not forwarded]")
        else:
            parts.append(str(item.model_dump_json(by_alias=True)))

    text = "\n".join(parts)
    if len(text) > _MAX_TOOL_RESULT_CHARS:
        head_len = _MAX_TOOL_RESULT_CHARS * 2 // 3
        tail_len = _MAX_TOOL_RESULT_CHARS // 3
        text = (
            f"{text[:head_len]}\n"
            "... [snapshot truncated to fit voice context] ...\n"
            f"{text[-tail_len:]}"
        )
    return text


def _tool_name(tool: object) -> str:
    if is_function_tool(tool):
        return get_function_info(tool).name
    if is_raw_function_tool(tool):
        return get_raw_function_info(tool).name
    return ""


class PlaywrightToolset(mcp.MCPToolset):
    """MCPToolset that drops disallowed tools right after they are fetched."""

    def __init__(self, *, disallowed: frozenset[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._disallowed = disallowed

    async def setup(self, *, reload: bool = False) -> "PlaywrightToolset":
        await super().setup(reload=reload)
        dropped = [t for t in self._tools if _tool_name(t) in self._disallowed]
        self.filter_tools(lambda tool: _tool_name(tool) not in self._disallowed)
        if dropped:
            logger.info(
                "excluded %d Playwright MCP tool(s): %s",
                len(dropped),
                ", ".join(_tool_name(t) for t in dropped),
            )
        return self


def create_browser_toolset() -> PlaywrightToolset:
    """Build the Playwright MCP toolset for the agent.

    Environment variables:
        ZARQ_BROWSER_HEADLESS: run without a visible window (used in CI sims).
        ZARQ_BROWSER_ISOLATED: use throwaway browser contexts instead of the
            persistent profile (no logins are kept between sessions).
    """
    args = [
        "-y",
        "@playwright/mcp@latest",
        "--browser",
        "chrome",  # use the locally installed Google Chrome
        "--output-dir",
        "browser-data",
    ]

    if os.environ.get("ZARQ_BROWSER_HEADLESS"):
        args.append("--headless")

    if os.environ.get("ZARQ_BROWSER_ISOLATED"):
        args.append("--isolated")
    else:
        args += ["--user-data-dir", os.path.join("browser-data", "profile")]

    return PlaywrightToolset(
        disallowed=_DISALLOWED_TOOLS,
        id="playwright",
        mcp_server=mcp.MCPServerStdio(
            # On Windows npx is a .cmd/.ps1 shim, so it must be run through cmd.
            "cmd",
            ["/c", "npx", *args],
            client_session_timeout_seconds=120,  # default 5s is far too low for pages
            tool_result_resolver=_resolve_tool_result,
        ),
        tool_options={
            name: mcp.MCPToolOptions(
                flags=ToolFlag.CANCELLABLE,
                on_duplicate="replace",
            )
            for name in _SLOW_TOOLS
        },
    )
