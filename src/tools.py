import asyncio
import logging

from langchain_community.tools import DuckDuckGoSearchRun
from livekit.agents import RunContext, function_tool


@function_tool
async def search_web(context: RunContext, query: str):
    try:
        results = await asyncio.to_thread(DuckDuckGoSearchRun().run, query)
        logging.info(f"Web search result for query '{query}': {results}")
        return results

    except Exception as e:
        logging.error(
            f"Error occurred while searching the web for query '{query}': {e}"
        )
        raise
