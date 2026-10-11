import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_mcp_adapters.client import MultiServerMCPClient

from ronnyx.api.routers import router as chat_router
from ronnyx.config import load_config
from ronnyx.core.agent import build_graph
from ronnyx.knowledge import load_knowledge_tools

logger = logging.getLogger("ronnyx")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    mcp = MultiServerMCPClient(config.mcp_servers)
    tools = await mcp.get_tools()
    logger.info(
        "Loaded %d tools from %d MCP servers", len(tools), len(config.mcp_servers)
    )

    custom = config.load_custom_tools()
    if custom:
        tools = tools + custom
        logger.info("Loaded %d custom tools", len(custom))

    if config.knowledge:
        knowledge_tools = await load_knowledge_tools(
            config.knowledge, config.llm_api_key
        )
        if knowledge_tools:
            tools = tools + knowledge_tools
            logger.info("Loaded knowledge base from '%s'", config.knowledge["root"])

    app.state.graph = build_graph(tools, config)
    app.state.tool_names = [t.name for t in tools]
    app.state.sessions = {}
    yield


app = FastAPI(title="Ronnyx", lifespan=lifespan)

app.include_router(chat_router, prefix="/api")


@app.get("/")
def root():
    return {"name": "ronnyx", "status": "ok"}
