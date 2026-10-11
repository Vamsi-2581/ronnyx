"""Optional GraphRAG knowledge base, exposed to the agent as search tools.

GraphRAG is an optional dependency (`pip install ronnyx[graphrag]`), so every
graphrag import stays inside a function.
"""

import logging
import os
from pathlib import Path

from langchain_core.tools import StructuredTool

logger = logging.getLogger("ronnyx")

COMMUNITY_LEVEL = 2
RESPONSE_TYPE = "Multiple Paragraphs"

INSTALL_HINT = "GraphRAG is not installed. Run: pip install 'ronnyx[graphrag]'"


def _absolutize_paths(model, root: Path) -> None:
    from pydantic import BaseModel

    for name, value in model:
        if isinstance(value, BaseModel):
            _absolutize_paths(value, root)
        elif (
            isinstance(value, str)
            and value
            and (name == "base_dir" or name.endswith("prompt"))
            and not Path(value).is_absolute()
        ):
            setattr(model, name, str(root / value))


def _load_graphrag_config(root: Path, api_key: str):
    from graphrag.config.load_config import load_config

    # settings.yaml references ${GRAPHRAG_API_KEY}; reuse the Ronnyx key unless set.
    os.environ.setdefault("GRAPHRAG_API_KEY", api_key)

    # load_config chdirs into the project root and leaves some paths (prompts,
    # cache) relative to it. Restore the cwd for the rest of Ronnyx and anchor
    # those paths to the root instead.
    root = root.resolve()
    cwd = Path.cwd()
    try:
        config = load_config(root_dir=root)
    finally:
        os.chdir(cwd)
    _absolutize_paths(config, root)
    return config


def init_project(root: Path, model: str) -> None:
    from graphrag.cli.initialize import initialize_project_at
    from graphrag.config.defaults import DEFAULT_EMBEDDING_MODEL

    initialize_project_at(
        path=root, force=False, model=model, embedding_model=DEFAULT_EMBEDDING_MODEL
    )


async def build_index(root: Path, api_key: str) -> bool:
    from graphrag import api
    from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks

    config = _load_graphrag_config(root, api_key)
    outputs = await api.build_index(
        config=config, callbacks=[ConsoleWorkflowCallbacks()]
    )
    return all(output.error is None for output in outputs)


async def load_tools(root: Path, api_key: str) -> list[StructuredTool]:
    from graphrag import api
    from graphrag.data_model.data_reader import DataReader
    from graphrag_storage import create_storage
    from graphrag_storage.tables.table_provider_factory import create_table_provider

    config = _load_graphrag_config(root, api_key)
    table_provider = create_table_provider(
        config.table_provider, storage=create_storage(config.output_storage)
    )
    reader = DataReader(table_provider)

    # Load the index once at startup; every search reuses these tables.
    entities = await reader.entities()
    communities = await reader.communities()
    community_reports = await reader.community_reports()
    text_units = await reader.text_units()
    relationships = await reader.relationships()
    covariates = (
        await reader.covariates() if await table_provider.has("covariates") else None
    )

    async def knowledge_local_search(query: str) -> str:
        response, _ = await api.local_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            text_units=text_units,
            relationships=relationships,
            covariates=covariates,
            community_level=COMMUNITY_LEVEL,
            response_type=RESPONSE_TYPE,
            query=query,
        )
        return str(response)

    async def knowledge_global_search(query: str) -> str:
        response, _ = await api.global_search(
            config=config,
            entities=entities,
            communities=communities,
            community_reports=community_reports,
            community_level=COMMUNITY_LEVEL,
            dynamic_community_selection=False,
            response_type=RESPONSE_TYPE,
            query=query,
        )
        return str(response)

    return [
        StructuredTool.from_function(
            coroutine=knowledge_local_search,
            name="knowledge_local_search",
            description=(
                "Search the user's indexed documents for specific facts about named "
                "people, projects, concepts or events and how they relate."
            ),
        ),
        StructuredTool.from_function(
            coroutine=knowledge_global_search,
            name="knowledge_global_search",
            description=(
                "Answer broad questions across all of the user's indexed documents, "
                "such as main themes, overall summaries or recurring topics."
            ),
        ),
    ]


async def load_knowledge_tools(knowledge: dict, api_key: str) -> list[StructuredTool]:
    """Load the knowledge tools, or return none if the index is unavailable.

    Like custom tools, a missing or broken knowledge base must not block startup.
    """
    root = Path(knowledge["root"])
    try:
        return await load_tools(root, api_key)
    except ImportError:
        logger.warning(INSTALL_HINT)
    except Exception as e:  # noqa: BLE001 - a missing index must not block startup
        logger.warning(
            "Knowledge base at '%s' is unavailable (%s). Run 'ronnyx-index' to build it.",
            root,
            e,
        )
    return []
