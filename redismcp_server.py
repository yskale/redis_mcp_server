#!/usr/bin/env python3
"""
Redis Graph MCP Server
Provides Model Context Protocol interface to query biomedical knowledge graph stored in Redis
"""

import os
import asyncio
import argparse
import redis
from redis.commands.graph import Graph

import json
import re
import httpx
from reasoner_transpiler.cypher import get_query as trapi_get_query
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    Tool, TextContent,
    Prompt, PromptArgument, PromptMessage, GetPromptResult,
)
from dotenv import load_dotenv
import uvicorn

NAME_RESOLUTION_URL = "https://name-resolution-sri.renci.org/lookup"
SAP_QDRANT_URL = "https://sap-qdrant.example.com/annotate/"

# Load environment variables from .env file
load_dotenv()
# Initialize MCP server
app = Server("redis-graph-server")

# Redis connection pool (initialized on first use)
redis_client = None
graph_db = None


async def fetch_synonyms(search_term: str) -> dict:
    """
    Enrich a search term by fetching synonyms and related concept identifiers from:
    - Name Resolution SRI: returns normalized CURIEs and preferred labels
    - SAP-BERT Qdrant: returns semantically similar biomedical concepts

    Returns {'curies': [...], 'labels': [...]} with up to 10 results from each source.
    """
    curies = []
    labels = []

    async with httpx.AsyncClient(timeout=10.0) as client:
        nr_coro = client.get(NAME_RESOLUTION_URL, params={"string": search_term, "limit": 10})
        sap_coro = client.post(
            SAP_QDRANT_URL,
            json={"text": search_term, "model_name": "sapbert", "count": 10, "args": {}}
        )
        nr_resp, sap_resp = await asyncio.gather(nr_coro, sap_coro, return_exceptions=True)

    # Name Resolution SRI: [{curie, label, synonyms, ...}, ...]
    if not isinstance(nr_resp, Exception) and nr_resp.status_code == 200:
        try:
            for item in nr_resp.json()[:10]:
                if curie := item.get("curie"):
                    curies.append(curie)
                if label := item.get("label"):
                    labels.append(label)
        except Exception:
            pass

    # SAP-BERT Qdrant: response structure varies; handle list or wrapped dict
    if not isinstance(sap_resp, Exception) and sap_resp.status_code == 200:
        try:
            sap_data = sap_resp.json()
            items = sap_data if isinstance(sap_data, list) else (
                sap_data.get("results") or sap_data.get("hits") or
                sap_data.get("annotations") or sap_data.get("concepts") or []
            )
            for item in items[:10]:
                if not isinstance(item, dict):
                    continue
                for id_field in ("id", "curie", "identifier", "concept_id"):
                    if val := item.get(id_field):
                        curies.append(val)
                        break
                for name_field in ("name", "label", "text", "concept_name"):
                    if val := item.get(name_field):
                        labels.append(val)
                        break
        except Exception:
            pass

    return {
        "curies": list(dict.fromkeys(curies)),   # deduplicate, preserve order
        "labels": list(dict.fromkeys(labels)),
    }


@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    """Expose reusable workflow prompts for chatbots and LLM clients."""
    return [
        Prompt(
            name="find_variables_for_concept",
            description=(
                "Find study variables in the BDC knowledge graph related to a biomedical concept. "
                "Enriches the concept with synonyms from Name Resolution SRI and SAP-BERT, "
                "then searches for connected study variables."
            ),
            arguments=[
                PromptArgument(
                    name="concept",
                    description="Biomedical concept to search for (e.g. 'asthma', 'diabetes', 'cholesterol')",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="explore_concept",
            description=(
                "Get a full picture of a concept in the knowledge graph: what entities are connected to it, "
                "which studies have measured it, and what variables are associated with it."
            ),
            arguments=[
                PromptArgument(
                    name="concept_id",
                    description="Concept CURIE identifier (e.g. 'MONDO:0004979' for asthma)",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="find_studies_for_disease",
            description=(
                "End-to-end workflow: given a disease or phenotype name, find all studies in the BDC "
                "knowledge graph that have collected data related to it, via synonym-enriched concept lookup."
            ),
            arguments=[
                PromptArgument(
                    name="disease_name",
                    description="Disease or phenotype name (e.g. 'asthma', 'hypertension')",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="explain_variable",
            description=(
                "Explain what a study variable measures and its full biomedical context: "
                "which concepts it is linked to and which studies it belongs to."
            ),
            arguments=[
                PromptArgument(
                    name="variable_id",
                    description="Study variable identifier (e.g. 'phv00430345.v1.p1')",
                    required=True,
                )
            ],
        ),
        Prompt(
            name="find_path_between_concepts",
            description=(
                "Discover how two biomedical concepts are related to each other through the knowledge graph "
                "and explain the connection path in plain language."
            ),
            arguments=[
                PromptArgument(
                    name="concept_a_id",
                    description="First concept CURIE (e.g. 'MONDO:0004979')",
                    required=True,
                ),
                PromptArgument(
                    name="concept_b_id",
                    description="Second concept CURIE (e.g. 'MONDO:0004784')",
                    required=True,
                ),
            ],
        ),
        Prompt(
            name="trapi_query_builder",
            description=(
                "Build and execute a TRAPI query graph against the knowledge graph. "
                "Guides the LLM to construct a valid TRAPI query for a given biomedical question, "
                "with automatic biolink predicate hierarchy expansion."
            ),
            arguments=[
                PromptArgument(
                    name="question",
                    description="Biomedical question to answer (e.g. 'What study variables are associated with asthma?')",
                    required=True,
                ),
            ],
        ),
    ]


@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> GetPromptResult:
    """Return the message template for the requested prompt."""

    if name == "find_variables_for_concept":
        concept = arguments.get("concept", "")
        return GetPromptResult(
            description=f"Find study variables related to '{concept}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"I want to find study variables in the BDC biomedical knowledge graph "
                            f"related to the concept '{concept}'.\n\n"
                            f"Please use the `search_concepts` tool with:\n"
                            f"  - search_term: \"{concept}\"\n"
                            f"  - find_variables: true\n\n"
                            f"This will automatically enrich '{concept}' with synonyms from "
                            f"Name Resolution SRI and SAP-BERT before searching.\n\n"
                            f"Once you have the results, summarize:\n"
                            f"1. How many synonyms were found and what they are\n"
                            f"2. How many study variables were matched\n"
                            f"3. The most relevant variables and which concepts linked them"
                        ),
                    ),
                )
            ],
        )

    elif name == "explore_concept":
        concept_id = arguments.get("concept_id", "")
        return GetPromptResult(
            description=f"Explore all connections for concept '{concept_id}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Explore the biomedical concept '{concept_id}' in the knowledge graph.\n\n"
                            f"Please run these tools in order:\n"
                            f"1. `get_concept_connections` with concept_id='{concept_id}' — to see all directly connected entities\n"
                            f"2. `find_connected_studies` with concept_id='{concept_id}' — to find studies that have measured this concept\n"
                            f"3. `get_concept_graph` with concept_id='{concept_id}' — for the full subgraph\n\n"
                            f"Then provide a clear summary:\n"
                            f"- What is this concept and what types of entities are connected to it?\n"
                            f"- How many study variables and studies are linked?\n"
                            f"- What are the most notable connections?"
                        ),
                    ),
                )
            ],
        )

    elif name == "find_studies_for_disease":
        disease_name = arguments.get("disease_name", "")
        return GetPromptResult(
            description=f"Find studies that collected data on '{disease_name}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"I want to find all studies in the BDC knowledge graph that have collected "
                            f"data related to '{disease_name}'.\n\n"
                            f"Step 1: Use `search_concepts` with search_term='{disease_name}' and find_variables=true "
                            f"to get synonym-enriched study variables.\n\n"
                            f"Step 2: For each unique concept_id returned, use `find_connected_studies` "
                            f"to get the full list of studies.\n\n"
                            f"Finally, summarize:\n"
                            f"- Which synonyms were matched in the graph\n"
                            f"- The complete list of studies found\n"
                            f"- How many unique variables and studies are available for '{disease_name}'"
                        ),
                    ),
                )
            ],
        )

    elif name == "explain_variable":
        variable_id = arguments.get("variable_id", "")
        return GetPromptResult(
            description=f"Explain study variable '{variable_id}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Please explain the study variable '{variable_id}' in plain language.\n\n"
                            f"Use `get_variable_details` with variable_id='{variable_id}' to retrieve its details.\n\n"
                            f"Then explain:\n"
                            f"1. What does this variable measure or represent?\n"
                            f"2. Which biomedical concepts is it linked to (diseases, phenotypes, procedures)?\n"
                            f"3. Which study does it belong to?\n"
                            f"4. How might a researcher use this variable in a study?"
                        ),
                    ),
                )
            ],
        )

    elif name == "find_path_between_concepts":
        concept_a = arguments.get("concept_a_id", "")
        concept_b = arguments.get("concept_b_id", "")
        return GetPromptResult(
            description=f"Find relationship path between '{concept_a}' and '{concept_b}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Discover how the concepts '{concept_a}' and '{concept_b}' are related "
                            f"in the biomedical knowledge graph.\n\n"
                            f"Use `find_concept_paths` with source_id='{concept_a}' and target_id='{concept_b}'.\n\n"
                            f"If no direct path is found, also try `expand_concept` on each to find "
                            f"shared neighbors.\n\n"
                            f"Then explain in plain language:\n"
                            f"1. Are these concepts directly or indirectly related?\n"
                            f"2. What is the relationship path between them?\n"
                            f"3. What does this connection mean biomedically?"
                        ),
                    ),
                )
            ],
        )

    elif name == "trapi_query_builder":
        question = arguments.get("question", "")
        return GetPromptResult(
            description=f"Build a TRAPI query to answer: '{question}'",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"I want to answer this biomedical question using the knowledge graph:\n\n"
                            f"\"{question}\"\n\n"
                            f"Please construct a TRAPI query graph and run it using the `trapi_query` tool.\n\n"
                            f"A TRAPI query graph has this structure:\n"
                            f"{{\n"
                            f"  \"nodes\": {{\n"
                            f"    \"<node_key>\": {{\"ids\": [\"<CURIE>\"], \"categories\": [\"biolink:<Type>\"]}}\n"
                            f"  }},\n"
                            f"  \"edges\": {{\n"
                            f"    \"<edge_key>\": {{\"subject\": \"<node_key>\", \"object\": \"<node_key>\", \"predicates\": [\"biolink:<predicate>\"]}}\n"
                            f"  }}\n"
                            f"}}\n\n"
                            f"Guidelines:\n"
                            f"- Use biolink categories like: biolink:Disease, biolink:PhenotypicFeature, biolink:StudyVariable, biolink:Gene\n"
                            f"- Use biolink predicates like: biolink:related_to, biolink:has_phenotype, biolink:associated_with\n"
                            f"- Omit 'ids' or 'categories' if unknown — the graph will match any node\n"
                            f"- Omit 'predicates' on an edge to match any relationship\n"
                            f"- The predicate hierarchy is automatically expanded (e.g. biolink:related_to includes all child predicates)\n\n"
                            f"If you're unsure of the exact CURIE, first use `search_concepts` to find it, "
                            f"then build the TRAPI query with the ID."
                        ),
                    ),
                )
            ],
        )

    else:
        raise ValueError(f"Unknown prompt: {name}")


def get_redis_connection():
    """Establish Redis connection with authentication"""
    global redis_client, graph_db

    if redis_client is None:
        host = os.getenv("REDIS_HOST", "localhost")
        port = int(os.getenv("REDIS_PORT", "6379"))
        password = os.getenv("REDIS_PASSWORD")
        graph_name = os.getenv("REDIS_GRAPH_NAME", "test")

        try:
            redis_client = redis.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            # Test connection
            redis_client.ping()
            graph_db = Graph(redis_client, graph_name)
        except Exception:
            raise

    return redis_client, graph_db


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for querying the Redis graph"""
    return [
        Tool(
            name="trapi_query",
            description=(
                "Query the knowledge graph using a standard TRAPI (Translator Reasoner API) query graph. "
                "Automatically handles the biolink predicate hierarchy — querying a parent predicate like "
                "'biolink:related_to' also matches all its child predicates (e.g. associated_with, correlated_with, etc.). "
                "Use this for structured biomedical queries without needing to know Cypher syntax. "
                "Nodes can be filtered by biolink category and/or specific IDs (CURIEs). "
                "Edges can be filtered by biolink predicate."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "qgraph": {
                        "type": "object",
                        "description": (
                            "TRAPI query graph with 'nodes' (dict of node_key → {ids?, categories?}) "
                            "and 'edges' (dict of edge_key → {subject, object, predicates?}). "
                            "Example: {\"nodes\": {\"disease\": {\"ids\": [\"MONDO:0004979\"], \"categories\": [\"biolink:Disease\"]}, "
                            "\"variable\": {\"categories\": [\"biolink:StudyVariable\"]}}, "
                            "\"edges\": {\"e0\": {\"subject\": \"disease\", \"object\": \"variable\"}}}"
                        ),
                        "properties": {
                            "nodes": {"type": "object"},
                            "edges": {"type": "object"}
                        },
                        "required": ["nodes", "edges"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return",
                        "default": 50
                    }
                },
                "required": ["qgraph"]
            }
        ),
        Tool(
            name="cypher_query",
            description="Execute a raw Cypher query on the Redis biomedical knowledge graph. Use backticks for labels with dots: `biolink.Disease`",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Cypher query to execute. Remember to escape biolink labels with backticks: (n:`biolink.Disease`)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Override LIMIT clause (optional)",
                        "default": 50
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_concepts",
            description=(
                "Search for biomedical concepts by name, OR find StudyVariables related to a concept with synonym enrichment. "
                "When find_variables=true, the search term is first expanded into synonyms and related concept identifiers "
                "by querying two external services in parallel: "
                "(1) Name Resolution SRI (https://name-resolution-sri.renci.org) — returns up to 10 normalized CURIEs and preferred labels; "
                "(2) SAP-BERT Qdrant (https://sap-qdrant.example.com) — returns up to 10 semantically similar biomedical concepts. "
                "The combined set of CURIEs and labels is then used to search the Redis knowledge graph for any StudyVariables "
                "connected to those concepts, providing broader coverage than a simple name match."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Concept name or keyword to search for (e.g., 'cholesterol', 'heart disease', 'asthma')"
                    },
                    "node_type": {
                        "type": "string",
                        "description": "Optional: Filter by node type (Disease, PhenotypicFeature, Procedure, Cell, OrganismTaxon, etc.)",
                        "enum": ["Disease", "PhenotypicFeature", "Procedure", "Cell", "OrganismTaxon", "Study", "Publication", "BiologicalProcess"]
                    },
                    "find_variables": {
                        "type": "boolean",
                        "description": "Set to true to enrich the search term with synonyms via Name Resolution SRI and SAP-BERT, then find StudyVariables connected to those concepts in the Redis graph",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20
                    }
                },
                "required": ["search_term"]
            }
        ),
        Tool(
            name="get_concept_graph",
            description="Get the complete subgraph for a biomedical concept including studies, variables, and related entities",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_id": {
                        "type": "string",
                        "description": "The concept ID (e.g., MONDO:0005015 for disease, phv00283870.v1.p1 for variable)"
                    },
                    "expand_depth": {
                        "type": "integer",
                        "description": "How many hops to follow (1-3)",
                        "default": 2
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum nodes to return",
                        "default": 100
                    }
                },
                "required": ["concept_id"]
            }
        ),
        Tool(
            name="get_concept_connections",
            description="List all entities directly connected to a concept, showing the relationship type, connected node type, name, and ID. Use node_type_filter to focus on a specific entity type (e.g. only StudyVariables or only Studies).",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_id": {
                        "type": "string",
                        "description": "Concept ID to analyze (works for any node: disease, phenotype, variable, etc.)"
                    },
                    "node_type_filter": {
                        "type": "string",
                        "description": "Optional: return only neighbors of this biolink type",
                        "enum": ["Disease", "PhenotypicFeature", "Procedure", "Cell", "OrganismTaxon", "Study", "StudyVariable", "BiologicalProcess"]
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of connections to return",
                        "default": 50
                    }
                },
                "required": ["concept_id"]
            }
        ),
        Tool(
            name="list_graph_schema",
            description="List all node types and relationships in the graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "show_counts": {
                        "type": "boolean",
                        "description": "Include count of each node type",
                        "default": True
                    }
                }
            }
        ),
        Tool(
            name="find_highly_connected_variables",
            description="Find StudyVariables with the most connections to biomedical concepts",
            inputSchema={
                "type": "object",
                "properties": {
                    "min_connections": {
                        "type": "integer",
                        "description": "Minimum number of connections",
                        "default": 10
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20
                    }
                }
            }
        ),
        Tool(
            name="search_variables_by_name",
            description=(
                "Search for StudyVariables by free-text name or by ID prefix/pattern. "
                "Unlike search_concepts, this searches only StudyVariable nodes and supports "
                "partial ID matching (e.g. 'phv00430' returns all variables whose ID starts with that prefix). "
                "Use this when you already know you want variables and have a name fragment or a partial phv ID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Variable name fragment or ID prefix (e.g., 'asthma', 'phv00430', 'bmi')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 20
                    }
                },
                "required": ["search_term"]
            }
        ),
        Tool(
            name="expand_concept",
            description="Expand a biomedical concept by finding related concepts, synonyms, and transitive connections up to N hops away",
            inputSchema={
                "type": "object",
                "properties": {
                    "concept_id": {
                        "type": "string",
                        "description": "Concept ID to expand (e.g., MONDO:0004784 for allergic asthma)"
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Maximum number of hops/relationships to traverse (1-3)",
                        "default": 2
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: Filter by specific relationship types (e.g., ['related_to', 'part_of'])"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of expanded concepts to return",
                        "default": 50
                    }
                },
                "required": ["concept_id"]
            }
        ),
        Tool(
            name="find_concept_paths",
            description="Find paths between two biomedical concepts through the knowledge graph",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {
                        "type": "string",
                        "description": "Source concept ID"
                    },
                    "target_id": {
                        "type": "string",
                        "description": "Target concept ID"
                    },
                    "max_path_length": {
                        "type": "integer",
                        "description": "Maximum path length to search (1-5)",
                        "default": 3
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of paths to return",
                        "default": 10
                    }
                },
                "required": ["source_id", "target_id"]
            }
        )
    ]


def results_to_list(result_set, header) -> list[dict]:
    """Convert a RedisGraph result set to a list of dicts keyed by column name."""
    if not result_set or not header:
        return []
    col_names = [col[1] for col in header]
    rows = []
    for row in result_set:
        rows.append({col: val for col, val in zip(col_names, row)})
    return rows


def to_json(data) -> str:
    return json.dumps(data, indent=2, default=str)


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        _, graph = get_redis_connection()
        
        if name == "trapi_query":
            qgraph = arguments["qgraph"]
            limit  = arguments.get("limit", 50)

            # Generate Cypher from TRAPI query graph using reasoner-transpiler.
            # Use memgraph dialect (uses id() instead of elementId()) — closer to RedisGraph.
            # reasoner=False gives plain Cypher without TRAPI result wrapping.
            cypher = trapi_get_query(qgraph, reasoner=False, dialect="memgraph")

            # RedisGraph uses backtick-wrapped dot notation: `biolink.Disease`
            # Transpiler outputs colon notation: `biolink:Disease` — convert it.
            cypher = re.sub(r'`biolink:(\w+)`', r'`biolink.\1`', cypher)

            # Append limit
            if "LIMIT" not in cypher.upper():
                cypher += f" LIMIT {limit}"

            result = graph.query(cypher)
            rows = results_to_list(result.result_set, result.header) if result.result_set else []
            return [TextContent(type="text", text=to_json({
                "qgraph": qgraph,
                "cypher_used": cypher,
                "total_results": len(rows),
                "results": rows,
            }))]

        elif name == "cypher_query":
            query = arguments["query"]
            result = graph.query(query)

            if result.result_set and len(result.result_set) > 0:
                rows = results_to_list(result.result_set, result.header)
                out = to_json({
                    "rows_returned": len(rows),
                    "execution_time_ms": result.execution_time,
                    "results": rows,
                })
                if len(out) > 50000:
                    out = out[:50000] + "\n... (truncated)"
                return [TextContent(type="text", text=out)]
            else:
                return [TextContent(type="text", text=to_json({"rows_returned": 0, "results": []}))]

        elif name == "search_concepts":
            search_term = arguments["search_term"]
            node_type = arguments.get("node_type")
            find_variables = arguments.get("find_variables", False)
            limit = arguments.get("limit", 20)

            if find_variables:
                synonyms = await fetch_synonyms(search_term)
                curies = synonyms["curies"]
                labels = synonyms["labels"]

                curie_list = ", ".join(f'"{c}"' for c in curies) if curies else None
                label_list = ", ".join(f'"{l}"' for l in labels) if labels else None

                where_clauses = []
                if curie_list:
                    where_clauses.append(f"concept.id IN [{curie_list}]")
                if label_list:
                    where_clauses.append(f"concept.name IN [{label_list}]")
                if not where_clauses:
                    where_clauses.append(f"concept.name CONTAINS '{search_term}'")
                where_expr = " OR ".join(where_clauses)

                query = f"""
                MATCH (concept)-[*1..2]-(v:`biolink.StudyVariable`)
                WHERE ({where_expr})
                  AND NOT labels(concept)[0] = 'biolink.StudyVariable'
                RETURN DISTINCT
                    labels(concept)[0] AS concept_type,
                    concept.id AS concept_id,
                    concept.name AS concept_name,
                    v.name AS variable_name,
                    v.id AS variable_id
                LIMIT {limit}
                """
                result = graph.query(query)
                rows = results_to_list(result.result_set, result.header) if result.result_set else []
                out = to_json({
                    "search_term": search_term,
                    "enrichment": {"curies": curies, "labels": labels},
                    "total_results": len(rows),
                    "variables": rows,
                })
                if len(out) > 50000:
                    out = out[:50000] + "\n... (truncated)"
                return [TextContent(type="text", text=out)]

            else:
                if node_type:
                    query = f"""
                    MATCH (n:`biolink.{node_type}`)
                    WHERE n.name CONTAINS '{search_term}'
                    RETURN labels(n)[0] AS type, n.name AS name, n.id AS id
                    LIMIT {limit}
                    """
                else:
                    query = f"""
                    MATCH (n)
                    WHERE n.name CONTAINS '{search_term}'
                    RETURN labels(n)[0] AS type, n.name AS name, n.id AS id
                    LIMIT {limit}
                    """
                result = graph.query(query)
                rows = results_to_list(result.result_set, result.header) if result.result_set else []
                out = to_json({
                    "search_term": search_term,
                    "node_type": node_type,
                    "total_results": len(rows),
                    "concepts": rows,
                })
                if len(out) > 50000:
                    out = out[:50000] + "\n... (truncated)"
                return [TextContent(type="text", text=out)]

        elif name == "get_concept_graph":
            concept_id = arguments["concept_id"]
            expand_depth = min(arguments.get("expand_depth", 2), 3)
            limit = arguments.get("limit", 100)

            if expand_depth == 1:
                query = f"""
                MATCH (concept {{id: "{concept_id}"}})-[r1]-(connected)
                RETURN concept.name AS concept,
                       type(r1) AS rel_type,
                       labels(connected)[0] AS connected_type,
                       connected.name AS connected_name,
                       connected.id AS connected_id
                LIMIT {limit}
                """
            else:
                query = f"""
                MATCH (concept {{id: "{concept_id}"}})-[r1]-(variable:`biolink.StudyVariable`)
                OPTIONAL MATCH (variable)-[r2]-(study:`biolink.Study`)
                OPTIONAL MATCH (variable)-[r3]-(related)
                WHERE related <> concept
                RETURN DISTINCT
                    concept.name AS concept,
                    concept.id AS concept_id,
                    labels(concept)[0] AS concept_type,
                    variable.name AS variable_name,
                    variable.id AS variable_id,
                    study.name AS study_name,
                    study.id AS study_id,
                    COUNT(DISTINCT related) AS related_concepts_count
                LIMIT {limit}
                """

            result = graph.query(query)
            if result.result_set:
                rows = results_to_list(result.result_set, result.header)
                out = to_json({"concept_id": concept_id, "expand_depth": expand_depth, "total_results": len(rows), "graph": rows})
                if len(out) > 50000:
                    out = out[:50000] + "\n... (truncated)"
                return [TextContent(type="text", text=out)]
            else:
                return [TextContent(type="text", text=to_json({"concept_id": concept_id, "total_results": 0, "graph": []}))]

        elif name == "get_concept_connections":
            concept_id = arguments["concept_id"]
            node_type_filter = arguments.get("node_type_filter")
            limit = arguments.get("limit", 50)

            type_clause = f":`biolink.{node_type_filter}`" if node_type_filter else ""

            summary_result = graph.query(f"""
            MATCH (concept {{id: "{concept_id}"}})-[r]-(connected{type_clause})
            WITH labels(connected)[0] AS entity_type, COUNT(*) AS count
            RETURN entity_type, count
            ORDER BY count DESC
            """)

            detail_result = graph.query(f"""
            MATCH (concept {{id: "{concept_id}"}})-[r]-(connected{type_clause})
            RETURN
                type(r) AS relationship,
                labels(connected)[0] AS connected_type,
                connected.name AS connected_name,
                connected.id AS connected_id
            ORDER BY labels(connected)[0], type(r), connected.name
            LIMIT {limit}
            """)

            summary = [{"entity_type": row[0], "count": row[1]} for row in (summary_result.result_set or [])]
            connections = results_to_list(detail_result.result_set, detail_result.header) if detail_result.result_set else []
            out = to_json({
                "concept_id": concept_id,
                "node_type_filter": node_type_filter,
                "summary": summary,
                "total_connections_shown": len(connections),
                "connections": connections,
            })
            if len(out) > 50000:
                out = out[:50000] + "\n... (truncated)"
            return [TextContent(type="text", text=out)]

        elif name == "list_graph_schema":
            show_counts = arguments.get("show_counts", True)

            if show_counts:
                query = """
                MATCH (n)
                WITH labels(n)[0] AS node_type, COUNT(*) AS count
                RETURN node_type, count
                ORDER BY count DESC
                """
            else:
                query = """
                MATCH (n)
                WITH DISTINCT labels(n)[0] AS node_type
                RETURN node_type
                ORDER BY node_type
                """

            result = graph.query(query)
            rows = results_to_list(result.result_set, result.header) if result.result_set else []
            return [TextContent(type="text", text=to_json({"schema": rows}))]

        elif name == "find_highly_connected_variables":
            min_connections = arguments.get("min_connections", 10)
            limit = arguments.get("limit", 20)

            query = f"""
            MATCH (v:`biolink.StudyVariable`)--(c)
            WITH v, COUNT(DISTINCT c) AS connection_count
            WHERE connection_count >= {min_connections}
            RETURN v.name AS variable_name, v.id AS variable_id, connection_count
            ORDER BY connection_count DESC
            LIMIT {limit}
            """

            result = graph.query(query)
            rows = results_to_list(result.result_set, result.header) if result.result_set else []
            return [TextContent(type="text", text=to_json({
                "min_connections": min_connections,
                "total_results": len(rows),
                "variables": rows,
            }))]

        elif name == "search_variables_by_name":
            search_term = arguments["search_term"]
            limit = arguments.get("limit", 20)

            query = f"""
            MATCH (v:`biolink.StudyVariable`)
            WHERE v.name CONTAINS '{search_term}' OR v.id CONTAINS '{search_term}'
            RETURN v.id AS variable_id, v.name AS variable_name
            LIMIT {limit}
            """

            result = graph.query(query)
            rows = results_to_list(result.result_set, result.header) if result.result_set else []
            out = to_json({"search_term": search_term, "total_results": len(rows), "variables": rows})
            if len(out) > 50000:
                out = out[:50000] + "\n... (truncated)"
            return [TextContent(type="text", text=out)]

        elif name == "expand_concept":
            concept_id = arguments["concept_id"]
            max_hops = min(arguments.get("max_hops", 2), 3)
            relationship_types = arguments.get("relationship_types")
            limit = arguments.get("limit", 50)

            if relationship_types and len(relationship_types) > 0:
                query = f"""
                MATCH path = (source {{id: "{concept_id}"}})-[r*1..{max_hops}]-(expanded)
                WHERE source <> expanded
                  AND ALL(rel in relationships(path) WHERE type(rel) IN {relationship_types})
                WITH expanded,
                     labels(expanded)[0] AS expanded_type,
                     length(path) AS hops,
                     [rel in relationships(path) | type(rel)] AS path_relationships
                RETURN DISTINCT
                    expanded.id AS concept_id,
                    expanded.name AS concept_name,
                    expanded_type,
                    hops,
                    path_relationships
                ORDER BY hops, expanded.name
                LIMIT {limit}
                """
            else:
                query = f"""
                MATCH path = (source {{id: "{concept_id}"}})-[r*1..{max_hops}]-(expanded)
                WHERE source <> expanded
                WITH expanded,
                     labels(expanded)[0] AS expanded_type,
                     length(path) AS hops,
                     [rel in relationships(path) | type(rel)] AS path_relationships
                RETURN DISTINCT
                    expanded.id AS concept_id,
                    expanded.name AS concept_name,
                    expanded_type,
                    hops,
                    path_relationships
                ORDER BY hops, expanded.name
                LIMIT {limit}
                """

            result = graph.query(query)
            rows = results_to_list(result.result_set, result.header) if result.result_set else []
            out = to_json({
                "concept_id": concept_id,
                "max_hops": max_hops,
                "relationship_types": relationship_types,
                "total_results": len(rows),
                "expanded": rows,
            })
            if len(out) > 50000:
                out = out[:50000] + "\n... (truncated)"
            return [TextContent(type="text", text=out)]

        elif name == "find_concept_paths":
            source_id = arguments["source_id"]
            target_id = arguments["target_id"]
            max_path_length = min(arguments.get("max_path_length", 3), 5)
            limit = arguments.get("limit", 10)

            # Find shortest paths between concepts
            query = f"""
            MATCH path = shortestPath((source {{id: "{source_id}"}})-[*1..{max_path_length}]-(target {{id: "{target_id}"}}))
            WITH path, length(path) AS path_length
            UNWIND nodes(path) AS node
            UNWIND relationships(path) AS rel
            WITH path, path_length,
                 collect(DISTINCT node.name) AS node_names,
                 collect(DISTINCT type(rel)) AS relationship_types
            RETURN
                path_length,
                node_names,
                relationship_types
            ORDER BY path_length
            LIMIT {limit}
            """

            result = graph.query(query)
            rows = results_to_list(result.result_set, result.header) if result.result_set else []
            out = to_json({
                "source_id": source_id,
                "target_id": target_id,
                "max_path_length": max_path_length,
                "total_paths": len(rows),
                "paths": rows,
            })
            if len(out) > 50000:
                out = out[:50000] + "\n... (truncated)"
            return [TextContent(type="text", text=out)]

        else:
            return [TextContent(type="text", text=to_json({"error": f"Unknown tool: {name}"}))]

    except Exception as e:
        return [TextContent(type="text", text=to_json({"error": f"Error executing tool '{name}': {str(e)}"}))]


def create_sse_app():
    """Create a plain ASGI app that serves the MCP server over SSE"""
    sse = SseServerTransport("/messages/")

    async def asgi_app(scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] == "http":
            if path == "/health":
                await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"status":"ok"}'})
            elif path == "/sse":
                async with sse.connect_sse(scope, receive, send) as streams:
                    await app.run(streams[0], streams[1], app.create_initialization_options())
            elif path.startswith("/messages/"):
                await sse.handle_post_message(scope, receive, send)
            else:
                await send({"type": "http.response.start", "status": 404, "headers": []})
                await send({"type": "http.response.body", "body": b"Not found"})
        elif scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await send({"type": "lifespan.shutdown.complete"})

    return asgi_app


async def run_stdio():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redis Graph MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="Transport mode (default: stdio)")
    parser.add_argument("--host", default="0.0.0.0", help="SSE host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="SSE port (default: 8000)")
    args = parser.parse_args()

    if args.transport == "sse":
        print(f"Starting Redis Graph MCP server (SSE) on http://{args.host}:{args.port}/sse")
        starlette_app = create_sse_app()
        uvicorn.run(starlette_app, host=args.host, port=args.port)
    else:
        asyncio.run(run_stdio())