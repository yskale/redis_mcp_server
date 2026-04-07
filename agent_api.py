#!/usr/bin/env python3
"""
Agent API — bridges user queries to vLLM (tool calling) + Redis MCP server.

Flow:
  POST /query  →  vLLM (Gemma-3 with tool calling)
                      ↕ tool calls
               MCP server (SSE) → Redis graph
"""

import os
import json
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.sse import sse_client

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://redis-mcp-server:8000/sse")
VLLM_URL       = os.getenv("VLLM_URL", "http://vllm-server.svc.cluster.local/v1")
MODEL          = os.getenv("MODEL", "google/gemma-3-12b-it")
MAX_TOOL_TURNS = int(os.getenv("MAX_TOOL_TURNS", "5"))

# ── MCP session (kept alive for lifetime of app) ──────────────────────────────

_mcp_session: ClientSession | None = None
_openai_tools: list[dict] = []
_mcp_cm = None


def _mcp_tool_to_openai(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_session, _openai_tools, _mcp_cm

    log.info(f"Connecting to MCP server at {MCP_SERVER_URL}")
    async with sse_client(MCP_SERVER_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            _mcp_session = session
            _openai_tools = [_mcp_tool_to_openai(t) for t in tools.tools]
            log.info(f"Loaded {len(_openai_tools)} tools from MCP server")
            yield

    _mcp_session = None
    _openai_tools = []


app = FastAPI(title="Redis Graph Agent API", lifespan=lifespan)

llm = AsyncOpenAI(
    base_url=VLLM_URL,
    api_key="EMPTY",
)

# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str
    max_results: int = 20
    system_prompt: str = (
        "You are a biomedical knowledge graph assistant with access to a Redis knowledge graph. "
        "Always use the available tools to retrieve data — never guess or make up answers.\n\n"
        "CRITICAL RULES for tool arguments:\n"
        "- When calling search_concepts, search_variables_by_name, or any search tool, "
        "extract only the KEY TERM from the user's question. "
        "For example: 'what variables are related to asthma' → search_term='asthma'. "
        "'find studies about cholesterol levels' → search_term='cholesterol'. "
        "NEVER pass the full question as the search_term.\n"
        "- Always provide ALL required arguments for a tool. "
        "Required args: search_concepts needs 'query', search_variables_by_name needs 'search_term', "
        "get_concept_graph needs 'concept_id', find_connected_studies needs 'concept_id'.\n"
        "- Start with search_concepts to find concept IDs, then use those IDs with other tools.\n\n"
        "Available tools summary:\n"
        "- search_concepts(query, node_type, find_variables, limit): search by keyword\n"
        "- search_variables_by_name(search_term, limit): find study variables by name\n"
        "- get_concept_graph(concept_id, hops, limit): get full subgraph for a concept\n"
        "- find_connected_studies(concept_id, limit): find studies linked to a concept\n"
        "- get_concept_connections(concept_id): count connections for a concept\n"
        "- cypher_query(query, limit): run a raw Cypher query\n"
        "- list_graph_schema(include_counts): show node types and relationships\n"
        "- find_highly_connected_variables(min_connections, limit): find well-connected variables\n"
        "- explore_concept_neighborhood(concept_id, hops): explore nearby concepts\n"
        "- expand_concept(concept_id, max_hops, limit): expand a concept's graph\n"
        "- find_concept_paths(source_id, target_id): find paths between two concepts\n"
        "- get_variable_details(variable_id): get details for a specific variable\n\n"
        "After retrieving data, provide a clear, structured summary of the findings."
    )


class QueryResponse(BaseModel):
    answer: str
    tools_used: list[str]
    tool_results: list[dict]


# ── Core agent loop ───────────────────────────────────────────────────────────

async def extract_keywords(query: str) -> str:
    """Use the LLM to extract the key biomedical term from a natural language query."""
    response = await llm.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Extract the single most important biomedical keyword or concept name from the user's question. Reply with ONLY the keyword, nothing else. Examples: 'what variables are related to asthma' → 'asthma', 'find studies about high cholesterol' → 'cholesterol', 'show me diabetes research' → 'diabetes'."},
            {"role": "user", "content": query},
        ],
    )
    keyword = (response.choices[0].message.content or query).strip().strip('"').strip("'")
    log.info(f"Extracted keyword: '{keyword}' from query: '{query}'")
    return keyword


async def run_agent(query: str, system_prompt: str) -> QueryResponse:
    if _mcp_session is None:
        raise HTTPException(503, "MCP server not connected")

    # Extract the key biomedical term so Gemma-3 uses it in tool args
    keyword = await extract_keywords(query)

    user_message = (
        f"{query}\n\n"
        f"[Key search term to use in tool arguments: '{keyword}']"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    tools_used: list[str] = []
    tool_results: list[dict] = []

    for turn in range(MAX_TOOL_TURNS):
        response = await llm.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=_openai_tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # Gemma-3 sometimes emits tool calls as plain text JSON instead of tool_calls field
        # Parse them out if tool_calls is empty but content looks like a tool call list
        parsed_text_calls = []
        if not msg.tool_calls and msg.content:
            try:
                text = msg.content.strip().strip("```json").strip("```").strip()
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed and "function" in parsed[0]:
                    parsed_text_calls = parsed
                    log.info(f"Parsed {len(parsed_text_calls)} tool calls from text content")
            except Exception:
                pass

        # No tool calls — final answer
        if not msg.tool_calls and not parsed_text_calls:
            return QueryResponse(
                answer=msg.content or "",
                tools_used=tools_used,
                tool_results=tool_results,
            )

        # Execute each tool call against MCP server
        if msg.tool_calls:
            messages.append(msg.model_dump(exclude_unset=True))

        # Handle standard tool_calls
        for tc in (msg.tool_calls or []):
            tool_name = tc.function.name
            tool_args: dict[str, Any] = json.loads(tc.function.arguments)

            log.info(f"Calling tool: {tool_name}({tool_args})")
            tools_used.append(tool_name)

            try:
                result = await _mcp_session.call_tool(tool_name, tool_args)
                content = result.content[0].text if result.content else "No results"
            except Exception as e:
                content = f"Tool error: {e}"

            tool_results.append({"tool": tool_name, "args": tool_args, "result": content})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
            })

        # Handle text-parsed tool calls (Gemma-3 fallback)
        if parsed_text_calls:
            all_results = []
            # Build a lookup of required args per tool
            tool_schema = {t["function"]["name"]: t["function"].get("parameters", {}) for t in _openai_tools}
            # Use extracted keyword for fallback arg filling
            user_query = keyword

            for tc in parsed_text_calls:
                tool_name = tc.get("function") or tc.get("name", "")
                tool_args = tc.get("parameters") or tc.get("arguments") or {}

                # Fill missing required string args with the user query as best-effort
                schema = tool_schema.get(tool_name, {})
                required = schema.get("required", [])
                props = schema.get("properties", {})
                for req_arg in required:
                    if req_arg not in tool_args:
                        arg_type = props.get(req_arg, {}).get("type", "string")
                        if arg_type == "string":
                            tool_args[req_arg] = user_query
                            log.warning(f"Missing required arg '{req_arg}' for {tool_name}, using query text as fallback")

                log.info(f"Calling tool (text-parsed): {tool_name}({tool_args})")
                tools_used.append(tool_name)
                try:
                    result = await _mcp_session.call_tool(tool_name, tool_args)
                    content = result.content[0].text if result.content else "No results"
                except Exception as e:
                    content = f"Tool error: {e}"
                tool_results.append({"tool": tool_name, "args": tool_args, "result": content})
                all_results.append(f"[{tool_name}]: {content}")

            # Ask model to summarize the results
            messages.append({"role": "user", "content": "Here are the tool results:\n\n" + "\n\n".join(all_results) + "\n\nPlease summarize the answer."})
            summary = await llm.chat.completions.create(model=MODEL, messages=messages)
            return QueryResponse(
                answer=summary.choices[0].message.content or "",
                tools_used=tools_used,
                tool_results=tool_results,
            )

    # Exceeded max turns — return whatever the last message was
    return QueryResponse(
        answer=messages[-1].get("content", "Max tool turns reached."),
        tools_used=tools_used,
        tool_results=tool_results,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    return await run_agent(request.query, request.system_prompt)


@app.get("/tools")
async def list_tools():
    return {"tools": _openai_tools, "count": len(_openai_tools)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "mcp_connected": _mcp_session is not None,
        "tools_loaded": len(_openai_tools),
        "model": MODEL,
    }
