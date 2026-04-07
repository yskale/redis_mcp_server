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
        "You are a biomedical knowledge graph assistant. "
        "Use the available tools to answer questions about diseases, "
        "phenotypes, study variables, and studies. "
        "Always use tools to retrieve data — do not guess."
    )


class QueryResponse(BaseModel):
    answer: str
    tools_used: list[str]
    tool_results: list[dict]


# ── Core agent loop ───────────────────────────────────────────────────────────

async def run_agent(query: str, system_prompt: str) -> QueryResponse:
    if _mcp_session is None:
        raise HTTPException(503, "MCP server not connected")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": query},
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

        # No tool calls — final answer
        if not msg.tool_calls:
            return QueryResponse(
                answer=msg.content or "",
                tools_used=tools_used,
                tool_results=tool_results,
            )

        # Execute each tool call against MCP server
        messages.append(msg.model_dump(exclude_unset=True))

        for tc in msg.tool_calls:
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
