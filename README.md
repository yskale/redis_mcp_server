# Redis Graph MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server for querying a biomedical knowledge graph stored in Redis (RedisGraph). Supports both **stdio** (Claude Desktop) and **SSE** (HTTP) transports.

## Overview

This server exposes a biomedical knowledge graph as MCP tools and prompts, enabling LLMs and AI agents to:

- Search for diseases, phenotypes, and other biomedical concepts by name
- Enrich searches with synonyms using [Name Resolution SRI](https://name-resolution-sri.renci.org) and [SAP-BERT](https://github.com/cambridgeltl/sapbert)
- Discover study variables connected to a concept across research datasets
- Traverse concept relationships and find paths between entities
- Execute raw Cypher queries for advanced use cases

## Architecture

```
User / Claude Desktop / API Client
            │
            ▼
    redis-graph-agent  (port 8080)
      - REST API for natural language queries
      - LLM reasoning via vLLM (tool-calling mode)
            │
            ▼
    redis-mcp-server  (port 8000)
      - MCP tools and prompts over SSE
      - Synonym enrichment via Name Resolution SRI + SAP-BERT
            │
            ▼
        Redis (RedisGraph)
```

---

## Using with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "redis-graph": {
      "url": "http://localhost:8000/sse"
    }
  }
}
```

Restart Claude Desktop after updating the config.

---

## Using the Agent API

### Via Swagger UI

Navigate to `/docs` on the running agent service.

### Via curl

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"query": "find study variables related to asthma"}'
```

### Request body

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural language question |
| `max_results` | int | 20 | Max results per tool call |
| `system_prompt` | string | built-in | Override the system prompt |

### Response

```json
{
  "answer": "Found 20 study variables related to asthma...",
  "tools_used": ["search_concepts"],
  "tool_results": [...]
}
```

---

## Available Tools

All tools return JSON. Below is each tool, what it does, and a concrete example.

---

### `search_concepts`

Search for biomedical concepts by name, or find study variables related to a concept with **automatic synonym enrichment**.

When `find_variables=true`, the term is expanded via Name Resolution SRI and SAP-BERT (up to 10 results each). The combined CURIEs and labels are then used to search the graph for connected `StudyVariable` nodes — providing broader coverage than a simple name match.

**Find concepts by name:**
```json
{ "search_term": "asthma" }
```

**Filter by node type:**
```json
{ "search_term": "asthma", "node_type": "Disease" }
```

**Find variables with synonym enrichment:**
```json
{ "search_term": "asthma", "find_variables": true, "limit": 20 }
```

---

### `get_concept_graph`

Get the subgraph around a concept.

- **Depth 1** — returns all directly connected nodes of any type
- **Depth 2** — traverses concept → StudyVariable → Study, with a count of how many other concepts each variable links to

**Direct neighbors:**
```json
{ "concept_id": "MONDO:0004979", "expand_depth": 1 }
```

**Variables and studies:**
```json
{ "concept_id": "MONDO:0004979", "expand_depth": 2, "limit": 50 }
```

---

### `get_concept_connections`

List every entity directly connected to a node — relationship type, node type, name, and ID. Returns a type-count summary plus the full leaf-node list. Also works with variable IDs. Optionally filter by node type.

**All connections:**
```json
{ "concept_id": "MONDO:0004979" }
```

**Only connected studies:**
```json
{ "concept_id": "MONDO:0004979", "node_type_filter": "Study" }
```

**Only connected variables:**
```json
{ "concept_id": "MONDO:0004979", "node_type_filter": "StudyVariable" }
```

**Inspect a variable's connections:**
```json
{ "concept_id": "phv00430345.v1.p1" }
```

---

### `search_variables_by_name`

Search for `StudyVariable` nodes by name fragment or by partial ID. Unlike `search_concepts`, this targets only variables and supports ID prefix matching.

**By name fragment:**
```json
{ "search_term": "bmi" }
```

**By partial phv ID:**
```json
{ "search_term": "phv00430" }
```

---

### `find_highly_connected_variables`

Find `StudyVariable` nodes with the most connections to biomedical concepts — useful for discovering the most broadly relevant variables across disease areas.

```json
{ "min_connections": 5, "limit": 20 }
```

---

### `expand_concept`

Traverse the concept graph N hops from a starting node, returning related concepts with hop distance and path relationship types. Optionally filter by relationship type.

**All related concepts within 2 hops:**
```json
{ "concept_id": "MONDO:0004979", "max_hops": 2 }
```

**Filter by relationship type:**
```json
{ "concept_id": "MONDO:0004979", "max_hops": 2, "relationship_types": ["related_to"] }
```

---

### `find_concept_paths`

Find the shortest path(s) between two biomedical concepts.

```json
{
  "source_id": "MONDO:0004979",
  "target_id": "MONDO:0004784",
  "max_path_length": 3
}
```

Returns the node names and relationship types along each path.

---

### `list_graph_schema`

List all node types in the graph with counts. Recommended as a first step for any LLM to understand what data is available.

```json
{ "show_counts": true }
```

---

### `cypher_query`

Execute a raw Cypher query directly against RedisGraph. Use backticks for biolink labels containing dots.

**Count all diseases:**
```cypher
MATCH (n:`biolink.Disease`) RETURN COUNT(n)
```

**Find concepts connected to a variable:**
```cypher
MATCH (v:`biolink.StudyVariable` {id: "phv00430345.v1.p1"})--(c)
RETURN labels(c)[0] AS type, c.name AS name, c.id AS id
LIMIT 20
```

---

## Available Prompts

When connected via Claude Desktop or another MCP client, these prompts appear as guided workflows:

| Prompt | Arguments | Description |
|---|---|---|
| `find_variables_for_concept` | `concept` | Synonym-enriched search for study variables |
| `explore_concept` | `concept_id` | Full exploration: connections, studies, subgraph |
| `find_studies_for_disease` | `disease_name` | End-to-end: disease name → matching studies |
| `explain_variable` | `variable_id` | Plain-language explanation of what a variable measures |
| `find_path_between_concepts` | `concept_a_id`, `concept_b_id` | Relationship path between two concepts |

---

## Local Development

### Prerequisites

- Python 3.12+
- Redis with RedisGraph module running locally

### Installation

```bash
pip install -r requirements.txt
```

### Run MCP server (stdio mode)

```bash
REDIS_HOST=localhost \
REDIS_PORT=6379 \
REDIS_PASSWORD=<password> \
REDIS_GRAPH_NAME=<graph_name> \
  python3 redismcp_server.py
```

### Run MCP server (SSE mode)

```bash
REDIS_HOST=localhost \
REDIS_PORT=6379 \
REDIS_PASSWORD=<password> \
REDIS_GRAPH_NAME=<graph_name> \
  python3 redismcp_server.py --transport sse --port 8000
```

### Run agent

```bash
MCP_SERVER_URL=http://localhost:8000/sse \
VLLM_URL=<your_vllm_endpoint>/v1 \
MODEL=<model_name> \
  uvicorn agent_api:app --host 0.0.0.0 --port 8080
```

---

## Kubernetes Deployment

```bash
# Deploy all services
kubectl apply -f k8s/mcp-server.yaml
kubectl apply -f k8s/agent.yaml
kubectl apply -f k8s/ingress.yaml

# Restart pods to pull a new image
kubectl rollout restart deployment/redis-mcp-server deployment/redis-graph-agent -n <namespace>

# Check pod status
kubectl get pods -n <namespace> -l 'app in (redis-mcp-server,redis-graph-agent)'
```

---

## Configuration

### MCP Server

| Variable | Description |
|---|---|
| `REDIS_HOST` | Redis hostname |
| `REDIS_PORT` | Redis port |
| `REDIS_PASSWORD` | Redis auth password |
| `REDIS_GRAPH_NAME` | RedisGraph graph name |

### Agent

| Variable | Description |
|---|---|
| `MCP_SERVER_URL` | SSE endpoint of the MCP server (e.g. `http://redis-mcp-server:8000/sse`) |
| `VLLM_URL` | vLLM OpenAI-compatible endpoint |
| `MODEL` | LLM model name served by vLLM |
| `MAX_TOOL_TURNS` | Max tool call iterations per query (default: `5`) |
