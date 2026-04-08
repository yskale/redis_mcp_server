# Redis Graph MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server for querying a biomedical knowledge graph stored in Redis (RedisGraph). Supports both stdio (Claude Desktop) and SSE (HTTP) transports.

## Deployed Endpoints (Kubernetes)

| Service | URL |
|---|---|
| Agent API (Swagger UI) | https://redis-agent.example.com/docs |
| Agent API (query) | https://redis-agent.example.com/query |
| MCP Server (SSE) | https://redis-mcp.example.com/sse |
| MCP Server (health) | https://redis-mcp.example.com/health |

## Architecture

```
User / Claude Desktop
        │
        ▼
redis-graph-agent  (port 8080)
  - Accepts natural language queries
  - Uses Gemma-3 via vLLM for reasoning
        │
        ▼
redis-mcp-server  (port 8000)
  - Exposes Redis graph tools via MCP/SSE
  - Synonym enrichment via Name Resolution SRI + SAP-BERT
        │
        ▼
search-redis-master  (namespace: your-namespace)
```

## Using with Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "redis-graph": {
      "url": "https://redis-mcp.example.com/sse"
    }
  }
}
```

Restart Claude Desktop after updating the config.

## Using the Agent API

### Via Swagger UI

Open https://redis-agent.example.com/docs in your browser.

### Via curl

```bash
curl -X POST https://redis-agent.example.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "find study variables related to asthma"}'
```

### Request body

| Field | Type | Default | Description |
|---|---|---|---|
| `query` | string | required | Natural language question |
| `max_results` | int | 20 | Max results per tool call |
| `system_prompt` | string | built-in | Override the system prompt |

---

## Available Tools

All tools return JSON. Below is each tool, what it does, and a concrete example.

---

### `search_concepts`

Search for biomedical concepts by name, or find study variables related to a concept with automatic synonym enrichment.

When `find_variables=true`, the term is expanded via **Name Resolution SRI** and **SAP-BERT** (up to 10 results each), then the combined CURIEs and labels are used to search for connected `StudyVariable` nodes.

**Example — find concepts:**
```json
{
  "search_term": "asthma",
  "find_variables": false
}
```
```json
{
  "search_term": "asthma",
  "node_type": "Disease",
  "find_variables": false
}
```

**Example — find variables with synonym enrichment:**
```json
{
  "search_term": "asthma",
  "find_variables": true,
  "limit": 20
}
```
Returns variables like `asthma_hospital_Yes12mo`, `esp_asthma_status_baseline` linked via 17 matched synonyms (MONDO:0004979, EFO:1002011, UMLS:C0581126, ...).

---

### `get_concept_graph`

Get the subgraph around a concept. At depth 1, returns all directly connected nodes of any type. At depth 2, traverses concept → StudyVariable → Study and includes how many other concepts each variable is linked to.

**Example — explore direct neighbors:**
```json
{
  "concept_id": "MONDO:0004979",
  "expand_depth": 1
}
```

**Example — get variables and studies:**
```json
{
  "concept_id": "MONDO:0004979",
  "expand_depth": 2,
  "limit": 50
}
```

---

### `get_concept_connections`

List every entity directly connected to a concept — including relationship type, node type, name, and ID. Returns a summary (count per entity type) plus the full list of leaf nodes. Optionally filter by node type.

**Example — all connections:**
```json
{
  "concept_id": "MONDO:0004979"
}
```

**Example — only connected studies:**
```json
{
  "concept_id": "MONDO:0004979",
  "node_type_filter": "Study"
}
```

**Example — only connected variables:**
```json
{
  "concept_id": "MONDO:0004979",
  "node_type_filter": "StudyVariable"
}
```

Also works with variable IDs to inspect a variable's connections:
```json
{
  "concept_id": "phv00430345.v1.p1"
}
```

---

### `search_variables_by_name`

Search for `StudyVariable` nodes by free-text name fragment or by partial phv ID. Unlike `search_concepts`, this targets only variables and supports ID prefix matching.

**Example — search by name fragment:**
```json
{
  "search_term": "bmi"
}
```

**Example — search by partial phv ID:**
```json
{
  "search_term": "phv00430"
}
```

---

### `find_highly_connected_variables`

Find `StudyVariable` nodes with the most connections to biomedical concepts — useful for discovering which variables are the most broadly relevant across disease areas.

**Example:**
```json
{
  "min_connections": 5,
  "limit": 20
}
```

---

### `expand_concept`

Traverse the concept graph N hops from a starting concept, returning related concepts with hop distance and path relationship types. Optionally filter by relationship type.

**Example — find all concepts within 2 hops:**
```json
{
  "concept_id": "MONDO:0004979",
  "max_hops": 2
}
```

**Example — only `related_to` relationships:**
```json
{
  "concept_id": "MONDO:0004979",
  "max_hops": 2,
  "relationship_types": ["related_to"]
}
```

---

### `find_concept_paths`

Find the shortest path(s) between two biomedical concepts through the knowledge graph.

**Example:**
```json
{
  "source_id": "MONDO:0004979",
  "target_id": "MONDO:0004784",
  "max_path_length": 3
}
```
Returns node names and relationship types along each path.

---

### `list_graph_schema`

List all node types in the graph with counts. Essential first step for any LLM to understand what data is available.

**Example:**
```json
{
  "show_counts": true
}
```

---

### `cypher_query`

Execute a raw Cypher query directly against RedisGraph. Use backticks for biolink labels with dots.

**Example — count all diseases:**
```cypher
MATCH (n:`biolink.Disease`) RETURN COUNT(n)
```

**Example — find concepts connected to a variable:**
```cypher
MATCH (v:`biolink.StudyVariable` {id: "phv00430345.v1.p1"})--(c)
RETURN labels(c)[0] AS type, c.name AS name, c.id AS id
LIMIT 20
```

---

## Available Prompts

When connected via Claude Desktop or another MCP client, these prompts appear as guided workflows:

| Prompt | Arguments | What it does |
|---|---|---|
| `find_variables_for_concept` | `concept` | Synonym-enriched variable search |
| `explore_concept` | `concept_id` | Full concept exploration (connections + studies + subgraph) |
| `find_studies_for_disease` | `disease_name` | End-to-end: disease name → studies |
| `explain_variable` | `variable_id` | Plain-language explanation of a variable |
| `find_path_between_concepts` | `concept_a_id`, `concept_b_id` | Relationship path between two concepts |

---

## Kubernetes Deployment

```bash
# Deploy
kubectl apply -f k8s/mcp-server.yaml
kubectl apply -f k8s/agent.yaml
kubectl apply -f k8s/ingress.yaml

# Restart (pull new image)
kubectl rollout restart deployment/redis-mcp-server deployment/redis-graph-agent -n your-namespace

# Check status
kubectl get pods -n your-namespace -l 'app in (redis-mcp-server,redis-graph-agent)'
```

## Local Development

### Prerequisites

- Python 3.12+
- Redis with RedisGraph module
- `kubectl` access to `your-namespace` namespace

### Installation

```bash
pip install -r requirements.txt
```

### Port-forward Redis

```bash
kubectl port-forward -n your-namespace svc/search-redis-master 6379:6379
```

### Run MCP server locally (stdio)

```bash
REDIS_HOST=localhost REDIS_PORT=6379 REDIS_PASSWORD=<password> REDIS_GRAPH_NAME=test \
  python3 redismcp_server.py
```

### Run MCP server locally (SSE)

```bash
REDIS_HOST=localhost REDIS_PORT=6379 REDIS_PASSWORD=<password> REDIS_GRAPH_NAME=test \
  python3 redismcp_server.py --transport sse --port 8000
```

### Run agent locally

```bash
MCP_SERVER_URL=http://localhost:8000/sse \
VLLM_URL=http://vllm-server.svc.cluster.local/v1 \
MODEL=google/gemma-3-12b-it \
  uvicorn agent_api:app --host 0.0.0.0 --port 8080
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | _(required)_ | Redis auth password |
| `REDIS_GRAPH_NAME` | `test` | RedisGraph graph name |
| `MCP_SERVER_URL` | _(required)_ | MCP server URL (agent only) |
| `VLLM_URL` | _(required)_ | vLLM endpoint (agent only) |
| `MODEL` | _(required)_ | LLM model name (agent only) |
| `MAX_TOOL_TURNS` | `5` | Max tool call iterations per query |
