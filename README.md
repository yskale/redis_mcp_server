# Redis Graph MCP Server

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server for querying a biomedical knowledge graph stored in Redis (RedisGraph). Supports both stdio (Claude Desktop) and SSE (HTTP) transports.

## Prerequisites

- Python 3.12+ (conda env `koios`)
- Redis with the RedisGraph module (version 4.x)
- `kubectl` access to the `your-namespace` namespace
- Port-forward to the Redis service (see [Connecting](#connecting))

## Installation

```bash
pip install -r requirements.txt
```

## Connecting

Redis runs in Kubernetes. You must port-forward before starting the server:

```bash
kubectl port-forward -n your-namespace svc/search-redis-master 6379:6379
```

Keep this running in a separate terminal.

## Configuration

Set via environment variables or a `.env` file in the server directory:

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | _(required)_ | Redis auth password |
| `REDIS_GRAPH_NAME` | `test` | RedisGraph graph name |

## Usage

### stdio mode (Claude Desktop)

This is the default. Claude Desktop spawns the server as a subprocess automatically.

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "redis-graph": {
      "command": "/Users/ykale/anaconda3/envs/koios/bin/python3",
      "args": ["/Users/ykale/redis_mcp_server/redismcp_server.py"],
      "env": {
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "REDIS_PASSWORD": "<password>",
        "REDIS_GRAPH_NAME": "test"
      }
    }
  }
}
```

Restart Claude Desktop after updating the config.

### SSE mode (HTTP server)

Exposes the MCP server as an HTTP service. Supports multiple simultaneous clients.

```bash
REDIS_HOST=localhost REDIS_PORT=6379 REDIS_PASSWORD=<password> REDIS_GRAPH_NAME=test \
  python3 redismcp_server.py --transport sse --port 8000
```

Options:

```
--transport   stdio | sse   (default: stdio)
--host        bind address  (default: 0.0.0.0)
--port        port number   (default: 8000)
```

SSE endpoints:
- `GET  /sse`        — establish SSE stream
- `POST /messages/`  — send MCP messages (session_id required)

### MCP Inspector (browser UI for testing)

```bash
# stdio
npx @modelcontextprotocol/inspector \
  /Users/ykale/anaconda3/envs/koios/bin/python3 \
  /Users/ykale/redis_mcp_server/redismcp_server.py

# SSE (start the server first)
npx @modelcontextprotocol/inspector http://localhost:8000/sse
```

## Available Tools

| Tool | Description |
|---|---|
| `cypher_query` | Execute a raw Cypher query on the graph |
| `search_concepts` | Search biomedical concepts by name (disease, phenotype, etc.) |
| `get_concept_graph` | Get the subgraph for a concept including studies and variables |
| `find_connected_studies` | Find all studies connected to a concept via StudyVariables |
| `get_concept_connections` | Count and list entity types connected to a concept |
| `list_graph_schema` | List all node types with counts |
| `find_highly_connected_variables` | Find StudyVariables with the most connections |
| `explore_concept_neighborhood` | Explore immediate neighbors of a concept |
| `search_variables_by_name` | Search StudyVariables by name or ID pattern |
| `get_variable_details` | Get full details for a specific StudyVariable |
| `expand_concept` | Find related concepts up to N hops away |
| `find_concept_paths` | Find shortest paths between two concepts |

## Example Cypher Queries

```cypher
-- Find all diseases
MATCH (n:`biolink.Disease`) RETURN n.name, n.id LIMIT 10

-- Find variables related to asthma
MATCH (c)--(v:`biolink.StudyVariable`)
WHERE c.name CONTAINS 'asthma'
RETURN c.name, v.name, v.id LIMIT 20

-- Find studies connected to a concept
MATCH (c {id: "MONDO:0004979"})--(v:`biolink.StudyVariable`)--(s:`biolink.Study`)
RETURN c.name, v.name, s.name LIMIT 20
```

## Architecture

```
Claude Desktop (stdio)
        │
        ▼
redismcp_server.py  ◄──── SSE (HTTP) ──── Browser / Python client / MCP Inspector
        │
        ▼
  localhost:6379
        │
        ▼
kubectl port-forward
        │
        ▼
search-redis-master-0  (namespace: your-namespace)
```
