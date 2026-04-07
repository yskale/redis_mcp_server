FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir uvicorn fastapi openai httpx

COPY redismcp_server.py .
COPY agent_api.py .

# Default: MCP server in SSE mode
# Override CMD in k8s for agent service
CMD ["python3", "redismcp_server.py", "--transport", "sse", "--host", "0.0.0.0", "--port", "8000"]
