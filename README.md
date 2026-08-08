# AI-learning

FastAPI service that wraps an LLM via OpenRouter — built step by step to learn tool calling, SSE streaming, and structured output.

## Setup

```bash
cp .env.example .env
# put OPENROUTER_API_KEY in .env
uv sync
uv run uvicorn app.main:app --reload
```

Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)  
Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

```bash
# -N disables curl buffering so you see SSE events as they arrive
curl -N -s http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the weather in London?"}'
```

SSE event types: `status`, `tool`, `delta` (text fragments), `done` (full reply + `tools_used`).
