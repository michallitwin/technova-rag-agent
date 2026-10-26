# TechNova RAG Agent

A customer-support agent for **TechNova**, a fictional electronics store. It answers questions by combining two sources:

- **store policy documents** (returns, delivery, warranty, payments) via RAG over a vector database,
- **the orders database** (SQL) via a read-only tool.

The agent is a hand-written **ReAct loop** (no framework): the LLM decides step by step which tool to call, the code runs it, and the result goes back into the history until the model can answer.

> Status: MVP in progress. Evaluation, API, Docker and tests are next (see [ROADMAP.md](ROADMAP.md)).

## Example

```
$ uv run --env-file .env python -m src.agent.react "Can I still return order 1042?"
Step 1: search_policy(query='return period') -> 2 chunks
Step 2: get_order(order_id=1042) -> delivered, 16 days ago
Step 3: repeated call -> forcing final answer

Answer: ... order 1042 was delivered 16 days ago, so you can still return it.
```

A one-step baseline agent (router → one tool → answer) answers the same question from the policy alone, without checking the order. The ReAct loop fixes this by using both tools.

## Question types

| Type | Example | Source |
|------|---------|--------|
| Policy | How long is the warranty? | documents (RAG) |
| Order | What is the status of order 1043? | SQL |
| Mixed | Can I still return order 1042? | both |
| Not in data | Do you have a store in Krakow? | none → "I don't know" |

## Architecture

```
User question
     |
     v
+--> LLM decides (JSON) -------------------------+
|        | {"tool": ..., "args": {...}}           | {"final_answer": "..."}
|        v                                        v
|   repeated call? -> break -> forced answer    Answer
|        v
|   run_tool()
|     search_policy(query)   Chroma: embeddings + cosine similarity
|     get_order(order_id)    SQLite (read-only), days_since_delivery
|        |
+--- result added to history
     max_steps reached -> forced answer
```

| Component | File |
|-----------|------|
| Settings from environment variables | `src/config.py`, `.env.example` |
| LLM client (Ollama now, Amazon Bedrock later) | `src/llm/client.py` |
| RAG: chunking, indexing, search | `src/rag/pipeline.py` |
| Tools: `search_policy`, `get_order`, `run_tool` | `src/agent/tools.py` |
| ReAct loop + CLI | `src/agent/react.py` |
| One-step baseline agent | `src/agent/router.py`, `src/agent/agent.py` |
| Final answer from tool results (shared) | `src/agent/answer.py` |
| Prompts (not in code) | `config/prompts/*.txt` |
| Orders database seed script | `scripts/create_db.py` |

## Quickstart

Requirements: Python 3.13, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/).

```bash
ollama pull llama3.2
uv sync
cp .env.example .env

uv run --env-file .env python -m scripts.create_db      # create the orders database
uv run --env-file .env python -m src.rag.pipeline       # build the vector index
uv run --env-file .env python -m src.agent.react "Can I still return order 1042?"
uv run --env-file .env python -m src.agent.react -v "..."   # show full tool results
```

## Design decisions

| Decision | Alternative | Why |
|----------|-------------|-----|
| Two stores: Chroma for documents, SQLite for orders | one vector DB | documents are searched by meaning, orders by exact ID |
| Hand-written ReAct loop first | LangGraph from day one | understand the mechanism; framework version comes later for comparison |
| Date arithmetic in the tool (`days_since_delivery`) | LLM computes days | small models make arithmetic mistakes; code is deterministic |
| Forced final answer when the model repeats a call or hits `max_steps` | stronger prompt only | a stricter prompt did not stop the loop; a rule in code always works |
| Prompts in `config/prompts/` | strings in code | single source of truth, change behaviour without touching code |
| Placeholders in tool examples | real values in examples | the model copied example values (`return period`, `1042`) into its calls |
| Sources appended from metadata | LLM writes citations | the small model ignored the citation format |
| `upsert` with stable chunk IDs | `add` | idempotent indexing: re-running does not duplicate chunks |
| Read-only SQL connection + `?` parameters | regular connection | no writes, no SQL injection |
| Settings from env vars (`--env-file .env`) | hardcoded values | same code locally, in Docker and on AWS |

## What didn't work

- **One-step router on mixed questions.** "Can I still return order 1042?" got a confident answer from the policy alone, without checking the order (a silent failure). Fixed by the ReAct loop.
- **The loop did not stop.** The model collected both facts but kept repeating calls until the step limit. A prompt rule did not help; a forced answer in code did.
- **Example bias.** The model copied values from the tool examples in the prompt (e.g. `get_order(1042)` for a question about another customer). Fixed with placeholders: steps for a warranty question dropped from 2 to 1.
- **Number comparison.** For an order delivered 38 days ago the model still said the 30-day return window was open. Under investigation.

## Security and limitations

- **No authentication in the MVP.** Anyone who knows an order number can see that order. In production, order lookups must be scoped to the logged-in customer: the customer ID comes from the session, never from text the user types to the LLM.
- No search by customer name, on purpose (privacy). The agent asks for the order number instead.
- Runs locally on a small model (`llama3.2`), so answers can be vague; evaluation will measure this.

## Evaluation

Coming next: a golden set of ~30 questions (policy / order / mixed / not in data) comparing the baseline and the ReAct agent on answer correctness, tool choice, hit@k, "I don't know" behaviour, steps per question, share of forced answers and latency.