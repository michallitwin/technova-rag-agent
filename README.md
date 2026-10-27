# TechNova RAG Agent

A customer-support agent for **TechNova**, a fictional electronics store. It answers questions by combining two sources:

- **store policy documents** (returns, delivery, warranty, payments) via RAG over a vector database,
- **the orders database** (SQL) via a read-only tool.

The agent is a hand-written **ReAct loop** (no framework): the LLM decides step by step which tool to call, the code runs it, and the result goes back into the history until the model can answer. Business decisions (return eligibility, delivery fee) are made **in code**; the LLM only explains them.

> Status: MVP in progress. The agent and its evaluation work; API, Docker, tests and a LangGraph version are next (see [ROADMAP.md](ROADMAP.md)).

## Results at a glance

Evaluated on a golden set of 31 questions with `llama3.2` running locally:

| | One-step baseline | ReAct agent |
|---|---|---|
| **Correct answers** | 28/31 (90%) | **29/31 (94%)** |
| Avg. tool calls per question | 1.0 | 1.06 |
| Avg. latency | 4.4 s | 5.1 s |
| Retrieval hit@2 | 100% | 100% |

Details, metric definitions and the history of changes are in [Evaluation](#evaluation).

## Example

```
$ uv run --env-file .env python -m src.agent.react "Can I still return order 1042?"
Step 1: get_order(order_id=1042) -> delivered, 16 days ago, return window open
Step 2: final answer

Answer: Yes, you can return the Smartwatch until 2026-11-09.
```

The one-step baseline (router → one tool → answer) often answers the same question from the policy alone, without checking the order.

## Question types

| Type | Example | Source |
|------|---------|--------|
| Policy | How long is the warranty? | documents (RAG) |
| Order | What is the status of order 1043? | SQL |
| Mixed | Can I still return order 1042? | order + business rule |
| Not in data | Do you ship to Japan? | none → "I don't have this information" |
| Missing info | Where is my order? | none → ask for the order number |

## Architecture

```
User question
     |
     v
+--> LLM decides (JSON) -------------------------+
|        | {"tool": ..., "args": {...}}           | {"final_answer": "..."}
|        v                                        v
|   tool already used? -> break -> forced answer  Answer
|        v
|   run_tool()
|     search_policy(query)   Chroma, cosine similarity, chunks below threshold dropped
|     get_order(order_id)    SQLite (read-only) + return_status, delivery_status
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
| Evaluation | `src/eval/run_eval.py`, `data/eval/golden_set.json` |
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

# evaluation with a frozen date (results are saved to data/eval/results/)
TODAY=2026-10-26 uv run --env-file .env python -m src.eval.run_eval

uv run ruff check .                                     # lint
```

## Evaluation

### Method

- **Golden set** (`data/eval/golden_set.json`): 31 questions in 5 categories (10 policy, 6 order, 8 mixed, 4 not in data, 3 missing info). Each question has the expected facts (with accepted wordings), the expected tools and, for policy questions, the expected policy section.
- **Correctness:** the answer must contain at least one accepted wording of every expected fact, matched as **whole words** (so `no` does not match `not` or `know`).
- **Retrieval hit@k:** measured separately from the LLM: is the expected section among the top-k chunks? This tells whether a wrong answer is the retriever's or the model's fault.
- Also measured: tool choice, tool calls per question, share of forced answers, latency.
- **Frozen date:** `TODAY=2026-10-26`, so answers about return windows do not change from day to day.
- Each run saves the full report (every answer) as JSON, so the grading can be re-run without calling the model again.

### Current results (v4)

| Metric | Baseline | ReAct |
|---|---|---|
| Correct, overall | 90% | **94%** |
| Correct, policy | 100% | 90% |
| Correct, order | 100% | 100% |
| Correct, mixed | 88% | **100%** |
| Correct, not in data | 100% | 75% |
| Correct, missing info | 33% | **100%** |
| Avg. tool calls | 1.0 | 1.06 |
| Forced answers | – | 23% |
| Avg. latency | 4.4 s | 5.1 s |

### History of changes

| Version | Change | Baseline | ReAct | ReAct tool calls | ReAct latency |
|---|---|---|---|---|---|
| v1 | first run (substring grading) | 20/30 | 21/30 | 2.27 | 8.4 s |
| v1b | **whole-word grading** (same answers, re-graded) | 18/30 | 20/30 | – | – |
| v2 | **similarity threshold** 0.35 in `search_policy` + missing wordings added to the key | 24/30 | 23/30 | 2.2 | 8.1 s |
| v3 | **delivery decision in code**, **one call per tool**, edge case: order of exactly $70.00 | 22/31 | **24/31** | **1.06** | **5.0 s** |
| v4 | unambiguous `delivery_status` ("NOT free delivery: ..."); key: grade the return **verdict** instead of the date, accept more wordings | 28/31 | **29/31** | 1.06 | 5.1 s |

What the history shows:

- **v1 → v1b:** the first grader was too lenient (`"no" in "I don't know"`). Fixing it lowered the scores but made them honest.
- **v2:** with no relevant paragraph, the agent used to invent facts ("Yes, we ship to Japan"). Dropping chunks below a similarity threshold (chosen from measured scores: answerable questions ≥ 0.40, unanswerable ≤ 0.30) cut these hallucinations. After v2 the simpler baseline was slightly better than ReAct: error analysis showed the loop calling extra tools "just in case" and getting distracted.
- **v3:** the new edge case (an order of exactly $70.00) immediately caught a wrong "free delivery" answer even though the code had computed the fee correctly.
- **v3:** allowing each tool once per question and putting the delivery-fee decision into the data made ReAct both better than the baseline and almost as cheap (1.06 tool calls instead of 2.2).

- **v4:** two changes at once, so the jump is not all agent improvement. The agent part: "NOT free delivery: ..." instead of "... is not over $70" (the small model missed the negation). The grading part: return questions are graded on the yes/no verdict (the date is a nice-to-have), and correct answers were no longer failed for wording ("cards", "check your order number").

### Known limitations of the evaluation

- **The agent was tuned on the same 31 questions it is evaluated on**, so 94% is optimistic. A held-out set of new questions is needed to check that the fixes generalise.
- 31 questions is a small set; one question is ~3 percentage points.
- Keyword grading checks facts, not tone or completeness; an LLM-as-a-judge (e.g. RAGAS) is planned for v2.
- Remaining ReAct errors: an incomplete policy answer (return shipping without "unless damaged") and "we do not have a store in Krakow" instead of "I don't know".

## Design decisions

| Decision | Alternative | Why |
|----------|-------------|-----|
| Two stores: Chroma for documents, SQLite for orders | one vector DB | documents are searched by meaning, orders by exact ID |
| Hand-written ReAct loop first | LangGraph from day one | understand the mechanism; framework version comes later for comparison |
| Return eligibility and delivery fee decided in code (`return_status`, `delivery_status`: ready-made sentences) | LLM compares numbers with the policy | the model said "yes" for a 38-day-old order with a 30-day window and "$59.99 is over $70", even with prompt rules. **Code decides, the LLM explains.** |
| Decision sentences start with the verdict ("NOT free delivery: ...", "Return NOT possible: ...") | negation in the middle ("... is not over $70") | the small model missed the negation for an order of exactly $70.00 |
| Ready-made decision **plus** a short prompt hint to use it | data only / prompt only | data alone was ignored for delivery, prompts alone were ignored for returns; together they work |
| Business rules in config (`RETURN_WINDOW_DAYS`, `FREE_DELIVERY_THRESHOLD`, `DELIVERY_FEE`) | read them from the policy text | deterministic; trade-off: the rule exists in config and in the policy document |
| Each tool at most once per question | allow repeated searches | the model kept re-searching with new words and then made unnecessary calls; with a 4-paragraph policy one search is enough |
| Similarity threshold for retrieved chunks | always pass top-k | weakly related paragraphs led to invented answers |
| Actionable tool errors ("Ask the customer for their order number.") | technical errors | the model follows a next step written in the data much better than a rule in the prompt |
| Forced final answer when the loop stalls or hits `max_steps` | stronger prompt only | a stricter prompt did not stop the loop; a rule in code always works |
| Frozen date (`TODAY`) for evaluation | real date | reproducible results; no flaky tests |
| Prompts in `config/prompts/` | strings in code | single source of truth, change behaviour without touching code |
| Placeholders in tool examples | real values in examples | the model copied example values (`return period`, `1042`) into its calls |
| Sources appended from metadata | LLM writes citations | the small model ignored the citation format |
| `upsert` with stable chunk IDs | `add` | idempotent indexing: re-running does not duplicate chunks |
| Read-only SQL connection + `?` parameters | regular connection | no writes, no SQL injection |
| Settings from env vars (`--env-file .env`) | hardcoded values | same code locally, in Docker and on AWS |

## What didn't work

- **One-step router on mixed questions.** "Can I still return order 1042?" got a confident answer from the policy alone, without checking the order (a silent failure). Fixed by the ReAct loop.
- **The loop did not stop.** The model collected both facts but kept repeating calls until the step limit. A prompt rule did not help; a forced answer in code did.
- **Example bias.** The model copied values from the tool examples in the prompt (e.g. `get_order(1042)` for a question about another customer). Fixed with placeholders.
- **Number comparison.** Three prompt versions did not stop wrong answers about the return window (0/3), and a boolean `return_window_open: false` was ignored too. A ready-made sentence fixed it: for a small model, the form of the data matters more than instructions.
- **Too strong error message.** "Ask the customer for their order number" made the agent ask for an order number on plain policy questions ("How do I report a defect?") after an unnecessary `get_order` call. Fixed at the root by allowing each tool only once.
- **Grading errors in both directions.** Substring matching passed wrong answers (`no` inside `know`); whole-word matching then failed correct ones (`card` vs `cards`). Both fixed by checking failures by hand before trusting the numbers.

## Security and limitations

- **No authentication in the MVP.** Anyone who knows an order number can see that order. In production, order lookups must be scoped to the logged-in customer: the customer ID comes from the session, never from text the user types to the LLM.
- The agent also repeats the customer's name to whoever gives the order number.
- No search by customer name, on purpose (privacy). The agent asks for the order number instead.
- Runs locally on a small model (`llama3.2`); some answers are incomplete (e.g. delivery cost without the free-delivery threshold).