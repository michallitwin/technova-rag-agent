# Roadmap

- [x] LangGraph version of the agent, compared with the hand-written loop on the same 31 questions

## Retrieval
- [ ] Hybrid search (BM25 + vectors, Reciprocal Rank Fusion with k=60) and reranking, with an ablation table (vectors / + BM25 / + reranker)
- [ ] Chunk size comparison with recall@k (check whether the splitter counts characters or tokens)

## Evaluation
- [ ] Held-out set of 10–15 new questions the agents were not tuned on
- [ ] LLM-as-a-judge (pass/fail), validated against hand-labelled answers
- [ ] Multi-turn test cases for conversation memory (e.g. "And when was it ordered?")
- [ ] Prompt-injection test case (a policy chunk with a malicious instruction)
- [ ] Evaluate tool choice and tool arguments separately from the final answer

## Answers and observability
- [ ] Citations checked against the retrieved chunks; tested "I don't have this information" answers
- [ ] Streaming answers; latency p50/p95 (time to first token and full answer)
- [ ] Tracing with Langfuse; token usage and cost per query in the evaluation report
- [X] Demo GIF in the README

## Known issues
- LangGraph agent sometimes answers order + policy questions from one tool (warranty for order 1042 without "24 months")
- Hallucination from related context: "Do you ship to Japan?" gets the Delivery section and a made-up "yes"
- Memory is stored but not always used ("And when was it ordered?" searches the policy instead of the earlier `get_order` result)
- LangGraph answers "I don't have this information" for order 1041 (M02) even though `get_order` returns a clear verdict; appeared after a local Ollama update

## Later: AWS
- [ ] Amazon Bedrock as the LLM, Titan embeddings, pgvector on RDS, deployment on ECS Fargate or App Runner
- [ ] PostgreSQL checkpointer instead of in-memory conversation memory
- [ ] Prompt caching, Bedrock Guardrails with retrieved content tagged as untrusted input
- [ ] Secrets in AWS Secrets Manager; GitHub Actions with OIDC and evaluation in CI
- [ ] Cost per query broken down by tokens, embeddings and reranking