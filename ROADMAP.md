# Roadmap

- [x] LangGraph version of the agent, compared with the hand-written loop on the same 31 questions (accuracy, tool calls, latency)
- [ ] Held-out set of 10–15 new questions the agent was not tuned on
- [ ] Prompt-injection test case (a policy chunk with a malicious instruction)
- [ ] Token usage per query (input/output) in the evaluation report
- [ ] Hybrid search (vectors + keyword, Reciprocal Rank Fusion) and reranking, with an ablation table
- [ ] LLM-as-a-judge (pass/fail), validated against hand-labelled answers

## Later: AWS

- [ ] Amazon Bedrock as the LLM, pgvector on RDS, deployment on ECS Fargate or App Runner
- [ ] Langfuse tracing, prompt caching, evaluation in CI