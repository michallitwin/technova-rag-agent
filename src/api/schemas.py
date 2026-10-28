"""Request and response models of the TechNova API."""

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Payload of the question to the support agent."""

    question: str = Field(
        min_length=3,
        max_length=500,
        examples=["Can I still return order 1042?"],
        description="Customer question in plain English",
    )

    @field_validator("question")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("Question must contain at least 3 non-space characters")
        return v


class Step(BaseModel):
    """One step of the agent loop."""

    call: dict | None = Field(description="Tool call chosen by the model (None if the reply was invalid JSON)")
    result: dict = Field(description="Result returned by the tool")


class AskResponse(BaseModel):
    """Answer of the support agent with the steps it took."""

    answer: str = Field(description="Final answer for the customer")
    steps: list[Step] = Field(description="Tool calls made by the agent, in order")
    forced: bool = Field(description="True if the code forced the final answer (the model did not finish on its own)")


class HealthResponse(BaseModel):
    """API health status response."""

    status: str = "ok"


class EvalSummaryResponse(BaseModel):
    """Summary of the latest evaluation on the golden set."""

    report: str = Field(description="File name of the evaluation report")
    model: str = Field(examples=["llama3.2"], description="LLM used in the evaluation")
    hit_at_k: float = Field(ge=0.0, le=1.0, description="Retrieval hit@k on policy questions")
    accuracy: float = Field(ge=0.0, le=1.0, description="Share of correct answers of the ReAct agent")
    avg_steps: float = Field(ge=0.0, description="Average number of tool calls per question")