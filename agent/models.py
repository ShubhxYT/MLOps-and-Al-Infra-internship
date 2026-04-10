from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

# All supported intent types
IntentType = Literal["create_file", "write_code", "summarize", "general_chat", "unknown"]


class Intent(BaseModel):
    """A single classified intent with extracted parameters."""

    intent_type: IntentType
    params: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ToolResult(BaseModel):
    """The result of executing one intent/tool."""

    intent_type: IntentType
    success: bool
    output: str
    file_path: Optional[str] = None


class PipelineResult(BaseModel):
    """Full result of one voice command pipeline run."""

    transcription: str
    intents: list[Intent]
    results: list[ToolResult]
