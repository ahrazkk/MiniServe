"""The subset of the OpenAI API request format we accept (unknown fields are ignored)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StreamOptions(BaseModel):
    include_usage: bool = False


class _Common(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    temperature: float = Field(1.0, ge=0.0)
    top_p: float = Field(1.0, gt=0.0, le=1.0)
    top_k: int = -1  # vLLM extension; 0 or -1 disables it
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    seed: int | None = None
    n: int = 1
    ignore_eos: bool = False  # vLLM extension, used by benchmarks to fix the output length

    @property
    def stop_strings(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)

    @property
    def include_usage(self) -> bool:
        return self.stream_options is not None and self.stream_options.include_usage


class CompletionRequest(_Common):
    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int | None = Field(16, ge=1)  # OpenAI's default for completions
    logprobs: int | None = None
    echo: bool = False


class ContentPart(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str = "text"
    text: str | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: str | list[ContentPart] | None = None

    def text(self) -> str:
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text or "" for part in self.content if part.type == "text")


class ChatCompletionRequest(_Common):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    logprobs: bool = False
