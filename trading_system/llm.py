from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import requests

from trading_system.config import TradingConfig
from trading_system.utils import RateLimiter


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class TokenUsage:
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    average_logprob: float | None = None
    average_probability: float | None = None


@dataclass(slots=True)
class LLMResponse:
    text: str
    usage: TokenUsage


@dataclass(slots=True)
class TokenUsageRecord:
    stage: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    average_logprob: float | None = None
    average_probability: float | None = None
    symbol: str | None = None
    role: str | None = None


class TokenUsageTracker:
    def __init__(self) -> None:
        self.records: list[TokenUsageRecord] = []

    def record(
        self,
        *,
        stage: str,
        usage: TokenUsage,
        symbol: str | None = None,
        role: str | None = None,
    ) -> None:
        self.records.append(
            TokenUsageRecord(
                stage=stage,
                provider=usage.provider,
                model=usage.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                average_logprob=usage.average_logprob,
                average_probability=usage.average_probability,
                symbol=symbol,
                role=role,
            )
        )

    def to_payload(self) -> dict[str, Any]:
        totals = self._aggregate(self.records)
        by_stage = {
            stage: self._aggregate(stage_records)
            for stage, stage_records in self._group_by(lambda item: item.stage).items()
        }
        by_model = {
            key: self._aggregate(model_records)
            for key, model_records in self._group_by(
                lambda item: f"{item.provider}:{item.model}"
            ).items()
        }
        return {
            "totals": totals,
            "by_stage": by_stage,
            "by_model": by_model,
            "calls": [asdict(record) for record in self.records],
        }

    def _group_by(
        self, key_builder: Any
    ) -> dict[str, list[TokenUsageRecord]]:
        grouped: dict[str, list[TokenUsageRecord]] = {}
        for record in self.records:
            key = key_builder(record)
            grouped.setdefault(key, []).append(record)
        return grouped

    @staticmethod
    def _aggregate(records: list[TokenUsageRecord]) -> dict[str, int]:
        return {
            "call_count": len(records),
            "input_tokens": sum(item.input_tokens for item in records),
            "output_tokens": sum(item.output_tokens for item in records),
            "total_tokens": sum(item.total_tokens for item in records),
        }


class LLMClient:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.config.validate_llm_provider()
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute, 60)

    def supports_warmup(self) -> bool:
        return self.config.llm_provider == "ollama"

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        json_mode: bool = True,
    ) -> LLMResponse:
        if self.config.llm_provider == "ollama":
            return self._generate_ollama(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                json_mode=json_mode,
            )
        if self.config.llm_provider == "openai":
            return self._generate_openai(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                json_mode=json_mode,
            )
        if self.config.llm_provider == "anthropic":
            return self._generate_anthropic(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        raise LLMError(f"Unsupported llm provider: {self.config.llm_provider}")

    def _generate_ollama(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        self.rate_limiter.acquire()
        response = self.session.post(
            f"{self.config.ollama_host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json" if json_mode else "",
                "options": {
                    "temperature": temperature,
                    "num_predict": max_output_tokens,
                    "top_p": 0.9,
                    "logprobs": True,
                    "top_logprobs": 5,
                },
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("response", "")).strip() or str(payload.get("thinking", "")).strip()
        if not text:
            raise LLMError(f"Ollama returned an empty response for model {model}")
        input_tokens = int(payload.get("prompt_eval_count") or 0)
        output_tokens = int(payload.get("eval_count") or 0)
        average_logprob, average_probability = self._summarize_logprobs(payload.get("logprobs"))
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                provider="ollama",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                average_logprob=average_logprob,
                average_probability=average_probability,
            ),
        )

    def _generate_openai(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        self.rate_limiter.acquire()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self.session.post(
            f"{self.config.openai_base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        try:
            message_content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("OpenAI response missing completion text.") from exc
        text = self._coerce_message_text(message_content)
        if not text:
            raise LLMError(f"OpenAI returned an empty response for model {model}")
        usage = body.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (input_tokens + output_tokens))
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                provider="openai",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            ),
        )

    def _generate_anthropic(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> LLMResponse:
        self.rate_limiter.acquire()
        response = self.session.post(
            f"{self.config.anthropic_base_url}/messages",
            headers={
                "x-api-key": self.config.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        if not text:
            raise LLMError(f"Anthropic returned an empty response for model {model}")
        usage = body.get("usage", {})
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        return LLMResponse(
            text=text,
            usage=TokenUsage(
                provider="anthropic",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    @staticmethod
    def _coerce_message_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        return ""

    @staticmethod
    def _summarize_logprobs(logprobs: Any) -> tuple[float | None, float | None]:
        if not isinstance(logprobs, list):
            return None, None
        values = []
        for item in logprobs:
            if not isinstance(item, dict):
                continue
            value = item.get("logprob")
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            return None, None
        average_logprob = sum(values) / len(values)
        average_probability = math.exp(average_logprob)
        return average_logprob, average_probability
