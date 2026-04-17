from __future__ import annotations

import json
from textwrap import dedent

from tenacity import retry, stop_after_attempt, wait_fixed

from trading_system.config import TradingConfig
from trading_system.llm import LLMClient, TokenUsageTracker
from trading_system.models import DebateResult, SymbolDebate, SymbolMarketData
from trading_system.utils import clamp


class DebateError(RuntimeError):
    pass


class OllamaDebateEngine:
    def __init__(self, config: TradingConfig, logger, token_tracker: TokenUsageTracker | None = None):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config)
        self.token_tracker = token_tracker

    def warmup(self) -> None:
        if not self.llm_client.supports_warmup():
            return
        for model in (self.config.get_debate_model(), self.config.get_decision_model()):
            try:
                self._llm_generate(
                    model=model,
                    prompt="Reply with a compact JSON object: {\"status\":\"ready\"}",
                    temperature=0.0,
                    stage="warmup",
                )
                self.logger.info("Warmed up model %s", model)
            except Exception as exc:  # noqa: BLE001
                raise DebateError(
                    f"Failed to warm up {self.config.llm_provider} model {model}: {exc}"
                ) from exc

    def run_debate_for_symbol(self, symbol_data: SymbolMarketData) -> SymbolDebate:
        bull = self._run_role(symbol_data, role="bull")
        bear = self._run_role(symbol_data, role="bear")
        return SymbolDebate(symbol=symbol_data.symbol, market_data=symbol_data, bull_case=bull, bear_case=bear)

    def _run_role(self, symbol_data: SymbolMarketData, role: str) -> DebateResult:
        prompt = self._build_role_prompt(symbol_data, role)
        raw, payload = self._generate_valid_debate_json(
            prompt=prompt,
            symbol=symbol_data.symbol,
            role=role,
        )
        return DebateResult(
            symbol=payload["symbol"],
            position=payload["position"],
            confidence=clamp(float(payload["confidence"]), 0.0, 1.0),
            arguments=[str(item) for item in payload["arguments"]],
            risks=[str(item) for item in payload["risks"]],
            key_levels={
                "support": payload["key_levels"].get("support"),
                "resistance": payload["key_levels"].get("resistance"),
            },
            raw_response=raw,
        )

    def _generate_valid_debate_json(
        self, prompt: str, symbol: str, role: str
    ) -> tuple[str, dict]:
        current_prompt = prompt
        last_error = "unknown error"
        raw = ""
        for attempt in range(self.config.ollama_retries):
            raw = self._llm_generate(
                model=self.config.get_debate_model(),
                prompt=current_prompt,
                temperature=self.config.debate_temperature if attempt == 0 else 0.0,
                stage="debate",
                symbol=symbol,
                role=role,
            )
            try:
                payload = self._extract_json(raw)
                self._validate_debate_payload(payload, symbol, role)
                return raw, payload
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                current_prompt = self._build_json_repair_prompt(raw, prompt, last_error)
                self.logger.warning(
                    "Debate JSON validation failed for %s/%s on attempt %s: %s raw=%r",
                    symbol,
                    role,
                    attempt + 1,
                    exc,
                    raw[:200],
                )
        raise DebateError(f"Unable to obtain valid debate JSON for {symbol}/{role}: {last_error}")

    def _build_role_prompt(self, symbol_data: SymbolMarketData, role: str) -> str:
        role_instruction = {
            "bull": (
                "You are the bull agent. You must argue only for a LONG position. "
                "Do not present bearish reasoning except to directly refute it."
            ),
            "bear": (
                "You are the bear agent. You must argue only for a SHORT position. "
                "Do not present bullish reasoning except to directly refute it."
            ),
        }[role]
        return dedent(
            f"""
            {role_instruction}

            Analyze the following structured market data and produce STRICT JSON only.
            No markdown, no prose outside the JSON object.

            Required schema:
            {{
              "symbol": "{symbol_data.symbol}",
              "position": "{role}",
              "confidence": 0.0,
              "arguments": ["concise point"],
              "risks": ["concise risk"],
              "key_levels": {{ "support": 0.0, "resistance": 0.0 }}
            }}

            Input data:
            {json.dumps(symbol_data.to_prompt_payload(), indent=2)}
            """
        ).strip()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _llm_generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        stage: str,
        symbol: str | None = None,
        role: str | None = None,
    ) -> str:
        response = self.llm_client.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=600,
            json_mode=True,
        )
        if self.token_tracker is not None:
            self.token_tracker.record(
                stage=stage,
                usage=response.usage,
                symbol=symbol,
                role=role,
            )
        return response.text

    @staticmethod
    def _extract_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise DebateError("No JSON object found in model response.")
            return json.loads(raw[start : end + 1])

    @staticmethod
    def _validate_debate_payload(payload: dict, expected_symbol: str, expected_role: str) -> None:
        required_fields = {"symbol", "position", "confidence", "arguments", "risks", "key_levels"}
        missing = required_fields - set(payload)
        if missing:
            raise DebateError(f"Debate payload missing fields: {sorted(missing)}")
        if payload["symbol"] != expected_symbol:
            raise DebateError(f"Debate payload symbol mismatch: {payload['symbol']} != {expected_symbol}")
        if payload["position"] != expected_role:
            raise DebateError(f"Debate payload role mismatch: {payload['position']} != {expected_role}")
        if not isinstance(payload["arguments"], list) or not isinstance(payload["risks"], list):
            raise DebateError("Debate payload arguments/risks must be lists.")
        if not isinstance(payload["key_levels"], dict):
            raise DebateError("Debate payload key_levels must be an object.")

    @staticmethod
    def _build_json_repair_prompt(raw: str, original_prompt: str, error: str) -> str:
        return dedent(
            f"""
            The previous output was invalid JSON for the required schema.
            Return ONLY ONE valid JSON object.
            No markdown, no explanation, no preamble, no trailing text.
            If any value is uncertain, still emit the required key with the best concise value.

            Required schema:
            {{
              "symbol": "TICKER",
              "position": "bull_or_bear",
              "confidence": 0.0,
              "arguments": ["concise point"],
              "risks": ["concise risk"],
              "key_levels": {{ "support": 0.0, "resistance": 0.0 }}
            }}

            Validation error:
            {error}

            Original task:
            {original_prompt}

            Invalid output:
            {raw}
            """
        ).strip()


DebateEngine = OllamaDebateEngine
