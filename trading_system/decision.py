from __future__ import annotations

import json
from textwrap import dedent

from tenacity import retry, stop_after_attempt, wait_fixed

from trading_system.config import TradingConfig
from trading_system.llm import LLMClient, TokenUsageTracker
from trading_system.models import SymbolDebate, TradeDecision
from trading_system.utils import clamp


class DecisionError(RuntimeError):
    pass


class DecisionEngine:
    def __init__(self, config: TradingConfig, logger, token_tracker: TokenUsageTracker | None = None):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config)
        self.token_tracker = token_tracker
        self._last_generation_probability: float | None = None

    def decide(self, debates: list[SymbolDebate]) -> list[TradeDecision]:
        try:
            prompt = self._build_decision_prompt(debates)
            raw, payload = self._generate_valid_decision_json(prompt)
            decisions = self._validate_and_normalize(
                payload,
                generation_probability=self._last_generation_probability,
            )
        except DecisionError as exc:
            self.logger.warning(
                "Batch decision generation failed, falling back to per-symbol decisions: %s",
                exc,
            )
            decisions = self._decide_per_symbol(debates)
        self.logger.info(
            "Decision model produced %s actionable decisions",
            len([item for item in decisions if item.action != "skip"]),
        )
        return decisions

    def _generate_valid_decision_json(self, prompt: str) -> tuple[str, dict]:
        current_prompt = prompt
        last_error = "unknown error"
        raw = ""
        for attempt in range(self.config.ollama_retries):
            raw = self._ollama_generate(
                prompt=current_prompt,
                temperature=self.config.decision_temperature if attempt == 0 else 0.0,
            )
            try:
                payload = self._extract_json(raw)
                if "decisions" not in payload:
                    raise DecisionError("Decision payload missing decisions list.")
                return raw, payload
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                current_prompt = self._build_json_repair_prompt(raw, prompt, last_error)
                self.logger.warning(
                    "Decision JSON validation failed on attempt %s: %s raw=%r",
                    attempt + 1,
                    exc,
                    raw[:200],
                )
        raise DecisionError(f"Unable to obtain valid decision JSON: {last_error}")

    def _build_decision_prompt(self, debates: list[SymbolDebate]) -> str:
        compact_payload = []
        for debate in debates:
            compact_payload.append(
                {
                    "symbol": debate.symbol,
                    "market_summary": {
                        "price_summary": debate.market_data.price_summary,
                        "score_breakdown": debate.market_data.score_breakdown,
                        "premarket": {
                            "latest_price": debate.market_data.premarket.latest_price,
                            "gap_pct": debate.market_data.premarket.gap_pct,
                            "volume": debate.market_data.premarket.volume,
                            "timestamp": debate.market_data.premarket.timestamp,
                        },
                    },
                    "bull_case": {
                        "confidence": debate.bull_case.confidence,
                        "arguments": debate.bull_case.arguments,
                        "risks": debate.bull_case.risks,
                        "key_levels": debate.bull_case.key_levels,
                    },
                    "bear_case": {
                        "confidence": debate.bear_case.confidence,
                        "arguments": debate.bear_case.arguments,
                        "risks": debate.bear_case.risks,
                        "key_levels": debate.bear_case.key_levels,
                    },
                }
            )
        return dedent(
            f"""
            You are the final portfolio decision model.
            Compare bull and bear arguments for each symbol and decide long, short, or skip.
            Output STRICT JSON only.

            Required schema:
            {{
              "decisions": [
                {{
                  "symbol": "AAPL",
                  "action": "long",
                  "confidence": 0.0,
                  "allocation": 0.0
                }}
              ]
            }}

            Rules:
            - action must be one of long, short, skip
            - confidence and allocation must be numbers between 0 and 1
            - keep skip allocations at 0
            - normalize allocations across non-skip trades so the total allocation is <= 1
            - prefer concentrated conviction over spreading tiny positions

            Input:
            {json.dumps(compact_payload, indent=2)}
            """
        ).strip()

    def _build_single_symbol_prompt(self, debate: SymbolDebate) -> str:
        payload = {
            "symbol": debate.symbol,
            "market_summary": {
                "price_summary": debate.market_data.price_summary,
                "score_breakdown": debate.market_data.score_breakdown,
                "premarket": {
                    "latest_price": debate.market_data.premarket.latest_price,
                    "gap_pct": debate.market_data.premarket.gap_pct,
                    "volume": debate.market_data.premarket.volume,
                    "timestamp": debate.market_data.premarket.timestamp,
                },
            },
            "bull_case": {
                "confidence": debate.bull_case.confidence,
                "arguments": debate.bull_case.arguments,
                "risks": debate.bull_case.risks,
                "key_levels": debate.bull_case.key_levels,
            },
            "bear_case": {
                "confidence": debate.bear_case.confidence,
                "arguments": debate.bear_case.arguments,
                "risks": debate.bear_case.risks,
                "key_levels": debate.bear_case.key_levels,
            },
        }
        return dedent(
            f"""
            You are the final portfolio decision model for ONE symbol.
            Compare the bull and bear cases and decide long, short, or skip.
            Output STRICT JSON only.

            Required schema:
            {{
              "symbol": "{debate.symbol}",
              "action": "long",
              "confidence": 0.0
            }}

            Rules:
            - action must be one of long, short, skip
            - confidence must be between 0 and 1
            - do not include any extra keys

            Input:
            {json.dumps(payload, indent=2)}
            """
        ).strip()

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def _ollama_generate(self, prompt: str, temperature: float) -> str:
        response = self.llm_client.generate(
            model=self.config.get_decision_model(),
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=1000,
            json_mode=True,
        )
        self._last_generation_probability = response.usage.average_probability
        if self.token_tracker is not None:
            self.token_tracker.record(stage="decision", usage=response.usage)
        return response.text

    @staticmethod
    def _extract_json(raw: str) -> dict:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1:
                raise DecisionError("No JSON object found in decision model response.")
            return json.loads(raw[start : end + 1])

    def _validate_and_normalize(
        self,
        payload: dict,
        generation_probability: float | None = None,
    ) -> list[TradeDecision]:
        if "decisions" not in payload or not isinstance(payload["decisions"], list):
            raise DecisionError("Decision payload missing decisions list.")

        decisions: list[TradeDecision] = []
        confidence_cap = self._decision_confidence_cap(generation_probability)
        for item in payload["decisions"]:
            action = str(item["action"]).lower()
            if action not in {"long", "short", "skip"}:
                raise DecisionError(f"Invalid action {action}")
            decisions.append(
                TradeDecision(
                    symbol=str(item["symbol"]).upper(),
                    action=action,
                    confidence=clamp(min(float(item["confidence"]), confidence_cap), 0.0, 1.0),
                    allocation=clamp(float(item["allocation"]), 0.0, 1.0),
                )
            )

        active = [item for item in decisions if item.action != "skip" and item.confidence >= self.config.min_confidence]
        total = sum(item.allocation for item in active)
        if total > 1.0 and total > 0:
            for item in decisions:
                if item in active:
                    item.allocation = item.allocation / total
        for item in decisions:
            if item.action == "skip" or item.confidence < self.config.min_confidence:
                item.action = "skip"
                item.allocation = 0.0
        return decisions

    def _decide_per_symbol(self, debates: list[SymbolDebate]) -> list[TradeDecision]:
        decisions: list[TradeDecision] = []
        for debate in debates:
            prompt = self._build_single_symbol_prompt(debate)
            try:
                payload = self._generate_valid_single_decision_json(prompt, debate.symbol)
                action = str(payload["action"]).lower()
                confidence = clamp(float(payload["confidence"]), 0.0, 1.0)
                decisions.append(
                    TradeDecision(
                        symbol=debate.symbol,
                        action=action if confidence >= self.config.min_confidence else "skip",
                        confidence=confidence if confidence >= self.config.min_confidence else 0.0,
                        allocation=0.0,
                    )
                )
            except DecisionError as exc:
                self.logger.warning(
                    "Skipping decision for %s after repeated single-symbol JSON failures: %s",
                    debate.symbol,
                    exc,
                )
                decisions.append(
                    TradeDecision(
                        symbol=debate.symbol,
                        action="skip",
                        confidence=0.0,
                        allocation=0.0,
                    )
                )

        active = [item for item in decisions if item.action != "skip"]
        total_confidence = sum(item.confidence for item in active)
        for item in active:
            item.allocation = item.confidence / total_confidence if total_confidence > 0 else 0.0
        return decisions

    def _generate_valid_single_decision_json(self, prompt: str, symbol: str) -> dict:
        current_prompt = prompt
        last_error = "unknown error"
        for attempt in range(self.config.ollama_retries):
            raw = self._ollama_generate(
                prompt=current_prompt,
                temperature=self.config.decision_temperature if attempt == 0 else 0.0,
            )
            try:
                payload = self._extract_json(raw)
                if payload.get("symbol") != symbol:
                    raise DecisionError(f"Single-symbol decision symbol mismatch for {symbol}")
                action = str(payload.get("action", "")).lower()
                if action not in {"long", "short", "skip"}:
                    raise DecisionError(f"Invalid action {action}")
                payload["confidence"] = min(
                    float(payload["confidence"]),
                    self._decision_confidence_cap(self._last_generation_probability),
                )
                return payload
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                current_prompt = self._build_single_symbol_json_repair_prompt(
                    raw,
                    prompt,
                    symbol,
                    last_error,
                )
                self.logger.warning(
                    "Single-symbol decision validation failed for %s on attempt %s: %s raw=%r",
                    symbol,
                    attempt + 1,
                    exc,
                    raw[:200],
                )
        raise DecisionError(f"Unable to obtain valid single-symbol decision JSON for {symbol}: {last_error}")

    @staticmethod
    def _build_json_repair_prompt(raw: str, original_prompt: str, error: str) -> str:
        return dedent(
            f"""
            The previous output was invalid JSON for the required schema.
            Return ONLY ONE valid JSON object.
            No markdown, no explanation, no preamble, no trailing text.

            Required schema:
            {{
              "decisions": [
                {{
                  "symbol": "AAPL",
                  "action": "long",
                  "confidence": 0.0,
                  "allocation": 0.0
                }}
              ]
            }}

            Validation error:
            {error}

            Original task:
            {original_prompt}

            Invalid output:
            {raw}
            """
        ).strip()

    @staticmethod
    def _decision_confidence_cap(generation_probability: float | None) -> float:
        if generation_probability is None:
            return 1.0
        return clamp(generation_probability, 0.0, 1.0)

    @staticmethod
    def _build_single_symbol_json_repair_prompt(
        raw: str,
        original_prompt: str,
        symbol: str,
        error: str,
    ) -> str:
        return dedent(
            f"""
            The previous output was invalid JSON for the required schema.
            Return ONLY ONE valid JSON object.
            No markdown, no explanation, no preamble, no trailing text.

            Required schema:
            {{
              "symbol": "{symbol}",
              "action": "long",
              "confidence": 0.0
            }}

            Validation error:
            {error}

            Original task:
            {original_prompt}

            Invalid output:
            {raw}
            """
        ).strip()
