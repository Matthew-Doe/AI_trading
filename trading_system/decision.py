from __future__ import annotations

import json
from textwrap import dedent

from tenacity import retry, stop_after_attempt, wait_fixed

from trading_system.confidence_calibration import ConfidenceCalibrator
from trading_system.config import TradingConfig
from trading_system.data import MarketDataService
from trading_system.llm import LLMClient, TokenUsageTracker
from trading_system.models import SymbolDebate, TradeDecision
from trading_system.utils import clamp


class DecisionError(RuntimeError):
    pass


class DecisionEngine:
    def __init__(
        self,
        config: TradingConfig,
        logger,
        token_tracker: TokenUsageTracker | None = None,
        market_data_service: MarketDataService | None = None,
        confidence_calibrator: ConfidenceCalibrator | None = None,
    ):
        self.config = config
        self.logger = logger
        self.llm_client = LLMClient(config)
        self.token_tracker = token_tracker
        self._last_generation_probability: float | None = None
        self.market_data_service = market_data_service or MarketDataService(config, logger)
        self.confidence_calibrator = confidence_calibrator or ConfidenceCalibrator(
            config=config,
            logger=logger,
            market_data_service=self.market_data_service,
        )

    def decide(self, debates: list[SymbolDebate]) -> list[TradeDecision]:
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
                payload = self._normalize_decision_payload(payload)
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
                    "market": {
                        "summary": debate.market_data.price_summary,
                        "indicators": {
                            "rsi": debate.market_data.indicators.rsi14,
                            "sma20": debate.market_data.indicators.sma20,
                            "sma50": debate.market_data.indicators.sma50,
                            "sma200": debate.market_data.indicators.sma200,
                        }
                    },
                    "bull": {"confidence": debate.bull_case.confidence, "args": debate.bull_case.arguments},
                    "bear": {"confidence": debate.bear_case.confidence, "args": debate.bear_case.arguments},
                }
            )
        return dedent(
            f"""
            [OUTPUT_REQUIREMENT]
            Return a JSON object matching the schema below. No explanation. No preamble.

            [SCHEMA]
            {{
              "decisions": [
                {{
                  "symbol": "TICKER",
                  "action": "long|short|skip",
                  "confidence": 0.50-0.99,
                  "allocation": 0.0-0.15,
                  "target_price": float,
                  "invalidation_price": float,
                  "reward_risk_ratio": float,
                  "reasoning_rebuttal": "str"
                }}
              ]
            }}

            [TRADING_RULES]
            1. Super Trend: If SMA20 > SMA50 > SMA200, ignore RSI overbought unless RSI > 92.
            2. R/R Floor: All trades MUST have Reward/Risk >= 2.5. Else SKIP.
            3. Granularity: Use the full range of confidence (0.50-0.99). AVOID '0.63'.
            4. Rebuttal: Explain why the opposing case risk is acceptable for each non-skip trade.

            [INPUT_DATA]
            {json.dumps(compact_payload, indent=2)}
            """
        ).strip()

    def _build_single_symbol_prompt(self, debate: SymbolDebate) -> str:
        payload = {
            "symbol": debate.symbol,
            "market_summary": {
                "price_summary": debate.market_data.price_summary,
                "indicators": {
                    "rsi14": debate.market_data.indicators.rsi14,
                    "sma20": debate.market_data.indicators.sma20,
                    "sma50": debate.market_data.indicators.sma50,
                    "sma200": debate.market_data.indicators.sma200,
                },
                "premarket": {
                    "latest_price": debate.market_data.premarket.latest_price,
                    "gap_pct": debate.market_data.premarket.gap_pct,
                    "volume": debate.market_data.premarket.volume,
                },
            },
            "bull_case": {
                "confidence": debate.bull_case.confidence,
                "arguments": debate.bull_case.arguments,
            },
            "bear_case": {
                "confidence": debate.bear_case.confidence,
                "arguments": debate.bear_case.arguments,
            },
        }
        return dedent(
            f"""
            Return ONLY one valid JSON object. No markdown. No prose. No thoughts.

            Required schema:
            {{
              "symbol": "{debate.symbol}",
              "action": "long|short|skip",
              "confidence": 0.0,
              "target_price": 0.0,
              "invalidation_price": 0.0,
              "reward_risk_ratio": 0.0,
              "reason": "12 words max"
            }}

            Rules:
            - Choose exactly one action: long, short, or skip.
            - Use confidence 0.50-0.99 for long/short, 0.0 for skip.
            - Non-skip trades require reward_risk_ratio >= 2.5.
            - If SMA20 > SMA50 > SMA200, ignore overbought RSI unless RSI > 92.
            - Keep reason short. Do not include analysis text outside JSON.

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

    @staticmethod
    def _normalize_decision_payload(payload: object) -> dict:
        if not isinstance(payload, dict):
            raise DecisionError("Decision payload must be a JSON object.")
        if "decisions" in payload:
            decisions = payload["decisions"]
            if not isinstance(decisions, list):
                raise DecisionError("Decision payload missing decisions list.")
            return payload

        if {"symbol", "action"}.issubset(payload):
            return {"decisions": [payload]}

        raise DecisionError("Decision payload missing decisions list.")

    def _validate_and_normalize(
        self,
        payload: dict,
        generation_probability: float | None = None,
    ) -> list[TradeDecision]:
        payload = self._normalize_decision_payload(payload)

        decisions: list[TradeDecision] = []
        confidence_cap = self._decision_confidence_cap(generation_probability)
        for item in payload["decisions"]:
            if not isinstance(item, dict):
                raise DecisionError("Decision item must be a JSON object.")
            action = str(item["action"]).lower()
            if action not in {"long", "short", "skip"}:
                raise DecisionError(f"Invalid action {action}")
            symbol = str(item["symbol"]).upper()
            raw_confidence = clamp(min(float(item["confidence"]), confidence_cap), 0.0, 1.0)
            calibrated_confidence = self._calibrate_confidence(symbol, action, raw_confidence)
            target_price = self._optional_float(item.get("target_price"))
            invalidation_price = self._optional_float(item.get("invalidation_price"))
            reward_risk_ratio = self._reward_risk_ratio(
                action=action,
                target_price=target_price,
                invalidation_price=invalidation_price,
                explicit_ratio=self._optional_float(item.get("reward_risk_ratio")),
            )
            if (
                action != "skip"
                and self.config.min_reward_risk_ratio > 0
                and reward_risk_ratio is not None
                and reward_risk_ratio < self.config.min_reward_risk_ratio
            ):
                action = "skip"
            decisions.append(
                TradeDecision(
                    symbol=symbol,
                    action=action,
                    confidence=calibrated_confidence,
                    allocation=clamp(float(item.get("allocation", 0.0)), 0.0, 1.0),
                    expected_move_pct=self._optional_float(item.get("expected_move_pct")),
                    target_price=target_price,
                    invalidation_price=invalidation_price,
                    time_horizon=self._optional_text(item.get("time_horizon")),
                    catalyst=self._optional_text(item.get("catalyst")),
                    reward_risk_ratio=reward_risk_ratio,
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

    @staticmethod
    def _optional_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _reward_risk_ratio(
        *,
        action: str,
        target_price: float | None,
        invalidation_price: float | None,
        explicit_ratio: float | None,
    ) -> float | None:
        if explicit_ratio is not None:
            return explicit_ratio
        if target_price is None or invalidation_price is None:
            return None
        if action == "long":
            risk = max(0.0, 1.0 - invalidation_price)
            reward = max(0.0, target_price - 1.0)
        elif action == "short":
            risk = max(0.0, invalidation_price - 1.0)
            reward = max(0.0, 1.0 - target_price)
        else:
            return None
        return reward / risk if risk > 0 else None

    def _decide_per_symbol(self, debates: list[SymbolDebate]) -> list[TradeDecision]:
        decisions: list[TradeDecision] = []
        for debate in debates:
            prompt = self._build_single_symbol_prompt(debate)
            try:
                payload = self._generate_valid_single_decision_json(prompt, debate.symbol)
                action = str(payload["action"]).lower()
                raw_confidence = clamp(float(payload.get("confidence", 0.0)), 0.0, 1.0)
                confidence = self._calibrate_confidence(debate.symbol, action, raw_confidence)
                
                target_price = self._optional_float(payload.get("target_price"))
                invalidation_price = self._optional_float(payload.get("invalidation_price"))
                reward_risk_ratio = self._reward_risk_ratio(
                    action=action,
                    target_price=target_price,
                    invalidation_price=invalidation_price,
                    explicit_ratio=self._optional_float(payload.get("reward_risk_ratio")),
                )

                decisions.append(
                    TradeDecision(
                        symbol=debate.symbol,
                        action=action if confidence >= self.config.min_confidence else "skip",
                        confidence=confidence if confidence >= self.config.min_confidence else 0.0,
                        allocation=0.0,
                        target_price=target_price,
                        invalidation_price=invalidation_price,
                        reward_risk_ratio=reward_risk_ratio,
                        catalyst=self._optional_text(payload.get("reason") or payload.get("reasoning_rebuttal")),
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
            # Default allocation logic if batch failed
            item.allocation = (item.confidence / total_confidence * 0.8) if total_confidence > 0 else 0.0
        return decisions

    def _calibrate_confidence(self, symbol: str, action: str, raw_confidence: float) -> float:
        if action == "skip" and raw_confidence <= 0.0:
            return 0.0
        return clamp(
            self.confidence_calibrator.calibrate(
                symbol=symbol,
                action=action,
                raw_confidence=raw_confidence,
            ),
            0.0,
            1.0,
        )

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
                payload = self._normalize_single_decision_payload(payload, symbol)
                if payload.get("symbol") != symbol:
                    raise DecisionError(f"Single-symbol decision symbol mismatch for {symbol}")
                action = str(payload.get("action", "")).lower()
                if action not in {"long", "short", "skip"}:
                    raise DecisionError(f"Invalid action {action}")
                # For single-symbol, we don't have generation_probability as easily but we can cap it.
                payload["confidence"] = min(
                    float(payload.get("confidence", 0.0)),
                    1.0
                )
                if payload.get("symbol") != symbol:
                    payload["symbol"] = symbol
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

    @classmethod
    def _normalize_single_decision_payload(cls, payload: object, symbol: str) -> dict:
        normalized = cls._normalize_decision_payload(payload)
        decisions = normalized["decisions"]
        symbol_matches = [
            item
            for item in decisions
            if isinstance(item, dict) and str(item.get("symbol", "")).upper() == symbol
        ]
        if len(symbol_matches) == 1:
            return symbol_matches[0]
        if len(decisions) == 1 and isinstance(decisions[0], dict):
            return decisions[0]
        raise DecisionError(f"Single-symbol decision symbol mismatch for {symbol}")

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
