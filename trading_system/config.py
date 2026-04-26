from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal environments
    def load_dotenv() -> bool:
        return False


load_dotenv()


def _parse_symbol_list(value: str, default: tuple[str, ...]) -> tuple[str, ...]:
    items = [item.strip().upper() for item in value.split(",") if item.strip()]
    return tuple(items) if items else default


def _parse_schedule_times(
    value: str | None, fallback_hour: int, fallback_minute: int
) -> tuple[tuple[int, int], ...]:
    if not value or not value.strip():
        return ((fallback_hour, fallback_minute),)

    schedule_times: list[tuple[int, int]] = []
    for item in value.split(","):
        candidate = item.strip()
        if not candidate:
            continue
        try:
            hour_text, minute_text = candidate.split(":", maxsplit=1)
            hour = int(hour_text)
            minute = int(minute_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid scheduled time '{candidate}'. Expected HH:MM entries in SCHEDULED_TIMES_ET."
            ) from exc
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid scheduled time '{candidate}'. Hour/minute out of range.")
        schedule_times.append((hour, minute))

    return tuple(schedule_times) if schedule_times else ((fallback_hour, fallback_minute),)


@dataclass(slots=True)
class TradingConfig:
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_paper_base_url: str = os.getenv(
        "ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets"
    )
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
    llm_debate_model: str = os.getenv("LLM_DEBATE_MODEL", "").strip()
    llm_decision_model: str = os.getenv("LLM_DECISION_MODEL", "").strip()
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_debate_model: str = os.getenv("OLLAMA_DEBATE_MODEL", "qwen3.5:4b")
    ollama_decision_model: str = os.getenv("OLLAMA_DECISION_MODEL", "qwen3.5:9b")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_debate_model: str = os.getenv("OPENAI_DEBATE_MODEL", "").strip()
    openai_decision_model: str = os.getenv("OPENAI_DECISION_MODEL", "").strip()
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
    anthropic_debate_model: str = os.getenv("ANTHROPIC_DEBATE_MODEL", "").strip()
    anthropic_decision_model: str = os.getenv("ANTHROPIC_DECISION_MODEL", "").strip()
    debate_temperature: float = float(os.getenv("OLLAMA_DEBATE_TEMPERATURE", "0.1"))
    decision_temperature: float = float(
        os.getenv("OLLAMA_DECISION_TEMPERATURE", "0.1")
    )
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    ollama_retries: int = int(os.getenv("OLLAMA_RETRIES", "3"))
    api_retries: int = int(os.getenv("API_RETRIES", "3"))
    max_requests_per_minute: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "100"))
    top_universe_size: int = int(os.getenv("TOP_UNIVERSE_SIZE", "200"))
    index_proxy_symbols: tuple[str, ...] = field(
        default_factory=lambda: _parse_symbol_list(
            os.getenv("INDEX_PROXY_SYMBOLS", "SPY,QQQ,DIA,IWM"),
            ("SPY", "QQQ", "DIA", "IWM"),
        )
    )
    candidate_count: int = int(os.getenv("CANDIDATE_COUNT", "20"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.60"))
    confidence_actionable_move_pct: float = float(
        os.getenv("CONFIDENCE_ACTIONABLE_MOVE_PCT", "0.02")
    )
    max_single_trade_pct: float = float(os.getenv("MAX_SINGLE_TRADE_PCT", "0.01"))
    cash_rich_trade_pct: float = float(os.getenv("CASH_RICH_TRADE_PCT", "0.10"))
    cash_rich_available_cash_threshold: float = float(
        os.getenv("CASH_RICH_AVAILABLE_CASH_THRESHOLD", "0.20")
    )
    high_confidence_trade_pct: float = float(os.getenv("HIGH_CONFIDENCE_TRADE_PCT", "0.10"))
    high_confidence_threshold: float = float(os.getenv("HIGH_CONFIDENCE_THRESHOLD", "0.95"))
    pending_order_review_max_gap_pct: float = float(
        os.getenv("PENDING_ORDER_REVIEW_MAX_GAP_PCT", "0.03")
    )
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))
    max_total_exposure: float = float(os.getenv("MAX_TOTAL_EXPOSURE", "0.80"))
    max_daily_loss_pct: float = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.02"))
    max_open_risk_pct: float = float(os.getenv("MAX_OPEN_RISK_PCT", "0.06"))
    max_position_weight: float = float(os.getenv("MAX_POSITION_WEIGHT", "0.10"))
    max_sector_exposure: float = float(os.getenv("MAX_SECTOR_EXPOSURE", "0.30"))
    max_correlation_exposure: float = float(os.getenv("MAX_CORRELATION_EXPOSURE", "0.30"))
    max_symbol_5d_return: float = float(os.getenv("MAX_SYMBOL_5D_RETURN", "0.35"))
    max_symbol_20d_return: float = float(os.getenv("MAX_SYMBOL_20D_RETURN", "0.50"))
    max_symbol_atr_pct: float = float(os.getenv("MAX_SYMBOL_ATR_PCT", "0.20"))
    max_symbol_volatility: float = float(os.getenv("MAX_SYMBOL_VOLATILITY", "1.50"))
    min_symbol_dollar_volume: float = float(os.getenv("MIN_SYMBOL_DOLLAR_VOLUME", "5000000"))
    max_long_rsi: float = float(os.getenv("MAX_LONG_RSI", "90"))
    min_short_rsi: float = float(os.getenv("MIN_SHORT_RSI", "10"))
    order_style: str = os.getenv("ORDER_STYLE", "bracket_limit").strip().lower()
    stop_atr_multiple: float = float(os.getenv("STOP_ATR_MULTIPLE", "1.5"))
    take_profit_r_multiple: float = float(os.getenv("TAKE_PROFIT_R_MULTIPLE", "2.0"))
    entry_limit_buffer_pct: float = float(os.getenv("ENTRY_LIMIT_BUFFER_PCT", "0.002"))
    max_slippage_pct: float = float(os.getenv("MAX_SLIPPAGE_PCT", "0.01"))
    min_reward_risk_ratio: float = float(os.getenv("MIN_REWARD_RISK_RATIO", "1.5"))
    buy_more_threshold: float = float(os.getenv("BUY_MORE_THRESHOLD", "0.05"))
    execute_orders: bool = os.getenv("EXECUTE_ORDERS", "false").lower() == "true"
    allow_shorting: bool = os.getenv("ALLOW_SHORTING", "true").lower() == "true"
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    telegram_approval_timeout_seconds: int = int(
        os.getenv("TELEGRAM_APPROVAL_TIMEOUT_SECONDS", "300")
    )
    telegram_poll_interval_seconds: int = int(
        os.getenv("TELEGRAM_POLL_INTERVAL_SECONDS", "5")
    )
    market_timezone: str = os.getenv("MARKET_TIMEZONE", "America/New_York")
    scheduled_hour: int = int(os.getenv("SCHEDULED_HOUR_ET", "8"))
    scheduled_minute: int = int(os.getenv("SCHEDULED_MINUTE_ET", "30"))
    scheduled_times: tuple[tuple[int, int], ...] = field(
        default_factory=lambda: _parse_schedule_times(
            os.getenv("SCHEDULED_TIMES_ET"),
            int(os.getenv("SCHEDULED_HOUR_ET", "8")),
            int(os.getenv("SCHEDULED_MINUTE_ET", "30")),
        )
    )
    benchmark_symbol: str = os.getenv("BENCHMARK_SYMBOL", "SPY").strip().upper()
    market_close_summary_hour: int = int(os.getenv("MARKET_CLOSE_SUMMARY_HOUR_ET", "20"))
    market_close_summary_minute: int = int(os.getenv("MARKET_CLOSE_SUMMARY_MINUTE_ET", "0"))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs")))
    run_dir: Path = field(default_factory=lambda: Path(os.getenv("RUN_DIR", "runs")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))
    universe_cache_ttl_hours: int = int(os.getenv("UNIVERSE_CACHE_TTL_HOURS", "12"))
    symbol_cache_ttl_hours: int = int(os.getenv("SYMBOL_CACHE_TTL_HOURS", "6"))
    news_limit: int = int(os.getenv("NEWS_LIMIT", "5"))

    def validate_for_live_run(self) -> None:
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise ValueError("Alpaca credentials are required for non-mock runs.")

    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def get_debate_model(self) -> str:
        if self.llm_debate_model:
            return self.llm_debate_model
        if self.llm_provider == "ollama":
            return self.ollama_debate_model
        if self.llm_provider == "openai":
            return self.openai_debate_model
        if self.llm_provider == "anthropic":
            return self.anthropic_debate_model
        raise ValueError(f"Unsupported llm provider: {self.llm_provider}")

    def get_decision_model(self) -> str:
        if self.llm_decision_model:
            return self.llm_decision_model
        if self.llm_provider == "ollama":
            return self.ollama_decision_model
        if self.llm_provider == "openai":
            return self.openai_decision_model
        if self.llm_provider == "anthropic":
            return self.anthropic_decision_model
        raise ValueError(f"Unsupported llm provider: {self.llm_provider}")

    def validate_llm_provider(self) -> None:
        if self.llm_provider not in {"ollama", "openai", "anthropic"}:
            raise ValueError(
                f"Unsupported llm provider: {self.llm_provider}. Expected ollama, openai, or anthropic."
            )
        if not self.get_debate_model():
            raise ValueError(f"No debate model configured for provider {self.llm_provider}.")
        if not self.get_decision_model():
            raise ValueError(f"No decision model configured for provider {self.llm_provider}.")
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic.")
