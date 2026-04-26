from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

import requests

from trading_system.config import TradingConfig
from trading_system.models import OrderPlan


class TelegramError(RuntimeError):
    pass


@dataclass(slots=True)
class TelegramApprovalDecision:
    approved: bool
    response_text: str | None = None
    approval_code: str | None = None


class TelegramNotifier:
    def __init__(self, config: TradingConfig, logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()

    def is_enabled(self) -> bool:
        return self.config.telegram_enabled()

    def send_message(self, text: str, reply_markup: dict[str, Any] | None = None) -> dict[str, Any] | None:
        if not self.is_enabled():
            return None
        try:
            response = self.session.post(
                self._api_url("sendMessage"),
                json={
                    "chat_id": self.config.telegram_chat_id,
                    "text": text,
                    **({"reply_markup": reply_markup} if reply_markup else {}),
                },
                timeout=self.config.request_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Telegram send failed: %s", exc)
            return None
        if not payload.get("ok"):
            self.logger.warning("Telegram API error: %s", payload)
            return None
        result = payload.get("result")
        return result if isinstance(result, dict) else None

    def request_trade_approval(
        self,
        *,
        run_id: str,
        order_plan: OrderPlan,
        standard_notional: float,
        requested_notional: float,
        cash_available: float,
    ) -> TelegramApprovalDecision:
        if not self.is_enabled():
            return TelegramApprovalDecision(approved=False)

        approval_code = secrets.token_hex(3).upper()
        offset = self._latest_update_offset()
        callback_approve = f"APPROVE:{approval_code}"
        callback_deny = f"DENY:{approval_code}"
        message = (
            f"Approval needed for {run_id}\n"
            f"Symbol: {order_plan.symbol}\n"
            f"Side: {order_plan.side}\n"
            f"Confidence: {order_plan.confidence:.2f}\n"
            f"Base cap: ${standard_notional:.2f} ({self.config.max_single_trade_pct:.0%} of capital)\n"
            f"Requested cap: ${requested_notional:.2f} ({self.config.high_confidence_trade_pct:.0%} of cash)\n"
            f"Cash available: ${cash_available:.2f}\n"
            f"Use the buttons below to approve or deny the larger order.\n"
            f"Code: {approval_code}"
        )
        sent_message = self.send_message(
            message,
            reply_markup={
                "inline_keyboard": [
                    [
                        {"text": "Approve", "callback_data": callback_approve},
                        {"text": "Deny", "callback_data": callback_deny},
                    ]
                ]
            },
        )
        message_id = int(sent_message.get("message_id", 0)) if sent_message else 0

        deadline = time.monotonic() + self.config.telegram_approval_timeout_seconds
        next_offset = offset
        while time.monotonic() < deadline:
            updates = self._get_updates(next_offset, timeout=self.config.telegram_poll_interval_seconds)
            for update in updates:
                next_offset = max(next_offset, int(update.get("update_id", 0)) + 1)
                callback_query = update.get("callback_query", {})
                callback_data = str(callback_query.get("data", "")).strip().upper()
                callback_id = str(callback_query.get("id", "")).strip()
                if callback_id and callback_data in {callback_approve, callback_deny}:
                    approved = callback_data == callback_approve
                    self._answer_callback_query(
                        callback_id,
                        "Approval granted." if approved else "Approval denied.",
                    )
                    if message_id:
                        self._edit_message_reply_markup(message_id)
                    return TelegramApprovalDecision(
                        approved=approved,
                        response_text=callback_data,
                        approval_code=approval_code,
                    )
                text = (
                    update.get("message", {}).get("text")
                    or update.get("edited_message", {}).get("text")
                    or ""
                ).strip()
                normalized = " ".join(text.upper().split())
                if normalized == f"APPROVE {approval_code}":
                    if message_id:
                        self._edit_message_reply_markup(message_id)
                    return TelegramApprovalDecision(True, text, approval_code)
                if normalized == f"DENY {approval_code}":
                    if message_id:
                        self._edit_message_reply_markup(message_id)
                    return TelegramApprovalDecision(False, text, approval_code)

        if message_id:
            self._edit_message_reply_markup(message_id)
        self.send_message(
            f"Approval timed out for {run_id} {order_plan.symbol}. Larger {self.config.high_confidence_trade_pct:.0%} cash order was not used."
        )
        return TelegramApprovalDecision(False, approval_code=approval_code)

    def send_trade_summary(self, *, run_id: str, order_plan: OrderPlan, payload: dict[str, Any]) -> None:
        if not self.is_enabled():
            return
        approval_line = ""
        if order_plan.telegram_approval_required:
            approval_line = (
                f"\nApproval override: {'granted' if order_plan.telegram_approval_granted else 'not granted'}"
            )
        message = (
            f"Trade update for {run_id}\n"
            f"{order_plan.symbol} {order_plan.side} qty={order_plan.qty}\n"
            f"Status: {payload.get('status')}\n"
            f"Confidence: {order_plan.confidence:.2f}\n"
            f"Notional: ${order_plan.notional:.2f}\n"
            f"Cap used: {order_plan.max_trade_pct:.0%}\n"
            f"Reason: {self._humanize_trade_reason(order_plan)}{approval_line}"
        )
        self.send_message(message)

    @staticmethod
    def _humanize_trade_reason(order_plan: OrderPlan) -> str:
        reason = order_plan.reason.strip()
        if not reason:
            return "Trade matched the current model signal."
        if "allocation=" in reason and "risk_qty=" in reason and "alloc_qty=" in reason:
            sizing = TelegramNotifier._parse_sizing_reason(reason)
            if not sizing:
                return "Trade matched the current model signal and was sized by risk and exposure limits."
            drivers: list[str] = []
            allocation = sizing.get("allocation")
            risk_qty = sizing.get("risk_qty")
            alloc_qty = sizing.get("alloc_qty")
            max_trade_qty = sizing.get("max_trade_qty")
            qty = order_plan.qty

            if allocation is not None:
                drivers.append(f"target allocation {allocation:.0%}")
            limiting_reasons: list[str] = []
            if max_trade_qty is not None and qty == int(max_trade_qty):
                limiting_reasons.append("max trade cap")
            if alloc_qty is not None and qty == int(alloc_qty):
                limiting_reasons.append("allocation size")
            if risk_qty is not None and qty == int(risk_qty):
                limiting_reasons.append("risk limit")

            if limiting_reasons:
                joined = ", ".join(limiting_reasons)
                drivers.append(f"sized by the tightest limit: {joined}")
            else:
                drivers.append("sized conservatively from allocation and risk limits")

            return "Trade matched the current model signal with " + "; ".join(drivers) + "."
        return reason

    @staticmethod
    def _parse_sizing_reason(reason: str) -> dict[str, float] | None:
        parsed: dict[str, float] = {}
        for part in reason.split(","):
            key, separator, value = part.strip().partition("=")
            if separator != "=" or not key or not value:
                continue
            try:
                parsed[key] = float(value)
            except ValueError:
                continue
        return parsed or None

    def _latest_update_offset(self) -> int:
        updates = self._get_updates(0, timeout=0)
        if not updates:
            return 0
        return max(int(update.get("update_id", 0)) for update in updates) + 1

    def _get_updates(self, offset: int, timeout: int) -> list[dict[str, Any]]:
        response = self.session.get(
            self._api_url("getUpdates"),
            params={
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": ["message", "edited_message", "callback_query"],
            },
            timeout=max(timeout + 5, self.config.request_timeout_seconds),
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise TelegramError(f"Telegram API error: {payload}")
        return [item for item in payload.get("result", []) if isinstance(item, dict)]

    def _api_url(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.config.telegram_bot_token}/{method}"

    def _answer_callback_query(self, callback_query_id: str, text: str) -> None:
        response = self.session.post(
            self._api_url("answerCallbackQuery"),
            json={"callback_query_id": callback_query_id, "text": text},
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()

    def _edit_message_reply_markup(self, message_id: int) -> None:
        response = self.session.post(
            self._api_url("editMessageReplyMarkup"),
            json={
                "chat_id": self.config.telegram_chat_id,
                "message_id": message_id,
                "reply_markup": {"inline_keyboard": []},
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
