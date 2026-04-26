from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from trading_system.config import TradingConfig
from trading_system.models import IndicatorSnapshot, PremarketSnapshot, SymbolMarketData
from trading_system.utils import RateLimiter, ensure_dir, read_json, safe_float, utc_timestamp, write_json


class DataIngestionError(RuntimeError):
    pass


class MarketDataService:
    def __init__(self, config: TradingConfig, logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()
        self.cache_dir = ensure_dir(self.config.cache_dir)
        self.universe_cache_path = self.cache_dir / "universe_top500.json"
        self.symbol_cache_dir = ensure_dir(self.cache_dir / "symbols")
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute, 60)
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                )
            }
        )
        self.alpaca_data_headers = {
            "APCA-API-KEY-ID": self.config.alpaca_api_key,
            "APCA-API-SECRET-KEY": self.config.alpaca_secret_key,
        }

    @staticmethod
    def _to_alpaca_symbol(symbol: str) -> str:
        return symbol.replace("-", ".")

    def is_market_day(self, at_time: datetime | None = None) -> bool:
        market_time = at_time or datetime.now(ZoneInfo(self.config.market_timezone))
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=market_time.date(), end_date=market_time.date()
        )
        return not schedule.empty

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def fetch_top_us_companies_by_market_cap(self, limit: int | None = None) -> list[dict]:
        target = limit or self.config.top_universe_size
        cached = self._read_json_cache(
            self.universe_cache_path, self.config.universe_cache_ttl_hours
        )
        if cached and len(cached) >= target:
            self.logger.info("Loaded universe from cache: %s symbols", len(cached))
            return cached[:target]

        companies: list[dict] = []

        for page in range(1, math.ceil(target / 100) + 1):
            url = (
                "https://companiesmarketcap.com/usa/largest-companies-in-the-usa-by-market-cap/"
                f"?page={page}"
            )
            self.rate_limiter.acquire()
            response = self.session.get(url, timeout=self.config.request_timeout_seconds)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.select("table tbody tr")

            if not rows:
                raise DataIngestionError("Unable to parse market cap universe source.")

            for row in rows:
                name_link = row.select_one("div.company-name a")
                company_name = row.select_one("div.company-name")
                company_code = row.select_one("div.company-code")
                numeric_cells = row.select("td.td-right[data-sort]")
                if not company_name or not company_code or len(numeric_cells) < 2:
                    continue
                symbol = company_code.get_text(" ", strip=True).upper()
                market_cap = self._parse_market_cap(numeric_cells[1].get_text(" ", strip=True))
                companies.append(
                    {
                        "symbol": symbol.replace(".", "-"),
                        "name": company_name.text.strip(),
                        "market_cap": market_cap,
                    }
                )
                if len(companies) >= target:
                    return companies[:target]

        if len(companies) < target:
            if cached:
                self.logger.warning(
                    "Universe refresh incomplete (%s). Falling back to stale cache with %s symbols.",
                    len(companies),
                    len(cached),
                )
                return cached[:target]
            raise DataIngestionError(
                f"Fetched only {len(companies)} companies, expected at least {target}."
            )

        companies = companies[:target]
        write_json(self.universe_cache_path, {"saved_at": utc_timestamp(), "payload": companies})
        return companies

    def build_universe(self, include_news: bool = False, as_of_date: datetime | None = None) -> list[SymbolMarketData]:
        companies = self._build_symbol_universe()
        results: list[SymbolMarketData] = []
        failures: list[str] = []
        self.logger.info("Building universe for %s symbols (as_of=%s)", len(companies), as_of_date)

        for company in companies:
            symbol = company["symbol"]
            try:
                symbol_data = self.fetch_symbol_market_data(
                    symbol=symbol,
                    market_cap=company["market_cap"],
                    include_news=include_news,
                    as_of_date=as_of_date,
                )
                results.append(symbol_data)
            except Exception as exc:  # noqa: BLE001
                failures.append(symbol)
                if as_of_date is None:
                    self.logger.warning("Skipping %s during ingest: %s", symbol, exc)

        if len(results) < min(50, self.config.top_universe_size // 2):
            raise DataIngestionError(
                f"Too few symbols loaded successfully ({len(results)}). Aborting run."
            )

        self.logger.info(
            "Universe build complete. success=%s failed=%s", len(results), len(failures)
        )
        return results

    def _build_symbol_universe(self) -> list[dict]:
        try:
            companies = self.fetch_top_us_companies_by_market_cap()
        except Exception as exc:  # noqa: BLE001
            companies = self._build_symbol_universe_from_cache()
            if not companies:
                raise
            self.logger.warning(
                "Falling back to cached symbol universe after remote universe fetch failure: %s",
                exc,
            )
        seen_symbols = {item["symbol"] for item in companies}
        for symbol in self.config.index_proxy_symbols:
            normalized = symbol.replace(".", "-").upper()
            if normalized in seen_symbols:
                continue
            companies.append(
                {
                    "symbol": normalized,
                    "name": f"{normalized} index proxy",
                    "market_cap": None,
                }
            )
            seen_symbols.add(normalized)
        return companies

    def _build_symbol_universe_from_cache(self) -> list[dict]:
        companies: list[dict] = []
        for path in sorted(self.symbol_cache_dir.glob("*.json")):
            payload = read_json(path).get("payload", {})
            symbol = str(payload.get("symbol", "")).upper()
            if not symbol:
                continue
            companies.append(
                {
                    "symbol": symbol,
                    "name": f"{symbol} cached snapshot",
                    "market_cap": payload.get("market_cap"),
                }
            )

        companies.sort(
            key=lambda item: (
                item["market_cap"] is None,
                -(item["market_cap"] or 0.0),
                item["symbol"],
            )
        )
        return companies[: self.config.top_universe_size]

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2), reraise=True)
    def fetch_symbol_market_data(
        self, 
        symbol: str, 
        market_cap: float | None, 
        include_news: bool = False,
        as_of_date: datetime | None = None,
    ) -> SymbolMarketData:
        # Use date-specific cache key for backtesting
        cache_key = symbol if as_of_date is None else f"{symbol}_{as_of_date.date().isoformat()}"
        cached = self._read_symbol_cache(cache_key)
        daily: pd.DataFrame
        premarket: PremarketSnapshot

        if cached:
            return self.apply_data_quality(self._symbol_market_data_from_cache(cached))

        try:
            daily = self._fetch_daily_bars(symbol, as_of_date=as_of_date)
            # Index -1 is as_of_date, index -2 is the previous day close
            last_close_ref = safe_float(daily["Close"].iloc[-2]) if len(daily) >= 2 else 0.0
            premarket = self._fetch_premarket_snapshot(symbol, last_close_ref, as_of_date=as_of_date)
        except Exception:
            raise

        # We compute indicators on data UP TO BUT NOT INCLUDING the as_of_date 
        history_for_indicators = daily.iloc[:-1] if as_of_date is not None else daily
        indicators = self._compute_indicators(history_for_indicators)
        
        close = safe_float(history_for_indicators["Close"].iloc[-1])
        volume = safe_float(history_for_indicators["Volume"].iloc[-1])
        high_20d = safe_float(history_for_indicators["High"].tail(20).max())
        low_20d = safe_float(history_for_indicators["Low"].tail(20).min())

        news_headlines = self._fetch_news_headlines(symbol) if include_news and as_of_date is None else []
        price_summary, raw_metrics = self._build_price_summary(symbol, history_for_indicators, premarket, indicators)

        symbol_data = SymbolMarketData(
            symbol=symbol,
            market_cap=market_cap,
            close=close,
            high_20d=high_20d,
            low_20d=low_20d,
            volume=volume,
            indicators=indicators,
            premarket=premarket,
            price_summary=price_summary,
            news_headlines=news_headlines,
            raw_metrics=raw_metrics,
        )
        symbol_data = self.apply_data_quality(symbol_data)
        self._write_symbol_cache(symbol_data, cache_key=cache_key)
        return symbol_data

    def apply_data_quality(self, symbol_data: SymbolMarketData) -> SymbolMarketData:
        flags = self._data_quality_flags(symbol_data)
        return replace(symbol_data, data_quality_flags=flags, is_tradeable=not flags)

    def _data_quality_flags(self, symbol_data: SymbolMarketData) -> list[str]:
        flags: list[str] = []
        metrics = symbol_data.raw_metrics
        indicators = symbol_data.indicators
        close = symbol_data.close
        atr_pct = indicators.atr14 / close if close > 0 else 0.0
        dollar_volume = close * symbol_data.volume

        if abs(metrics.get("price_change_20d", 0.0)) > self.config.max_symbol_20d_return:
            flags.append("extreme_20d_return")
        if abs(metrics.get("price_change_5d", 0.0)) > self.config.max_symbol_5d_return:
            flags.append("extreme_5d_return")
        if atr_pct > self.config.max_symbol_atr_pct:
            flags.append("extreme_atr_pct")
        if indicators.volatility20 > self.config.max_symbol_volatility:
            flags.append("extreme_volatility")
        if dollar_volume < self.config.min_symbol_dollar_volume:
            flags.append("insufficient_dollar_volume")
        if indicators.sma200 > 0 and close > 0:
            sma200_ratio = max(indicators.sma200, close) / max(min(indicators.sma200, close), 1e-9)
            if sma200_ratio > 5.0:
                flags.append("sma200_discontinuity")
        if indicators.rsi14 <= 0.5 or indicators.rsi14 >= 99.5:
            flags.append("rsi_extreme")
        if (
            symbol_data.premarket.gap_pct is not None
            and abs(symbol_data.premarket.gap_pct) > 0.20
            and safe_float(symbol_data.premarket.volume) <= 0
        ):
            flags.append("unsupported_premarket_gap")
        return flags

    def _fetch_daily_bars(self, symbol: str, as_of_date: datetime | None = None) -> pd.DataFrame:
        end = as_of_date or datetime.now(UTC)
        start = end - timedelta(days=365)
        
        if self.config.alpaca_api_key and self.config.alpaca_secret_key:
            try:
                daily = self._fetch_daily_bars_alpaca(symbol, start=start, end=end)
                if not daily.empty:
                    return daily
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Alpaca daily bars failed for %s: %s", symbol, exc)

        daily = self._yf_download(
            tickers=symbol,
            start=start.strftime("%Y-%m-%d"),
            end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )
        if daily.empty:
            raise DataIngestionError(f"No daily data returned for {symbol}")

        if isinstance(daily.columns, pd.MultiIndex):
            daily.columns = daily.columns.get_level_values(0)

        daily = daily.dropna().tail(220)
        return daily

    def _fetch_daily_bars_alpaca(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        alpaca_symbol = self._to_alpaca_symbol(symbol)
        self.rate_limiter.acquire()
        response = self.session.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            headers=self.alpaca_data_headers,
            params={
                "symbols": alpaca_symbol,
                "timeframe": "1Day",
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "limit": 500,
                "adjustment": "raw",
                "feed": "iex",
                "sort": "asc",
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        bars = payload.get("bars", {}).get(alpaca_symbol, [])
        if not bars:
            return pd.DataFrame()

        frame = pd.DataFrame(bars)
        frame = frame.rename(
            columns={
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "t": "Timestamp",
            }
        )
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
        frame = frame.set_index("Timestamp")[["Open", "High", "Low", "Close", "Volume"]]
        return frame

    def _fetch_premarket_snapshot(
        self, symbol: str, last_close: float, as_of_date: datetime | None = None
    ) -> PremarketSnapshot:
        if as_of_date is not None:
            # Re-fetch with as_of_date to get the 'day T' open
            daily = self._fetch_daily_bars(symbol, as_of_date=as_of_date)
            if daily.empty or len(daily) < 2:
                return PremarketSnapshot(None, None, None, None)
            
            day_t = daily.iloc[-1]
            day_prev = daily.iloc[-2]
            
            open_price = safe_float(day_t["Open"])
            prev_close = safe_float(day_prev["Close"])
            gap_pct = ((open_price - prev_close) / prev_close) if prev_close else 0.0
            
            return PremarketSnapshot(
                latest_price=open_price,
                gap_pct=gap_pct,
                volume=safe_float(day_t["Volume"]) * 0.05, # Simulated low premarket volume
                timestamp=as_of_date.isoformat(),
            )

        if self.config.alpaca_api_key and self.config.alpaca_secret_key:
            try:
                premarket = self._fetch_premarket_snapshot_alpaca(symbol, last_close)
                return premarket
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Alpaca premarket snapshot failed for %s: %s", symbol, exc)

        intraday = self._yf_download(
            tickers=symbol,
            period="2d",
            interval="5m",
            progress=False,
            auto_adjust=False,
            prepost=True,
            threads=False,
        )
        if intraday.empty:
            return PremarketSnapshot(None, None, None, None)

        if isinstance(intraday.columns, pd.MultiIndex):
            intraday.columns = intraday.columns.get_level_values(0)

        intraday = intraday.dropna()
        if intraday.empty:
            return PremarketSnapshot(None, None, None, None)

        ny_tz = ZoneInfo(self.config.market_timezone)
        if intraday.index.tz is None:
            intraday.index = intraday.index.tz_localize("UTC").tz_convert(ny_tz)
        else:
            intraday.index = intraday.index.tz_convert(ny_tz)

        premarket = intraday[
            (intraday.index.time >= datetime.strptime("04:00", "%H:%M").time())
            & (intraday.index.time < datetime.strptime("09:30", "%H:%M").time())
        ]

        if premarket.empty:
            return PremarketSnapshot(None, None, None, None)

        latest = premarket.iloc[-1]
        latest_price = safe_float(latest["Close"])
        gap_pct = ((latest_price - last_close) / last_close) if last_close else None
        return PremarketSnapshot(
            latest_price=latest_price,
            gap_pct=gap_pct,
            volume=safe_float(premarket["Volume"].sum()),
            timestamp=premarket.index[-1].isoformat(),
        )

    def _fetch_premarket_snapshot_alpaca(self, symbol: str, last_close: float) -> PremarketSnapshot:
        alpaca_symbol = self._to_alpaca_symbol(symbol)
        now = datetime.now(UTC)
        start = now - timedelta(days=2)
        self.rate_limiter.acquire()
        response = self.session.get(
            "https://data.alpaca.markets/v2/stocks/bars",
            headers=self.alpaca_data_headers,
            params={
                "symbols": alpaca_symbol,
                "timeframe": "5Min",
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": now.isoformat().replace("+00:00", "Z"),
                "limit": 1000,
                "adjustment": "raw",
                "feed": "iex",
                "sort": "asc",
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        bars = payload.get("bars", {}).get(alpaca_symbol, [])
        if not bars:
            return PremarketSnapshot(None, None, None, None)

        frame = pd.DataFrame(bars)
        frame = frame.rename(columns={"c": "Close", "v": "Volume", "t": "Timestamp"})
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], utc=True)
        frame = frame.set_index("Timestamp")[["Close", "Volume"]]

        ny_tz = ZoneInfo(self.config.market_timezone)
        frame.index = frame.index.tz_convert(ny_tz)
        premarket = frame[
            (frame.index.time >= datetime.strptime("04:00", "%H:%M").time())
            & (frame.index.time < datetime.strptime("09:30", "%H:%M").time())
        ]
        if premarket.empty:
            return PremarketSnapshot(None, None, None, None)

        latest = premarket.iloc[-1]
        latest_price = safe_float(latest["Close"])
        gap_pct = ((latest_price - last_close) / last_close) if last_close else None
        return PremarketSnapshot(
            latest_price=latest_price,
            gap_pct=gap_pct,
            volume=safe_float(premarket["Volume"].sum()),
            timestamp=premarket.index[-1].isoformat(),
        )

    def _compute_indicators(self, daily: pd.DataFrame) -> IndicatorSnapshot:
        high = daily["High"]
        low = daily["Low"]
        close = daily["Close"]
        volume = daily["Volume"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr14 = safe_float(tr.rolling(14).mean().iloc[-1])

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
        rs = gain / loss
        rsi14 = safe_float((100 - (100 / (1 + rs))).iloc[-1], default=50.0)

        returns = close.pct_change()
        volatility20 = safe_float((returns.rolling(20).std() * np.sqrt(252)).iloc[-1])

        return IndicatorSnapshot(
            atr14=atr14,
            rsi14=rsi14,
            sma20=safe_float(close.rolling(20).mean().iloc[-1]),
            sma50=safe_float(close.rolling(50).mean().iloc[-1]),
            sma200=safe_float(close.rolling(200).mean().iloc[-1]),
            volatility20=volatility20,
            avg_volume20=safe_float(volume.rolling(20).mean().iloc[-1]),
        )

    def _build_price_summary(
        self,
        symbol: str,
        daily: pd.DataFrame,
        premarket: PremarketSnapshot,
        indicators: IndicatorSnapshot,
    ) -> tuple[str, dict[str, float]]:
        recent = daily.tail(30).copy()
        close = recent["Close"]
        volume = recent["Volume"]
        returns = close.pct_change().dropna()

        price_change_5d = safe_float((close.iloc[-1] / close.iloc[-6]) - 1) if len(close) > 5 else 0.0
        price_change_20d = safe_float((close.iloc[-1] / close.iloc[0]) - 1)
        volume_ratio = safe_float(volume.iloc[-1] / indicators.avg_volume20, default=1.0)
        range_position = safe_float(
            (close.iloc[-1] - recent["Low"].min()) /
            max(recent["High"].max() - recent["Low"].min(), 1e-9)
        )
        summary = (
            f"{symbol} last close {close.iloc[-1]:.2f}; 5d return {price_change_5d:.2%}; "
            f"20d return {price_change_20d:.2%}; current volume vs 20d average {volume_ratio:.2f}x; "
            f"20d range position {range_position:.2f}; ATR14 {indicators.atr14:.2f}; "
            f"RSI14 {indicators.rsi14:.1f}; SMA20/50/200 "
            f"{indicators.sma20:.2f}/{indicators.sma50:.2f}/{indicators.sma200:.2f}; "
            f"20d annualized volatility {indicators.volatility20:.2%}; "
            f"premarket gap {safe_float(premarket.gap_pct):.2%}."
        )
        return summary, {
            "price_change_5d": price_change_5d,
            "price_change_20d": price_change_20d,
            "volume_ratio": volume_ratio,
            "range_position": range_position,
            "premarket_gap_pct": safe_float(premarket.gap_pct),
            "recent_volatility": safe_float(returns.std() * np.sqrt(252)),
        }

    def _fetch_news_headlines(self, symbol: str) -> list[str]:
        try:
            self.rate_limiter.acquire()
            ticker = yf.Ticker(symbol)
            news_items = getattr(ticker, "news", []) or []
            headlines = [
                item.get("title", "").strip()
                for item in news_items[: self.config.news_limit]
                if item.get("title")
            ]
            return headlines
        except Exception:  # noqa: BLE001
            return []

    def fetch_close_to_close_return(self, symbol: str) -> tuple[float, float, float]:
        daily = self._fetch_daily_bars(symbol)
        closes = daily["Close"].dropna().tail(2)
        if len(closes) < 2:
            raise DataIngestionError(f"Insufficient close data returned for {symbol}")
        previous_close = safe_float(closes.iloc[-2])
        latest_close = safe_float(closes.iloc[-1])
        close_to_close_return = ((latest_close - previous_close) / previous_close) if previous_close else 0.0
        return previous_close, latest_close, close_to_close_return

    def fetch_forward_close_window(
        self,
        symbol: str,
        *,
        as_of: datetime,
        trading_days_ahead: int = 3,
    ) -> tuple[float, float, str]:
        try:
            daily = self._fetch_daily_bars(
                symbol,
                as_of_date=as_of + timedelta(days=trading_days_ahead + 7),
            )
        except TypeError:
            daily = self._fetch_daily_bars(symbol)
        closes = daily["Close"].dropna().copy()
        if closes.empty:
            raise DataIngestionError(f"No close history returned for {symbol}")

        market_tz = ZoneInfo(self.config.market_timezone)
        if closes.index.tz is None:
            closes.index = closes.index.tz_localize("UTC").tz_convert(market_tz)
        else:
            closes.index = closes.index.tz_convert(market_tz)

        closes = closes.sort_index()
        as_of_local = as_of.astimezone(market_tz)
        known = closes.loc[:as_of_local]
        if known.empty:
            raise DataIngestionError(
                f"No known close available for {symbol} at {as_of_local.isoformat()}"
            )

        forward_position = len(known) - 1 + trading_days_ahead
        if forward_position >= len(closes):
            raise DataIngestionError(
                f"Insufficient forward close history for {symbol} at {as_of_local.date()}"
            )

        reference_close = safe_float(known.iloc[-1])
        forward_close = safe_float(closes.iloc[forward_position])
        forward_as_of = closes.index[forward_position].isoformat()
        return reference_close, forward_close, forward_as_of

    def _yf_download(self, **kwargs) -> pd.DataFrame:
        self.rate_limiter.acquire()
        return yf.download(**kwargs)

    def _read_json_cache(self, path: Path, ttl_hours: int) -> list[dict] | None:
        if not path.exists():
            return None
        payload = read_json(path)
        saved_at = payload.get("saved_at")
        data = payload.get("payload")
        if not saved_at or not isinstance(data, list):
            return None
        saved_at_dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
        if datetime.now(UTC) - saved_at_dt > timedelta(hours=ttl_hours):
            return None
        return data

    def _read_symbol_cache(self, symbol: str) -> dict | None:
        path = self.symbol_cache_dir / f"{symbol}.json"
        if not path.exists():
            return None
        payload = read_json(path)
        saved_at = payload.get("saved_at")
        if not saved_at:
            return None
        saved_at_dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
        if datetime.now(UTC) - saved_at_dt > timedelta(hours=self.config.symbol_cache_ttl_hours):
            # For backtesting, we allow stale cache if it matches the as_of_date
            return payload
        return payload

    def _write_symbol_cache(self, symbol_data: SymbolMarketData, cache_key: str | None = None) -> None:
        key = cache_key or symbol_data.symbol
        write_json(
            self.symbol_cache_dir / f"{key}.json",
            {"saved_at": utc_timestamp(), "payload": symbol_data},
        )

    @staticmethod
    def _symbol_market_data_from_cache(payload: dict) -> SymbolMarketData:
        item = payload["payload"]
        return SymbolMarketData(
            symbol=item["symbol"],
            market_cap=item.get("market_cap"),
            close=item["close"],
            high_20d=item["high_20d"],
            low_20d=item["low_20d"],
            volume=item["volume"],
            indicators=IndicatorSnapshot(**item["indicators"]),
            premarket=PremarketSnapshot(**item["premarket"]),
            price_summary=item["price_summary"],
            news_headlines=item.get("news_headlines", []),
            score_breakdown=item.get("score_breakdown", {}),
            raw_metrics=item.get("raw_metrics", {}),
            data_quality_flags=item.get("data_quality_flags", []),
            is_tradeable=item.get("is_tradeable", True),
        )

    @staticmethod
    def _parse_market_cap(text: str) -> float | None:
        cleaned = text.replace("$", "").replace(",", "").strip().upper()
        if not cleaned:
            return None
        multipliers = {"T": 1_000_000_000_000, "B": 1_000_000_000, "M": 1_000_000}
        suffix = cleaned[-1]
        if suffix in multipliers:
            return safe_float(cleaned[:-1]) * multipliers[suffix]
        return safe_float(cleaned, default=0.0) or None
