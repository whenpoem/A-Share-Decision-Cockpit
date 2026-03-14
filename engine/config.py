from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMProviderConfig:
    api_base: str
    api_key: str
    model_name: str
    timeout_seconds: float = 25.0
    temperature: float = 0.1
    max_tokens: int = 1200


@dataclass
class LiveBrokerConfig:
    provider: str
    enabled: bool = False
    account_id: str = ""
    terminal_path: str = ""
    terminal_user: str = ""
    terminal_password: str = ""


@dataclass
class Settings:
    project_root: Path
    storage_root: Path
    market_storage: Path
    text_storage: Path
    disclosure_storage: Path
    disclosure_pdf_storage: Path
    disclosure_text_storage: Path
    artifact_storage: Path
    db_path: Path
    primary_provider: LLMProviderConfig
    fallback_provider: LLMProviderConfig
    live_broker: LiveBrokerConfig
    system_db_path: Path
    system_storage_root: Path
    current_mode: str = "paper"
    market_provider: str = "baostock"
    text_provider: str = "akshare"
    text_lookback_days: int = 30
    max_news_per_symbol: int = 6
    max_announcements_per_symbol: int = 6
    max_announcement_body_chars: int = 4000
    text_fallback_to_derived: bool = True
    market_universe_size: int = 60
    candidate_pool_size: int = 12
    avoid_pool_size: int = 6
    max_events_per_symbol: int = 8
    max_position_weight: float = 0.10
    max_sector_weight: float = 0.25
    max_gross_exposure: float = 0.80
    max_daily_turnover: float = 0.35
    stop_loss_pct: float = 0.08
    trailing_stop_pct: float = 0.06
    take_profit_pct: float = 0.18
    time_stop_days: int = 20
    daily_loss_limit: float = 0.03
    portfolio_drawdown_limit: float = 0.08
    fee_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_bps: float = 5.0
    default_capital: float = 1_000_000.0
    fallback_to_sample_market: bool = True
    blacklist_symbols: list[str] = field(default_factory=list)
    default_watchlist: list[str] = field(
        default_factory=lambda: [
            "000001",
            "000333",
            "000651",
            "000858",
            "002415",
            "002594",
            "300750",
            "600036",
            "600519",
            "601318",
            "601398",
            "601899",
        ]
    )

    @classmethod
    def load(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        system_storage_root = Path(os.getenv("ASHARE_STORAGE_ROOT", root / "storage")).resolve()
        storage_root = system_storage_root
        market_storage = storage_root / "market"
        text_storage = storage_root / "text"
        disclosure_storage = text_storage / "disclosures"
        disclosure_pdf_storage = disclosure_storage / "pdf"
        disclosure_text_storage = disclosure_storage / "text"
        artifact_storage = storage_root / "artifacts"
        for path in (
            storage_root,
            market_storage,
            text_storage,
            disclosure_storage,
            disclosure_pdf_storage,
            disclosure_text_storage,
            artifact_storage,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=root,
            storage_root=storage_root,
            system_storage_root=system_storage_root,
            market_storage=market_storage,
            text_storage=text_storage,
            disclosure_storage=disclosure_storage,
            disclosure_pdf_storage=disclosure_pdf_storage,
            disclosure_text_storage=disclosure_text_storage,
            artifact_storage=artifact_storage,
            db_path=Path(os.getenv("ASHARE_DB_PATH", storage_root / "state.db")).resolve(),
            system_db_path=Path(
                os.getenv("ASHARE_SYSTEM_DB_PATH", system_storage_root / "state.db")
            ).resolve(),
            primary_provider=LLMProviderConfig(
                api_base=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                timeout_seconds=float(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "25")),
                temperature=float(os.getenv("DEEPSEEK_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("DEEPSEEK_MAX_TOKENS", "1200")),
            ),
            fallback_provider=LLMProviderConfig(
                api_base=os.getenv(
                    "QWEN_API_BASE",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
                api_key=os.getenv("QWEN_API_KEY", ""),
                model_name=os.getenv("QWEN_MODEL", "qwen-plus"),
                timeout_seconds=float(os.getenv("QWEN_TIMEOUT_SECONDS", "25")),
                temperature=float(os.getenv("QWEN_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "1200")),
            ),
            live_broker=LiveBrokerConfig(
                provider=os.getenv("ASHARE_LIVE_BROKER", "qmt_ready"),
                enabled=_bool_env("ASHARE_LIVE_ENABLED", False),
                account_id=os.getenv("ASHARE_LIVE_ACCOUNT_ID", ""),
                terminal_path=os.getenv("ASHARE_QMT_TERMINAL_PATH", ""),
                terminal_user=os.getenv("ASHARE_QMT_USER", ""),
                terminal_password=os.getenv("ASHARE_QMT_PASSWORD", ""),
            ),
            market_provider=os.getenv("ASHARE_MARKET_PROVIDER", "baostock"),
            text_provider=os.getenv("ASHARE_TEXT_PROVIDER", "akshare"),
            text_lookback_days=int(os.getenv("ASHARE_TEXT_LOOKBACK_DAYS", "30")),
            max_news_per_symbol=int(os.getenv("ASHARE_MAX_NEWS_PER_SYMBOL", "6")),
            max_announcements_per_symbol=int(
                os.getenv("ASHARE_MAX_ANNOUNCEMENTS_PER_SYMBOL", "6")
            ),
            max_announcement_body_chars=int(
                os.getenv("ASHARE_MAX_ANNOUNCEMENT_BODY_CHARS", "4000")
            ),
            text_fallback_to_derived=_bool_env("ASHARE_TEXT_FALLBACK_TO_DERIVED", True),
            market_universe_size=int(os.getenv("ASHARE_MARKET_UNIVERSE_SIZE", "60")),
            candidate_pool_size=int(os.getenv("ASHARE_CANDIDATE_POOL_SIZE", "12")),
            avoid_pool_size=int(os.getenv("ASHARE_AVOID_POOL_SIZE", "6")),
            max_events_per_symbol=int(os.getenv("ASHARE_MAX_EVENTS_PER_SYMBOL", "8")),
            max_position_weight=float(os.getenv("ASHARE_MAX_POSITION_WEIGHT", "0.10")),
            max_sector_weight=float(os.getenv("ASHARE_MAX_SECTOR_WEIGHT", "0.25")),
            max_gross_exposure=float(os.getenv("ASHARE_MAX_GROSS_EXPOSURE", "0.80")),
            max_daily_turnover=float(os.getenv("ASHARE_MAX_DAILY_TURNOVER", "0.35")),
            stop_loss_pct=float(os.getenv("ASHARE_STOP_LOSS_PCT", "0.08")),
            trailing_stop_pct=float(os.getenv("ASHARE_TRAILING_STOP_PCT", "0.06")),
            take_profit_pct=float(os.getenv("ASHARE_TAKE_PROFIT_PCT", "0.18")),
            time_stop_days=int(os.getenv("ASHARE_TIME_STOP_DAYS", "20")),
            daily_loss_limit=float(os.getenv("ASHARE_DAILY_LOSS_LIMIT", "0.03")),
            portfolio_drawdown_limit=float(
                os.getenv("ASHARE_PORTFOLIO_DRAWDOWN_LIMIT", "0.08")
            ),
            fee_rate=float(os.getenv("ASHARE_FEE_RATE", "0.0003")),
            stamp_tax_rate=float(os.getenv("ASHARE_STAMP_TAX_RATE", "0.001")),
            slippage_bps=float(os.getenv("ASHARE_SLIPPAGE_BPS", "5")),
            default_capital=float(os.getenv("ASHARE_DEFAULT_CAPITAL", "1000000")),
            fallback_to_sample_market=_bool_env("ASHARE_FALLBACK_TO_SAMPLE_MARKET", True),
            blacklist_symbols=[
                symbol.strip().zfill(6)
                for symbol in os.getenv("ASHARE_BLACKLIST_SYMBOLS", "").split(",")
                if symbol.strip()
            ],
            default_watchlist=[
                symbol.strip().zfill(6)
                for symbol in os.getenv(
                    "ASHARE_DEFAULT_WATCHLIST",
                    "000001,000333,000651,000858,002415,002594,300750,600036,600519,601318,601398,601899",
                ).split(",")
                if symbol.strip()
            ],
        )

    def for_mode(self, mode: str) -> "Settings":
        mode_root = (self.system_storage_root / mode).resolve()
        market_storage = mode_root / "market"
        text_storage = mode_root / "text"
        disclosure_storage = text_storage / "disclosures"
        disclosure_pdf_storage = disclosure_storage / "pdf"
        disclosure_text_storage = disclosure_storage / "text"
        artifact_storage = mode_root / "artifacts"
        for path in (
            mode_root,
            market_storage,
            text_storage,
            disclosure_storage,
            disclosure_pdf_storage,
            disclosure_text_storage,
            artifact_storage,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return replace(
            self,
            current_mode=mode,
            storage_root=mode_root,
            market_storage=market_storage,
            text_storage=text_storage,
            disclosure_storage=disclosure_storage,
            disclosure_pdf_storage=disclosure_pdf_storage,
            disclosure_text_storage=disclosure_text_storage,
            artifact_storage=artifact_storage,
            db_path=mode_root / "state.db",
        )
