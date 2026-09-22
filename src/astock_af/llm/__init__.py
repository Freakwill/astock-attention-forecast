"""LLM overlay: news-grounded directional view to complement the numeric model."""

from .client import ChatClient, LLMError, load_api_key, parse_json_reply
from .news import (
    NewsBundle,
    NewsItem,
    akshare_news,
    bundle_frame,
    collect,
    eastmoney_news,
    save_bundle,
)
from .overlay import OverlayCall, latest_forecast, run_overlay, score_previous

__all__ = [
    "ChatClient",
    "LLMError",
    "NewsBundle",
    "NewsItem",
    "OverlayCall",
    "akshare_news",
    "bundle_frame",
    "collect",
    "eastmoney_news",
    "latest_forecast",
    "load_api_key",
    "parse_json_reply",
    "run_overlay",
    "save_bundle",
    "score_previous",
]
