#!/usr/bin/env python3
"""
每日 AI 新聞簡報自動抓取與摘要腳本。

功能說明：
1. 使用 feedparser 抓取過去 28 小時內的國內外 AI 科技 RSS 新聞。
2. 透過新版 google-genai SDK 呼叫 gemini-2.5-flash，以 Structured Outputs（Pydantic Schema）
   精選 6 至 8 則最重要的 AI 新聞，並整理為台灣繁體中文簡報。
3. 讀取並更新同目錄下的 archive.json，以台灣時間（UTC+8）日期為鍵值，
   覆蓋同日舊資料、將最新簡報置於陣列最前方，並最多保留最近 60 天紀錄。
"""

from __future__ import annotations

import calendar
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import feedparser
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

try:
    import certifi
except ImportError:
    certifi = None  # type: ignore[assignment]

# 台灣時區（UTC+8）
TAIWAN_TZ = timezone(timedelta(hours=8))

# 預設抓取之國內外 AI 科技新聞 RSS 來源
RSS_FEEDS: list[dict[str, str]] = [
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
    },
    {
        "name": "The Verge AI",
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
    },
    {
        "name": "VentureBeat AI",
        "url": "https://venturebeat.com/category/ai/feed/",
    },
    {
        "name": "科技新報 AI",
        "url": "https://technews.tw/category/ai/feed/",
    },
    {
        "name": "INSIDE 硬塞",
        "url": "https://www.inside.com.tw/feed/tag/AI",
    },
]

LOOKBACK_HOURS = 28
MAX_ARCHIVE_DAYS = 60
MAX_CANDIDATE_ARTICLES = 80
HTTP_TIMEOUT_SECONDS = 20
DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]
MODEL_NAME = os.environ.get("GEMINI_MODEL", DEFAULT_MODELS[0])

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = BASE_DIR / "archive.json"
ENV_PATH = BASE_DIR / ".env"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36 ai-daily-briefing/1.0"
)


class NewsCategory(str, Enum):
    """嚴格限定的四大 AI 新聞領域分類。"""

    MODEL_TECH = "模型與技術突破"
    BUSINESS_MARKET = "商業與市場動態"
    HARDWARE_APP = "硬體與應用落地"
    POLICY_SECURITY = "政策法規與資安隱私"


class NewsItem(BaseModel):
    """單則 AI 簡報新聞的結構化資料模型。"""

    category: NewsCategory = Field(
        description="新聞分類，嚴格限定為「模型與技術突破」、「商業與市場動態」、「硬體與應用落地」、「政策法規與資安隱私」四者之一。"
    )
    source: str = Field(
        description="媒體來源名稱，例如 TechCrunch AI、The Verge AI、VentureBeat AI、科技新報 AI、INSIDE 硬塞。"
    )
    title: str = Field(
        description="精準且具資訊量的台灣繁體中文新聞標題。"
    )
    bullets: list[str] = Field(
        min_length=2,
        max_length=3,
        description="2 至 3 點精煉的條列式核心重點，使用流暢的台灣繁體中文。",
    )
    analysis: str = Field(
        description="簡短精闢的商業或技術影響分析（約 1 至 2 句台灣繁體中文）。"
    )
    url: str = Field(
        description="原始報導完整網址，必須直接沿用輸入清單中的對應連結，不可自行捏造或修改。"
    )


class DailyBriefing(BaseModel):
    """每日精選 AI 新聞清單。"""

    items: list[NewsItem] = Field(
        min_length=1,
        max_length=8,
        description="精選出 6 至 8 則當日最重要的 AI 新聞（若候選新聞不足 6 則，則依實際重要新聞數量整理）。",
    )


SYSTEM_INSTRUCTION = """你是一位資深的台灣科技與 AI 產業總編輯，負責每天為忙碌的技術決策者、工程師與創業者整理「每日 AI 簡報」。

請嚴格遵守以下編採與用語規範：
1. 一律使用道地的「台灣繁體中文（zh-TW）」與台灣慣用技術名詞（例如：模型、推論、算力、晶片、伺服器、資料庫、專案、介面、程式碼、開源、預設），避免中國大陸用語或生硬直譯。
2. 從提供的候選新聞中，精選出 6 至 8 則最具代表性、產業影響力或技術突破性的新聞；若有多家媒體報導同一事件，請整併去重並保留資訊最完整的一則來源與原始連結。
3. 每則新聞必須歸類至以下四個領域之一，不可自創分類：
   - 模型與技術突破
   - 商業與市場動態
   - 硬體與應用落地
   - 政策法規與資安隱私
4. 每則新聞的 `bullets` 必須包含 2 至 3 點精煉重點，清楚交代「發生什麼事、關鍵數據或規格、主要參與方」。
5. `analysis` 請以精簡的 1 至 2 句話點出該新聞背後的商業意義、產業競爭格局或技術落地影響。
6. `url` 必須原封不動採用輸入候選清單中的真實連結，嚴禁編造、縮短或更改網址。
"""


def load_local_env(env_path: Path) -> None:
    """若同目錄存在 .env 檔案，載入尚未設定的環境變數（不覆蓋既有系統環境變數）。"""
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:
        print(f"[警告] 讀取 .env 失敗：{exc}", file=sys.stderr)


def clean_html(raw_text: str, max_length: int = 420) -> str:
    """移除 HTML 標籤與多餘空白並截斷過長摘要。"""
    if not raw_text:
        return ""
    text = html.unescape(raw_text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_length:
        text = text[: max_length - 1].rstrip() + "…"
    return text


def parse_entry_datetime(entry: Any) -> datetime | None:
    """從 feedparser 條目解析發布時間（回傳 UTC aware datetime）。"""
    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        parsed_struct = entry.get(field)
        if parsed_struct:
            try:
                timestamp = calendar.timegm(parsed_struct)
                return datetime.fromtimestamp(timestamp, tz=timezone.utc)
            except (OverflowError, TypeError, ValueError):
                continue
    return None


def fetch_single_feed(feed_info: dict[str, str], cutoff_utc: datetime) -> list[dict[str, Any]]:
    """抓取單一 RSS 來源並篩選過去指定時數內的文章。"""
    source_name = feed_info["name"]
    feed_url = feed_info["url"]
    articles: list[dict[str, Any]] = []

    req = urllib.request.Request(
        feed_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        },
    )

    ssl_context = (
        ssl.create_default_context(cafile=certifi.where())
        if certifi is not None
        else ssl.create_default_context()
    )

    try:
        with urllib.request.urlopen(
            req, timeout=HTTP_TIMEOUT_SECONDS, context=ssl_context
        ) as resp:
            raw_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[警告] 抓取 RSS 失敗（{source_name} - {feed_url}）：{exc}", file=sys.stderr)
        return articles

    parsed = feedparser.parse(raw_bytes)
    if parsed.bozo and not parsed.entries:
        print(f"[警告] 解析 RSS 格式異常（{source_name}）：{parsed.bozo_exception}", file=sys.stderr)
        return articles

    for entry in parsed.entries:
        published_dt = parse_entry_datetime(entry)
        if published_dt is None or published_dt < cutoff_utc:
            continue

        title = clean_html(entry.get("title", ""), max_length=240)
        link = (entry.get("link") or "").strip()
        summary_raw = entry.get("summary") or entry.get("description") or ""
        if not summary_raw and entry.get("content"):
            first_content = entry["content"][0]
            if isinstance(first_content, dict):
                summary_raw = first_content.get("value", "")
        summary = clean_html(summary_raw, max_length=420)

        if not title or not link:
            continue

        articles.append(
            {
                "source": source_name,
                "title": title,
                "url": link,
                "summary": summary,
                "published_utc": published_dt,
            }
        )

    print(f"[RSS] {source_name}：過去 {LOOKBACK_HOURS} 小時內取得 {len(articles)} 則文章")
    return articles


def fetch_recent_articles() -> list[dict[str, Any]]:
    """彙整所有 RSS 來源的文章、依網址去重並依發布時間由新到舊排序。"""
    now_utc = datetime.now(timezone.utc)
    cutoff_utc = now_utc - timedelta(hours=LOOKBACK_HOURS)
    print(f"開始抓取過去 {LOOKBACK_HOURS} 小時內的 AI 新聞（截止時間：{cutoff_utc.isoformat()}）...")

    collected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for feed_info in RSS_FEEDS:
        for article in fetch_single_feed(feed_info, cutoff_utc):
            normalized_url = article["url"].split("#", 1)[0].rstrip("/")
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            collected.append(article)

    collected.sort(key=lambda item: item["published_utc"], reverse=True)
    if len(collected) > MAX_CANDIDATE_ARTICLES:
        collected = collected[:MAX_CANDIDATE_ARTICLES]

    print(f"共彙整 {len(collected)} 則候選新聞。")
    return collected


def build_prompt(articles: list[dict[str, Any]], today_tw: str) -> str:
    """組合提供給 Gemini 模型的候選新聞清單提示詞。"""
    lines = [
        f"今日日期（台灣時間）：{today_tw}",
        f"以下為過去 {LOOKBACK_HOURS} 小時內從國內外科技媒體抓取的 {len(articles)} 則 AI 相關候選新聞：",
        "",
    ]
    for idx, art in enumerate(articles, start=1):
        published_tw = art["published_utc"].astimezone(TAIWAN_TZ).strftime("%Y-%m-%d %H:%M")
        lines.append(f"[{idx}] 來源：{art['source']} | 發布時間：{published_tw}")
        lines.append(f"標題：{art['title']}")
        lines.append(f"網址：{art['url']}")
        if art["summary"]:
            lines.append(f"摘要：{art['summary']}")
        lines.append("")

    target_count = "6 至 8 則" if len(articles) >= 6 else f"{len(articles)} 則"
    lines.append(
        f"請從上述候選清單中精選 {target_count} 最重要的 AI 新聞，整理為結構化的台灣繁體中文每日簡報。"
    )
    return "\n".join(lines)


def curate_news_with_gemini(articles: list[dict[str, Any]], today_tw: str) -> DailyBriefing:
    """呼叫 Gemini 2.5 Flash 與 Structured Outputs 精選並摘要新聞。"""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "找不到環境變數 GEMINI_API_KEY。請先設定您的 Gemini API 金鑰後再執行，例如：\n"
            '  PowerShell: $env:GEMINI_API_KEY="您的金鑰"\n'
            '  Bash/Zsh:   export GEMINI_API_KEY="您的金鑰"'
        )

    client = genai.Client(api_key=api_key)
    prompt = build_prompt(articles, today_tw)

    candidate_models = [MODEL_NAME] + [m for m in DEFAULT_MODELS if m != MODEL_NAME]
    max_attempts_per_model = 2
    response = None
    last_error: Exception | None = None

    for model_id in candidate_models:
        for attempt in range(1, max_attempts_per_model + 1):
            print(f"正在呼叫 {model_id} 進行新聞精選與繁體中文摘要（第 {attempt}/{max_attempts_per_model} 次）...")
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        response_mime_type="application/json",
                        response_schema=DailyBriefing,
                        temperature=0.3,
                    ),
                )
                break
            except Exception as exc:
                last_error = exc
                err_msg = str(exc)
                is_not_found = "404" in err_msg or "NOT_FOUND" in err_msg
                is_transient = any(
                    code in err_msg
                    for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")
                )
                if is_not_found:
                    print(f"[警告] 模型 {model_id} 不可用，嘗試下一個備援模型...", file=sys.stderr)
                    break
                if is_transient and attempt < max_attempts_per_model:
                    wait_seconds = 5 * attempt
                    print(
                        f"[警告] {model_id} 暫時忙碌（{exc}），將於 {wait_seconds} 秒後重試...",
                        file=sys.stderr,
                    )
                    time.sleep(wait_seconds)
                    continue
                if is_transient:
                    print(f"[警告] {model_id} 持續忙碌，嘗試切換至下一個備援模型...", file=sys.stderr)
                    break
                raise
        if response is not None:
            break

    if response is None:
        raise RuntimeError(f"所有候選模型均無法完成請求，最後錯誤：{last_error}")

    if response.parsed is not None and isinstance(response.parsed, DailyBriefing):
        return response.parsed

    if not response.text:
        raise RuntimeError("Gemini 回傳空內容，無法解析結構化新聞簡報。")

    return DailyBriefing.model_validate_json(response.text)


def load_archive(path: Path) -> list[dict[str, Any]]:
    """讀取既有的 archive.json，若檔案不存在則初始化為空陣列。"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        print("[警告] archive.json 根節點不是陣列，將重新初始化。", file=sys.stderr)
        return []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[警告] 讀取 archive.json 失敗（{exc}），將重新初始化。", file=sys.stderr)
        return []


def update_archive(briefing: DailyBriefing, now_tw: datetime) -> list[dict[str, Any]]:
    """移除同日期舊資料、將今日簡報插入陣列最前方，最多保留最近 60 天並寫回。"""
    today_str = now_tw.strftime("%Y-%m-%d")
    archive = load_archive(ARCHIVE_PATH)

    # 移除同日期（YYYY-MM-DD）的舊資料
    filtered = [entry for entry in archive if isinstance(entry, dict) and entry.get("date") != today_str]

    today_record: dict[str, Any] = {
        "date": today_str,
        "generated_at": now_tw.isoformat(timespec="seconds"),
        "items": [item.model_dump(mode="json") for item in briefing.items],
    }

    updated = [today_record, *filtered][:MAX_ARCHIVE_DAYS]
    ARCHIVE_PATH.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated


def main() -> int:
    load_local_env(ENV_PATH)

    now_tw = datetime.now(TAIWAN_TZ)
    today_tw = now_tw.strftime("%Y-%m-%d")
    print(f"=== 每日 AI 新聞簡報產生器（台灣日期：{today_tw}） ===")

    articles = fetch_recent_articles()
    if not articles:
        print("[錯誤] 過去 28 小時內未抓取到任何 RSS 文章，請檢查網路連線或 RSS 來源狀態。", file=sys.stderr)
        return 1

    try:
        briefing = curate_news_with_gemini(articles, today_tw)
    except Exception as exc:
        print(f"[錯誤] 呼叫 Gemini API 失敗：{exc}", file=sys.stderr)
        return 1

    updated_archive = update_archive(briefing, now_tw)
    print(
        f"完成！已精選 {len(briefing.items)} 則新聞寫入 {ARCHIVE_PATH.name}，"
        f"目前共保存 {len(updated_archive)} 天歷史紀錄。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
