#!/usr/bin/env python3
"""
每日 AI 新聞簡報自動抓取與摘要腳本。

功能說明：
1. 使用 feedparser 抓取過去 28 小時內的國內外 AI 科技 RSS 新聞；
   回補歷史日期時，自動搭配 TechCrunch 分頁 RSS 與 Google News 日期區間 RSS 彙整完整候選池。
2. 透過新版 google-genai SDK 呼叫 Gemini Flash 模型，以 Structured Outputs（Pydantic Schema）
   精選 7 至 10 則最重要的 AI 新聞，整理為台灣繁體中文簡報。
3. 自動帶入過去最多 60 天的歷史標題清單進行比對，減少跨日重複報導，
   並標記「相關前期報導（related_history）」供前端點擊跳轉前情提要。
4. 讀取並更新同目錄下的 archive.json，依台灣時間（UTC+8）日期由新到舊排序，最多保留最近 60 天。
"""

from __future__ import annotations

import argparse
import calendar
import html
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
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

# 歷史日期回補用的 Google News 站點查詢設定（搭配 TechCrunch 分頁 RSS）
HISTORICAL_GNEWS_SITES: list[dict[str, str]] = [
    {
        "name": "科技新報 AI",
        "query": "site:technews.tw AI",
        "locale": "hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    },
    {
        "name": "The Verge AI",
        "query": "site:theverge.com AI",
        "locale": "hl=en-US&gl=US&ceid=US:en",
    },
    {
        "name": "INSIDE 硬塞",
        "query": "site:inside.com.tw AI",
        "locale": "hl=zh-TW&gl=TW&ceid=TW:zh-Hant",
    },
    {
        "name": "VentureBeat AI",
        "query": "site:venturebeat.com AI",
        "locale": "hl=en-US&gl=US&ceid=US:en",
    },
]

LOOKBACK_HOURS = 28
MAX_ARCHIVE_DAYS = 60
MAX_CANDIDATE_ARTICLES = 80
MAX_CANDIDATES_PER_DAY_IN_BATCH = 24
HTTP_TIMEOUT_SECONDS = 20
GEMINI_TIMEOUT_MS = 180_000

DEFAULT_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
]
MODEL_NAME = os.environ.get("GEMINI_MODEL", DEFAULT_MODELS[0])

# 同一次執行批次中記住已驗證成功的模型，並排除不可用或無配額模型
_LAST_WORKING_MODEL: str | None = None
_DISABLED_MODELS: set[str] = set()

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


class RelatedNews(BaseModel):
    """與本則新聞高度相關的前期歷史報導。"""

    date: str = Field(
        description="前期報導的台灣日期（YYYY-MM-DD），必須嚴格來自輸入提供的 60 天歷史存檔清單。"
    )
    title: str = Field(
        description="前期報導的完整標題，必須完全照抄輸入提供的 60 天歷史存檔清單，不可改寫或捏造。"
    )
    relation: str = Field(
        description="15 至 35 字的精簡台灣繁體中文脈絡說明，指出本則新聞與該前期報導的延續或關聯。"
    )


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
    related_history: list[RelatedNews] = Field(
        default_factory=list,
        max_length=2,
        description="最多 2 筆與本則新聞高度相關的前期歷史報導（僅限從提供的 60 天歷史存檔清單挑選；若無明確脈絡關聯請回傳空陣列 []）。",
    )


class DailyBriefing(BaseModel):
    """每日精選 AI 新聞清單。"""

    items: list[NewsItem] = Field(
        min_length=1,
        max_length=10,
        description="精選出 7 至 10 則當日最重要的 AI 新聞（若候選新聞不足 7 則，則依實際重要新聞數量整理）。",
    )


class DayBriefingEntry(BaseModel):
    """多日合併批次中單一日期的簡報結構。"""

    date: str = Field(
        description="該日簡報的台灣日期（YYYY-MM-DD），必須與輸入提供的目標日期完全一致。"
    )
    items: list[NewsItem] = Field(
        min_length=1,
        max_length=10,
        description="該日精選的 7 至 10 則最重要的 AI 新聞（若該日候選新聞不足 7 則，則依實際數量整理）。",
    )


class MultiDayBriefing(BaseModel):
    """單次 API 呼叫同時整理多個日期的 AI 簡報輸出。"""

    days: list[DayBriefingEntry] = Field(
        description="依輸入的目標日期順序（由舊到新）逐日回傳精選簡報。"
    )


class RelinkEntry(BaseModel):
    """單則既有新聞的相關舊聞回填結果。"""

    date: str = Field(description="該則既有新聞所屬日期（YYYY-MM-DD）。")
    index: int = Field(description="該則新聞在該日期清單中的 1-based 編號。")
    related_history: list[RelatedNews] = Field(
        default_factory=list,
        max_length=2,
        description="最多 2 筆與本則新聞高度相關的前期歷史報導（僅限嚴格早於該日且來自歷史清單的報導，無關聯則為 []）。",
    )


class RelinkResult(BaseModel):
    """批次為多個既有日期補齊相關舊聞脈絡的輸出格式。"""

    entries: list[RelinkEntry]


SYSTEM_INSTRUCTION = """你是一位資深的台灣科技與 AI 產業總編輯，負責每天為忙碌的技術決策者、工程師與創業者整理「每日 AI 簡報」。

請嚴格遵守以下編採與用語規範：
1. 一律使用道地的「台灣繁體中文（zh-TW）」與台灣慣用技術名詞（例如：模型、推論、算力、晶片、伺服器、資料庫、專案、介面、程式碼、開源、預設），避免中國大陸用語或生硬直譯。
2. 從提供的候選新聞中，精選出 **7 至 10 則**最具代表性、產業影響力或技術突破性的新聞（若候選新聞不足 7 則，則依實際重要新聞數量整理）；若有多家媒體報導同一事件，請整併去重並保留資訊最完整的一則來源與原始連結。
3. 若提示詞附有「過去 60 天歷史存檔標題清單」：
   - 避免重複收錄前幾天已報導過且沒有實質新進展的同一事件。
   - 若當日新聞屬於某個既有事件的後續進展、同一公司／模型／法規的延續動態，請在 `related_history` 中標註最多 2 筆高度相關的前期報導。
   - `related_history` 的 `date` 與 `title` **必須嚴格逐字照抄**歷史存檔清單中的內容，嚴禁修改標題或自行編造不存在的歷史報導；若無高度相關的前期報導，請直接回傳空陣列 `[]`。
4. 每則新聞必須歸類至以下四個領域之一，不可自創分類：
   - 模型與技術突破
   - 商業與市場動態
   - 硬體與應用落地
   - 政策法規與資安隱私
5. 每則新聞的 `bullets` 必須包含 2 至 3 點精煉重點，清楚交代「發生什麼事、關鍵數據或規格、主要參與方」。
6. `analysis` 請以精簡的 1 至 2 句話點出該新聞背後的商業意義、產業競爭格局或技術落地影響。
7. `url` 必須原封不動採用輸入候選清單中的真實連結，嚴禁編造、縮短或更改網址。
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


def get_ssl_context() -> ssl.SSLContext:
    """建立含 certifi CA 憑證的 SSL Context。"""
    if certifi is not None:
        return ssl.create_default_context(cafile=certifi.where())
    return ssl.create_default_context()


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


def strip_gnews_title_suffix(title: str) -> str:
    """移除 Google News RSS 標題尾端附帶的媒體名稱後綴。"""
    return re.sub(r"\s+-\s+[^-]+$", "", title).strip()


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


def fetch_feed_raw(feed_url: str, source_name: str) -> feedparser.FeedParserDict | None:
    """抓取並解析單一 RSS/Atom URL。"""
    req = urllib.request.Request(
        feed_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml, */*",
        },
    )
    try:
        with urllib.request.urlopen(
            req, timeout=HTTP_TIMEOUT_SECONDS, context=get_ssl_context()
        ) as resp:
            raw_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"[警告] 抓取 RSS 失敗（{source_name} - {feed_url}）：{exc}", file=sys.stderr)
        return None

    parsed = feedparser.parse(raw_bytes)
    if parsed.bozo and not parsed.entries:
        print(f"[警告] 解析 RSS 格式異常（{source_name}）：{parsed.bozo_exception}", file=sys.stderr)
        return None
    return parsed


def extract_articles_from_parsed(
    parsed: feedparser.FeedParserDict,
    source_name: str,
    start_utc: datetime,
    end_utc: datetime | None = None,
    is_gnews: bool = False,
) -> list[dict[str, Any]]:
    """從 feedparser 結果抽出指定時間區間內的文章清單。"""
    articles: list[dict[str, Any]] = []
    for entry in parsed.entries:
        published_dt = parse_entry_datetime(entry)
        if published_dt is None or published_dt < start_utc:
            continue
        if end_utc is not None and published_dt > end_utc:
            continue

        title = clean_html(entry.get("title", ""), max_length=240)
        if is_gnews:
            title = strip_gnews_title_suffix(title)
        link = (entry.get("link") or "").strip()
        summary_raw = entry.get("summary") or entry.get("description") or ""
        if not summary_raw and entry.get("content"):
            first_content = entry["content"][0]
            if isinstance(first_content, dict):
                summary_raw = first_content.get("value", "")
        summary = clean_html(summary_raw, max_length=420)
        if is_gnews and summary == title:
            summary = ""

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
    return articles


def fetch_recent_articles(
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    exclude_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    """彙整預設 RSS 來源的文章、依網址去重並依發布時間由新到舊排序。"""
    now_utc = datetime.now(timezone.utc)
    if start_utc is None:
        start_utc = now_utc - timedelta(hours=LOOKBACK_HOURS)

    print(f"開始抓取過去 {LOOKBACK_HOURS} 小時內的 AI 新聞（截止時間：{start_utc.isoformat()}）...")
    collected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for feed_info in RSS_FEEDS:
        parsed = fetch_feed_raw(feed_info["url"], feed_info["name"])
        if parsed is None:
            continue
        items = extract_articles_from_parsed(parsed, feed_info["name"], start_utc, end_utc)
        print(f"[RSS] {feed_info['name']}：取得 {len(items)} 則文章")
        for article in items:
            normalized_url = article["url"].split("#", 1)[0].rstrip("/")
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            collected.append(article)

    return finalize_candidates(collected, exclude_urls)


def fetch_historical_pool(date_list: list[str]) -> list[dict[str, Any]]:
    """一次抓取涵蓋指定歷史日期清單的完整文章池（TechCrunch 分頁 RSS ＋ Google News 日期區間 RSS）。"""
    parsed_dates = [datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=TAIWAN_TZ) for d in date_list]
    min_day = min(parsed_dates)
    max_day = max(parsed_dates)
    pool_start_utc = min_day.astimezone(timezone.utc)
    pool_end_utc = (max_day + timedelta(days=1) - timedelta(seconds=1)).astimezone(timezone.utc)

    print(
        f"開始預載歷史文章池（{min_day.strftime('%Y-%m-%d')} ~ {max_day.strftime('%Y-%m-%d')}）..."
    )
    pool: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    def add_batch(items: list[dict[str, Any]]) -> None:
        for art in items:
            norm = art["url"].split("#", 1)[0].rstrip("/")
            if norm in seen_urls:
                continue
            seen_urls.add(norm)
            pool.append(art)

    # 1. TechCrunch AI 分頁 RSS（自動翻頁直到早於目標最早日期）
    for page in range(1, 22):
        url = f"https://techcrunch.com/category/artificial-intelligence/feed/?paged={page}"
        parsed = fetch_feed_raw(url, f"TechCrunch AI (p{page})")
        if parsed is None or not parsed.entries:
            break
        add_batch(
            extract_articles_from_parsed(parsed, "TechCrunch AI", pool_start_utc, pool_end_utc)
        )
        entry_dts = [parse_entry_datetime(e) for e in parsed.entries]
        valid_dts = [dt for dt in entry_dts if dt is not None]
        if valid_dts and min(valid_dts) < pool_start_utc:
            break

    # 2. 預設 RSS（涵蓋最近 1～3 天的直接來源）
    for feed_info in RSS_FEEDS:
        if feed_info["name"] == "TechCrunch AI":
            continue
        parsed = fetch_feed_raw(feed_info["url"], feed_info["name"])
        if parsed is None:
            continue
        add_batch(
            extract_articles_from_parsed(parsed, feed_info["name"], pool_start_utc, pool_end_utc)
        )

    # 3. Google News 站點日期區間 RSS（每 7 天切一段以避開單次 100 筆上限）
    chunk_start = min_day - timedelta(days=1)
    while chunk_start <= max_day:
        chunk_end = min(chunk_start + timedelta(days=7), max_day + timedelta(days=1))
        after_str = chunk_start.strftime("%Y-%m-%d")
        before_str = (chunk_end + timedelta(days=1)).strftime("%Y-%m-%d")
        for site in HISTORICAL_GNEWS_SITES:
            full_q = urllib.parse.quote(f"{site['query']} after:{after_str} before:{before_str}")
            gnews_url = f"https://news.google.com/rss/search?q={full_q}&{site['locale']}"
            parsed = fetch_feed_raw(gnews_url, f"GNews/{site['name']}")
            if parsed is not None:
                add_batch(
                    extract_articles_from_parsed(
                        parsed,
                        site["name"],
                        pool_start_utc,
                        pool_end_utc,
                        is_gnews=True,
                    )
                )
            time.sleep(1.2)
        chunk_start = chunk_end + timedelta(days=1)

    print(f"歷史文章池預載完成，共彙整 {len(pool)} 則去重候選文章。")
    return pool


def finalize_candidates(
    collected: list[dict[str, Any]],
    exclude_urls: set[str] | None = None,
) -> list[dict[str, Any]]:
    """過濾跨日重複網址並排序截斷候選文章。"""
    if exclude_urls:
        fresh = [
            art
            for art in collected
            if art["url"].split("#", 1)[0].rstrip("/") not in exclude_urls
        ]
        if len(fresh) >= 7:
            collected = fresh

    collected.sort(key=lambda item: item["published_utc"], reverse=True)
    if len(collected) > MAX_CANDIDATE_ARTICLES:
        collected = collected[:MAX_CANDIDATE_ARTICLES]
    return collected


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


def get_history_before(effective_date: str) -> list[tuple[str, str]]:
    """取得嚴格早於 effective_date、最多 60 天內的歷史存檔（日期, 標題）清單。"""
    archive = load_archive(ARCHIVE_PATH)
    prior_days = [
        entry
        for entry in archive
        if isinstance(entry, dict)
        and isinstance(entry.get("date"), str)
        and entry["date"] < effective_date
    ]
    prior_days.sort(key=lambda e: e["date"], reverse=True)
    prior_days = prior_days[:MAX_ARCHIVE_DAYS]

    history: list[tuple[str, str]] = []
    for day in prior_days:
        day_date = day["date"]
        for item in day.get("items") or []:
            if isinstance(item, dict) and item.get("title"):
                history.append((day_date, str(item["title"]).strip()))
    return history


def sanitize_related_history(
    related_list: list[RelatedNews],
    valid_history: list[tuple[str, str]],
) -> list[RelatedNews]:
    """嚴格驗證模型回傳的相關舊聞是否真實存在於歷史存檔，並正規化為精確標題。"""
    if not related_list or not valid_history:
        return []

    by_date: dict[str, list[str]] = {}
    for d, t in valid_history:
        by_date.setdefault(d, []).append(t)

    cleaned: list[RelatedNews] = []
    seen_pairs: set[tuple[str, str]] = set()

    for rel in related_list:
        d = rel.date.strip()
        t = rel.title.strip()
        titles_on_date = by_date.get(d)
        if not titles_on_date:
            continue

        matched_title: str | None = None
        if t in titles_on_date:
            matched_title = t
        else:
            # 若標點或空白略有差異，嘗試子字串安全比對
            for cand in titles_on_date:
                if (len(t) >= 10 and t in cand) or (len(cand) >= 10 and cand in t):
                    matched_title = cand
                    break

        if matched_title is None:
            continue
        pair = (d, matched_title)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        cleaned.append(
            RelatedNews(date=d, title=matched_title, relation=rel.relation.strip())
        )
        if len(cleaned) >= 2:
            break

    return cleaned


def build_prompt(
    articles: list[dict[str, Any]],
    effective_date: str,
    history: list[tuple[str, str]],
) -> str:
    """組合提供給 Gemini 模型的候選新聞與 60 天歷史存檔提示詞。"""
    lines = [
        f"簡報日期（台灣時間）：{effective_date}",
        f"以下為該日期區間內從國內外科技媒體抓取的 {len(articles)} 則 AI 相關候選新聞：",
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

    if history:
        lines.append(
            f"=== 過去最多 {MAX_ARCHIVE_DAYS} 天歷史存檔清單（共 {len(history)} 則，供去重與建立 related_history 脈絡連結用）==="
        )
        for h_date, h_title in history:
            lines.append(f"- [{h_date}] {h_title}")
        lines.append("")

    target_count = "7 至 10 則" if len(articles) >= 7 else f"{len(articles)} 則"
    lines.append(
        f"請從上述候選新聞中精選 {target_count} 最重要的 AI 新聞，整理為結構化的台灣繁體中文每日簡報。"
        "若與上述歷史存檔清單中的前期新聞有明確延續關係或脈絡關聯，請在 `related_history` 標註最多 2 筆（date 與 title 須完全照抄清單）；無明確關聯請留空陣列 `[]`。"
    )
    return "\n".join(lines)


def build_multi_day_prompt(
    day_articles: list[tuple[str, list[dict[str, Any]]]],
    history: list[tuple[str, str]],
) -> str:
    """組合單次呼叫處理多個連續日期的精簡提示詞，大幅節省 API 請求次數。"""
    date_range = f"{day_articles[0][0]} ~ {day_articles[-1][0]}"
    lines = [
        f"請一次為以下 {len(day_articles)} 個日期（{date_range}）分別整理「每日 AI 簡報」，並依日期由舊到新回傳。",
        "每個日期請精選 **7 至 10 則**最重要的 AI 新聞（若該日候選新聞不足 7 則，則依實際重要新聞整理）。",
        "請確保不同日期之間避免重複收錄沒有新進展的同一則事件；若後續日期的新聞與前面日期或歷史存檔有明確延續關係，請在 `related_history` 標註最多 2 筆嚴格早於該日期的相關報導（`date` 與 `title` 須精確一致）。",
        "",
    ]

    if history:
        lines.append(
            f"=== 早於 {day_articles[0][0]} 的歷史存檔標題清單（共 {len(history)} 則，供去重與建立 related_history 用）==="
        )
        for h_date, h_title in history:
            lines.append(f"- [{h_date}] {h_title}")
        lines.append("")

    for d_str, arts in day_articles:
        lines.append(f"================ 目標日期：{d_str}（候選 {len(arts)} 則）================")
        for idx, art in enumerate(arts, start=1):
            published_tw = art["published_utc"].astimezone(TAIWAN_TZ).strftime("%H:%M")
            lines.append(f"[{idx}] {art['source']} ({published_tw}) | {art['title']}")
            lines.append(f"URL: {art['url']}")
            if art["summary"]:
                short_sum = art["summary"][:200].rstrip()
                lines.append(f"摘要：{short_sum}")
        lines.append("")

    lines.append(
        "請務必為上述每一個目標日期都輸出一筆對應的 `days` 項目，且 `date` 與目標日期完全相同。"
    )
    return "\n".join(lines)


def generate_structured_with_fallback(
    client: genai.Client,
    prompt: str,
    schema: type[BaseModel],
    system_instruction: str,
) -> Any:
    """呼叫 Gemini 結構化輸出（多輪重試應對 503 尖峰、記住可用模型並排除無配額模型）。"""
    global _LAST_WORKING_MODEL

    base_models: list[str] = []
    for m in [_LAST_WORKING_MODEL, MODEL_NAME, *DEFAULT_MODELS]:
        if m and m not in base_models and m not in _DISABLED_MODELS:
            base_models.append(m)

    max_rounds = 5
    last_error: Exception | None = None
    response = None

    for round_idx in range(1, max_rounds + 1):
        active_models = [m for m in base_models if m not in _DISABLED_MODELS]
        if not active_models:
            break

        for model_id in active_models:
            print(f"正在呼叫 {model_id}（第 {round_idx}/{max_rounds} 輪）...")
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=0.3,
                        max_output_tokens=32768,
                    ),
                )
                _LAST_WORKING_MODEL = model_id
                break
            except Exception as exc:
                last_error = exc
                err_msg = str(exc)
                short_err = err_msg.replace("\n", " ")[:200]
                is_not_found = "404" in err_msg or "NOT_FOUND" in err_msg
                is_zero_or_daily_quota = (
                    "limit: 0" in err_msg
                    or "PerDay" in err_msg
                    or "requests per day" in err_msg
                )
                is_transient = any(
                    code in err_msg
                    for code in (
                        "503",
                        "UNAVAILABLE",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "500",
                        "INTERNAL",
                        "DEADLINE_EXCEEDED",
                        "timeout",
                        "Timeout",
                    )
                )
                if is_not_found or is_zero_or_daily_quota:
                    print(
                        f"[警告] 模型 {model_id} 不可用或已達每日配額（{short_err}），本批次不再嘗試。",
                        file=sys.stderr,
                    )
                    _DISABLED_MODELS.add(model_id)
                    continue
                if is_transient:
                    print(
                        f"[警告] {model_id} 暫時忙碌（{short_err}），切換下一個模型...",
                        file=sys.stderr,
                    )
                    continue
                raise

        if response is not None:
            break

        if round_idx < max_rounds and any(m not in _DISABLED_MODELS for m in base_models):
            wait_seconds = 12 * round_idx
            print(
                f"[警告] 本輪所有可用模型皆暫時尖峰滿載，將於 {wait_seconds} 秒後進入第 {round_idx + 1} 輪重試...",
                file=sys.stderr,
            )
            time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError(f"所有候選模型均無法完成請求，最後錯誤：{last_error}")

    if response.parsed is not None and isinstance(response.parsed, schema):
        return response.parsed

    if not response.text:
        raise RuntimeError("Gemini 回傳空內容，無法解析結構化結果。")

    return schema.model_validate_json(response.text)


def create_gemini_client() -> genai.Client:
    """建立附帶合理逾時設定的 Gemini API 用戶端。"""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "找不到環境變數 GEMINI_API_KEY。請先設定您的 Gemini API 金鑰後再執行，例如：\n"
            '  PowerShell: $env:GEMINI_API_KEY="您的金鑰"\n'
            '  Bash/Zsh:   export GEMINI_API_KEY="您的金鑰"'
        )
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
    )


def curate_news_with_gemini(
    client: genai.Client,
    articles: list[dict[str, Any]],
    effective_date: str,
    history: list[tuple[str, str]],
) -> DailyBriefing:
    """呼叫 Gemini 精選 7～10 則繁體中文新聞並驗證 60 天相關舊聞連結。"""
    prompt = build_prompt(articles, effective_date, history)
    briefing: DailyBriefing = generate_structured_with_fallback(
        client=client,
        prompt=prompt,
        schema=DailyBriefing,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    for item in briefing.items:
        item.related_history = sanitize_related_history(item.related_history, history)
    return briefing


def relink_existing_dates_batch(client: genai.Client, target_dates: list[str]) -> int:
    """以單次 API 呼叫為多個既有的 archive.json 日期一次補齊 60 天相關舊聞脈絡。"""
    archive = load_archive(ARCHIVE_PATH)
    by_date = {
        entry["date"]: entry
        for entry in archive
        if isinstance(entry, dict) and isinstance(entry.get("date"), str)
    }

    valid_targets = [d for d in sorted(target_dates) if d in by_date and (by_date[d].get("items") or [])]
    if not valid_targets:
        print("[資訊] 沒有符合條件的既有日期需要補齊相關舊聞脈絡。")
        return 0

    # 收集涵蓋所有目標日期前期的 60 天歷史標題清單
    max_target = max(valid_targets)
    full_history = get_history_before(max_target)
    if not full_history:
        print("[資訊] 尚無前期歷史存檔可供關聯比對，略過。")
        return 0

    lines = [
        f"請一次為以下 {len(valid_targets)} 個既有日期的新聞，比對「歷史存檔清單」並找出高度相關的前期報導。",
        "規則：每則新聞最多標註 2 筆高度相關且日期「嚴格早於該新聞日期」的前期報導；`date` 與 `title` 必須完全逐字照抄歷史清單，不可捏造；若無明確延續關係請回傳空陣列 `[]`。",
        "",
    ]

    for d_str in valid_targets:
        items = by_date[d_str].get("items") or []
        lines.append(f"=== 待補齊日期：{d_str}（共 {len(items)} 則）===")
        for idx, item in enumerate(items, start=1):
            lines.append(f"[{d_str} #{idx}] 標題：{item.get('title', '')}")
            bullets = item.get("bullets") or []
            if bullets:
                lines.append(f"重點：{'；'.join(str(b) for b in bullets[:2])}")
        lines.append("")

    lines.append(f"=== 可參照的歷史存檔清單（共 {len(full_history)} 則）===")
    for h_date, h_title in full_history:
        lines.append(f"- [{h_date}] {h_title}")

    system_msg = (
        "你是一位資深的台灣科技總編輯，負責為指定日期的既有 AI 新聞標記過去 60 天內高度相關的前期報導脈絡。"
        "請只挑選具有明確延續關係、因果脈絡、同事件進展或同公司／模型競爭演變且日期嚴格早於該新聞的前期報導；"
        "date 與 title 必須完全逐字照抄輸入的歷史存檔清單，不可捏造；沒有明確關聯請回傳空陣列 []。"
    )

    print(
        f"=== 單次批次補齊 {len(valid_targets)} 個日期的 60 天相關舊聞脈絡"
        f"（{', '.join(valid_targets)}，歷史清單 {len(full_history)} 則） ==="
    )
    result: RelinkResult = generate_structured_with_fallback(
        client=client,
        prompt="\n".join(lines),
        schema=RelinkResult,
        system_instruction=system_msg,
    )

    # 依每個目標日期各自嚴格驗證只連結早於該日期的歷史報導
    grouped: dict[tuple[str, int], list[RelatedNews]] = {}
    for entry in result.entries:
        grouped[(entry.date.strip(), entry.index)] = entry.related_history

    for d_str in valid_targets:
        day_history = get_history_before(d_str)
        items = by_date[d_str].get("items") or []
        linked_count = 0
        for idx, item in enumerate(items, start=1):
            raw_rel = grouped.get((d_str, idx), [])
            cleaned = sanitize_related_history(raw_rel, day_history)
            item["related_history"] = [r.model_dump(mode="json") for r in cleaned]
            if cleaned:
                linked_count += 1
        print(f"完成！{d_str} 共為 {linked_count}/{len(items)} 則新聞建立前情提要脈絡。")

    ARCHIVE_PATH.write_text(
        json.dumps(archive, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


def process_multi_day_batch(
    client: genai.Client,
    batch_dates: list[str],
    pooled_articles: list[dict[str, Any]],
) -> int:
    """以單次 Gemini API 呼叫同時整理多個連續日期，大幅節省免費層每日請求次數。"""
    now_tw = datetime.now(TAIWAN_TZ)
    ordered_dates = sorted(batch_dates)
    day_articles: list[tuple[str, list[dict[str, Any]]]] = []

    for d_str in ordered_dates:
        day_start_tw = datetime.strptime(d_str, "%Y-%m-%d").replace(tzinfo=TAIWAN_TZ)
        day_end_tw = day_start_tw + timedelta(days=1) - timedelta(seconds=1)
        start_utc = day_start_tw.astimezone(timezone.utc)
        end_utc = day_end_tw.astimezone(timezone.utc)

        exclude_urls = get_existing_urls_except(d_str)
        day_items = [
            art for art in pooled_articles if start_utc <= art["published_utc"] <= end_utc
        ]
        candidates = finalize_candidates(day_items, exclude_urls)[:MAX_CANDIDATES_PER_DAY_IN_BATCH]
        if candidates:
            day_articles.append((d_str, candidates))
        else:
            print(f"[警告] 日期 {d_str} 在歷史文章池中無候選文章，略過。", file=sys.stderr)

    if not day_articles:
        return 0

    earliest_date = day_articles[0][0]
    history = get_history_before(earliest_date)
    prompt = build_multi_day_prompt(day_articles, history)

    print(
        f"=== 多日合併批次整理：{day_articles[0][0]} ~ {day_articles[-1][0]}"
        f"（共 {len(day_articles)} 天、單次 API 呼叫，歷史清單 {len(history)} 則） ==="
    )
    try:
        multi: MultiDayBriefing = generate_structured_with_fallback(
            client=client,
            prompt=prompt,
            schema=MultiDayBriefing,
            system_instruction=SYSTEM_INSTRUCTION,
        )
    except Exception as exc:
        print(
            f"[錯誤] 呼叫 Gemini API 失敗（批次 {day_articles[0][0]}~{day_articles[-1][0]}）：{exc}",
            file=sys.stderr,
        )
        return 1

    target_set = {d for d, _ in day_articles}
    returned_days = sorted(
        [d for d in multi.days if d.date.strip() in target_set],
        key=lambda d: d.date.strip(),
    )

    if not returned_days:
        print("[錯誤] 模型未回傳任何符合目標日期的簡報資料。", file=sys.stderr)
        return 1

    for day_entry in returned_days:
        d_str = day_entry.date.strip()
        # 逐日寫入 archive.json，讓同批次後面的日期可安全驗證連結到前面的日期
        valid_hist = get_history_before(d_str)
        for item in day_entry.items:
            item.related_history = sanitize_related_history(item.related_history, valid_hist)
        linked_count = sum(1 for item in day_entry.items if item.related_history)
        updated_archive = update_archive(
            DailyBriefing(items=day_entry.items),
            now_tw,
            target_date_str=d_str,
        )
        print(
            f"完成！已精選 {len(day_entry.items)} 則新聞（其中 {linked_count} 則附帶前情提要脈絡）"
            f"寫入 {ARCHIVE_PATH.name}（日期：{d_str}），目前共保存 {len(updated_archive)} 天歷史紀錄。"
        )
    return 0


def update_archive(
    briefing: DailyBriefing,
    now_tw: datetime,
    target_date_str: str | None = None,
) -> list[dict[str, Any]]:
    """移除同日期舊資料、寫入指定日期簡報並依日期由新到舊排序，最多保留最近 60 天。"""
    date_str = target_date_str or now_tw.strftime("%Y-%m-%d")
    archive = load_archive(ARCHIVE_PATH)

    filtered = [
        entry for entry in archive if isinstance(entry, dict) and entry.get("date") != date_str
    ]

    record: dict[str, Any] = {
        "date": date_str,
        "generated_at": now_tw.isoformat(timespec="seconds"),
        "items": [item.model_dump(mode="json") for item in briefing.items],
    }

    updated = [record, *filtered]
    updated.sort(key=lambda entry: str(entry.get("date", "")), reverse=True)
    updated = updated[:MAX_ARCHIVE_DAYS]

    ARCHIVE_PATH.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return updated


def get_existing_urls_except(target_date_str: str) -> set[str]:
    """收集其他已存檔日期使用過的網址，回補歷史日期時優先避免跨日重複。"""
    urls: set[str] = set()
    for entry in load_archive(ARCHIVE_PATH):
        if not isinstance(entry, dict) or entry.get("date") == target_date_str:
            continue
        for item in entry.get("items") or []:
            if isinstance(item, dict) and item.get("url"):
                urls.add(str(item["url"]).split("#", 1)[0].rstrip("/"))
    return urls


def expand_date_spec(spec: str) -> list[str]:
    """解析日期清單（支援逗號分隔與 YYYY-MM-DD..YYYY-MM-DD 區間語法，由舊到新排序）。"""
    dates: list[str] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if ".." in token:
            start_s, end_s = [x.strip() for x in token.split("..", 1)]
            cur = datetime.strptime(start_s, "%Y-%m-%d")
            end = datetime.strptime(end_s, "%Y-%m-%d")
            step = 1 if cur <= end else -1
            while True:
                dates.append(cur.strftime("%Y-%m-%d"))
                if cur == end:
                    break
                cur += timedelta(days=step)
        else:
            datetime.strptime(token, "%Y-%m-%d")
            dates.append(token)
    # 去重並維持原順序
    return list(dict.fromkeys(dates))


def process_single_date(
    client: genai.Client,
    target_date_str: str | None = None,
    pooled_articles: list[dict[str, Any]] | None = None,
) -> int:
    """抓取並更新單一日期（未指定時抓取過去 28 小時並記為台灣今日）。"""
    now_tw = datetime.now(TAIWAN_TZ)

    if target_date_str:
        day_start_tw = datetime.strptime(target_date_str, "%Y-%m-%d").replace(tzinfo=TAIWAN_TZ)
        day_end_tw = day_start_tw + timedelta(days=1) - timedelta(seconds=1)
        start_utc = day_start_tw.astimezone(timezone.utc)
        end_utc = day_end_tw.astimezone(timezone.utc)

        print(f"=== 每日 AI 新聞簡報回補（目標台灣日期：{target_date_str}） ===")
        exclude_urls = get_existing_urls_except(target_date_str)
        if pooled_articles is not None:
            day_items = [
                art for art in pooled_articles if start_utc <= art["published_utc"] <= end_utc
            ]
            articles = finalize_candidates(day_items, exclude_urls)
            print(f"從歷史文章池篩出 {len(articles)} 則候選新聞。")
        else:
            articles = fetch_recent_articles(start_utc, end_utc, exclude_urls=exclude_urls)
        effective_date = target_date_str
    else:
        effective_date = now_tw.strftime("%Y-%m-%d")
        print(f"=== 每日 AI 新聞簡報產生器（台灣日期：{effective_date}） ===")
        articles = fetch_recent_articles()

    if not articles:
        print(f"[警告] 日期 {effective_date} 未抓取到任何 RSS 文章，略過。", file=sys.stderr)
        return 1

    history = get_history_before(effective_date)
    if history:
        print(f"已載入過去最多 {MAX_ARCHIVE_DAYS} 天歷史存檔共 {len(history)} 則標題供脈絡比對。")

    try:
        briefing = curate_news_with_gemini(client, articles, effective_date, history)
    except Exception as exc:
        print(f"[錯誤] 呼叫 Gemini API 失敗（{effective_date}）：{exc}", file=sys.stderr)
        return 1

    linked_count = sum(1 for item in briefing.items if item.related_history)
    updated_archive = update_archive(briefing, now_tw, target_date_str=effective_date)
    print(
        f"完成！已精選 {len(briefing.items)} 則新聞（其中 {linked_count} 則附帶前情提要脈絡）"
        f"寫入 {ARCHIVE_PATH.name}（日期：{effective_date}），目前共保存 {len(updated_archive)} 天歷史紀錄。"
    )
    return 0


def main() -> int:
    load_local_env(ENV_PATH)

    parser = argparse.ArgumentParser(description="抓取並生成每日 AI 新聞簡報")
    parser.add_argument(
        "--dates",
        type=str,
        default="",
        help="選填：指定要回補的台灣日期（支援逗號分隔或 YYYY-MM-DD..YYYY-MM-DD 區間）",
    )
    parser.add_argument(
        "--relink-dates",
        type=str,
        default="",
        help="選填：僅為 archive.json 中已存在的日期補齊 60 天相關舊聞脈絡（單次合併呼叫）",
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        default=0,
        help="選填：每次 API 呼叫合併處理的天數（例如 5 代表 5 天合為 1 次呼叫，預設 1）",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="選填：回補日期時自動略過 archive.json 中已存在的日期",
    )
    args = parser.parse_args()

    try:
        client = create_gemini_client()
    except Exception as exc:
        print(f"[錯誤] {exc}", file=sys.stderr)
        return 1

    raw_dates = args.dates.strip() or os.environ.get("TARGET_DATES", "").strip()
    raw_relink = args.relink_dates.strip() or os.environ.get("RELINK_DATES", "").strip()
    env_batch = os.environ.get("BATCH_DAYS", "").strip()
    batch_days = args.batch_days or (int(env_batch) if env_batch.isdigit() else 1)
    batch_days = max(1, min(batch_days, 7))
    skip_existing = args.skip_existing or (
        os.environ.get("SKIP_EXISTING", "").strip().lower() in ("1", "true", "yes")
    )

    if raw_dates:
        date_list = expand_date_spec(raw_dates)
        if skip_existing:
            existing_dates = {
                str(e.get("date"))
                for e in load_archive(ARCHIVE_PATH)
                if isinstance(e, dict) and e.get("date")
            }
            before_len = len(date_list)
            date_list = [d for d in date_list if d not in existing_dates]
            skipped = before_len - len(date_list)
            if skipped:
                print(f"[資訊] 已自動略過 {skipped} 個已存在的存檔日期。")

        if date_list:
            pooled = fetch_historical_pool(date_list)
            if batch_days > 1:
                chunks = [
                    date_list[i : i + batch_days]
                    for i in range(0, len(date_list), batch_days)
                ]
                for idx, chunk in enumerate(chunks):
                    code = process_multi_day_batch(client, chunk, pooled)
                    if code != 0:
                        return code
                    if idx < len(chunks) - 1:
                        time.sleep(10)
            else:
                for idx, d in enumerate(date_list):
                    code = process_single_date(client, target_date_str=d, pooled_articles=pooled)
                    if code != 0:
                        return code
                    if idx < len(date_list) - 1:
                        time.sleep(10)

    if raw_relink:
        relink_list = expand_date_spec(raw_relink)
        code = relink_existing_dates_batch(client, relink_list)
        if code != 0:
            return code

    if raw_dates or raw_relink:
        return 0

    return process_single_date(client, target_date_str=None)


if __name__ == "__main__":
    raise SystemExit(main())
