#!/usr/bin/env python3
"""
Stage 1 — Collect.

Pulls every episode of The Startup CPG Podcast from its RSS feed, downloads the
publisher-provided plain-text transcript for each, and writes one JSON file per
episode to ./data/episodes/.

These transcripts are speaker-labeled and timestamped -- far better than YouTube
auto-captions. The feed is the source of truth; re-run this weekly and it will
only fetch what's new.

    pip install requests feedparser
    python fetch_transcripts.py
"""

from __future__ import annotations

import json
import re
import time
from html import unescape
from pathlib import Path

import feedparser
import requests

FEED_URL = "https://feeds.transistor.fm/startupcpg"
OUT_DIR = Path("data/episodes")
SLEEP_BETWEEN = 0.4          # be a polite citizen
TIMEOUT = 30

# The transcript URL lives in a <podcast:transcript> tag. feedparser doesn't
# surface the podcast: namespace cleanly, so we grab it off the raw XML instead.
TRANSCRIPT_RE = re.compile(
    r'<podcast:transcript\s+url="([^"]+)"', re.IGNORECASE
)

# "00:56\nDaniel Scharff\nHello, everyone..." -> we want structured turns.
# Transistor's format is: optional timestamp line, speaker line, then body.
TURN_RE = re.compile(
    r"^(?:(\d{1,2}:\d{2}(?::\d{2})?)\s*\n)?"   # optional mm:ss / hh:mm:ss
    r"([A-Z][^\n]{0,60}?)\s*\n"                 # speaker name
    r"(.+?)(?=\n\n|\Z)",                        # body, up to blank line
    re.MULTILINE | re.DOTALL,
)


def strip_html(raw: str) -> str:
    """Show notes come as CDATA HTML. We want the text -- it's a free, human-written
    topic label for the episode and is genuinely useful supervision later."""
    text = re.sub(r"<br\s*/?>", "\n", raw or "")
    text = re.sub(r"</(p|li|ul|div)>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    return unescape(text).strip()


def to_seconds(ts: str | None) -> int | None:
    if not ts:
        return None
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_turns(raw: str) -> list[dict]:
    """Split a raw transcript into speaker turns. Falls back to one big blob if
    the format ever changes -- better to degrade than to crash mid-run."""
    raw = raw.replace('\r\n', '\n').replace('\r', '\n')
    turns = []
    for ts, speaker, body in TURN_RE.findall(raw):
        body = " ".join(body.split())
        if not body:
            continue
        turns.append({
            "t": to_seconds(ts),
            "speaker": speaker.strip(),
            "text": body,
        })
    if not turns:
        turns = [{"t": None, "speaker": None, "text": " ".join(raw.split())}]
    return turns


def transcript_urls_by_guid(feed_xml: str) -> dict:
    """Map each <item>'s guid -> transcript url, straight off the raw XML."""
    mapping = {}
    for item in re.findall(r"<item>.*?</item>", feed_xml, re.DOTALL):
        guid = re.search(r"<guid[^>]*>(.*?)</guid>", item, re.DOTALL)
        turl = TRANSCRIPT_RE.search(item)
        if guid and turl:
            mapping[guid.group(1).strip()] = turl.group(1)
    return mapping


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "cpg-research/1.0"

    print(f"Fetching feed: {FEED_URL}")
    raw_feed = session.get(FEED_URL, timeout=TIMEOUT).text
    feed = feedparser.parse(raw_feed)
    tmap = transcript_urls_by_guid(raw_feed)
    print(f"  {len(feed.entries)} episodes, {len(tmap)} with transcripts\n")

    fetched = skipped = failed = 0

    for entry in feed.entries:
        guid = entry.get("id", "").strip()
        turl = tmap.get(guid)
        if not turl:
            skipped += 1
            continue

        # Stable, sortable filename. Episode numbers repeat across the two feeds
        # (the "#256" in titles and the podcast:episode tag disagree), so the
        # guid is the only thing we can actually trust as a key.
        slug = guid[:8]
        out_path = OUT_DIR / f"{slug}.json"
        if out_path.exists():
            skipped += 1
            continue

        try:
            raw_transcript = session.get(turl, timeout=TIMEOUT).text
        except Exception as e:                       # noqa: BLE001
            print(f"  FAIL {entry.get('title','?')[:50]}: {e}")
            failed += 1
            continue

        turns = parse_turns(raw_transcript)

        # podcast:person with role="Host" tells us who's hosting. Anyone else
        # speaking is a guest. This is the hook for the "topic x guest role"
        # cut later, so it's worth capturing now.
        hosts = [
            t.get("term") or t.get("value")
            for t in entry.get("tags", [])
        ] if False else []
        host_match = re.search(
            r'<podcast:person role="Host"[^>]*>([^<]+)</podcast:person>',
            raw_feed[raw_feed.find(guid):raw_feed.find(guid) + 8000],
        )
        host = host_match.group(1).strip() if host_match else None

        record = {
            "guid": guid,
            "title": entry.get("title"),
            "published": entry.get("published"),
            "episode": entry.get("itunes_episode"),
            "duration_sec": entry.get("itunes_duration"),
            "link": entry.get("link"),
            "host": host,
            "show_notes": strip_html(entry.get("summary", "")),
            "transcript_url": turl,
            "turns": turns,
            "word_count": sum(len(t["text"].split()) for t in turns),
        }

        out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        fetched += 1
        print(f"  [{fetched:>3}] {record['word_count']:>6,}w  {record['title'][:60]}")
        time.sleep(SLEEP_BETWEEN)

    print(f"\nDone. fetched={fetched} skipped={skipped} failed={failed}")
    print(f"Output: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
