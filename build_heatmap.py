#!/usr/bin/env python3
"""
Stages 2-5 — Chunk, discover topics, classify, and render the heatmap.

Run fetch_transcripts.py first. Then:

    pip install sentence-transformers umap-learn hdbscan pandas seaborn matplotlib anthropic
    export ANTHROPIC_API_KEY=...
    python build_heatmap.py

Two deliberate design choices worth understanding before you change anything:

1. The unit of analysis is the CHUNK, not the episode. Talking points live at
   the paragraph level. Counting per-episode throws away almost all the signal
   and gives you a heatmap where every cell is ~1.

2. Topics are DISCOVERED first (cluster + LLM-label), then LOCKED and used as a
   fixed taxonomy for a cheap classification pass. Hand-writing the taxonomy up
   front feels faster and reliably produces the wrong categories -- you'll
   invent "influencer marketing" and miss that half the corpus is actually
   arguing about deductions and chargebacks.
"""

import json
import os
import re
from collections import Counter
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

EPISODES = Path("data/episodes")
ARTIFACTS = Path("data/artifacts")
TARGET_WORDS = 220          # chunk size; ~1-2 min of speech
MIN_WORDS = 60              # drop "yeah, totally" turns
MODEL = "claude-haiku-4-5-20251001"

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# --------------------------------------------------------------------------
# Stage 2 — Chunk
# --------------------------------------------------------------------------
def load_chunks() -> pd.DataFrame:
    rows = []
    for path in sorted(EPISODES.glob("*.json")):
        ep = json.loads(path.read_text())
        host = (ep.get("host") or "").lower()

        buf, buf_words, buf_speaker, buf_t = [], 0, None, None
        for turn in ep["turns"]:
            words = len(turn["text"].split())
            if buf_speaker is None:
                buf_speaker, buf_t = turn["speaker"], turn["t"]
            buf.append(turn["text"])
            buf_words += words

            if buf_words >= TARGET_WORDS:
                rows.append(_row(ep, host, buf, buf_speaker, buf_t))
                buf, buf_words, buf_speaker, buf_t = [], 0, None, None

        if buf_words >= MIN_WORDS:
            rows.append(_row(ep, host, buf, buf_speaker, buf_t))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    print(f"{len(df):,} chunks from {df.guid.nunique()} episodes")
    return df


def _row(ep, host, buf, speaker, t):
    sp = (speaker or "").lower()
    return {
        "guid": ep["guid"],
        "title": ep["title"],
        "published": ep["published"],
        "speaker": speaker,
        # Host vs guest is the single most valuable axis you have. Host turns
        # are mostly questions and framing; guest turns are where the actual
        # claims live. Keep them separable.
        "is_host": bool(host and sp and (sp in host or host in sp)),
        "t": t,
        "text": " ".join(buf),
    }


# --------------------------------------------------------------------------
# Stage 3 — Discover topics (embed -> reduce -> cluster -> LLM-label)
# --------------------------------------------------------------------------
def discover_topics(df: pd.DataFrame, sample: int = 6000) -> dict:
    """Cluster a sample of GUEST chunks and have an LLM name each cluster."""
    guest = df[~df.is_host]
    sub = guest.sample(min(sample, len(guest)), random_state=0).reset_index(drop=True)

    print("Embedding...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = encoder.encode(sub.text.tolist(), batch_size=64, show_progress_bar=True)

    print("Reducing + clustering...")
    reduced = umap.UMAP(
        n_neighbors=15, n_components=5, metric="cosine", random_state=0
    ).fit_transform(vecs)
    labels = hdbscan.HDBSCAN(
        min_cluster_size=25, metric="euclidean"
    ).fit_predict(reduced)
    sub["cluster"] = labels

    n = len([c for c in set(labels) if c != -1])
    print(f"{n} clusters ({(labels == -1).mean():.0%} noise)")

    # Show the LLM real examples from each cluster and let it write the label.
    # Reading the clusters yourself first is worth the 20 minutes -- this is the
    # step where you learn what the corpus is actually about.
    digest = []
    for c in sorted(set(labels)):
        if c == -1:
            continue
        examples = sub[sub.cluster == c].text.head(8).tolist()
        digest.append(f"--- Cluster {c} ({(labels == c).sum()} chunks) ---\n"
                      + "\n".join(f"* {e[:300]}" for e in examples))

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": (
            "These are clusters of excerpts from a CPG (consumer packaged goods) "
            "founder podcast. For each cluster, write a short, concrete topic "
            "label (3-6 words) that a CPG operator would recognize. Prefer "
            "specific industry language over generic business-speak: "
            "'trade spend and deductions' not 'financial management'.\n\n"
            "Then collapse near-duplicate clusters and return a FINAL taxonomy "
            "of 20-30 topics as a JSON array of {\"topic\": str, \"description\": str}. "
            "Return only the JSON array, no preamble.\n\n" + "\n\n".join(digest)
        )}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    taxonomy = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2))
    print(f"\nTaxonomy ({len(taxonomy)} topics):")
    for t in taxonomy:
        print(f"  - {t['topic']}")
    return taxonomy


# --------------------------------------------------------------------------
# Stage 4 — Classify every chunk against the locked taxonomy
# --------------------------------------------------------------------------
def classify(df: pd.DataFrame, taxonomy: list, batch: int = 20) -> pd.DataFrame:
    """Multi-label: a chunk about co-packer margins hits two topics, not one.
    For 20-30k chunks, move this to the Batch API -- same prompt, half the cost,
    and you're not babysitting a loop for an hour."""
    menu = "\n".join(f"{i}. {t['topic']}" for i, t in enumerate(taxonomy))
    out = []

    for start in range(0, len(df), batch):
        rows = df.iloc[start:start + batch]
        numbered = "\n\n".join(
            f"[{i}] {r.text[:1200]}" for i, r in enumerate(rows.itertuples())
        )
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": (
                f"Topics:\n{menu}\n\n"
                "For each excerpt below, return the topic indices it substantively "
                "covers (0-3 of them; empty list if it's just chitchat, intros, or "
                "ad reads). Return only a JSON object mapping excerpt index to a "
                "list of topic indices.\n\n" + numbered
            )}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            parsed = json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M))
        except json.JSONDecodeError:
            parsed = {}
        for i in range(len(rows)):
            out.append([taxonomy[j]["topic"] for j in parsed.get(str(i), [])
                        if j < len(taxonomy)])
        print(f"\r  classified {min(start + batch, len(df)):,}/{len(df):,}", end="")

    df = df.copy()
    df["topics"] = out
    df.to_json(ARTIFACTS / "classified.jsonl", orient="records", lines=True)
    print()
    return df


# --------------------------------------------------------------------------
# Stage 5 — Heatmaps
# --------------------------------------------------------------------------
def render(df: pd.DataFrame) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    long = df.explode("topics").dropna(subset=["topics"])

    # (a) topic x quarter, as SHARE of that quarter's chunks. Share, not count --
    #     otherwise you're just plotting how many episodes they shipped.
    pivot = (
        long.pivot_table(index="topics", columns="quarter",
                         values="guid", aggfunc="count")
        .fillna(0)
    )
    pivot = pivot.div(pivot.sum(axis=0), axis=1)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(max(12, pivot.shape[1] * 0.45), pivot.shape[0] * 0.4 + 2))
    sns.heatmap(pivot, cmap="rocket_r", cbar_kws={"label": "share of chunks"},
                linewidths=0.4, linecolor="white")
    plt.title("Startup CPG: what the show talks about, over time")
    plt.xlabel(""); plt.ylabel("")
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "heatmap_topic_by_quarter.png", dpi=160)
    print(f"wrote {ARTIFACTS / 'heatmap_topic_by_quarter.png'}")

    # (b) host vs guest. Probably your highest-signal cut: the gap between what
    #     the host keeps ASKING about and what guests keep VOLUNTEERING is where
    #     the interesting stuff is.
    roles = (
        long.assign(role=lambda d: d.is_host.map({True: "host", False: "guest"}))
        .pivot_table(index="topics", columns="role", values="guid", aggfunc="count")
        .fillna(0)
    )
    roles = roles.div(roles.sum(axis=0), axis=1)
    roles["gap"] = roles.get("guest", 0) - roles.get("host", 0)
    roles.sort_values("gap").to_csv(ARTIFACTS / "host_vs_guest.csv")
    print(f"wrote {ARTIFACTS / 'host_vs_guest.csv'}")

    # (c) co-occurrence: which topics always travel together.
    pairs = Counter()
    for topics in df.topics:
        for a in topics:
            for b in topics:
                if a < b:
                    pairs[(a, b)] += 1
    co = pd.Series(pairs).unstack().fillna(0)
    co.to_csv(ARTIFACTS / "cooccurrence.csv")
    print(f"wrote {ARTIFACTS / 'cooccurrence.csv'}")


if __name__ == "__main__":
    chunks = load_chunks()
    tax_path = ARTIFACTS / "taxonomy.json"
    taxonomy = (json.loads(tax_path.read_text()) if tax_path.exists()
                else discover_topics(chunks))
    render(classify(chunks, taxonomy))
