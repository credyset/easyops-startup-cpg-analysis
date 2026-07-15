# Handoff: Startup CPG Podcast — Transcript Corpus & Topic Heatmap

**For:** the agent working with Logic Agency
**Goal:** collect the ~100 most recent Startup CPG Podcast episodes as clean, speaker-labeled transcripts, then build a "heatmap" of the most common talking points across them.
**Status:** pipeline designed and coded (steps 1–5). Step 1 (collection) needs to run in an environment with open outbound network — see *Why this is being handed off* below. Everything after step 1 is ready to run.

---

## TL;DR for the receiving agent

1. Run `fetch_transcripts.py` → produces `data/episodes/*.json` (one file per episode). **~3 minutes, needs internet access to `transistor.fm`.**
2. Run `build_heatmap.py` → chunks, embeds, clusters, LLM-labels a topic taxonomy, classifies every chunk, and renders the heatmap. **Needs `pip` access and an `ANTHROPIC_API_KEY`.**
3. `taxonomy_v0.json` is a *hand-drafted hypothesis* taxonomy built from only 8 episodes. Do **not** treat it as ground truth. Use it only to sanity-check what the clustering discovers (step 3 inside `build_heatmap.py`). Where the two disagree, the clustering wins — it saw the whole corpus; the hypothesis saw eight.

That's the whole job. The rest of this document is the *why* behind each decision, so you don't have to reverse-engineer the reasoning.

---

## The key insight (don't skip this)

The original ask was to scrape the **YouTube channel** (@startupcpg) and build a talking-points heatmap. **Do not scrape YouTube.** The YouTube channel is a repost of *The Startup CPG Podcast*, and the podcast publishes clean transcripts for free via its RSS feed.

- Feed URL: `https://feeds.transistor.fm/startupcpg`
- Every `<item>` in the feed ends with a tag like:
  `<podcast:transcript url="https://share.transistor.fm/s/<id>/transcript.txt" type="text/plain"/>`
- Those transcripts are **speaker-labeled and timestamped** — e.g. real `"Daniel Scharff"` / `"Hans Eisenbeis"` turns, punctuated, with `mm:ss` markers. That is night-and-day better than YouTube auto-captions (which are an unpunctuated word-soup with no speaker turns and bulk-download IP blocking).

Speaker labels are the single most valuable property of this data, because they let you separate **host questions** from **guest answers**, and filter by guest type (founder / buyer / VC / co-packer). Most of the analytical value downstream depends on that separation. Preserve it.

One more freebie: each episode's RSS entry contains a hand-written **"Listen in as they cover:"** bulleted list in the show notes. That's ~100 episodes of human-authored topic labels — free supervision. `fetch_transcripts.py` captures the full show notes into each JSON record. Consider mining those bullets as a cheap cross-check on the clustering (see "Shortcut worth taking" below).

---

## Why this is being handed off (the blocker)

The pipeline was authored in a sandbox whose outbound network is **allowlisted to package registries only** (PyPI, npm, GitHub, Ubuntu mirrors, `api.anthropic.com`). `transistor.fm` is not on the list, so the collection step returns:

```
HTTP/2 403
x-deny-reason: host_not_allowed
```

There is a second tool (`web_fetch`) that *can* reach Transistor, but it streams content back through the model's context window one URL at a time. At ~100 episodes × ~7k words, that's ~700k words — roughly an order of magnitude too large to page through that way. And Transistor's RSS XML repeats each episode's notes three times (`<description>`, `<content:encoded>`, `<itunes:summary>` are byte-identical), so even *listing* the feed burns budget fast — a single fetch of the feed only surfaced 8 of the ~340 items before truncating.

**Net:** collection (step 1) needs an environment with normal outbound internet. Everything downstream (steps 2–5) only needs PyPI + the Anthropic API, both of which were available — so those steps are fully specified and ready. If your environment (Logic Agency's) has open network, you can run the entire thing end-to-end with no further design work.

---

## The pipeline, stage by stage

### Stage 1 — Collect  (`fetch_transcripts.py`)
Parse the RSS feed once, pull every `transcript.txt`, and write one JSON per episode to `data/episodes/`. Each record captures: `guid`, `title`, `published`, `episode` number, `duration_sec`, `link`, `host`, `show_notes` (de-HTML'd), `transcript_url`, structured `turns` (each with timestamp, speaker, text), and a `word_count`.

Design notes baked into the code:
- **The guid is the only trustworthy key.** Episode numbers disagree between the `#256`-style title prefixes and the `<podcast:episode>` tag, so filenames are derived from the guid, not the episode number.
- **Idempotent / resumable:** it skips episodes whose JSON already exists, so re-running weekly only fetches new episodes. The feed *is* the diff.
- **Polite:** 0.4s between requests, custom User-Agent.
- **Degrades instead of crashing:** if the speaker-turn regex ever fails to match (format change), it falls back to a single-blob turn rather than dropping the episode.

To limit to the **100 most recent** episodes specifically: the feed is already in reverse-chronological order, so either (a) stop after 100 successful writes, or (b) fetch all ~340 and slice by `published` date afterward. Fetching all of them is cheap and gives you more to work with; recommend (b) unless there's a reason to cap.

### Stage 2 — Chunk  (inside `build_heatmap.py`, `load_chunks()`)
Split each transcript into speaker turns, then group into ~220-word windows (roughly 1–2 min of speech). **The unit of analysis is the chunk, not the episode.** Talking points live at the paragraph level; counting per-episode collapses all the signal (every episode would score ~1 on its topics). Each chunk carries an `is_host` boolean derived by matching the speaker name against the episode's host — this is the hook for the host-vs-guest analysis later, and it's the highest-signal cut in the whole project.

### Stage 3 — Discover topics  (`discover_topics()`)
**Discover the taxonomy; don't hand-write it.** Embed guest chunks locally (`all-MiniLM-L6-v2`, free), reduce with UMAP, cluster with HDBSCAN (this is BERTopic in spirit). You get ~40–80 natural clusters. Then an LLM reads the top examples from each cluster and writes a label, then collapses near-duplicates into a final ~20–30-topic taxonomy saved to `data/artifacts/taxonomy.json`.

Rationale: hand-authored taxonomies encode the author's priors and reliably produce the *wrong* categories — you invent "influencer marketing" and miss that half the corpus is actually arguing about trade spend, deductions, and chargebacks. Reading the clusters yourself before accepting the labels is worth ~20 minutes; it's where you learn what the corpus is actually about.

### Stage 4 — Classify  (`classify()`)
With the taxonomy locked, run a cheap multi-label classification pass over **all** chunks with a fast model (`claude-haiku-4-5`). Multi-label because one chunk can hit two topics (e.g. co-packer margins = manufacturing + unit economics). For the full corpus (~20–30k chunks) move this to the **Batch API** — same prompt, ~half the cost, and you're not babysitting a loop. Empty label lists are expected for intros, ad reads, and chitchat; that's a feature.

### Stage 5 — Heatmap  (`render()`)
Three outputs:
- **`heatmap_topic_by_quarter.png`** — topic × quarter, cell = *share* of that quarter's chunks (share, not raw count — otherwise you're just plotting how many episodes shipped). This is a **trend** map: you'll see clean-label / non-UPF / GLP-1 / tariffs climbing and other subjects fading.
- **`host_vs_guest.csv`** — the gap between what the host keeps *asking* about and what guests keep *volunteering*. Probably the highest-signal artifact.
- **`cooccurrence.csv`** — which topics always travel together; reveals the industry's mental model.

---

## Important caveats about `taxonomy_v0.json`

This file was drafted by hand from **only 8 episodes** (the ones that fit in context before the collection blocker). It is a *hypothesis to test*, not a deliverable. Known likely errors, stated up front so they can be checked:

- **It has no line item for trade spend / deductions / chargebacks.** None of the 8 sample episodes touched it. Across ~100+ episodes this is very likely a top-5 topic. If the clustering surfaces it, that's a win for the clustering, not a bug.
- **"Founder origin story" will dominate any raw frequency count** and tell you almost nothing — nearly every episode has one. It's included as an explicit topic precisely so it can be *excluded/down-weighted*, not so it can inflate the map.
- **Fundraising is over-represented** (5 of 28 topics) because 3 of the 8 sample episodes were fundraising-flavored. That's sampling noise.

If you drop this file at `data/artifacts/taxonomy.json`, `build_heatmap.py` will *skip* discovery (Stage 3) and classify against it directly. **Recommended: don't.** Let Stage 3 run on the real corpus, then diff its induced taxonomy against this v0 to see where the 8-episode sample misled.

---

## "Shortcut worth taking" — the show-notes bullets

Before (or alongside) the full embedding pipeline, consider just extracting the "Listen in as they cover:" bullets from every episode's `show_notes` field and counting/clustering those. They're hand-written topic tags, already in the JSON, and may get you ~70% of the heatmap in an afternoon. Use it as a sanity check: if the bullet-based picture and the transcript-clustering picture disagree wildly, something in the clustering is off.

---

## Cost / scale sanity check

- ~340 episodes × ~40 min × ~150 wpm ≈ **~2M words ≈ ~2.7M tokens** of transcript.
- Embeddings: **free** (run locally).
- Classification via Haiku on the Batch API: **a few dollars** for the whole corpus.
- This is a cheap project. The expensive part is judgment on the taxonomy, not compute.

---

## Files in this bundle

| File | What it is |
|---|---|
| `HANDOFF.md` | This document. |
| `fetch_transcripts.py` | Stage 1. RSS → per-episode JSON with speaker-labeled turns. Needs internet to `transistor.fm`. |
| `build_heatmap.py` | Stages 2–5. Chunk → embed → cluster → LLM-label → classify → heatmap. Needs `pip` + `ANTHROPIC_API_KEY`. |
| `taxonomy_v0.json` | Hand-drafted hypothesis taxonomy (n=8). A checking tool, NOT ground truth. |

## Runbook

```bash
# 0. deps
pip install requests feedparser sentence-transformers umap-learn hdbscan \
            pandas seaborn matplotlib anthropic
export ANTHROPIC_API_KEY=...

# 1. collect  (needs open network; ~3 min)
python fetch_transcripts.py            # -> data/episodes/*.json

# 2. analyze  (discovers taxonomy, classifies, renders)
python build_heatmap.py                # -> data/artifacts/{taxonomy.json,
                                        #    heatmap_topic_by_quarter.png,
                                        #    host_vs_guest.csv, cooccurrence.csv}
```

To cap at the 100 most recent episodes, either stop `fetch_transcripts.py` after 100 writes, or fetch all and filter `data/episodes/*.json` by the `published` field before running `build_heatmap.py`.

## Open questions to confirm with the requester
- **100 vs all ~340?** The ask said "100 most recent"; the pipeline can do either. All-340 is only marginally more expensive and gives longer trend history.
- **Definition of "talking point."** Two valid readings: (a) *topics* (what subjects come up), which is what this pipeline delivers; (b) *claims/advice* ("distributor X takes Y% margin", "don't go national before velocity hits Z"). If (b) is wanted, add a claim-extraction pass in Stage 4 that pulls assertions rather than topic tags, then dedupe and count across episodes — the disagreements between guests are usually the most interesting output.
