# Plan: Boots answers only from your files

Goal: you point Boots at files (a folder, an upload, a link), or ask it to find and download them. Boots reads them and then answers **only from those files**, says where each answer came from (file and page), and says "the files don't say" when they don't.

Status: steps 1–3 built and deployed on 2026-11-18 (see HANDOVER.md §8.5). Steps 4–5 are not started.

## 1. What the Pi can handle

The thinking happens on Groq, in the cloud. The Pi only has to read files, cut them into passages, store them and find the right passages for each question. A Pi 4 is enough for that.

Measured on the Pi today:

- Raspberry Pi 4 with 4 cores, 7.6 GB RAM (about 7 GB free) and 48 GB of free disk.
- In the agent's Python: `numpy` and `onnxruntime` are installed. SQLite full-text search (FTS5) works.
- Not installed yet: `pypdf`, `python-docx`, `tesseract`, `pdftotext`.

| Capability | On the Pi? | Notes |
|---|---|---|
| Text, Markdown, HTML, CSV | Yes, easily | Standard library; reuse `web_tools._readable_text` for HTML. |
| PDFs with real text | Yes | `pdftotext` (apt `poppler-utils`) does about 1–3 s for a 100-page PDF; `pypdf` is the pure-Python fallback. |
| Word, PowerPoint, Excel | Yes | `python-docx`, `python-pptx`, `openpyxl`; all small. |
| Keyword search over passages | Yes, fast | SQLite FTS5: under 50 ms even for thousands of documents. |
| Meaning-based search (embeddings) | Yes, but slow to index | A small ONNX model (about 35–130 MB) runs on `onnxruntime`. Indexing is roughly 10–40 passages a second, so a 100-page PDF takes about 10–40 s, in the background. One question takes under 0.1 s. |
| Scanned PDFs and photos (OCR) | Possible, slow | `tesseract` takes about 5–15 s a page. Fine for a few pages; for a long scan, send it to DGX3. |
| Understanding pictures, charts, diagrams | No | Needs a vision model. Captions and text around a figure are still used. |
| Audio and video files | Not locally | Could go through Deepgram (already paid for) as a later add-on. |
| Very large libraries | Yes, up to roughly 10,000 documents | Disk and RAM are not the limit; indexing time is. |

What actually limits answers isn't the Pi:

- **Groq cost per question.** Each grounded answer sends about 6 passages (around 2,000–3,000 tokens) with the question. That's several times a normal chat turn, so the number of passages is capped.
- **Voice length.** Spoken replies stay 1–2 sentences. Longer answers go to the chat panel, with sources.
- **"Summarise this whole book"** can't fit in one call. Boots would summarise section by section in the background (several Groq calls) and report back, like a website build.

## 2. Build steps

### Step 1. Library and reading files (`library_tools.py`, new)

- Files live in `~/boots-library/<collection>/`, for example `~/boots-library/thesis/`. A collection is one set of files you're asking about.
- When a file is added: pull out the text, split it into passages of about 800 characters that overlap slightly, and store them in `~/boots-library/index.sqlite` with the file name, page number and heading.
- Re-adding a file that hasn't changed is skipped, using a hash of its contents.
- Same shape as the other tool modules (`TOOL_SPECS` + `dispatch()`), so `agent.py`, the dashboard chat and `mcp_server.py` all pick it up.
- Paths stay locked inside the home folder, as `files_tools._resolve` does now.

### Step 2. Tools Boots can call

- `add_files(path_or_url, collection)` adds a file, a folder or a link.
- `find_and_download(topic, collection, max_files)` uses `web_search` to find PDFs and pages, downloads them and adds them. It runs as a background job ("On it, I'll tell you when they're in") through the existing `ME3_BG_TOOLS` system.
- `list_library()` lists the collections, how many files each has, and which collection is in use.
- `use_collection(name)` starts answering from that collection; `stop_using_files()` ends it.
- `ask_files(question)` returns the best passages, each tagged with its file and page.
- `remove_files(name)`.

### Step 3. Answer-only-from-files mode

While a collection is in use:

- Every question first goes through `ask_files`. The code does this, not the model, so it can't be skipped.
- The prompt gains a short addendum: answer only from the passages given; name the file and page; if the passages don't answer it, say "your files don't cover that" and offer to search the web. Never fill gaps from general knowledge.
- Web search and fetch are switched off while the mode is on, unless you say "check the web too".
- If no passage scores above a threshold, Boots answers "the files don't say" **without calling Groq**. That saves a call and stops it guessing.
- Music, lights, timers and other device commands keep working.
- The dashboard header shows "Answering from: thesis", so it's clear the mode is on.

### Step 4. Better search (optional; after step 3 works)

- Add the small embedding model and combine its results with keyword search. That finds "how much did it cost" in a passage that says "the budget was ₱2M".
- `bge-small-en` covers English only. For Tagalog or Taglish files, use `multilingual-e5-small` (about 120 MB).
- Index in the background, and fall back to keyword search until a file's embeddings are done.

### Step 5. Dashboard

- A Library card: drag-and-drop upload (`POST /library/upload`), the list of collections and files with their status (reading, ready, failed), and a button to use a collection.
- Chat replies show source chips ("thesis.pdf p. 12"); clicking one opens that passage.

### Step 6. Tests

- Offline tests in `tests/` (no Groq calls): reading each file type, splitting, search ranking, the "files don't say" threshold, and the path lock.
- Live check: about 5–8 Groq calls in total. One question the files answer, one they don't, one download job, and one device command while the mode is on.

## 3. Decisions for you

1. **How strict?** Strict means only from the files, plus device commands. Lenient means files first, and Boots may add general knowledge if it labels it as such.
2. **Which file types matter first?** PDFs and Word are assumed.
3. **Scanned documents?** If yes, install `tesseract` now or send them to DGX3.
4. **Language of the files:** English only, or Tagalog too? This decides the embedding model.
5. **Downloads:** any site, or only PDFs and pages from search results? Is there a size cap per file (for example 25 MB)?

## 4. Order and size

Steps 1–3 are the useful core and need no new hardware: about a day of work, 0 Groq calls until the live check. Steps 4–5 come after you've tried it. Step 6 runs alongside each step.
