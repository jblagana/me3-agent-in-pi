#!/usr/bin/env python3
# Boots -- Library tools: answer only from your files (2026-11-18).
#
# You point Boots at files (a path, a folder, a link) or ask it to find and
# download some. They're read, cut into short passages and stored in a
# SQLite full-text index. While a collection is "in use", every LLM call
# gets the best-matching passages for the user's latest words, and the
# model is told to answer only from them, naming the file and page.
#
# Same contract as files_tools.py: TOOL_SPECS + dispatch(), short speakable
# results. agent.py, the dashboard chat (bootsui/boots_ui.py) and
# mcp_server.py all import it.
#
# State lives on disk (BOOTS_LIBRARY, default ~/boots-library), not in
# memory: the tools usually run in the MCP subprocess, while the agent and
# the dashboard read which collection is in use.
#   ~/boots-library/index.sqlite   files + passages (FTS5)
#   ~/boots-library/state.json     {"active": "<collection>", "web": false}
#   ~/boots-library/<collection>/  downloaded copies (local files are
#                                  indexed where they are, not copied)
#
# Reading: text/Markdown/CSV/JSON as-is; HTML via web_tools' readable-text
# pass; PDF via `pdftotext` (poppler-utils), else pypdf; Word (.docx) by
# reading its XML directly (no extra package). Scanned PDFs (no text layer)
# are reported, not OCR'd.
import hashlib
import html as _html
import json
import os
import re
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request
import zipfile

LIB_ROOT = os.environ.get("BOOTS_LIBRARY") or os.path.join(os.path.expanduser("~"), "boots-library")
DB_PATH = os.path.join(LIB_ROOT, "index.sqlite")
STATE_PATH = os.path.join(LIB_ROOT, "state.json")

MAX_FILE_BYTES = int(os.environ.get("BOOTS_LIBRARY_MAX_MB", "25")) * 1024 * 1024
MAX_FILES_PER_ADD = 200
MAX_DOWNLOADS = 5
CHUNK_CHARS = 900          # one passage
CHUNK_OVERLAP = 150        # carried into the next passage
TOP_K = 6                  # passages per question (~1.5k tokens)
NOTE_CHARS = 5200          # cap on the passages block in the prompt
TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".log", ".rst"}
HTML_EXT = {".html", ".htm"}
SUPPORTED = TEXT_EXT | HTML_EXT | {".pdf", ".docx"}
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".cache"}

_STOP = set("""a an and are as at be been but by can could did do does for from had has
have how i if in into is it its me my of on or our please say says said should so
than that the their them then there these they this to was we were what when where
which who whom why will with would you your about tell does did any all also just
okay ok hey boots file files document documents doc docs page""".split())


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
def _db():
    os.makedirs(LIB_ROOT, exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.execute("""create table if not exists files(
        id integer primary key, collection text, name text, path text,
        sha text, pages integer, chars integer, added real, source text)""")
    c.execute("""create virtual table if not exists chunks using fts5(
        text, collection unindexed, file_id unindexed, page unindexed,
        tokenize='porter unicode61')""")
    return c


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:40] or "my-files"


def get_state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except Exception:
        return {}


def _set_state(**kw) -> None:
    st = get_state()
    st.update(kw)
    os.makedirs(LIB_ROOT, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f)
    os.replace(tmp, STATE_PATH)


def active_collection() -> str:
    return get_state().get("active") or ""


def web_blocked() -> bool:
    """True while files-only mode is on and the user hasn't allowed the web."""
    st = get_state()
    return bool(st.get("active")) and not st.get("web")


# ---------------------------------------------------------------------------
# reading files -> [(page_no, text)]
# ---------------------------------------------------------------------------
def _read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [(1, f.read())]


def _html_text(raw: str) -> str:
    try:
        import web_tools
        return web_tools._readable_text(raw)
    except Exception:
        t = re.sub(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", raw, flags=re.S | re.I)
        return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()


def _read_html(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [(1, _html_text(f.read()))]


def _read_pdf(path):
    try:
        out = subprocess.run(["pdftotext", "-enc", "UTF-8", path, "-"],
                             capture_output=True, timeout=120)
        if out.returncode == 0:
            pages = out.stdout.decode("utf-8", "replace").split("\f")
            return [(i + 1, p) for i, p in enumerate(pages) if p.strip()] or [(1, "")]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("I can't read PDFs yet: poppler-utils or pypdf needs installing")
    r = PdfReader(path)
    return [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(r.pages)]


def _read_docx(path):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    paras = []
    for p in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        t = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
        if "<w:tab/>" in p and not t:
            continue
        t = _html.unescape(t).strip()
        if t:
            paras.append(t)
    # Word has no fixed pages; ~3000 characters is about one printed page.
    pages, cur, n = [], [], 0
    for t in paras:
        cur.append(t)
        n += len(t)
        if n >= 3000:
            pages.append("\n\n".join(cur))
            cur, n = [], 0
    if cur:
        pages.append("\n\n".join(cur))
    return [(i + 1, p) for i, p in enumerate(pages)] or [(1, "")]


def extract(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXT:
        return _read_text(path)
    if ext in HTML_EXT:
        return _read_html(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    raise ValueError(f"I can't read {ext or 'that kind of'} files yet")


def chunk_pages(pages):
    """[(page, text)] -> [(page, passage)], ~CHUNK_CHARS each, slight overlap."""
    out = []
    for page, text in pages:
        text = re.sub(r"[ \t]+", " ", text or "")
        paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z0-9•\-])", text) if p.strip()]
        buf = ""
        for p in paras:
            while len(p) > CHUNK_CHARS:              # one huge paragraph
                cut = p.rfind(". ", 0, CHUNK_CHARS)
                cut = cut + 1 if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
                if buf:
                    out.append((page, buf.strip()))
                    buf = ""
                out.append((page, p[:cut].strip()))
                p = p[max(0, cut - CHUNK_OVERLAP):]
            if len(buf) + len(p) + 1 > CHUNK_CHARS and buf:
                out.append((page, buf.strip()))
                buf = buf[-CHUNK_OVERLAP:] + " " + p
            else:
                buf = (buf + "\n" + p) if buf else p
        if buf.strip():
            out.append((page, buf.strip()))
    return [(pg, t) for pg, t in out if len(t) > 20]


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------
def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def index_file(path, collection, source="", display_name=""):
    """Index one file. Returns (status, name, pages, note): status is
    'added', 'same' (unchanged, skipped), 'empty' or 'error'."""
    name = display_name or os.path.basename(path)
    try:
        if os.path.getsize(path) > MAX_FILE_BYTES:
            return ("error", name, 0, f"bigger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
        sha = _sha(path)
        c = _db()
        try:
            row = c.execute("select id, sha from files where collection=? and path=?",
                            (collection, path)).fetchone()
            if row and row[1] == sha:
                return ("same", name, 0, "")
            pages = extract(path)
            chunks = chunk_pages(pages)
            chars = sum(len(t) for _, t in pages)
            if not chunks:
                note = ("it looks scanned (no text layer); I can't read scans yet"
                        if path.lower().endswith(".pdf") else "no readable text")
                return ("empty", name, len(pages), note)
            if row:
                c.execute("delete from chunks where file_id=?", (str(row[0]),))
                c.execute("delete from files where id=?", (row[0],))
            cur = c.execute("insert into files(collection,name,path,sha,pages,chars,added,source)"
                            " values(?,?,?,?,?,?,?,?)",
                            (collection, name, path, sha, len(pages), chars, time.time(), source))
            fid = str(cur.lastrowid)
            c.executemany("insert into chunks(text,collection,file_id,page) values(?,?,?,?)",
                          [(t, collection, fid, pg) for pg, t in chunks])
            c.commit()
            pdf_thin = path.lower().endswith(".pdf") and chars < 80 * len(pages)
            return ("added", name, len(pages),
                    "some pages look scanned and were skipped" if pdf_thin else "")
        finally:
            c.close()
    except ValueError as e:
        return ("error", name, 0, str(e))
    except Exception as e:
        return ("error", name, 0, f"couldn't read it ({type(e).__name__})")


def _home_path(path):
    import files_tools
    return files_tools._resolve(path)        # locked inside ~/boots-files (2026-11-18)


def _summarise(results, collection):
    added = [r for r in results if r[0] == "added"]
    same = [r for r in results if r[0] == "same"]
    bad = [r for r in results if r[0] in ("error", "empty")]
    parts = []
    if added:
        pages = sum(r[2] for r in added)
        names = ", ".join(r[1] for r in added[:4]) + (f" and {len(added) - 4} more" if len(added) > 4 else "")
        parts.append(f"Added {len(added)} file{'s' if len(added) != 1 else ''} to '{collection}' "
                     f"({pages} page{'s' if pages != 1 else ''}): {names}.")
    if same:
        parts.append(f"{len(same)} already in it, unchanged.")
    for r in bad[:3]:
        parts.append(f"Skipped {r[1]}: {r[3]}.")
    if len(bad) > 3:
        parts.append(f"And {len(bad) - 3} more skipped.")
    notes = [f"{r[1]}: {r[3]}" for r in added if r[3]]
    if notes:
        parts.append("Note: " + "; ".join(notes[:2]) + ".")
    return " ".join(parts) or "Nothing was added."


def _activate_if_added(results, collection):
    if any(r[0] in ("added", "same") for r in results):
        _set_state(active=collection, web=False)
        return (f" Files-only mode is on: I'll answer from '{collection}' until you say "
                "stop using the files.")
    return ""


# ---------------------------------------------------------------------------
# downloading
# ---------------------------------------------------------------------------
_PRIVATE = re.compile(r"^(localhost|127\.|0\.|10\.|192\.168\.|169\.254\.|172\.(1[6-9]|2\d|3[01])\.|\[?::1)")


def _public_url(u: str) -> str:
    u = (u or "").strip()
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    host = urllib.parse.urlparse(u).hostname or ""
    if not host or _PRIVATE.match(host):
        raise ValueError("I can only download from public websites")
    return u


def download(url, collection):
    """Download one public URL into the collection folder. Returns the path."""
    u = _public_url(url)
    try:
        import web_tools
        ua = web_tools.USER_AGENT
    except Exception:
        ua = "BootsVoiceAgent/1.0"
    req = urllib.request.Request(u, headers={"User-Agent": ua, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=20) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        data = r.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"it's bigger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    path_part = urllib.parse.urlparse(u).path
    base = os.path.basename(path_part) or urllib.parse.urlparse(u).hostname
    stem, ext = os.path.splitext(base)
    if data[:5] == b"%PDF-" or "pdf" in ctype:
        ext = ".pdf"
    elif "wordprocessingml" in ctype:
        ext = ".docx"
    elif ext.lower() not in SUPPORTED:
        ext = ".html" if ("html" in ctype or b"<html" in data[:2000].lower()) else ".txt"
    if ext == ".html":
        text = _html_text(data.decode("utf-8", "replace"))
        if len(text) < 200:
            raise ValueError("the page had almost no readable text")
    folder = os.path.join(LIB_ROOT, collection)
    os.makedirs(folder, exist_ok=True)
    fname = (re.sub(r"[^A-Za-z0-9._-]+", "-", stem)[:60].strip("-") or "page") + ext
    path = os.path.join(folder, fname)
    k = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{os.path.splitext(fname)[0]}-{k}{ext}")
        k += 1
    with open(path, "wb") as f:
        f.write(data)
    return path


def _search_urls(topic, n):
    """URLs for a topic: Tavily if there's a key, else Wikipedia search."""
    urls = []
    try:
        import web_tools
        key = web_tools._tavily_key()
    except Exception:
        web_tools, key = None, ""
    if key:
        for q in (f"{topic} filetype:pdf", topic):
            try:
                req = urllib.request.Request(
                    "https://api.tavily.com/search",
                    data=json.dumps({"api_key": key, "query": q, "max_results": n + 2,
                                     "search_depth": "basic"}).encode(),
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=12) as r:
                    for res in json.loads(r.read().decode("utf-8", "replace")).get("results") or []:
                        u = (res.get("url") or "").strip()
                        if u and u not in urls:
                            urls.append(u)
            except Exception:
                pass
            if len(urls) >= n:
                break
    if len(urls) < n and web_tools is not None:
        body = web_tools._get("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="
                              + urllib.parse.quote(topic) + "&srlimit=5&format=json")
        # Wikipedia's full-text search returns loose matches ("solar dryer for
        # mangoes" -> LG Electronics, Sugarcane): keep a page only when its
        # title + snippet hold at least half of the topic's keywords.
        kws = set(_keywords(topic))
        try:
            for hit in (json.loads(body).get("query") or {}).get("search") or []:
                words = set(_keywords(hit.get("title", "") + " " + web_tools._strip_html(hit.get("snippet", ""))))
                if kws and len(kws & words) < max(1, (len(kws) + 1) // 2):
                    continue
                u = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(hit["title"].replace(" ", "_"))
                if u not in urls:
                    urls.append(u)
        except Exception:
            pass
    # PDFs first: they're usually the actual documents.
    urls.sort(key=lambda u: 0 if u.lower().split("?")[0].endswith(".pdf") else 1)
    return urls


def _stage(key, status, say=""):
    try:
        import web_tools
        web_tools._stage("find_and_download", key, status, say)
    except Exception:
        pass


def _wiki_text_file(url, collection):
    """Wikipedia pages: save the plain-text extract (the HTML is mostly menus)."""
    m = re.match(r"https?://([a-z\-]+)\.(?:m\.)?wikipedia\.org/wiki/([^?#]+)", url)
    if not m:
        return None
    import web_tools
    body = web_tools._get(f"https://{m.group(1)}.wikipedia.org/w/api.php?action=query&prop=extracts"
                          f"&explaintext=1&redirects=1&format=json&titles={m.group(2)}")
    pages = (json.loads(body).get("query") or {}).get("pages") or {}
    text = "\n\n".join((p.get("extract") or "") for p in pages.values()).strip()
    if len(text) < 200:
        return None
    folder = os.path.join(LIB_ROOT, collection)
    os.makedirs(folder, exist_ok=True)
    title = urllib.parse.unquote(m.group(2))
    path = os.path.join(folder, re.sub(r"[^A-Za-z0-9._-]+", "-", title)[:60] + "-wikipedia.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(title.replace("_", " ") + "\n\n" + text)
    return path


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def _keywords(text):
    words = re.findall(r"[A-Za-z0-9À-ɏ]+", (text or "").lower())
    out = []
    for w in words:
        if (len(w) > 1 or w.isdigit()) and w not in _STOP and w not in out:
            out.append(w)
    return out[:16]


def search(question, collection=None, k=TOP_K, also=""):
    """Best passages for a question: [(file name, page, text, coverage)].
    coverage = share of the question's keywords found in the passage.
    `also` (the previous user message) is used when the question itself
    has fewer than 2 keywords ("what about page 3?")."""
    coll = collection or active_collection()
    if not coll:
        return []
    kws = _keywords(question)
    if len(kws) < 2 and also:
        kws += [w for w in _keywords(also) if w not in kws]
    if not kws:
        return []
    q = " OR ".join('"%s"' % w.replace('"', "") for w in kws)
    c = _db()
    try:
        rows = c.execute(
            "select c.text, c.page, f.name, bm25(chunks) as s from chunks c "
            "join files f on f.id = cast(c.file_id as integer) "
            "where chunks match ? and c.collection = ? order by s limit ?",
            (q, coll, k * 3)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        c.close()
    out = []
    for text, page, name, _s in rows:
        low = text.lower()
        cov = sum(1 for w in kws if w in low) / len(kws)
        out.append((name, int(page or 1), text, cov))
    # bm25 order, but drop near-misses when better passages exist
    best = max((r[3] for r in out), default=0)
    out = [r for r in out if r[3] >= best * 0.5][:k]
    return out


def _cite(name, page):
    return f"{name}, page {page}" if name.lower().endswith((".pdf", ".docx")) else name


def grounding_note(question, also=""):
    """System-prompt block for files-only mode, or '' when it's off.
    Added to EVERY LLM call while a collection is in use."""
    coll = active_collection()
    if not coll:
        return ""
    hits = search(question, coll, also=also)
    head = (f"\n\nFILES-ONLY MODE is on (collection '{coll}'). Answer the user's questions ONLY from "
            "the passages below. Name the source in a few words (for example 'your thesis.pdf, page "
            "12 says...'). If the passages don't contain the answer, say your files don't cover "
            "that and offer to look it up on the web; do not answer from general knowledge or "
            "guess, even if you know the answer. Web search is off in this mode"
            + (" (the user allowed it: you may use it, and label what came from the web)."
               if get_state().get("web") else
               "; if the user agrees to use the web, call use_collection with allow_web=true.")
            + " Device commands (music, lights, volume, timers, alarms, reminders) and the library"
            " tools still work as usual.")
    good = [h for h in hits if h[3] >= 0.34 or (len(_keywords(question)) <= 2 and h[3] > 0)]
    if not good:
        return head + ("\nNo passage in the files matches the user's latest message. If it is a "
                       "question, say the files don't cover it; if it's a command, just do it.")
    lines, used = [], 0
    for i, (name, page, text, _cov) in enumerate(good, 1):
        t = re.sub(r"\s+", " ", text)
        if used + len(t) > NOTE_CHARS:
            t = t[:max(0, NOTE_CHARS - used)]
        if not t:
            break
        lines.append(f"[{i}] {_cite(name, page)}: {t}")
        used += len(t)
    return head + "\nPassages for the user's latest message:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def _tool_add_files(path="", collection=""):
    p = (path or "").strip()
    if not p:
        return "Tell me which file, folder or link to add."
    coll = _slug(collection or active_collection() or "my files")
    if re.match(r"^(https?://|www\.)", p, re.I):
        try:
            fp = _wiki_text_file(_public_url(p), coll) or download(p, coll)
        except ValueError as e:
            return f"I couldn't add that link: {e}."
        except Exception as e:
            return f"I couldn't download that link ({type(e).__name__})."
        res = [index_file(fp, coll, source=p)]
        return _summarise(res, coll) + _activate_if_added(res, coll)
    try:
        ap = _home_path(p)
    except ValueError as e:
        return str(e)
    if os.path.isdir(ap):
        files = []
        for dp, dns, fns in os.walk(ap):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".")]
            for fn in sorted(fns):
                if os.path.splitext(fn)[1].lower() in SUPPORTED:
                    files.append(os.path.join(dp, fn))
        if not files:
            return f"There are no files I can read in {os.path.basename(ap)} (I read PDF, Word, text, Markdown, HTML and CSV)."
        more = len(files) - MAX_FILES_PER_ADD
        res = [index_file(f, coll) for f in files[:MAX_FILES_PER_ADD]]
        out = _summarise(res, coll)
        if more > 0:
            out += f" I stopped at {MAX_FILES_PER_ADD} files; {more} more were left out."
        return out + _activate_if_added(res, coll)
    if not os.path.isfile(ap):
        return f"I can't find {p}."
    res = [index_file(ap, coll)]
    return _summarise(res, coll) + _activate_if_added(res, coll)


def _tool_find_and_download(topic="", collection="", max_files=3):
    t = (topic or "").strip()
    if not t:
        return "Tell me what the files should be about."
    try:
        n = max(1, min(MAX_DOWNLOADS, int(max_files or 3)))
    except (TypeError, ValueError):
        n = 3
    coll = _slug(collection or t)
    _stage("search", f"searching the web for files about {t}")
    urls = _search_urls(t, n)
    if not urls:
        return f"I couldn't find any files about {t} online."
    res, tried = [], 0
    for u in urls:
        if sum(1 for r in res if r[0] in ("added", "same")) >= n or tried >= n + 4:
            break
        tried += 1
        _stage(f"dl{tried}", f"downloading {urllib.parse.urlparse(u).hostname}")
        try:
            fp = _wiki_text_file(u, coll) or download(u, coll)
            res.append(index_file(fp, coll, source=u))
        except Exception as e:
            res.append(("error", urllib.parse.urlparse(u).hostname or u, 0,
                        str(e) if isinstance(e, ValueError) else "download failed"))
    ok = [r for r in res if r[0] in ("added", "same")]
    if not ok:
        return f"I found links about {t} but couldn't download any readable files."
    out = _summarise([r for r in res if r[0] != "error"] + [r for r in res if r[0] == "error"][:1], coll)
    return out + _activate_if_added(res, coll)


def _tool_list_library():
    c = _db()
    try:
        rows = c.execute("select collection, count(*), sum(pages) from files group by collection "
                         "order by max(added) desc").fetchall()
    finally:
        c.close()
    act = active_collection()
    if not rows:
        return "Your library is empty. Point me at a file or folder, or ask me to find files on a topic."
    parts = [f"'{c}' ({n} file{'s' if n != 1 else ''}, {p or 0} pages)" for c, n, p in rows[:8]]
    out = "Your library has " + ", ".join(parts) + "."
    out += (f" I'm answering from '{act}' right now." if act else " Files-only mode is off.")
    return out


def _tool_list_collection(collection=""):
    coll = _slug(collection) if collection else active_collection()
    if not coll:
        return "No collection is in use. Say which one."
    c = _db()
    try:
        rows = c.execute("select name, pages from files where collection=? order by added",
                         (coll,)).fetchall()
    finally:
        c.close()
    if not rows:
        return f"There's no collection called '{coll}'."
    names = ", ".join(f"{n} ({p} p.)" if n.lower().endswith((".pdf", ".docx")) else n
                      for n, p in rows[:15])
    return f"'{coll}' has {len(rows)} files: {names}" + (" and more." if len(rows) > 15 else ".")


def _tool_use_collection(name="", allow_web=False):
    coll = _slug(name)
    c = _db()
    try:
        n = c.execute("select count(*) from files where collection=?", (coll,)).fetchone()[0]
        if not n:
            have = [r[0] for r in c.execute("select distinct collection from files")]
    finally:
        c.close()
    if not n:
        return (f"There's no collection called '{coll}'."
                + (f" You have: {', '.join(have[:6])}." if have else " Your library is empty."))
    web = str(allow_web).lower() in ("true", "1", "yes")
    _set_state(active=coll, web=web)
    return (f"Files-only mode is on: I'll answer from '{coll}' ({n} files)"
            + (", and I may use the web too, labelled as such." if web else ".")
            + " Say stop using the files to turn it off.")


def _tool_stop_using_files():
    was = active_collection()
    _set_state(active="", web=False)
    return (f"Done, I'm no longer limited to '{was}'." if was else "Files-only mode was already off.")


def _tool_ask_files(question="", collection=""):
    coll = _slug(collection) if collection else active_collection()
    if not coll:
        return "No collection is in use. Say which one to search."
    hits = search(question, coll)
    if not hits:
        return f"Nothing in '{coll}' matches that."
    return "Passages:\n" + "\n".join(
        "[%d] %s: %s" % (i, _cite(n, p), re.sub(r"\s+", " ", t)[:700])
        for i, (n, p, t, _c) in enumerate(hits, 1))


def _tool_remove_files(name="", collection=""):
    coll = _slug(collection) if collection else active_collection()
    target = (name or "").strip()
    if not coll:
        return "Say which collection."
    c = _db()
    try:
        if target.lower() in ("", "all", "everything", coll):
            rows = c.execute("select id, path from files where collection=?", (coll,)).fetchall()
        else:
            rows = c.execute("select id, path from files where collection=? and lower(name) like ?",
                             (coll, "%" + target.lower() + "%")).fetchall()
        if not rows:
            return f"I couldn't find {target or 'any files'} in '{coll}'."
        for fid, path in rows:
            c.execute("delete from chunks where file_id=?", (str(fid),))
            c.execute("delete from files where id=?", (fid,))
            # only delete copies Boots downloaded; your own files stay on disk
            if os.path.normpath(path).startswith(os.path.normpath(os.path.join(LIB_ROOT, coll)) + os.sep):
                try:
                    os.remove(path)
                except OSError:
                    pass
        c.commit()
        left = c.execute("select count(*) from files where collection=?", (coll,)).fetchone()[0]
    finally:
        c.close()
    if not left and active_collection() == coll:
        _set_state(active="", web=False)
    return (f"Removed {len(rows)} file{'s' if len(rows) != 1 else ''} from '{coll}'."
            + ("" if left else " The collection is empty now, so files-only mode is off."))


TOOL_SPECS = [
    ("add_files", "Add a file, a folder, or a web link to the user's library so you can answer "
                  "from it. Turns on files-only mode for that collection.",
     {"path": "string", "collection": "string"}),
    ("find_and_download", "Search the web for documents on a topic, download up to max_files "
                          "(default 3, max 5) into a collection and turn on files-only mode.",
     {"topic": "string", "collection": "string", "max_files": "integer"}),
    ("list_library", "List the user's file collections and which one is in use.", {}),
    ("list_collection", "List the files in a collection (default: the one in use).",
     {"collection": "string"}),
    ("use_collection", "Answer only from this collection of files from now on. allow_web=true "
                       "also allows web search.",
     {"name": "string", "allow_web": "boolean"}),
    ("stop_using_files", "Turn off files-only mode.", {}),
    ("ask_files", "Search a collection for passages about a question (files-only mode already "
                  "adds passages automatically; use this for a different collection or a "
                  "second search).",
     {"question": "string", "collection": "string"}),
    ("remove_files", "Remove a file (or name='all' for the whole collection) from the library.",
     {"name": "string", "collection": "string"}),
]

OPTIONAL_PARAMS = {
    "add_files": ("collection",),
    "find_and_download": ("collection", "max_files"),
    "list_collection": ("collection",),
    "use_collection": ("allow_web",),
    "ask_files": ("collection",),
    "remove_files": ("collection",),
}

NAMES = {n for n, _d, _p in TOOL_SPECS}

_DISPATCH = {
    "add_files": lambda **kw: _tool_add_files(kw.get("path", ""), kw.get("collection", "")),
    "find_and_download": lambda **kw: _tool_find_and_download(
        kw.get("topic", ""), kw.get("collection", ""), kw.get("max_files", 3)),
    "list_library": lambda **kw: _tool_list_library(),
    "list_collection": lambda **kw: _tool_list_collection(kw.get("collection", "")),
    "use_collection": lambda **kw: _tool_use_collection(kw.get("name", ""), kw.get("allow_web", False)),
    "stop_using_files": lambda **kw: _tool_stop_using_files(),
    "ask_files": lambda **kw: _tool_ask_files(kw.get("question", ""), kw.get("collection", "")),
    "remove_files": lambda **kw: _tool_remove_files(kw.get("name", ""), kw.get("collection", "")),
}


def dispatch(tool: str, **kw) -> str:
    fn = _DISPATCH.get(tool)
    if fn is None:
        return f"I don't know how to do '{tool}'."
    try:
        return fn(**kw)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Something went wrong with the library: {type(e).__name__}."
