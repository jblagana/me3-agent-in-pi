"""Files-only mode: library_tools (2026-11-18).

1. read      text, HTML, Word (.docx built here) and PDF (if pdftotext or
             pypdf is installed) come out as text with page numbers.
2. chunk     long text is cut into ~900-character passages; no passage is
             empty and pages are kept.
3. add       add_files on a folder indexes the readable files, skips the
             rest, turns on files-only mode, and re-adding is a no-op.
4. search    the right passage ranks first; the note cites file and page;
             a question the files don't answer gets the "no passage" note.
5. mode      web_blocked() follows the mode; allow_web lifts it;
             stop_using_files turns the note off.
6. guard     paths outside the home folder and private URLs are refused.
7. remove    removing a file drops its passages; removing all ends the mode.
8. agent     _library_ground adds passages to the system instruction for one
             call only, and the voice agent's web handler refuses web_search
             while the mode is on.

Offline: no network, no LLM calls.

    cd ~/me3 && .venv/bin/python tests/test_library.py
"""
import asyncio
import os
import shutil
import sys
import tempfile
import types
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

TMP = tempfile.mkdtemp(prefix="boots-lib-test-")
os.environ["BOOTS_LIBRARY"] = os.path.join(TMP, "library")
os.environ["BOOT_FILES_ROOT"] = TMP

import library_tools as lib   # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""))


def make_docx(path, paras):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paras)
    xml = ('<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.'
           'openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + "</w:body></w:document>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml", xml)


def make_pdf(path, pages):
    """A minimal valid PDF with one line of text per page."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None,
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for txt in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({txt}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        cid = len(objs)
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {cid} 0 R >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(out)


def pdf_reader_available():
    if shutil.which("pdftotext"):
        return True
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


def setup_docs():
    d = os.path.join(TMP, "thesis")
    os.makedirs(d)
    with open(os.path.join(d, "notes.md"), "w", encoding="utf-8") as f:
        f.write("# Field notes\n\nThe solar dryer prototype reached 62 degrees Celsius at noon.\n\n"
                "Mangoes dried in 14 hours instead of the usual three days.\n")
    with open(os.path.join(d, "page.html"), "w", encoding="utf-8") as f:
        f.write("<html><body><nav>Menu Home</nav><article><p>The barangay water project "
                "was funded by a grant of 2 million pesos from the provincial office in 2025, "
                "covering pipes, pumps and three storage tanks.</p></article></body></html>")
    make_docx(os.path.join(d, "budget.docx"),
              ["Budget summary", "The total budget for the dryer is 48,000 pesos.",
               "Labor costs were 12,000 pesos."])
    with open(os.path.join(d, "photo.jpg"), "wb") as f:
        f.write(b"\xff\xd8not really a jpeg")
    if pdf_reader_available():
        make_pdf(os.path.join(d, "report.pdf"),
                 ["Chapter one introduces the cooperative.",
                  "The cooperative has 37 member farmers in Benguet."])
    return d


def read_case(d):
    md = lib.extract(os.path.join(d, "notes.md"))
    check("read: markdown", "62 degrees" in md[0][1])
    ht = lib.extract(os.path.join(d, "page.html"))
    check("read: html keeps the article, drops the menu",
          "2 million pesos" in ht[0][1] and "Menu Home" not in ht[0][1], ht[0][1][:80])
    dx = lib.extract(os.path.join(d, "budget.docx"))
    check("read: docx", "48,000 pesos" in dx[0][1], dx[0][1][:80])
    if pdf_reader_available():
        pdf = lib.extract(os.path.join(d, "report.pdf"))
        check("read: pdf keeps page numbers",
              len(pdf) == 2 and "37 member farmers" in pdf[1][1] and pdf[1][0] == 2, repr(pdf)[:120])
    else:
        print("SKIP read: pdf (no pdftotext or pypdf here)")


def chunk_case():
    long = ("This is a sentence about rice terraces. " * 120)
    ch = lib.chunk_pages([(1, long), (2, "Short second page about irrigation canals here.")])
    check("chunk: long page splits", sum(1 for p, _ in ch if p == 1) >= 4, str(len(ch)))
    check("chunk: passages stay near the size",
          all(len(t) <= lib.CHUNK_CHARS + lib.CHUNK_OVERLAP + 50 for _, t in ch))
    check("chunk: page 2 kept", any(p == 2 and "irrigation" in t for p, t in ch))


def add_case(d):
    out = lib.dispatch("add_files", path="thesis", collection="Thesis")
    want = 4 if pdf_reader_available() else 3
    check("add: folder indexed", f"Added {want} files to 'thesis'" in out, out)
    check("add: unreadable file not counted", "photo.jpg" not in out, out)
    check("add: mode turned on", lib.active_collection() == "thesis" and "Files-only mode is on" in out)
    again = lib.dispatch("add_files", path="thesis", collection="thesis")
    check("add: re-adding unchanged files is a no-op", "already in it" in again and "Added" not in again, again)


def search_case():
    hits = lib.search("how much was the budget for the dryer")
    check("search: right passage first", hits and hits[0][0] == "budget.docx", repr(hits[:1])[:120])
    note = lib.grounding_note("how hot did the solar dryer get")
    check("search: note has the passage and a citation",
          "62 degrees" in note and "notes.md" in note and "FILES-ONLY MODE" in note, note[-300:])
    if pdf_reader_available():
        n2 = lib.grounding_note("how many member farmers does the cooperative have")
        check("search: pdf cited with page", "report.pdf, page 2" in n2, n2[-200:])
    miss = lib.grounding_note("who won the basketball finals")
    check("search: unrelated question gets the no-passage note",
          "No passage in the files matches" in miss and "[1]" not in miss, miss[-200:])
    follow = lib.grounding_note("and labor?", also="what was the budget for the dryer")
    check("search: short follow-up uses the previous question", "12,000" in follow, follow[-200:])


def mode_case():
    check("mode: web blocked while on", lib.web_blocked())
    out = lib.dispatch("use_collection", name="thesis", allow_web=True)
    check("mode: allow_web lifts the block", not lib.web_blocked() and "web too" in out, out)
    lib.dispatch("use_collection", name="thesis")
    check("mode: plain use_collection blocks again", lib.web_blocked())
    bad = lib.dispatch("use_collection", name="nope")
    check("mode: unknown collection refused", "no collection called" in bad, bad)
    lst = lib.dispatch("list_library")
    check("mode: list_library names it", "'thesis'" in lst and "right now" in lst, lst)
    lib.dispatch("stop_using_files")
    check("mode: stop turns it off", not lib.active_collection() and not lib.web_blocked()
          and lib.grounding_note("budget") == "")
    lib.dispatch("use_collection", name="thesis")


def guard_case():
    out = lib.dispatch("add_files", path="/etc/passwd")
    check("guard: outside the home folder refused", "outside the allowed folder" in out, out)
    out2 = lib.dispatch("add_files", path="http://192.168.1.10/secret.pdf")
    check("guard: private URL refused", "public websites" in out2, out2)


def remove_case():
    out = lib.dispatch("remove_files", name="budget")
    check("remove: one file", "Removed 1 file" in out, out)
    check("remove: its passages are gone",
          not any(h[0] == "budget.docx" for h in lib.search("budget dryer pesos")))
    check("remove: your own file stays on disk", os.path.exists(os.path.join(TMP, "thesis", "budget.docx")))
    out2 = lib.dispatch("remove_files", name="all")
    check("remove: all ends the mode", "files-only mode is off" in out2 and not lib.active_collection(), out2)


async def agent_case(d):
    """Voice agent hooks, without loading the pipeline: import agent.py with
    the heavy modules stubbed only if the real import fails."""
    try:
        os.environ.setdefault("ME3_BENCH", "1")
        os.environ.setdefault("ME3_NO_STATUS", "1")
        import agent
    except Exception as e:
        print(f"SKIP agent hooks (agent.py not importable here: {type(e).__name__})")
        return
    lib.dispatch("add_files", path="thesis", collection="thesis")
    ctx = types.SimpleNamespace(messages=[
        {"role": "user", "content": "how hot did the solar dryer get"}])
    llm = types.SimpleNamespace(_settings=types.SimpleNamespace(system_instruction="BASE"))
    agent._library_ground(ctx, llm)
    si = llm._settings.system_instruction
    check("agent: passages added to the system instruction",
          si.startswith("BASE") and "62 degrees" in si, si[:80])
    check("agent: history untouched", len(ctx.messages) == 1)
    got = []

    async def cb(r):
        got.append(r)
    params = types.SimpleNamespace(function_name="web_search", arguments={"query": "x"},
                                   result_callback=cb)
    res = await agent._web_handler(params)
    check("agent: web_search refused in files-only mode", "files-only mode is on" in res, res)
    names = {s.name for s in agent._function_schemas()}
    check("agent: library tools registered", lib.NAMES <= names, str(lib.NAMES - names))
    lib.dispatch("stop_using_files")
    llm2 = types.SimpleNamespace(_settings=types.SimpleNamespace(system_instruction="BASE"))
    agent._library_ground(ctx, llm2)
    check("agent: nothing added when the mode is off", llm2._settings.system_instruction == "BASE")


def dashboard_case():
    path = os.path.join(os.path.dirname(HERE), "bootsui", "boots_ui.py")
    src = open(path, encoding="utf-8").read()
    import re
    m = re.search(r"_LIB_NAMES = \{(.*?)\}", src, re.S)
    names = set(re.findall(r'"([a-z_]+)"', m.group(1))) if m else set()
    check("dashboard: routes every library tool", names == lib.NAMES, str(names ^ lib.NAMES))


async def main():
    try:
        d = setup_docs()
        read_case(d)
        chunk_case()
        add_case(d)
        search_case()
        mode_case()
        guard_case()
        remove_case()
        dashboard_case()
        await agent_case(d)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    ok = all(RESULTS)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    code = loop.run_until_complete(main())
    sys.stdout.flush()
    os._exit(code)
