"""
Wikipedia Training engine for OryvexAI.

Give it a topic; it turns the matching Wikipedia article into chat training data.
Standard library only. The web panel's "Training" page drives it (see server.py).

THE PIPELINE (each step is logged live, see WikiTrainer._run)

  1. SEARCH   MediaWiki API  action=query&list=search  -> ranked article titles.
              The best hit is used; disambiguation pages are skipped automatically.
  2. FETCH    MediaWiki API  action=parse&prop=text    -> the article as rendered HTML
              (what the page itself is built from: infobox, menus, references, ...).
  3. CLEAN    HTMLParser walk that drops everything that is not prose: scripts, styles,
              tables / infoboxes, navboxes, the table of contents, hatnotes, image
              thumbnails, edit links, citation markers like [12], "See also",
              "References", "External links" sections ... What is left is a list of
              (section, paragraph) pairs.
  4. CHUNK    Paragraphs of one section are merged into chunks of roughly `chunk_size`
              characters. Over-long paragraphs are split on sentence boundaries, so a
              chunk never stops in the middle of a sentence.
  5. Q&A      Rule-based synthetic question-answer generation, no model needed:
                * definition   "What is <Title>?"            <- first sentences of the lead
                * overview     "Tell me about <Title>."      <- the whole lead chunk
                * section      "Tell me about <section> in <Title>." <- a body chunk
                * who          "By whom was X invented?"     <- "X was invented by Y ..."
                * when         "What happened in 1969 ...?"  <- a sentence containing a year
                * what         "What is <X>?"                <- "X is a/an/the ..." sentences
              Every answer is a sentence (or chunk) copied from the article, so the
              pairs are grounded in the text. Answers go through the same quality
              filter as Learn mode (length, script, no e-mails / long numbers,
              data/blocklist.txt).
  6. CONVERT  Every pair becomes ONE line in the data/chat.jsonl format:
                {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
              The internal state (`WikiTrainer.records`) holds exactly these objects and
              nothing else; each one is validated strictly before it is accepted.
              `chat_jsonl()` serialises that state, and the panel's download button
              serves it.

Wikipedia text is CC BY-SA 4.0. Keep that in mind when you share a model trained on it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
from html.parser import HTMLParser
from pathlib import Path

from learn import (SYSTEM, FetchError, Fetcher, clean_text, good_paragraph,
                   load_blocklist, summary_answer)

LANGS = ("en", "fa")
WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
LOG_LIMIT = 2000
MAX_TOPICS = 20
PREVIEW_LIMIT = 8
ROLES = ("system", "user", "assistant")


def detect_lang(text: str) -> str:
    """'fa' if the topic is written in Persian/Arabic script, otherwise 'en'."""
    return "fa" if re.search(r"[\u0600-\u06ff]", text) else "en"


# =================================================================== 3. CLEANING
class _ArticleParser(HTMLParser):
    """Walks Wikipedia's rendered HTML and keeps only (section, paragraph) prose."""

    VOID = {"br", "hr", "img", "input", "link", "meta", "source", "wbr", "area", "col", "base"}
    DROP_TAGS = {"script", "style", "noscript", "table", "figure", "figcaption", "nav", "aside",
                 "form", "svg", "iframe", "template", "button", "select", "textarea", "audio",
                 "video", "math", "head"}
    DROP_CLASSES = {
        "infobox", "navbox", "vertical-navbox", "sidebar", "reflist", "references",
        "mw-references-wrap", "toc", "hatnote", "navigation-not-searchable", "thumb",
        "tsingle", "gallery", "metadata", "ambox", "mbox-small", "mw-editsection",
        "noprint", "reference", "mw-empty-elt", "sistersitebox", "side-box", "shortdescription",
        "refbegin", "citation", "catlinks", "printfooter", "mw-cite-backlink", "navbar",
        "error", "plainlinks", "portal", "dablink", "rellink", "hlist", "mw-authority-control",
    }
    DROP_SECTIONS = {
        "references", "external links", "see also", "notes", "further reading", "bibliography",
        "sources", "footnotes", "citations", "gallery", "works cited",
        "منابع", "پانویس", "جستارهای وابسته", "پیوند به بیرون", "برای مطالعه بیشتر", "یادداشت‌ها",
    }
    BLOCK = {"p", "li", "dd", "dt", "blockquote", "h2", "h3", "h4", "div", "tr", "br"}
    HEADINGS = {"h2": 2, "h3": 3, "h4": 4}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []   # (tag, is_dropped)
        self.drop = 0
        self.buf: list[str] = []
        self.heading_level = 0
        self.section = ""                          # "" = lead
        self.paras: list[tuple[str, str]] = []     # (section, text)
        self.removed = 0                           # how many junk elements were dropped
        self.disambiguation = False

    # -- helpers
    def _flush(self):
        text = " ".join("".join(self.buf).split())
        self.buf = []
        if not text:
            return
        if self.heading_level:
            self.section = text.replace("[edit]", "").strip()
            self.heading_level = 0
        else:
            self.paras.append((self.section, text))

    def _dropped(self, tag: str, attrs: dict) -> bool:
        if tag in self.DROP_TAGS:
            return True
        classes = set((attrs.get("class") or "").split())
        if classes & self.DROP_CLASSES:
            return True
        if tag == "sup" and (classes & {"reference", "noprint"} or (attrs.get("id") or "").startswith("cite_ref")):
            return True
        if tag in ("div", "span", "section") and (attrs.get("role") in ("navigation", "note", "presentation")):
            return True
        if attrs.get("aria-hidden") == "true" and tag != "div":
            return True
        return False

    # -- HTMLParser hooks
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        # only the notice box counts; ordinary links to disambiguation pages carry class "mw-disambig"
        if tag in ("div", "table") and {"disambigbox", "dmbox-disambig"} & set((a.get("class") or "").split()):
            self.disambiguation = True
        if tag in self.VOID:
            if tag == "br" and not self.drop:
                self._flush()
            return
        dropped = self._dropped(tag, a)
        self.stack.append((tag, dropped))
        if dropped:
            self.drop += 1
            self.removed += 1
            return
        if self.drop:
            return
        if tag in self.BLOCK:
            self._flush()
        if tag in self.HEADINGS:
            self.heading_level = self.HEADINGS[tag]

    def handle_startendtag(self, tag, attrs):
        if tag == "br" and not self.drop:
            self._flush()

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        # close the nearest matching open tag (tolerates sloppy HTML)
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                closed = self.stack[i:]
                del self.stack[i:]
                self.drop -= sum(1 for _, d in closed if d)
                self.drop = max(0, self.drop)
                break
        else:
            return
        if not self.drop and tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if not self.drop:
            self.buf.append(data)


_CITE = re.compile(r"\[(?:\d+|[a-z]|note \d+|citation needed|clarification needed|when\?|who\?|edit)\]", re.I)
_SPACE_PUNCT = re.compile(r"\s+([,.;:!?؟،])")
_ZW = re.compile(r"[​‌‍⁠﻿\xa0]")
_PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def clean_article_html(html: str) -> dict:
    """HTML -> {"paragraphs": [(section, text)], "removed": n, "disambiguation": bool}."""
    p = _ArticleParser()
    p.feed(html)
    p._flush()
    out: list[tuple[str, str]] = []
    for section, text in p.paras:
        if section.strip().lower() in _ArticleParser.DROP_SECTIONS:
            continue
        text = _ZW.sub(" ", text)
        text = _CITE.sub("", text)
        text = _SPACE_PUNCT.sub(r"\1", " ".join(text.split()))
        text = clean_text(text, limit=4000)
        if len(text) >= 40 and len(text.split()) >= 6:
            out.append((section.strip(), text))
    return {"paragraphs": out, "removed": p.removed, "disambiguation": p.disambiguation}


# =================================================================== 4. CHUNKING
_ABBR = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "jr", "sr", "no", "inc", "ltd", "prof", "ca",
         "c", "fig", "approx", "est", "gen", "col", "lt", "mt", "ft", "e.g", "i.e", "u.s", "u.k"}
_END = re.compile(r"[.!?؟]['\")\]”’]*$")


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that does not break on 'Dr.', 'U.S.', 'J. R. R.' and the like."""
    words, out, cur = text.split(), [], []
    for i, w in enumerate(words):
        cur.append(w)
        if not _END.search(w):
            continue
        bare = w.rstrip(".!?؟'\")]”’").lower()
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if w.endswith(".") and (bare in _ABBR or (len(bare) == 1 and bare.isalpha())):
            continue
        if nxt and not (nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in "\"'“(" or "؀" <= nxt[0] <= "ۿ"):
            continue
        out.append(" ".join(cur))
        cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def make_chunks(paragraphs: list[tuple[str, str]], chunk_size: int = 700) -> list[dict]:
    """Merge paragraphs per section into ~chunk_size char chunks, cut only at sentence ends."""
    chunks: list[dict] = []
    cur_sec, cur = None, ""

    def close():
        nonlocal cur
        if cur.strip():
            chunks.append({"section": cur_sec or "", "text": cur.strip()})
        cur = ""

    for section, para in paragraphs:
        if section != cur_sec:
            close()
            cur_sec = section
        for sent in split_sentences(para):
            if cur and len(cur) + len(sent) + 1 > chunk_size:
                close()
            cur = (cur + " " + sent).strip()
        if len(cur) >= chunk_size * 0.8:
            close()
    close()
    return chunks


# =================================================================== 5. Q&A GENERATION
Q_TEMPLATES = {
    "en": {
        "define": ["What is {t}?", "Who or what is {t}?", "Can you define {t}?", "Give me a short definition of {t}."],
        "overview": ["Tell me about {t}.", "Explain {t}.", "What can you tell me about {t}?", "Give me an overview of {t}."],
        "section": ["Tell me about {s} in the context of {t}.", "What should I know about {s} regarding {t}?",
                    "Explain the {s} of {t}.", "What does Wikipedia say about {s} in the article on {t}?"],
        "year": ["What happened in {y} in relation to {t}?", "What is notable about {y} for {t}?",
                 "What do you know about {t} in {y}?"],
        "thing": ["What is {x}?", "Can you explain what {x} is?", "Define {x}."],
        "more": ["Tell me more about {t} (part {n}).", "What else should I know about {t} (part {n})?",
                 "Give me more background on {t} (part {n})."],
    },
    "fa": {
        "define": ["{t} چیست؟", "{t} کیست یا چیست؟", "می‌شه {t} رو تعریف کنی؟"],
        "overview": ["درباره {t} توضیح بده.", "می‌شه درباره {t} بگی؟", "یه نمای کلی از {t} بده."],
        "section": ["درباره {s} در موضوع {t} توضیح بده.", "در مورد {s} در {t} چه می‌دانی؟"],
        "year": ["در سال {y} چه اتفاقی برای {t} افتاد؟", "درباره {t} در سال {y} چه می‌دانی؟"],
        "thing": ["{x} چیست؟"],
        "more": ["بیشتر درباره {t} بگو (بخش {n}).", "چه نکته دیگری درباره {t} باید بدانم (بخش {n})؟"],
    },
}
_YEAR = re.compile(r"(?<!\d)(1[0-9]{3}|20[0-4][0-9])(?!\d|s\b|')")
_BY = re.compile(
    r"^(?P<subj>[A-Z][^,;]{2,70}?) (?P<aux>was|were|is|are) (?:first |originally |later )?"
    r"(?P<verb>founded|invented|discovered|written|developed|created|designed|proposed|named|built|"
    r"established|published|released|introduced|formulated|coined|composed|painted|directed|"
    r"produced|described|launched|led|started) by (?P<who>[^,.;]{3,80})")
_IS_A = re.compile(r"^(?P<subj>[A-Z][\w'’\-]*(?: [\w'’\-]+){0,4}) (?:is|are|was|were) (?:an?|the) (?P<rest>.{25,})$")
_BAD_SUBJ = re.compile(r"^(?:He|She|It|They|This|That|These|Those|There|Its|His|Her|Their|Here|We|You|I|One)\b")


def _pick(options: list[str], key: str) -> str:
    return options[int(hashlib.sha1(key.encode("utf-8")).hexdigest(), 16) % len(options)]


def _answer_ok(a: str, blocklist: list[str]) -> bool:
    return 40 <= len(a) <= 900 and good_paragraph(a, blocklist)


def _trim(text: str, max_chars: int) -> str:
    """Cut at a sentence boundary so the answer never ends mid-sentence."""
    out = ""
    for s in split_sentences(text):
        if out and len(out) + len(s) + 1 > max_chars:
            break
        out = (out + " " + s).strip()
    return out if out else text[:max_chars].rsplit(" ", 1)[0]


def generate_pairs(title: str, chunks: list[dict], lang: str, max_pairs: int,
                   blocklist: list[str], progress=None) -> list[tuple[str, str, str]]:
    """Return [(kind, question, answer)] built from the chunks. `progress(i, n, added)` is optional."""
    T = Q_TEMPLATES[lang]
    base_title = re.sub(r"\s*[\(（].*?[\)）]\s*$", "", title).strip().lower()
    pairs: list[tuple[str, str, str]] = []
    seen_q: set[str] = set()

    def add(kind: str, q: str, a: str) -> bool:
        a = " ".join(a.split())
        k = q.strip().lower()
        if len(pairs) >= max_pairs or k in seen_q or not _answer_ok(a, blocklist):
            return False
        seen_q.add(k)
        pairs.append((kind, q.strip(), a))
        return True

    for i, ch in enumerate(chunks):
        before = len(pairs)
        text, sec = ch["text"], ch["section"]
        sentences = split_sentences(text)
        key = f"{title}|{i}"
        if i == 0:  # first chunk = the article's opening: definition + overview
            short = summary_answer(text)
            if short:
                add("define", _pick(T["define"], key + "d").format(t=title), short)
            add("overview", _pick(T["overview"], key + "o").format(t=title), _trim(text, 700))
        elif not sec:  # later chunks of the unnamed lead: must NOT reuse the definition questions
            add("more", _pick(T["more"], key + "m").format(t=title, n=i + 1), _trim(text, 700))
        else:
            add("section", _pick(T["section"], key + "s").format(s=sec, t=title), _trim(text, 700))

        facts = 0
        for s in sentences:
            if facts >= 2 or len(pairs) >= max_pairs:
                break
            if lang == "en":
                m = _BY.match(s)
                if m:
                    q = f"By whom {'was' if m['aux'] in ('was', 'is') else 'were'} {m['subj']} {m['verb']}?"
                    if m["aux"] in ("is", "are"):
                        q = f"By whom {m['aux']} {m['subj']} {m['verb']}?"
                    if add("who", q, s):
                        facts += 1
                        continue
                m = _IS_A.match(s)
                if m and not _BAD_SUBJ.match(m["subj"]) and m["subj"].lower() not in (title.lower(), base_title):
                    if add("thing", _pick(T["thing"], key + m["subj"]).format(x=m["subj"]), s):
                        facts += 1
                        continue
            ym = _YEAR.search(s.translate(_PERSIAN_DIGITS))
            if ym and add("year", _pick(T["year"], key + ym.group(1)).format(y=ym.group(1), t=title), s):
                facts += 1
        if progress:
            progress(i + 1, len(chunks), len(pairs) - before)
        if len(pairs) >= max_pairs:
            break
    return pairs


# =================================================================== 6. chat.jsonl STATE
def make_record(question: str, answer: str) -> dict:
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


def validate_record(rec) -> bool:
    """Strictly the chat.jsonl shape: {"messages":[system, user, assistant]} and nothing else."""
    if not isinstance(rec, dict) or set(rec) != {"messages"}:
        return False
    msgs = rec["messages"]
    if not isinstance(msgs, list) or len(msgs) != 3:
        return False
    for m, role in zip(msgs, ROLES):
        if not isinstance(m, dict) or set(m) != {"role", "content"} or m["role"] != role:
            return False
        if not isinstance(m["content"], str) or not m["content"].strip():
            return False
    return True


# =================================================================== the manager
class WikiTrainer:
    """Runs the pipeline in a background thread; the web panel polls / streams its state."""

    def __init__(self, data_dir: str | None = None, engine=None, ckpt: str | None = None):
        root = Path(__file__).resolve().parent
        self.root = root
        self.data_dir = Path(data_dir) if data_dir else root / "data"
        self.ckpt = Path(ckpt) if ckpt else root / "checkpoints" / "oryvex.pt"
        self.engine = engine               # live model: hot-reloaded after training
        self.proc: subprocess.Popen | None = None
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.logs: list[dict] = []
        self.next_id = 1
        self.records: list[dict] = []      # THE internal state: strictly chat.jsonl objects
        self.state = self._fresh()

    @staticmethod
    def _fresh() -> dict:
        return {"phase": "idle", "step": 0, "steps": 6, "train_step": 0, "train_total": 0,
                "loss": None, "val_loss": None, "eta_min": None, "topic": "", "lang": "en", "title": "",
                "url": "", "topic_n": 0, "topics": 1, "progress": 0.0, "chunks": 0, "paragraphs": 0, "removed": 0,
                "pairs": 0, "format": "chat.jsonl", "ready": False}

    # ---- logging / status
    def _log(self, msg: str, level: str = "info"):
        with self.cond:
            self.logs.append({"id": self.next_id, "t": time.strftime("%H:%M:%S"), "level": level, "msg": msg})
            self.next_id += 1
            del self.logs[:-LOG_LIMIT]
            self.cond.notify_all()

    def _set(self, **kw):
        with self.cond:
            self.state.update(kw)
            self.cond.notify_all()

    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def status(self, since: int | None = None) -> dict:
        with self.lock:
            prev = [{"question": r["messages"][1]["content"], "answer": r["messages"][2]["content"]}
                    for r in self.records[:PREVIEW_LIMIT]]
            return {**self.state, "running": self.running(), "records": len(self.records), "preview": prev,
                    "last_log": self.logs[-1]["id"] if self.logs else 0,
                    **({"logs": [e for e in self.logs if e["id"] > since]} if since is not None else {})}

    def wait_logs(self, since: int, timeout: float = 15.0) -> tuple[list[dict], dict]:
        """Block until there are log lines newer than `since` (or timeout); used by the SSE stream."""
        with self.cond:
            if not any(e["id"] > since for e in self.logs[-1:]):
                self.cond.wait(timeout)
            return [e for e in self.logs if e["id"] > since], dict(self.state, running=self.running(),
                                                                    records=len(self.records))

    # ---- state -> file
    def chat_jsonl(self) -> str:
        with self.lock:
            return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.records)

    # ---- control
    @staticmethod
    def _normalise(o: dict):
        raw = o.get("topics", o.get("topic", ""))
        if isinstance(raw, str):
            raw = re.split(r"[\n;,،]+", raw)
        topics: list[str] = []
        for t in raw or []:
            t = " ".join(str(t).split())
            if t and t.lower() not in [x.lower() for x in topics]:
                topics.append(t)
        if not topics:
            return "Type at least one topic."
        if len(topics) > MAX_TOPICS:
            return f"Too many topics (max {MAX_TOPICS} per run)."
        if any(len(t) > 120 for t in topics):
            return "A topic is too long (max 120 characters)."

        def clamp(v, lo, hi, d):
            try:
                return max(lo, min(hi, int(v)))
            except (TypeError, ValueError):
                return d
        contact = " ".join(str(o.get("contact", "")).split())[:100]
        if contact and not re.fullmatch(r"[\w.+-]+@[\w-]+\.[\w.-]+|https?://\S+", contact):
            return "Contact must be an e-mail address or a website URL (or leave it empty)."
        return {"topics": topics, "contact": contact, "append": bool(o.get("append", True)),
                "train": bool(o.get("train", True)), "steps": clamp(o.get("steps", 1500), 50, 20000, 1500),
                "max_pairs": clamp(o.get("max_pairs", 60), 5, 400, 60),   # per topic
                "chunk_size": clamp(o.get("chunk_size", 700), 300, 1500, 700)}

    def start(self, opts: dict) -> str | None:
        o = self._normalise(opts)
        if isinstance(o, str):
            return o
        with self.lock:
            if self.running():
                return "A run is already in progress."
            self.stop_event.clear()
            self.logs.clear()
            self.state = {**self._fresh(), "phase": "searching", "topic": ", ".join(o["topics"]), "lang": "auto",
                          "topic_n": 0, "topics": len(o["topics"]), "steps": 7 if o["train"] else 6}
            self.thread = threading.Thread(target=self._run, args=(o,), daemon=True)
            self.thread.start()
        return None

    def stop(self):
        self.stop_event.set()
        p = self.proc
        if p and p.poll() is None:  # ask train.py to save what it has and exit
            try:
                p.send_signal(signal.SIGINT if os.name == "posix" else signal.SIGTERM)
            except Exception:
                pass

    def clear(self) -> str | None:
        with self.lock:
            if self.running():
                return "Stop the current run first."
            self.records = []
            self.logs.clear()
            self.state = self._fresh()
            self.cond.notify_all()
        return None

    def join(self):
        if self.thread:
            self.thread.join()

    # ---- the pipeline
    def _api(self, fetcher: Fetcher, lang: str, params: dict) -> dict:
        base = os.environ.get("ORYVEX_WIKI_API", WIKI_API).format(lang=lang)
        url = base + "?" + urllib.parse.urlencode({"format": "json", "formatversion": "2", **params})
        for attempt in range(6):
            try:
                data, _, _ = fetcher.get(url, "application/json", check_robots=False)  # documented API
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503) or attempt == 5:
                    raise
                try:
                    wait = float(e.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    wait = 0
                wait = min(120.0, max(wait, 8.0 * 2 ** attempt))
                self._log(f"Wikipedia asked us to slow down (HTTP {e.code}). Waiting {wait:.0f}s, "
                          f"then retrying ({attempt + 1}/5)...", "warn")
                if self.stop_event.wait(wait):
                    raise InterruptedError
        j = json.loads(data.decode("utf-8"))
        if "error" in j:
            raise FetchError(j["error"].get("info", "Wikipedia API error"))
        return j

    def _step(self, n: int, phase: str, msg: str):
        t = self.state
        base = t.get("topic_n", 1) - 1
        self._set(step=n, phase=phase, progress=(base + (n - 1) / 6) / max(1, t.get("topics", 1)))
        self._log(f"[{n}/6] {msg}", "step")

    def _stopped(self) -> bool:
        if self.stop_event.is_set():
            self._set(phase="stopped")
            self._log("Stopped by user. The previous chat.jsonl state was left untouched.", "warn")
            return True
        return False

    def _process_topic(self, fetcher, topic, o, blocklist):
        """Steps 1-5 for one topic. Returns [(kind, q, a)] or None if stopped."""
        lang = detect_lang(topic)
        self._set(lang=lang)
        self._log(f"Language detected automatically: {'Persian' if lang == 'fa' else 'English'} -> {lang}.wikipedia.org")
        # 1. SEARCH
        self._step(1, "searching", f"Searching Wikipedia ({lang}) for '{topic}'...")
        res = self._api(fetcher, lang, {"action": "query", "list": "search", "srsearch": topic,
                                        "srnamespace": "0", "srlimit": "5", "srprop": "snippet"})
        hits = [h["title"] for h in res.get("query", {}).get("search", [])]
        if not hits:
            raise FetchError(f"No Wikipedia article found for '{topic}'.")
        self._log(f"Search returned {len(hits)} candidate(s): " + "; ".join(hits))
        if self._stopped():
            return None

        # 2. FETCH (try candidates until one is a real article)
        article = None
        for title in hits:
            self._step(2, "fetching", f"Fetching article '{title}' (rendered HTML)...")
            page = self._api(fetcher, lang, {"action": "parse", "page": title, "prop": "text",
                                             "redirects": "1", "disableeditsection": "1",
                                             "disablelimitreport": "1", "disabletoc": "1"})
            parsed = page.get("parse", {})
            html = parsed.get("text", "")
            if isinstance(html, dict):
                html = html.get("*", "")
            self._log(f"Downloaded {len(html):,} characters of raw HTML for '{parsed.get('title', title)}'.")
            # 3. CLEAN
            self._step(3, "cleaning", "Cleaning text: removing menus, infoboxes, tables, references, edit links...")
            cleaned = clean_article_html(html)
            if (cleaned["disambiguation"] and len(cleaned["paragraphs"]) < 15) or len(cleaned["paragraphs"]) < 2:
                self._log(f"'{title}' is a disambiguation / stub page, trying the next candidate.", "warn")
                continue
            article = (parsed.get("title", title), cleaned)
            break
        if article is None:
            raise FetchError(f"None of the search results for '{topic}' was a usable article.")
        title, cleaned = article
        paras = cleaned["paragraphs"]
        chars = sum(len(t) for _, t in paras)
        url = f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        self._set(title=title, url=url, paragraphs=self.state["paragraphs"] + len(paras),
                  removed=self.state["removed"] + cleaned["removed"])
        self._log(f"Removed {cleaned['removed']} non-prose elements. Kept {len(paras)} paragraphs "
                  f"({chars:,} characters of clean text) in {len({s for s, _ in paras})} section(s).")
        if self._stopped():
            return None

        # 4. CHUNK
        self._step(4, "chunking", f"Splitting text into logical chunks (~{o['chunk_size']} characters, sentence-aligned)...")
        chunks = make_chunks(paras, o["chunk_size"])
        self._set(chunks=self.state["chunks"] + len(chunks))
        self._log(f"Created {len(chunks)} chunks.")
        if not chunks:
            raise FetchError("The article had no usable text.")
        if self._stopped():
            return None

        # 5. Q&A
        self._step(5, "generating", "Generating Q&A pairs from the chunks...")
        base = self.state["topic_n"] - 1
        total = self.state["topics"]

        def progress(i, n, added):
            self._set(progress=(base + (4 + i / n) / 6) / total, pairs=self.state["pairs"] + added)
            self._log(f"Generating Q&A pairs... chunk {i}/{n} -> +{added} pair(s)")
            if self.stop_event.is_set():
                raise InterruptedError

        try:
            pairs = generate_pairs(title, chunks, lang, o["max_pairs"], blocklist, progress)
        except InterruptedError:
            self._stopped()
            return None
        kinds: dict[str, int] = {}
        for k, _, _ in pairs:
            kinds[k] = kinds.get(k, 0) + 1
        self._log("Pair types: " + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none"))
        return pairs

    def _append_jsonl(self, path: Path, records: list[dict]) -> int:
        """Append records to a chat .jsonl file, skipping questions that are already in it."""
        known: set[str] = set()
        needs_nl = False
        if path.exists():
            raw = path.read_text(encoding="utf-8")
            needs_nl = bool(raw) and not raw.endswith("\n")
            for line in raw.splitlines():
                try:
                    msgs = json.loads(line)["messages"]
                    known.add(next(m["content"] for m in msgs if m["role"] == "user").strip().lower())
                except Exception:
                    continue
        new = []
        for r in records:
            q = r["messages"][1]["content"].strip().lower()
            if q not in known:
                known.add(q)
                new.append(r)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            if needs_nl:
                f.write("\n")
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._log(f"Appended {len(new)} new conversation(s) to {path.name} "
                  f"({len(records) - len(new)} duplicate(s) skipped).", "ok")
        return len(new)

    def _train_model(self, o: dict):
        """Step 7: fine-tune the model on the data (train.py --resume) and stream its output live."""
        steps = o["steps"]
        self._set(phase="training", step=7, progress=0.0, train_step=0, train_total=steps)
        self._log(f"[7/7] Training the model on the new data ({steps} steps, resuming {self.ckpt.name})...", "step")
        if not self.ckpt.exists():
            raise FileNotFoundError(f"No model at {self.ckpt}. Train one first: python train.py")
        shutil.copy2(self.ckpt, self.ckpt.with_suffix(".prev.pt"))   # backup, so the Learn page can roll back
        cmd = [sys.executable, "-u", str(self.root / "train.py"), "--resume", "--steps", str(steps),
               "--batch-size", "16", "--lr", "5e-4", "--eval-every", str(max(50, steps // 4)),
               "--out", str(self.ckpt), "--data-dir", str(self.data_dir), "--no-demo"]
        dev = getattr(self.engine, "device", None)
        if dev and dev != "auto":
            cmd += ["--device", str(dev)]
        self.proc = subprocess.Popen(cmd, cwd=str(self.root), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, encoding="utf-8", errors="replace")
        step_re = re.compile(r"step\s+(\d+)/(\d+)\s*\|\s*loss\s+([\d.]+).*?eta\s+([\d.]+)")
        val_re = re.compile(r"validation loss\s+([\d.]+)")
        tail: list[str] = []
        for line in self.proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail = (tail + [line])[-6:]
            m, v = step_re.search(line), val_re.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                done = max(0, done - (total - steps))
                self._set(train_step=done, loss=float(m.group(3)), eta_min=float(m.group(4)),
                          progress=min(1.0, done / steps))
            if v:
                self._set(val_loss=float(v.group(1)))
            self._log("train> " + line, "info")
        rc = self.proc.wait()
        self.proc = None
        if rc != 0 and not self.stop_event.is_set():
            raise RuntimeError("Training failed: " + " | ".join(tail[-3:]))
        self._set(phase="reloading")
        if self.engine is not None:
            self._log("Loading the new weights into the running chat...")
            self.engine.reload(str(self.ckpt))
            self._log("Done. The chat page now uses the updated model.", "ok")
        else:
            self._log("Restart the server to use the new weights.", "warn")
        if self.stop_event.is_set():
            self._set(phase="stopped")
            self._log("Stopped. The latest saved checkpoint is live.", "warn")
        else:
            self._set(phase="done", progress=1.0)

    def _run(self, o: dict):
        topics, n = o["topics"], len(o["topics"])
        try:
            if o["contact"]:
                os.environ["ORYVEX_CONTACT"] = o["contact"]  # identifies us in the User-Agent, as Wikimedia asks
            blocklist = load_blocklist(self.data_dir)
            fetcher = Fetcher(delay=2.5, max_bytes=8_000_000)  # slow and polite: ~1 request / 2.5 s
            self._log(f"Training request: {n} topic(s) {topics} "
                      f"max_pairs/topic={o['max_pairs']} chunk_size={o['chunk_size']}")
            all_pairs: list[tuple[str, str, str]] = []
            seen_q: set[str] = set()
            failed = 0
            for i, topic in enumerate(topics, 1):
                self._set(topic_n=i)
                self._log(f"=== Topic {i}/{n}: '{topic}' ===", "step")
                try:
                    pairs = self._process_topic(fetcher, topic, o, load_blocklist(self.data_dir) or blocklist)
                except InterruptedError:
                    self._stopped()
                    return
                except Exception as e:  # one bad topic must not kill the whole run
                    failed += 1
                    self._log(f"Skipping '{topic}': {e}", "error")
                    continue
                if pairs is None:
                    return  # stopped
                for kind, q, a in pairs:
                    if q.lower() not in seen_q:
                        seen_q.add(q.lower())
                        all_pairs.append((kind, q, a))
            if not all_pairs:
                raise FetchError("No question-answer pairs were produced.")

            # 6. CONVERT -> chat.jsonl state (replaces the previous state atomically)
            self._set(step=6, phase="converting", progress=0.98, topic_n=n)
            self._log(f"[6/6] Converting {len(all_pairs)} pairs to chat.jsonl format and validating...", "step")
            records = [make_record(q, a) for _, q, a in all_pairs]
            bad = [r for r in records if not validate_record(r)]
            if bad:
                raise ValueError(f"{len(bad)} record(s) failed chat.jsonl validation.")
            with self.cond:
                self.records = records
                self.state.update(phase="done" if not o["train"] else "converting", step=6, progress=1.0,
                                  pairs=len(records), ready=True)
                self.cond.notify_all()
            size = len(self.chat_jsonl().encode("utf-8"))
            self._log(f"State set to chat.jsonl: {len(records)} valid conversation(s) from {n - failed}/{n} topic(s), "
                      f"{size:,} bytes. Ready to download.", "ok")
            if o["append"]:
                self._append_jsonl(self.data_dir / "chat.jsonl", records)
            if o["train"]:
                # learned.jsonl is oversampled by train.py, so a few hundred new examples are not drowned out
                self._append_jsonl(self.data_dir / "learned.jsonl", records)
            if failed:
                self._log(f"{failed} topic(s) failed. Run them again later (duplicates are skipped when appending).", "warn")
            if o["train"]:
                self._train_model(o)
        except Exception as e:  # keep the panel alive whatever happens
            self._set(phase="error")
            self._log(f"Error: {e}", "error")
