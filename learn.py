"""
Learn mode for OryvexAI: collect public text from the web, turn it into training
data, fine-tune the model, and hot-reload it.

What it does (and does not do)
  * Wikipedia (English / Persian) through the official MediaWiki API: random
    articles or topics you name. Each article becomes (a) a chat example
    "Tell me about X" -> first sentences of the article and (b) raw text.
  * Optional extra pages you list yourself (and, if you choose, links on the
    same site up to depth 2). robots.txt is obeyed, requests are rate limited,
    private/internal addresses are blocked, and it identifies itself honestly.
  * It is NOT "the whole internet". Every run is bounded (page caps, depth caps).
    A 4M-parameter model cannot absorb the web; more, cleaner data helps only
    up to what the model size can hold.

Everything is standard library. Output goes to:
    data/learned.jsonl   chat examples        data/learned.txt   raw paragraphs
    data/learned_state.json  what was already collected (no duplicates)
    data/blocklist.txt   optional words/domains to skip, one per line

CLI (no web panel needed):
    python learn.py --lang en fa --random 100 --train --steps 300
    python learn.py --lang en --topics "Neural network" "Tehran"
    python learn.py --urls https://example.com/article --consent
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import ipaddress
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_WIKI_API = "https://{lang}.wikipedia.org/w/api.php"
LANGS = ("en", "fa")


def user_agent() -> str:
    contact = os.environ.get("ORYVEX_CONTACT", "set ORYVEX_CONTACT to your email or site")
    return f"OryvexAI-Learner/1.0 (personal learning project; {contact})"


# =============================================================== polite fetching
class FetchError(Exception):
    pass


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, fetcher: "Fetcher"):
        self.fetcher = fetcher

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.fetcher.check_url(newurl)  # re-validate every redirect hop
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Fetcher:
    def __init__(self, delay: float = 1.0, timeout: int = 15, max_bytes: int = 2_000_000):
        self.delay, self.timeout, self.max_bytes = delay, timeout, max_bytes
        self._last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._opener = urllib.request.build_opener(_SafeRedirect(self))

    def check_url(self, url: str):
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise FetchError("only http/https URLs are supported")
        if os.environ.get("ORYVEX_ALLOW_PRIVATE") == "1":
            return
        try:
            infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80))
        except OSError:
            raise FetchError(f"cannot resolve {p.hostname}")
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                raise FetchError("internal/private addresses are blocked")

    def _raw_get(self, url: str, accept: str, max_bytes: int | None = None):
        host = urllib.parse.urlparse(url).netloc
        wait = self.delay - (time.time() - self._last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={
            "User-Agent": user_agent(), "Accept": accept, "Accept-Encoding": "identity"})
        limit = max_bytes or self.max_bytes
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                ctype = r.headers.get("Content-Type", "")
                data = r.read(limit + 1)
                final = r.geturl()
        finally:
            self._last[host] = time.time()
        if len(data) > limit:
            raise FetchError("page too large")
        return data, ctype, final

    def robots_ok(self, url: str) -> bool:
        p = urllib.parse.urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        rp = self._robots.get(base)
        if rp is None:
            rp = urllib.robotparser.RobotFileParser()
            try:
                data, _, _ = self._raw_get(base + "/robots.txt", "text/plain", 500_000)
                rp.parse(data.decode("utf-8", "replace").splitlines())
            except urllib.error.HTTPError as e:
                if e.code in (401, 403):
                    rp.disallow_all = True
                else:
                    rp.allow_all = True
            except Exception:
                rp.disallow_all = True  # can't read robots.txt -> be conservative
            rp.modified()
            self._robots[base] = rp
        return rp.can_fetch(user_agent(), url)

    def get(self, url: str, accept: str = "text/html,text/plain;q=0.8", check_robots: bool = True):
        self.check_url(url)
        if check_robots and not self.robots_ok(url):
            raise FetchError("blocked by robots.txt")
        return self._raw_get(url, accept)


# =============================================================== text extraction
class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "aside", "form", "svg",
            "iframe", "template", "button", "select", "head"}
    BLOCK = {"p", "li", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "td", "div",
             "br", "tr", "section", "article", "dd", "dt"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.buf: list[str] = []
        self.paras: list[str] = []
        self.links: list[str] = []
        self.noindex = False

    def _flush(self):
        text = " ".join("".join(self.buf).split())
        self.buf = []
        if text:
            self.paras.append(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta" and (a.get("name") or "").lower() == "robots" and "noindex" in (a.get("content") or "").lower():
            self.noindex = True
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self._flush()
        if tag == "a" and a.get("href"):
            self.links.append(a["href"])

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if self.skip == 0:
            self.buf.append(data)


def html_to_paragraphs(raw: str):
    ex = _TextExtractor()
    ex.feed(raw)
    ex._flush()
    return ex.paras, ex.links, ex.noindex


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_LONGNUM = re.compile(r"\d[\d\s().-]{8,}\d")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200e\u200f\u202a-\u202e]")


def load_blocklist(data_dir: Path) -> list[str]:
    path = data_dir / "blocklist.txt"
    if not path.exists():
        return []
    return [ln.strip().lower() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def good_paragraph(p: str, blocklist: list[str]) -> bool:
    if len(p) < 60 or len(p.split()) < 8:
        return False
    if _EMAIL.search(p) or _LONGNUM.search(p):  # skip likely personal data
        return False
    low = p.lower()
    if any(b in low for b in blocklist):
        return False
    letters = [c for c in p if c.isalpha()]
    if not letters:
        return False
    ok = sum(1 for c in letters if ("a" <= c.lower() <= "z") or ("\u0600" <= c <= "\u06ff"))
    return ok / len(letters) >= 0.85  # English / Persian script only


def clean_text(p: str, limit: int = 1200) -> str:
    p = _CTRL.sub("", p)
    p = " ".join(p.split())
    if len(p) > limit:
        cut = p[:limit]
        p = cut[: cut.rfind(" ")] if " " in cut else cut
    return p


_SENT = re.compile(r"(?<=[.!?؟])\s+")
_PAREN = re.compile(r"\s*[\(（][^()（）]*[\)）]")


def summary_answer(extract: str, max_chars: int = 420) -> str:
    text = " ".join(extract.split())
    for _ in range(2):
        text = _PAREN.sub("", text)  # drop pronunciations / dates in brackets
    out = ""
    for s in _SENT.split(text):
        if not s:
            continue
        if out and len(out) + len(s) + 1 > max_chars:
            break
        out = (out + " " + s).strip()
        if len(out) >= 160:
            break
    if len(out) > max_chars + 120 or not out:
        return ""
    if len(out) > max_chars:
        cut = out[:max_chars]
        out = cut[: cut.rfind(" ")].rstrip(",;:") + "."
    return out


QUESTIONS = {
    "en": ["Tell me about {t}.", "What is {t}?", "Explain {t}.", "Who or what is {t}?", "What can you tell me about {t}?"],
    "fa": ["درباره {t} توضیح بده.", "{t} چیست؟", "{t} کیست یا چیست؟", "می‌شه درباره {t} بگی؟"],
}
SYSTEM = "You are OryvexAI, a sharp and dedicated AI assistant."


# =============================================================== the manager
class LearnManager:
    """Runs one learn job at a time in a background thread."""

    def __init__(self, data_dir: str = "data", ckpt: str = "checkpoints/oryvex.pt",
                 engine=None, device: str = "auto"):
        self.data_dir = (ROOT / data_dir) if not os.path.isabs(data_dir) else Path(data_dir)
        self.ckpt = (ROOT / ckpt) if not os.path.isabs(ckpt) else Path(ckpt)
        self.engine, self.device = engine, device
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.log = collections.deque(maxlen=200)
        self.s = self._fresh_state()
        self.blocklist = load_blocklist(self.data_dir)
        self._state_path = self.data_dir / "learned_state.json"
        self.seen = self._load_seen()

    # ---- state helpers
    @staticmethod
    def _fresh_state():
        return {"phase": "idle", "message": "Idle.", "pages": 0, "articles": 0, "paragraphs": 0,
                "skipped": 0, "errors": 0, "step": 0, "total_steps": 0, "loss": None,
                "val_loss": None, "eta_min": None, "started": None}

    def _load_seen(self):
        try:
            d = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            d = {}
        return {"titles": set(d.get("titles", [])), "urls": set(d.get("urls", [])),
                "hashes": set(d.get("hashes", []))}

    def _save_seen(self):
        d = {"titles": sorted(self.seen["titles"]), "urls": sorted(self.seen["urls"]),
             "hashes": sorted(self.seen["hashes"])[-200_000:]}
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d), encoding="utf-8")
        os.replace(tmp, self._state_path)

    def _set(self, **kw):
        with self.lock:
            self.s.update(kw)

    def _inc(self, key, n=1):
        with self.lock:
            self.s[key] += n

    def _say(self, msg: str):
        with self.lock:
            self.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
            self.s["message"] = msg

    def _count_lines(self, name, sep="\n"):
        p = self.data_dir / name
        if not p.exists():
            return 0
        text = p.read_text(encoding="utf-8", errors="replace")
        return text.count("\n") if sep == "\n" else text.count(sep)

    def status(self) -> dict:
        with self.lock:
            out = dict(self.s)
            out["log"] = list(self.log)[-60:]
        out["running"] = bool(self.thread and self.thread.is_alive())
        out["data"] = {"chat_examples": self._count_lines("learned.jsonl"),
                       "paragraphs": self._count_lines("learned.txt", "\n\n")}
        out["can_rollback"] = self.ckpt.with_suffix(".prev.pt").exists() and not out["running"]
        if self.engine is not None:
            out["model"] = self.engine.info()
        return out

    # ---- public control
    def start(self, opts: dict) -> str | None:
        """Returns an error string, or None if the job started."""
        if self.thread and self.thread.is_alive():
            return "A learn job is already running."
        o = self._normalise(opts)
        if isinstance(o, str):
            return o
        self.stop_event.clear()
        with self.lock:
            self.s = self._fresh_state()
            self.s["started"] = time.time()
            self.log.clear()
        self.thread = threading.Thread(target=self._job, args=(o,), daemon=True)
        self.thread.start()
        return None

    def stop(self):
        self.stop_event.set()
        p = self.proc
        if p and p.poll() is None:
            try:
                p.send_signal(signal.SIGINT if os.name == "posix" else signal.SIGTERM)
            except Exception:
                pass

    def join(self):
        if self.thread:
            self.thread.join()

    def rollback(self) -> str | None:
        prev = self.ckpt.with_suffix(".prev.pt")
        if self.thread and self.thread.is_alive():
            return "Wait for the running job to finish."
        if not prev.exists():
            return "No previous model to restore."
        shutil.copy2(prev, self.ckpt)
        if self.engine is not None:
            self.engine.reload(str(self.ckpt))
        self._say("Restored the previous model.")
        return None

    @staticmethod
    def _normalise(o: dict):
        langs = [l for l in (o.get("langs") or []) if l in LANGS]
        topics = [t.strip() for t in (o.get("topics") or []) if str(t).strip()][:50]
        urls = [u.strip() for u in (o.get("urls") or []) if str(u).strip()][:50]
        mode = "topics" if o.get("mode") == "topics" else "random"
        if urls and not o.get("consent"):
            return "Confirm that you may use those pages (tick the checkbox) before crawling URLs."
        if mode == "topics" and not topics and langs:
            return "Enter at least one topic, or switch to random articles."
        if not langs and not urls:
            return "Choose at least one source."
        clamp = lambda v, lo, hi, d: max(lo, min(hi, int(v))) if str(v).lstrip("-").isdigit() else d
        return {
            "langs": langs, "mode": mode, "topics": topics, "urls": urls,
            "count": clamp(o.get("count", 40), 1, 2000, 40),
            "depth": clamp(o.get("depth", 0), 0, 2, 0),
            "max_pages": clamp(o.get("max_pages", 30), 1, 500, 30),
            "train": bool(o.get("train", True)),
            "steps": clamp(o.get("steps", 300), 20, 5000, 300),
            "delay": max(0.5, float(o.get("delay", 1.0))),
        }

    # ---- the job
    def _job(self, o: dict):
        try:
            self.blocklist = load_blocklist(self.data_dir)
            fetcher = Fetcher(delay=o["delay"])
            self._set(phase="collecting")
            for lang in o["langs"]:
                if self.stop_event.is_set():
                    break
                self._wikipedia(fetcher, lang, o)
            if o["urls"] and not self.stop_event.is_set():
                self._crawl(fetcher, o)
            self._save_seen()
            new = self.s["articles"] + self.s["paragraphs"]
            self._say(f"Collected {self.s['articles']} chat examples and {self.s['paragraphs']} text paragraphs.")
            if self.stop_event.is_set():
                self._set(phase="stopped")
                self._say("Stopped.")
                return
            if o["train"] and new > 0:
                self._train(o["steps"])
            elif o["train"]:
                self._say("Nothing new was collected, so training was skipped.")
                self._set(phase="done")
            else:
                self._set(phase="done")
        except Exception as e:  # keep the panel alive whatever happens
            self._set(phase="error")
            self._say(f"Error: {e}")

    # ---- Wikipedia
    def _wiki_call(self, fetcher: Fetcher, lang: str, extra: dict) -> list[dict]:
        base = os.environ.get("ORYVEX_WIKI_API", DEFAULT_WIKI_API).format(lang=lang)
        params = {"action": "query", "format": "json", "formatversion": "2", "prop": "extracts",
                  "exintro": "1", "explaintext": "1", "exlimit": "max", "redirects": "1", **extra}
        url = base + "?" + urllib.parse.urlencode(params)
        data, _, _ = fetcher.get(url, "application/json", check_robots=False)  # documented API, rate limited
        self._inc("pages")
        pages = json.loads(data.decode("utf-8")).get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        return sorted(pages, key=lambda p: p.get("index", 0))

    def _wikipedia(self, fetcher: Fetcher, lang: str, o: dict):
        label = "English" if lang == "en" else "Persian"
        self._say(f"Wikipedia ({label}): {'random articles' if o['mode'] == 'random' else 'topics'}...")
        if o["mode"] == "random":
            got, tries, dry = 0, 0, 0
            while got < o["count"] and tries < o["count"] // 5 + 8 and dry < 3 and not self.stop_event.is_set():
                tries += 1
                before = got
                try:
                    pages = self._wiki_call(fetcher, lang, {"generator": "random", "grnnamespace": "0", "grnlimit": "20"})
                except Exception as e:
                    self._inc("errors")
                    self._say(f"Request failed: {e}")
                    if tries >= 3 and got == 0:
                        break
                    continue
                for pg in pages:
                    if got >= o["count"]:
                        break
                    got += self._add_article(lang, pg)
                dry = dry + 1 if got == before else 0  # stop when batches stop yielding new articles
                self._say(f"Wikipedia ({label}): {got}/{o['count']} articles")
        else:
            for topic in o["topics"]:
                if self.stop_event.is_set():
                    break
                try:
                    pages = self._wiki_call(fetcher, lang, {"generator": "search", "gsrsearch": topic,
                                                           "gsrnamespace": "0", "gsrlimit": "5"})
                except Exception as e:
                    self._inc("errors")
                    self._say(f"'{topic}' failed: {e}")
                    continue
                added = sum(self._add_article(lang, pg) for pg in pages)
                self._say(f"Topic '{topic}': {added} new articles")

    def _add_article(self, lang: str, page: dict) -> int:
        title, extract = (page.get("title") or "").strip(), (page.get("extract") or "").strip()
        key = f"{lang}:{title}"
        low = (title + " " + extract[:200]).lower()
        if (not title or key in self.seen["titles"] or len(extract) < 120
                or title.startswith(("List of", "فهرست")) or "may refer to" in low
                or "ابهام‌زدایی" in title or "ممکن است به" in extract[:120]
                or any(b in low for b in self.blocklist)):
            self._inc("skipped")
            return 0
        answer = summary_answer(extract)
        if not answer or not good_paragraph(answer, self.blocklist):
            self._inc("skipped")
            return 0
        self.seen["titles"].add(key)
        q = QUESTIONS[lang][int(hashlib.sha1(key.encode()).hexdigest(), 16) % len(QUESTIONS[lang])].format(t=title)
        conv = {"messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": q},
                             {"role": "assistant", "content": answer}],
                "source": f"wikipedia:{lang}:{title}", "license": "CC BY-SA 4.0"}
        with open(self.data_dir / "learned.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
        self._add_paragraphs([clean_text(extract)])
        self._inc("articles")
        return 1

    def _add_paragraphs(self, paras: list[str]) -> int:
        n = 0
        with open(self.data_dir / "learned.txt", "a", encoding="utf-8") as f:
            for p in paras:
                p = clean_text(p)
                h = hashlib.sha1(p.encode("utf-8")).hexdigest()[:12]
                if h in self.seen["hashes"] or not good_paragraph(p, self.blocklist):
                    continue
                self.seen["hashes"].add(h)
                f.write(p + "\n\n")
                n += 1
        self._inc("paragraphs", n)
        return n

    # ---- arbitrary pages
    def _crawl(self, fetcher: Fetcher, o: dict):
        queue = collections.deque((u, 0) for u in o["urls"])
        visited, fetched = set(), 0
        self._say(f"Crawling your URLs (depth {o['depth']}, max {o['max_pages']} pages, robots.txt obeyed)...")
        while queue and fetched < o["max_pages"] and not self.stop_event.is_set():
            url, depth = queue.popleft()
            url = urllib.parse.urldefrag(url)[0]
            if url in visited or url in self.seen["urls"]:
                continue
            visited.add(url)
            try:
                data, ctype, final = fetcher.get(url)
                self._inc("pages")
                fetched += 1
                if "html" not in ctype.lower() and "text/plain" not in ctype.lower():
                    self._inc("skipped")
                    continue
                m = re.search(r"charset=([\w-]+)", ctype, re.I)
                raw = data.decode(m.group(1) if m else "utf-8", "replace")
                paras, links, noindex = html_to_paragraphs(raw) if "html" in ctype.lower() else (raw.split("\n\n"), [], False)
                if noindex:
                    self._inc("skipped")
                    continue
                self.seen["urls"].add(url)
                n = self._add_paragraphs(paras)
                self._say(f"{url} -> {n} paragraphs")
                if depth < o["depth"]:
                    host = urllib.parse.urlparse(final).netloc
                    for href in links:
                        nxt = urllib.parse.urljoin(final, href)
                        pr = urllib.parse.urlparse(nxt)
                        if pr.netloc == host and pr.scheme in ("http", "https") and not re.search(
                                r"\.(png|jpe?g|gif|svg|pdf|zip|gz|mp[34]|css|js|ico|woff2?)(\?|$)", pr.path, re.I):
                            queue.append((nxt, depth + 1))
            except Exception as e:
                self._inc("errors")
                self._say(f"Skipped {url}: {e}")

    # ---- fine-tuning in a subprocess
    def _train(self, steps: int):
        self._set(phase="training", step=0, total_steps=steps)
        if self.ckpt.exists():
            shutil.copy2(self.ckpt, self.ckpt.with_suffix(".prev.pt"))
        cmd = [sys.executable, "-u", str(ROOT / "train.py"), "--resume", "--steps", str(steps),
               "--batch-size", "16", "--lr", "5e-4", "--eval-every", str(max(50, steps // 3)),
               "--out", str(self.ckpt), "--data-dir", str(self.data_dir), "--no-demo"]
        dev = getattr(self.engine, "device", self.device)
        if dev and dev != "auto":
            cmd += ["--device", dev]
        self._say(f"Training {steps} steps on the collected data...")
        self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, encoding="utf-8", errors="replace")
        step_re = re.compile(r"step\s+(\d+)/(\d+)\s*\|\s*loss\s+([\d.]+).*?eta\s+([\d.]+)")
        val_re = re.compile(r"validation loss\s+([\d.]+)")
        tail = collections.deque(maxlen=8)
        for line in self.proc.stdout:
            line = line.rstrip()
            tail.append(line)
            m = step_re.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                base = total - steps
                self._set(step=max(0, done - base), loss=float(m.group(3)), eta_min=float(m.group(4)))
            v = val_re.search(line)
            if v:
                self._set(val_loss=float(v.group(1)))
                self._say(f"Validation loss {v.group(1)} (checkpoint saved)")
        rc = self.proc.wait()
        self.proc = None
        if rc != 0 and not self.stop_event.is_set():
            self._set(phase="error")
            self._say("Training failed: " + " | ".join(list(tail)[-3:]))
            return
        self._set(phase="reloading")
        if self.engine is not None:
            self._say("Loading the new weights...")
            self.engine.reload(str(self.ckpt))
        self._set(phase="stopped" if self.stop_event.is_set() else "done")
        self._say("Learning finished. The updated model is live." if not self.stop_event.is_set()
                  else "Stopped. The latest saved checkpoint is live.")


# =============================================================== command line
def main():
    ap = argparse.ArgumentParser(description="OryvexAI learn mode (collect public text, fine-tune)")
    ap.add_argument("--lang", nargs="*", default=["en"], choices=LANGS)
    ap.add_argument("--random", type=int, default=0, help="N random Wikipedia articles per language")
    ap.add_argument("--topics", nargs="*", default=[], help="Wikipedia topics to look up")
    ap.add_argument("--urls", nargs="*", default=[], help="extra pages to read (robots.txt obeyed)")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--max-pages", type=int, default=30)
    ap.add_argument("--consent", action="store_true", help="confirm you may use the --urls pages")
    ap.add_argument("--train", action="store_true", help="fine-tune after collecting")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()

    mgr = LearnManager()
    langs = a.lang if (a.random or a.topics) else []
    opts = {"langs": langs, "mode": "topics" if a.topics else "random", "topics": a.topics,
            "count": a.random or 40, "urls": a.urls, "depth": a.depth, "max_pages": a.max_pages,
            "consent": a.consent, "train": a.train, "steps": a.steps}
    err = mgr.start(opts)
    if err:
        sys.exit(err)
    shown = 0
    while mgr.thread.is_alive():
        time.sleep(0.5)
        log = mgr.status()["log"]
        for line in log[shown:]:
            print(line, flush=True)
        shown = len(log)
    for line in mgr.status()["log"][shown:]:
        print(line)


if __name__ == "__main__":
    main()
