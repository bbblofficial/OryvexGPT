"""
Inference engine: loads a trained OryvexGPT checkpoint and chats with memory.
Used by both the terminal chat and the web panel.
"""
from __future__ import annotations

import codecs
import threading
from dataclasses import dataclass, field

import torch

from model import GPTConfig, OryvexGPT
from tokenizer import BPETokenizer, encode_chat, EOS, BYTE_OFFSET, PAD, BOS, SYSTEM, USER, ASSISTANT

BOT_NAME = "OryvexAI"
DEFAULT_CKPT = "checkpoints/oryvex.pt"
MAX_USER_CHARS = 2000


@dataclass
class GenConfig:
    max_new_tokens: int = 200
    temperature: float = 0.5   # 0 = greedy. Raise for more variety.
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.0  # try 1.05-1.15 if replies loop


@dataclass
class ChatSession:
    messages: list = field(default_factory=list)    # what the model sees (trimmed to fit context)
    transcript: list = field(default_factory=list)  # full log shown in the web UI

    def reset(self):
        self.messages.clear()
        self.transcript.clear()


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


class OryvexEngine:
    def __init__(self, model: OryvexGPT, tok: BPETokenizer, system: str, device: str,
                 gen: GenConfig | None = None, meta: dict | None = None):
        self.model, self.tok, self.system, self.device = model, tok, system, device
        self.gen = gen or GenConfig()
        self.meta = meta or {}
        self.lock = threading.Lock()  # one generation at a time
        self.block = model.cfg.block_size

    @classmethod
    def load(cls, path: str = DEFAULT_CKPT, device: str = "auto", gen: GenConfig | None = None,
             system: str | None = None):
        device = pick_device(device)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = OryvexGPT(GPTConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        tok = BPETokenizer([tuple(m) for m in ckpt["merges"]])
        return cls(model, tok, system or ckpt["system"], device, gen, ckpt.get("meta", {}))

    # ------------------------------------------------------------- prompt
    def _build_ids(self, session: ChatSession) -> list[int]:
        budget = self.block - 48  # leave room to write; the window slides if a reply runs long
        while True:
            ids, _ = encode_chat(self.tok, session.messages, self.system, add_generation_prompt=True)
            if len(ids) <= budget or len(session.messages) <= 1:
                break
            session.messages.pop(0)  # forget the oldest turn first
            while len(session.messages) > 1 and session.messages[0]["role"] != "user":
                session.messages.pop(0)
        return ids[:1] + ids[-(budget - 1):] if len(ids) > budget else ids

    # ----------------------------------------------------------- sampling
    def _sample(self, logits: torch.Tensor, produced: list[int]) -> int:
        g = self.gen
        logits = logits.float().clone()
        logits[[PAD, BOS, SYSTEM, USER, ASSISTANT]] = float("-inf")
        if g.repetition_penalty != 1.0 and produced:
            idx = torch.tensor(sorted(set(produced)), device=logits.device)
            vals = logits[idx]
            logits[idx] = torch.where(vals > 0, vals / g.repetition_penalty, vals * g.repetition_penalty)
        if g.temperature <= 0:
            return int(torch.argmax(logits))
        logits = logits / g.temperature
        if g.top_k and g.top_k < logits.numel():
            kth = torch.topk(logits, g.top_k).values[-1]
            logits[logits < kth] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        if 0 < g.top_p < 1:
            sp, si = torch.sort(probs, descending=True)
            cut = torch.cumsum(sp, dim=0) - sp > g.top_p
            sp[cut] = 0
            probs = torch.zeros_like(probs).scatter(0, si, sp)
            probs = probs / probs.sum()
        return int(torch.multinomial(probs, 1))

    @torch.inference_mode()
    def _generate(self, ids: list[int], cancel: threading.Event):
        x = torch.tensor([ids], device=self.device)
        logits, _, past = self.model(x, use_cache=True)
        total, produced = len(ids), []
        for _ in range(self.gen.max_new_tokens):
            if cancel.is_set():
                return
            nxt = self._sample(logits[0, -1], produced)
            if nxt < BYTE_OFFSET:  # <eos> (or any special) ends the reply
                return
            produced.append(nxt)
            yield nxt
            if total + 1 > self.block:  # context full: slide the window and re-read the tail
                ctx = (ids + produced)[-(self.block // 2):]
                logits, _, past = self.model(torch.tensor([ctx], device=self.device), use_cache=True)
                total = len(ctx)
            else:
                step = torch.tensor([[nxt]], device=self.device)
                logits, _, past = self.model(step, past=past, use_cache=True)
                total += 1

    # -------------------------------------------------------------- public
    def stream_reply(self, session: ChatSession, user_text: str, cancel: threading.Event | None = None):
        """Generator yielding the reply in text chunks; updates the session memory."""
        cancel = cancel or threading.Event()
        user_text = user_text.strip()[:MAX_USER_CHARS]
        with self.lock:
            user_msg = {"role": "user", "content": user_text}
            session.messages.append(user_msg)
            session.transcript.append(dict(user_msg))
            parts: list[str] = []
            try:
                ids = self._build_ids(session)
                dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
                for t in self._generate(ids, cancel):
                    piece = dec.decode(self.tok.token_bytes[t])
                    if piece:
                        parts.append(piece)
                        yield piece
                tail = dec.decode(b"", final=True)
                if tail:
                    parts.append(tail)
                    yield tail
            finally:
                reply = "".join(parts).strip()
                if reply:
                    session.messages.append({"role": "assistant", "content": reply})
                    session.transcript.append({"role": "assistant", "content": reply})
                else:
                    if user_msg in session.messages:
                        session.messages.remove(user_msg)
                    session.transcript.pop()

    def reply(self, session: ChatSession, user_text: str) -> str:
        return "".join(self.stream_reply(session, user_text))

    def info(self) -> dict:
        params = sum(p.numel() for p in self.model.parameters())
        return {
            "name": BOT_NAME,
            "model": f"OryvexGPT {params / 1e6:.1f}M",
            "params": params,
            "device": self.device,
            "context": self.block,
            "steps": self.meta.get("steps"),
            "val_loss": self.meta.get("val_loss"),
        }
