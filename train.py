#!/usr/bin/env python3
"""
Train OryvexAI from scratch: tokenizer + transformer weights, nothing pretrained.

    python train.py                         # auto-pick a size for your hardware
    python train.py --preset small --steps 6000
    python train.py --resume                # continue from checkpoints/oryvex.pt

Data it reads from ./data (created automatically on first run):
    *.jsonl   one conversation per line: {"messages":[{"role":"user","content":"..."},
                                                      {"role":"assistant","content":"..."}]}
    *.txt     plain text (trains next-word prediction; paragraphs split by blank lines)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import sys
import time

import torch

import datagen
from model import GPTConfig, OryvexGPT, PRESETS, preset_config
from tokenizer import BPETokenizer, encode_chat, BOS, EOS
from engine import pick_device


def load_examples(data_dir: str, dataset_size: int):
    os.makedirs(data_dir, exist_ok=True)
    jsonl = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not [p for p in jsonl if os.path.basename(p) != "learned.jsonl"]:
        n = datagen.write_dataset(os.path.join(data_dir, "chat.jsonl"), dataset_size)
        print(f"[data] generated starter dataset: {n} conversations -> {data_dir}/chat.jsonl")
        jsonl = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))

    chats, learned_chats = [], []
    for path in jsonl:
        is_learned = os.path.basename(path) == "learned.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                msgs = json.loads(line)["messages"]
                system = next((m["content"] for m in msgs if m["role"] == "system"), datagen.SYSTEM)
                turns = [m for m in msgs if m["role"] in ("user", "assistant")]
                if turns and turns[-1]["role"] == "assistant":
                    (learned_chats if is_learned else chats).append((system, turns))

    raws, learned_raws = [], []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            paras = [p.strip() for p in f.read().split("\n\n") if p.strip()]
        (learned_raws if os.path.basename(path) == "learned.txt" else raws).extend(paras)
    print(f"[data] {len(chats)} conversations, {len(raws)} raw paragraphs | "
          f"learned from the web: {len(learned_chats)} conversations, {len(learned_raws)} paragraphs")
    return chats, raws, learned_chats, learned_raws


def build_streams(tok, chats, raws, seed=0, learned=((), ())):
    """Pack examples into one token stream (+ loss mask). Web-learned data is oversampled
    so it makes up roughly half of the tokens; otherwise a few hundred new examples would
    be drowned out by the base set."""
    def enc(chat_list, raw_list):
        out = [encode_chat(tok, turns, system) for system, turns in chat_list]
        for p in raw_list:
            ids = [BOS] + tok.encode(p, special=False) + [EOS]
            out.append((ids, [1] * len(ids)))
        return out

    base = enc(chats, raws)
    random.Random(seed).shuffle(base)
    n_val = max(1, len(base) // 33)
    val, train = base[:n_val], base[n_val:]

    extra = enc(*learned)
    if extra:
        base_tokens = sum(len(i) for i, _ in train)
        extra_tokens = max(1, sum(len(i) for i, _ in extra))
        rep = max(1, min(20, round(0.5 * base_tokens / extra_tokens)))
        print(f"[data] web-learned data repeated x{rep} per epoch ({extra_tokens} tokens)")
        train = train + extra * rep
        random.Random(seed + 1).shuffle(train)

    def pack(exs):
        ids, mask = [], []
        for i, m in exs:
            ids += i
            mask += m
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    return pack(train), pack(val)


def get_batch(stream, block, batch, device):
    ids, mask = stream
    starts = torch.randint(0, len(ids) - block - 1, (batch,))
    idx = starts[:, None] + torch.arange(block)[None, :]
    x, y, m = ids[idx], ids[idx + 1], mask[idx + 1]
    y = y.masked_fill(~m, -100)  # only learn from assistant text
    return x.to(device), y.to(device)


def lr_at(step, total, base, warmup=100, floor=0.1):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t)))


def save_ckpt(path, model, tok, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "config": model.cfg.to_dict(),
        "merges": [list(m) for m in tok.merges],
        "system": datagen.SYSTEM,
        "meta": meta,
    }, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, stream, block, batch, device, iters=20):
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(stream, block, batch, device)
        total += model(x, y)[1].item()
    model.train()
    return total / iters


def main():
    p = argparse.ArgumentParser(description="Train OryvexAI from scratch")
    p.add_argument("--preset", default="auto", choices=["auto", *PRESETS])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--block-size", type=int, default=None, help="context length (tokens)")
    p.add_argument("--vocab-size", type=int, default=2500)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--dataset-size", type=int, default=12000, help="size of the generated starter set")
    p.add_argument("--out", default="checkpoints/oryvex.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--resume", action="store_true", help="continue training from --out")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--no-demo", action="store_true", help="skip the quick test at the end")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    preset = a.preset if a.preset != "auto" else ("small" if device == "cuda" else "nano")
    defaults = {"nano": (3000, 32, 2e-3), "small": (5000, 32, 1e-3),
                "base": (8000, 32, 6e-4), "large": (12000, 24, 3e-4)}[preset]
    steps = a.steps or defaults[0]
    batch = a.batch_size or (defaults[1] if device != "cpu" else min(defaults[1], 16))
    lr = a.lr or defaults[2]

    chats, raws, l_chats, l_raws = load_examples(a.data_dir, a.dataset_size)

    if a.resume and os.path.exists(a.out):
        ck = torch.load(a.out, map_location="cpu", weights_only=True)
        tok = BPETokenizer([tuple(m) for m in ck["merges"]])
        cfg = GPTConfig(**ck["config"])
        start = ck.get("meta", {}).get("steps", 0)
        print(f"[resume] {a.out} (already trained {start} steps)")
    else:
        ck, start = None, 0
        print("[tokenizer] training byte-level BPE from scratch...")
        texts = ([datagen.SYSTEM] + [m["content"] for _, turns in chats + l_chats for m in turns]
                 + raws + l_raws)
        tok = BPETokenizer().train(texts, a.vocab_size)
        cfg = preset_config(preset, tok.vocab_size, a.block_size, a.dropout)

    model = OryvexGPT(cfg)
    if ck:
        model.load_state_dict(ck["model"])
    model.to(device).train()
    train_s, val_s = build_streams(tok, chats, raws, a.seed, (l_chats, l_raws))
    if len(train_s[0]) <= cfg.block_size + 2:
        sys.exit("Not enough training data for this block size. Add more data to ./data")

    n_params = model.num_params()
    print(f"[model] OryvexGPT '{preset}': {n_params / 1e6:.2f}M parameters | vocab {cfg.vocab_size} | "
          f"context {cfg.block_size} | layers {cfg.n_layer} heads {cfg.n_head} width {cfg.n_embd}")
    print(f"[train] device {device} | {len(train_s[0]) / 1e6:.2f}M training tokens | "
          f"batch {batch} x {cfg.block_size} | {steps} steps | peak lr {lr}")

    decay = [p_ for p_ in model.parameters() if p_.dim() >= 2]
    no_decay = [p_ for p_ in model.parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    meta = {"preset": preset, "steps": start, "val_loss": None, "params": n_params}
    t0, run_loss, best_val = time.time(), 0.0, float("inf")
    step = start
    try:
        while step < start + steps:
            for g in opt.param_groups:
                g["lr"] = lr_at(step - start, steps, lr, warmup=min(100, max(10, steps // 10)))
            x, y = get_batch(train_s, cfg.block_size, batch, device)
            with torch.autocast(device_type="cuda" if use_amp else "cpu", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)[:2]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            step += 1

            if step % 50 == 0:
                el = time.time() - t0
                eta = el / (step - start) * (start + steps - step)
                print(f"step {step:5d}/{start + steps} | loss {run_loss / 50:.4f} | "
                      f"{el:6.0f}s elapsed | eta {eta / 60:5.1f} min", flush=True)
                run_loss = 0.0
            if step % a.eval_every == 0 or step == start + steps:
                val = evaluate(model, val_s, cfg.block_size, batch, device)
                meta.update(steps=step, val_loss=round(val, 4))
                print(f"  >> validation loss {val:.4f}  (saved {a.out})", flush=True)
                save_ckpt(a.out, model, tok, meta)
    except KeyboardInterrupt:
        print("\n[train] interrupted - saving what we have...")
        meta["steps"] = step
        save_ckpt(a.out, model, tok, meta)

    print(f"[done] {a.out} | {n_params / 1e6:.2f}M parameters | {step} steps | {time.time() - t0:.0f}s")
    if not a.no_demo:
        demo(a.out, device)


def demo(path, device):
    from engine import OryvexEngine, ChatSession
    eng = OryvexEngine.load(path, device)
    s = ChatSession()
    print("\n--- quick test ---")
    for q in ["Hi", "Who are you?", "What is 17 + 25?", "My name is Dara.", "What's my name?"]:
        print(f"you > {q}\nbot > {eng.reply(s, q)}")


if __name__ == "__main__":
    main()
