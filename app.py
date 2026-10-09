#!/usr/bin/env python3
"""
Chat with your from-scratch OryvexAI.

    python app.py                 # terminal chat
    python app.py --web           # web panel at http://127.0.0.1:8000
    python app.py --temperature 0.8 --ckpt checkpoints/oryvex.pt
"""
from __future__ import annotations

import argparse
import os
import sys

from engine import BOT_NAME, DEFAULT_CKPT, ChatSession, GenConfig, OryvexEngine


class C:
    GREEN, CYAN, DIM, RED, BOLD, RESET = "\033[92m", "\033[96m", "\033[2m", "\033[91m", "\033[1m", "\033[0m"


def run_cli(engine: OryvexEngine):
    os.system("")  # enable ANSI colors on Windows terminals
    info = engine.info()
    session = ChatSession()
    print(f"\n{C.BOLD}{BOT_NAME}{C.RESET}  {C.DIM}{info['model']} ({info['params']:,} parameters) | "
          f"context {info['context']} tokens | device {info['device']}{C.RESET}")
    print(f"{C.DIM}Trained from scratch. No Ollama, no API.  Commands: /reset  /info  /exit  (Ctrl+C stops a reply){C.RESET}\n")
    while True:
        try:
            user = input(f"{C.GREEN}{C.BOLD}You     >{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not user:
            continue
        if user.startswith("/"):
            cmd = user.lower()
            if cmd in ("/exit", "/quit", "/q"):
                print("Bye.")
                return
            if cmd == "/reset":
                session.reset()
                print(f"{C.DIM}Memory cleared.{C.RESET}")
            elif cmd == "/info":
                print(f"{C.DIM}{info}{C.RESET}")
            else:
                print(f"{C.DIM}/reset  clear memory\n/info   model details\n/exit   quit{C.RESET}")
            continue
        print(f"{C.CYAN}{C.BOLD}{BOT_NAME} >{C.RESET} ", end="", flush=True)
        gen = engine.stream_reply(session, user)
        try:
            for chunk in gen:
                print(chunk, end="", flush=True)
            print("\n")
        except KeyboardInterrupt:
            gen.close()
            print(f"\n{C.DIM}[stopped]{C.RESET}\n")


def main():
    p = argparse.ArgumentParser(description=f"{BOT_NAME}: chat with your own from-scratch model")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="trained checkpoint (made by train.py)")
    p.add_argument("--web", action="store_true", help="start the web panel instead of the terminal chat")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--temperature", type=float, default=GenConfig.temperature, help="0 = greedy, higher = more varied")
    p.add_argument("--top-k", type=int, default=GenConfig.top_k)
    p.add_argument("--top-p", type=float, default=GenConfig.top_p)
    p.add_argument("--repetition-penalty", type=float, default=GenConfig.repetition_penalty)
    p.add_argument("--max-new-tokens", type=int, default=GenConfig.max_new_tokens)
    p.add_argument("--system", default=None, help="override the system prompt (works best if it matches training)")
    a = p.parse_args()

    if not os.path.exists(a.ckpt):
        sys.exit(f"No trained model at '{a.ckpt}'.\nTrain one first:  python train.py")
    gen = GenConfig(a.max_new_tokens, a.temperature, a.top_k, a.top_p, a.repetition_penalty)
    engine = OryvexEngine.load(a.ckpt, a.device, gen, a.system)
    if a.web:
        from server import run_web
        run_web(engine, a.host, a.port, not a.no_browser)
    else:
        run_cli(engine)


if __name__ == "__main__":
    main()
