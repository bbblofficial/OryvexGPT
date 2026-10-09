# OryvexAI (from scratch)

A language model that is entirely yours: **its own architecture, its own
tokenizer, its own parameters (weights), trained by you**. Nothing is
downloaded, nothing is pretrained, no Ollama, no external API.

| Part | File | What it is |
|---|---|---|
| Tokenizer | `tokenizer.py` | Byte-level BPE written from scratch (works for English, Persian, any language) |
| Model | `model.py` | GPT-style transformer in PyTorch (attention, MLP, KV cache) |
| Training data | `datagen.py`, `data/` | Starter chat dataset (English + Persian) that you can extend |
| Training | `train.py` | Learns the tokenizer and the weights, saves `checkpoints/oryvex.pt` |
| Chat + memory | `engine.py`, `app.py` | Terminal chat |
| Web panel | `server.py`, `web/index.html` | Chat UI in your browser |

## Run it

1. Install Python 3.9+ and run `install.bat` (Windows) or `./install.sh` (macOS/Linux).
   Or just: `pip install torch`
2. Chat:
   - Web panel: `run_web.bat` / `./run_web.sh`  (http://127.0.0.1:8000)
   - Terminal: `run_cli.bat` / `./run_cli.sh`

The zip includes a **starter checkpoint** (`checkpoints/oryvex.pt`) that was
trained briefly, so it answers immediately. It is deliberately small and
undertrained. Train it properly on your machine for much better results:

    python train.py                  # auto size: "nano" on CPU, "small" on GPU
    python train.py --steps 6000     # train longer
    python train.py --resume --steps 3000   # keep improving the current model

Training prints loss every 50 steps and saves a checkpoint every 250 steps.
Ctrl+C stops safely and saves. First run builds the tokenizer (about 15 seconds).

## Model sizes (parameters)

| Preset | Parameters | Good for |
|---|---|---|
| `nano` | ~3.9M | CPU, trains in about an hour |
| `small` | ~11.7M | GPU or patient CPU |
| `base` | ~26.8M | GPU (8 GB+) |
| `large` | ~87M | strong GPU |

    python train.py --preset small
    python train.py --preset base --block-size 512 --batch-size 16

Changing preset or tokenizer settings means training a new model (no `--resume`).
Delete `checkpoints/oryvex.pt` first, or use `--out checkpoints/other.pt`.

## Teaching it your own knowledge

A from-scratch model only knows what is in its training data. Out of the box
that is the starter set from `datagen.py`: identity, small talk, arithmetic,
capitals, short explanations, Python snippets, and remembering facts you tell it
during a chat (your name, your city, a number...).

Add data, then retrain:

- **Conversations**: put a `.jsonl` file in `data/`, one conversation per line:

      {"messages":[{"role":"user","content":"What is Docker?"},{"role":"assistant","content":"Docker packages an app and its dependencies into a container."}]}

- **Plain text**: put `.txt` files in `data/` (books, articles, notes). Paragraphs are
  separated by blank lines. This teaches fluent language, but not chat behavior.

Then `python train.py --out checkpoints/oryvex.pt` (fresh tokenizer, fresh weights).
Delete `data/chat.jsonl` to regenerate the starter set, or edit `datagen.py`.

## Be realistic about quality

- A model with 4 to 90 million parameters, trained on one laptop, is a learning
  project, not a ChatGPT replacement. It will be good at what is in its data and
  will make things up outside it.
- Quality comes from **data quantity and variety**, then **model size and training
  time**. Hundreds of MB of clean text/conversations plus a GPU is where it starts
  to feel general-purpose.
- Lower `--temperature` (0.2 to 0.5) gives steadier answers, higher (0.8+) more
  variety. Try `--repetition-penalty 1.1` if it loops.

## Options

    python app.py --help
    python train.py --help

Web panel binds to 127.0.0.1 by default. `--host 0.0.0.0` exposes it to your
network with no login; only do that on networks you trust.

## Host it for testing: GitHub + GitHub Actions + trycloudflare

1. Create a new GitHub repository and push this folder (the trained
   `checkpoints/oryvex.pt` is about 15 MB, which is fine for a normal push):

       git init
       git add .
       git commit -m "OryvexAI"
       git branch -M main
       git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
       git push -u origin main

2. On GitHub open **Actions** -> **Host OryvexAI (trycloudflare test)** ->
   **Run workflow**, choose how many minutes to stay online (max 330).
3. When the job reaches "Start Cloudflare quick tunnel", the log and the job
   summary show a link like `https://something-random.trycloudflare.com`.

Things to know:
- The link changes on every run and dies when the job ends. This is for testing,
  not permanent hosting.
- The panel has no login. Anyone who gets the link can chat with it and use your
  runner's minutes, so share it carefully.
- Runs on the free GitHub CPU runner, so replies are slower than on a GPU.
- A permanent site needs a real server or VPS (run `python app.py --web --host 0.0.0.0`
  behind a reverse proxy and add authentication).

## Known weaknesses of the shipped starter model

It is a 3.9M-parameter model trained for 1800 steps on a single slow CPU core.
It answers capitals, definitions, Python snippets, Persian greetings and
"remember my name" well. It still gets **arithmetic wrong**, sometimes answers
"Who are you?" incorrectly in English, and may greet with an invented name.
These are data/training limits, not bugs: add more examples to `data/`, then
train longer (`python train.py --resume --steps 3000`) or use `--preset small`.
