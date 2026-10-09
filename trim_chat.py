"""Undo Wikipedia appends: keep only the first N lines of data/chat.jsonl (a backup is saved first).

    python trim_chat.py 12000          # the starter dataset has 12000 lines
"""
import shutil
import sys
from pathlib import Path

path = Path("data/chat.jsonl")
if len(sys.argv) != 2 or not sys.argv[1].isdigit():
    sys.exit(__doc__)
keep = int(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
if len(lines) <= keep:
    sys.exit(f"{path} has {len(lines)} lines, nothing to trim.")
shutil.copy(path, path.with_suffix(".jsonl.bak"))
path.write_text("\n".join(lines[:keep]) + "\n", encoding="utf-8")
print(f"Kept {keep} lines, removed {len(lines) - keep}. Backup: {path.with_suffix('.jsonl.bak')}")
