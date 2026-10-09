#!/usr/bin/env bash
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Done. Run ./run_web.sh to chat, or ./train.sh to train more."
