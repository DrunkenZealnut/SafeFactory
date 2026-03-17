#!/bin/bash
# SafeFactory gunicorn startup script
# macOS fork safety: must be set BEFORE Python starts
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "$0")"
source venv/bin/activate

# Kill existing process on port 5001
lsof -ti:5001 | xargs kill -9 2>/dev/null
sleep 1

nohup venv/bin/gunicorn web_app:app --bind 127.0.0.1:5001 --workers 2 --timeout 180 > app.log 2>&1 &
echo "SafeFactory started (PID: $!)"
