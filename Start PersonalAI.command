#!/bin/zsh

cd "$HOME/PersonalAI" || exit 1

CHAT_PID=""
CHAT_RUNNING=0

cleanup() {
  if [[ -n "$CHAT_PID" ]]; then
    kill "$CHAT_PID" 2>/dev/null || true
  fi
}

pause_before_close() {
  echo
  read -r "REPLY?Press Return to close this window. "
}

trap 'cleanup; exit 130' INT TERM
trap cleanup EXIT

if ! curl -fsS --max-time 3 http://127.0.0.1:1234/v1/models | grep -q 'qwen/qwen3.5-2b'; then
  echo "LM Studio is not ready with Qwen 2B. Open LM Studio, load Qwen3.5 2B, then run this launcher again."
  pause_before_close
  exit 1
fi

if curl -fsS --max-time 2 http://127.0.0.1:8000/ >/dev/null; then
  CHAT_RUNNING=1
elif lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 8000 is occupied, but the PersonalAI page did not respond. Close the other process using that port, then try again."
  pause_before_close
  exit 1
fi

if (( CHAT_RUNNING )); then
  echo "PersonalAI is already running at http://127.0.0.1:8000."
  pause_before_close
  exit 0
fi

./.venv/bin/python app.py &
CHAT_PID=$!
echo "Starting PersonalAI chat..."
for attempt in {1..30}; do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/ >/dev/null; then
    CHAT_RUNNING=1
    break
  fi
  if ! kill -0 "$CHAT_PID" 2>/dev/null; then
    echo "PersonalAI stopped before it became ready. See the error above."
    exit 1
  fi
  sleep 1
done
if (( ! CHAT_RUNNING )); then
  echo "PersonalAI did not become ready within 30 seconds. See the error above."
  exit 1
fi

echo ""
echo "PersonalAI is ready at http://127.0.0.1:8000."
echo "Leave this window open while using the assistant. Press Control-C to stop services started by this launcher."

if [[ -n "$CHAT_PID" ]]; then
  wait "$CHAT_PID"
fi
