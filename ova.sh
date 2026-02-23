#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

OVA_DIR="$ROOT_DIR/.ova"
BACKEND_PID="$OVA_DIR/backend.pid"
BACKEND_GROUP="$OVA_DIR/backend.group"
BACKEND_LOG="$OVA_DIR/backend.log"
FRONTEND_PID="$OVA_DIR/frontend.pid"
FRONTEND_GROUP="$OVA_DIR/frontend.group"
FRONTEND_LOG="$OVA_DIR/frontend.log"

BACKEND_PORT=5173
FRONTEND_PORT=8000

OVA_PROFILE="${OVA_PROFILE:-default}"

LLM="ministral-3:3b-instruct-2512-q4_K_M"
MLX_MODELS=("mlx-community/Kokoro-82M-4bit" "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-4bit" "mlx-community/parakeet-tdt-0.6b-v3")
CUDA_MODELS=("hexgrad/Kokoro-82M" "nvidia/parakeet-tdt-0.6b-v3" "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

usage() {
  cat <<'EOF'
Usage: ./ova.sh [OPTIONS] <command>

Options:
  OVA_PROFILE=<profile>  Set the profile to use (default: default)

Commands:
  install --cuda | --mlx
            Create uv env, install deps for CUDA/MLX, and fetch models
  start     Start backend + frontend server (non-blocking)
  stop      Stop running services
  help      Show this message

Example:
  ./ova.sh install --cuda
  OVA_PROFILE=dua ./ova.sh start
EOF
}

die() {
  echo "ova: $*" >&2
  exit 1
}

ensure_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing '$1' in PATH"
}

ensure_pip() {
  if uv run --no-sync python3 - <<'PY'
try:
    import pip  # noqa: F401
except Exception as exc:
    raise SystemExit(1) from exc
PY
  then
    echo "pip already available in uv venv"
    return 0
  fi

  echo "Installing pip into uv venv"
  if uv run --no-sync python3 -m ensurepip --upgrade; then
    return 0
  fi
  uv pip install --upgrade pip
}

is_running() {
  local pidfile=$1
  [[ -f "$pidfile" ]] || return 1
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  ps -p "$pid" >/dev/null 2>&1
}

port_open() {
  local port=$1
  python3 - <<PY
import socket
import sys

port = int("${port}")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.2)
try:
    sock.connect(("127.0.0.1", port))
    sock.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
}

wait_for_port() {
  local name=$1
  local port=$2
  local pidfile=$3
  local logfile=$4
  local timeout=${5:-10}
  local success_msg=${6:-}

  local start
  start="$(date +%s)"
  while true; do
    if ! is_running "$pidfile"; then
      echo "$name failed to start (process exited). See log: $logfile"
      return 1
    fi

    if port_open "$port"; then
      if [[ -n "$success_msg" ]]; then
        echo "$success_msg"
      else
        echo "$name started successfully (port $port)"
      fi
      return 0
    fi

    local now
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "$name failed to start within ${timeout}s. See log: $logfile"
      return 1
    fi

    sleep 0.25
  done
}

wait_for_log_message() {
  local name=$1
  local logfile=$2
  local pidfile=$3
  local message=$4
  local attempts=${5:-120}
  local interval=${6:-1}
  local success_msg=${7:-}

  for ((i=1; i<=attempts; i++)); do
    if ! is_running "$pidfile"; then
      echo "$name failed to start (process exited). See log: $logfile"
      return 1
    fi

    if [[ -f "$logfile" ]] && grep -Fq "$message" "$logfile"; then
      if [[ -n "$success_msg" ]]; then
        echo "$success_msg"
      else
        echo "$name started successfully"
      fi
      return 0
    fi

    sleep "$interval"
  done

  echo "$name failed to start within ${attempts}s. See log: $logfile"
  return 1
}

start_detached() {
  local logfile=$1
  local pidfile=$2
  local groupfile=$3
  shift 3

  mkdir -p "$OVA_DIR"
  : > "$logfile"

  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" >"$logfile" 2>&1 < /dev/null &
    echo "$!" > "$pidfile"
    echo "1" > "$groupfile"
  else
    nohup "$@" >"$logfile" 2>&1 < /dev/null &
    echo "$!" > "$pidfile"
    rm -f "$groupfile"
  fi
}

stop_service() {
  local name=$1
  local pidfile=$2
  local groupfile=$3

  if [[ ! -f "$pidfile" ]]; then
    echo "$name not running"
    return 0
  fi

  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]]; then
    rm -f "$pidfile" "$groupfile"
    echo "$name not running"
    return 0
  fi

  if ! ps -p "$pid" >/dev/null 2>&1; then
    rm -f "$pidfile" "$groupfile"
    echo "$name not running"
    return 0
  fi

  if [[ -f "$groupfile" ]]; then
    kill -- -"$pid" >/dev/null 2>&1 || true
  else
    kill "$pid" >/dev/null 2>&1 || true
  fi

  for _ in {1..25}; do
    if ps -p "$pid" >/dev/null 2>&1; then
      sleep 0.2
    else
      break
    fi
  done

  if ps -p "$pid" >/dev/null 2>&1; then
    if [[ -f "$groupfile" ]]; then
      kill -9 -- -"$pid" >/dev/null 2>&1 || true
    else
      kill -9 "$pid" >/dev/null 2>&1 || true
    fi
  fi

  rm -f "$pidfile" "$groupfile"
  echo "$name stopped"
}

ensure_ollama_model() {
  local model=$1
  local models
  if models="$(ollama list 2>/dev/null || true)"; then
    if echo "$models" | awk 'NR>1{print $1}' | grep -qx "$model"; then
      echo "Ollama model present: $model"
      return 0
    fi
  fi

  echo "Pulling Ollama model: $model"
  ollama pull "$model"
}

ensure_hf_models() {
  local backend=$1
  ensure_cmd uvx
  local -a models
  case "$backend" in
    cuda)
      models=("${CUDA_MODELS[@]}")
      ;;
    mlx)
      models=("${MLX_MODELS[@]}")
      ;;
    *)
      die "unknown backend for HF models: $backend (expected cuda or mlx)"
      ;;
  esac

  local cache_list
  cache_list="$(uvx hf cache list 2>/dev/null || true)"
  for repo_id in "${models[@]}"; do
    if [[ -n "$cache_list" ]] && echo "$cache_list" | grep -Fq "$repo_id"; then
      echo "HF model present: $repo_id"
    else
      echo "Downloading HF model: $repo_id"
      uvx hf download "$repo_id"
    fi
  done
}

cmd="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$cmd" in
  install)
    ensure_cmd uv
    ensure_cmd ollama
    install_backend=""
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --cuda)
          [[ -z "$install_backend" ]] || die "install accepts exactly one of --cuda or --mlx"
          install_backend="cuda"
          ;;
        --mlx)
          [[ -z "$install_backend" ]] || die "install accepts exactly one of --cuda or --mlx"
          install_backend="mlx"
          ;;
        *)
          die "unknown install option: $1 (expected --cuda or --mlx)"
          ;;
      esac
      shift
    done

    [[ -n "$install_backend" ]] || die "install requires one of --cuda or --mlx"

    printf 'backend=%s\n' "$install_backend" > "$ROOT_DIR/.config"
    echo "Wrote backend config: $ROOT_DIR/.config"

    uv venv --clear
    if [[ "$install_backend" == "cuda" ]]; then
      uv pip install -e ".[cuda]"
    else
      uv pip install --prerelease=allow -e ".[mlx]"
    fi
    ensure_ollama_model "$LLM"
    ensure_hf_models "$install_backend"
    ;;
  start)
    ensure_cmd uv
    echo "Starting Outrageous Voice Assistant..."
    echo "Starting front-end..."
    if is_running "$FRONTEND_PID"; then
      if port_open "$FRONTEND_PORT"; then
        echo "Front-end already running (port $FRONTEND_PORT)"
      else
        echo "Front-end running but not responding on port $FRONTEND_PORT"
      fi
    else
      start_detached "$FRONTEND_LOG" "$FRONTEND_PID" "$FRONTEND_GROUP" \
        uv run --no-sync python3 -m http.server
      wait_for_port "Web server" "$FRONTEND_PORT" "$FRONTEND_PID" "$FRONTEND_LOG" 5 \
        "Front-end started successfully (port $FRONTEND_PORT)"
    fi

    echo "Starting back-end..."
    if is_running "$BACKEND_PID"; then
      if port_open "$BACKEND_PORT"; then
        echo "Back-end already running (port $BACKEND_PORT)"
      else
        echo "Back-end running but not responding on port $BACKEND_PORT"
      fi
    else
      start_detached "$BACKEND_LOG" "$BACKEND_PID" "$BACKEND_GROUP" \
        bash -c "OVA_PROFILE='$OVA_PROFILE' uv run --no-sync uvicorn ova.api:app --reload --port \"$BACKEND_PORT\""
      wait_for_log_message "Backend" "$BACKEND_LOG" "$BACKEND_PID" \
        "Application startup complete." 60 1 \
        "Back-end started successfully (port $BACKEND_PORT)"
    fi
    echo "All services are up and running."
    echo "Now just fire up your browser and go to http://localhost:$FRONTEND_PORT and enjoy!!!"
    ;;
  stop)
    stop_service "Backend" "$BACKEND_PID" "$BACKEND_GROUP"
    stop_service "Web server" "$FRONTEND_PID" "$FRONTEND_GROUP"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    die "unknown command: $cmd"
    ;;
esac
