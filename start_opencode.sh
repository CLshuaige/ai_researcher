#!/usr/bin/env bash
set -euo pipefail

OPENCODE_PATTERN="opencode serve"
CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/configs/opencode.json"
WAIT_SECONDS=3

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

############################
# 1. Check if OpenCode is installed
############################
if ! command -v opencode &> /dev/null; then
  log "ERROR: OpenCode is not installed."
  log "Please install OpenCode first:"
  log "  npm install -g opencode-ai"
  log "  # or"
  log "  curl -fsSL https://opencode.ai/install | bash"
  exit 1
fi

log "OpenCode version: $(opencode --version 2>&1 || echo 'unknown')"

############################
# 2. Stop existing OpenCode server
############################
log "Checking existing OpenCode server processes..."
if pgrep -f "$OPENCODE_PATTERN" > /dev/null; then
  log "Stopping existing OpenCode server (SIGTERM)..."
  pkill -15 -f "$OPENCODE_PATTERN"
  sleep $WAIT_SECONDS
fi

# Fallback force kill
if pgrep -f "$OPENCODE_PATTERN" > /dev/null; then
  log "Force killing remaining OpenCode server (SIGKILL)..."
  pkill -9 -f "$OPENCODE_PATTERN"
fi

############################
# 3. Read port from config
############################
if ! PORT=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['server']['port'])"); then
  log "ERROR: Could not read port from config file: $CONFIG_PATH"
  exit 1
fi

log "Using port $PORT from configuration"

############################
# 4. Clean up port
############################
log "Checking port $PORT status..."
if lsof -i :$PORT > /dev/null 2>&1; then
  log "Port $PORT is occupied. Cleaning up..."
  lsof -i :$PORT || true

  # Kill processes using the port
  lsof -ti :$PORT | xargs kill -9 2>/dev/null || true
  sleep 2
fi

############################
# 5. Final port check
############################
if lsof -i :$PORT > /dev/null 2>&1; then
  log "ERROR: Port $PORT is still not clean. Abort."
  lsof -i :$PORT
  exit 1
fi

log "Port $PORT is clean."

############################
# 6. Verify configuration file
############################
if [ ! -f "$CONFIG_PATH" ]; then
  log "ERROR: Configuration file not found: $CONFIG_PATH"
  exit 1
fi

log "Using configuration: $CONFIG_PATH"

############################
# 7. Start OpenCode server
############################
log "Starting OpenCode server..."

# Set config via environment variable
export OPENCODE_CONFIG="$CONFIG_PATH"

exec opencode serve \
  --hostname 127.0.0.1
