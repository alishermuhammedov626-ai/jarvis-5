#!/usr/bin/env bash
#
# JARVIS 5 - one-shot server backtest.
#
#   bash scripts/run_server_backtest.sh [DAYS] [CONFIG]
#
#   DAYS    history to download and test (default 365)
#   CONFIG  optional JSON config (default: built-in defaults, 17x / 0.35 %)
#
# Steps: environment check -> tests -> data download (resumable) ->
#        primary backtest -> full validation suite -> archive.
#
# Long runs: launch it detached so an SSH drop cannot kill it --
#   nohup bash scripts/run_server_backtest.sh 365 > jarvis5.log 2>&1 &
#   tail -f jarvis5.log
set -euo pipefail

DAYS="${1:-365}"
CONFIG="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data}"
OUT_DIR="${OUT_DIR:-results}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
NAME="jarvis5_${DAYS}d_${STAMP}"
PY="${PYTHON:-python3}"

CFG_ARGS=()
[ -n "$CONFIG" ] && CFG_ARGS=(--config "$CONFIG")

echo "==================================================================="
echo " JARVIS 5 server backtest"
echo " days=$DAYS  config=${CONFIG:-<defaults>}  run=$NAME"
echo " started $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "==================================================================="

echo
echo "[1/5] Environment"
"$PY" --version
"$PY" -c "import requests" 2>/dev/null || {
    echo "  installing requests ..."; "$PY" -m pip install -r requirements.txt; }
mkdir -p "$DATA_DIR" "$OUT_DIR"

echo
echo "[2/5] Tests (no-look-ahead, determinism, rules)"
"$PY" -m unittest discover -s tests -q

echo
echo "[3/5] Historical data (resumable; re-running skips what is cached)"
"$PY" -m jarvis5 download --days "$DAYS" --data-dir "$DATA_DIR" "${CFG_ARGS[@]}"

echo
echo "[4/5] Primary backtest"
"$PY" -m jarvis5 backtest --days "$DAYS" --data-dir "$DATA_DIR" \
      --out "$OUT_DIR" --name "$NAME" "${CFG_ARGS[@]}"

echo
echo "[5/5] Full validation (walk-forward, Monte Carlo, robustness,"
echo "      parameter sensitivity, production gate)"
echo "      This is the long one - hours on a full year of data."
"$PY" -m jarvis5 full --days "$DAYS" --data-dir "$DATA_DIR" \
      --out "$OUT_DIR" --name "$NAME" "${CFG_ARGS[@]}"

ARCHIVE="${OUT_DIR}/${NAME}.tar.gz"
tar -czf "$ARCHIVE" -C "$OUT_DIR" $(cd "$OUT_DIR" && ls | grep "^${NAME}" | grep -v '\.tar\.gz$')

echo
echo "==================================================================="
echo " DONE  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " summary : ${OUT_DIR}/${NAME}_summary.txt"
echo " analysis: ${OUT_DIR}/${NAME}_analysis.json"
echo " archive : ${ARCHIVE}"
echo "==================================================================="
