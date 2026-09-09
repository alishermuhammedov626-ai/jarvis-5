#!/usr/bin/env bash
#
# JARVIS 5 — market move cause research.
#
#   bash scripts/run_research.sh [DAYS] [CONFIG]
#
# Stage 1 (synthetic) always runs first and stage 2 is blocked unless it
# passes.  The trading engine is NOT touched by any of this.
#
# Long runs: launch detached so an SSH drop cannot kill it --
#   nohup bash scripts/run_research.sh 365 > research.log 2>&1 &
#   tail -f research.log
set -euo pipefail

DAYS="${1:-365}"
CONFIG="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data}"
OUT_DIR="${OUT_DIR:-reports}"
PY="${PYTHON:-python3}"
COINS="${COINS:-DOGEUSDT,1000PEPEUSDT,1000SHIBUSDT,PUMPUSDT,1000BONKUSDT,WIFUSDT}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"

CFG_ARGS=()
[ -n "$CONFIG" ] && CFG_ARGS=(--config "$CONFIG")

echo "==================================================================="
echo " JARVIS 5 RESEARCH   days=$DAYS   config=${CONFIG:-<defaults>}"
echo " started $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "==================================================================="

echo
echo "[1/4] Tests"
"$PY" -m unittest discover -s tests -q

echo
echo "[2/4] Stage 1 — synthetic validation (no real data touched)"
"$PY" -m jarvis5.research.synthetic --cycles 6 --out "$OUT_DIR" "${CFG_ARGS[@]}"

echo
echo "[3/4] Historical data (resumable)"
"$PY" -m jarvis5 download --days "$DAYS" --data-dir "$DATA_DIR" --symbols "$COINS"

echo
echo "[4/4] Stage 2 — research on real data"
"$PY" -m jarvis5.research --coins "$COINS" --days "$DAYS" \
      --data-dir "$DATA_DIR" --out "$OUT_DIR" "${CFG_ARGS[@]}"

ARCHIVE="${OUT_DIR}/research_${DAYS}d_${STAMP}.tar.gz"
tar -czf "$ARCHIVE" -C "$OUT_DIR" . --exclude='*.tar.gz'

echo
echo "==================================================================="
echo " DONE  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " read first: ${OUT_DIR}/FINAL_REPORT.md"
echo " archive   : ${ARCHIVE}"
echo "==================================================================="
