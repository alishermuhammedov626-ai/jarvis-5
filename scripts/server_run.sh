#!/usr/bin/env bash
#
# ============================================================================
#  JARVIS 5 — full server run
#  preflight -> tests -> synthetic gate -> download -> backtest -> research
# ============================================================================
#
#  Usage, from the project root on the server:
#
#      bash scripts/server_run.sh                 # 365 days, all 6 coins
#      bash scripts/server_run.sh 90              # shorter first pass
#      bash scripts/server_run.sh 365 fast        # research stride 15 (~3x faster)
#      bash scripts/server_run.sh 365 backtest    # skip the research stage
#      bash scripts/server_run.sh 365 research    # skip the trading backtest
#
#  This takes hours. Run it detached so an SSH drop cannot kill it:
#
#      nohup bash scripts/server_run.sh 365 > jarvis5_run.log 2>&1 &
#      tail -f jarvis5_run.log
#
#  Every stage is resumable: the kline cache is only topped up, so re-running
#  after an interruption does not re-download what is already on disk.
#
#  Environment overrides:
#      SKIP_PREFLIGHT=1   proceed even though the preflight check failed
#      SKIP_DOWNLOAD=1    use the candles already in DATA_DIR
#      SKIP_TESTS=1       skip the test suite (not recommended)
#      COINS=...          override the symbol list
#      DATA_DIR / OUT_DIR / REPORT_DIR / VENV / PYTHON
# ============================================================================
set -euo pipefail

case "${1:-}" in
    -h|--help|help)
        sed -n '3,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
esac

DAYS="${1:-365}"
MODE="${2:-all}"            # all | fast | backtest | research

[[ "$DAYS" =~ ^[0-9]+$ ]] || { echo "DAYS must be a number, got '$DAYS'"; exit 2; }
case "$MODE" in
    all|fast|backtest|research) ;;
    *) echo "MODE must be all|fast|backtest|research, got '$MODE'"; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${DATA_DIR:-data}"
OUT_DIR="${OUT_DIR:-results}"
REPORT_DIR="${REPORT_DIR:-reports}"
COINS="${COINS:-DOGEUSDT,1000PEPEUSDT,1000SHIBUSDT,PUMPUSDT,1000BONKUSDT,WIFUSDT}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
NAME="jarvis5_${DAYS}d_${STAMP}"
VENV="${VENV:-.venv}"

RESEARCH_ARGS=()
[ "$MODE" = "fast" ] && RESEARCH_ARGS=(--config config/research_fast.json)

step() { printf '\n\033[1m========== %s ==========\033[0m\n' "$1"; }
fail() { printf '\n!! %s\n' "$1" >&2; exit 1; }

echo "============================================================"
echo " JARVIS 5 SERVER RUN"
echo " days=$DAYS  mode=$MODE  run=$NAME"
echo " host=$(hostname)  started $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"

# ---------------------------------------------------------------- 1. python
step "1/7  Python environment"
PY_BIN="${PYTHON:-python3}"
command -v "$PY_BIN" >/dev/null || fail "python3 not found. On Debian/Ubuntu:
   apt-get update && apt-get install -y python3 python3-venv python3-pip"

if [ ! -d "$VENV" ]; then
    echo "creating virtualenv in $VENV ..."
    "$PY_BIN" -m venv "$VENV" 2>/dev/null || fail "python3-venv is missing. Run:
   apt-get install -y python3-venv"
fi
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -V
pip install --quiet --upgrade pip >/dev/null 2>&1 || true
pip install --quiet -r requirements.txt
echo "dependencies ready (requests only; the engine is pure stdlib)"

# ------------------------------------------------------------- 2. preflight
step "2/7  Preflight"
mkdir -p "$DATA_DIR" "$OUT_DIR" "$REPORT_DIR"
if ! python scripts/preflight.py --days "$DAYS" --symbols "$COINS" \
        --data-dir "$DATA_DIR" --json "$OUT_DIR/preflight_${STAMP}.json"; then
    if [ "${SKIP_PREFLIGHT:-0}" = "1" ]; then
        echo "preflight failed but SKIP_PREFLIGHT=1 — continuing anyway"
    else
        fail "preflight failed — fix the reported issue before starting a long run.
   To inspect without the network:  python scripts/preflight.py --offline
   To override deliberately:        SKIP_PREFLIGHT=1 bash scripts/server_run.sh $DAYS"
    fi
fi

# ----------------------------------------------------------------- 3. tests
step "3/7  Tests (no look-ahead, determinism, rules, research)"
if [ "${SKIP_TESTS:-0}" = "1" ]; then
    echo "SKIPPED (SKIP_TESTS=1)"
else
    python -m unittest discover -s tests -q || fail "tests failed — not proceeding"
fi

# ------------------------------------------------------- 4. synthetic gate
step "4/7  Research stage 1 — synthetic validation (no real data touched)"
python -m jarvis5.research.synthetic --cycles 6 --out "$REPORT_DIR" \
    || fail "the synthetic gate failed — real-data research stays blocked"

# -------------------------------------------------------------- 5. download
step "5/7  Historical data (resumable — cached candles are not re-fetched)"
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
    echo "SKIPPED (SKIP_DOWNLOAD=1) — using what is already in $DATA_DIR"
else
    python -m jarvis5 download --days "$DAYS" --data-dir "$DATA_DIR" \
        --symbols "$COINS" || fail "download failed — see the message above.
   The exchange may be unreachable from this host; run:
     python scripts/preflight.py --days $DAYS"
fi
echo
du -sh "$DATA_DIR" 2>/dev/null || true
ls -1 "$DATA_DIR"/*_1m.csv 2>/dev/null | wc -l | xargs -I{} echo "{} symbol files cached"

# -------------------------------------------------------------- 6. backtest
if [ "$MODE" != "research" ]; then
    step "6/7  Trading backtest (17x, 0.35% risk, 6 coins)"
    python -m jarvis5 backtest --days "$DAYS" --data-dir "$DATA_DIR" \
        --out "$OUT_DIR" --name "$NAME" --symbols "$COINS" \
        || fail "backtest failed. If it says 'no cached data', the candles in
   $DATA_DIR do not cover the last $DAYS days — check with:
     python -m jarvis5 download --days $DAYS --data-dir $DATA_DIR"
else
    step "6/7  Trading backtest — SKIPPED (mode=research)"
fi

# -------------------------------------------------------------- 7. research
if [ "$MODE" != "backtest" ]; then
    step "7/7  Research stage 2 — real data"
    python -m jarvis5.research --coins "$COINS" --days "$DAYS" \
        --data-dir "$DATA_DIR" --out "$REPORT_DIR" \
        ${RESEARCH_ARGS[@]+"${RESEARCH_ARGS[@]}"} \
        || fail "research failed — see the message above"
else
    step "7/7  Research — SKIPPED (mode=backtest)"
fi

# ---------------------------------------------------------------- archive
ARCHIVE="${OUT_DIR}/${NAME}.tar.gz"
LIST="$(mktemp)"
find "$REPORT_DIR" -type f ! -name '*.tar.gz' -print > "$LIST" 2>/dev/null || true
find "$OUT_DIR" -maxdepth 1 -type f -name "${NAME}*" ! -name '*.tar.gz' \
     -print >> "$LIST" 2>/dev/null || true
if [ -s "$LIST" ]; then
    # -P keeps absolute paths usable when DATA_DIR/OUT_DIR were overridden
    tar -czPf "$ARCHIVE" --files-from "$LIST" 2>/dev/null \
        || echo "(archive step skipped: tar reported an error)"
else
    echo "(nothing to archive)"
fi
rm -f "$LIST"

echo
echo "============================================================"
echo " DONE  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "============================================================"
echo " Read first:"
echo "   ${REPORT_DIR}/FINAL_REPORT.md        research answers (30 questions)"
if [ "$MODE" != "research" ]; then
echo "   ${OUT_DIR}/${NAME}_summary.txt        backtest result"
fi
echo " Everything, zipped:"
echo "   ${ARCHIVE}"
echo
case "$ARCHIVE" in
    /*) ARCHIVE_ABS="$ARCHIVE" ;;
    *)  ARCHIVE_ABS="${ROOT}/${ARCHIVE}" ;;
esac
echo " Copy it to your machine with:"
echo "   scp root@\$(hostname -I | awk '{print \$1}'):${ARCHIVE_ABS} ."
echo "============================================================"
