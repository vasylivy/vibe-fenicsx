#!/usr/bin/env bash
# Launch the Rayleigh-Darcy porous-convection showcase under MPI.
#
# Usage:
#   ./run_porous.sh [RA] [N] [extra solver args...]
#
# Examples:
#   ./run_porous.sh                 # Ra=4000 coarse showcase (default knobs)
#   ./run_porous.sh 2000 96         # Ra=2000 on an L*96 x 96 grid
#   ./run_porous.sh 4000 64 --no-live --t-end 0.08
#
# Override the rank count with POROUS_NP (default 4).
#
# Live view (headless): rank 0 rewrites $OUTDIR/live.png in place each refresh.
# Open that file in VS Code (or any viewer that reloads on disk change) to watch
# the run evolve.  With a local DISPLAY the solver shows its own window instead.
set -euo pipefail

RA="${1:-4000}"
N="${2:-64}"
shift $(( $# > 2 ? 2 : $# )) || true

OUTDIR="${POROUS_OUTDIR:-results/porous}"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR/live.png"     # don't show a stale frame from a past run

mpirun -np "${POROUS_NP:-4}" \
    python examples/porous_convection.py --ra "$RA" --N "$N" "$@" &
run_pid=$!

# Headless: open the live PNG in VS Code once the first frame lands; it refreshes
# in place as the run advances.  Falls back to a hint when the `code` CLI is
# absent (e.g. a plain terminal or CI).
if [ -z "${DISPLAY:-}" ]; then
    if command -v code >/dev/null 2>&1; then
        for _ in $(seq 1 150); do
            [ -f "$OUTDIR/live.png" ] && { code "$OUTDIR/live.png"; break; }
            kill -0 "$run_pid" 2>/dev/null || break   # run ended before a frame
            sleep 0.2
        done
    else
        echo "Live view: open ${OUTDIR}/live.png in VS Code (refreshes in place)."
    fi
fi

wait "$run_pid"
