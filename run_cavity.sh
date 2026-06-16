#!/usr/bin/env bash
# Launch the lid-driven cavity showcase under MPI.
#
# Usage:
#   ./run_cavity.sh [RE] [N] [extra solver args...]
#
# Examples:
#   ./run_cavity.sh                 # Re=1000 showcase (default knobs)
#   ./run_cavity.sh 400 48          # Re=400 on a 48x48 mesh
#   ./run_cavity.sh 1000 96 --no-live --steady-tol 1e-4
#
# Override the rank count with CAVITY_NP (default 4).
#
# Live view (headless): rank 0 rewrites $OUTDIR/live.png in place each refresh.
# Open that file in VS Code (or any viewer that reloads on disk change) to watch
# the run evolve.  With a local DISPLAY the solver shows its own window instead.
set -euo pipefail

RE="${1:-1000}"
N="${2:-64}"
shift $(( $# > 2 ? 2 : $# )) || true

OUTDIR="${CAVITY_OUTDIR:-results/cavity}"
mkdir -p "$OUTDIR"
rm -f "$OUTDIR/live.png"     # don't show a stale frame from a past run

mpirun -np "${CAVITY_NP:-4}" \
    python examples/lid_driven_cavity.py --re "$RE" --N "$N" "$@" &
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
