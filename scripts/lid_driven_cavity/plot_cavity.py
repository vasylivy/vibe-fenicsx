#!/usr/bin/env python3
"""Visualize the lid-driven cavity from solver frame dumps.

Reads `frame_*.npz` files (and an optional `centerlines.npz`) written by the
cavity solver and produces, for a run prefix `OUT`:

    OUT_montage.png      evolution montage (several frames)
    OUT.gif              animation of velocity magnitude + streamlines
    OUT_steady.png       final (steady) frame, standalone
    OUT_centerlines.png  steady centerline u(y), v(x) vs Ghia et al. (1982)

Each `frame_*.npz` holds vertex-sampled fields (no FEniCS needed here):
    points (N,2), u (N,2), p (N,), t (scalar).
`centerlines.npz` holds: y, u_vert, x, v_horiz, Re, N.

Pure matplotlib/numpy + imageio, so it runs anywhere the .npz files are copied.

Usage:
    python3 scripts/lid_driven_cavity/plot_cavity.py <frames_dir> [out_prefix] [gold_dir]

`out_prefix` defaults to "<frames_dir>/cavity"; `gold_dir` to "gold" — each Re's
benchmark is read from `<gold_dir>/ghia_re<Re>.csv` when present.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.tri import LinearTriInterpolator, Triangulation  # noqa: E402

SPEED_LEVELS = np.linspace(0.0, 1.0, 21)  # fixed scale: lid speed = 1
STREAM_GRID = 44


def _render_frame(ax, npz_path, with_stream=True):
    """Draw velocity-magnitude contours (+ streamlines) for one frame."""
    d = np.load(npz_path)
    pts = d["points"]
    x, y = pts[:, 0], pts[:, 1]
    u = d["u"]
    speed = np.hypot(u[:, 0], u[:, 1])
    tri = Triangulation(x, y)

    cf = ax.tricontourf(tri, speed, levels=SPEED_LEVELS, cmap="viridis", extend="max")
    if with_stream and float(speed.max()) > 1e-9:
        xi = np.linspace(0.0, 1.0, STREAM_GRID)
        yi = np.linspace(0.0, 1.0, STREAM_GRID)
        Xg, Yg = np.meshgrid(xi, yi)
        ug = np.asarray(LinearTriInterpolator(tri, u[:, 0])(Xg, Yg).filled(0.0))
        vg = np.asarray(LinearTriInterpolator(tri, u[:, 1])(Xg, Yg).filled(0.0))
        ax.streamplot(xi, yi, ug, vg, color="white", density=1.1,
                      linewidth=0.5, arrowsize=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"t = {float(d['t']):.2f}")
    return cf


def _fig_to_image(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[..., :3].copy()


def make_montage(frames, out_png, ncols=3, nrows=2):
    picks = np.linspace(0, len(frames) - 1, ncols * nrows).round().astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.2 * nrows))
    cf = None
    for ax, idx in zip(axes.ravel(), picks):
        cf = _render_frame(ax, frames[idx])
    fig.suptitle("Lid-driven cavity — velocity magnitude & streamlines", y=0.99)
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(cf, cax=cbar_ax, label="|u|")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def make_gif(frames, out_gif, fps=12):
    import imageio.v2 as imageio

    images = []
    for f in frames:
        fig, ax = plt.subplots(figsize=(4.4, 4.4))
        _render_frame(ax, f)
        fig.tight_layout()
        images.append(_fig_to_image(fig))
        plt.close(fig)
    imageio.mimsave(out_gif, images, fps=fps, loop=0)
    print(f"wrote {out_gif} ({len(images)} frames)")


def make_steady(frame, out_png):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    cf = _render_frame(ax, frame)
    fig.colorbar(cf, ax=ax, shrink=0.85, label="|u|")
    ax.set_title("Lid-driven cavity, steady state (Re=1000)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def _read_ghia(path):
    """Read the Ghia CSV (skipping '#'-comment lines); return a name->array dict."""
    header, rows = None, []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [h.strip() for h in row]
                continue
            rows.append([float(c) for c in row])
    arr = np.array(rows, dtype=float)
    return {name: arr[:, i] for i, name in enumerate(header)}


def make_centerlines(entries, out_png):
    """Compare solver centerlines against Ghia, one row of (u(y), v(x)) per Re.

    `entries` is a list of (centerlines_npz_path, ghia_csv_path_or_None),
    rendered top-to-bottom in increasing Re.
    """
    loaded = []
    for cl_npz, ghia_csv in entries:
        c = np.load(cl_npz)
        g = _read_ghia(ghia_csv) if ghia_csv and Path(ghia_csv).exists() else None
        loaded.append((int(c["Re"]), c, g))
    loaded.sort(key=lambda e: e[0])

    nrows = len(loaded)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 4.4 * nrows), squeeze=False)
    for row, (Re, c, g) in enumerate(loaded):
        axu, axv = axes[row]
        label = f"FEniCSx (N={int(c['N'])})"
        axu.plot(c["u_vert"], c["y"], "-", color="C0", label=label)
        axv.plot(c["x"], c["v_horiz"], "-", color="C0", label=label)
        if g is not None:
            axu.plot(g["u"], g["y"], "o", color="C3", ms=5, label="Ghia et al. 1982")
            axv.plot(g["x"], g["v"], "o", color="C3", ms=5, label="Ghia et al. 1982")
        axu.set_xlabel("u")
        axu.set_ylabel("y")
        axu.set_title(f"Re={Re}: u along vertical centerline x=0.5")
        axv.set_xlabel("x")
        axv.set_ylabel("v")
        axv.set_title(f"Re={Re}: v along horizontal centerline y=0.5")
        for ax in (axu, axv):
            ax.grid(True, ls=":", lw=0.5)
            ax.legend()
    fig.suptitle("Cavity centerline velocities vs Ghia et al. (1982)")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.03 / nrows))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    frames_dir = Path(argv[1])
    out_prefix = argv[2] if len(argv) > 2 else str(frames_dir / "cavity")
    # Third arg, if given, is the gold directory (default "gold"); each Re's
    # benchmark is gold/ghia_re<Re>.csv.
    gold_dir = Path(argv[3]) if len(argv) > 3 else Path("gold")

    frames = sorted(frames_dir.glob("frame_*.npz"))
    if not frames:
        print(f"no frame_*.npz in {frames_dir}", file=sys.stderr)
        return 1

    make_montage(frames, f"{out_prefix}_montage.png")
    make_gif(frames, f"{out_prefix}.gif")
    make_steady(frames[-1], f"{out_prefix}_steady.png")

    # Centerlines: a single centerlines.npz, or centerlines_re<Re>.npz from the
    # validation sweep. Each is matched to gold/ghia_re<Re>.csv by its Re.
    cls = sorted(frames_dir.glob("centerlines*.npz"))
    if cls:
        entries = []
        for f in cls:
            Re = int(np.load(f)["Re"])
            ghia = gold_dir / f"ghia_re{Re}.csv"
            entries.append((f, ghia if ghia.exists() else None))
        make_centerlines(entries, f"{out_prefix}_centerlines.png")
    else:
        print(f"(no centerlines*.npz in {frames_dir} — skipping centerline comparison)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
