#!/usr/bin/env python3
"""Visualize Rayleigh-Darcy porous convection from solver frame dumps.

Reads `frame_*.npz` files (plus optional `nu_timeseries.npz` and a Nu-Ra CSV)
written by the porous-convection solver and produces, for a run prefix `OUT`:

    OUT_montage.png      evolution montage — temperature with the plume "fingers"
    OUT.gif              animation of the temperature field
    OUT_field.png        final temperature field, standalone
    OUT_nu_timeseries.png  Nusselt number Nu(t)
    OUT_nu_scaling.png   time-averaged Nu vs Ra against the 2-D correlation

Each `frame_*.npz` holds vertex-sampled fields (no FEniCS needed here):
    points (N,2), T (N,), u (N,2), t (scalar), Ra, L.
`nu_timeseries.npz` holds: t, nu, Ra, nu_avg.
The Nu-Ra CSV (e.g. `results/porous/nu_scaling.csv`, written by the sweep) has
columns `Ra,Nu`; the plot draws those points against the published correlation.

Pure matplotlib/numpy + imageio, so it runs anywhere the .npz files are copied.

Usage:
    python3 scripts/porous_convection/plot_porous.py <frames_dir> [out_prefix] [nu_scaling.csv]

`out_prefix` defaults to "<frames_dir>/porous"; the CSV to
"<frames_dir>/nu_scaling.csv" when present.
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

T_LEVELS = np.linspace(0.0, 1.0, 21)  # temperature scale: cold=0 (top) .. hot=1 (bottom)
STREAM_GRID = 60

# 2-D Rayleigh-Darcy correlation Nu ~ 0.0069 Ra + 2.75, from Hewitt, Neufeld &
# Lister (PRL 108:224503, 2012); see docs/porous_convection.md for what the slope
# and intercept mean.
NU_SLOPE = 0.0069
NU_OFFSET = 2.75


def _render_frame(ax, npz_path, with_stream=True):
    """Draw the temperature field (+ streamlines) for one frame."""
    d = np.load(npz_path)
    pts = d["points"]
    x, y = pts[:, 0], pts[:, 1]
    T = d["T"]
    L = float(d["L"]) if "L" in d else float(x.max())
    tri = Triangulation(x, y)

    cf = ax.tricontourf(tri, T, levels=T_LEVELS, cmap="RdBu_r", extend="both")
    if with_stream and "u" in d:
        u = d["u"]
        speed = np.hypot(u[:, 0], u[:, 1])
        if float(speed.max()) > 1e-9:
            xi = np.linspace(0.0, L, int(STREAM_GRID * L))
            yi = np.linspace(0.0, 1.0, STREAM_GRID)
            Xg, Yg = np.meshgrid(xi, yi)
            ug = np.asarray(LinearTriInterpolator(tri, u[:, 0])(Xg, Yg).filled(0.0))
            vg = np.asarray(LinearTriInterpolator(tri, u[:, 1])(Xg, Yg).filled(0.0))
            ax.streamplot(xi, yi, ug, vg, color="k", density=1.0,
                          linewidth=0.4, arrowsize=0.5)
    ax.set_xlim(0, L)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"t = {float(d['t']):.3f}")
    return cf


def _fig_to_image(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return buf.reshape(h, w, 4)[..., :3].copy()


def make_montage(frames, out_png, ncols=2, nrows=3):
    picks = np.linspace(0, len(frames) - 1, ncols * nrows).round().astype(int)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.0 * nrows))
    cf = None
    for ax, idx in zip(axes.ravel(), picks):
        cf = _render_frame(ax, frames[idx], with_stream=False)
    fig.suptitle("Rayleigh-Darcy convection — temperature plumes (fingering)", y=0.99)
    fig.tight_layout(rect=(0, 0, 0.92, 0.97))
    cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(cf, cax=cbar_ax, label="T")
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"wrote {out_png}")


def make_gif(frames, out_gif, fps=10):
    import imageio.v2 as imageio

    images = []
    for f in frames:
        d = np.load(f)
        L = float(d["L"]) if "L" in d else 2.0
        fig, ax = plt.subplots(figsize=(4.2 * L, 4.2))
        _render_frame(ax, f, with_stream=False)
        fig.tight_layout()
        images.append(_fig_to_image(fig))
        plt.close(fig)
    imageio.mimsave(out_gif, images, fps=fps, loop=0)
    print(f"wrote {out_gif} ({len(images)} frames)")


def make_field(frame, out_png):
    d = np.load(frame)
    L = float(d["L"]) if "L" in d else 2.0
    Ra = float(d["Ra"]) if "Ra" in d else None
    fig, ax = plt.subplots(figsize=(4.6 * L, 4.6))
    cf = _render_frame(ax, frame, with_stream=True)
    fig.colorbar(cf, ax=ax, shrink=0.85, label="T")
    title = "Porous convection — temperature & flow"
    if Ra is not None:
        title += f" (Ra={Ra:.0f})"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def make_nu_timeseries(nu_npz, out_png):
    d = np.load(nu_npz)
    t, nu = d["t"], d["nu"]
    Ra = float(d["Ra"]) if "Ra" in d else None
    nu_avg = float(d["nu_avg"]) if "nu_avg" in d else float(np.mean(nu[len(nu) // 2:]))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(t, nu, "-", color="C0", lw=0.9, label="Nu(t)")
    ax.axhline(nu_avg, color="C3", ls="--", lw=1.2,
               label=f"time-avg Nu = {nu_avg:.2f}")
    if Ra is not None:
        ax.axhline(NU_SLOPE * Ra + NU_OFFSET, color="C2", ls=":", lw=1.2,
                   label=f"0.0069 Ra + 2.75 = {NU_SLOPE * Ra + NU_OFFSET:.2f}")
    ax.set_xlabel("t")
    ax.set_ylabel("Nu")
    ttl = "Nusselt number history"
    if Ra is not None:
        ttl += f", Ra={Ra:.0f}"
    ax.set_title(ttl)
    ax.grid(True, ls=":", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def _read_nu_csv(path):
    """Read a Ra,Nu CSV (skipping '#'-comment lines); return (Ra, Nu) arrays."""
    header, rows = None, []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if header is None:
                header = [h.strip().lower() for h in row]
                continue
            rows.append([float(c) for c in row])
    arr = np.array(rows, dtype=float)
    cols = {name: arr[:, i] for i, name in enumerate(header)}
    return cols["ra"], cols["nu"]


def make_nu_scaling(csv_path, out_png):
    Ra, Nu = _read_nu_csv(csv_path)
    order = np.argsort(Ra)
    Ra, Nu = Ra[order], Nu[order]
    rr = np.linspace(Ra.min() * 0.8, Ra.max() * 1.1, 100)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(rr, NU_SLOPE * rr + NU_OFFSET, "-", color="C2",
            label="0.0069 Ra + 2.75  (Hewitt et al. 2012)")
    ax.plot(Ra, Nu, "o", color="C0", ms=7, label="FEniCSx (time-averaged)")
    ax.set_xlabel("Ra")
    ax.set_ylabel("Nu")
    ax.set_title("Heat transport: Nu vs Rayleigh-Darcy number")
    ax.grid(True, ls=":", lw=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"wrote {out_png}")


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    frames_dir = Path(argv[1])
    out_prefix = argv[2] if len(argv) > 2 else str(frames_dir / "porous")
    nu_csv = argv[3] if len(argv) > 3 else str(frames_dir / "nu_scaling.csv")

    frames = sorted(frames_dir.glob("frame_*.npz"))
    if frames:
        make_montage(frames, f"{out_prefix}_montage.png")
        make_gif(frames, f"{out_prefix}.gif")
        make_field(frames[-1], f"{out_prefix}_field.png")
    else:
        print(f"no frame_*.npz in {frames_dir}", file=sys.stderr)

    nu_ts = frames_dir / "nu_timeseries.npz"
    if nu_ts.exists():
        make_nu_timeseries(nu_ts, f"{out_prefix}_nu_timeseries.png")
    else:
        print(f"(no {nu_ts} — skipping Nu(t) plot)")

    if nu_csv and Path(nu_csv).exists():
        make_nu_scaling(nu_csv, f"{out_prefix}_nu_scaling.png")
    else:
        print(f"(no {nu_csv} — skipping Nu-Ra plot)")

    return 0 if frames else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
