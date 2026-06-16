#!/usr/bin/env python3
"""Contour-plot the Poisson (steady-heat) solution on the z=0.5 mid-plane.

Reads an .npz written by the solver (the `POISSON_SLICE_NPZ` output) with:

    points    (M, 2) float  -- (x, y) of the mid-plane nodes
    u         (M,)   float  -- u_h on the plane (the FE solution)
    u_exact   (M,)   float  -- sin(pi x) sin(pi y) sin(pi/2)      [optional]
    error     (M,)   float  -- u_h - u_exact                      [optional]

Renders a filled contour of u_h and, when present, the pointwise error, over the
unit square. Pure matplotlib/numpy -- no FEniCS import, so it runs anywhere the
.npz can be copied.

Usage:
    python3 scripts/poisson_mms/plot_poisson_slice.py <slice.npz> [output.png]

If the output path is omitted, writes alongside the input as <slice>.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2

    in_path = Path(argv[1])
    out_path = Path(argv[2]) if len(argv) > 2 else in_path.with_suffix(".png")

    data = np.load(in_path)
    pts = data["points"]
    x, y = pts[:, 0], pts[:, 1]
    u = data["u"]
    has_err = "error" in data.files

    ncols = 2 if has_err else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 5.2), squeeze=False)
    ax = axes[0, 0]

    filled = ax.tricontourf(x, y, u, levels=21, cmap="inferno")
    ax.tricontour(x, y, u, levels=11, colors="k", linewidths=0.3, alpha=0.5)
    fig.colorbar(filled, ax=ax, shrink=0.85, label=r"$u_h$")
    title = r"Poisson solution $u_h$ on the $z=0.5$ plane"
    if "n" in data.files:
        title += rf"  (Q1, $n={int(data['n'])}^3$)"
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    if has_err:
        err = data["error"]
        ax = axes[0, 1]
        amax = float(np.max(np.abs(err))) or 1.0
        filled = ax.tricontourf(
            x, y, err, levels=21, cmap="coolwarm", vmin=-amax, vmax=amax
        )
        fig.colorbar(filled, ax=ax, shrink=0.85, label=r"$u_h - u$")
        ax.set_title("pointwise error $u_h - u$")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
