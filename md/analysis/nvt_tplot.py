#!/usr/bin/env python3
"""Temperature-vs-time figure for NVT runs: raw instantaneous T (thin) plus a running mean
(thick, default 25 frames = 50 ps at the 2 ps report stride; edge-corrected, so the first and
last half-window average only the frames that exist), with the setpoint line.

    python md/analysis/nvt_tplot.py                                  # reported 1 ns run
    python md/analysis/nvt_tplot.py --csv ATOMTRICK=runs/md/a_nvt_energy.csv --out fig.png

Without ``--csv`` the script reads ``md/results/reported_1ns/``.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REPORTED = [("ATOMTRICK", os.path.join(_RES, "reported_1ns", "atomtrick_nvt_energy.csv"))]
TITLE = "NVT temperature vs time — Ac-Ala3-NHMe, 300 K, 1 ns, release driver (seed 2024)"
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def read(path, t_max):
    t, T = [], []
    for r in csv.DictReader(open(path)):
        tp = float(r["time_ps"])
        if tp <= t_max:
            t.append(tp)
            T.append(float(r["T_K"]))
    return np.array(t), np.array(T)


def smooth(y, w):
    """Centred running mean of width ``w`` that stays unbiased at both ends: within w/2 frames
    of an edge it averages only the frames that exist.  (A plain ``np.convolve(y, k, "same")``
    zero-pads, so the curve droops at both ends of the trace, to about half the true value at
    the first and the last frame.)"""
    w = max(1, min(int(w), len(y)))
    k = np.ones(w)
    return np.convolve(y, k, mode="same") / np.convolve(np.ones_like(y), k, mode="same")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", action="append", default=None, metavar="LABEL=PATH",
                    help="NVT energy CSV (repeatable); default: the reported run")
    ap.add_argument("--t_max", type=float, default=1000.0)
    ap.add_argument("--window", type=int, default=25, help="running-mean width in frames")
    ap.add_argument("--setpoint", type=float, default=300.0)
    ap.add_argument("--title", default=TITLE)
    ap.add_argument("--out", default="nvt_temperature_1ns.png")
    args = ap.parse_args()
    arms = [tuple(s.split("=", 1)) for s in args.csv] if args.csv else REPORTED

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for i, (label, path) in enumerate(arms):
        t, T = read(path, args.t_max)
        c = COLORS[i % len(COLORS)]
        ax.plot(t, T, color=c, alpha=0.22, lw=0.6)
        ax.plot(t, smooth(T, args.window), color=c, lw=2.0,
                label=f"{label}  (running mean; ⟨T⟩={T.mean():.1f} K, σ={T.std():.1f} K)")
    ax.axhline(args.setpoint, color="k", ls="--", lw=1.1, alpha=0.75,
               label=f"setpoint {args.setpoint:g} K")
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("temperature (K)")
    ax.set_title(args.title)
    ax.set_xlim(0, args.t_max)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
