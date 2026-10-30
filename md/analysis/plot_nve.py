#!/usr/bin/env python3
"""NVE total-energy drift figure: per-atom E_tot deviation (thin), 21-frame rolling mean (bold,
42 ps at the 2 ps stride), least-squares fit (dotted) and the temperature trace per arm, over
[0, --hi] ps.  The fits use the same definition as ``nve_drift.py``.

    python md/analysis/plot_nve.py                      # reported 1 ns run -> nve_drift_1ns.png
    python md/analysis/plot_nve.py --csv "mine=runs/md/a_nve_energy.csv" --out f.png

Without ``--csv`` the script reads ``md/results/reported_1ns/``.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
EXTRA_COLORS = ["#0a7d4f", "#003049", "#c1121f", "#8338ec", "#fb8500"]
# (label, csv, colour, line width, line style)
REPORTED_ARMS = [   # the reported 1 ns release-driver run
    ("ATOMTRICK", os.path.join(_RES, "reported_1ns", "atomtrick_nve_energy.csv"),
     EXTRA_COLORS[0], 1.1, "-"),
]


def load(p, hi, natoms):
    r = sorted((float(x["time_ps"]), float(x["Etot_eV"]), float(x["T_K"]))
               for x in csv.DictReader(open(p)) if float(x["time_ps"]) <= hi)
    e0 = r[0][1]
    return ([a[0] for a in r], [(a[1] - e0) / natoms * 1000.0 for a in r], [a[2] for a in r])


def slope(t, d):
    t = [x / 1000 for x in t]
    mx, my = sum(t) / len(t), sum(d) / len(d)
    return sum((a - mx) * (b - my) for a, b in zip(t, d)) / sum((a - mx) ** 2 for a in t)


def roll(t, d, w=21):
    h = w // 2
    return ([t[i] for i in range(h, len(t) - h)],
            [sum(d[i - h:i + h + 1]) / w for i in range(h, len(d) - h)])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", action="append", default=None, metavar="LABEL=PATH",
                    help="NVE energy CSV (repeatable); default: the reported run")
    ap.add_argument("--hi", type=float, default=1000.0, help="window end in ps")
    ap.add_argument("--natoms", type=int, default=42)
    ap.add_argument("--out", default="nve_drift_1ns.png")
    args = ap.parse_args()
    if args.csv:
        arms = []
        for i, s in enumerate(args.csv):
            lab, path = s.split("=", 1)
            arms.append((lab, path, EXTRA_COLORS[i % len(EXTRA_COLORS)], 1.1, "-"))
    else:
        arms = REPORTED_ARMS
    HI = args.hi

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    for lab, f, c, lw, ls in arms:
        t, d, T = load(f, HI, args.natoms)
        s = slope(t, d)
        ax1.plot(t, d, lw=0.5, ls="-", color=c, alpha=0.20)
        rt, rd = roll(t, d)
        ax1.plot(rt, rd, lw=1.6, ls=ls, color=c, label=f"{lab}   {s:+.2f} meV/atom/ns")
        my = sum(d) / len(d)
        mt = sum(t) / len(t)
        ax1.plot([0, HI], [my + s * (0 - mt) / 1000, my + s * (HI - mt) / 1000],
                 lw=2.4, ls=":", color=c, alpha=0.85)
        ax2.plot(t, T, lw=0.4, ls="-", color=c, alpha=0.45)
        ax2.axhline(sum(T) / len(T), color=c, ls=":", lw=1.4)
        ax2.text(HI + 6, sum(T) / len(T), f"{sum(T)/len(T):.0f} K", color=c, fontsize=8,
                 va="center", fontweight="bold")

    ax1.axhline(0, color="k", lw=0.6, ls=":")
    ax1.set_ylabel(r"$\Delta E_\mathrm{tot}$ / atom   (meV)")
    ax1.set_title(f"Ac-Ala3-NHMe — {HI:g} ps NVE total-energy drift\n"
                  "same box, same seed, same initial frame", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.92,
               title="bold = 42 ps rolling mean · dotted = least-squares fit",
               title_fontsize=7.5)
    ax1.grid(alpha=0.25)
    ax1.set_xlim(0, HI)
    ax2.set_xlabel("time (ps)")
    ax2.set_ylabel("T (K)")
    ax2.set_xlim(0, HI)
    ax2.grid(alpha=0.25)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
