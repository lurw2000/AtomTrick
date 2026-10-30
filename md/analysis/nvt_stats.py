#!/usr/bin/env python3
"""NVT thermostat statistics over a time window (stdlib only).

For each NVT energy CSV (``md/md_openmm.py --ensemble nvt``): mean T with its standard error
(sigma_T / sqrt(frames); consecutive 2 ps frames are weakly correlated, lag-1 autocorrelation
about 0.1, so this understates it by roughly 10 %), population std of T, and T drift =
least-squares slope of T vs time (K/ns) with its standard error, over [--t0, --t1] ps.  A correct,
stationary 300 K run has <T> ~ setpoint, sigma_T ~ T*sqrt(2/N_f) (about 38 K for 42 atoms)
and a drift consistent with zero.  With sigma_T ~ 40 K over 500 frames, the slope's own 1-sigma
uncertainty is about 6 K/ns.

    python md/analysis/nvt_stats.py                 # reported 1 ns run (md/results/reported_1ns/)
    python md/analysis/nvt_stats.py --csv mine=runs/md/atomtrick_nvt_energy.csv

Expected (reported run, release driver, 3N dof): ATOMTRICK 310.6 (SEM 1.73) K / 38.6 K /
+2.60 +- 5.99 K/ns.
"""
import argparse
import csv
import os

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REPORTED = [("ATOMTRICK", os.path.join(_RES, "reported_1ns", "atomtrick_nvt_energy.csv"))]


def read(path):
    t, T = [], []
    for r in csv.DictReader(open(path)):
        t.append(float(r["time_ps"]))
        T.append(float(r["T_K"]))
    return t, T


def slope(x, y):
    """Least-squares slope and its standard error (ordinary least squares)."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a0 = my - b * mx
    s2 = sum((c - a0 - b * a) ** 2 for a, c in zip(x, y)) / max(n - 2, 1)
    return b, (s2 / sxx) ** 0.5


def mean(v):
    return sum(v) / len(v)


def std(v):
    mu = mean(v)
    return (sum((a - mu) ** 2 for a in v) / len(v)) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", action="append", default=None, metavar="LABEL=PATH",
                    help="NVT energy CSV (repeatable); default: the reported run")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=1000.0)
    args = ap.parse_args()
    arms = [tuple(s.split("=", 1)) for s in args.csv] if args.csv else REPORTED

    print("| model | window | reached (ps) | frames | mean T (K) | SEM (K) | T std (K) | "
          "T drift (K/ns) | drift SE (K/ns) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, path in arms:
        t, T = read(path)
        idx = [i for i in range(len(t)) if args.t0 <= t[i] <= args.t1]
        if len(idx) < 3:
            print(f"| {label} | [{args.t0:g},{args.t1:g}] ps | — | {len(idx)} | — | — | — | — | — |")
            continue
        tw = [t[i] for i in idx]; Tw = [T[i] for i in idx]
        ns = [(x - tw[0]) / 1000.0 for x in tw]
        b, se = slope(ns, Tw)
        print(f"| **{label}** | [{args.t0:g},{args.t1:g}] ps | {tw[-1]:.0f} | {len(idx)} | "
              f"{mean(Tw):.1f} | {std(Tw) / len(Tw) ** 0.5:.2f} | {std(Tw):.1f} | {b:+.2f} | "
              f"{se:.2f} |")


if __name__ == "__main__":
    main()
