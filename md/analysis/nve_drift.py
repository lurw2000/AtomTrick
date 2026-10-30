#!/usr/bin/env python3
"""NVE energy-drift table over configurable time windows (stdlib only).

For every energy CSV written by ``md/md_openmm.py`` (columns tag, step, time_ps, Epot_eV,
Ekin_eV, Etot_eV, T_K) and every window [t0, t1] (ps, inclusive):

* drift  = least-squares slope of  dE(t) = (E_tot(t) - E_tot(first frame in window)) / N_atoms
           against time, in meV/atom/ns, and its standard error (ordinary least squares; the
           2 ps frames are treated as independent, which the energy autocorrelation supports);
* RMS    = population std of dE over the window, in meV/atom (the fluctuation scale a drift
           must exceed to be meaningful);
* <T>, sigma_T over the window, frame count, and the last time actually present (a window
  longer than the run is flagged).

Without ``--csv`` the arm is the reported 1 ns release-driver run committed in
``md/results/reported_1ns/`` (default windows 100 500 690 1000 ps):

    python md/analysis/nve_drift.py                         # -> reported_1ns/nve_drift.md
    python md/analysis/nve_drift.py --per 100 --t_max 1000  # -> nve_drift_per100ps.md
    python md/analysis/nve_drift.py --csv mine=runs/md/atomtrick_nve_energy.csv --windows 1000

Expected (reported run): [0,1000] ps -> ATOMTRICK -0.17 +- 0.16 (RMS 1.02, <T> 281.2 K); the
+- is the slope's standard error.
"""
import argparse
import csv
import os
import statistics

_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
REPORTED_ARMS = [   # the reported 1 ns release-driver run
    ("ATOMTRICK", os.path.join(_RES, "reported_1ns", "atomtrick_nve_energy.csv")),
]


def load(path):
    rows = sorted((float(r["time_ps"]), float(r["Etot_eV"]), float(r["T_K"]))
                  for r in csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def slope(x, y):
    """Least-squares slope and its standard error (ordinary least squares)."""
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a0 = my - b * mx
    s2 = sum((c - a0 - b * a) ** 2 for a, c in zip(x, y)) / max(n - 2, 1)
    return b, (s2 / sxx) ** 0.5


def fit(rows, lo, hi, natoms):
    w = [r for r in rows if lo <= r[0] <= hi]
    if len(w) < 3:
        return None
    e0 = w[0][1]
    t_ns = [r[0] / 1000.0 for r in w]
    d = [(r[1] - e0) / natoms * 1000.0 for r in w]        # meV/atom
    T = [r[2] for r in w]
    b, se = slope(t_ns, d)
    return {"n": len(w), "last": w[-1][0], "drift": b, "drift_se": se,
            "rms": statistics.pstdev(d), "T": sum(T) / len(T), "Tsd": statistics.pstdev(T)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", action="append", default=None, metavar="LABEL=PATH",
                    help="energy CSV (repeatable); default: the reported run")
    ap.add_argument("--natoms", type=int, default=42, help="atoms (Ac-Ala3-NHMe: 42)")
    ap.add_argument("--t0", type=float, default=0.0, help="window start (ps)")
    ap.add_argument("--windows", type=float, nargs="+", default=[100, 500, 690, 1000],
                    help="window END times in ps (each window is [t0, end])")
    ap.add_argument("--per", type=float, default=None,
                    help="instead: consecutive windows of this length (ps) up to --t_max")
    ap.add_argument("--t_max", type=float, default=1000.0)
    args = ap.parse_args()

    arms = [tuple(s.split("=", 1)) for s in args.csv] if args.csv else REPORTED_ARMS
    data = [(label, load(path)) for label, path in arms]
    if args.per:
        wins, t = [], args.t0
        while t + args.per <= args.t_max + 1e-9:
            wins.append((t, t + args.per))
            t += args.per
    else:
        wins = [(args.t0, hi) for hi in args.windows]

    print(f"Drift = LSQ slope of (E_tot - E_tot(first frame)) / {args.natoms} atoms vs time, "
          "with its standard error (OLS); RMS = population std of that per-atom deviation.\n")
    print("| arm | window (ps) | frames | last t (ps) | drift (meV/atom/ns) | "
          "drift SE (meV/atom/ns) | RMS (meV/atom) | <T> (K) | sigma_T (K) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, rows in data:
        for lo, hi in wins:
            r = fit(rows, lo, hi, args.natoms)
            if r is None:
                print(f"| {label} | [{lo:g}, {hi:g}] | <3 | — | — | — | — | — | — |")
                continue
            short = " (run ends)" if r["last"] < hi - 1e-6 else ""
            print(f"| {label} | [{lo:g}, {hi:g}] | {r['n']} | {r['last']:g}{short} | "
                  f"{r['drift']:+.2f} | {r['drift_se']:.2f} | {r['rms']:.2f} | {r['T']:.1f} | "
                  f"{r['Tsd']:.1f} |")


if __name__ == "__main__":
    main()
