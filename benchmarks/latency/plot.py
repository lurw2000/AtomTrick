#!/usr/bin/env python3
"""Plot median inference latency vs number of atoms for all models (rMD17 + MD22),
one panel per batch size, from the long CSV written by merge.py.

    python benchmarks/latency/plot.py --long results/h100_fp32_2026-10-20/latency_all_long.csv --out fig

Runs headless (Agg); needs only matplotlib + the stdlib csv module.  Model labels are
styled by name; ATOMTRICK rows are recognised by their "(eager)" / "(compiled)" (or
"(makefx)") suffix.  Compiled-baseline rows, "<baseline> (compiled)" or
"<baseline> (compiled: <variant>)", take the baseline's colour with a solid line and a
filled marker; the eager baseline then becomes a dashed line with hollow markers of the
same colour.  When the long CSV has merge.py's ``selected`` column, rows with
``selected == 0`` (the compiled variant not shown in the tables) are skipped.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE_STYLE = {
    "Equiformer":   ("#FF7F0E", "D", "-"),
    "EquiformerV2": ("#D62728", "s", "-"),
    "GotenNet":     ("#2CA02C", "^", "-"),
    "MACE":         ("#9467BD", "v", "-"),
}
EAGER_STYLE = ("#7F7F7F", "o", "--")
COMPILED_STYLE = ("#1F77B4", "o", "-")


def base_of_compiled(model):
    """'MACE (compiled)' / 'MACE (compiled: fused)' -> 'MACE' if MACE is a known baseline."""
    if model.endswith(")") and " (compiled" in model:
        base = model[:model.rindex(" (compiled")]
        if base in BASE_STYLE:
            return base
    return None


def style_of(model, baselines_with_compiled=()):
    """-> ((color, marker, linestyle), linewidth, filled)"""
    if model in BASE_STYLE:
        color, marker, ls = BASE_STYLE[model]
        if model in baselines_with_compiled:           # eager next to its compiled curve
            return (color, marker, "--"), 1.8, False
        return (color, marker, ls), 1.8, True
    base = base_of_compiled(model)
    if base is not None:
        color, marker, _ = BASE_STYLE[base]
        return (color, marker, "-"), 2.2, True
    if model.endswith("(eager)"):
        return EAGER_STYLE, 1.8, True
    if model.endswith("(compiled)") or model.endswith("(makefx)"):
        return COMPILED_STYLE, 2.6, True
    return ("#8C564B", "x", ":"), 1.5, True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--long", required=True, help="merge.py's long CSV")
    ap.add_argument("--out", default="latency_all",
                    help="output path stem; writes <out>.png and <out>.pdf")
    ap.add_argument("--batch_sizes", nargs="+", default=["1", "4"])
    ap.add_argument("--title", default="H100 inference latency (energy + forces): "
                                       "ATOMTRICK vs baselines — rMD17 + MD22")
    args = ap.parse_args()

    with open(args.long) as f:
        rows = [r for r in csv.DictReader(f) if r.get("selected", "1") != "0"]
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    with_compiled = {base_of_compiled(m) for m in models} - {None}

    fig, axes = plt.subplots(1, len(args.batch_sizes), figsize=(6 * len(args.batch_sizes), 4.6),
                             squeeze=False)
    for ax, bs in zip(axes[0], args.batch_sizes):
        for model in models:
            pts = sorted((int(r["atoms"]), float(r["latency_median_ms"])) for r in rows
                         if r["model"] == model and r["batch_size"] == bs
                         and r["status"] == "ok" and r["latency_median_ms"])
            if not pts:
                continue
            (color, marker, ls), lw, filled = style_of(model, with_compiled)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker,
                    linestyle=ls, label=model, linewidth=lw, markersize=7,
                    markerfacecolor=color if filled else "white",
                    markeredgecolor="white" if filled else color,
                    markeredgewidth=0.5 if filled else 1.2, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of atoms (rMD17 9-24 → MD22 42-370)")
        ax.set_ylabel("Inference latency (ms), median")
        ax.set_title(f"Batch size {bs}")
        ax.grid(True, which="both", alpha=0.25, linewidth=0.4, zorder=0)
        if len(models) <= 7:
            ax.legend(fontsize=8, framealpha=0.95, edgecolor="none", loc="upper left")

    fig.suptitle(args.title, fontsize=12)
    if len(models) > 7:          # eager + compiled baselines: one shared legend below the panels
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center",
                   ncol=min(5, (len(labels) + 1) // 2), bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    print(f"wrote {args.out}.png\nwrote {args.out}.pdf")


if __name__ == "__main__":
    main()
