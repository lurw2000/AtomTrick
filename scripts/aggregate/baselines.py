"""Published baseline numbers used by the aggregation scripts, with sources.

Units everywhere: energy MAE in meV (total energy per structure) for rMD17 and
MD22; SPICE energies in meV/atom; forces in meV/Å (per Cartesian component).
Conversion from kcal/mol: 1 kcal/mol = 43.364 meV (as in the paper).

Every value is copied from the cited table.  Where a source reports kcal/mol
(GotenNet; the MD22 compilation), the ORIGINAL kcal/mol numbers are stored and
converted here, so each entry can be checked against its source directly; the
MD22 baselines the paper prints (its MD22 MAE tables) are used as printed.
``python baselines.py --check`` runs consistency checks against the paper.

    python scripts/aggregate/baselines.py                 # print all tables
    python scripts/aggregate/baselines.py --benchmark md22
    python scripts/aggregate/baselines.py --check
"""

import argparse
import sys

KCALMOL_TO_MEV = 43.364

RMD17_MOLECULES = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
                   "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"]
MD22_MOLECULES = ["Ac-Ala3-NHMe", "DHA", "stachyose", "AT-AT", "AT-AT-CG-CG",
                  "buckyball-catcher", "double-walled_nanotube"]
MD22_ATOMS = {"Ac-Ala3-NHMe": 42, "DHA": 56, "stachyose": 87, "AT-AT": 60,
              "AT-AT-CG-CG": 118, "buckyball-catcher": 148, "double-walled_nanotube": 370}
SPICE_SUBSETS = ["PubChem", "DES370K Monomers", "DES370K Dimers", "Dipeptides",
                 "Solvated Amino Acids", "Water", "QMugs"]


def _row(values, keys):
    return {k: v for k, v in zip(keys, values)}


def _kcal(values, keys):
    """kcal/mol -> meV, keeping None for 'not reported'."""
    return {k: (None if v is None else v * KCALMOL_TO_MEV) for k, v in zip(keys, values)}


# ---------------------------------------------------------------------------
# rMD17 (test MAE; 950 train / 50 val, standard splits)
# ---------------------------------------------------------------------------
# NequIP, Allegro, BOTNet, MACE, TensorNet: the values printed in the ATOMTRICK
# paper's rMD17 force and energy MAE tables, which reproduce the
# numbers published by each method in meV / meV/Å.  `decimals=None` means "as
# printed" (ties are decided at the printed precision of each value).
_M = RMD17_MOLECULES
RMD17 = {
    "NequIP": {
        "citation": "Batzner et al., Nat. Commun. 13, 2453 (2022)",
        "source": "ATOMTRICK paper, rMD17 MAE tables (F, E)",
        "family": "CG",
        "decimals": None,
        "force": _row([8.2, 2.9, 0.3, 2.8, 5.1, 1.3, 5.9, 4.0, 1.6, 3.1], _M),
        "energy": _row([2.3, 0.7, 0.04, 0.4, 0.8, 0.9, 1.4, 0.7, 0.3, 0.4], _M),
    },
    "Allegro": {
        "citation": "Musaelian et al., Nat. Commun. 14, 579 (2023)",
        "source": "ATOMTRICK paper, rMD17 MAE tables (F, E)",
        "family": "CG-free",
        "decimals": None,
        "force": _row([7.3, 2.6, 0.2, 2.1, 3.6, 0.9, 4.9, 2.9, 1.8, 1.8], _M),
        "energy": _row([2.3, 1.2, 0.3, 0.4, 0.6, 0.2, 1.5, 0.9, 0.4, 0.6], _M),
    },
    "BOTNet": {
        "citation": "Batatia et al., Nat. Mach. Intell. 7, 56 (2025) "
                    "(The design space of E(3)-equivariant atom-centred potentials)",
        "source": "ATOMTRICK paper, rMD17 MAE tables (F, E)",
        "family": "CG",
        "decimals": None,
        "force": _row([8.5, 3.3, 0.3, 3.2, 5.8, 1.8, 5.8, 4.3, 1.9, 3.2], _M),
        "energy": _row([2.3, 0.7, 0.03, 0.4, 0.8, 0.2, 1.3, 0.8, 0.4, 0.4], _M),
    },
    "MACE": {
        "citation": "Batatia et al., NeurIPS 2022 (MACE)",
        "source": "ATOMTRICK paper, rMD17 MAE tables (F, E)",
        "family": "CG",
        "decimals": None,
        "force": _row([6.6, 3.0, 0.3, 2.1, 4.1, 1.6, 4.8, 3.1, 1.5, 2.1], _M),
        "energy": _row([2.2, 1.2, 0.4, 0.4, 0.8, 0.5, 1.3, 0.9, 0.5, 0.5], _M),
    },
    "TensorNet": {
        "citation": "Simeon & De Fabritiis, NeurIPS 2023 (TensorNet)",
        "source": "ATOMTRICK paper, rMD17 MAE tables (F, E)",
        "family": "CG",
        "decimals": None,
        "force": _row([8.9, 3.1, 0.3, 3.5, 5.4, 1.6, 5.9, 4.6, 1.7, 3.1], _M),
        "energy": _row([2.4, 0.7, 0.02, 0.5, 0.8, 0.2, 1.3, 0.8, 0.3, 0.4], _M),
    },
    # GotenNet reports kcal/mol (averaged over the 5 standard splits); stored in
    # kcal/mol exactly as in GotenNet Table 4 and converted.  Forces convert to
    # 5.663 2.094 0.204 2.090 3.599 1.041 4.024 3.048 1.132 1.808 (mean 2.470).
    "GotenNet": {
        "citation": "Aykent & Xia, ICLR 2025 (GotenNet), Table 4",
        "source": "GotenNet Table 4 (kcal/mol, kcal/mol/Å) x 43.364",
        "family": "CG-free",
        "decimals": 3,
        "force": _kcal([0.1306, 0.0483, 0.0047, 0.0482, 0.0830, 0.0240, 0.0928, 0.0703,
                        0.0261, 0.0417], _M),
        "energy": _kcal([0.0358, 0.0121, 0.0005, 0.0071, 0.0129, 0.0039, 0.0212, 0.0141,
                         0.0044, 0.0064], _M),
    },
}

# ---------------------------------------------------------------------------
# MD22 (test MAE)
# ---------------------------------------------------------------------------
# Raw source: the kcal/mol values compiled in GotenNet (ICLR 2025) Table 3, which
# collects them from the original publications (sGDML: Chmiela et al., Sci. Adv.
# 9, eadf0873 (2023); Allegro: Musaelian et al. 2023; MACE: Kovács et al., J.
# Chem. Phys. 159, 044118 (2023); Equiformer: Liao & Smidt, ICLR 2023; ViSNet:
# Wang et al., Nat. Commun. 15, 313 (2024); QuinNet: Wang et al., NeurIPS 2023).
# Order: Ac-Ala3-NHMe (GotenNet's "Tetrapeptide"), DHA, stachyose, AT-AT,
# AT-AT-CG-CG, buckyball-catcher, double-walled nanotube; None = not reported.
_D = MD22_MOLECULES
_MD22_KCAL = {   # name: (energy kcal/mol, force kcal/mol/Å)
    "sGDML": ([0.3902, 1.3117, 4.0497, 0.7235, 1.3885, 1.1962, 4.0122],
              [0.7968, 0.7474, 0.6744, 0.6911, 0.7028, 0.6820, 0.5231]),
    "Allegro": ([0.1019, 0.1153, 0.2485, 0.1428, 0.3933, 0.5258, 2.2097],
                [0.1068, 0.0732, 0.0971, 0.0952, 0.1280, 0.0887, 0.3428]),
    "MACE": ([0.0620, 0.1317, 0.1244, 0.1093, 0.1578, None, None],
             [0.0876, 0.0646, 0.0876, 0.0992, 0.1153, None, None]),
    "Equiformer": ([0.0828, 0.1788, 0.1404, 0.1309, 0.1510, 0.3978, 1.1945],
                   [0.0804, 0.0506, 0.0635, 0.0960, 0.1252, 0.1114, 0.2747]),
    "ViSNet": ([0.0796, 0.1526, 0.1283, 0.1688, 0.1995, 0.4421, 1.0339],
               [0.0972, 0.0668, 0.0869, 0.1070, 0.1563, 0.1335, 0.3959]),
    "QuinNet": ([0.0840, 0.1200, 0.2300, 0.1400, 0.3800, 0.5624, 1.8130],
                [0.0681, 0.0515, 0.0543, 0.0687, 0.1273, 0.1091, 0.2473]),
    "GotenNet-S": ([0.0589, 0.0640, 0.0751, 0.0640, 0.0964, 0.3432, 0.9993],
                   [0.0719, 0.0496, 0.0512, 0.0632, 0.0824, 0.0838, 0.2464]),
    "GotenNet-B": ([0.0505, 0.0575, 0.0673, 0.0544, 0.0923, 0.3032, 0.6641],
                   [0.0567, 0.0421, 0.0427, 0.0478, 0.0744, 0.0789, 0.1888]),
}
# Values printed in the ATOMTRICK paper's MD22 energy (meV) and force
# (meV/Å) MAE tables, for the first five systems.  They are the conversions above
# rounded to 0.1 except four cells that differ by 0.1 (stachyose sGDML E 175.7
# vs 175.6 and F 29.3 vs 29.2; AT-AT-CG-CG MACE E 6.9 vs 6.8 and Equiformer E
# 6.6 vs 6.5), presumably taken from the original papers' own unit conversions.
_PAPER_MD22 = {   # name: (energy x5, force x5)
    "sGDML": ([16.9, 56.9, 175.7, 31.4, 60.2], [34.6, 32.4, 29.3, 30.0, 30.5]),
    "Allegro": ([4.4, 5.0, 10.8, 6.2, 17.1], [4.6, 3.2, 4.2, 4.1, 5.6]),
    "MACE": ([2.7, 5.7, 5.4, 4.7, 6.9], [3.8, 2.8, 3.8, 4.3, 5.0]),
    "Equiformer": ([3.6, 7.8, 6.1, 5.7, 6.6], [3.5, 2.2, 2.8, 4.2, 5.4]),
    "ViSNet": ([3.5, 6.6, 5.6, 7.3, 8.7], [4.2, 2.9, 3.8, 4.6, 6.8]),
    "GotenNet-B": ([2.2, 2.5, 2.9, 2.4, 4.0], [2.5, 1.8, 1.9, 2.1, 3.2]),
}
_MD22_CITATION = {
    "sGDML": "Chmiela et al., Sci. Adv. 9, eadf0873 (2023)",
    "Allegro": "Musaelian et al., Nat. Commun. 14, 579 (2023)",
    "MACE": "Kovács et al., J. Chem. Phys. 159, 044118 (2023)",
    "Equiformer": "Liao & Smidt, ICLR 2023",
    "ViSNet": "Wang et al., Nat. Commun. 15, 313 (2024)",
    "QuinNet": "Wang et al., NeurIPS 2023",
    "GotenNet-S": "Aykent & Xia, ICLR 2025 (GotenNet-S)",
    "GotenNet-B": "Aykent & Xia, ICLR 2025 (GotenNet-B)",
}
_MD22_FAMILY = {"sGDML": "kernel", "MACE": "CG", "Equiformer": "CG"}


def _md22_entry(name):
    """Paper-printed values (1 decimal) where the paper prints them; otherwise
    exact conversions of GotenNet Table 3 (2 decimals).  GotenNet-B always uses
    the exact conversions (F 2.459 1.826 1.852 2.073 3.226 3.421 8.187, E 2.19
    2.49 2.92 2.36 4.00 13.15 28.80), as does every buckyball / nanotube force
    (sGDML 29.57/22.68, Allegro 3.85/14.87, Equiformer 4.83/11.91, ViSNet
    5.79/17.17, QuinNet 4.73/10.72, GotenNet-S 3.63/10.68, GotenNet-B 3.42/8.19)."""
    e_kcal, f_kcal = _MD22_KCAL[name]
    energy, force = _kcal(e_kcal, _D), _kcal(f_kcal, _D)
    decimals = {m: 2 for m in _D}
    paper = _PAPER_MD22.get(name) if name != "GotenNet-B" else None
    if paper:
        for i, m in enumerate(_D[:5]):
            energy[m], force[m], decimals[m] = paper[0][i], paper[1][i], 1
    return {
        "citation": _MD22_CITATION[name],
        "source": ("ATOMTRICK paper, MD22 MAE tables (first five systems); other systems: "
                   "GotenNet (ICLR 2025) Table 3, kcal/mol x 43.364") if paper else
                  "GotenNet (ICLR 2025) Table 3 (kcal/mol, kcal/mol/Å) x 43.364",
        "family": _MD22_FAMILY.get(name, "CG-free"),
        "decimals": decimals,
        "energy": energy,
        "force": force,
    }


MD22 = {name: _md22_entry(name) for name in _MD22_KCAL}
# Columns of the paper's MD22 force and energy MAE tables.
MD22_PAPER_COLUMNS = ["sGDML", "Allegro", "MACE", "Equiformer", "ViSNet", "GotenNet-B"]

# ---------------------------------------------------------------------------
# SPICE / MACE-OFF23 test set, per subset
# ---------------------------------------------------------------------------
# Kovács et al., "MACE-OFF23: Transferable machine learning force fields for
# organic molecules", arXiv:2312.15211, SI Table S1 (test errors).  E in
# meV/atom, F in meV/Å.  Order per subset: (E MAE, E RMSE, F MAE, F RMSE).
# Table S1 also has a Tripeptides column, which our test file does not contain.
_S1 = {
    "MACE-OFF23-S": {
        "PubChem": (1.41, 2.74, 35.68, 61.83),
        "DES370K Monomers": (1.04, 1.47, 17.63, 26.15),
        "DES370K Dimers": (0.98, 1.51, 16.31, 28.48),
        "Dipeptides": (0.84, 1.26, 25.07, 36.97),
        "Solvated Amino Acids": (1.60, 1.98, 38.56, 53.55),
        "Water": (1.67, 2.07, 28.53, 39.33),
        "QMugs": (1.03, 1.28, 41.45, 62.46),
    },
    "MACE-OFF23-M": {
        "PubChem": (0.91, 2.02, 20.57, 40.51),
        "DES370K Monomers": (0.63, 0.90, 9.36, 14.31),
        "DES370K Dimers": (0.58, 0.91, 9.02, 16.81),
        "Dipeptides": (0.52, 0.85, 14.27, 22.25),
        "Solvated Amino Acids": (1.21, 1.55, 23.26, 32.19),
        "Water": (0.76, 0.99, 15.27, 21.40),
        "QMugs": (0.69, 0.89, 23.58, 36.73),
    },
    "MACE-OFF23-L": {
        "PubChem": (0.88, 2.48, 14.75, 33.34),
        "DES370K Monomers": (0.59, 0.84, 6.58, 10.27),
        "DES370K Dimers": (0.54, 0.87, 6.62, 12.81),
        "Dipeptides": (0.42, 0.70, 10.19, 16.19),
        "Solvated Amino Acids": (0.98, 1.32, 19.43, 26.91),
        "Water": (0.83, 0.99, 13.57, 18.78),
        "QMugs": (0.45, 0.58, 16.93, 27.17),
    },
}
SPICE_METRICS = ["e_mae_meV_atom", "e_rmse_meV_atom", "f_mae_meVA", "f_rmse_meVA"]
SPICE = {
    name: {
        "citation": "Kovács et al., MACE-OFF23, arXiv:2312.15211 (2023), SI Table S1",
        "source": "MACE-OFF23 SI Table S1 (E meV/atom, F meV/Å)",
        "decimals": 2,
        "subsets": {s: dict(zip(SPICE_METRICS, v)) for s, v in table.items()},
    }
    for name, table in _S1.items()
}

BENCHMARKS = {"rmd17": RMD17, "md22": MD22, "spice": SPICE}


def value_decimals(value, model_entry, molecule=None):
    """Printed precision of a baseline value (for tie decisions)."""
    dec = model_entry.get("decimals")
    if isinstance(dec, dict):
        return dec.get(molecule, 2)
    if dec is not None:
        return dec
    text = repr(float(value))
    return len(text.split(".")[1]) if "." in text and "e" not in text else 0


def get(benchmark, model, quantity, molecule):
    """Baseline value (meV or meV/Å) or None if not reported."""
    return BENCHMARKS[benchmark][model][quantity].get(molecule)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(v, nd):
    return "–" if v is None else f"{v:.{nd}f}"


def print_rmd17(out=sys.stdout):
    models = list(RMD17)
    for q, unit in (("force", "meV/Å"), ("energy", "meV")):
        print(f"\n### rMD17 {q} MAE ({unit})\n", file=out)
        print("| Molecule | " + " | ".join(models) + " |", file=out)
        print("|---|" + "---|" * len(models), file=out)
        for m in RMD17_MOLECULES:
            cells = []
            for name in models:
                v = RMD17[name][q][m]
                cells.append(_fmt(v, value_decimals(v, RMD17[name], m)))
            print(f"| {m} | " + " | ".join(cells) + " |", file=out)
        means = [sum(RMD17[n][q].values()) / len(RMD17_MOLECULES) for n in models]
        print("| mean | " + " | ".join(f"{x:.3f}" for x in means) + " |", file=out)


def print_md22(out=sys.stdout):
    models = list(MD22)
    for q, unit in (("force", "meV/Å"), ("energy", "meV")):
        print(f"\n### MD22 {q} MAE ({unit})\n", file=out)
        print("| Molecule | atoms | " + " | ".join(models) + " |", file=out)
        print("|---|---|" + "---|" * len(models), file=out)
        for m in MD22_MOLECULES:
            cells = [_fmt(MD22[n][q][m], value_decimals(MD22[n][q][m], MD22[n], m))
                     for n in models]
            print(f"| {m} | {MD22_ATOMS[m]} | " + " | ".join(cells) + " |", file=out)


def print_spice(out=sys.stdout):
    models = list(SPICE)
    labels = {"e_mae_meV_atom": "E MAE (meV/atom)", "e_rmse_meV_atom": "E RMSE (meV/atom)",
              "f_mae_meVA": "F MAE (meV/Å)", "f_rmse_meVA": "F RMSE (meV/Å)"}
    print("\n### SPICE (MACE-OFF23 test, SI Table S1)\n", file=out)
    print("| Subset | Metric | " + " | ".join(models) + " |", file=out)
    print("|---|---|" + "---|" * len(models), file=out)
    for s in SPICE_SUBSETS:
        for k in SPICE_METRICS:
            cells = [_fmt(SPICE[n]["subsets"][s][k], 2) for n in models]
            print(f"| {s} | {labels[k]} | " + " | ".join(cells) + " |", file=out)


def self_check():
    """Consistency with the numbers printed in the paper and the exact conversions."""
    problems = []
    goten_f = [5.663, 2.094, 0.204, 2.090, 3.599, 1.041, 4.024, 3.048, 1.132, 1.808]
    goten_e = [1.552, 0.525, 0.022, 0.308, 0.559, 0.169, 0.919, 0.611, 0.191, 0.278]
    for m, f, e in zip(RMD17_MOLECULES, goten_f, goten_e):
        if round(RMD17["GotenNet"]["force"][m], 3) != f:
            problems.append(f"GotenNet rMD17 F {m}: {RMD17['GotenNet']['force'][m]:.4f} vs {f}")
        if round(RMD17["GotenNet"]["energy"][m], 3) != e:
            problems.append(f"GotenNet rMD17 E {m}: {RMD17['GotenNet']['energy'][m]:.4f} vs {e}")
    known = {("sGDML", "energy", "stachyose"), ("sGDML", "force", "stachyose"),
             ("MACE", "energy", "AT-AT-CG-CG"), ("Equiformer", "energy", "AT-AT-CG-CG")}
    for name, (es, fs) in _PAPER_MD22.items():
        for i, m in enumerate(MD22_MOLECULES[:5]):
            for q, printed, kc in (("energy", es[i], _MD22_KCAL[name][0][i]),
                                   ("force", fs[i], _MD22_KCAL[name][1][i])):
                conv = kc * KCALMOL_TO_MEV
                if abs(round(conv, 1) - printed) > 1e-9:
                    tag = "known" if (name, q, m) in known else "MISMATCH"
                    problems.append(f"{tag}: MD22 {name} {q} {m}: GotenNet-T3 conversion "
                                    f"{conv:.2f} vs paper {printed}")
    c8 = {"buckyball-catcher": [29.57, 3.85, 4.83, 5.79, 4.73, 3.63, 3.42],
          "double-walled_nanotube": [22.68, 14.87, 11.91, 17.17, 10.72, 10.68, 8.19]}
    for m, vals in c8.items():
        for name, v in zip(["sGDML", "Allegro", "Equiformer", "ViSNet", "QuinNet",
                            "GotenNet-S", "GotenNet-B"], vals):
            if round(MD22[name]["force"][m], 2) != v:
                problems.append(f"MD22 {name} F {m}: {MD22[name]['force'][m]:.3f} vs {v}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print published baseline tables.")
    ap.add_argument("--benchmark", choices=["rmd17", "md22", "spice", "all"], default="all")
    ap.add_argument("--check", action="store_true",
                    help="Check conversions against the paper values.")
    args = ap.parse_args(argv)
    if args.check:
        problems = self_check()
        bad = 0
        for p in problems:
            if not p.startswith(("known", "MISMATCH")):
                p = "MISMATCH: " + p
            bad += p.startswith("MISMATCH")
            print(p)
        print("baselines self-check:", "OK" if not bad else f"{bad} mismatch(es)")
        return 1 if bad else 0
    if args.benchmark in ("rmd17", "all"):
        print_rmd17()
    if args.benchmark in ("md22", "all"):
        print_md22()
    if args.benchmark in ("spice", "all"):
        print_spice()
    return 0


if __name__ == "__main__":
    sys.exit(main())
