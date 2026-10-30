#!/usr/bin/env python3
"""Verify processed dataset NPZs against the recorded array checksums.

Why not just ``md5sum -c checksums.md5``?  ``numpy.savez_compressed`` stamps
every zip member with the current time, so an NPZ you regenerate with
``prepare_*.py`` NEVER has the same file md5 as ours even when every array is
bit-identical.  ``checksums.md5`` is therefore only useful for byte-for-byte
copies of our files.  This script instead hashes the raw array payload of
each ``.npy`` member (the bytes after the npy header), which is independent of
zip timestamps, compression level and npy header padding.

Pure standard library (zipfile + hashlib): runs anywhere, no numpy needed.

Usage (from the repository root)::

    python scripts/data/verify_data.py                  # check every file in the manifest that exists
    python scripts/data/verify_data.py --only rmd17     # only paths containing "rmd17"
    python scripts/data/verify_data.py --strict         # missing files count as failures

Maintainers regenerate the manifest with ``--write`` (``--prefix`` sets the
repo-relative directory recorded for the given files)::

    python scripts/data/verify_data.py --write scripts/data/checksums_arrays.tsv \\
        --prefix data/rmd17/ /path/to/data/rmd17/*_default.npz
    python scripts/data/verify_data.py --write scripts/data/checksums_arrays.tsv --append \\
        --prefix data/md22/ /path/to/data/md22/*_default.npz

Exit status: 0 if every checked array matches, 1 otherwise.  Pickled object
arrays (``subset_names`` in the SPICE files) are reported but never fail the
check, because pickle bytes can differ across numpy/python versions.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import struct
import sys
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "scripts", "data", "checksums_arrays.tsv")
MAGIC = b"\x93NUMPY"
CHUNK = 1 << 22


def _read_exact(fh, n):
    buf = b""
    while len(buf) < n:
        part = fh.read(n - len(buf))
        if not part:
            raise ValueError("truncated .npy member")
        buf += part
    return buf


def hash_member(zf: zipfile.ZipFile, name: str):
    """Return (descr, shape, fortran_order, md5-of-payload) for one .npy member."""
    with zf.open(name) as fh:
        head = _read_exact(fh, 8)
        if head[:6] != MAGIC:
            raise ValueError(f"{name}: not an .npy member")
        major = head[6]
        if major == 1:
            (hlen,) = struct.unpack("<H", _read_exact(fh, 2))
        elif major in (2, 3):
            (hlen,) = struct.unpack("<I", _read_exact(fh, 4))
        else:
            raise ValueError(f"{name}: unsupported npy version {major}")
        header = ast.literal_eval(_read_exact(fh, hlen).decode("latin1"))
        md5 = hashlib.md5()
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            md5.update(block)
    shape = "x".join(str(s) for s in header["shape"]) or "scalar"
    return str(header["descr"]), shape, bool(header["fortran_order"]), md5.hexdigest()


def hash_npz(path: str):
    """Yield (member, descr, shape, md5) for every array in an npz file."""
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if not name.endswith(".npy"):
                continue
            descr, shape, fortran, md5 = hash_member(zf, name)
            if fortran:
                descr += ",F"
            yield name[:-4], descr, shape, md5


def write_manifest(out: str, files, prefix: str, append: bool) -> int:
    mode = "a" if append else "w"
    with open(out, mode) as fh:
        if not append:
            fh.write("# path\tarray\tdtype\tshape\tmd5_of_array_payload\n")
        for f in files:
            rel = prefix + os.path.basename(f)
            for member, descr, shape, md5 in hash_npz(f):
                fh.write(f"{rel}\t{member}\t{descr}\t{shape}\t{md5}\n")
            print(f"hashed {rel}")
    return 0


def check_manifest(manifest: str, root: str, only: str | None, strict: bool) -> int:
    expected = {}
    with open(manifest) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            path, member, descr, shape, md5 = line.rstrip("\n").split("\t")
            if only and only not in path:
                continue
            expected.setdefault(path, {})[member] = (descr, shape, md5)

    n_ok = n_bad = n_missing = n_soft = 0
    for path in sorted(expected):
        full = os.path.join(root, path)
        if not os.path.exists(full):
            n_missing += 1
            print(f"MISSING  {path}")
            continue
        got = {m: (d, s, h) for m, d, s, h in hash_npz(full)}
        problems, soft = [], []
        for member, (descr, shape, md5) in expected[path].items():
            if member not in got:
                problems.append(f"{member}: absent")
                continue
            g_descr, g_shape, g_md5 = got[member]
            if (g_descr, g_shape) != (descr, shape):
                problems.append(f"{member}: dtype/shape {g_descr} {g_shape} != {descr} {shape}")
            elif g_md5 != md5:
                (soft if descr.startswith("|O") else problems).append(f"{member}: payload md5 differs")
        extra = sorted(set(got) - set(expected[path]))
        if extra:
            soft.append("extra arrays: " + ",".join(extra))
        if problems:
            n_bad += 1
            print(f"FAIL     {path}: " + "; ".join(problems))
        else:
            n_ok += 1
            print(f"OK       {path}" + (f"   (note: {'; '.join(soft)})" if soft else ""))
        n_soft += bool(soft)

    print(f"\n{n_ok} OK, {n_bad} FAILED, {n_missing} missing"
          + (f", {n_soft} with non-fatal notes" if n_soft else ""))
    if n_bad or (strict and n_missing):
        return 1
    if n_ok == 0:
        print("nothing was checked (no manifest file exists under --root?)")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--root", default=REPO_ROOT,
                    help="directory the manifest paths are relative to (default: repo root)")
    ap.add_argument("--only", default=None, help="check only paths containing this substring")
    ap.add_argument("--strict", action="store_true", help="treat missing files as failures")
    ap.add_argument("--write", metavar="OUT", default=None,
                    help="(maintainers) hash the given npz files into manifest OUT")
    ap.add_argument("--append", action="store_true", help="with --write: append to OUT")
    ap.add_argument("--prefix", default="", help="with --write: repo-relative dir recorded per file")
    ap.add_argument("files", nargs="*")
    args = ap.parse_args()
    if args.write:
        if not args.files:
            ap.error("--write needs npz files")
        return write_manifest(args.write, args.files, args.prefix, args.append)
    if args.files:
        ap.error("positional files are only used with --write")
    return check_manifest(args.manifest, args.root, args.only, args.strict)


if __name__ == "__main__":
    sys.exit(main())
