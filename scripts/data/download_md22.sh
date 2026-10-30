#!/usr/bin/env bash
# Download the MD22 dataset (sGDML format, one .npz per system).
#
#   Chmiela et al. (2023), "Accurate global machine learning force fields for molecules with
#   hundreds of atoms", Science Advances 9, eadf0873.  Distributed via http://www.sgdml.org/.
#   Please check the dataset's terms of use on sgdml.org before redistributing.
#
# Usage (from the repo root):
#   bash scripts/data/download_md22.sh [DEST] [system ...]   # DEST defaults to data/raw/md22
#
# Result: DEST/md22_<system>.npz for the 7 systems (~565 MB in total):
#   Ac-Ala3-NHMe DHA AT-AT stachyose AT-AT-CG-CG buckyball-catcher double-walled_nanotube
#
# Next step:
#   python scripts/data/prepare_md22.py --raw_dir DEST --out_dir data/md22
#
# URLs checked on 2026-09-28: every file below returned HTTP 200 with the byte size of our copy.
# If the host moves, set MD22_BASE_URL to the new directory; the md5 check still applies.
set -euo pipefail

DEST=${1:-data/raw/md22}
shift || true
BASE=${MD22_BASE_URL:-https://www.quantum-machine.org/gdml/repo/datasets}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_download_common.sh
source "$HERE/_download_common.sh"

# md5 of the files we trained on (also in scripts/data/checksums.md5).  A case statement rather
# than an associative array so the script also runs under macOS's bash 3.2.
md5_for() {
    case "$1" in
        Ac-Ala3-NHMe)           echo f130a28fe97a49c4c0d535e58b7fb4d8 ;;
        DHA)                    echo cbb86fed19f6d2698f41be5c90f86047 ;;
        AT-AT)                  echo 1b73fd567f84c5bdd58a4c1f382c0b7b ;;
        stachyose)              echo 7b3798cec7d6729c50ae6b9d963cd817 ;;
        AT-AT-CG-CG)            echo 43261f58a24ca5a54aa6b8ab4947a8d2 ;;
        buckyball-catcher)      echo d010c8c076e7f11d20e7477f193458ac ;;
        double-walled_nanotube) echo 5ca2b6fe68ebdf3b6e382727448424af ;;
        *)                      echo "" ;;
    esac
}
SYSTEMS=(Ac-Ala3-NHMe DHA AT-AT stachyose AT-AT-CG-CG buckyball-catcher double-walled_nanotube)
if [[ $# -gt 0 ]]; then
    SYSTEMS=("$@")
fi

for mol in "${SYSTEMS[@]}"; do
    want=$(md5_for "$mol")
    if [[ -z "$want" ]]; then
        echo "ERROR: unknown MD22 system '$mol'" >&2
        echo "       known: Ac-Ala3-NHMe DHA AT-AT stachyose AT-AT-CG-CG buckyball-catcher double-walled_nanotube" >&2
        exit 1
    fi
    out="$DEST/md22_${mol}.npz"
    fetch "$BASE/md22_${mol}.npz" "$out"
    check_md5 "$out" "$want"
done
echo "[done] MD22 files in $DEST"
