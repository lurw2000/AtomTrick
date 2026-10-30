#!/usr/bin/env bash
# Download the revised MD17 (rMD17) dataset.
#
#   Christensen & von Lilienfeld (2020), "On the role of gradients for machine learning of
#   molecular energies and forces".  figshare DOI 10.6084/m9.figshare.12672038, license CC0-1.0.
#   File: rmd17.tar.bz2 (figshare file id 23950376, 1,066,301,513 bytes).
#
# Usage (from the repo root):
#   bash scripts/data/download_rmd17.sh [DEST]          # DEST defaults to data/raw
#
# Result:
#   DEST/rmd17/npz_data/rmd17_<molecule>.npz            (10 molecules, 100k frames each)
#   DEST/rmd17/splits/index_{train,test}_0{1..5}.csv    (official 1000/1000 splits; we use split 01)
#
# Next step:
#   python scripts/data/prepare_rmd17.py --raw_dir DEST/rmd17 --out_dir data/rmd17
#
# The URL was checked on 2026-09-28 (figshare API: rmd17.tar.bz2, md5 below).  If figshare moves
# the file, set RMD17_URL to any mirror of rmd17.tar.bz2; the md5 check still applies.
set -euo pipefail

DEST=${1:-data/raw}
URL=${RMD17_URL:-https://ndownloader.figshare.com/files/23950376}
MD5=cb1a927628d96f2e966025da4fb63d18

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_download_common.sh
source "$HERE/_download_common.sh"

ARCHIVE="$DEST/rmd17.tar.bz2"
fetch "$URL" "$ARCHIVE"
check_md5 "$ARCHIVE" "$MD5"

if [[ -d "$DEST/rmd17/npz_data" && -d "$DEST/rmd17/splits" ]]; then
    echo "[skip] $DEST/rmd17 already extracted"
else
    echo "[untar] $ARCHIVE -> $DEST"
    tar -xjf "$ARCHIVE" -C "$DEST"
    # The archive is expected to unpack to rmd17/{npz_data,splits}.  Be tolerant of a different
    # top-level folder name so the layout the processor needs is always DEST/rmd17.
    if [[ ! -d "$DEST/rmd17/npz_data" ]]; then
        found=$(find "$DEST" -maxdepth 3 -type d -name npz_data | head -n 1)
        if [[ -z "$found" ]]; then
            echo "ERROR: no npz_data/ directory found after extracting $ARCHIVE" >&2
            exit 1
        fi
        top=$(dirname "$found")
        echo "[move] $top -> $DEST/rmd17"
        mkdir -p "$DEST/rmd17"
        mv "$top/npz_data" "$top/splits" "$DEST/rmd17/"
    fi
fi

n=$(ls "$DEST"/rmd17/npz_data/rmd17_*.npz 2>/dev/null | wc -l || true)
echo "[done] $n molecule files in $DEST/rmd17/npz_data (expected 10)"
echo "       verify extracted files:  md5sum -c --ignore-missing scripts/data/checksums.md5"
echo "       (paths in checksums.md5 assume DEST=data/raw)"
