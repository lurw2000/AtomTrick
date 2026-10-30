#!/usr/bin/env bash
# Download the SPICE train/test split released with MACE-OFF23.
#
#   Kovacs et al., "MACE-OFF23: Transferable machine learning force fields for organic molecules"
#   (arXiv:2312.15211).  Data: University of Cambridge Apollo repository, DOI 10.17863/CAM.107498
#   (handle 1810/366661), MIT license.  These are MACE-OFF23's own files: the filtered training set
#   (951,005 configurations, their train+val) and the held-out test set (50,195 configurations;
#   whether their Table S1 is computed on exactly this file, or on a filtered copy, is not
#   documented).  The DFT labels are derived from SPICE 1 (Eastman et al.).
#
# Usage (from the repo root):
#   bash scripts/data/download_spice_maceoff.sh [DEST]       # DEST defaults to data/raw/spice_maceoff
#
# Result (~1.7 GB download, ~5.2 GB extracted):
#   DEST/train_large_neut_no_bad_clean.xyz   DEST/test_large_neut_all.xyz
#
# Next step:
#   python scripts/data/prepare_spice_maceoff.py --raw_dir DEST --out_dir data/spice_maceoff
#
# Verify: the Apollo bitstream URLs were checked on 2026-09-28 (the server returned the expected
# file names and byte sizes, 1,619,411,395 and 85,448,756).  Apollo occasionally resets long
# downloads; re-running resumes the partial file.  If the direct links stop working, download the
# two tarballs by hand from https://doi.org/10.17863/CAM.107498 into DEST and re-run this script.
set -euo pipefail

DEST=${1:-data/raw/spice_maceoff}
APOLLO=${APOLLO_BASE_URL:-https://www.repository.cam.ac.uk/bitstreams}

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=_download_common.sh
source "$HERE/_download_common.sh"

# name | Apollo bitstream uuid | md5 of the tarball we used
FILES="
train_large_neut_no_bad_clean b185b5ab-91cf-489a-9302-63bfac42824a 882f32ecdfaf0065025f0f3cd8010d86
test_large_neut_all           cb8351dd-f09c-413f-921c-67a702a7f0c5 bf1f7c7ff1714e6ac6af2cdc642a130a
"

echo "$FILES" | while read -r name uuid md5; do
    [[ -z "$name" ]] && continue
    tarball="$DEST/$name.tar.gz"
    if [[ -s "$DEST/$name.xyz" ]]; then
        echo "[skip] $DEST/$name.xyz already extracted"
        continue
    fi
    fetch "$APOLLO/$uuid/download" "$tarball"
    check_md5 "$tarball" "$md5"
    echo "[untar] $tarball"
    # The tarballs were made on macOS: skip the AppleDouble "._*" stubs.  GNU tar may print
    # "Ignoring unknown extended header keyword 'LIBARCHIVE.xattr...'" -- harmless.
    tar -xzf "$tarball" -C "$DEST" --exclude='._*'
    if [[ ! -s "$DEST/$name.xyz" ]]; then
        # Tolerate a leading directory inside the tarball.
        found=$(find "$DEST" -maxdepth 3 -name "$name.xyz" | head -n 1)
        if [[ -z "$found" ]]; then
            echo "ERROR: $name.xyz not found after extracting $tarball" >&2
            exit 1
        fi
        mv "$found" "$DEST/$name.xyz"
    fi
done
echo "[done] SPICE (MACE-OFF23) xyz files in $DEST"
echo "       the tarballs can be deleted once prepare_spice_maceoff.py has run"
