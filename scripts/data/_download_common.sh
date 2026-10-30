# Shared helpers for scripts/data/download_*.sh.  Source this file; do not run it.
#
#   fetch URL OUT          resumable download (curl, falls back to wget); skips if OUT exists
#   check_md5 FILE MD5     verify an md5 (warns and continues when no md5 tool is installed)

fetch() {
    local url=$1 out=$2
    if [[ -s "$out" ]]; then
        echo "[skip] $out already exists"
        return 0
    fi
    mkdir -p "$(dirname "$out")"
    echo "[get ] $url"
    echo "       -> $out"
    if command -v curl >/dev/null 2>&1; then
        # -C - resumes a partial .part file; --retry covers the connection resets we saw
        # from both figshare and Apollo on large files.
        curl -L --fail --retry 5 --retry-delay 5 -C - -o "$out.part" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -c -O "$out.part" "$url"
    else
        echo "ERROR: need curl or wget" >&2
        return 1
    fi
    mv "$out.part" "$out"
}

_md5_of() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then   # macOS
        md5 -q "$1"
    else
        echo ""
    fi
}

check_md5() {
    local file=$1 want=$2 got
    got=$(_md5_of "$file")
    if [[ -z "$got" ]]; then
        echo "[warn] no md5 tool found; not verifying $file"
    elif [[ "$got" == "$want" ]]; then
        echo "[ ok ] md5 $file"
    else
        echo "ERROR: md5 mismatch for $file" >&2
        echo "       expected $want" >&2
        echo "       got      $got" >&2
        echo "       (delete the file and re-run to download it again)" >&2
        return 1
    fi
}
