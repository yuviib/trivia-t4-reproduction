#!/usr/bin/env bash
# Rebuilds the tested layout: upstream TRivia at a pinned commit, the fixes in
# patches/trivia-fixes.patch applied, and reproduction/ copied in.
# Usage: ./setup.sh [destination]     (default: ./TRivia)
set -euo pipefail

UPSTREAM="${TRIVIA_UPSTREAM:-https://github.com/opendatalab/TRivia.git}"
COMMIT="fdaebd7d22f5ee8ac379bc929b3f751af7cae36b"
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$HERE/TRivia}"

git clone "$UPSTREAM" "$DEST"
git -C "$DEST" checkout -q "$COMMIT"
git -C "$DEST" apply "$HERE/patches/trivia-fixes.patch"
cp -r "$HERE/reproduction" "$DEST/reproduction"
echo "Ready: $DEST (upstream ${COMMIT:0:7} + fixes + reproduction/)"
