#!/usr/bin/env bash
# Build and run tests at each commit from BASE..HEAD, one at a time.
# Usage: scripts/test_each_commit.sh [BASE] [-- <runtests.py args>]
#   BASE defaults to fbaba1738 (branch tip before the breakup commits).
set -euo pipefail

BASE="${1:-fbaba1738}"
shift || true
if [ "${1:-}" = "--" ]; then shift; fi
RUNTESTS_ARGS=("$@")

ORIG_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMITS=($(git rev-list --reverse "${BASE}..HEAD"))

cleanup() { git checkout -q "$ORIG_BRANCH"; }
trap cleanup EXIT

for c in "${COMMITS[@]}"; do
    subject="$(git log -1 --format=%s "$c")"
    echo "=============================================="
    echo "commit $c  $subject"
    echo "=============================================="
    git checkout -q "$c"
    make -j"$(nproc)" DIM=2 DEBUG=False
    python3 scripts/runtests.py "${RUNTESTS_ARGS[@]}"
done
