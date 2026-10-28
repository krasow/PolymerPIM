#!/usr/bin/env bash
# Clones upmem-legion at a pinned commit for RUNTIME=legion.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${1:-${ROOT}/opt/upmem-legion}"
LEGION_REPO="${LEGION_REPO:-git@github.com:PrescienceLab/upmem-legion.git}"
LEGION_REV="${LEGION_REV:-a6da57041}"

if [[ -d "${DEST_DIR}/.git" ]]; then
    # Follow the pin when it moves.
    if ! git -C "${DEST_DIR}" cat-file -e "${LEGION_REV}^{commit}" 2>/dev/null; then
        git -C "${DEST_DIR}" fetch "${LEGION_REPO}"
    fi
    git -C "${DEST_DIR}" checkout --detach "${LEGION_REV}"
    exit 0
elif [[ -e "${DEST_DIR}" ]]; then
    echo "Cannot install upmem-legion: ${DEST_DIR} already exists" >&2
    exit 1
fi

git clone "${LEGION_REPO}" "${DEST_DIR}"
git -C "${DEST_DIR}" checkout --detach "${LEGION_REV}"
