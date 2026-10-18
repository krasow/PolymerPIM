#!/usr/bin/env bash
# Clones upmem-legion (Legion with the UPMEM Realm module) for RUNTIME=legion.
# The build writes objects next to its sources, so it gets its own checkout
# rather than sharing a working tree.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${1:-${ROOT}/opt/upmem-legion}"
LEGION_REPO="${LEGION_REPO:-git@github.com:PrescienceLab/upmem-legion.git}"
LEGION_REV="${LEGION_REV:-c1e3202ba}"

if [[ -d "${DEST_DIR}/.git" ]]; then
    echo "upmem-legion already installed: ${DEST_DIR}"
    exit 0
elif [[ -e "${DEST_DIR}" ]]; then
    echo "Cannot install upmem-legion: ${DEST_DIR} already exists" >&2
    exit 1
fi

git clone "${LEGION_REPO}" "${DEST_DIR}"
git -C "${DEST_DIR}" checkout --detach "${LEGION_REV}"
