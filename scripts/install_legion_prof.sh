#!/usr/bin/env bash
# Builds legion_prof from the pinned upmem-legion; installs rustup if needed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGION_DIR="${LEGION_DIR:-${ROOT}/opt/upmem-legion}"
DEST_DIR="${1:-${ROOT}/opt/legion_prof}"
CARGO="${CARGO_HOME:-$HOME/.cargo}/bin/cargo"

if [ ! -d "$LEGION_DIR/tools/legion_prof_rs" ]; then
    echo "No upmem-legion at $LEGION_DIR; run scripts/install_legion.sh first" >&2
    exit 1
fi

if ! command -v cargo >/dev/null 2>&1 && [ ! -x "$CARGO" ]; then
    echo "Installing Rust (rustup, no PATH changes)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
        sh -s -- -y --no-modify-path --profile minimal
fi
command -v cargo >/dev/null 2>&1 && CARGO=cargo

echo "Building legion_prof into $DEST_DIR..."
"$CARGO" install --locked --path "$LEGION_DIR/tools/legion_prof_rs" --root "$DEST_DIR"

echo "Installed $DEST_DIR/bin/legion_prof"
