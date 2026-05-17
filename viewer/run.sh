#!/usr/bin/env bash
# Convenience launcher for Linux / WSL.
# Forces Qt to use PySide6's bundled plugins (works around the common
# "conda Qt version mismatch" error) and prefers the wayland backend
# (works on WSLg + most modern Linux desktops).
#
# Usage:  bash viewer/run.sh
# Or:     chmod +x viewer/run.sh && ./viewer/run.sh
set -euo pipefail

PYSIDE_PLUGINS=$(python -c "import os, PySide6; print(os.path.join(os.path.dirname(PySide6.__file__), 'Qt', 'plugins'))")
export QT_PLUGIN_PATH="$PYSIDE_PLUGINS"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-wayland}"

# Run from repo root so `python -m viewer` resolves the package.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
exec python -m viewer "$@"
