#!/usr/bin/env bash
# Run as your normal user; only system package installation uses sudo.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
skip_system=0
case "${1:-}" in
    --skip-system) skip_system=1 ;;
    -h|--help)
        echo "Usage: bash setup_linux.sh [--skip-system]"
        echo "Creates .venv using Python 3.10–3.12 and installs requirements.txt."
        echo "Installs Ubuntu/Debian system dependencies unless --skip-system is given."
        echo "Set PYTHON=/path/to/python3.12 to select an interpreter."
        exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
esac
if (( $# > 1 )); then
    echo "Too many arguments; use --help." >&2
    exit 1
fi
if [[ "$(uname -s)" != Linux ]]; then
    echo "Run this script on the Linux receiver." >&2
    exit 1
fi

python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    for candidate in python3.12 python3.11 python3.10; do
        if command -v "$candidate" >/dev/null 2>&1; then
            python_bin="$candidate"
            break
        fi
    done
fi
if [[ -z "$python_bin" ]]; then
    echo "Install Python 3.10, 3.11, or 3.12, then rerun this script." >&2
    exit 1
fi
check_python() {
    "$1" -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "This project requires Python 3.10–3.12"'
}
check_python "$python_bin"

if (( ! skip_system )); then
    if ! command -v apt-get >/dev/null 2>&1; then
        echo "Automatic system setup supports Ubuntu/Debian." >&2
        echo "Install your Python venv support, OpenGL, GLib, and EGL libraries, then rerun with --skip-system." >&2
        exit 1
    fi
    elevate=()
    if (( EUID != 0 )); then
        elevate=(sudo)
    fi
    python_version="$("$python_bin" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    venv_package="python${python_version}-venv"
    venv_packages=()
    if ! "$python_bin" -m venv --help >/dev/null 2>&1; then
        if ! apt-cache show "$venv_package" >/dev/null 2>&1; then
            echo "${python_bin} does not provide venv, and ${venv_package} is unavailable from apt." >&2
            exit 1
        fi
        venv_packages=("$venv_package")
    fi
    "${elevate[@]}" apt-get update
    "${elevate[@]}" apt-get install -y "${venv_packages[@]}" libgl1 libglib2.0-0 libegl1
fi

cd "$project_dir"
if [[ -e .venv ]]; then
    if [[ ! -x .venv/bin/python ]]; then
        echo ".venv exists but is not a usable Linux virtual environment. Move it aside and rerun." >&2
        exit 1
    fi
    check_python .venv/bin/python
else
    "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python - <<'PY'
import cv2
import mediapipe
import numpy
import pythonosc
import mido
import matplotlib
from classes.app import OSCGestureApp
from pathlib import Path

if not Path("hand_landmarker.task").is_file():
    raise SystemExit("Missing hand_landmarker.task: copy the model into the project root.")
print("Dependencies and application imports OK.")
PY

printf '\nSetup complete. From the project directory, run:\n'
echo '  source .venv/bin/activate'
echo '  python main.py --camera-url http://<macbook-ip>:8080/stream'
