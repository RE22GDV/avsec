#!/usr/bin/env bash
# =============================================================================
#  avsec — launcher for Linux and macOS
#
#  Creates the virtual environment on first run, installs dependencies, then
#  either opens the web interface or runs the command you asked for.
#
#    ./run.sh              web interface (default)
#    ./run.sh ui           web interface
#    ./run.sh test         the test suite
#    ./run.sh demo         one frame through every method
#    ./run.sh check        dataset + protocol checks
#    ./run.sh research     the full research programme (long)
#    ./run.sh lab          ISITIA 2021: reproduce the scheme + repeat on photos
#    ./run.sh subst-lab    B2s: keyed substitution on top of the permutation
#    ./run.sh verify       recheck every published number, runs nothing
#    ./run.sh sources      download the natural UAV photographs
#    ./run.sh shell        a shell with the environment active
#    ./run.sh <anything>   passed straight to `avsec`, e.g. ./run.sh budget
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PY="$VENV/bin/python"

# ------------------------------------------------------------------ interpreter
#
# Pick a Python the scientific stack actually ships wheels for.  The newest
# installed interpreter is routinely ahead of numpy/scipy wheel availability,
# and then pip falls back to building scipy from source - which needs a Fortran
# compiler and fails on a normal desktop.  Known-good versions come first.
find_python() {
    if [ -n "${AVSEC_PYTHON:-}" ]; then echo "$AVSEC_PYTHON"; return; fi
    for v in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$v" >/dev/null 2>&1 &&
           "$v" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3,10) else 1)'                >/dev/null 2>&1; then
            command -v "$v"; return
        fi
    done
}

if [ ! -x "$PY" ]; then
    BASE_PY="$(find_python)"
    if [ -z "$BASE_PY" ]; then
        echo "[avsec] ERROR: no Python 3.10+ found." >&2
        echo "        Install one, or set AVSEC_PYTHON=/path/to/python" >&2
        exit 1
    fi
    echo "[avsec] interpreter: $BASE_PY ($("$BASE_PY" -V 2>&1))"
    echo "[avsec] creating the virtual environment in $VENV ..."
    "$BASE_PY" -m venv "$VENV"
    echo "[avsec] installing dependencies (a few minutes on first run) ..."
    "$PY" -m pip install --upgrade pip --quiet
    if ! "$PY" -m pip install -r requirements-dev.txt --quiet --only-binary=:all:
    then
        echo "[avsec] pre-built packages unavailable; allowing source builds ..."
        if ! "$PY" -m pip install -r requirements-dev.txt; then
            echo >&2
            echo "[avsec] ERROR: dependency installation failed." >&2
            echo "        The usual cause is a Python version numpy/scipy do not" >&2
            echo "        ship wheels for yet. Remove .venv and retry with:" >&2
            echo "            AVSEC_PYTHON=python3.12 ./run.sh" >&2
            exit 1
        fi
    fi
    "$PY" -m pip install -e . --quiet
    echo "[avsec] environment ready."
    echo
fi

export PYTHONPATH="$PWD/src"
export PYTHONIOENCODING=utf-8

CMD="${1:-ui}"
shift || true

case "$CMD" in
  ui)
    echo "[avsec] web interface: http://127.0.0.1:8765/"
    echo "[avsec] press Ctrl+C to stop."
    exec "$PY" -m avsec.cli ui --port 8765 "$@"
    ;;
  test)
    exec "$PY" -m pytest tests -q "$@"
    ;;
  demo)
    "$PY" -m avsec.cli demo --config configs/smoke.yaml --output runs/demo
    echo "[avsec] images and metrics written to runs/demo"
    ;;
  check)
    "$PY" -m avsec.cli dataset validate --config configs/research_main.yaml --output runs/_check
    "$PY" -m avsec.cli protocol-check --config configs/research_main.yaml --output runs/_check
    ;;
  research)
    echo "[avsec] this runs the full programme and takes about an hour."
    echo "[avsec] it resumes if interrupted — just run it again."
    "$PY" scripts/run_matrix.py configs/research_main.yaml runs/main 16
    "$PY" scripts/run_program.py configs/research_main.yaml runs/main 16
    "$PY" -m avsec.cli analyze --input runs/main --plan configs/analysis_plan.yaml
    "$PY" -m avsec.cli plots --input runs/main
    "$PY" -m avsec.cli verify --input runs/main
    "$PY" scripts/publish.py runs/main
    ;;
  subst-lab)
    echo "[avsec] B2s: substitution table, attacks, channel cost ..."
    "$PY" -m avsec.cli subst-lab --config configs/research_main.yaml --output results/b2s
    echo "[avsec] tables and figures written to results/b2s"
    ;;

  lab)
    echo "[avsec] ISITIA 2021: reproducing the scheme and repeating it on photos."
    "$PY" -m avsec.cli lab --config configs/research_main.yaml --output results/lab
    echo "[avsec] figure and tables written to results/lab"
    ;;
  verify)
    echo "[avsec] rechecking every published number from the published tables."
    "$PY" -m avsec.cli verify --input results/main
    ;;
  sources)
    "$PY" scripts/fetch_drone_photo.py
    "$PY" scripts/fetch_natural_sources.py
    ;;
  shell)
    echo "[avsec] environment active. Type \`avsec --help\` or \`exit\`."
    exec "${SHELL:-/bin/bash}" --rcfile <(echo "source $VENV/bin/activate")
    ;;
  *)
    exec "$PY" -m avsec.cli "$CMD" "$@"
    ;;
esac
