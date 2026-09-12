@echo off
REM ============================================================================
REM  avsec - launcher for Windows
REM
REM  Creates the virtual environment on first run, installs dependencies, then
REM  either opens the web interface or runs the command you asked for.
REM
REM    run.bat              web interface (default)
REM    run.bat ui           web interface
REM    run.bat test         the test suite
REM    run.bat demo         one frame through every method
REM    run.bat check        dataset + protocol checks
REM    run.bat research     the full research programme (long)
REM    run.bat verify       recheck every published number, runs nothing
REM    run.bat sources      download the natural UAV photographs
REM    run.bat shell        a shell with the environment active
REM    run.bat <anything>   passed straight to `avsec`, e.g. run.bat budget
REM
REM  Set AVSEC_PYTHON to force a specific interpreter, e.g.
REM    set AVSEC_PYTHON=C:\Python312\python.exe
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "VENV=.venv"
set "PY=%VENV%\Scripts\python.exe"

REM ---------------------------------------------------------------- interpreter
if not exist "%PY%" (
    call :find_python
    if not defined BASE_PY (
        echo.
        echo [avsec] ERROR: no suitable Python found.
        echo         Install Python 3.12 from https://www.python.org/downloads/
        echo         and tick "Add python.exe to PATH", or set AVSEC_PYTHON.
        exit /b 1
    )
    echo [avsec] interpreter: !BASE_PY!
    echo [avsec] creating the virtual environment in %VENV% ...
    "!BASE_PY!" -m venv "%VENV%"
    if not exist "%PY%" (
        echo [avsec] ERROR: could not create a virtual environment with !BASE_PY!
        exit /b 1
    )
    echo [avsec] installing dependencies ^(a few minutes on first run^) ...
    "%PY%" -m pip install --upgrade pip --quiet
    "%PY%" -m pip install -r requirements-dev.txt --quiet --only-binary=:all:
    if !errorlevel! neq 0 (
        echo.
        echo [avsec] Pre-built packages were not available for this interpreter,
        echo [avsec] retrying and allowing source builds ...
        "%PY%" -m pip install -r requirements-dev.txt
        if !errorlevel! neq 0 (
            echo.
            echo [avsec] ERROR: dependency installation failed.
            echo         The usual cause is a Python version that numpy/scipy do
            echo         not yet ship wheels for. Remove the .venv folder and
            echo         retry with an older interpreter, for example:
            echo             set AVSEC_PYTHON=py -3.12
            echo             run.bat
            exit /b 1
        )
    )
    "%PY%" -m pip install -e . --quiet
    echo [avsec] environment ready.
    echo.
)

set "PYTHONPATH=%CD%\src"
set "PYTHONIOENCODING=utf-8"

REM ------------------------------------------------------------------ dispatch
set "CMD=%~1"
if "%CMD%"=="" set "CMD=ui"

if /i "%CMD%"=="ui" (
    echo [avsec] web interface: http://127.0.0.1:8765/
    echo [avsec] press Ctrl+C to stop.
    "%PY%" -m avsec.cli ui --port 8765
    goto :eof
)

if /i "%CMD%"=="test" (
    "%PY%" -m pytest tests -q
    goto :eof
)

if /i "%CMD%"=="demo" (
    "%PY%" -m avsec.cli demo --config configs/smoke.yaml --output runs/demo
    echo [avsec] images and metrics written to runs\demo
    goto :eof
)

if /i "%CMD%"=="check" (
    "%PY%" -m avsec.cli dataset validate --config configs/research_main.yaml --output runs/_check
    "%PY%" -m avsec.cli protocol-check --config configs/research_main.yaml --output runs/_check
    goto :eof
)

if /i "%CMD%"=="research" (
    echo [avsec] this runs the full programme and takes about an hour.
    echo [avsec] it resumes if interrupted - just run it again.
    "%PY%" scripts\run_matrix.py configs\research_main.yaml runs\main 16
    "%PY%" scripts\run_program.py configs\research_main.yaml runs\main 16
    "%PY%" -m avsec.cli analyze --input runs/main --plan configs/analysis_plan.yaml
    "%PY%" -m avsec.cli plots --input runs/main
    "%PY%" -m avsec.cli verify --input runs/main
    "%PY%" scripts\publish.py runs\main
    goto :eof
)

if /i "%CMD%"=="verify" (
    echo [avsec] rechecking every published number from the published tables.
    "%PY%" -m avsec.cli verify --input results/main
    goto :eof
)

if /i "%CMD%"=="sources" (
    "%PY%" scripts\fetch_drone_photo.py
    "%PY%" scripts\fetch_natural_sources.py
    goto :eof
)

if /i "%CMD%"=="shell" (
    echo [avsec] environment active. Type `avsec --help` or `exit`.
    cmd /k "%VENV%\Scripts\activate.bat"
    goto :eof
)

REM anything else goes to the CLI as-is
"%PY%" -m avsec.cli %*
goto :eof


REM ----------------------------------------------------------------------------
REM  Pick an interpreter that the scientific stack actually ships wheels for.
REM
REM  `py -3` selects the NEWEST installed Python, which is routinely ahead of
REM  numpy/scipy wheel availability - on this machine that meant Python 3.13 and
REM  a failed attempt to compile scipy from source for want of a Fortran
REM  compiler.  Known-good versions are tried oldest-supported first.
REM ----------------------------------------------------------------------------
:find_python
set "BASE_PY="
if defined AVSEC_PYTHON (
    set "BASE_PY=%AVSEC_PYTHON%"
    goto :eof
)
where py >nul 2>nul
if !errorlevel! equ 0 (
    for %%v in (3.12 3.11 3.10 3.13) do (
        if not defined BASE_PY (
            py -%%v -c "import sys" >nul 2>nul
            if !errorlevel! equ 0 (
                for /f "delims=" %%p in ('py -%%v -c "import sys;print(sys.executable)"') do set "BASE_PY=%%p"
            )
        )
    )
)
if not defined BASE_PY (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        for /f "delims=" %%p in ('python -c "import sys;print(sys.executable)"') do set "BASE_PY=%%p"
    )
)
goto :eof
