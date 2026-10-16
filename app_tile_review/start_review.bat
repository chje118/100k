@echo off
setlocal EnableExtensions
title Slide review

REM ================================================================
REM  SETTINGS
REM ================================================================

REM Conda environment the app runs in
set "ENV_NAME=review"

REM Optional: conda installation folder (the one containing Scripts\activate.bat).
REM Leave empty to find it automatically.
set "CONDA_ROOT="

REM review_app.py (and its static\ folder) next to this file:
REM   E:\Christine\repos\100k\app_tile_review\
set "APP=%~dp0review_app.py"

set "LYMPHOMA_DIR=E:\Christine\100k_lymphoma"
set "ENDOMETRIUM_DIR=E:\Christine\100k_endometrium"

REM Quick-pick diagnosis buttons (comma separated, empty = free text only)
set "LYMPHOMA_DX=FL,DLBCL,cHL,Nodular hyperplasia"
set "ENDOMETRIUM_DX=Normal endometrium,Endometrial polyp,Hyperplasia,Atypical hyperplasia, Adenocarcinoma"

REM ================================================================

if not exist "%APP%" (
    echo review_app.py not found next to this file:
    echo   %APP%
    goto :end
)

REM ---------- find conda ----------
if defined CONDA_ROOT goto :check_conda
REM 1) conda already known in this window / on PATH
for /f "delims=" %%C in ('where conda.bat 2^>nul') do for %%D in ("%%~dpC..") do (
    if not defined CONDA_ROOT if exist "%%~fD\Scripts\activate.bat" set "CONDA_ROOT=%%~fD"
)
REM 2) common install locations (this user, all users, any user on this machine)
for %%R in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniforge3" "C:\miniconda3" "C:\anaconda3" "C:\miniforge3" "E:\miniconda3" "E:\anaconda3") do (
    if not defined CONDA_ROOT if exist "%%~R\Scripts\activate.bat" set "CONDA_ROOT=%%~R"
)
if not defined CONDA_ROOT for /d %%U in ("C:\Users\*") do for %%N in (miniconda3 anaconda3 miniforge3) do (
    if not defined CONDA_ROOT if exist "%%~U\%%N\envs\%ENV_NAME%\python.exe" set "CONDA_ROOT=%%~U\%%N"
)

:check_conda
if not defined CONDA_ROOT goto :no_conda
if not exist "%CONDA_ROOT%\Scripts\activate.bat" goto :no_conda

REM ---------- activate the environment ----------
call "%CONDA_ROOT%\Scripts\activate.bat" "%ENV_NAME%"
if /i not "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" (
    echo.
    echo   Could not activate the conda environment "%ENV_NAME%"
    echo   using conda in: %CONDA_ROOT%
    goto :end
)
python -c "import flask, openslide" 2>nul
if errorlevel 1 (
    echo.
    echo   The "%ENV_NAME%" environment is missing packages. Install with:
    echo     conda install -n %ENV_NAME% -c conda-forge flask pillow waitress openslide openslide-python
    goto :end
)

REM ---------- choose project ----------
echo.
echo   Slide review
echo   ------------
echo     1   Lymphoma
echo     2   Endometrium
echo.
choice /c 12 /n /m "  Choose 1 or 2: "
if errorlevel 2 goto :endometrium

:lymphoma
set "PROJECT=%LYMPHOMA_DIR%"
set "DX=%LYMPHOMA_DX%"
set "PORT=8050"
set "NAME=Lymphoma"
goto :run

:endometrium
set "PROJECT=%ENDOMETRIUM_DIR%"
set "DX=%ENDOMETRIUM_DX%"
set "PORT=8051"
set "NAME=Endometrium"

:run
title Slide review - %NAME%
if not exist "%PROJECT%\" (
    echo.
    echo   Folder not found: %PROJECT%
    goto :end
)

REM Already running? Then just open the page.
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
    echo.
    echo   %NAME% review is already running - opening it in the browser.
    start "" "http://localhost:%PORT%/"
    timeout /t 3 >nul
    exit /b
)

echo.
echo   Environment: %CONDA_DEFAULT_ENV%   (%CONDA_PREFIX%)
echo   Project:     %PROJECT%
echo.
echo   Starting %NAME% review ... the browser opens by itself.
echo   KEEP THIS WINDOW OPEN while you work. Close it when you are done.
echo.
python "%APP%" "%PROJECT%" --port %PORT% --diagnoses "%DX%"
goto :end

:no_conda
echo.
echo   Could not find a conda installation with the "%ENV_NAME%" environment.
echo   Open this file in Notepad and set CONDA_ROOT at the top, e.g.
echo     set "CONDA_ROOT=C:\Users\yourname\miniconda3"
echo   (find it in Anaconda Prompt with:  where conda)

:end
echo.
pause
