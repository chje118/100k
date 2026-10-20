"""
Run ONCE, from the Python environment that runs the review app
(the one with openslide, flask and waitress):

    python make_shortcut.py                 # shortcut called "Slide review"
    python make_shortcut.py "Slide review DVP"

Creates a desktop shortcut. Double-clicking it starts review_app.py with
pythonw.exe (no terminal window) and opens the app in the browser.
The shortcut can be copied, renamed, or pinned to the taskbar / Start menu.
Put a review_app.ico next to this file before running to give it an icon.
"""

import subprocess
import sys
from pathlib import Path


def ps_quote(s) -> str:
    """Single-quoted PowerShell string."""
    return "'" + str(s).replace("'", "''") + "'"


def main() -> None:
    if sys.platform != "win32":
        sys.exit("make_shortcut.py only works on Windows.")

    here = Path(__file__).resolve().parent
    app = here / "review_app.py"
    if not app.is_file():
        sys.exit(f"review_app.py not found in {here}")

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        sys.exit(f"pythonw.exe not found next to {sys.executable}")

    # Fail here (with a terminal) rather than silently on double-click.
    for mod, required in (("openslide", True), ("flask", True), ("waitress", False)):
        try:
            __import__(mod)
        except Exception as e:
            if required:
                sys.exit(f"'{mod}' cannot be imported in this environment ({e}). "
                         "Run make_shortcut.py from the environment that runs the review app.")
            print(f"Note: '{mod}' is not installed (pip install {mod}); the app still works without it.")

    name = sys.argv[1] if len(sys.argv) > 1 else "Slide review"
    icon = here / "review_app.ico"
    script = "\n".join([
        "$desktop = [Environment]::GetFolderPath('Desktop')",
        f"$lnk = Join-Path $desktop {ps_quote(name + '.lnk')}",
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($lnk)",
        f"$s.TargetPath = {ps_quote(pythonw)}",
        f"$s.Arguments = {ps_quote(chr(34) + str(app) + chr(34))}",
        f"$s.WorkingDirectory = {ps_quote(here)}",
        "$s.Description = 'Pathologist slide review'",
        f"$s.IconLocation = {ps_quote(icon)}" if icon.is_file() else "",
        "$s.Save()",
        "Write-Output $lnk",
    ])
    r = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit("Could not create the shortcut:\n" + r.stderr)
    print("Created:", r.stdout.strip())
    print("Double-click it to start the review app.")


if __name__ == "__main__":
    main()