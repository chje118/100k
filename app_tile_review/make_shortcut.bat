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


def finish(msg: str, ok: bool = True) -> None:
    """Print the result and keep the window open (needed when double-clicked)."""
    print()
    print(msg)
    print()
    print(f"Python used: {sys.executable}")
    try:
        input("Press Enter to close this window...")
    except EOFError:
        pass
    sys.exit(0 if ok else 1)


def ps_quote(s) -> str:
    """Single-quoted PowerShell string."""
    return "'" + str(s).replace("'", "''") + "'"


def main() -> None:
    if sys.platform != "win32":
        finish("make_shortcut.py only works on Windows.", ok=False)

    here = Path(__file__).resolve().parent
    app = here / "review_app.py"
    if not app.is_file():
        finish(f"review_app.py not found in {here}", ok=False)

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.is_file():
        finish(f"pythonw.exe not found next to {sys.executable}", ok=False)

    # Fail here (with a terminal) rather than silently on double-click.
    for mod, required in (("openslide", True), ("flask", True), ("waitress", False)):
        try:
            __import__(mod)
        except Exception as e:
            if required:
                finish(f"'{mod}' cannot be imported with this Python ({e}).\n"
                       "Windows opened make_shortcut.py with the wrong Python. Instead open the\n"
                       "Anaconda Prompt, run  conda activate <your env>  and then\n"
                       "python make_shortcut.py  in this folder.", ok=False)
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
        finish("Could not create the shortcut:\n" + r.stderr, ok=False)
    finish(f"Created: {r.stdout.strip()}\nDouble-click it to start the review app.")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # never let the window vanish without a message
        finish(f"Something went wrong: {e!r}", ok=False)