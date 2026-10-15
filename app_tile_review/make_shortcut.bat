@echo off
REM Creates the "Slide review" icon in E:\Christine that starts the review app.
REM Keep this file next to start_review.bat and slide_review.ico, and double-click it once.

set "TARGET_DIR=E:\Christine"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$l=Join-Path '%TARGET_DIR%' 'Slide review.lnk';" ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($l);" ^
  "$s.TargetPath='%~dp0start_review.bat';" ^
  "$s.WorkingDirectory='%~dp0';" ^
  "$s.IconLocation='%~dp0slide_review.ico,0';" ^
  "$s.Description='Open the pathologist slide review';" ^
  "$s.Save();" ^
  "Write-Host ('Shortcut created: ' + $l)"
pause
