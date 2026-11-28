# Menu entry discovered automatically by ABL-Production-Control.ps1.
$ErrorActionPreference = 'Stop'
$python = 'C:\Users\earld\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$control = 'C:\sbv_repo\abl_csv_repo\scripts\abl_newsroom_control.py'
& $python -B $control run
exit $LASTEXITCODE
