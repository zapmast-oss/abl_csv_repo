# Adjacent launchers are discovered by ABL Production Control.
$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot 'abl_eab.py'
if (!(Test-Path -LiteralPath $entry)) { $entry = 'C:\sbv_repo\abl_csv_repo\tools\eab\abl_eab.py' }
& python.exe -B $entry open
exit $LASTEXITCODE
