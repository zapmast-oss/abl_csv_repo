# Discovered automatically by the existing ABL Production Control menu.
$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot 'abl_production.py'
if (!(Test-Path -LiteralPath $entry)) { $entry = 'C:\sbv_repo\abl_csv_repo\tools\production\abl_production.py' }
& python.exe -B $entry open
exit $LASTEXITCODE
