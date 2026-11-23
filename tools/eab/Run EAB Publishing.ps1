# The EAB publishing console operates on the local public website.
$ErrorActionPreference = 'Stop'
$entry = Join-Path $PSScriptRoot 'abl_eab.py'
if (!(Test-Path -LiteralPath $entry)) { $entry = 'C:\sbv_repo\abl_csv_repo\tools\eab\abl_eab.py' }
$runtime = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (!(Test-Path -LiteralPath $runtime)) { $runtime = 'python.exe' }
& $runtime -B $entry console
exit $LASTEXITCODE
