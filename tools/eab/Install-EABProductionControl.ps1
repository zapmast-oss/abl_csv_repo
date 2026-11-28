[CmdletBinding(SupportsShouldProcess)]
param([string]$MenuDirectory = (Join-Path $env:USERPROFILE 'Scripts'))

$ErrorActionPreference = 'Stop'
$menu = Join-Path $MenuDirectory 'ABL-Production-Control.ps1'
if (!(Test-Path -LiteralPath $menu -PathType Leaf)) { throw "Production Control menu missing: $menu" }
$launchers = @('Open EAB Website.ps1', 'Run EAB Publishing.ps1')
foreach ($name in $launchers) {
    $source = Join-Path $PSScriptRoot $name
    $destination = Join-Path $MenuDirectory $name
    if ((Test-Path -LiteralPath $destination) -and
        (Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $destination).Hash) {
        throw "Existing operation differs. Review before replacing: $destination"
    }
}
foreach ($name in $launchers) {
    $source = Join-Path $PSScriptRoot $name
    $destination = Join-Path $MenuDirectory $name
    if (!(Test-Path -LiteralPath $destination) -and $PSCmdlet.ShouldProcess($destination, 'Add EAB menu operation')) {
        [IO.File]::Copy($source, $destination, $false)
    }
    if (Test-Path -LiteralPath $destination) { Write-Output "EAB operation verified: $destination" }
}
