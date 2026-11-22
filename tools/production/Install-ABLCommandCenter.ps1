<# Adds two operations to the existing Production Control menu through script discovery. #>
[CmdletBinding(SupportsShouldProcess)]
param([string]$MenuDirectory = (Join-Path $env:USERPROFILE 'Scripts'))

$ErrorActionPreference = 'Stop'
$menu = Join-Path $MenuDirectory 'ABL-Production-Control.ps1'
if (!(Test-Path -LiteralPath $menu -PathType Leaf)) { throw "Production Control menu missing: $menu" }
$launchers = @('Open ABL Command Center.ps1', 'Run ABL Production Session.ps1')
foreach ($launcher in $launchers) {
    $source = Join-Path $PSScriptRoot $launcher
    $destination = Join-Path $MenuDirectory $launcher
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $destination).Hash) {
            throw "Existing operation differs. Review before replacing: $destination"
        }
    }
}
foreach ($launcher in $launchers) {
    $source = Join-Path $PSScriptRoot $launcher
    $destination = Join-Path $MenuDirectory $launcher
    if (!(Test-Path -LiteralPath $destination) -and $PSCmdlet.ShouldProcess($destination, 'Add Production Control operation')) {
        [IO.File]::Copy($source, $destination, $false)
    }
    if ((Test-Path -LiteralPath $destination) -and
        (Get-FileHash -LiteralPath $source).Hash -eq (Get-FileHash -LiteralPath $destination).Hash) {
        Write-Output "Production Control operation verified: $destination"
    }
}
