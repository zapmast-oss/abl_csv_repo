# Install the existing menu's additional operation and a limited per-user poll.
[CmdletBinding()]
param([ValidateSet('Menu','Task','All')][string]$Component = 'All')
$ErrorActionPreference = 'Stop'
$repo = 'C:\sbv_repo\abl_csv_repo'
$control = Join-Path $repo 'scripts\abl_newsroom_control.py'
$pythonw = 'C:\Users\earld\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe'
$menu = 'C:\Users\earld\Scripts'
$taskName = 'ABL Daily Newsroom'
if (!(Test-Path -LiteralPath $control) -or !(Test-Path -LiteralPath $pythonw)) { throw 'Newsroom control or runtime missing.' }
if ($Component -in 'All','Menu') {
    if (!(Test-Path -LiteralPath (Join-Path $menu 'ABL-Production-Control.ps1'))) { throw 'Existing Production Control menu missing.' }
    $src = Join-Path $PSScriptRoot 'Run ABL Newsroom Now.ps1'
    $dst = Join-Path $menu 'Run ABL Newsroom Now.ps1'
    if (!(Test-Path -LiteralPath $dst)) { [IO.File]::Copy($src,$dst,$false) }
    if ((Get-FileHash -LiteralPath $src).Hash -ne (Get-FileHash -LiteralPath $dst).Hash) { throw "Conflicting existing menu operation: $dst" }
    Write-Output "Production Control operation verified: $dst"
}
if ($Component -in 'All','Task') {
    $arguments = '-B "' + $control + '" poll'
    $service = New-Object -ComObject 'Schedule.Service'
    $service.Connect()
    $folder = $service.GetFolder('\')
    $existing = @($folder.GetTasks(0) | Where-Object Name -eq $taskName)
    if ($existing.Count -eq 0) {
        $definition = $service.NewTask(0)
        $definition.RegistrationInfo.Description = 'Check the live ABL CSV export every five minutes. Verify and archive internal coverage for changed exports.'
        $definition.Principal.UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
        $definition.Principal.LogonType = 3 # TASK_LOGON_INTERACTIVE_TOKEN
        $definition.Principal.RunLevel = 0 # Least privilege
        $definition.Settings.StartWhenAvailable = $true
        $definition.Settings.DisallowStartIfOnBatteries = $false
        $definition.Settings.StopIfGoingOnBatteries = $false
        $definition.Settings.MultipleInstances = 2 # Ignore if still running
        $definition.Settings.ExecutionTimeLimit = 'PT2H'
        $trigger = $definition.Triggers.Create(2) # Daily, repeat for the day
        $trigger.StartBoundary = (Get-Date).AddMinutes(1).ToString('yyyy-MM-ddTHH:mm:ss')
        $trigger.DaysInterval = 1
        $trigger.Repetition.Interval = 'PT5M'
        $trigger.Repetition.Duration = 'P1D'
        $action = $definition.Actions.Create(0)
        $action.Path = $pythonw
        $action.Arguments = $arguments
        $action.WorkingDirectory = $repo
        [void]$folder.RegisterTaskDefinition($taskName,$definition,2,$null,$null,3)
    }
    $task = $folder.GetTask($taskName)
    $action = $task.Definition.Actions.Item(1)
    $trigger = $task.Definition.Triggers.Item(1)
    if (!$task.Enabled -or $action.Path -ne $pythonw -or $action.Arguments -ne $arguments -or
        $action.WorkingDirectory -ne $repo -or $task.Definition.Actions.Count -ne 1 -or
        $trigger.Repetition.Interval -ne 'PT5M' -or $task.Definition.Principal.RunLevel -ne 0) {
        throw "Existing $taskName registration differs from the newsroom polling operation."
    }
    Write-Output "Scheduled operation verified: $taskName. Next run $($task.NextRunTime)."
}
