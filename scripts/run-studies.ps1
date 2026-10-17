#Requires -Version 7
<#
.SYNOPSIS
    Generates, trains and aggregates studies one after another; safe to interrupt and resume.

.DESCRIPTION
    For every study: `compute generate`, `compute train --skip-trained` and
    `main.py aggregate --plot-runs`. Output goes to the console and to
    logs/run-studies-<date>-<time>.log.

    Resuming: a run counts as trained once its artifacts/metadata.json holds its current
    config (metadata.json is written last), so after Ctrl+C or a crash only the runs that
    were in progress train again. Aggregation is skipped while results/summary.md is newer
    than study.toml, runs/index.csv and every metadata.json of the study.

    The system does not go to sleep while the script runs; closing the lid may still
    suspend it, depending on the power settings.

.PARAMETER Studies
    Study directories relative to the repository root; by default the studies of part one,
    the cheapest first.

.PARAMETER Jobs
    Runs trained in parallel (default: the number of logical processors).

.PARAMETER ForceAggregate
    Aggregate even when results/ is up to date (e.g. after a change of the plotting code).

.EXAMPLE
    pwsh -File scripts/run-studies.ps1

.EXAMPLE
    pwsh -File scripts/run-studies.ps1 -Jobs 6 data/experiments/regression/cube-losses
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments)]
    [string[]] $Studies,
    [int] $Jobs = [Environment]::ProcessorCount,
    [switch] $ForceAggregate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Exit codes of native commands are checked explicitly.
$PSNativeCommandUseErrorActionPreference = $false

$DefaultStudies = @(
    'data/experiments/classification/simple-losses'
    'data/experiments/classification/three-gauss-losses'
    'data/experiments/regression/activation-losses'
    'data/experiments/regression/cube-losses'
    'data/experiments/classification/simple-activations'
    'data/experiments/classification/three-gauss-activations'
    'data/experiments/regression/activation-activations'
    'data/experiments/regression/cube-activations'
    'data/experiments/classification/simple-architectures'
    'data/experiments/classification/three-gauss-architectures'
    'data/experiments/regression/activation-architectures'
    'data/experiments/regression/cube-architectures'
)
if (-not $Studies) {
    $Studies = $DefaultStudies
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Compute = Join-Path $Root 'src/compute/target/release' ($IsWindows ? 'compute.exe' : 'compute')
$LogDir = Join-Path $Root 'logs'
$Log = Join-Path $LogDir ('run-studies-{0:yyyyMMdd-HHmmss}.log' -f (Get-Date))
# Python buffers its output when piped; the progress should appear as it is printed.
$env:PYTHONUNBUFFERED = '1'

function Write-Log([string] $Text = '') {
    Write-Host $Text
    Add-Content -LiteralPath $Log -Value $Text
}

# Runs a native command with its stdout and stderr shown live and appended to the log;
# returns its exit code.
function Invoke-Logged([string] $File, [string[]] $Arguments) {
    Write-Log "> $(Split-Path $File -Leaf) $($Arguments -join ' ')"
    & $File @Arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $Log -Append | Out-Host
    return $LASTEXITCODE
}

function Format-Duration([TimeSpan] $Duration) {
    return '{0:hh\:mm\:ss}' -f $Duration
}

function Get-RunCount([string] $Study) {
    $index = Join-Path $Study 'runs/index.csv'
    return (Get-Content -LiteralPath $index | Measure-Object).Count - 1
}

# Whether results/summary.md is newer than everything aggregation reads.
function Test-ResultsCurrent([string] $Study) {
    $summary = Join-Path $Study 'results/summary.md'
    if (-not (Test-Path -LiteralPath $summary)) {
        return $false
    }
    $stamp = (Get-Item -LiteralPath $summary).LastWriteTimeUtc
    $inputs = @(
        Get-Item -LiteralPath (Join-Path $Study 'study.toml'), (Join-Path $Study 'runs/index.csv')
        Get-ChildItem -LiteralPath (Join-Path $Study 'runs') -Recurse -File -Filter 'metadata.json'
    )
    return -not ($inputs | Where-Object { $_.LastWriteTimeUtc -gt $stamp })
}

# Keeps the system awake while the script runs (the display may still turn off).
$ES_CONTINUOUS = [uint32] 2147483648
$ES_SYSTEM_REQUIRED = [uint32] 1
if ($IsWindows) {
    Add-Type -Namespace Win32 -Name Power -MemberDefinition @'
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint flags);
'@
}

$results = [System.Collections.Generic.List[object]]::new()
$current = $null
$finished = $false
$started = Get-Date
Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    if ($IsWindows) {
        [void] [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
    }
    Write-Log "run-studies: $($Studies.Count) studies, $Jobs jobs, log $Log"

    $current = 'build'
    if ((Invoke-Logged 'cargo' @('build', '--release', '--manifest-path', 'src/compute/Cargo.toml')) -ne 0) {
        throw 'cargo build failed'
    }
    if ((Invoke-Logged 'uv' @('sync', '--system-certs', '--project', 'src/analysis')) -ne 0) {
        throw 'uv sync failed'
    }

    for ($i = 0; $i -lt $Studies.Count; $i++) {
        $study = $Studies[$i].TrimEnd('/', '\')
        $current = $study
        $studyStarted = Get-Date
        Write-Log
        Write-Log ('[{0}/{1}] {2} ({3:HH:mm:ss})' -f ($i + 1), $Studies.Count, $study, $studyStarted)
        $status = 'done'
        $runs = $null
        $aggregated = '-'

        if (-not (Test-Path -LiteralPath (Join-Path $study 'study.toml'))) {
            Write-Log "error: $study/study.toml does not exist"
            $status = 'failed'
        }
        elseif ((Invoke-Logged $Compute @('generate', $study)) -ne 0) {
            $status = 'failed'
        }
        else {
            $runs = Get-RunCount $study
            Write-Log "$runs runs"
            if ((Invoke-Logged $Compute @('train', $study, '--jobs', $Jobs, '--skip-trained')) -ne 0) {
                $status = 'incomplete'
            }
            if (-not $ForceAggregate -and (Test-ResultsCurrent $study)) {
                Write-Log 'results/ up to date, aggregation skipped'
                $aggregated = 'skipped'
            }
            elseif ((Invoke-Logged 'uv' @('run', '--project', 'src/analysis', 'src/analysis/main.py', 'aggregate', $study, '--plot-runs')) -ne 0) {
                $status = 'failed'
                $aggregated = 'failed'
            }
            else {
                $aggregated = 'yes'
            }
        }

        $duration = (Get-Date) - $studyStarted
        Write-Log "${study}: $status in $(Format-Duration $duration)"
        $results.Add([pscustomobject] @{
                Study     = $study
                Status    = $status
                Runs      = $runs
                Aggregate = $aggregated
                Time      = Format-Duration $duration
            })
    }
    $current = $null
    $finished = $true
}
catch {
    Write-Log "error: $_"
    $finished = $true
    $results.Add([pscustomobject] @{ Study = $current; Status = 'failed' })
}
finally {
    if ($IsWindows) {
        [void] [Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS)
    }
    Pop-Location
    Write-Log
    if ($results.Count) {
        $table = $results | Format-Table -AutoSize | Out-String
        Write-Log $table.Trim()
    }
    Write-Log "total time $(Format-Duration ((Get-Date) - $started))"
    if (-not $finished) {
        Write-Log "interrupted during: $current; run the script again to resume"
    }
}

$failed = @($results | Where-Object Status -ne 'done').Count
exit ($failed -gt 0 ? 1 : 0)
