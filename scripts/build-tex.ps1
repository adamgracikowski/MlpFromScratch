<#
.SYNOPSIS
    Compile a LaTeX document to PDF and clean up build artifacts.

.DESCRIPTION
    PowerShell counterpart of scripts/build-tex.sh. Unlike the bash version it
    does not use latexmk (MiKTeX's latexmk needs Perl, which is usually not on
    PATH outside Git Bash). Instead it runs the build steps itself:

    1. Changes into the directory of the .tex file, so relative paths to the
       bibliography (.bib) and images (\includegraphics) resolve as usual.
    2. Runs pdflatex with -output-directory=.latexbuild.
    3. Runs biber (biblatex documents) or bibtex (\bibliography documents)
       if the document has a bibliography.
    4. Re-runs pdflatex until LaTeX stops asking for a rerun (at most
       MaxRuns passes in total), so cross-references and citations are stable.
    5. Copies the PDF next to the .tex file under the requested name and
       removes .latexbuild/, so only the sources and the PDF remain.

    On failure the script prints the LaTeX errors, saves the full log as
    <output>.log next to the .tex file and exits with code 1. Undefined
    citations/references and biber warnings are reported but do not fail
    the build.

    Requirements: pdflatex and biber/bibtex on PATH (e.g. MiKTeX or TeX Live).
    Works in Windows PowerShell 5.1 and PowerShell 7+. If script execution is
    blocked, run it with:
        powershell -ExecutionPolicy Bypass -File scripts\build-tex.ps1 ...

.PARAMETER TexFile
    Source document (relative or absolute path to a .tex file).

.PARAMETER OutputName
    Name of the resulting PDF, written next to the .tex file. Defaults to the
    name of the .tex file with a .pdf extension; the .pdf extension is added
    if missing.

.PARAMETER Keep
    Keep the .latexbuild/ directory (useful for debugging).

.PARAMETER MaxRuns
    Maximum number of pdflatex passes (default: 5).

.EXAMPLE
    .\scripts\build-tex.ps1 docs\report.tex

.EXAMPLE
    .\scripts\build-tex.ps1 docs\report.tex report.pdf

.EXAMPLE
    .\scripts\build-tex.ps1 docs\report.tex -Keep
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$TexFile,

    [Parameter(Position = 1)]
    [string]$OutputName,

    [switch]$Keep,

    [ValidateRange(2, 10)]
    [int]$MaxRuns = 5
)

$ErrorActionPreference = 'Stop'
$BuildDir = '.latexbuild'
# Messages after which another pdflatex pass can change the output. Biber's
# "Please (re)run Biber" is not listed: biber already ran once, and the message
# stays when a citation key is missing, so rerunning would not help.
$RerunPattern = 'Rerun to get|Please rerun LaTeX|Label\(s\) may have changed|rerunfilecheck Warning'

function Fail([string]$Message) {
    [Console]::Error.WriteLine($Message)
    exit 1
}

# Runs a native tool, returns its exit code and keeps its output for error reports.
function Invoke-Tool([string]$Exe, [string[]]$Arguments) {
    $previous = $ErrorActionPreference
    # Native tools write progress to stderr; do not let that stop the script.
    $ErrorActionPreference = 'Continue'
    $script:LastToolOutput = & $Exe @Arguments 2>&1
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    return $code
}

foreach ($tool in 'pdflatex', 'biber', 'bibtex') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Fail "Error: $tool not found on PATH (install MiKTeX or TeX Live)."
    }
}
if (-not (Test-Path -LiteralPath $TexFile -PathType Leaf)) {
    Fail "Error: file '$TexFile' not found."
}
if ([IO.Path]::GetExtension($TexFile) -ne '.tex') {
    Fail "Error: '$TexFile' does not have a .tex extension."
}

$texItem = Get-Item -LiteralPath $TexFile
$texDir = $texItem.DirectoryName
$texName = $texItem.Name
$job = [IO.Path]::GetFileNameWithoutExtension($texName)

if (-not $OutputName) { $OutputName = "$job.pdf" }
if ([IO.Path]::GetExtension($OutputName) -ne '.pdf') { $OutputName = "$OutputName.pdf" }
if ($OutputName -match '[\\/]') {
    Fail "Error: give a bare PDF file name (no directory); it is written next to '$texName'."
}
$logCopy = [IO.Path]::GetFileNameWithoutExtension($OutputName) + '.log'

Push-Location -LiteralPath $texDir
try {
    if (Test-Path -LiteralPath $BuildDir) { Remove-Item -LiteralPath $BuildDir -Recurse -Force }
    New-Item -ItemType Directory -Path $BuildDir | Out-Null

    $log = Join-Path $BuildDir "$job.log"
    $pdf = Join-Path $BuildDir "$job.pdf"
    $bcf = Join-Path $BuildDir "$job.bcf"
    $aux = Join-Path $BuildDir "$job.aux"
    $latexArgs = @('-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
        "-output-directory=$BuildDir", $texName)

    # Prints the failure report and stops the script.
    function Stop-Build([string]$Step, [int]$Code) {
        [Console]::Error.WriteLine("Compilation failed at step '$Step' (exit code $Code).")
        $stepLog = if ($Step -eq 'pdflatex') { $log } else { Join-Path $BuildDir "$job.blg" }
        if (Test-Path -LiteralPath $stepLog) {
            if ($Step -eq 'pdflatex') {
                # Errors in file:line:error format and classic '!' lines.
                Select-String -LiteralPath $stepLog -CaseSensitive -Pattern '^(!|.*:\d+: )' -Context 0, 3 |
                Select-Object -First 15 |
                ForEach-Object {
                    [Console]::Error.WriteLine($_.Line)
                    $_.Context.PostContext | Where-Object { $_.Trim() } |
                    ForEach-Object { [Console]::Error.WriteLine("    $_") }
                }
            }
            else {
                Select-String -LiteralPath $stepLog -CaseSensitive -Pattern 'ERROR|error' |
                Select-Object -First 15 |
                ForEach-Object { [Console]::Error.WriteLine($_.Line) }
            }
            Copy-Item -LiteralPath $stepLog -Destination $logCopy -Force
            [Console]::Error.WriteLine("Full log: $(Join-Path $texDir $logCopy)")
        }
        else {
            $script:LastToolOutput | Select-Object -Last 20 |
            ForEach-Object { [Console]::Error.WriteLine($_.ToString()) }
        }
        Pop-Location
        if (-not $Keep) { Remove-Item -LiteralPath (Join-Path $texDir $BuildDir) -Recurse -Force -ErrorAction SilentlyContinue }
        exit 1
    }

    Write-Host "Compiling: $(Join-Path $texDir $texName)"

    $code = Invoke-Tool 'pdflatex' $latexArgs
    if ($code -ne 0) { Stop-Build 'pdflatex' $code }
    $runs = 1

    # Bibliography: biblatex writes a .bcf file, classic BibTeX leaves \bibdata in the .aux file.
    if (Test-Path -LiteralPath $bcf) {
        $code = Invoke-Tool 'biber' @("--input-directory=$BuildDir", "--output-directory=$BuildDir", $job)
        if ($code -ne 0) { Stop-Build 'biber' $code }
        $needRerun = $true
    }
    elseif ((Test-Path -LiteralPath $aux) -and (Select-String -LiteralPath $aux -CaseSensitive -Pattern '\\bibdata' -Quiet)) {
        # bibtex runs on the .aux file in the build directory but must find .bib files next to the .tex file.
        $previousBibInputs = $env:BIBINPUTS
        $env:BIBINPUTS = "$texDir;$previousBibInputs"
        $code = Invoke-Tool 'bibtex' @((Join-Path $BuildDir $job))
        $env:BIBINPUTS = $previousBibInputs
        if ($code -ne 0) { Stop-Build 'bibtex' $code }
        $needRerun = $true
    }
    else {
        $needRerun = Select-String -LiteralPath $log -CaseSensitive -Pattern $RerunPattern -Quiet
    }

    while ($needRerun -and $runs -lt $MaxRuns) {
        $code = Invoke-Tool 'pdflatex' $latexArgs
        if ($code -ne 0) { Stop-Build 'pdflatex' $code }
        $runs++
        $needRerun = Select-String -LiteralPath $log -CaseSensitive -Pattern $RerunPattern -Quiet
    }
    if ($needRerun) {
        Write-Warning "LaTeX still asks for a rerun after $MaxRuns passes; references may be off."
    }
    if (-not (Test-Path -LiteralPath $pdf)) {
        Stop-Build 'pdflatex' 0
    }

    # Warnings that do not stop the build but leave the document broken.
    $undefined = Select-String -LiteralPath $log -CaseSensitive -Pattern 'Citation .* undefined|Reference .* undefined'
    if ($undefined) {
        Write-Warning 'Undefined citations or references:'
        $undefined | ForEach-Object { $_.Line } | Sort-Object -Unique | Select-Object -First 20 |
        ForEach-Object { Write-Warning $_ }
    }
    $blg = Join-Path $BuildDir "$job.blg"
    if (Test-Path -LiteralPath $blg) {
        $bibMessages = Select-String -LiteralPath $blg -CaseSensitive -Pattern '^(ERROR|WARN)|^Warning--'
        if ($bibMessages) {
            Write-Warning 'Bibliography tool messages:'
            $bibMessages | Select-Object -First 20 | ForEach-Object { Write-Warning $_.Line }
        }
    }

    Copy-Item -LiteralPath $pdf -Destination $OutputName -Force
    if (Test-Path -LiteralPath $logCopy) { Remove-Item -LiteralPath $logCopy -Force }
    if (-not $Keep) { Remove-Item -LiteralPath $BuildDir -Recurse -Force }

    Write-Host "Done: $(Join-Path $texDir $OutputName) ($runs pdflatex passes)"
}
finally {
    # Stop-Build already popped the location before exiting.
    if ((Get-Location).Path -eq $texDir) { Pop-Location }
}
