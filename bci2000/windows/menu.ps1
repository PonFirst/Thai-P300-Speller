<#
The Thai P300 speller session, step by step (Windows). Started by ThaiSpeller.cmd in the main folder.

Asks for the participant code once, then shows the session steps in order with what is already done:
look at the signal, check run, calibration (two parts), fit, the two spelling runs, reports. Each step
calls bci2000\run_session.ps1 or the Python tools with the right files, so nobody types a command or
Thai text. The sentences come from sentences.json next to this file.

Without the menu (scripts, the package build's smoke test):
  menu.ps1 -Subject S01 -Session 001 -Step check|cal1|cal2|fit|fixed|transformer|reports [-Fake] [-Auto] [-Reps 1]
-Fake rehearses with the fake headset; its files go to collection\recordings\bci2000_rehearsal.
This file stays ASCII: Windows PowerShell 5.1 misreads other characters in a script without a BOM.
#>
param(
    [string]$Subject = "",
    [string]$Session = "",
    [string]$Step = "",
    [switch]$Fake,
    [switch]$Auto,
    [int]$Reps = 0
)
$ErrorActionPreference = "Stop"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$RunSession = Join-Path $Repo "bci2000\run_session.ps1"
$Helper = Join-Path $Repo "bci2000\session_helper.py"
$PY = Join-Path $Repo "python\python.exe"
if (-not (Test-Path $PY)) {
    try { $PY = Join-Path ((& conda info --base) | Out-String).Trim() "envs\p300speller\python.exe" } catch { }
}
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Sentences = Get-Content -Raw -Encoding UTF8 (Join-Path $PSScriptRoot "sentences.json") | ConvertFrom-Json
$Interactive = -not $Step

function Pause-Menu { if ($Interactive) { [void](Read-Host "`nPress Enter to go back to the menu") } }

# --- who and what -----------------------------------------------------------------------------------
if ($Interactive) {
    Write-Host "`nThai P300 Speller`n"
    if (-not (Test-Path $PY)) { Write-Host "Python is missing: re-extract the ThaiSpeller zip."; Pause-Menu; exit 2 }
    $answer = Read-Host "Use the real headset? Type Y for yes, or R to rehearse without it [Y]"
    if ($answer -match '^[Rr]') { $Fake = $true }
}
while ($Subject -notmatch '^[A-Za-z0-9]+$') {
    if (-not $Interactive) { Write-Error "-Subject must be letters and digits only"; exit 2 }
    $Subject = Read-Host "Participant code (letters and digits only, e.g. S01; never a name)"
}
if (-not $Session -and $Interactive) { $Session = Read-Host "Session number [001]" }
if (-not $Session) { $Session = "001" }
if ($Session -notmatch '^[0-9]{1,3}$') { Write-Error "The session number must be up to 3 digits"; exit 2 }

$DataRoot = Join-Path $Repo ("collection\recordings\" + $(if ($Fake) { "bci2000_rehearsal" } else { "bci2000" }))
$Dir = Join-Path $DataRoot "$Subject$Session"
$Stem = if ($Fake) { "demo_$Subject" } else { $Subject }
$ModelPrm = Join-Path $Repo "models\${Stem}_bci2000.prm"
$Manifest = "models\${Stem}_manifest.json"
$Lmbin = "models\thai_gpt.lmbin"

function Get-Runs {
    if (-not (Test-Path $Dir)) { return @() }
    $lines = @(& $PY $Helper runs $Dir)
    return @($lines | Where-Object { $_ -match "`t" } | ForEach-Object {
        $kind, $path = $_ -split "`t", 2
        [pscustomobject]@{ Kind = $kind; Path = $path }
    })
}
function Latest([string]$kind) {
    $r = @(Get-Runs | Where-Object { $_.Kind -eq $kind })
    if ($r.Count) { return $r[-1].Path } else { return $null }
}

function Invoke-Run([string]$mode, [hashtable]$extra = @{}) {
    $common = @{ Subject = $Subject; Session = $Session; Data = $DataRoot }
    if ($Fake) { $common.Fake = $true }
    if ($Auto) { $common.Auto = $true }
    if ($Reps -gt 0 -and $mode -ne "check") { $common.Reps = $Reps }
    # Everything the run prints goes to the screen; a function would otherwise return it with the exit code
    & $RunSession $mode @common @extra | Out-Host
    return $LASTEXITCODE
}

function Invoke-Python([string[]]$arguments) {
    Push-Location $Repo
    try { & $PY @arguments | Out-Host; return $LASTEXITCODE } finally { Pop-Location }
}

# --- the steps --------------------------------------------------------------------------------------
function Step-View {
    Write-Host "`nThe Source window shows the live signal. Nothing is recorded. Close the Operator window when done."
    $extra = @{}
    if ($Fake) { $extra.Fake = $true }
    & $RunSession view @extra | Out-Host
    return 0
}

function Step-Check {
    Write-Host "`nCheck run: about 1.5 minutes. Follow the instructions in this window:"
    Write-Host "blink hard (20 s), then eyes closed (20 s), then clench the jaw (20 s)."
    $code = Invoke-Run "check"
    if ($code -ne 0) { return $code }
    $file = Latest "check"
    if ($file) {
        Write-Host ""
        Push-Location $Repo
        try { & $PY "tools\h13_montage_check.py" $file | Tee-Object -FilePath (Join-Path $Dir "h13_check.txt") | Out-Host }
        finally { Pop-Location }
        Write-Host "`nGo on only if all four lines say PASS. Otherwise fix the electrodes and run the check again."
    }
    return 0
}

function Step-Calibration([int]$part) {
    $text = $Sentences.calibration[$part - 1]
    Write-Host "`nCalibration, part $part. Press Start in the Operator window."
    Write-Host "Look at the highlighted letter and silently count its flashes. Blink only between letters."
    return (Invoke-Run "calibrate" @{ Text = $text })
}

function Step-Fit {
    $runs = @(Get-Runs | Where-Object { $_.Kind -eq "calibration" } | ForEach-Object { $_.Path })
    if (-not $runs.Count) { Write-Host "`nNo calibration run yet: do the calibration first."; return 2 }
    $arguments = @("-m", "pipeline.fit_from_recording", "--id", $Subject, "--recording") + $runs +
                 @("--manifest", $Manifest, "--lm-transformer-checkpoint", $Lmbin)
    if ($Fake) { $arguments += "--allow-demo" }
    if (Test-Path $ModelPrm) {
        if ($Interactive) {
            $answer = Read-Host "A model for $Subject already exists. Replace it? (y/N)"
            if ($answer -notmatch '^[Yy]') { return 0 }
        }
        $arguments += "--force"
    }
    Write-Host "`nFitting on $($runs.Count) calibration run(s)..."
    $code = Invoke-Python $arguments
    if ($code -eq 0) { Write-Host "`nLook at the out-of-fold AUC above: about 0.70 or more is good." }
    return $code
}

function Step-Online([string]$condition) {
    if (-not (Test-Path $ModelPrm)) { Write-Host "`nNo model yet: fit the classifier first."; return 2 }
    if ($condition -eq "fixed") { Write-Host "`nSpelling with all 120 flashes per letter. Press Start in the Operator window." }
    else { Write-Host "`nSpelling with dynamic stopping and the Transformer. Press Start in the Operator window." }
    return (Invoke-Run "online" @{ Model = $ModelPrm; Condition = $condition; Text = $Sentences.online })
}

function Step-Reports {
    $code = 0
    $opened = @()
    foreach ($kind in @("fixed", "transformer", "uniform")) {
        $file = Latest $kind
        if (-not $file) { continue }
        $html = Join-Path $Dir "report_$kind.html"
        Write-Host "`n--- $kind run ---"
        $c = Invoke-Python @("-m", "stopping.summarize_experiment", "--recording", $file, "--html", $html,
                             "--json", (Join-Path $Dir "report_$kind.json"))
        if ($c -ne 0) { $code = $c } elseif (Test-Path $html) { $opened += $html }
        $theta = if ($kind -eq "fixed") { @("0.5", "0.7", "0.8", "0.85", "0.9", "0.95") } else { @("0.5", "0.7", "0.8", "0.85", "0.9") }
        $replay = @("-m", "stopping.evaluate_recording", "--manifest", $Manifest, "--recording", $file, "--theta") + $theta +
                  @("--out", (Join-Path $Dir "replay_$kind.csv"))
        if ($kind -eq "fixed") { $replay += @("--lam", "0", "0.7") }
        if ($Fake) { $replay += "--allow-demo" }
        $c = Invoke-Python $replay
        if ($c -ne 0) { $code = $c }
    }
    if (-not $opened.Count) { Write-Host "`nNo spelling run yet."; return 2 }
    Write-Host "`nReports and tables are in $Dir"
    if ($Interactive) { foreach ($h in $opened) { Start-Process $h } }
    return $code
}

function Invoke-Step([string]$name) {
    switch ($name) {
        "view" { return Step-View }
        "check" { return Step-Check }
        "cal1" { return Step-Calibration 1 }
        "cal2" { return Step-Calibration 2 }
        "fit" { return Step-Fit }
        "fixed" { return Step-Online "fixed" }
        "transformer" { return Step-Online "transformer" }
        "reports" { return Step-Reports }
        default { Write-Host "Unknown step $name"; return 2 }
    }
}

if (-not $Interactive) {
    $result = @(Invoke-Step $Step)
    exit [int]$result[-1]
}

# --- the menu ---------------------------------------------------------------------------------------
$items = @(
    @{ Key = "1"; Step = "view"; Label = "Look at the signal (nothing is recorded)" },
    @{ Key = "2"; Step = "check"; Label = "Check run (1.5 min)" },
    @{ Key = "3"; Step = "cal1"; Label = "Calibration, part 1 (6 min)" },
    @{ Key = "4"; Step = "cal2"; Label = "Calibration, part 2 (7 min; rest a few minutes first)" },
    @{ Key = "5"; Step = "fit"; Label = "Fit the classifier (1-2 min)" },
    @{ Key = "6"; Step = "fixed"; Label = "Spelling, all flashes (8.5 min)" },
    @{ Key = "7"; Step = "transformer"; Label = "Spelling, dynamic stopping + Transformer (5-8.5 min)" },
    @{ Key = "8"; Step = "reports"; Label = "Reports" }
)
while ($true) {
    $runs = @(Get-Runs)
    $count = { param($k) @($runs | Where-Object { $_.Kind -eq $k }).Count }
    $done = @{
        check = (& $count "check") -gt 0
        cal1 = (& $count "calibration") -ge 1
        cal2 = (& $count "calibration") -ge 2
        fit = Test-Path $ModelPrm
        fixed = (& $count "fixed") -gt 0
        transformer = (& $count "transformer") -gt 0
        reports = Test-Path (Join-Path $Dir "report_fixed.html")
    }
    Write-Host ""
    Write-Host "============================================================"
    $mode = if ($Fake) { "REHEARSAL (fake headset, not real data)" } else { "real headset" }
    Write-Host " Participant $Subject, session $Session - $mode"
    Write-Host "============================================================"
    foreach ($i in $items) {
        $mark = if ($done[$i.Step]) { "done" } else { "" }
        Write-Host ("  {0}  {1,-58} {2}" -f $i.Key, $i.Label, $mark)
    }
    Write-Host "  0  Quit"
    $choice = Read-Host "`nType a step number and press Enter"
    if ($choice -eq "0") { break }
    $item = $items | Where-Object { $_.Key -eq $choice } | Select-Object -First 1
    if (-not $item) { continue }
    [void](Invoke-Step $item.Step)
    Pause-Menu
}
