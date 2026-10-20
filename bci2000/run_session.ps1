<#
Run one BCI2000 session run of the Thai checkerboard speller on Windows, configured the same way every
time. The Windows counterpart of bci2000/run_session.sh (macOS): same modes, checks, logs and output.

  .\bci2000\run_session.ps1 view                                 # live signal, nothing recorded
  .\bci2000\run_session.ps1 check     -Subject S01               # 60 s: blink / eyes closed / jaw (H3 + H13)
  .\bci2000\run_session.ps1 calibrate -Subject S01 -Text "<calibration sentence>" [-Reps 5]
  .\bci2000\run_session.ps1 online    -Subject S01 -Model models\S01_bci2000.prm -Text "<sentence>"
                                      [-Condition transformer|uniform|fixed] [-Reps 5]
Most people use bci2000\windows\menu.ps1 (ThaiSpeller.cmd) instead, which runs these in order.

-Port: the headset's Bluetooth COM port; "auto" (the default) finds the paired headset.
Common options: -Session 001 (BCI2000 SubjectSession), -Data DIR (default collection\recordings\bci2000),
  -Window 1280x800, -Fake (rehearse with tools/fake_unicorn_bt.py instead of -Port; the data marks itself
  synthetic), -Auto (start the run by itself 2 s after configuring instead of waiting for Start).

Modules: UnicornSerialSource always; calibrate and check use DummySignalProcessing (no classifier), online
uses CheckerboardSignalProcessing with the fitted model. Conditions (online):
  transformer  Bayesian stopping + the Transformer prior (the model .prm's own settings; the default)
  uniform      Bayesian stopping + uniform prior (the no-LM control)
  fixed        no stopping: every flash of NumberOfSequences repetitions, the best mean score

The Operator window opens configured: check the signal in the Source window, then press Start. The script
ends when that run ends; run it again for the next run (BCI2000 numbers runs R01, R02, ...).
If PowerShell refuses to run scripts: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
param(
    [Parameter(Position = 0)][string]$Mode,
    [string]$Subject = "",
    [string]$Session = "001",
    [string]$Text = "",
    [string]$Model = "",
    [string]$Condition = "transformer",
    [int]$Reps = 0,
    [switch]$Free,
    [switch]$Fake,
    [switch]$Auto,
    [string]$Data = "",
    [string]$Window = "1280x800",
    [string]$Port = "auto"
)
$ErrorActionPreference = "Stop"

function Die([string]$message) {
    [Console]::Error.WriteLine("run_session: $message")
    exit 2
}
function Usage {
    Get-Content $PSCommandPath | Select-Object -Skip 1 -First 24 | ForEach-Object { Write-Host $_ }
    exit 2
}
# BCI2000 script lines take forward slashes; Windows accepts them everywhere.
function Slash([string]$path) { return $path.Replace('\', '/') }

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
# The Windows package brings its own Python and BCI2000 next to the code; a developer setup uses the
# p300speller conda environment and a BCI2000 source build.
$PY = Join-Path $Repo "python\python.exe"
if (-not (Test-Path $PY)) {
    try { $PY = Join-Path ((& conda info --base) | Out-String).Trim() "envs\p300speller\python.exe" } catch { }
}
$Prog = Join-Path $Repo "bci2000\prog"
if (-not (Test-Path (Join-Path $Prog "BCI2000Shell.exe"))) {
    $Prog = if ($env:BCI2000_ROOT) { Join-Path $env:BCI2000_ROOT "prog" } else { Join-Path $HOME "BCI2000\trunk\prog" }
}
$SpellerPrm = Join-Path $Repo "bci2000\parms\ThaiCheckerboard.prm"
$FakePort = 9000
if (-not $Data) { $Data = Join-Path $Repo "collection\recordings\bci2000" }
# Thai text in and out of Python, whatever the console code page
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if ($Mode -notin @("calibrate", "online", "check", "view")) { Usage }
if ($Mode -eq "view" -and -not $Subject) { $Subject = "VIEW" }
if ($Subject -notmatch '^[A-Za-z0-9]+$') { Die "-Subject must be letters and digits only (BCI2000 builds file names from it)" }
if ($Session -notmatch '^[0-9]{1,3}$') { Die "-Session must be a number of up to 3 digits" }
if ($Window -notmatch '^[0-9]+x[0-9]+$') { Die "-Window must look like 1280x800" }
if ($Reps -lt 0) { Die "-Reps must be a positive integer" }
if ($Condition -notin @("transformer", "uniform", "fixed")) { Die "-Condition must be transformer, uniform or fixed" }
if (-not $PY -or -not (Test-Path $PY)) { Die "no Python found: use the Windows package, or create the p300speller conda environment" }

foreach ($m in @("BCI2000Shell.exe", "UnicornSerialSource.exe", "DummySignalProcessing.exe",
                 "CheckerboardSignalProcessing.exe", "CheckerboardSpeller.exe", "DummyApplication.exe")) {
    if (-not (Test-Path (Join-Path $Prog $m))) { Die "missing ${Prog}\${m}: build first (bci2000\WINDOWS.md)" }
}
$running = Get-Process -Name Operator, UnicornSerialSource, DummySignalProcessing, CheckerboardSignalProcessing,
    CheckerboardSpeller, DummyApplication -ErrorAction SilentlyContinue
if ($running) { Die "BCI2000 is already running; close it first" }
& $PY (Join-Path $Repo "tools\make_bci2000_parms.py") --check | Out-Null
if ($LASTEXITCODE -ne 0) { Die "bci2000\parms\ThaiCheckerboard.prm is out of date: run tools\make_bci2000_parms.py" }

$Helper = Join-Path $Repo "bci2000\session_helper.py"
function Encode([string]$t) {
    # BCI2000 parameter encoding of a text (ASCII %XX), refusing characters that are not in the matrix
    $out = & $PY $Helper encode $t
    if ($LASTEXITCODE -ne 0) { exit 2 }
    return ($out | Out-String).Trim()
}

$SP = "DummySignalProcessing"; $App = "CheckerboardSpeller"; $Settings = @()
switch ($Mode) {
    "calibrate" {
        if (-not $Text) { Die "calibrate needs -Text (the calibration sentence)" }
        $encoded = Encode $Text
        $n = if ($Reps -gt 0) { $Reps } else { 5 }
        $Settings = @("Set parameter InterpretMode 2", "Set parameter TextToSpell $encoded",
                      "Set parameter NumberOfSequences $n")
    }
    "online" {
        if (-not $Model) { Die "online needs -Model models\<id>_bci2000.prm (written by fit_from_recording)" }
        if (-not (Test-Path $Model -PathType Leaf)) { Die "no model at $Model" }
        $Model = (Resolve-Path $Model).Path
        $SP = "CheckerboardSignalProcessing"
        if ($Free) {
            if ($Text) { Die "-Free and -Text exclude each other" }
            $modeLines = @("Set parameter InterpretMode 1")
        } else {
            if (-not $Text) { Die "online needs -Text (copy spelling, scored) or -Free" }
            $encoded = Encode $Text
            $modeLines = @("Set parameter InterpretMode 2", "Set parameter TextToSpell $encoded")
        }
        $condLines = @()
        switch ($Condition) {
            "transformer" {
                if (-not (Select-String -Path $Model -SimpleMatch "LanguageModel= 1" -Quiet)) {
                    Die "$Model has no Transformer prior: refit with --lm-transformer-checkpoint, or use -Condition uniform"
                }
            }
            "uniform" { $condLines = @("Set parameter LanguageModel 0") }
            "fixed" { $condLines = @("Set parameter StoppingRule 0") }
        }
        # 5 repetitions (120 flashes) in every condition: dynamic stopping's t_max is 120 too, so it never
        # gets more flashes than the fixed baseline.
        $n = if ($Reps -gt 0) { $Reps } else { 5 }
        $Settings = @("Load parameterfile `"$(Slash $Model)`"") + $modeLines + $condLines +
                    @("Set parameter NumberOfSequences $n")
    }
    "check" { $App = "DummyApplication" }
    "view" { $App = "DummyApplication" }
}

if ($Fake) {
    $SourceArgs = "--TcpAddress=127.0.0.1:$FakePort"
    Write-Host "*** -Fake: SYNTHETIC EEG from tools/fake_unicorn_bt.py. This is a rehearsal, not data. ***"
} else {
    if (-not $Port -or $Port -eq "auto") {
        $found = @(& $PY $Helper find-headset)
        if ($LASTEXITCODE -ne 0 -or -not $found) { exit 2 }   # the helper said why
        $Port = ($found[-1] | Out-String).Trim()
        Write-Host "Headset port: $Port"
    }
    if ($Port -notmatch '^COM[0-9]+$') {
        Die ("-Port must be the headset's COM port, e.g. -Port COM5: Windows Settings > Bluetooth & devices > " +
             "More Bluetooth settings > COM Ports, the `"Outgoing`" port of UN-...")
    }
    # The COM ports Windows has now (the registry: System.IO.Ports is missing from PowerShell 7)
    $ports = @()
    $comm = Get-ItemProperty -Path "HKLM:\HARDWARE\DEVICEMAP\SERIALCOMM" -ErrorAction SilentlyContinue
    if ($comm) { $ports = @($comm.PSObject.Properties | Where-Object { $_.Name -notlike "PS*" } | ForEach-Object { $_.Value }) }
    if ($Port -notin $ports) { Die "$Port does not exist (COM ports now: $($ports -join ', ')): pair the headset and switch it on" }
    $SourceArgs = "--SerialPort=$Port"
}

# Work folder for the generated script and the system log. BCI2000's command line can't take a quoted
# path, so use the 8.3 short form when the temp path has spaces (e.g. a user name with a space).
$Work = Join-Path ([System.IO.Path]::GetTempPath()) ("bci2000_session_" + [System.Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $Work | Out-Null
if ($Work -match ' ') {
    $Work = (New-Object -ComObject Scripting.FileSystemObject).GetFolder($Work).ShortPath
    if ($Work -match ' ') { Die "the temp folder path has spaces and no short form; set TEMP to a folder without spaces" }
}
$SystemLog = Join-Path $Work "system.log"
# Looking only: BCI2000 still reserves a file name when it configures, so keep that out of the recordings
if ($Mode -eq "view") { $Data = $Work }

$FakeProc = $null
if ($Fake) {
    if (Get-NetTCPConnection -LocalPort $FakePort -State Listen -ErrorAction SilentlyContinue) {
        Die "port $FakePort is busy (an old fake headset? close the python process running fake_unicorn_bt.py)"
    }
    $FakeProc = Start-Process -FilePath $PY -ArgumentList @("`"$(Join-Path $Repo 'tools\fake_unicorn_bt.py')`"", "--port", $FakePort) `
        -PassThru -NoNewWindow -RedirectStandardOutput (Join-Path $Work "fake_headset.log") `
        -RedirectStandardError (Join-Path $Work "fake_headset.err")
    Start-Sleep -Seconds 1
}

try {
    New-Item -ItemType Directory -Path $Data -Force | Out-Null
    $w, $h = $Window.Split("x")
    $windowLines = @()
    if ($App -eq "CheckerboardSpeller") {   # DummyApplication has no window parameters
        $windowLines = @("Set parameter WindowLeft 0", "Set parameter WindowTop 40",
                         "Set parameter WindowWidth $w", "Set parameter WindowHeight $h")
    }
    $Dir = Join-Path $Data "$Subject$Session"
    $before = @(Get-ChildItem -Path $Dir -Filter *.dat -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })

    if ($Mode -eq "view") {
        $run = @("Set config", "Wait for Resting 30", "Show window",
                 "Echo Looking only: nothing is recorded. Close the Operator window when you are done.",
                 "Wait for Running 86400")
    } elseif ($Mode -eq "check") {
        $run = @("Set config", "Wait for Resting 30", "Show window",
                 "Echo Check the traces in the Source window. The recording starts in 5 s: 20 s blinking, 20 s eyes closed, 20 s jaw clench.",
                 "Sleep 5", "Start",
                 "Echo NOW: blink hard, about once a second (20 s)", "Sleep 20",
                 "Echo NOW: close your eyes and relax (20 s)", "Sleep 20",
                 "Echo NOW: clench your jaw, on and off (20 s)", "Sleep 20",
                 "Stop", "Wait for Suspended 10")
    } elseif ($Auto) {
        $run = @("Set config", "Wait for Resting 30", "Sleep 2", "Start", "Wait for Suspended 3600")
    } else {
        $run = @("Set config", "Wait for Resting 30", "Show window",
                 "Echo Configured. Check the signal in the Source window, then press Start in the Operator window.",
                 "Wait for Running", "Wait for Suspended")
    }

    # The Source window shows the raw signal: every display filter off (set explicitly, since the Operator
    # remembers the last choice). The recording is never filtered either way.
    $script = @(
        'Change directory $BCI2000LAUNCHDIR',
        "Reset system",
        "Startup system localhost --SystemLogFile=$(Slash $SystemLog)",
        "Start executable UnicornSerialSource --local $SourceArgs",
        "Start executable $SP --local",
        "Start executable $App --local",
        "Wait for Connected 30",
        "Set VisProperty SRCD.HPFilter off",
        "Set VisProperty SRCD.LPFilter off",
        "Set VisProperty SRCD.NotchFilter off",
        "Load parameterfile `"$(Slash $SpellerPrm)`""
    ) + $Settings + @(
        "Set parameter DataDirectory `"$(Slash $Data)`"",
        "Set parameter SubjectName $Subject",
        "Set parameter SubjectSession $Session"
    ) + $windowLines + $run + @("Quit")
    [System.IO.File]::WriteAllLines((Join-Path $Work "session.bat"), $script, (New-Object System.Text.UTF8Encoding($false)))

    $label = "run_session: $Mode, subject $Subject, session $Session"
    if ($Mode -eq "online") { $label += ", condition $Condition" }
    Write-Host $label
    Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
    Push-Location $Prog
    try {
        # Not the exit status: BCI2000Shell exits nonzero after any Echo even when every command worked
        # (measured on macOS). Real failures are the errors in the system log and a missing recording.
        & (Join-Path $Prog "BCI2000Shell.exe") (Join-Path $Work "session.bat")
    } finally {
        Pop-Location
    }

    Write-Host ""
    if ($Mode -eq "view") { exit 0 }
    $errors = @()
    if (Test-Path $SystemLog) {
        $errors = @(Select-String -Path $SystemLog -Pattern "Error")
        if ($errors) {
            Write-Host "--- BCI2000 errors ---"
            $errors | ForEach-Object { Write-Host $_.Line }
            if ($errors | Where-Object { $_.Line -match "maximum texture size" }) {
                Write-Host ("The graphics driver can't draw a window this large. Use a smaller one, e.g. " +
                            "ThaiSpeller.cmd -Window 1000x700, or update the graphics driver.")
            }
        }
    }
    # Only a file this run created counts: never report an earlier run's recording as this one's. An empty
    # file is BCI2000's placeholder from a configuration that never started, not a recording.
    $latest = Get-ChildItem -Path $Dir -Filter *.dat -ErrorAction SilentlyContinue |
        Where-Object { $before -notcontains $_.FullName -and $_.Length -gt 0 } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) {
        Write-Host "No recording was written by this run (system log: $SystemLog)."
        exit 1
    }
    $latest = $latest.FullName
    Write-Host "Recorded: $latest"
    # BCI2000's system log (module errors, the headset's packet-loss report) beside the recording it belongs to
    $logCopy = [System.IO.Path]::ChangeExtension($latest, ".system.log")
    if (Test-Path $SystemLog) {
        Copy-Item $SystemLog $logCopy
        Write-Host "System log: $logCopy"
    }
    if ($errors) {
        Write-Host "BCI2000 reported errors during this run (above), so its recording may be incomplete. Repeat the step."
        exit 1
    }
    switch ($Mode) {
        "calibrate" {
            # Every calibration run of this session folder, oldest first: speller runs (they have flashes)
            # without a classifier.
            $out = @(& $PY $Helper calibration-runs $Dir)
            $quoted = ($out | Select-Object -Skip 1 | ForEach-Object { "`"$_`"" }) -join " "
            Write-Host "Calibration runs in this session: $($out[0]) (do another run, or fit on all of them)"
            $demo = if ($Fake) { " --allow-demo" } else { "" }
            # One manifest per subject; the Transformer as its NumPy-readable export (no PyTorch on Windows)
            $stem = if ($Fake) { "demo_$Subject" } else { $Subject }
            Write-Host "Next: & `"$PY`" -m pipeline.fit_from_recording --id $Subject --recording $quoted --manifest models\${stem}_manifest.json --lm-transformer-checkpoint models\thai_gpt.lmbin$demo"
        }
        "online" { Write-Host "Summary: & `"$PY`" -m stopping.summarize_experiment --recording `"$latest`"" }
        "check" { Write-Host "Checks: & `"$PY`" tools\h13_montage_check.py `"$latest`"" }
    }
    exit 0   # not BCI2000Shell's status, which is nonzero after any Echo
} finally {
    # A BCI2000 script that stops on an error never reaches its Quit line, which would leave BCI2000 open and
    # block the next run. Nothing of BCI2000 was running when this run started (checked above), so all of it is ours.
    Get-Process -Name Operator, UnicornSerialSource, DummySignalProcessing, CheckerboardSignalProcessing,
        CheckerboardSpeller, DummyApplication, BCI2000Shell -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    if ($FakeProc -and -not $FakeProc.HasExited) {
        Start-Sleep -Seconds 1
        Stop-Process -Id $FakeProc.Id -Force -ErrorAction SilentlyContinue
    }
}
