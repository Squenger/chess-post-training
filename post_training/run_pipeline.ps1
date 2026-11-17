# End-to-end training pipeline (data prep, SFT, RL, export)
param(
    [switch]$ForceData,
    [switch]$SkipRL,
    [switch]$SkipEngineEval
)

$ErrorActionPreference = "Continue"
$env:UV_LINK_MODE = "copy"

$PostTrainingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PostTrainingDir
Set-Location $RepoRoot

function Clear-VenvReadOnly {
    param([string]$VenvPath)
    if (Test-Path $VenvPath) {
        Get-ChildItem $VenvPath -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReadOnly } |
            ForEach-Object { $_.Attributes = $_.Attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly) }
    }
}

function Invoke-Step {
    param([string]$Label, [scriptblock]$Cmd)
    Write-Host "`n[$Label]" -ForegroundColor Cyan
    $global:LASTEXITCODE = 0
    & $Cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[error] $Label failed (code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Pre-flight: unlock read-only dirs in .venv and ensure CUDA torch + deps are intact
if (Test-Path "$RepoRoot\.venv") {
    Clear-VenvReadOnly "$RepoRoot\.venv"
    Get-ChildItem "$RepoRoot\.venv\Lib\site-packages\*.dist-info" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        if (-not (Test-Path (Join-Path $_.FullName "RECORD"))) {
            $pkg = ($_.Name -split "-")[0]
            Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
            $pkgDir = Join-Path "$RepoRoot\.venv\Lib\site-packages" $pkg
            if (Test-Path $pkgDir) { Remove-Item -Recurse -Force $pkgDir -ErrorAction SilentlyContinue }
        }
    }
}

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }

$global:LASTEXITCODE = 0
& $PythonExe -c "import sys; exec('try:\n from sympy import S\n import mpmath, torch\n sys.exit(0 if torch.cuda.is_available() else 2)\nexcept Exception:\n sys.exit(1)')"
$CheckCode = $LASTEXITCODE

if ($CheckCode -eq 2 -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "[setup] Installing CUDA PyTorch into .venv..." -ForegroundColor Yellow
    Clear-VenvReadOnly "$RepoRoot\.venv"
    Remove-Item -Recurse -Force "$RepoRoot\.venv\Lib\site-packages\sympy*", "$RepoRoot\.venv\Lib\site-packages\mpmath*" -ErrorAction SilentlyContinue
    $global:LASTEXITCODE = 0
    uv pip install --link-mode=copy --reinstall-package torch --reinstall-package sympy --reinstall-package mpmath torch sympy mpmath --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Clear-VenvReadOnly "$RepoRoot\.venv"
} elseif ($CheckCode -ne 0 -and $CheckCode -ne 2) {
    Write-Host "[setup] Repairing sympy/mpmath..." -ForegroundColor Yellow
    Clear-VenvReadOnly "$RepoRoot\.venv"
    Remove-Item -Recurse -Force "$RepoRoot\.venv\Lib\site-packages\sympy*", "$RepoRoot\.venv\Lib\site-packages\mpmath*" -ErrorAction SilentlyContinue
    $global:LASTEXITCODE = 0
    uv pip install --link-mode=copy --reinstall sympy mpmath
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Clear-VenvReadOnly "$RepoRoot\.venv"
}

$global:LASTEXITCODE = 0
& $PythonExe -c "import sys; exec('try:\n import datasets, tqdm\n sys.exit(0)\nexcept Exception:\n sys.exit(1)')"
if ($LASTEXITCODE -ne 0) {
    Clear-VenvReadOnly "$RepoRoot\.venv"
    $global:LASTEXITCODE = 0
    uv pip install --link-mode=copy "datasets>=2.18" "stockfish>=3.28" "tqdm>=4.66"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Clear-VenvReadOnly "$RepoRoot\.venv"
}

Invoke-Step "1/5 Inspect repo & Stockfish" {
    & $PythonExe "$PostTrainingDir/inspect_repo.py" --download-stockfish
}

$TrainDataPt = Join-Path $PostTrainingDir "data\sft_datamix\train_datamix.pt"
$ValDataPt   = Join-Path $PostTrainingDir "data\sft_datamix\val_positions.pt"

if ((Test-Path $TrainDataPt) -and (Test-Path $ValDataPt) -and (-not $ForceData)) {
    Write-Host "`n[2/5 Data prep] Found existing dataset, skipping (use -ForceData to rebuild)." -ForegroundColor Green
} else {
    Invoke-Step "2/5 Data prep" {
        & $PythonExe "$PostTrainingDir/prepare_data.py"
    }
}

Invoke-Step "3/5 SFT training" {
    & $PythonExe "$PostTrainingDir/train_sft.py"
}

if (-not $SkipRL) {
    Invoke-Step "4/5 RL GRPO + Q-distillation" {
        & $PythonExe "$PostTrainingDir/train_rl_grpo.py"
    }
}

Invoke-Step "5/5 Export & verification" {
    if ($SkipEngineEval) {
        & $PythonExe "$PostTrainingDir/export_submission.py"
    } else {
        & $PythonExe "$PostTrainingDir/export_submission.py" --run_engine_eval
    }
}

Write-Host "`nDone. Model saved to $PostTrainingDir\final_model\model.safetensors" -ForegroundColor Green
