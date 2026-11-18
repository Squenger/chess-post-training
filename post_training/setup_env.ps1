# Install CUDA PyTorch, Python dependencies, and Stockfish on Windows
$ErrorActionPreference = "Continue"
$env:UV_LINK_MODE = "copy"

$PostTrainingDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $PostTrainingDir
New-Item -ItemType Directory -Force -Path (Join-Path $PostTrainingDir "bin") | Out-Null
Set-Location $RepoRoot

function Clear-VenvReadOnly {
    param([string]$VenvPath)
    if (Test-Path $VenvPath) {
        Get-ChildItem $VenvPath -Recurse -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Attributes -band [System.IO.FileAttributes]::ReadOnly } |
            ForEach-Object { $_.Attributes = $_.Attributes -band (-bnot [System.IO.FileAttributes]::ReadOnly) }
    }
}

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

Write-Host "[1/3] Installing CUDA PyTorch..."
$global:LASTEXITCODE = 0
uv pip install --link-mode=copy --reinstall-package torch --reinstall-package sympy --reinstall-package mpmath torch sympy mpmath --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Clear-VenvReadOnly "$RepoRoot\.venv"

Write-Host "[2/3] Installing Python packages..."
$global:LASTEXITCODE = 0
uv pip install --link-mode=copy "chess==1.11.2" "safetensors>=0.4" "numpy>=1.24" "huggingface_hub>=0.24" "cryptography>=42" "datasets>=2.18" "stockfish>=3.28" "tqdm>=4.66" "wandb>=0.16" "accelerate>=0.28"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Clear-VenvReadOnly "$RepoRoot\.venv"

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
Write-Host "[3/3] Checking Stockfish and GPU..."
$global:LASTEXITCODE = 0
& $PythonExe "$PostTrainingDir/inspect_repo.py" --download-stockfish
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete. Run .\post_training\run_pipeline.ps1" -ForegroundColor Green
