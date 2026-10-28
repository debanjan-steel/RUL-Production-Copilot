# ==============================================================
# Auto-activate .venv when this folder is opened in PowerShell
# VS Code handles activation automatically via .vscode/settings.json
# ==============================================================

$venvPath = Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "[OK] .venv activated (Python 3.13)" -ForegroundColor Green
} else {
    Write-Host "[WARNING] .venv not found. Run: python -m venv .venv" -ForegroundColor Yellow
}
