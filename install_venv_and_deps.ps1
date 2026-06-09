<#
PowerShell helper script to create a local virtual environment named .venv
and install packages from requirements.txt. Run this from the workspace root
in VS Code integrated terminal (PowerShell).
#>

$venv = ".venv"
if (-not (Test-Path $venv)) {
    Write-Host "Creating virtual environment: $venv"
    python -m venv $venv
} else {
    Write-Host "Virtual environment already exists: $venv"
}

$activate = Join-Path $venv "Scripts\Activate.ps1"
if (Test-Path $activate) {
    Write-Host "Activating virtual environment for this session..."
    & $activate
} else {
    Write-Host "Activation script not found at $activate"
}

Write-Host "Upgrading pip and installing requirements..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Installation finished. To activate the venv in future terminals, run:" 
Write-Host "    .\\.venv\\Scripts\\Activate.ps1"

Write-Host "If torch installation fails on Windows, visit https://pytorch.org/get-started/locally/ for the correct wheel command."
