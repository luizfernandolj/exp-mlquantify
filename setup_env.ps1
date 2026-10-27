# Cria o ambiente virtual e instala as dependencias (Windows / PowerShell).
#
#   .\setup_env.ps1
#   .\setup_env.ps1 -Python "C:\caminho\para\python3.11.exe"

param([string]$Python = "")

$ErrorActionPreference = "Stop"

if (-not $Python) {
    # Procura o Python 3.11 pelo lancador py; cai para o python do PATH.
    try { $Python = (& py -3.11 -c "import sys; print(sys.executable)") }
    catch { $Python = "python" }
}

Write-Host "Interpretador: $Python"
& $Python --version

& $Python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
Write-Host "Ambiente pronto. Verificando as bibliotecas..."
& .\.venv\Scripts\python.exe -c @"
import importlib.metadata as md
for p in ('mlquantify', 'quapy', 'quantificationlib', 'qunfold', 'scikit-learn'):
    print(f'  {p:20s} {md.version(p)}')
"@

Write-Host ""
Write-Host "Proximos passos:"
Write-Host "  .\.venv\Scripts\python.exe build_catalog.py"
Write-Host "  .\.venv\Scripts\python.exe run_experiments.py --run-id final"
