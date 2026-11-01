# Full pipeline for Windows PowerShell: exploration -> CV + final fit -> predictions -> report -> check.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
foreach ($script in "eda", "train", "predict", "make_report", "check_submission") {
    Write-Host "== src/$script.py"
    python "src/$script.py"
    if ($LASTEXITCODE -ne 0) { throw "src/$script.py failed" }
}
