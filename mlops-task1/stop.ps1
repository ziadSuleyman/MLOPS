# Shuts the environment down. The pgdata volume is never touched, so the
# loaded data survives and .\start.ps1 brings everything back in seconds.
#
# ASCII only, on purpose: see the note at the top of start.ps1.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "`nStopping JupyterLab if it is running..." -ForegroundColor Cyan
$jupyter = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
           Where-Object { $_.CommandLine -like "*jupyterlab*" }
if ($jupyter) {
    $jupyter | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force
        Write-Host "    Stopped PID $($_.ProcessId)." -ForegroundColor Green
    }
} else {
    Write-Host "    Not running." -ForegroundColor Gray
}

Write-Host "`nStopping the database..." -ForegroundColor Cyan
docker compose down

Write-Host "`nDone. Data is kept in the mlops-task1_pgdata volume." -ForegroundColor Green
Write-Host "Run .\start.ps1 to bring it all back." -ForegroundColor DarkGray
