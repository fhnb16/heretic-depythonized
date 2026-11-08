# Renders the UI with the mocked Tauri API in headless Microsoft Edge, saves screenshots
# and reports JavaScript errors.
# Usage: pwsh apps/heretic-gui/ui-test/screenshots.ps1 [-Scenarios home,run] [-Lang ru] [-Theme light] [-Skin heretic] [-Out target/ui-shots]
param(
    [string[]]$Scenarios = @("home", "new", "gguf", "goal", "params", "run", "results", "chat", "export", "savegguf", "settings", "failed", "update", "updating", "about"),
    [string]$Lang = "ru",
    [string]$Theme = "light",
    [string]$Skin = "heretic",
    [string]$Out = "target/ui-shots",
    [int]$Port = 8737
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path

$edge = @(
    "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edge) { throw "Microsoft Edge not found" }

# Minimal static file server (ES modules can't be loaded from file:// URLs). POST /__report
# stores the scenario report sent by the mock.
$server = Start-Job -ArgumentList $root.Path, $Port, $Out -ScriptBlock {
    param($root, $port, $out)
    $types = @{ ".html" = "text/html; charset=utf-8"; ".js" = "text/javascript; charset=utf-8"; ".css" = "text/css; charset=utf-8"; ".png" = "image/png" }
    $listener = [System.Net.HttpListener]::new()
    $listener.Prefixes.Add("http://localhost:$port/")
    $listener.Start()
    while ($listener.IsListening) {
        $context = $listener.GetContext()
        $request = $context.Request
        if ($request.Url.AbsolutePath -eq "/__quit") {
            $context.Response.Close()
            $listener.Stop()
            break
        }
        if ($request.Url.AbsolutePath -eq "/__report") {
            $reader = [System.IO.StreamReader]::new($request.InputStream)
            $body = $reader.ReadToEnd()
            [System.IO.File]::WriteAllText((Join-Path $out "report-$($request.QueryString['scenario']).json"), $body)
            $context.Response.Close()
            continue
        }
        $path = [Uri]::UnescapeDataString($request.Url.AbsolutePath.TrimStart("/"))
        $file = Join-Path $root $path
        if (Test-Path $file -PathType Leaf) {
            $bytes = [System.IO.File]::ReadAllBytes($file)
            $context.Response.ContentType = $types[[System.IO.Path]::GetExtension($file)] ?? "application/octet-stream"
            $context.Response.OutputStream.Write($bytes, 0, $bytes.Length)
        } else {
            $context.Response.StatusCode = 404
        }
        $context.Response.Close()
    }
}

$failed = $false
try {
    Start-Sleep -Milliseconds 800
    foreach ($scenario in $Scenarios) {
        $report = Join-Path $Out "report-$scenario.json"
        Remove-Item $report -ErrorAction SilentlyContinue
        $url = "http://localhost:$Port/ui-test/index.html?scenario=$scenario&lang=$Lang&theme=$Theme&skin=$Skin"
        $png = Join-Path $Out "$scenario-$Lang-$Theme-$Skin.png"
        Remove-Item $png -ErrorAction SilentlyContinue
        $userData = Join-Path $Out ".edge-profile"
        $process = Start-Process -FilePath $edge -PassThru -WindowStyle Hidden -ArgumentList @(
            "--headless=new", "--disable-gpu", "--no-first-run", "--user-data-dir=`"$userData`"",
            "--window-size=1280,860", "--virtual-time-budget=15000", "--screenshot=`"$png`"", $url
        )
        # Newer Edge launchers return at once and render in a child process, so wait for the
        # report and the screenshot themselves rather than for the launcher to exit.
        $deadline = (Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $deadline -and -not ((Test-Path $report) -and (Test-Path $png))) {
            Start-Sleep -Milliseconds 250
        }
        $process.WaitForExit(5000) | Out-Null

        if (-not (Test-Path $report)) {
            Write-Host "$scenario`: no report (scenario did not finish)" -ForegroundColor Yellow
            $failed = $true
            continue
        }
        $result = Get-Content $report -Raw | ConvertFrom-Json
        if ($result.errors.Count -gt 0) {
            Write-Host "$scenario`: ERRORS" -ForegroundColor Red
            $result.errors | ForEach-Object { Write-Host "  $_" }
            $failed = $true
        } else {
            Write-Host "$scenario`: ok -> $png" -ForegroundColor Green
        }
    }
} finally {
    # GetContext() blocks, so Stop-Job alone would wait about two minutes for the listener.
    try { Invoke-WebRequest "http://localhost:$Port/__quit" -TimeoutSec 3 | Out-Null } catch {}
    Wait-Job $server -Timeout 5 | Out-Null
    Stop-Job $server -ErrorAction SilentlyContinue
    Remove-Job $server -Force -ErrorAction SilentlyContinue
}

if ($failed) { exit 1 }
