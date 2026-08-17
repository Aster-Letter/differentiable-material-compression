$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$evidenceRoot = Join-Path $repoRoot 'outputs\analysis\ue-bc-latent-feasibility-v1'
$wrapper = Join-Path $repoRoot 'scripts\export_ue_insights_trace_v1.cmd'
$variants = @(
    'lantern_material_render_160k_rgba8',
    'lantern_material_render_160k_bc7'
)
$results = @()

foreach ($pass in @(1, 2)) {
    $passRoot = Join-Path $evidenceRoot ("raw\timing\pass{0}" -f $pass)
    foreach ($variant in $variants) {
        $tracePath = Join-Path $passRoot "$variant.utrace"
        $eventsPath = Join-Path $passRoot "$variant.gpu_events.csv"
        $logPath = Join-Path $passRoot "$variant.insights-export.log"
        if (!(Test-Path -LiteralPath $tracePath)) {
            throw "Missing trace: $tracePath"
        }
        $reused = (Test-Path -LiteralPath $eventsPath) -and (Test-Path -LiteralPath $logPath)
        if (!$reused) {
            if ((Test-Path -LiteralPath $eventsPath) -or (Test-Path -LiteralPath $logPath)) {
                throw "Refusing partial overwrite for pass $pass variant $variant"
            }
            Write-Host "Exporting pass $pass variant $variant"
            & $wrapper $tracePath $eventsPath $logPath gpu_events
            if ($LASTEXITCODE -ne 0) {
                throw "Insights export failed for pass $pass variant $variant with $LASTEXITCODE"
            }
        }
        if (!(Test-Path -LiteralPath $eventsPath) -or (Get-Item -LiteralPath $eventsPath).Length -le 80) {
            throw "Missing or unexpectedly small events file: $eventsPath"
        }
        $results += [ordered]@{
            pass = $pass
            variant_id = $variant
            reused_existing = $reused
            events_path = $eventsPath.Substring($repoRoot.Length + 1).Replace('\', '/')
            events_sha256 = (Get-FileHash -LiteralPath $eventsPath -Algorithm SHA256).Hash.ToLowerInvariant()
            export_log_path = $logPath.Substring($repoRoot.Length + 1).Replace('\', '/')
            export_log_sha256 = (Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
}

$report = [ordered]@{
    schema_version = 1
    status = 'complete'
    filter = [ordered]@{
        thread = 'GPU0-Graphics0'
        timers = @('Frame*', 'BasePass')
        columns = @('ThreadId', 'ThreadName', 'TimerId', 'TimerName', 'StartTime', 'EndTime', 'Duration', 'Depth')
    }
    results = $results
}
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evidenceRoot 'gpu_timing_export_report.json') -Encoding utf8
Write-Host "Exported or verified $($results.Count) GPU event files."
