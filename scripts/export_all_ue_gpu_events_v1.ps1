$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$evidenceRoot = Join-Path $repoRoot 'outputs\analysis\ue-runtime-evidence-v1'
$wrapper = Join-Path $repoRoot 'scripts\export_ue_insights_trace_v1.cmd'
$variants = @(
    'lantern_source_core4',
    'lantern_raw_q4',
    'lantern_material_render_20k',
    'lantern_material_render_160k',
    'helmet_source_core4',
    'helmet_c4_affine_80k',
    'helmet_r0b_direct_scalar',
    'helmet_c4_dtf_160k',
    'helmet_c5_dtf_120k'
)
$results = @()

foreach ($pass in @(1, 2)) {
    $passRoot = Join-Path $evidenceRoot ("raw\traces\pass{0}" -f $pass)
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
        if (!(Test-Path -LiteralPath $eventsPath) -or !(Test-Path -LiteralPath $logPath)) {
            throw "Missing exported events or log for pass $pass variant $variant"
        }
        $eventsInfo = Get-Item -LiteralPath $eventsPath
        if ($eventsInfo.Length -le 80) {
            throw "Exported events file is unexpectedly small: $eventsPath"
        }
        $results += [ordered]@{
            pass = $pass
            variant_id = $variant
            reused_existing = $reused
            events_path = $eventsPath.Substring($repoRoot.Length + 1).Replace('\', '/')
            events_bytes = $eventsInfo.Length
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
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $evidenceRoot 'gpu_events_export_report.json') -Encoding utf8
Write-Host "Exported or verified $($results.Count) GPU event files."
