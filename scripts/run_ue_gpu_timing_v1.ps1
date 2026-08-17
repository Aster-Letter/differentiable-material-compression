param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2)]
    [int]$PassNumber,
    [int]$MaxVariants = 0,
    [switch]$SkipCompleted,
    [string]$UnrealEditor = $env:UE_EDITOR_EXE
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repoRoot 'ue_demo\CGCompressionDemo\CGCompressionDemo.uproject'
if ([string]::IsNullOrWhiteSpace($UnrealEditor)) {
    throw 'Set UE_EDITOR_EXE or pass -UnrealEditor with the UE 5.8 UnrealEditor.exe path.'
}
$editor = (Resolve-Path -LiteralPath $UnrealEditor).Path
$evidenceRoot = Join-Path $repoRoot 'outputs\analysis\ue-runtime-evidence-v1'
$traceRoot = Join-Path $evidenceRoot ("raw\traces\pass{0}" -f $PassNumber)
$ddcRoot = Join-Path $repoRoot '.scratch\ue-ddc'
$contractPath = Join-Path $evidenceRoot 'measurement_contract.json'
$expectedContractSha256 = '652fd3761a3d8817e7f026eb067206098a12ad565c6677ab15b13ce679433bb2'
$forwardOrder = @(
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
$variants = if ($PassNumber -eq 1) { $forwardOrder } else { @($forwardOrder[($forwardOrder.Count - 1)..0]) }
$execCommands = @(
    'r.SetRes 1920x1080w',
    'r.ScreenPercentage 100',
    'r.DynamicRes.OperationMode 0',
    'r.VSync 0',
    't.MaxFPS 0',
    'r.Streaming.FullyLoadUsedTextures 1',
    'sg.ViewDistanceQuality 3',
    'sg.AntiAliasingQuality 3',
    'sg.ShadowQuality 3',
    'sg.GlobalIlluminationQuality 3',
    'sg.ReflectionQuality 3',
    'sg.PostProcessQuality 3',
    'sg.TextureQuality 3',
    'sg.EffectsQuality 3',
    'sg.FoliageQuality 3',
    'sg.ShadingQuality 3',
    'r.MaterialQualityLevel 1',
    'stat gpu',
    'stat unit'
) -join ','

$actualContractSha256 = (Get-FileHash -LiteralPath $contractPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualContractSha256 -ne $expectedContractSha256) {
    throw "Measurement contract hash mismatch: $actualContractSha256"
}

New-Item -ItemType Directory -Force -Path $traceRoot | Out-Null
$results = @()
$launched = 0

foreach ($variant in $variants) {
    $tracePath = Join-Path $traceRoot ("{0}.utrace" -f $variant)
    $logPath = Join-Path $traceRoot ("{0}.log" -f $variant)
    if ((Test-Path -LiteralPath $tracePath) -or (Test-Path -LiteralPath $logPath)) {
        if ($SkipCompleted -and (Test-Path -LiteralPath $tracePath) -and (Test-Path -LiteralPath $logPath)) {
            Write-Host "Skipping completed pass $PassNumber variant $variant"
            $traceInfo = Get-Item -LiteralPath $tracePath
            $logInfo = Get-Item -LiteralPath $logPath
            $results += [ordered]@{
                pass = $PassNumber
                variant_id = $variant
                order_index = [array]::IndexOf($variants, $variant)
                reused_existing = $true
                exit_code = 0
                trace_path = $tracePath.Substring($repoRoot.Length + 1).Replace('\', '/')
                trace_bytes = $traceInfo.Length
                trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
                log_path = $logPath.Substring($repoRoot.Length + 1).Replace('\', '/')
                log_bytes = $logInfo.Length
                log_sha256 = (Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.ToLowerInvariant()
                contract_sha256 = $actualContractSha256
                formal_holdout_accessed = $false
            }
            continue
        }
        throw "Refusing to overwrite existing output for pass $PassNumber variant $variant"
    }
    if ($MaxVariants -gt 0 -and $launched -ge $MaxVariants) {
        break
    }

    $map = "/Game/CGCompression/PerformanceEvidenceV1/Maps/Single/$variant"
    $arguments = @(
        $project,
        $map,
        '-game',
        '-d3d12',
        '-RenderOffscreen',
        '-ForceRes',
        '-ResX=1920',
        '-ResY=1080',
        '-Windowed',
        '-NoSound',
        '-NoSplash',
        '-NoVSync',
        '-SECONDS=50',
        '-DDC=InstalledNoZenLocalFallback',
        "-LocalDataCachePath=$ddcRoot",
        '-trace=gpu,frame,cpu,bookmark',
        "-tracefile=$tracePath",
        "-ExecCmds=`"$execCommands`"",
        "-abslog=$logPath"
    )
    $startedAt = Get-Date
    Write-Host "Starting pass $PassNumber variant $variant at $($startedAt.ToString('o'))"
    $process = Start-Process -FilePath $editor -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
    $finishedAt = Get-Date
    if ($process.ExitCode -ne 0) {
        throw "UE exited with code $($process.ExitCode) for pass $PassNumber variant $variant"
    }
    if (!(Test-Path -LiteralPath $tracePath) -or !(Test-Path -LiteralPath $logPath)) {
        throw "Missing trace or log for pass $PassNumber variant $variant"
    }
    $traceInfo = Get-Item -LiteralPath $tracePath
    $logInfo = Get-Item -LiteralPath $logPath
    if ($traceInfo.Length -le 0 -or $logInfo.Length -le 0) {
        throw "Empty trace or log for pass $PassNumber variant $variant"
    }
    $results += [ordered]@{
        pass = $PassNumber
        variant_id = $variant
        order_index = [array]::IndexOf($variants, $variant)
        started_at = $startedAt.ToString('o')
        finished_at = $finishedAt.ToString('o')
        wall_seconds = ($finishedAt - $startedAt).TotalSeconds
        exit_code = $process.ExitCode
        trace_path = $tracePath.Substring($repoRoot.Length + 1).Replace('\', '/')
        trace_bytes = $traceInfo.Length
        trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
        log_path = $logPath.Substring($repoRoot.Length + 1).Replace('\', '/')
        log_bytes = $logInfo.Length
        log_sha256 = (Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.ToLowerInvariant()
        contract_sha256 = $actualContractSha256
        formal_holdout_accessed = $false
    }
    $launched += 1
}

$runReport = [ordered]@{
    schema_version = 1
    status = 'complete'
    pass = $PassNumber
    ordering = if ($PassNumber -eq 1) { 'forward' } else { 'reverse' }
    requested_seconds = 50
    warmup_seconds_after_first_gpu_frame = 30
    measurement_windows = 5
    window_seconds = 3
    resolution = @(1920, 1080)
    rhi = 'DirectX 12'
    shader_platform = 'PCD3D_SM6'
    material_quality = 'High'
    scalability = 'Epic'
    results = $results
}
$reportPath = Join-Path $traceRoot 'run_report.json'
$runReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportPath -Encoding utf8
Write-Host "Completed $launched variant(s); report: $reportPath"
