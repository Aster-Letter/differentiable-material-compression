param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(1, 2)]
    [int]$PassNumber,
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
$evidenceRoot = Join-Path $repoRoot 'outputs\analysis\ue-bc-latent-feasibility-v1'
$traceRoot = Join-Path $evidenceRoot ("raw\timing\pass{0}" -f $PassNumber)
$ddcRoot = Join-Path $repoRoot '.scratch\ue-ddc'
$configPath = Join-Path $repoRoot 'configs\eval\ue_bc_latent_visual_v1.json'
$expectedConfigSha256 = 'd7ca91b31bc9d66899e8abfd709bbef7a5252ca165ee8f3f3c99afc92f998071'
$actualConfigSha256 = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualConfigSha256 -ne $expectedConfigSha256) {
    throw "Visual config hash mismatch: $actualConfigSha256"
}
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$sourceVariant = $config.variants | Where-Object { $_.id -eq 'lantern_material_render_160k_bc7' }
if (@($sourceVariant).Count -ne 1) {
    throw 'Expected exactly one Lantern material-render 160k visual variant.'
}

$forwardOrder = @(
    'lantern_material_render_160k_rgba8',
    'lantern_material_render_160k_bc7'
)
$variants = if ($PassNumber -eq 1) { $forwardOrder } else { @($forwardOrder[1], $forwardOrder[0]) }
$maps = @{
    lantern_material_render_160k_rgba8 = $sourceVariant.rgba8_map
    lantern_material_render_160k_bc7 = $sourceVariant.bc7_map
}
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

New-Item -ItemType Directory -Force -Path $traceRoot | Out-Null
$results = @()
foreach ($variant in $variants) {
    $tracePath = Join-Path $traceRoot ("{0}.utrace" -f $variant)
    $logPath = Join-Path $traceRoot ("{0}.log" -f $variant)
    if ((Test-Path -LiteralPath $tracePath) -or (Test-Path -LiteralPath $logPath)) {
        if ($SkipCompleted -and (Test-Path -LiteralPath $tracePath) -and (Test-Path -LiteralPath $logPath)) {
            Write-Host "Skipping completed pass $PassNumber variant $variant"
        } else {
            throw "Refusing to overwrite existing output for pass $PassNumber variant $variant"
        }
    } else {
        $arguments = @(
            $project,
            $maps[$variant],
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
        map = $maps[$variant]
        order_index = [array]::IndexOf($variants, $variant)
        exit_code = 0
        trace_path = $tracePath.Substring($repoRoot.Length + 1).Replace('\', '/')
        trace_bytes = $traceInfo.Length
        trace_sha256 = (Get-FileHash -LiteralPath $tracePath -Algorithm SHA256).Hash.ToLowerInvariant()
        log_path = $logPath.Substring($repoRoot.Length + 1).Replace('\', '/')
        log_bytes = $logInfo.Length
        log_sha256 = (Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.ToLowerInvariant()
        visual_config_sha256 = $actualConfigSha256
        formal_holdout_accessed = $false
    }
}

$runReport = [ordered]@{
    schema_version = 1
    status = 'complete'
    pass = $PassNumber
    ordering = if ($PassNumber -eq 1) { 'rgba8_then_bc7' } else { 'bc7_then_rgba8' }
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
Write-Host "Completed timing pass $PassNumber; report: $reportPath"
