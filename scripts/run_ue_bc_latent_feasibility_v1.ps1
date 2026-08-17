param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Setup', 'Residency', 'VisualSetup', 'Visual')]
    [string]$Phase,
    [string]$UnrealEditor = $env:UE_EDITOR_EXE,
    [string]$VariantId = '',
    [ValidateSet('RGBA8', 'BC7', 'Both')]
    [string]$VisualKind = 'Both',
    [string]$ReplicateId = '',
    [switch]$KeepOpen
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repoRoot 'ue_demo\CGCompressionDemo\CGCompressionDemo.uproject'
$configPath = Join-Path $repoRoot 'configs\eval\ue_bc_latent_feasibility_v1.json'
$visualConfigPath = Join-Path $repoRoot 'configs\eval\ue_bc_latent_visual_v1.json'
$evidenceRoot = Join-Path $repoRoot 'outputs\analysis\ue-bc-latent-feasibility-v1'
$pythonRoot = Join-Path $repoRoot 'ue_demo\CGCompressionDemo\Content\Python'

if ([string]::IsNullOrWhiteSpace($UnrealEditor)) {
    throw 'Set UE_EDITOR_EXE or pass -UnrealEditor with the UE 5.8 UnrealEditor.exe path.'
}
$editor = (Resolve-Path -LiteralPath $UnrealEditor).Path
$editorCmd = Join-Path (Split-Path -Parent $editor) 'UnrealEditor-Cmd.exe'
if (!(Test-Path -LiteralPath $editorCmd)) {
    throw "UnrealEditor-Cmd.exe not found beside $editor"
}
New-Item -ItemType Directory -Force -Path $evidenceRoot | Out-Null

if ($Phase -eq 'Setup') {
    $logPath = Join-Path $evidenceRoot 'ue_setup.log'
    if (Test-Path -LiteralPath $logPath) {
        throw "Refusing to overwrite existing setup log: $logPath"
    }
    $scriptPath = Join-Path $pythonRoot 'setup_ue_bc_latent_feasibility_v1.py'
    $arguments = @(
        $project,
        '-run=pythonscript',
        "-script=$scriptPath",
        '-unattended',
        '-nop4',
        '-nosplash',
        '-d3d12',
        "-abslog=$logPath"
    )
    $process = Start-Process -FilePath $editorCmd -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "UE BC latent setup failed with exit code $($process.ExitCode)"
    }
    $report = Get-Content -Raw (Join-Path $evidenceRoot 'ue_setup_report.json') | ConvertFrom-Json
    if ($report.status -ne 'complete' -or $report.variants.Count -ne 3) {
        throw "UE BC latent setup report is incomplete: $($report.status)"
    }
    Write-Host 'BC latent setup completed for three variants.'
    exit 0
}

if ($Phase -eq 'VisualSetup') {
    $logPath = Join-Path $evidenceRoot 'ue_visual_setup.log'
    if (Test-Path -LiteralPath $logPath) {
        throw "Refusing to overwrite existing visual setup log: $logPath"
    }
    $scriptPath = Join-Path $pythonRoot 'setup_ue_bc_latent_visual_v1.py'
    $arguments = @(
        $project,
        '-run=pythonscript',
        "-script=$scriptPath",
        '-unattended',
        '-nop4',
        '-nosplash',
        '-d3d12',
        "-abslog=$logPath"
    )
    $process = Start-Process -FilePath $editorCmd -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "UE BC latent visual setup failed with exit code $($process.ExitCode)"
    }
    $report = Get-Content -Raw (Join-Path $evidenceRoot 'ue_visual_setup_report.json') | ConvertFrom-Json
    if ($report.status -ne 'complete' -or $report.map_count -ne 6) {
        throw "UE BC latent visual setup report is incomplete: $($report.status)"
    }
    Write-Host 'BC latent visual setup completed for six paired maps.'
    exit 0
}

$activeConfigPath = if ($Phase -eq 'Visual') { $visualConfigPath } else { $configPath }
$config = Get-Content -Raw $activeConfigPath | ConvertFrom-Json
$selectedVariants = @($config.variants)
if (![string]::IsNullOrWhiteSpace($VariantId)) {
    $selectedVariants = @($selectedVariants | Where-Object { $_.id -eq $VariantId })
    if ($selectedVariants.Count -ne 1) {
        throw "Expected exactly one variant named $VariantId, got $($selectedVariants.Count)"
    }
}

if ($Phase -eq 'Visual') {
    $kinds = if ($VisualKind -eq 'Both') { @('RGBA8', 'BC7') } else { @($VisualKind) }
    $visualLogRoot = Join-Path $evidenceRoot 'raw\visual'
    New-Item -ItemType Directory -Force -Path $visualLogRoot | Out-Null
    $env:UE_BC_LATENT_VISUAL_EDITOR_AUTOSTART = '1'
    if ($KeepOpen) {
        if ($selectedVariants.Count -ne 1 -or $kinds.Count -ne 1) {
            throw '-KeepOpen requires exactly one -VariantId and one -VisualKind.'
        }
        $env:UE_BC_LATENT_VISUAL_KEEP_OPEN = '1'
    }
    try {
        foreach ($variant in $selectedVariants) {
            foreach ($kind in $kinds) {
                $id = [string]$variant.id
                $kindLower = $kind.ToLowerInvariant()
                $replicateSuffix = if ([string]::IsNullOrWhiteSpace($ReplicateId)) { '' } else { "__$ReplicateId" }
                if ($ReplicateId -notmatch '^[A-Za-z0-9_-]*$') {
                    throw 'ReplicateId may contain only letters, digits, underscores, and hyphens.'
                }
                $map = if ($kind -eq 'BC7') { [string]$variant.bc7_map } else { [string]$variant.rgba8_map }
                $logPath = Join-Path $visualLogRoot "${id}__${kindLower}${replicateSuffix}.log"
                $reportPath = Join-Path $evidenceRoot "visual_runs\${id}__${kindLower}${replicateSuffix}.json"
                $screenshotPath = Join-Path $evidenceRoot "screenshots\${id}__${kindLower}${replicateSuffix}.png"
                if ((Test-Path -LiteralPath $logPath) -or (Test-Path -LiteralPath $reportPath) -or (Test-Path -LiteralPath $screenshotPath)) {
                    throw "Refusing to overwrite visual evidence for ${id}__${kindLower}"
                }
                $env:UE_BC_LATENT_VISUAL_VARIANT = $id
                $env:UE_BC_LATENT_VISUAL_KIND = $kindLower
                $env:UE_BC_LATENT_VISUAL_REPLICATE = $ReplicateId
                $arguments = @(
                    $project,
                    $map,
                    '-d3d12',
                    '-unattended',
                    '-DisablePlugins=StylusInput',
                    '-ForceRes',
                    '-ResX=1920',
                    '-ResY=1080',
                    '-Windowed',
                    '-NoSound',
                    '-NoSplash',
                    '-NoVSync',
                    "-abslog=$logPath"
                )
                if ($KeepOpen) {
                    $process = Start-Process -FilePath $editor -ArgumentList $arguments -PassThru
                    $deadline = [DateTime]::UtcNow.AddSeconds(120)
                    $captureReady = $false
                    while ([DateTime]::UtcNow -lt $deadline) {
                        if ($process.HasExited) {
                            throw "UE visual run exited before inspection handoff for ${id}__${kindLower}"
                        }
                        if (Test-Path -LiteralPath $reportPath) {
                            $pendingReport = Get-Content -Raw $reportPath | ConvertFrom-Json
                            if ($pendingReport.status -eq 'failed') {
                                throw "UE visual run failed for ${id}__${kindLower}: $($pendingReport.error)"
                            }
                            if ($pendingReport.status -eq 'complete_kept_open') {
                                $captureReady = $true
                                break
                            }
                        }
                        Start-Sleep -Seconds 1
                    }
                    if (!$captureReady) {
                        Stop-Process -Id $process.Id -Force
                        throw "UE visual run timed out for ${id}__${kindLower}"
                    }
                }
                else {
                    $process = Start-Process -FilePath $editor -ArgumentList $arguments -PassThru -WindowStyle Hidden
                    if (!$process.WaitForExit(120000)) {
                        Stop-Process -Id $process.Id -Force
                        throw "UE visual run timed out for ${id}__${kindLower}"
                    }
                    if ($process.ExitCode -ne 0) {
                        throw "UE visual run failed for ${id}__${kindLower} with exit code $($process.ExitCode)"
                    }
                }
                $report = Get-Content -Raw $reportPath | ConvertFrom-Json
                $acceptedStatus = if ($KeepOpen) { 'complete_kept_open' } else { 'complete' }
                if ($report.status -ne $acceptedStatus) {
                    throw "UE visual report is incomplete for ${id}__${kindLower}: $($report.status)"
                }
                Write-Host "Captured fixed frame for ${id}__${kindLower}"
            }
        }
    }
    finally {
        Remove-Item Env:UE_BC_LATENT_VISUAL_EDITOR_AUTOSTART -ErrorAction SilentlyContinue
        Remove-Item Env:UE_BC_LATENT_VISUAL_VARIANT -ErrorAction SilentlyContinue
        Remove-Item Env:UE_BC_LATENT_VISUAL_KIND -ErrorAction SilentlyContinue
        Remove-Item Env:UE_BC_LATENT_VISUAL_REPLICATE -ErrorAction SilentlyContinue
        Remove-Item Env:UE_BC_LATENT_VISUAL_KEEP_OPEN -ErrorAction SilentlyContinue
    }
    exit 0
}

$rawRoot = Join-Path $evidenceRoot 'raw\residency'
New-Item -ItemType Directory -Force -Path $rawRoot | Out-Null
$captureScript = Join-Path $pythonRoot 'capture_ue_bc_latent_residency_v1.py'
$captureSource = Get-Content -Raw -LiteralPath $captureScript
$forbiddenRuntimeApis = @(
    'get_editor_subsystem',
    'EditorLevelLibrary',
    'EditorActorSubsystem',
    'UnrealEditorSubsystem'
)
foreach ($api in $forbiddenRuntimeApis) {
    if ($captureSource.Contains($api)) {
        throw "Runtime capture script contains forbidden editor API: $api"
    }
}
$env:UE_BC_LATENT_CAPTURE_AUTOSTART = '1'
try {
    foreach ($variant in $selectedVariants) {
        $id = [string]$variant.id
        $logPath = Join-Path $rawRoot "$id.log"
        $reportPath = Join-Path $evidenceRoot "residency_runs\$id.json"
        if ((Test-Path -LiteralPath $logPath) -or (Test-Path -LiteralPath $reportPath)) {
            throw "Refusing to overwrite residency evidence for $id"
        }
        $env:UE_BC_LATENT_VARIANT = $id
        $arguments = @(
            $project,
            [string]$variant.destination_map,
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
            "-abslog=$logPath"
        )
        $process = Start-Process -FilePath $editor -ArgumentList $arguments -PassThru -Wait -WindowStyle Hidden
        if ($process.ExitCode -ne 0) {
            throw "UE residency run failed for $id with exit code $($process.ExitCode)"
        }
        $report = Get-Content -Raw $reportPath | ConvertFrom-Json
        if ($report.status -ne 'complete') {
            throw "UE residency report is incomplete for ${id}: $($report.status)"
        }
        Write-Host "Captured warmed residency for $id"
    }
}
finally {
    Remove-Item Env:UE_BC_LATENT_CAPTURE_AUTOSTART -ErrorAction SilentlyContinue
    Remove-Item Env:UE_BC_LATENT_VARIANT -ErrorAction SilentlyContinue
}
