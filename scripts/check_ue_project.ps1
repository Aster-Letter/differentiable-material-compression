param(
    [string]$ProjectPath = 'ue_demo\CGCompressionDemo\CGCompressionDemo.uproject',
    [switch]$RunHeadless
)

$ErrorActionPreference = 'Stop'

function Assert-Match {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [Parameter(Mandatory = $true)][string]$Pattern,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if ($Text -notmatch $Pattern) {
        throw "$Label is not configured as expected (pattern: $Pattern)"
    }
    Write-Host "${Label}: OK"
}

$project = Get-Item -LiteralPath (Resolve-Path -LiteralPath $ProjectPath) -ErrorAction Stop
$projectRoot = $project.Directory.FullName
$descriptor = Get-Content -LiteralPath $project.FullName -Raw | ConvertFrom-Json
$association = [string]$descriptor.EngineAssociation

$builds = Get-ItemProperty 'HKCU:\Software\Epic Games\Unreal Engine\Builds' -ErrorAction Stop
$engineProperty = $builds.PSObject.Properties | Where-Object Name -eq $association | Select-Object -First 1
if ($null -eq $engineProperty) {
    throw "No installed Unreal Engine matches association $association"
}

$engineRoot = [string]$engineProperty.Value
$buildVersionPath = Join-Path $engineRoot 'Engine\Build\Build.version'
$editorCmd = Join-Path $engineRoot 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$build = Get-Content -LiteralPath $buildVersionPath -Raw | ConvertFrom-Json

Write-Host '== Project =='
Write-Host "uproject: $($project.FullName)"
Write-Host "engine: $($build.MajorVersion).$($build.MinorVersion).$($build.PatchVersion) changelist $($build.Changelist)"
Write-Host "engine root: $engineRoot"

$engineConfigPath = Join-Path $projectRoot 'Config\DefaultEngine.ini'
$engineConfig = Get-Content -LiteralPath $engineConfigPath -Raw

Write-Host "`n== Render contract =="
Assert-Match $engineConfig '(?m)^DefaultGraphicsRHI=DefaultGraphicsRHI_DX12\s*$' 'DX12 RHI'
Assert-Match $engineConfig '(?m)^\+D3D12TargetedShaderFormats=PCD3D_SM6\s*$' 'Shader Model 6'
Assert-Match $engineConfig '(?m)^TargetedHardwareClass=Desktop\s*$' 'Desktop target'
Assert-Match $engineConfig '(?m)^DefaultGraphicsPerformance=Maximum\s*$' 'Maximum quality'
Assert-Match $engineConfig '(?m)^r\.RayTracing=False\s*$' 'Hardware ray tracing disabled'
Assert-Match $engineConfig '(?m)^r\.Substrate=False\s*$' 'Substrate disabled'

Write-Host "`n== Generated directory ignore rules =="
foreach ($name in @('DerivedDataCache', 'Intermediate', 'Saved')) {
    $path = Join-Path $projectRoot $name
    if (Test-Path -LiteralPath $path) {
        git check-ignore --quiet -- $path
        if ($LASTEXITCODE -ne 0) {
            throw "$path is generated but not ignored by Git"
        }
        Write-Host "${name}: ignored"
    }
}

$logsRoot = Join-Path $projectRoot 'Saved\Logs'
$latestLog = Get-ChildItem -LiteralPath $logsRoot -File -Filter *.log -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

Write-Host "`n== Latest editor log =="
if ($null -eq $latestLog) {
    Write-Host 'No editor log found yet.'
} else {
    Write-Host "$($latestLog.FullName) ($($latestLog.LastWriteTime))"
    Select-String -LiteralPath $latestLog.FullName -Pattern @(
        'LogInit: Engine Version:',
        'LogD3D12RHI: Chosen D3D12 Adapter',
        'LogRHI: RHI D3D12 with Feature Level',
        'LogConfig: Set CVar \[\[r\.RayTracing:',
        'LogConfig: Set CVar \[\[r\.Substrate:'
    ) | ForEach-Object { Write-Host $_.Line }

    $staleRuntimeSettings = Select-String -LiteralPath $latestLog.FullName -Pattern @(
        'LogConfig: Set CVar \[\[r\.RayTracing:1\]\]',
        'LogConfig: Set CVar \[\[r\.Substrate:1\]\]'
    )
    if ($staleRuntimeSettings) {
        Write-Warning 'Latest editor boot predates the checked-in render contract; restart Unreal Editor and rerun this check.'
    }

    $handledEnsures = Select-String -LiteralPath $latestLog.FullName -Pattern 'Handled ensure:'
    if ($handledEnsures) {
        Write-Warning "Latest log contains $($handledEnsures.Count) handled ensure(s); inspect before performance capture."
    }
}

if ($RunHeadless) {
    if (Get-Process UnrealEditor -ErrorAction SilentlyContinue) {
        throw 'Close the interactive Unreal Editor before the headless boot test.'
    }

    Write-Host "`n== Headless boot =="
    & $editorCmd $project.FullName -Unattended -NullRHI -NoSplash -NoSound -NoShaderCompile '-ExecCmds=Quit'
    if ($LASTEXITCODE -ne 0) {
        throw "UnrealEditor-Cmd failed with exit code $LASTEXITCODE"
    }
    Write-Host 'Headless boot: OK'
}
