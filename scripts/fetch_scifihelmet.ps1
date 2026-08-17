[CmdletBinding()]
param(
    [string]$Destination = "assets/source/glTF-Sample-Assets"
)

$ErrorActionPreference = "Stop"
$repository = "https://github.com/KhronosGroup/glTF-Sample-Assets.git"
$resolvedDestination = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Destination))

if (Test-Path -LiteralPath $resolvedDestination) {
    throw "Destination already exists: $resolvedDestination"
}

git clone --depth 1 --filter=blob:none --sparse $repository $resolvedDestination
if ($LASTEXITCODE -ne 0) {
    throw "Sparse clone failed with exit code $LASTEXITCODE"
}

git -C $resolvedDestination sparse-checkout set Models/SciFiHelmet
if ($LASTEXITCODE -ne 0) {
    throw "Sparse checkout failed with exit code $LASTEXITCODE"
}

$entry = Join-Path $resolvedDestination "Models/SciFiHelmet/glTF/SciFiHelmet.gltf"
if (-not (Test-Path -LiteralPath $entry)) {
    throw "SciFiHelmet entry file was not downloaded: $entry"
}

Write-Host "SciFiHelmet is ready at $entry"
