$ErrorActionPreference = 'Stop'

function Find-Executable {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string[]]$Fallbacks = @()
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    foreach ($candidate in $Fallbacks) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return $null
}

$programFiles = [Environment]::GetFolderPath('ProgramFiles')
$programFilesX86 = [Environment]::GetFolderPath('ProgramFilesX86')
$cudaRoot = Join-Path $programFiles 'NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$nvcc = Find-Executable 'nvcc' @((Join-Path $cudaRoot 'bin\nvcc.exe'))
$blender = Find-Executable 'blender' @(
    (Join-Path $programFiles 'Blender Foundation\Blender 5.2\blender.exe')
)
$epicLauncher = Join-Path $programFilesX86 'Epic Games\Launcher\Portal\Binaries\Win64\EpicGamesLauncher.exe'

Write-Host '== System tools =='
foreach ($name in @('git', 'cmake', 'ninja', 'nvidia-smi')) {
    $path = Find-Executable $name
    if ($null -eq $path) {
        Write-Host "${name}: MISSING"
    } else {
        Write-Host "${name}: $path"
    }
}
Write-Host "nvcc: $(if ($nvcc) { $nvcc } else { 'MISSING' })"
Write-Host "blender: $(if ($blender) { $blender } else { 'MISSING' })"
Write-Host "Epic Launcher: $(if (Test-Path -LiteralPath $epicLauncher) { $epicLauncher } else { 'MISSING' })"

Write-Host "`n== GPU =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0) {
    throw "nvidia-smi failed with exit code $LASTEXITCODE"
}

Write-Host "`n== CUDA compiler =="
if ($null -eq $nvcc) {
    throw 'CUDA 12.6 nvcc was not found.'
}
& $nvcc --version
if ($LASTEXITCODE -ne 0) {
    throw "nvcc failed with exit code $LASTEXITCODE"
}

Write-Host "`n== Python packages =="
$pythonCheck = "import sys, torch, nvdiffrast; print('python:', sys.version.replace(chr(10), ' ')); print('torch:', torch.__version__); print('torch cuda:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('nvdiffrast:', nvdiffrast.__file__)"
conda run -n cg-frontier python -c $pythonCheck
if ($LASTEXITCODE -ne 0) {
    throw "Python package check failed with exit code $LASTEXITCODE"
}

Write-Host "`n== nvdiffrast smoke test =="
$env:CUDA_HOME = $cudaRoot
conda run -n cg-frontier python -m pytest tests/test_nvdiffrast_smoke.py -q -p no:cacheprovider
if ($LASTEXITCODE -ne 0) {
    throw "nvdiffrast smoke test failed with exit code $LASTEXITCODE"
}
