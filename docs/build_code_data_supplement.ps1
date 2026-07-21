$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$staging = Join-Path $repoRoot "tmp\code_and_data_supplement"
$archive = Join-Path $repoRoot "docs\code_and_data_supplement.zip"

if (-not $staging.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to use a staging directory outside the repository."
}

if (Test-Path -LiteralPath $staging) {
    Remove-Item -LiteralPath $staging -Recurse -Force
}
New-Item -ItemType Directory -Path $staging | Out-Null

function Copy-ArtifactFile {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    $source = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Missing artifact file: $RelativePath"
    }
    $destination = Join-Path $staging $RelativePath
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination
}

function Copy-ArtifactTree {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    $source = Join-Path $repoRoot $RelativePath
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Missing artifact directory: $RelativePath"
    }
    $destination = Join-Path $staging $RelativePath
    New-Item -ItemType Directory -Path (Split-Path $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse
}

Copy-Item -LiteralPath (Join-Path $PSScriptRoot "code_data_supplement_README.md") -Destination (Join-Path $staging "README.md")
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "code_data_supplement_setup.py") -Destination (Join-Path $staging "setup.py")
Copy-ArtifactFile "requirements.txt"
Copy-ArtifactTree "src"

$dataFiles = @(
    "data\prepare_data.py",
    "data\download_longbench.py",
    "data\gen_hotpotqa_synthetic.py",
    "data\processed\hotpotqa_test.json",
    "data\processed\2wikimultihopqa_test.json",
    "data\processed\musique_test.json",
    "data\processed\longbench_multifieldqa_en_test.json",
    "data\processed\longbench_musique_test.json",
    "data\processed\longbench_narrativeqa_test.json",
    "data\processed\longbench_qasper_test.json",
    "data\processed\longbench_2wikimqa_test.json"
)
$dataFiles | ForEach-Object { Copy-ArtifactFile $_ }

$experimentFiles = @(
    "experiments\run_parallel.py",
    "experiments\run_quick_exp.py",
    "experiments\run_comparison.py",
    "experiments\run_oracle.py",
    "experiments\run_easv.py",
    "experiments\reproduce_paper_results.py",
    "experiments\_paired_stats.py",
    "experiments\_paired_stats_longbench.py",
    "experiments\_budget_curve.py",
    "experiments\_hop_curve.py",
    "experiments\_extend_musique_cot_sc_hop4.py",
    "experiments\_decomp_diagnosis.py",
    "experiments\_verify_probe.py",
    "experiments\analyze_hotpotqa_types.py",
    "experiments\gen_figures.py"
)
$experimentFiles | ForEach-Object { Copy-ArtifactFile $_ }
Copy-ArtifactTree "experiments\configs"

$resultTrees = @(
    "experiments\results\longbench_multifieldqa_en_8b",
    "experiments\results\longbench_musique_8b",
    "experiments\results\longbench_narrativeqa_8b",
    "experiments\results\longbench_qasper_8b",
    "experiments\results\longbench_2wikimqa_8b",
    "experiments\results\n500_fullctx_8b",
    "experiments\results\musique_n500_8b",
    "experiments\results\oracle_musique_8b",
    "experiments\results\p2_qwenplus",
    "experiments\results\budget_curve_8b",
    "experiments\results\easv_musique_8b",
    "experiments\results\verify_probe",
    "experiments\results\grounded_test",
    "experiments\results\repair_test",
    "experiments\results\case_study"
)
$resultTrees | ForEach-Object { Copy-ArtifactTree $_ }

$resultFiles = @(
    "experiments\results\hotpotqa_gers_sc_results.json",
    "experiments\results\hotpotqa_gers_adaptive_cv_results.json",
    "experiments\results\hotpotqa_gers_adaptive_cv2_results.json",
    "experiments\results\2wikimultihopqa_standard_cot_results.json",
    "experiments\results\2wikimultihopqa_gers_adaptive_cv2_results.json"
)
$resultFiles | ForEach-Object { Copy-ArtifactFile $_ }

$testFiles = @(
    "tests\__init__.py",
    "tests\test_graph.py",
    "tests\test_consistency.py",
    "tests\test_chain_generation.py",
    "tests\test_baselines.py",
    "tests\test_utils.py"
)
$testFiles | ForEach-Object { Copy-ArtifactFile $_ }

Get-ChildItem -LiteralPath $staging -Recurse -Directory |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache") } |
    Sort-Object FullName -Descending |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }
Get-ChildItem -LiteralPath $staging -Recurse -File -Include "*.pyc", "*.pyo" |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }

if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}
Compress-Archive -Path (Join-Path $staging "*") -DestinationPath $archive -CompressionLevel Optimal

$sizeMB = (Get-Item -LiteralPath $archive).Length / 1MB
if ($sizeMB -gt 50) {
    throw ("Archive is {0:N2} MB, exceeding the 50 MB submission limit." -f $sizeMB)
}

Write-Host ("Created {0} ({1:N2} MB)" -f $archive, $sizeMB)
