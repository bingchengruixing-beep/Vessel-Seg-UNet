# 全量实验: 依次运行 4 组配置, 每组训练后分别做原始/后处理评估。
# 用法: .\.venv\Scripts\pwsh ... 或 powershell -File experiments/run_experiments.ps1
# 不用 $ErrorActionPreference="Stop": 子进程 stderr(如 tqdm 进度条)会被当成
# 致命错误误杀训练。统一用 $LASTEXITCODE 判断每一步是否失败。
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:NO_PROXY = "*"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "venv not found: $Python" }

$LogDir = Join-Path $Root "experiments\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Experiments = @(
    @{ Name = "exp1_baseline";            Desc = "UNet + BCEDice" },
    @{ Name = "exp2_focal_tversky";       Desc = "UNet + FocalTversky" },
    @{ Name = "exp3_cl_dice";             Desc = "UNet + BCEDice + clDice" },
    @{ Name = "exp4_attn_focal_cldice";   Desc = "AttentionUNet + FocalTversky + clDice" }
)

foreach ($Exp in $Experiments) {
    $Name = $Exp.Name
    $Cfg = Join-Path $Root "configs\experiments\$Name.yaml"
    $CkptDir = Join-Path $Root "checkpoints\$Name"
    Write-Host "`n===== $($Exp.Desc) [$Name] =====" -ForegroundColor Cyan

    Write-Host "[train] $Name"
    & $Python "train.py" --config $Cfg *> (Join-Path $LogDir "train_$Name.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "TRAIN FAILED ($Name): $LASTEXITCODE" -ForegroundColor Red; continue }

    $Best = Join-Path $CkptDir "best_model.pth"
    if (-not (Test-Path $Best)) { Write-Host "No best_model.pth for $Name" -ForegroundColor Red; continue }

    Write-Host "[eval raw] $Name"
    & $Python "evaluate.py" --checkpoint $Best --config $Cfg --output-dir "results\experiments\${Name}_raw" *> (Join-Path $LogDir "eval_${Name}_raw.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "EVAL RAW FAILED ($Name): $LASTEXITCODE" -ForegroundColor Red }

    Write-Host "[eval postprocess] $Name"
    & $Python "evaluate.py" --checkpoint $Best --config $Cfg --postprocess --output-dir "results\experiments\${Name}_pp" *> (Join-Path $LogDir "eval_${Name}_pp.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "EVAL PP FAILED ($Name): $LASTEXITCODE" -ForegroundColor Red }
}

Write-Host "`nAll experiments finished." -ForegroundColor Green
