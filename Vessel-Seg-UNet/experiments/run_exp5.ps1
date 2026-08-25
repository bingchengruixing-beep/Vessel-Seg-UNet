# 自有数据实验: 基线 + clDice 两组训练, 各自原始/后处理评估 + 跨域评估。
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:NO_PROXY = "*"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "experiments\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Experiments = @(
    @{ Name = "exp5_own_baseline";  Cfg = "configs\experiments\exp5_own_baseline.yaml" },
    @{ Name = "exp5_own_cl_dice";   Cfg = "configs\experiments\exp5_own_cl_dice.yaml" }
)

foreach ($Exp in $Experiments) {
    $Name = $Exp.Name
    $Cfg = $Exp.Cfg
    Write-Host "`n===== $Name =====" -ForegroundColor Cyan
    & $Python "train.py" --config $Cfg *> (Join-Path $LogDir "train_$Name.log")
    if ($LASTEXITCODE -ne 0) { Write-Host "TRAIN FAILED ($Name): $LASTEXITCODE" -ForegroundColor Red; continue }
    $Best = Join-Path $Root "checkpoints\$Name\best_model.pth"
    if (-not (Test-Path $Best)) { Write-Host "No best_model.pth ($Name)" -ForegroundColor Red; continue }

    # 自有验证集评估(原始 + 后处理)
    & $Python "evaluate.py" --checkpoint $Best --config $Cfg --output-dir "results\experiments\${Name}_raw" *> (Join-Path $LogDir "eval_${Name}_raw.log")
    & $Python "evaluate.py" --checkpoint $Best --config $Cfg --postprocess --output-dir "results\experiments\${Name}_pp" *> (Join-Path $LogDir "eval_${Name}_pp.log")

    # 跨域: 用 DIAS 验证集评估(exp1 配置指向 DIAS val)
    & $Python "evaluate.py" --checkpoint $Best --config "configs\experiments\exp1_baseline.yaml" --output-dir "results\experiments\${Name}_on_dias" *> (Join-Path $LogDir "eval_${Name}_on_dias.log")
}

# 反向跨域: DIAS 训练的 exp3 检查点在自有验证集上
$Exp3Best = Join-Path $Root "checkpoints\exp3_cl_dice\best_model.pth"
if (Test-Path $Exp3Best) {
    & $Python "evaluate.py" --checkpoint $Exp3Best --config "configs\experiments\exp5_own_cl_dice.yaml" --output-dir "results\experiments\exp3_cl_dice_on_own" *> (Join-Path $LogDir "eval_exp3_cl_dice_on_own.log")
}

Write-Host "`nAll exp5 experiments finished." -ForegroundColor Green
