"""汇总所有实验的评估报告,输出 Markdown 对比表。

用法: .venv/Scripts/python.exe experiments/summarize_results.py
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = [
    ("exp1_baseline", "UNet + BCEDice", "基线"),
    ("exp2_focal_tversky", "UNet + FocalTversky", "损失对比"),
    ("exp3_cl_dice", "UNet + BCEDice + clDice", "中心线监督"),
    ("exp4_attn_focal_cldice", "AttentionUNet + FocalTversky + clDice", "组合"),
]


def parse_report(path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    metrics = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+([0-9.]+)\s*±\s*([0-9.]+)", line)
        if m:
            metrics[m.group(1).lower()] = (float(m.group(2)), float(m.group(3)))
    return metrics


def parse_train_log(name):
    log = ROOT / "experiments" / "logs" / f"train_{name}.log"
    if not log.exists():
        return None, None
    raw_bytes = log.read_bytes()
    # PowerShell 5.1 的 *> 重定向输出 UTF-16,按字节特征自动选择编码
    if raw_bytes[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw_bytes.decode("utf-16", errors="ignore")
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")
    best_dice = None
    for line in text.splitlines():
        m = re.search(r"[Bb]est Dice:\s*([0-9.]+)", line)
        if m:
            best_dice = float(m.group(1))
    epochs = None
    m = re.search(r"Early stopping after (\d+) epochs", text)
    if m:
        epochs = int(m.group(1))
    else:
        matches = re.findall(r"Epoch \[(\d+)/\d+\]", text)
        epochs = int(matches[-1]) if matches else None
    return best_dice, epochs


def fmt(value):
    if value is None:
        return "—"
    return f"{value:.4f}"


lines = ["# 预实验报告(DIAS 开源于集)—— 非最终数据", ""]
lines.append("> ⚠️ 本报告对应预实验阶段使用的 DIAS 开源于集(50 对);正式完整数据集(177 对自有标注)见 RESULTS.md\n")
lines.append("> 数据集: DIAS(train 30 / val 20,800×800 灰度 DSA,前景占比 ~6%)\n")
lines.append("> 训练策略(全部一致): AdamW lr=1e-4、warmup 5 + cosine、grad clip 1.0、EMA 0.999、seed 42、AMP、batch 2、早停 patience 10\n")

lines.append("## 验证集指标(best_model,EMA 权重)\n")
lines.append("| 实验 | 模型 + 损失 | 模式 | Dice | IoU | Precision | Recall | 训练最佳Dice | 完成Epoch |")
lines.append("|------|-------------|------|------|-----|-----------|--------|--------------|-----------|")

rows = []
for name, desc, tag in EXPERIMENTS:
    raw = parse_report(ROOT / "results" / "experiments" / f"{name}_raw" / "eval_report.txt")
    pp = parse_report(ROOT / "results" / "experiments" / f"{name}_pp" / "eval_report.txt")
    best_dice, epochs = parse_train_log(name)
    if raw is None:
        continue
    rows.append((name, desc, "原始", raw, best_dice, epochs))
    if pp is not None:
        rows.append((name, desc, "+后处理", pp, best_dice, epochs))

for name, desc, mode, metrics, best_dice, epochs in rows:
    lines.append(
        f"| {name} | {desc} | {mode} | {fmt(metrics.get('dice', (None,))[0])} "
        f"| {fmt(metrics.get('iou', (None,))[0])} | {fmt(metrics.get('precision', (None,))[0])} "
        f"| {fmt(metrics.get('recall', (None,))[0])} | {fmt(best_dice)} | {epochs if epochs is not None else '—'} |"
    )

lines.append("")
lines.append("## 结论")
lines.append("")
lines.append(
    "1. **clDice 中心线监督是本轮最大赢家**: exp3(UNet + BCEDice + clDice)原始 Dice **0.7029**,"
    "较基线 exp1(0.6861)提升 **+1.7pp**,IoU 提升 +1.6pp;且 Precision/Recall 最均衡(0.706/0.716),"
    "逐样本标准差最小(0.045),稳定性最好。骨架监督在不改模型架构的前提下即可获得拓扑层面的提升。"
)
lines.append("")
lines.append(
    "2. **Focal Tversky 按设计工作,但不改变 Dice**: exp2 的 Recall 从 0.661 提升到 **0.758(+9.7pp)**,"
    "Precision 相应下降(0.748→0.644)——α=0.7/β=0.3 的漏检惩罚生效。"
    "若应用场景更看重\"少漏血管\"(如术前评估),选 Focal Tversky;若看重整体分割质量,选 clDice 方案。"
)
lines.append("")
lines.append(
    "3. **exp4(AttentionUNet + FocalTversky + clDice)组合不占优**: 原始 Dice 0.6699,低于所有 UNet 配置。"
    "注意 exp4 因系统休眠被中断于 51/80 epoch(未触发早停,最佳 0.6660@epoch42 仍在爬升),"
    "且 AttentionUNet 参数量(30.9M)在 30 张训练图上易过拟合。小数据上应优先损失改进,而非加大模型。"
)
lines.append("")
lines.append(
    "4. **默认后处理对细血管数据集有害**: min_component_size=50 会把真实细支血管当噪声删除,"
    "四组实验后处理后 Dice 均下降 2~2.5pp。DIAS 上建议 min_component_size≤10 或关闭后处理;"
    "部署时按数据集血管口径重新标定该参数。"
)
lines.append("")
lines.append(
    "5. **训练成本**: clDice 的逐 epoch Zhang-Suen 骨架计算(CPU,numpy)使 exp3 每 epoch ~55s(基线 ~7s)。"
    "后续可把骨架降采样到一半分辨率计算(拓扑损失对分辨率不敏感)或预计算+几何同步,"
    "可将开销降回接近基线水平。"
)
lines.append("")
lines.append(
    "6. **环境备注**: 训练期间笔记本多次休眠导致墙钟时间大幅膨胀,exp4 因此被中断;"
    "重跑请保持接通电源/关闭睡眠,或直接 `powershell -File experiments/run_experiments.ps1` 重跑 exp4。"
)

out = ROOT / "experiments" / "results_summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("written:", out)
