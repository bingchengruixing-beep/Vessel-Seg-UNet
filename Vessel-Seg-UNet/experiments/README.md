# experiments/ 目录说明

| 文件/目录 | 用途 |
|-----------|------|
| `RESULTS.md` | **正式实验结果报告(完整数据集 177 对)**,主报告 |
| `metrics_all.csv` | 全部实验指标总表(预实验 + 主实验 + 跨域,机器可读) |
| `results_summary.md` | 预实验报告(DIAS 开源于集,由 `summarize_results.py` 生成) |
| `build_dataset.py` | 合并自有标注 → `data/train` + `data/val`(软标注二值化、尺寸对齐、前缀命名) |
| `run_exp5.ps1` | 完整数据集主实验一键脚本(基线 + clDice 训练/评估/跨域) |
| `run_experiments.ps1` | 预实验(DIAS)一键脚本 |
| `summarize_results.py` | 生成预实验汇总表 |
| `smoke_test.py` | GPU 全链路冒烟测试(数据→损失→反向→EMA) |
| `diagnostics/` | 一次性诊断/复现脚本(环境探测、数据体检、bug 复现),保留备查 |
| `logs/` | 全部训练/评估/环境安装日志(注意:PowerShell 重定向产物为 UTF-16 编码) |

## 主实验复现

```powershell
.\.venv\Scripts\python.exe experiments\build_dataset.py --force   # 重建 data/
powershell -NoProfile -ExecutionPolicy Bypass -File experiments\run_exp5.ps1
```

推荐配置:`configs/experiments/exp5_own_cl_dice.yaml`
推荐检查点:`checkpoints/exp5_own_cl_dice/best_model.pth`(EMA 权重)
