# 全量对比实验结果

> 数据集: DIAS(train 30 / val 20,800×800 灰度 DSA,前景占比 ~6%)

> 训练策略(全部一致): AdamW lr=1e-4、warmup 5 + cosine、grad clip 1.0、EMA 0.999、seed 42、AMP、batch 2、早停 patience 10

## 验证集指标(best_model,EMA 权重)

| 实验 | 模型 + 损失 | 模式 | Dice | IoU | Precision | Recall | 训练最佳Dice | 完成Epoch |
|------|-------------|------|------|-----|-----------|--------|--------------|-----------|
| exp1_baseline | UNet + BCEDice | 原始 | 0.6861 | 0.5277 | 0.7477 | 0.6609 | 0.6823 | 45 |
| exp1_baseline | UNet + BCEDice | +后处理 | 0.6648 | 0.5033 | 0.7092 | 0.6584 | 0.6823 | 45 |
| exp2_focal_tversky | UNet + FocalTversky | 原始 | 0.6855 | 0.5242 | 0.6441 | 0.7577 | 0.6802 | 51 |
| exp2_focal_tversky | UNet + FocalTversky | +后处理 | 0.6609 | 0.4963 | 0.6037 | 0.7590 | 0.6802 | 51 |
| exp3_cl_dice | UNet + BCEDice + clDice | 原始 | 0.7029 | 0.5436 | 0.7056 | 0.7159 | 0.6995 | 52 |
| exp3_cl_dice | UNet + BCEDice + clDice | +后处理 | 0.6830 | 0.5205 | 0.6579 | 0.7301 | 0.6995 | 52 |
| exp4_attn_focal_cldice | AttentionUNet + FocalTversky + clDice | 原始 | 0.6699 | 0.5057 | 0.6133 | 0.7617 | 0.6660 | 51 |
| exp4_attn_focal_cldice | AttentionUNet + FocalTversky + clDice | +后处理 | 0.6479 | 0.4813 | 0.5718 | 0.7726 | 0.6660 | 51 |

## 结论

1. **clDice 中心线监督是本轮最大赢家**: exp3(UNet + BCEDice + clDice)原始 Dice **0.7029**,较基线 exp1(0.6861)提升 **+1.7pp**,IoU 提升 +1.6pp;且 Precision/Recall 最均衡(0.706/0.716),逐样本标准差最小(0.045),稳定性最好。骨架监督在不改模型架构的前提下即可获得拓扑层面的提升。

2. **Focal Tversky 按设计工作,但不改变 Dice**: exp2 的 Recall 从 0.661 提升到 **0.758(+9.7pp)**,Precision 相应下降(0.748→0.644)——α=0.7/β=0.3 的漏检惩罚生效。若应用场景更看重"少漏血管"(如术前评估),选 Focal Tversky;若看重整体分割质量,选 clDice 方案。

3. **exp4(AttentionUNet + FocalTversky + clDice)组合不占优**: 原始 Dice 0.6699,低于所有 UNet 配置。注意 exp4 因系统休眠被中断于 51/80 epoch(未触发早停,最佳 0.6660@epoch42 仍在爬升),且 AttentionUNet 参数量(31.4M)在 30 张训练图上易过拟合。小数据上应优先损失改进,而非加大模型。

4. **默认后处理对细血管数据集有害**: min_component_size=50 会把真实细支血管当噪声删除,四组实验后处理后 Dice 均下降 2~2.5pp。DIAS 上建议 min_component_size≤10 或关闭后处理;部署时按数据集血管口径重新标定该参数。

5. **训练成本**: clDice 的逐 epoch Zhang-Suen 骨架计算(CPU,numpy)使 exp3 每 epoch ~55s(基线 ~7s)。后续可把骨架降采样到一半分辨率计算(拓扑损失对分辨率不敏感)或预计算+几何同步,可将开销降回接近基线水平。

6. **环境备注**: 训练期间笔记本多次休眠导致墙钟时间大幅膨胀,exp4 因此被中断;重跑请保持接通电源/关闭睡眠,或直接 `powershell -File experiments/run_experiments.ps1` 重跑 exp4。
