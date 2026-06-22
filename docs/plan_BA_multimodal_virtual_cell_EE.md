# B+A 组合路线详细计划：多模态虚拟细胞解析 EE 线粒体状态

- 文档版本：v1（2026-06-22）
- 课题主线：B（EE hit calling / 线粒体状态图谱）+ A（多模态 virtual-cell 表型预测）
- 推荐题目：**A multimodal virtual-cell framework reveals separable mitochondrial potential and biogenesis programs in energy-expenditure perturbation screens**
- 中文题目：**多模态虚拟细胞框架解析能量消耗扰动中的线粒体膜电位与生物合成解耦机制**

---

## 0. 一句话定位

用 HepG2 EE siRNA KD 的 **DRUG-seq × brightfield/DINOv2 × TMRM/MitoTracker** 三模态数据，先建立可信的线粒体功能状态图谱，再训练一个严格防泄漏验证的多模态 virtual-cell 模型，回答：**能否从扰动转录组和无标记/线粒体影像中预测并解释 EE 相关线粒体表型？**

---

## 1. 核心科学假设

### H1 — EE 表型不是单轴 TMRM 高低，而是至少两个可分离的线粒体状态

TMRM ch2 与 MitoTracker ch4 的组合可以把传统 TMRM 信号拆成两条机制轴：

| 机制轴 | 操作定义 | 生物学解释 | 阳性锚点 |
| --- | --- | --- | --- |
| per-mito ΔΨm | ch2 / ch4 | 单个线粒体膜电位 / 偶联状态 | BAM15 ↓ |
| mito mass / biogenesis | ch4 / ch1 | 线粒体质量 / 生物合成 | MK8722 ↑ |

预期结果：候选 EE 靶点会形成不同状态簇，例如 uncoupler-like、biogenesis-like、energizer-like 和 toxic-collapse，而不是简单的高低排序。

### H2 — 转录组、明场形态和线粒体影像是互补模态，应做融合而非相互替代

现有分析显示每个模态内部可重复，但模态间相关弱。这不是失败，而是说明它们观测不同层级：

- DRUG-seq：扰动后的分子程序与 MoA。
- Brightfield C1：无标记细胞状态、形态和泛化特征。
- C24 TMRM+MitoTracker：线粒体特异表型和 Seahorse 近似读出。

预期结果：多模态模型在 leave-target-out 排序上应优于任何单模态模型，尤其在 top-hit enrichment、Spearman 和机制分类上提升。

### H3 — virtual-cell 的价值不是大模型拟合，而是小样本下的扰动表示和反事实排序

本数据是孔级 mini-bulk，每扰动约 6 个重复，不能按大规模单细胞 VAE 来做。应从强基线、小模型、严格验证开始，把 vCell 定位为：

1. 学习扰动 latent embedding。
2. 预测目标 KD 的线粒体状态。
3. 为未验证靶点做反事实排序和机制解释。

---

## 2. 当前可用数据地基

### 已完成资产

| 资产 | 文件 / 位置 | 用途 |
| --- | --- | --- |
| DRUG-seq processed AnnData | `data/processed/adata_drugseq_processed.h5ad` | 表达、HVG、KD-QC、QC flags |
| vCell-ready matrix | `data/processed/drugseq_vcell.npz` | NTC=0 固化，vCell 直接训练 |
| 三模态 AnnData | `data/processed/adata_multimodal.h5ad` | 表达 + C1/C24 DINOv2 + TMRM 表型 |
| 靶点级汇总 | `data/processed/targets_summary.csv` | hit calling 的核心表 |
| 孔级注释 | `data/processed/wells_annotation.csv` | 过滤毒性/QC、重复聚合 |
| 真实 Seahorse 验证 | `data/seahorse_vAssay_validation.csv` | 外部功能真值，n=16 target |
| vAssay review 结果 | `output/2026-06-11/vassay_systematic/systematic_benchmark.csv` | 防泄漏验证框架 |

### 已知关键事实

- 1440 孔，约 180 扰动，24 板，所有孔已三模态对齐。
- 批次-靶点强混淆，必须使用 plate-wise within-NTC 标准化。
- KD 分层：strong 46 / weak 96 / failed 28 / unknown 7。
- 毒性孔约 4%，必须 flag/penalty，不应简单删除所有信息。
- MitoTracker 轴已揭示 MK8722 的 TMRM 升高主要来自 mitochondrial mass/biogenesis，而不是 per-mito ΔΨm 增高。
- vAssay 同域排序可用，但 siRNA 域外推弱；后续模型必须使用 leave-target-out / leave-plate-out 验证。

---

## 3. 总体研究目标

### Main objective

建立一个以 **机制表型图谱 + 多模态 virtual-cell 预测器** 为核心的 EE 靶点发现框架，用于：

1. 发现和分类 EE 相关线粒体状态。
2. 预测靶点 KD 对 per-mito ΔΨm、mitochondrial mass、TMRM area 和 Seahorse-like readout 的影响。
3. 解释 top hit 的转录组 MoA，并输出可实验验证的候选靶点优先级。

### Specific aims

| Aim | 名称 | 对应路线 | 主要问题 |
| --- | --- | --- | --- |
| Aim 1 | EE mitochondrial state atlas | B | 哪些靶点产生 uncoupling / biogenesis / energizer 状态？ |
| Aim 2 | Multimodal virtual-cell predictor | A | 多模态模型能否比单模态更好地预测线粒体表型？ |
| Aim 3 | Mechanistic target prioritization | B+A | 哪些候选最值得做 Seahorse / 二次 KD 验证？ |

---

## 4. Aim 1：EE 线粒体状态图谱与 hit calling

### 4.1 输入

使用 `targets_summary.csv` 和 `wells_annotation.csv` 作为主输入，优先使用非毒性孔、QC-pass 孔；对 failed KD 不直接删除，而是在优先级中降权。

### 4.2 表型轴定义

| 表型 | 列名 | 方向解释 | 主用途 |
| --- | --- | --- | --- |
| per-mito ΔΨm | `pheno_permito_dpsi_z` | 下降 = uncoupling-like | 解偶联 / 偶联状态 |
| mito mass | `pheno_mitomass_z` | 上升 = biogenesis-like | 线粒体生物合成 |
| TMRM intensity | `pheno_intensity_z` | 综合 ΔΨm + mass | 历史兼容 |
| TMRM high-potential area | `pheno_area_z` | 下降 = ΔΨm area collapse | BAM15-like 主轴 |
| vAssay AUC / MB | `vassay_pred_AUC_*`, `vassay_pred_MB_*` | Seahorse-like readout | 辅助排序，信排序不信绝对值 |

### 4.3 靶点状态分类

建议先用透明规则，之后再用 clustering 验证。

| 类别 | 判定逻辑 | 生物学读法 |
| --- | --- | --- |
| uncoupler-like | per-mito ΔΨm 显著下降，mito mass 不显著上升或次要 | 类 BAM15，可能促进 proton leak / ΔΨm collapse |
| biogenesis-like | mito mass 显著上升，per-mito ΔΨm 不塌缩 | 类 MK8722，可能增加线粒体质量/代谢容量 |
| energizer-like | per-mito ΔΨm 和 mito mass 同向上升，毒性低 | 潜在提高线粒体功能，但需警惕 hyperpolarization |
| toxic-collapse | ΔΨm/area 下降且 cell_count 明显下降或 QC fail 高 | 毒性/死亡驱动，低优先级 |
| neutral / uncertain | 效应弱、KD failed 或重复不稳定 | 暂不推进 |

### 4.4 共识 hit score

建议定义一个可解释的 target-level 分数，而不是一开始训练黑箱分类器。

```text
consensus_score = phenotype_strength
                + kd_confidence_bonus
                + replicate_consistency_bonus
                + transcriptomic_moa_bonus
                + crossmodal_support_bonus
                - toxicity_penalty
                - qc_penalty
```

推荐初版权重：

| 组件 | 建议定义 | 权重方向 |
| --- | --- | --- |
| phenotype_strength | max(abs(permito_z), abs(mitomass_z), abs(area_z))，clip 到 10 | 主分 |
| kd_confidence | strong +2, weak +1, failed -3, unknown -1 | 强 KD 加分 |
| replicate_consistency | split-half 同向 / bootstrap CI 不跨 0 | 加分 |
| transcriptomic_moa | 与 BAM15/MK8722 signature 相似或相关通路富集 | 加分 |
| crossmodal_support | 表达、BF、C24 或 vAssay readout 至少两模态支持 | 加分 |
| toxicity | tox_rate > 0.2 开始扣分，>0.5 强扣 | 扣分 |
| qc_fail | qc_fail_rate 高扣分 | 扣分 |

### 4.5 关键图表

1. per-mito ΔΨm vs mito mass 二维状态图，点大小为 KD confidence，颜色为 hit class。
2. BAM15/MK8722/ATP5B/SLC25A4/PSMC3 阳性对照位置图。
3. top hit waterfall plot：按 consensus_score 排序。
4. top hit 的孔级重复箱线图，显示重复稳定性。
5. toxicity vs phenotype strength 图，标出 toxic-collapse。
6. phenotype class × KD tier 堆叠图。

### 4.6 Aim 1 成功标准

- 至少形成 3 个清晰机制状态簇。
- top hit 不是由 toxic wells 或 failed KD 主导。
- BAM15/MK8722 能在图谱上作为正确锚点分离。
- 产出 10–20 个优先级候选，并分成 immediate / secondary / deprioritized 三档。

---

## 5. Aim 2：多模态 virtual-cell 表型预测

### 5.1 预测任务

优先预测 target-level 或 well-level 的批内标准化表型。建议两层任务都做，但论文/汇报主结果用 target-level，避免孔级重复造成过度乐观。

| 任务 | y | 用途 |
| --- | --- | --- |
| T1 | `pheno_permito_dpsi_z` | 预测 uncoupling-like 状态 |
| T2 | `pheno_mitomass_z` | 预测 biogenesis-like 状态 |
| T3 | `pheno_area_z` | 兼容传统 TMRM high-potential area |
| T4 | real Seahorse AUC%（n=16） | 外部 sanity，不作为主训练目标 |
| T5 | vAssay pred_AUC / pred_MB | 辅助任务，权重低，避免把 proxy 当真值 |

### 5.2 输入模态

| 模态 | 特征 | 使用方式 |
| --- | --- | --- |
| Transcriptome | HVG lognorm / zscore、pathway scores、DE signature | 主模态，解释 MoA |
| Brightfield C1 | `X_dino_c1` | 泛化形态输入，优先用于模型外推 |
| Mito imaging C24 | `X_dino_c24` | 线粒体特异输入，辅助但需警惕 circularity |
| Metadata/QC | KD tier、tox flag、plate | 不作为生物预测输入；用于分层/过滤/校正 |

重要原则：C24 与 TMRM/MitoTracker 表型接近，可能带来“读出泄漏”。因此模型报告要分成三组：

1. **Expression-only**：最严格、最可泛化。
2. **Expression + BF**：推荐主模型，避免用线粒体染料直接预测线粒体表型。
3. **Expression + BF + C24**：上限模型，用来估计线粒体影像能补多少信息。

### 5.3 模型梯度

按复杂度从低到高推进，每一步必须打败前一步再升级。

#### Level 0 — 强基线

| 模型 | 输入 | 目的 |
| --- | --- | --- |
| Mean / plate NTC baseline | 无 | 最低基线 |
| Ridge / ElasticNet | 表达 HVG 或 pathway score | 小样本稳健基线 |
| PLSRegression | 表达 HVG → 多表型 | 多输出低秩基线 |
| DINO Ridge | C1 / C24 | 单影像模态基线 |
| Late-fusion Ridge | 表达 + C1 + C24 PCA | 简单融合基线 |

#### Level 1 — 可解释多模态融合

建议先做 target-level 的 late-fusion / multi-view latent model：

```text
expression features  ─┐
brightfield features ─┼─> low-dimensional shared latent ─> phenotype heads
C24 features         ─┘
```

可选实现：

- PCA/PLS per modality + Ridge head。
- MultiOutputRegressor + ElasticNet。
- sparse CCA / PLS 作为解释性 cross-modal latent。
- 简单 stacking：每个模态先 LOTO 生成 OOF prediction，再由 meta-model 融合。

#### Level 2 — vCell-style latent-additive perturbation model

在强基线站稳后，再做轻量 vCell extension：

```text
control / NTC expression -> encoder -> basal latent z
perturbation id          -> embedding p_k
z + p_k                  -> decoder -> expression reconstruction
z + p_k                  -> phenotype heads -> TMRM / mito mass / Seahorse-like readouts
```

模型约束：

- latent_dim 小，例如 8–32。
- perturb embedding 加 L2 正则。
- decoder 使用 log-normalized expression + MSE，暂不启用 NB decoder。
- phenotype head 权重高于 expression reconstruction，避免只学表达重构。
- batch/plate 不作为可泛化输入，只用于 adversarial 或 residual check；初版不建议复杂化。

#### Level 3 — 反事实和 zero-shot 方向

如果 Level 2 有信号，再探索 gene-program perturbation embedding：

- 用基因通路、GO、Reactome、protein family、人遗传注释给 perturbation embedding 加先验。
- 目标是对未实验靶点做近似 zero-shot 排序。
- 这属于 paper 加分项，不作为第一阶段交付硬要求。

### 5.4 防泄漏验证设计

所有模型必须报告以下 validation scheme：

| Scheme | Split | 回答的问题 | 主指标 |
| --- | --- | --- | --- |
| random well CV | 随机孔 | 仅作 sanity，不作为结论 | 不主报 |
| group_plate | 留整板 | 能否跨板泛化 | Spearman / MAE |
| leave-target-out | 留整靶点 | 能否预测新靶点 | Spearman / top-k enrichment |
| leave-batch-out | 留整批次 | 最严格，但靶点混淆强 | 作为压力测试 |
| Seahorse external | n=16 true Seahorse | 方向是否对真实功能读出有意义 | Spearman / rank agreement |

主结论只允许来自 **leave-target-out** 和 **group_plate**。random CV 只能放 supplement 或方法审计。

### 5.5 指标

| 指标 | 用途 | 备注 |
| --- | --- | --- |
| Spearman ρ | 主指标 | hit ranking 最重要 |
| Pearson r | 趋势指标 | 受 outlier 影响，辅助 |
| MAE | 绝对误差 | 不能单独解释 |
| top-k enrichment | top predicted 是否富集真实 strong phenotype | 最贴近 hit calling |
| AUROC / AUPRC | 若把 hit class 二值化 | 注意类别不平衡 |
| calibration plot | 预测强度是否回归 NTC | vAssay 已有压缩问题，必须监控 |
| bootstrap CI | 小样本不确定性 | 每个靶点给置信区间 |

### 5.6 Aim 2 成功标准

最低可接受：

- Expression + BF 在 leave-target-out 的 Spearman 明显高于 expression-only 和 mean baseline。
- 对 per-mito ΔΨm 与 mito mass 至少一条轴达到稳定可重复预测。
- top 10 predicted hits 中富集 Aim 1 的 strong phenotype / low toxicity hits。

理想结果：

- 多模态模型同时提升 per-mito ΔΨm 和 mito mass 两条轴。
- C1/BF 在新靶点泛化上优于 C24 或至少提供互补信息。
- 模型在 n=16 Seahorse 外部验证上方向一致，即使绝对值不准。

失败也有价值：

- 如果多模态不能提升，说明当前数据中功能表型主要由 post-transcriptional / imaging-only 状态决定；这仍可支持 Aim 1 生物学图谱和 C 路线的 virtual assay validation story。

---

## 6. Aim 3：机制解释与候选靶点优先级

### 6.1 转录组 MoA 分析

对每个靶点建立批内 NTC-relative signature：

- log fold-change / moderated effect size。
- 不建议一开始做复杂 DESeq2；孔级 mini-bulk + 小 n 可先用 limma-like linear model / ridge-stabilized effect。
- 对 strong phenotype hit 单独做 pathway enrichment。

推荐 gene set：

- OXPHOS / ETC complex I–V。
- Mitochondrial biogenesis / PGC1A / NRF1 / TFAM。
- FAO / lipid metabolism。
- AMPK / mTOR / insulin signaling。
- ER stress / ISR / unfolded protein response。
- Proteostasis / autophagy / mitophagy。
- Apoptosis / cytotoxicity。

### 6.2 Reference signature matching

以阳性对照作为机制锚点：

| Reference | 预期 signature | 用途 |
| --- | --- | --- |
| BAM15 | uncoupling / ΔΨm collapse | uncoupler-like 转录组相似性 |
| MK8722 | AMPK / mitochondrial mass up | biogenesis-like 转录组相似性 |
| ATP5B / SLC25A4 | ETC / mitochondrial stress | 线粒体功能扰动参照 |
| PSMC3 | toxicity / proteasome stress | toxic-collapse 负面参照 |

输出每个 target 的 connectivity score：

```text
moa_score[target, reference] = correlation(signature_target, signature_reference)
```

### 6.3 候选分层

| 档位 | 定义 | 后续动作 |
| --- | --- | --- |
| Tier 1 | 表型强、KD strong/weak 可信、毒性低、MoA 支持、多模态模型也预测高 | Seahorse / independent siRNA 优先验证 |
| Tier 2 | 表型强但 KD weak 或 MoA 不清 | 二次 KD 或 CRISPRi 前先查试剂 |
| Tier 3 | 模型预测高但当前表型弱 | 可能是假阴性或 KD 不足，适合 rescue/更强 KD |
| Deprioritized | toxic、failed KD、QC 不稳或 MoA 指向非特异应激 | 暂不推进 |

### 6.4 最终输出字段

每个候选 target 输出：

- target symbol。
- hit class。
- consensus score。
- per-mito ΔΨm effect。
- mito mass effect。
- TMRM area/intensity effect。
- vAssay AUC/MB rank。
- KD tier / kd_frac_drop。
- tox_rate / qc_fail_rate。
- BAM15/MK8722/PSMC3 signature similarity。
- key enriched pathways。
- model predicted phenotype and uncertainty。
- final recommendation。

---

## 7. 分阶段实施计划

### Phase 1 — 机制图谱和透明 hit calling

目标：把 Aim 1 做完整，先形成可信生物学图谱。

任务：

1. 从 `targets_summary.csv` 重建 target-level phenotype matrix。
2. 加入 bootstrap CI / split-half consistency。
3. 生成 hit class 和 consensus_score。
4. 画二维机制状态图、top-hit waterfall、toxicity overlay。
5. 输出 `B_A_candidate_priority.csv` 和一页 summary。

Go / no-go：

- 如果 top hits 大多 toxic 或 failed KD，先调整分数和过滤策略。
- 如果 BAM15/MK8722 锚点不能稳定分离，停止进入模型阶段，回查标准化。

### Phase 2 — 单模态和简单融合基线

目标：建立可辩护的 baseline ladder。

任务：

1. Expression-only Ridge/PLS。
2. C1-only Ridge。
3. C24-only Ridge。
4. Expression + C1 late fusion。
5. Expression + C1 + C24 upper-bound fusion。
6. 全部使用 group_plate 和 leave-target-out。

Go / no-go：

- 如果所有模型都不优于 mean baseline，转向 Aim 1 + C 路线，不强行做 vCell。
- 如果 Expression + C1 有增益，进入 Phase 3。

### Phase 3 — vCell-style 多任务模型

目标：训练轻量 virtual-cell 模型，同时预测表达重构和线粒体表型。

任务：

1. 基于 `drugseq_vcell.npz` 建立 NTC control → perturbation latent 的训练脚本。
2. 增加 phenotype heads：permito、mitomass、area、intensity。
3. 加入 modality heads 或 late-fusion embedding。
4. 与 Phase 2 OOF 结果同口径比较。
5. 输出 per-target predicted phenotype + uncertainty。

Go / no-go：

- 如果 vCell 不超过 late-fusion baseline，仍保留为 negative result，不作为主结论。
- 如果 vCell 提升 top-k 或机制分类，作为主模型。

### Phase 4 — MoA 和候选优先级整合

目标：把模型预测、机制图谱和转录组解释合成 candidate package。

任务：

1. 计算每个 target 的 reference signature matching。
2. 做 pathway enrichment / gene set score。
3. 生成 final priority table。
4. 挑选 Tier 1 / Tier 2 / model-rescue candidates。
5. 写 internal report / slide deck。

### Phase 5 — 可选前瞻验证设计

目标：为 wet-lab 申请/讨论准备一套最小验证 panel。

建议 panel：12–24 个 target。

| 类别 | 数量 | 选择原则 |
| --- | --- | --- |
| strong uncoupler-like | 4–6 | per-mito ΔΨm 强下降、毒性低 |
| strong biogenesis-like | 4–6 | mito mass 强上升、per-mito 不塌缩 |
| model-high / phenotype-moderate | 2–4 | 模型预测高，可能当前 KD 不足 |
| phenotype-high / model-low | 2–4 | 模型盲点，用于检验泛化边界 |
| negative controls | 2–4 | neutral 或 failed KD |

验证读出：

- independent siRNA 或 CRISPRi。
- Seahorse OCR/ECAR/AUC。
- TMRM + MitoTracker + cell count。
- viability / cell number。
- qPCR/DRUG-seq 小 panel 确认 KD。

---

## 8. 预期主图设计

### Figure 1 — Dataset and mitochondrial-state decomposition

- 实验设计示意：siRNA KD → DRUG-seq + imaging。
- 通道解释：BF、TMRM、nuclei、MitoTracker。
- BAM15/MK8722 展示 per-mito ΔΨm 与 mito mass 解耦。

### Figure 2 — EE mitochondrial state atlas

- target-level per-mito ΔΨm vs mito mass 图。
- hit class 标注。
- KD tier / toxicity overlay。
- top candidates 标注。

### Figure 3 — Multimodal prediction benchmark

- expression-only、C1-only、C24-only、fusion、vCell 的 LOTO Spearman。
- top-k enrichment。
- calibration / prediction compression check。

### Figure 4 — Mechanistic interpretation

- top hit × pathway heatmap。
- BAM15/MK8722/PSMC3 signature connectivity。
- selected candidates 的 per-target evidence strips。

### Figure 5 — Candidate prioritization and prospective validation panel

- consensus score waterfall。
- Tier 1/Tier 2 表格。
- proposed validation panel。

---

## 9. 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| KD 普遍偏弱 | 假阴性多，机制解释变弱 | KD tier 分层；failed KD 不作为阴性证据；设计 model-rescue group |
| 批次-靶点混淆 | 泛化评估困难 | 只用 within-NTC 标准化；主验证 group_plate + leave-target-out |
| C24 与 TMRM 表型接近 | 可能读出泄漏 | 主模型用 expression + BF；C24 作为 upper bound |
| Seahorse 真值少 | 无法直接训练真实功能模型 | Seahorse 仅作外部 sanity；主目标是 TMRM/MitoTracker 机制轴 |
| vCell 过拟合 | 模型故事不稳 | 先做 Ridge/PLS/late fusion；vCell 必须打败基线才主报 |
| 模态间相关弱 | 融合模型可能无提升 | 将其解释为互补层级；若融合无提升，主线转为机制图谱 + 可信验证框架 |

---

## 10. 最小可交付物

### 第一批交付

| 产物 | 建议文件 |
| --- | --- |
| target-level state atlas 表 | `output/2026-06-22/ba_multimodal_plan/B_A_state_atlas.csv` |
| candidate priority 表 | `output/2026-06-22/ba_multimodal_plan/B_A_candidate_priority.csv` |
| 机制图谱 | `output/2026-06-22/ba_multimodal_plan/figs/mitochondrial_state_atlas.png` |
| hit calling summary | `output/2026-06-22/ba_multimodal_plan/B_A_hit_calling_summary.md` |

### 第二批交付

| 产物 | 建议文件 |
| --- | --- |
| baseline benchmark | `output/2026-06-22/ba_multimodal_plan/multimodal_baseline_benchmark.csv` |
| OOF predictions | `output/2026-06-22/ba_multimodal_plan/multimodal_oof_predictions.csv` |
| model comparison 图 | `output/2026-06-22/ba_multimodal_plan/figs/model_benchmark.png` |
| final report | `output/2026-06-22/ba_multimodal_plan/B_A_multimodal_virtual_cell_report.md` |

---

## 11. 近期执行清单

### Step 1 — 先做 Aim 1，可快速收敛

- [ ] 写 `scripts/ba_state_atlas.py`：读取 `targets_summary.csv`，计算 bootstrap/split-half consistency、hit class、consensus score。
- [ ] 输出 state atlas + priority table。
- [ ] 画 per-mito ΔΨm vs mito mass 图。
- [ ] 人工 review top 30，检查 toxic / failed KD / known biology。

### Step 2 — 再做 Aim 2 baseline ladder

- [ ] 写 `scripts/ba_multimodal_benchmark.py`：expression/C1/C24/fusion 的 group_plate 与 leave-target-out CV。
- [ ] 统一输出 OOF prediction、Spearman、top-k enrichment。
- [ ] 比较 expression-only vs expression+BF vs expression+BF+C24。

### Step 3 — 决定是否进入 vCell extension

- [ ] 如果 fusion baseline 有稳定增益，添加轻量 phenotype heads。
- [ ] 如果没有增益，停止模型复杂化，把结果转成机制图谱 + validation framework。

### Step 4 — 写候选包

- [ ] 结合 MoA signature 和 pathway score。
- [ ] 形成 Tier 1/Tier 2/deprioritized 清单。
- [ ] 准备 wet-lab validation panel。

---

## 12. 推荐决策

建议把本课题主叙事定为：

> **先用 MitoTracker-resolved imaging 建立 EE 线粒体状态图谱，再用严格防泄漏的多模态 virtual-cell 模型测试这些状态是否能被扰动转录组和无标记形态预测。**

执行优先级：

1. **Aim 1 先行**：机制图谱和 hit calling 最稳，能快速产出候选靶点。
2. **Aim 2 以基线驱动**：先证明 expression+BF 或 fusion 确实有增益，再进入 vCell。
3. **Aim 3 收口**：所有结果最终落到可验证 target panel，而不是只停留在模型分数。

这条路线的学术价值在于把三个通常混在一起的问题分开：

1. 生物学上，TMRM 信号可拆成 per-mito ΔΨm 与 mitochondrial mass。
2. 方法学上，virtual assay 必须防 label leakage 和 domain shift。
3. 转化上，多模态模型的真正用途是候选靶点排序和机制解释，而不是宣称绝对预测 Seahorse 数值。
