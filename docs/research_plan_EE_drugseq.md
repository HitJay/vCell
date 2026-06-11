# 研究计划：HepG2 能量消耗（EE）siRNA 敲低 × DRUG-seq × TMRM 影像

- 文档版本：v4（2026-06-11）— 订正通道身份（**ch1 = 明场 BF、ch3 = 核荧光**）；新增 §12.4 实证 BF 特征泛化最优 + 三模态纳入 BF/线粒体双影像块
- 作者：Qiuye Jin (Jay) / NNRCC
- 数据位置（软链接）：`data/drug-seq/` → `../../vAssay_archieve/UHYG/drug-seq`
- 读取环境：`/data/user/QYJI/miniforge3/envs/scvi/bin/python`（`base` 环境无 `anndata`）
- 关联框架：本仓库 `vCell`（latent-additive conditional VAE，扰动响应建模）
- 状态：主线 D 数据地基已交付；影像 C1(BF)/C24(线粒体) 特征全 24 板齐全；三模态已孔级对齐（1440/1440）。下一步：主线 B / 多模态融合建模。

---

## 1. 背景与科学问题

**能量消耗（Energy Expenditure, EE）** 是代谢与肥胖治疗的核心可调节表型。增强细胞/线粒体的产热与氧化（解偶联、脂肪酸氧化、线粒体生物合成）可提高 EE，是减重靶点发现的重要方向。

本数据集在 **HepG2（人肝癌、肝代谢模型）** 中，对约 170 个 EE 相关候选基因做了 **siRNA 敲低（KD）**，并对每个孔同时采集两类读出：

1. **DRUG-seq**：孔级 mini-bulk 转录组（高通量、低成本的全基因组表达谱）。
2. **TMRM 高内涵影像**：TMRM 染料反映 **线粒体膜电位（ΔΨm）**，作为 EE / 线粒体功能的代理表型。

```mermaid
flowchart LR
    KD["siRNA 敲低<br/>(~170 EE 靶点 + 对照)"] --> Well["HepG2 孔<br/>(96-well, 4day)"]
    Well --> RNA["DRUG-seq<br/>表达谱 36601 基因"]
    Well --> IMG["TMRM Operetta 影像<br/>线粒体膜电位 ΔΨm"]
    RNA --> Q1["表达 → 表型预测 / MoA"]
    IMG --> Q2["EE 表型 hit calling"]
    Q1 --> OUT["EE 靶点优先级 + 机制解释"]
    Q2 --> OUT
```

**核心科学问题：**

1. 哪些基因 KD 能产生类似已知 EE 调节剂（解偶联 / AMPK 激活）的线粒体表型？（**hit calling**）
2. 孔级转录组能否预测、并机制性地解释 TMRM 影像表型？（**表达 → 表型**）
3. 每个靶点 KD 扰动了哪些代谢通路，能否构建 EE 靶点的功能图谱？（**MoA**）

---

## 2. 数据集描述

### 2.1 文件清单（`data/drug-seq/`）

| 文件 | 大小 | 内容 |
| --- | --- | --- |
| `adata.h5ad` | 31 MB | 主数据：**1440 孔 × 36601 基因**，原始 UMI counts（CSC 稀疏，min 1 / max 4174 / mean 33） |
| `adata_with_image_4features.h5ad` | 224 MB | 上述 + 4 个 TMRM 影像指标已并入 `obs` |
| `image_4features_aggregate.csv` | 150 KB | 孔级聚合后的 4 个影像指标 |
| `data_ingestion.py` | — | h5ad 勘验报告 CLI（`-f <file>`） |
| `process_image_aggregation.py` | — | 从 Operetta CSV 聚合影像指标并并入 adata |
| `demo.ipynb` | — | 简易演示（仅查看唯一值/列名） |

### 2.2 实验设计

- **样本单位**：孔（well）级 mini-bulk，**非单细胞**；共 **1440 孔**，全部 **4day** 时点。
- **批次**：5 个（`OFGM-0724` 360、`OFGM-1127` 360、`OFGM-0916` 360、`OFGM-0618` 240、`OFGM-1205` 120）。
- **板**：24 板 × 60 孔。
- **类别 `category`**：Target 1092 / 阳性对照 PC 204 / 阴性对照 NC 144。
- **扰动 `group`**：180 种。
  - `NTC` 144 孔（si-非靶向对照，**= vCell 的 control**）。
  - 阳性对照：`BAM15`（线粒体解偶联剂）、`MK8722`（AMPK 激活剂）、`ATP5B`、`SLC25A4`、`PSMC3` 等。
  - 约 170 个候选靶点 siRNA，每个约 6 个重复孔。

### 2.3 TMRM 影像 4 指标

均为 TMRM 通道（ch2，线粒体膜电位）相对明场通道（ch1，BF）的归一化比值（原始 `image_4features` 的归一化分母；ch1 身份详见 §12.4）：

| 指标 | 含义 | 所属轴 |
| --- | --- | --- |
| `ch2_ch1_intensity_area_ratio` | TMRM 强度 / ch1 面积 | **强度轴** |
| `ch2_ch1_area_area_ratio` | 高电位面积 / ch1 面积 | **高电位面积轴** |
| `ch2_intensity_cell_count_ratio` | TMRM 强度 / 细胞数 | 强度轴（**含毒性混杂**） |
| `ch2_area_cell_count_ratio` | 高电位面积 / 细胞数 | 高电位面积轴（**含毒性混杂**） |

> 孔位通过 `r02c02 → B02` 换算，把影像与转录组按 (`well`, `tmrm_operetta_data_file_name`) 配对。

### 2.4 `obs` 数据字典

| 列 | 类型 | 说明 |
| --- | --- | --- |
| `sample` / `sample_id` | str | 样本唯一 ID |
| `sample_name` | cat | 可读名（如 `siSEC16A_P1_1`） |
| `group` | cat | **扰动靶点 / 对照标签（建模关键列，= `pert_key`）** |
| `category` | cat | Target / PC / NC |
| `batch` / `batch2` / `batch_raw` | cat | 实验批次（**与靶点高度混淆，见 §3.1**） |
| `plate` / `plate_raw` | cat | 板编号 |
| `well` | cat | 孔位（连接影像） |
| `time` | cat | 时点（全部 4day） |
| `num_umis` | float | 文库大小（总 UMI），QC 用 |
| `num_features` | float | 检出基因数，QC 用 |
| `mt_percentage` | float | 线粒体基因比例，QC / 毒性判读用 |
| `tmrm_operetta_data_file_name` | cat | 对应 TMRM 影像板（连接影像） |
| `ch2_*`（4 列） | float | TMRM 影像指标（**仅在 `*_with_image` 文件**） |

---

## 3. 数据现状与关键约束（建模前必须正视的地基事实）

> 以下结论均来自对本数据集的实际勘验（2026-06-10）。它们直接决定课题设计的可行边界。

### 3.1 批次与靶点几乎完全混淆 → 必须"批内相对 NTC 标准化"

- 175 个 Target 靶点中 **98%（171 个）只出现在单一批次/板**。
- 因此 **绝不能跨批次直接比较原始值**；否则靶点效应与批次效应不可分。
- 救济点：**`NTC` 横跨全部 5 个批次**（每批 12–36 孔），`BAM15` / `MK8722` 也覆盖多批次。
- **设计原则（贯穿所有课题）**：任何比较都以 **批内（plate/batch 内）相对同批 NTC 的标准化**（plate-wise z-score 或 ratio-to-NTC）为基础。

### 3.2 影像 assay window 真实，且存在两条正交的 EE 轴

阳性对照相对 NTC 的 z 效应量（用 NTC 的标准差，|z|>2 视为强信号）：

| 对照 | n | intensity_area | area_area | intensity_cell_count | area_cell_count | 解读 |
| --- | --- | --- | --- | --- | --- | --- |
| **BAM15**（解偶联剂） | 72 | −0.15 | **−4.83** | +1.20 | **−4.77** | 高电位**面积塌缩**（ΔΨm↓，符合解偶联生物学） |
| **MK8722**（AMPK 激活） | 72 | **+4.50** | +1.95 | **+3.91** | +1.86 | TMRM **强度升高**（线粒体活性/生物合成↑） |
| ATP5B | 48 | −1.54 | −1.05 | +3.39 | +1.13 | 中等、方向混合 |
| SLC25A4 | 12 | −1.39 | −1.84 | −0.94 | −1.60 | 弱–中，趋于下降 |

- 原始均值对照（节选）：`area_area` 指标 NTC≈0.612、**BAM15≈0.067（大幅塌缩）**、MK8722≈0.832（升高）；`intensity_area` 指标 NTC≈0.063、**MK8722≈0.117（翻倍）**。
- **结论**：4 个指标可归为 **"强度轴"（MK8722 ↑ 锚定）** 与 **"高电位面积轴"（BAM15 ↓ 锚定）** 两条互补方向，构成可用于 EE hit calling 的二维表型空间。

### 3.3 KD 效率普遍偏弱 → KD-QC 必做

KD 孔自身靶基因 logCPM 相对 NTC 的下降（Δ）：

| 靶点 | nKD | Δ logCPM | 靶点 | nKD | Δ logCPM |
| --- | --- | --- | --- | --- | --- |
| EPS15 | 12 | −0.60 | DGAT2 | 6 | −0.37 |
| SLC25A4 | 12 | −0.44 | CKB | 6 | −0.32 |
| PSMC3 | 30 | −0.40 | GRB14 | 6 | −0.31 |
| HSPA4 | 6 | −0.37 | SIRT4 | 6 | −0.23 |
| PGM1 | 6 | −0.24 | SEC16A | 6 | −0.01 |
| **SHISA5** | 6 | **+0.20（未敲下）** | | | |

- 多数靶点仅下降约 20–45%，部分（SHISA5、SEC16A）几乎未见敲低。
- **含义**：阴性结果不可直接解释为"该基因无功能"，可能只是 KD 不足。
- **必做**：对每个靶点先做 KD 效率打分（批内相对 NTC），**按 KD 效率分层/过滤**后再做下游分析。
- 注意：`group` 标签命名与 `var['symbol']` 可能存在别名差异（例如 `ATP5B` 在 `symbol` 中查不到，疑为新命名 `ATP5F1B`），KD-QC 需先做**基因符号别名解析**。

### 3.4 毒性假信号 → 必须做毒性去卷积

- 以细胞数为分母的指标（`*_cell_count_ratio`）在细胞大量死亡时会异常放大：`PSMC3` 的 `intensity_cell_count` z≈**+61**，Target 类该指标均值出现 **`inf`**（部分孔细胞数→0）。
- **必做**：用 `num_umis` / `num_features` / `mt_percentage` / 细胞计数把 **"解偶联导致的 ΔΨm 下降"** 与 **"毒性死亡导致的下降"** 区分开；标记并隔离毒性孔。
- **优先采用不含细胞数分母的指标**（`ch2_ch1_*`）作为主表型，`*_cell_count_*` 仅作辅助 + 毒性 flag。
- **升级（见 §3.5）**：原始影像 CSV 的 **`cell_count` 绝对值已确认可得**，毒性去卷积无需再从 ratio 反推，可直接用真实细胞数 + `ch1_area`（汇合度）构建毒性分数。

### 3.5 原始影像数据可得性核查（2026-06-10，风险消除 + 重大正面发现）

> 主线 D 方案曾标注一个数据可得性风险：聚合脚本 [process_image_aggregation.py](../../vAssay_archieve/UHYG/drug-seq/process_image_aggregation.py) 只保留了 4 个 ratio、丢弃了 `cell_count`。经核查，**原始 CSV 完好无损，风险解除**，并发现额外数据资产。

- **路径**：`/NNRCC_Image/processed_data/UHYG/2025/<板名>/<板名>.csv`，**24/24 板全部存在**，16 列结构完全一致。
- **粒度更细**：原始为 **field（视野）级**，每板 540 行 = 60 孔 × 9 视野；之前的 `image_4features` 是 well-mean 聚合 → 现在可重做聚合并附带**孔内变异（SEM/CV）**用于 QC。
- **`cell_count` 直接可得**：min 3 / max 1201 / mean 534，无零值 → 毒性去卷积用真实细胞数，不必反推。
- **完整原始通道**：`ch1_intensity/area`（明场 BF）、`ch2_intensity/area`（TMRM ΔΨm）、`ch4_intensity/area`（MitoTracker）、`cell_count`（来自 ch3 核分割）→ 可自由重算任意 ratio，不受既往聚合限制。另有 `readout.csv`（干净 8 列原始读出）。
- **🔬 ch4 通道：MitoTracker（线粒体质量），已由实验方确认（2026-06-10）**。原始 CSV 含 `ch4_intensity/area` + 4 个 `ch4_ch1_*` ratio，100% 有信号，但既往 `image_4features` 聚合完全没用它。
  - 通道含义（2026-06-11 订正，见 §12.4）：**ch1 = 明场 brightfield（BF，无标记），ch2 = TMRM（膜电位 ΔΨm），ch3 = 核荧光（Hoechst，用于分割计数），ch4 = MitoTracker（线粒体质量）**。
  - **关键机制解析**：`ch2/ch1` 把"膜电位"和"线粒体数量"卷在一起；MitoTracker 通道把两者拆开：
    - **per-mito ΔΨm（偶联状态）= ch2/ch4** → 解偶联检测（BAM15 ↓）。
    - **线粒体质量/生物合成 = ch4/ch1** → 生物合成检测（MK8722 ↑）。
  - **实测验证（批内 NTC z）**：MK8722 在 `ch2/ch1` 上的巨幅 +20 信号，到 per-mito ΔΨm（ch2/ch4 强度）几乎消失（中位 +1.6），而线粒体质量轴 +8.3 → **MK8722 的"TMRM 升高"主要来自线粒体生物合成（数量变多），而非每个线粒体更带电**。BAM15 则在 per-mito ΔΨm 面积轴 −15（膜电位真实塌缩）。这是 MitoTracker 通道才能解开的机制。
  - 已纳入主线 D pipeline：新增 `pheno_permito_dpsi_z`（per-mito ΔΨm）与 `pheno_mitomass_z`（线粒体质量）两条机制 z 轴，hit 方向细分为 uncoupler_like / biogenesis_like / energizer_like（见 §9.7）。

---

## 4. 研究主线（课题设计与优先级）

执行顺序：**D（地基）→ B（hit calling）→ A（预测模型）→ C（MoA）→ E（整合）**。

### 主线 D —— 数据地基工程（最先，所有课题复用）

- **目标**：产出可复用的 **干净孔级表型矩阵 + KD 效率表 + 毒性 flag**，作为 B/A/C 的唯一可信输入。
- **方法（概要）**：表达侧 CPM+log1p / HVG / 批内相对 NTC 标准化；影像侧 field→well 聚合 + 批内 plate-wise z（BAM15/MK8722 锚定方向）；QC 含 KD 效率打分（符号别名解析）、毒性去卷积（真实 cell_count）、低质量孔标记。
- **产出**：`data/processed/`（处理后 h5ad + vCell-ready npz + 孔级/靶点级注释表）+ `output/` QC 报告。
- **风险**：批次校正过度去信号 → 用阳性对照效应量做"过校正"监控。
- **➡ 完整设计、6 阶段数据流、交付物清单与默认执行参数见 [§9 主线 D 详细设计](#9-主线-d-详细设计数据地基)。**

### 主线 B —— EE hit calling（最快出可跟进结论）

- **目标**：找出表型上"类 BAM15（解偶联）"或"类 MK8722（产能增强）"的靶点。
- **方法**：
  1. 影像端：在 §3.2 的二维表型空间里，对每个靶点计算批内相对 NTC 的效应量与方向，毒性孔过滤。
  2. 转录组端（MoA matching）：每个靶点的差异表达 signature 与 BAM15/MK8722 signature 求连接度相似性。
  3. 共识：影像 + 转录组两端取交集，输出共识 hit。
- **产出**：排序的候选靶点表（效应量、方向、KD 效率、毒性 flag、转录组一致性）。

### 主线 A —— 转录组 → TMRM 表型预测（vCell / perturb-seq→TMRM 落地）

- **目标**：用孔级表达预测 4 个 TMRM 指标，并反推驱动基因/通路（呼应既往 perturb-seq→TMRM R²≈0.75 的工作）。
- **方法**：批内标准化后，表达（HVG / OXPHOS·产热·FAO 通路评分）→ 回归 / TabPFN 预测影像表型；SHAP / 通路权重做可解释性。
- **产出**：一个"表达 → 线粒体表型"预测器（让便宜的转录组充当贵影像的代理）+ 驱动基因列表。
- **与 vCell 的结合**：见 §5。

### 主线 C —— 转录组 MoA 与靶点功能图谱

- **目标**：刻画每个靶点 KD 扰动的通路；将 180 个扰动聚成功能模块。
- **方法**：N≈6 重复的 pseudobulk 差异表达（DESeq2 / limma-voom，批内对 NTC）→ OXPHOS / 线粒体生物合成 / FAO / 产热 / ISR 通路评分 → 靶点按"转录组 + 影像"联合聚类。
- **产出**：靶点–通路热图 + 功能模块划分，解释 hit 的机制类别。

### 主线 E —— 跨数据整合与安全性注释（加分）

- 与既往 **EE/TMRM image-only 筛选（约 300 靶点，DINOv2+TabPFN R²≈0.76）** 交叉验证 hit 可重复性。
- 多模态融合（表达 + 影像）预测 EE 表型。
- hit 靶点接 **安全性 landscape（RIC-349）** 做成药性/安全注释，输出优先级清单。

---

## 5. 与 vCell 框架的结合

`vCell` 是 latent-additive conditional VAE：把表达谱编码为 basal 潜变量，每个扰动是潜空间中的一个学习向量，解码重建扰动后表达，可回答反事实问题（"这些 control 孔在扰动 k 下会是什么样"）。

**接入方式：**

- `data.data_path = data/drug-seq/adata.h5ad`（或带影像版本）。
- `data.pert_key = group`（drug-seq 的扰动列）。
- **control 映射**：vCell 约定 perturbation id `0` = control；需把 `NTC` 映射为 control。**[待确认]** `PerturbationDataset` 对 `.h5ad` 字符串标签的 control 识别逻辑。

**反事实用法（服务主线 A）**：用 vCell 预测某靶点 KD 下的表达谱 → 接表达→表型预测器，实现"虚拟 KD → 预测 TMRM 表型"，可用于对未做实验的靶点做零样本预测。

**必须正视的技术约束（诚实评估）：**

1. **样本量小**：vCell 原设计针对单细胞（每扰动数百细胞），本数据是孔级 bulk，仅 1440 样本、每扰动约 6 重复 → VAE 易过拟合。对策：更小模型 / 更强正则 / 先用线性 latent-additive 基线打底。
2. **raw counts vs MSE 解码**：vCell 默认 Gaussian(MSE) 重构，适合 log-normalized 表达；本数据是原始 counts（max 4174）→ 需先 normalize+log1p，或采用 roadmap 中的 **负二项（NB）解码器**。
3. **批次混淆**：§3.1 的批次效应需在建模前/中处理（批内标准化或把 batch 作为协变量），否则潜空间会编码批次而非生物学。

---

## 6. 路线图（里程碑，不含时间承诺）

- **M1 — 数据地基**：标准化表达矩阵 + 4 维表型 z 矩阵 + KD 效率表 + 毒性 flag（主线 D）。
- **M2 — EE hit list**：影像 + 转录组共识的候选靶点排序表（主线 B）。
- **M3 — 表达→表型预测器**：可复用模型 + 驱动基因/通路（主线 A）。
- **M4 — MoA 图谱**：靶点–通路热图与功能模块（主线 C）。
- **M5 — 整合与安全注释**：跨数据验证 + 成药性/安全优先级（主线 E）。

---

## 7. 准备工作清单

- [x] 在 `data/drug-seq` 建立指向数据文件夹的软链接（相对路径，已加入 `.gitignore`）。
- [x] 勘验数据：维度、设计平衡性、assay window、KD 效率、毒性信号（本文档 §2–§3）。
- [x] 原始影像 CSV 可得性核查：24/24 板全在、含 `cell_count` 与 ch4、field 级粒度（§3.5）。
- [x] 确认读取/分析环境：`scvi` env 已具备 `anndata`（`/data/user/QYJI/miniforge3/envs/scvi/bin/python`）。
- [x] 基因符号别名解析表（`group` 标签 ↔ `var['symbol']`）—— 主线 D 阶段 0 产出（`gene_symbol_map.csv`；仅 `ATP5B→ATP5F1B` 1 条别名）。
- [x] 主线 D 管线脚本：批内标准化 + KD-QC + 毒性去卷积（已实现并跑通，见 §9 与 §9.7）。
- [x] **ch4 影像通道染料身份已确认 = MitoTracker**（线粒体质量）；已纳入 per-mito ΔΨm / 线粒体质量两条机制轴（§12.4）。

---

## 8. 数据访问片段（参考）

```bash
# 读取环境（base 无 anndata）
PY=/data/user/QYJI/miniforge3/envs/scvi/bin/python
```

```python
import anndata as ad

# 通过软链接访问（工作区相对路径）
adata = ad.read_h5ad("data/drug-seq/adata.h5ad")                      # 表达（raw counts）
adata_img = ad.read_h5ad("data/drug-seq/adata_with_image_4features.h5ad")  # 表达 + 4 影像指标

# 关键列
#   adata.obs['group']     -> 扰动标签（NTC = control）
#   adata.obs['category']  -> Target / PC / NC
#   adata.obs['batch']     -> 批次（与靶点高度混淆，须批内标准化）
#   adata_img.obs[['ch2_ch1_intensity_area_ratio',
#                  'ch2_ch1_area_area_ratio',
#                  'ch2_intensity_cell_count_ratio',
#                  'ch2_area_cell_count_ratio']]  -> TMRM 表型
```

---

*本文档为活文档（living document），随分析进展更新版本号与里程碑状态。*

---

## 9. 主线 D 详细设计（数据地基）

> 本节是主线 D 的可交付实施方案。用户已拍板"按默认执行"，默认参数见 §9.6。

### 9.0 定位
产出一套**批次校正 + QC + 毒性去卷积后的干净孔级数据集 + 全套注释表**，作为 B/A/C 三条下游课题的**唯一可信输入（single source of truth）**。

### 9.1 数据流（6 阶段，模块化）

```mermaid
flowchart TD
    IN1["adata.h5ad<br/>raw counts 1440×36601"] --> S0
    IN2["24 板原始 Operetta CSV<br/>field 级, 含 cell_count + ch4"] --> S0
    S0["阶段0 加载/对齐<br/>+ 基因符号别名解析"] --> S1
    S1["阶段1 表达QC<br/>批内离群孔检测"] --> S2
    S2["阶段2 归一化+批次处理<br/>CPM→log1p→批内 z / ratio-to-NTC"] --> S3
    S3["阶段3 KD效率打分<br/>strong/weak/failed 分层"] --> S4
    S4["阶段4 影像表型+毒性去卷积<br/>field→well, 2轴z表型 + 真实cell_count毒性"] --> S5
    S5["阶段5 打包交付物 + QC报告"]
```

| 阶段 | 做什么 | 关键点 |
| --- | --- | --- |
| **0 加载/对齐** | 读 adata + 24 板 CSV；well 对齐校验；建 `group↔symbol` 别名表 | `ATP5B→ATP5F1B` 类别名必须解决（KD-QC 依赖） |
| **1 表达 QC** | `num_umis/num_features/mt_percentage` 批内 MAD/分位数离群检测 | 只标记不删（默认） |
| **2 归一化+批次** | CPM→log1p→HVG（线粒体/OXPHOS/产热/FAO 强制保留）→批内相对 NTC z | 过校正监控：BAM15/MK8722 signature 保留 |
| **3 KD 打分** | 靶点自身基因批内相对 NTC 的 ΔlogCPM → strong/weak/failed | SHISA5/SEC16A 应落 failed |
| **4 影像表型+毒性** | field→well 聚合（带 SEM/CV）；批内 z 二轴表型；**真实 cell_count + ch1_area** 建毒性分数 | 主表型用 `ch2_ch1_*`；ch4=MitoTracker 机制轴已纳入（§12.4） |
| **5 打包** | 写 h5ad + npz + 3 张注释表 + QC 报告 | 见 §9.2 |

### 9.2 交付物清单

**代码**（新增，不动现有 vCell 模型代码）
- `src/vcell/data/drugseq.py` — 6 阶段函数化 pipeline（可单测）
- `scripts/prep_drugseq.py` — CLI 入口
- `configs/drugseq_prep.yaml` — 阈值/路径/方法开关
- `tests/test_drugseq_prep.py` — 冒烟测试（小子集 + 阳性对照 sanity）

**数据产物**（`data/processed/`，已纳入 .gitignore）
| 文件 | 内容 |
| --- | --- |
| `adata_drugseq_processed.h5ad` | 主产物：layers=`counts/lognorm/zscore`，obs 带全部注释 |
| `drugseq_vcell.npz` | vCell 直接可吃：`X`(lognorm)、`pert`、`control_index`(NTC→0 已固化)、`num_perturbations`、meta |
| `wells_annotation.csv` | 孔级（1440 行）：QC flag、KD 分数/层、毒性 flag、二轴表型 z |
| `targets_summary.csv` | 靶点级（~180 行）：n 重复、KD 层、表型效应量+方向、毒性比例、hit 候选标记 |
| `gene_symbol_map.csv` | `group`↔`symbol` 别名映射 |

**报告**（`output/2026-06-10/`）
- QC 报告（图+md）：PCA/UMAP 着色 batch vs group、阳性对照 window、KD 分层分布、毒性分布、过校正监控。

### 9.3 验证标准（怎么证明地基是对的）
- **阳性对照 sanity**：校正后 BAM15 仍落高电位面积轴↓、MK8722 仍落强度轴↑（设 z 阈值断言）。
- **批次效应下降**：校正后 PCA 上 batch 解释方差下降、NTC 跨批次聚拢。
- **KD-QC**：报告 strong KD 靶点占比。
- **毒性**：cell_count 爆炸孔被正确隔离、计数合理。
- **可重复**：固定 seed + pytest 冒烟测试通过。

### 9.4 与 vCell 对接（必须处理的 2 点）
1. **control 映射**：交付走 `npz` 路线，直接把 `control_index` 固化为 NTC→0，**绕过** [load_h5ad](../src/vcell/data/dataset.py#L116) 默认 `control_label="control"` 与 NTC 不匹配的问题。
2. **归一化**：交付的 `X` 已是 log-normalized，匹配 vCell 默认 MSE 解码，避免 raw counts 爆炸。

### 9.5 依赖与环境
- 主路径仅需 `anndata/numpy/pandas/scipy/scikit-learn`（`scvi` env 已备）。
- scVI/Harmony 为**可选**高级分支，默认不启用（见 9.6-A）。

### 9.6 默认执行参数（本次采用）
| 决策点 | 默认选择 |
| --- | --- |
| A 批次校正 | **轻量"批内 ratio-to-NTC / z-score"**（可解释、快）；scVI 仅预留接口，不默认跑 |
| B 低质量/毒性孔 | **只标记不删**，下游按 flag 自行过滤 |
| C 交付格式 | **h5ad + npz 都给** |
| D 符号别名 | **本地启发式 + 手工补已知别名**（不依赖联网） |
| E vCell 核心改动 | 默认**不改**；走 npz 绕过 control_label（若后续需 h5ad 路线再加配置项） |
| ch4 通道 | **已确认 = MitoTracker**；纳入 per-mito ΔΨm（ch2/ch4）+ 线粒体质量（ch4/ch1）两条机制轴 |

### 9.7 执行结果（2026-06-10，已落地）

代码：[src/vcell/data/drugseq.py](../src/vcell/data/drugseq.py)、[scripts/prep_drugseq.py](../scripts/prep_drugseq.py)、[scripts/report_drugseq.py](../scripts/report_drugseq.py)、[configs/drugseq_prep.yaml](../configs/drugseq_prep.yaml)、[tests/test_drugseq_prep.py](../tests/test_drugseq_prep.py)。

运行：`python scripts/prep_drugseq.py --config configs/drugseq_prep.yaml` → `python scripts/report_drugseq.py`（用 `scvi` env）。全量测试 22 passed。

关键结果：
- 1440 孔 × 36601 基因，HVG **2037**（含 EE 通路基因强制保留）。
- **批次校正有效**：PCA 上原始 log-norm 的 plate 聚簇（尤其 OFGM-1205 完全分离）在批内 NTC z 后消失（图：`output/2026-06-10/figs/pca_batch_correction.png`）。
- **阳性对照方向正确**：BAM15 `pheno_area_z≈-27`（解偶联→面积轴↓）、MK8722 `pheno_intensity_z≈+23`（AMPK→强度轴↑）。
- **MitoTracker 机制轴（ch4 确认后纳入）**：BAM15 `pheno_permito_dpsi_z≈-19`（per-mito ΔΨm 塌缩）；MK8722 `pheno_mitomass_z≈+13`（线粒体质量↑）而 per-mito 仅微升 → 揭示 MK8722 是**生物合成驱动**，非每个线粒体更带电（图：`output/2026-06-10/figs/mechanism_axes.png`）。
- **KD 分层**：strong 46 / weak 96 / failed 28 / unknown 7；SHISA5、SEC16A 落 failed（与勘验一致）。
- **毒性**：判据改为 `cell_count < 0.3×同批 NTC median`（损失>70%），标记 **51 孔（4%）**；PSMC3（蛋白酶体，median 0.21）被正确识别，BAM15/MK8722（median≈1.0）正确放过。
- **EE hit 初筛 113**，按机制细分 **uncoupler_like 94 / biogenesis_like 18（含 SIRT4、NDUFAF1 等）/ energizer_like 1**（交给主线 B 精细化）。

交付物（`data/processed/`，均已 git 忽略）：`adata_drugseq_processed.h5ad`、`drugseq_vcell.npz`（NTC→0、log-norm、vCell 可直接加载，已验证）、`wells_annotation.csv`、`targets_summary.csv`、`gene_symbol_map.csv`、`prep_summary.json`；QC 报告 `output/2026-06-10/QC_report_drugseq.md` + 4 图（含机制轴 `mechanism_axes.png`）。

---

## 10. 原始图像位置与影像特征提取准备（2026-06-10）

> 为后续用**视觉基础模型（DINOv2 等）提取影像特征**做准备，已系统盘点原始图像位置并生成机读 manifest。

### 10.1 原始图像位置

- **根目录**：`/NNRCC_Image/processed_data/UHYG/2025/<板名>/`，24/24 板齐全。
- **每板内容**：
  - `projection/` — 投影后 16-bit **TIFF**（4 通道 × 540 视野 = 2160 张/板）。
  - `jpg/` — 8-bit **JPG** 预览（同样 2160 张/板）。
  - `<板名>.csv` / `readout.csv` — field 级 CellProfiler 风格读出（含 `cell_count`、ch1/2/4 强度面积）。
  - `csv/` — **已有 DINOv2 预提取特征**（384 维）+ vAssay readout 预测（`pred_MB` / `pred_AUC`）。
- **规模**：**51,840 张 TIFF（+ 51,840 JPG）** = 24 板 × 60 孔 × 9 视野 × 4 通道，已逐一 stat 验证**零缺失**。

### 10.2 通道含义（4 通道）

| 通道 | 染料 / 含义 | 用途 |
| --- | --- | --- |
| ch1 | 明场 brightfield（BF，无标记） | 形态学特征、归一化分母 |
| ch2 | **TMRM**（线粒体膜电位 ΔΨm） | EE 主表型 |
| ch3 | 核荧光（Hoechst/DAPI） | Cellpose 核分割 + 计数 |
| ch4 | **MitoTracker**（线粒体质量） | per-mito ΔΨm 归一化、生物合成轴 |

### 10.3 ⚠ 两种 TIFF 命名风格（已自动处理）

跨板存在两种 TIFF 命名，manifest 脚本**逐板自动探测**：

- `A_hyphen`（仅 `UHYG_20250411_1` 1 板）：`r02-c02-f01-ch2-01.tiff`
- `B_operetta`（其余 23 板）：`r02c02f01p01-ch2sk1fk1fl1.tiff`
- JPG 全部为 Operetta 风格：`r02c02f01p01-ch2sk1fk1fl1.jpg`

### 10.4 图像 manifest（机读，供视觉模型批量读取）

- 脚本：[scripts/inventory_images.py](../scripts/inventory_images.py)（`--check-exists` 验证全部路径）。
- 产物（`data/processed/`，已 git 忽略）：
  - `image_manifest.csv` — **51,840 行**，列：`plate_batch / image_plate / tiff_style / group / category / well / field / channel / dye / tiff_path / jpg_path / tiff_exists`。每行一张图，已关联到扰动靶点 `group` 与孔 `well`。
  - `image_manifest_plate_summary.csv` — 每板可用性汇总（24/24 complete）。
- 用法：按 `well`/`group` 直接对接 `wells_annotation.csv`、`drugseq_vcell.npz`，实现**影像特征 ↔ 转录组 ↔ TMRM 表型**三模态对齐。

### 10.5 现有 vAssay 影像 pipeline（DINOv2 → Seahorse 预测）

现有成熟管线在 [/das/user/QYJI/1_Pipeline](/das/user/QYJI/1_Pipeline)（git repo）：

```
jpg → 3_1_DINOv2_small.py (facebook/dinov2-small, 每通道 384 维)
    → DINO2_features.csv (2160 行 = 540 视野 × 4 通道, 带 Channel 列)
    → 4_FeatureAggregation.py (按 channel=[..] 选通道 + 孔级均值)
    → *_C<通道>_ID_aggre.csv (60 孔 × 384 维)
    → 5_vAssay_Prediction.py (TabPFN 模型)
    → vAssay_readout_C<x>.csv 加 pred_MB, pred_AUC
```

- **⚠ 纠正：`C` 后的数字 = 输入通道编号组合（不是 DINOv2 checkpoint 版本）**。已由数据级验证（C24_aggre = 通道 (2,4) 的 DINOv2 特征均值，误差 8.88e-16）：

| 标记 | 输入通道 | 含义 |
| --- | --- | --- |
| **C1** | ch1 | 明场 BF（无标记，泛化建模最佳输入，见 §12.4） |
| **C12** | ch1+ch2 | 核 + TMRM(ΔΨm) |
| **C14** | ch1+ch4 | 核 + MitoTracker(质量) |
| **C24** | ch2+ch4 | **TMRM + MitoTracker（最贴 EE/线粒体）** |

- **预测目标 = Seahorse**：`pred_MB`（Maximal Breath 最大呼吸）+ `pred_AUC`；模型为 **TabPFN**（`weights/tabpfn_{MB,AUC}_{C1,C24}_v1.pkl`）。
- 当前投产只剩 **C1 + C24** 两套模型（C12/C14 在 `weights_old/`）→ 这解释了 csv 盘点中 C1（24 板）、C24（19 板）覆盖最全。
- **通道覆盖**：C1 全 24 板 / C24 19 板 / C14 7 板 / C12 3 板。原始图像（51,840 张）全齐 → 可用统一视觉模型（或 `3_2_DINOv2_giant.py`）重提全板保证可比。这是主线 A/E（影像特征 → EE 表型 / 多模态融合）的输入准备。

### 10.6 C24 特征补齐（2026-06-10，全 24 板齐全）

5 个板（`UHYG_20250804_1/2/3`、`UHYG_20250908_1`、`UHYG_20250915_1`）原缺 C24 聚合 + 预测，但**最贵的原始 DINOv2 特征（`DINO2_features.csv`）全 24 板都在** → 只需补两步轻量操作（通道 [2,4] 聚合 + TabPFN 预测），无需 GPU 重提。

- 脚本：[/das/user/QYJI/1_Pipeline/backfill_C24.py](/das/user/QYJI/1_Pipeline/backfill_C24.py)（复刻 `4_FeatureAggregation.py` + `5_vAssay_Prediction.py`）。
- 运行环境：**cp3 env + 必须 `SCIPY_ARRAY_API=1`**（tabpfn 2.1.0 的硬性要求）：
  ```bash
  cd /das/user/QYJI/1_Pipeline
  SCIPY_ARRAY_API=1 /data/user/QYJI/miniforge3/envs/cp3/bin/python backfill_C24.py
  ```
- **验证（多角度全过）**：
  - 聚合逻辑：特征未变的板（0624_1）我的聚合 vs 官方 = **1.78e-15**（完全一致）。
  - TabPFN 确定性：同输入两次预测差 = **0.00**（无随机性）。
  - 模型可复现：官方 aggre 喂当前模型，19 板 pred 误差 MB <1.2% / AUC <0.6%（系统性小差异 = 官方历史用稍早模型版本）。
  - 旁证发现：部分旧板的 DINOv2 特征后被重提过（0714_3 官方特征 0.7599 ≠ 现 raw 0.7386）→ 官方旧 readout 用旧特征；补齐的 5 板用当前最新特征，自洽。
- 结果：**24/24 板 C24 齐全**，格式一致（387 列 × 60 孔），pred_MB 均值 [1.33, 1.47] 落在原有板 [1.32, 1.52] 内。
- **注意**：C1（核形态）与 C24（线粒体）readout 孔级/靶点级相关性都不高（靶点级 pred_MB Pearson −0.07），阳性对照方向常相反 → 二者捕捉**互补非冗余**信息，EE/线粒体分析须用 **C24**，勿假设不同通道预测一致。

---

## 11. Seahorse 金标准验证（2026-06-10）— vAssay 真实准确度

> 部分 target 同时做了**真实 Seahorse 测量**，构成 vAssay（C24 影像 → TabPFN）的金标准对照。这是评估"准确度"的硬指标。

### 11.1 数据来源与口径

- 真值来源：`image.png`（部分 target 测了真实 Seahorse），已逐行转录为 [data/seahorse_vAssay_validation.csv](../data/seahorse_vAssay_validation.csv)（git 跟踪，手工金标准源数据，非脚本产物）。
- 字段：`target / seahorse_assay_date / seahorse_AUC_value / _pct / vassay_image_date / vassay_AUC_value / _pct / needs_repeat / notes`。
- **口径**：`_pct` = 相对**同板 NTC** 归一化（NTC=1），是跨板可比的正确指标；`vassay_AUC_value` 即 pipeline 的 `pred_AUC`（NTC 112.55 / PSMC3 57.69 吻合，确认数据链路一致）。
- `0530_Plate3 needs repeat`（影像质量存疑）已用 `needs_repeat` 标记。

### 11.2 准确度（target 级，vAssay vs 真实 Seahorse AUC%）

| 版本 | n | Pearson | Spearman | MAE |
| --- | --- | --- | --- | --- |
| 历史 vAssay（图中值） | 12 | 0.76 | 0.35 | 0.18 |
| **当前补齐 vAssay** | **16** | **0.77** | **0.55** | 0.20 |

- 脚本：[scripts/validate_seahorse.py](../scripts/validate_seahorse.py)（`scvi` env）；产物 `output/2026-06-10/figs/seahorse_validation.png` + `seahorse_validation_targets.csv`。
- 当前补齐版**排序能力更好**（Spearman 0.35 → 0.55），且多覆盖 4 个图中未填值的 target（RREB1 / SLC12A8 / UBE2D4 / ZC3H7B）——这是补齐 5 板的直接价值。

### 11.3 关键结论（诚实评估）

- **趋势 / 排序正确**：PSMC3（真值最低 0.31）vAssay 也预测最低；CEP68 / RREB1（真值高）也预测高。Pearson 0.77 与既往 vAssay R²≈0.75 量级吻合 → C24 影像确实承载 Seahorse 信息。
- **系统性高估（回归到 NTC）**：散点图所有点都在 y=x 上方，预测整体被压缩贴近 1.0（SEC16A 真值 0.60 但预测 0.999），对**强 EE 抑制靶不够敏感**，当前版压缩更明显。
- **下游指导**：用 `vAssay_AUC` 做 EE hit 排序时**信相对排序（Spearman），勿信绝对预测值**。这也再次印证 C24（线粒体通道）对 Seahorse 有效，C1 对不上。

---

## 12. 三模态对齐（2026-06-10）

> 把**转录组 × C24 影像 × TMRM 机制轴表型**在孔级统一对齐成一个 AnnData，供下游多模态建模。

### 12.1 对齐结构与覆盖

- 对齐键：`(image_plate, well)`，**1440/1440 孔三模态 100% 对齐，零缺失**。
- 代码：[src/vcell/data/multimodal.py](../src/vcell/data/multimodal.py) + [scripts/align_multimodal.py](../scripts/align_multimodal.py)；产物 `data/processed/adata_multimodal.h5ad`（git 忽略）。

| 模态 | 存放位置 | 内容 |
| --- | --- | --- |
| 转录组 | `layers['counts'/'lognorm']`，`obsm['X_lognorm_hvg'/'X_zscore_hvg']` | 36601 基因 / 2037 HVG |
| C24 影像 | `obsm['X_dino_c24']`（384 维）+ `obs['vassay_pred_MB'/'vassay_pred_AUC']` | DINOv2 特征 + Seahorse 预测 |
| TMRM 表型 | `obs['pheno_*_z']` + `kd_tier` + `tox_flag` | 4 条机制轴 + KD + 毒性 |

### 12.2 一致性分析（关键发现）

脚本 [scripts/analyze_multimodal.py](../scripts/analyze_multimodal.py)；图 `output/2026-06-10/figs/multimodal_consistency.png`。靶点级、批内 NTC 标准化、排除毒性孔。

- **每个模态内部信号真实可重复**（split-half reliability）：TMRM 机制轴 r 0.75–0.88、Seahorse pred_MB 0.87、pred_AUC 0.60 —— 都很高。**唯一例外**：OXPHOS 转录组评分 r 仅 **0.12**（孔级 bulk 转录组单基因评分噪声大，靶点信号弱）。
- **但模态之间几乎不相关**（靶点级 Pearson 多在 ±0.25 内）：permito ΔΨm ↔ pred_AUC −0.24、mitomass ↔ pred_MB +0.23、OXPHOS ↔ pred_AUC +0.16。
- **TMRM 内部高度自洽**（permito ↔ area 0.73、intensity ↔ area 0.63），说明低跨模态相关不是噪声所致。

### 12.3 结论与建模指引

- **这不是对齐错误**，而是真实的数据结构：三个模态各自可重复，但测量**互补的生物学层面**（mRNA 表达 ≠ 膜电位 ≠ Seahorse 呼吸）。
- **多模态的价值在"互补信息"而非"冗余验证"** → 下游应做**融合建模**（如转录组 + 影像联合预测 EE），而非期待模态间相互印证。
- OXPHOS 单基因评分信号弱 → 转录组侧应改用**通路评分 / 差异表达 signature**（主线 C）而非朴素基因均值。

### 12.4 ⚠ 通道身份订正 + 明场（BF）特征更适合建模

**通道身份订正**（2026-06-11，据 [2_tmrm.py](/das/user/QYJI/1_Pipeline/2_tmrm.py) 代码 + 实图核对，纠正之前的错误）：

| 通道 | 真实身份 | 证据 |
| --- | --- | --- |
| **ch1** | **明场 brightfield（BF，无标记）** | 处理时 `cv2.bitwise_not` 反转；实图为灰底半透明贴壁细胞 |
| ch2 | TMRM 荧光（ΔΨm） | Otsu 取亮信号 |
| **ch3** | **核荧光 Hoechst/DAPI** | Cellpose `model_type='nuclei'` 分割计数；实图黑底亮核 |
| ch4 | MitoTracker 荧光（质量） | Otsu 取亮信号 |

→ **C1 = ch1 = 明场（BF）**（数据级验证 C1_aggre = 通道 (1) 均值）。`cell_count` 来自 ch3 核分割。

**实证：BF 特征泛化最好**（留一靶交叉验证 LOGO，预测真实 Seahorse，n=16 靶，图 `output/2026-06-10/figs/feature_modality_benchmark.png`）：

| 特征 | LOGO CV Pearson |
| --- | --- |
| **C1 / 明场 BF** | **+0.38（最强）** |
| transcriptome HVG | +0.26 |
| C24 / 线粒体染料 | +0.02（最弱） |

- **反直觉但关键**：C24 的 vAssay 0.77（§11）是 pipeline TabPFN 在**同分布**上的拟合；这里是对**全新 target 泛化**（LOGO，杜绝泄漏）。C24 线粒体染料特征**过度专门化**，泛化差；**BF 明场编码更广义、可迁移的细胞形态/状态**，反而泛化最好。
- 印证建模哲学：**无标记 BF 特征更适合做泛化建模的输入**（更便宜、更通用、避免用下游表型染料当输入的循环性）。
- 三模态对象已同时纳入 **`obsm['X_dino_c1']`（BF）** 与 **`obsm['X_dino_c24']`（线粒体）** 两个独立 384 维影像块（[align_multimodal.py](../scripts/align_multimodal.py) `--channels C1 C24`），建模时按需选用：BF 做泛化输入，C24 做线粒体表型相关。

---

*主线 D 数据地基已交付；影像侧 C24 全 24 板齐全并经 Seahorse 金标准验证；三模态已孔级对齐（1440/1440，含 BF + 线粒体双影像块）。下一步：主线 B（EE hit calling，Spearman 口径）或多模态融合建模（转录组 + BF 影像 → EE 表型）。*

---

## 13. vAssay 历史模型 review + 防泄漏重评估（2026-06-11）

> 对既往 vAssay（影像 DINOv2 → TabPFN → Seahorse）训练工作做系统 review，并建立**防泄漏评估框架**，给出可信的泛化性能。

### 13.1 既往工作与问题

- 训练资产已拷入仓库：`data/vassay_train/train_C{1,12,14,24}.csv`（264 孔 × 384 特征）+ `train_C14_giant.csv`（1536）+ `legacy_weights/`（8 个旧 TabPFN，git 忽略）。
- `Treatment` 15 种 = siRNA（siATP5B/siMFN2/siMFF/siMIEF2/siAAC/siETS1/siSLC22A3/siNTC）+ 化合物（BAM15/CCCP/FCCP/DMSO/No add/Smol）。
- **核心问题**：旧 `B2-Modeling.ipynb` 用 `KFold(shuffle=True)` 随机划分 → 同板/同处理跨 train/test 泄漏 → readme 报告的 R²≈0.77 偏乐观。

### 13.2 防泄漏评估框架（已实现）

- 代码：[src/vcell/vassay/](../src/vcell/vassay/) + [scripts/vassay_benchmark.py](../scripts/vassay_benchmark.py) / [vassay_crossdomain.py](../scripts/vassay_crossdomain.py) / [vassay_summary.py](../scripts/vassay_summary.py) + [tests/test_vassay.py](../tests/test_vassay.py)（5 测试，全量 29 passed）。
- CV 方案：`random`（复现旧泄漏值）/ `group_plate`（泛化到新板）/ `group_treatment`（泛化到新扰动，部署相关）/ `logo_treatment`。
- 运行：Ridge 用 `scvi` env；TabPFN 用 `SCIPY_ARRAY_API=1` + `cp3` env。

### 13.3 关键结果（修正上次过重的判断）

AUC 预测，随机 CV vs 诚实分组 CV（Ridge，out-of-fold）：

| 通道 | random r | group_treatment r | group_treatment Spearman |
| --- | --- | --- | --- |
| C24 | 0.86 | 0.66 | **0.64** |
| C12 | 0.84 | 0.64 | 0.61 |
| C14 | 0.86 | 0.64 | 0.61 |
| C1 | 0.82 | 0.58 | 0.55 |
| mean baseline | — | — | −0.40 |

- **泄漏真实但非灾难**：`R²` 在分组 CV 下转负（对跨板 scale 漂移极敏感），但 **Pearson/Spearman 只降 ~0.2**。
- **排序能力是真的**：`group_treatment`（泛化到新扰动）下 C24 Spearman **0.64**，远超 mean baseline 的 **−0.40** → 模型学到了**可迁移的排序信号**。
- ⚠ **自我修正**：上次只看 `R²=−1.3` 下"泛化≈0"的结论过重了。R² 看绝对标定，而 **EE hit 的真正用途是排序（Spearman），它在诚实 CV 下稳健在 0.6+**。
- 诚实通道排序（group_treatment Spearman）：**C24 ≈ C12 ≈ C14 > C1**。（与 §12.4 drug-seq LOGO 上 BF 最优不同 —— 那里是纯 siRNA 域 + y 来自真实 Seahorse；此处是化合物+siRNA 混合域 + y 来自同批 Seahorse。）
- **TabPFN 印证**（同框架，AUC，out-of-fold）：C24 random R² 0.80/ρ 0.85 → group_plate R² −1.57 → group_treatment **R² 0.35/r 0.69/ρ 0.67**；C1 random ρ 0.83 → group_treatment ρ 0.58。TabPFN 比 Ridge 略好（C24 group_treatment R² 转正 0.35），但**泄漏模式与排序信号完全一致** → 结论与模型无关，稳健。

### 13.4 跨域外部验证（P3）

legacy vAssay 预测 vs drug-seq 真实 Seahorse（n=12 靶，[vassay_crossdomain.py](../scripts/vassay_crossdomain.py)）：

| 子集 | n | Pearson | Spearman | MAE |
| --- | --- | --- | --- | --- |
| 全部 | 12 | 0.76 | 0.35 | 0.18 |
| 剔除 needs_repeat | 9 | 0.73 | **0.18** | 0.18 |

- **跨到纯 siRNA 域排序能力弱**（Spearman 0.18–0.35）：Pearson 0.76 主要靠 PSMC3 一个低值点撑起。
- 印证 §13.1 的域偏移担忧：训练域（化合物+siRNA 混合）→ 应用域（纯 siRNA KD）有分布偏移。

### 13.5 结论与建议

- **可信用途**：vAssay 在同域内可做 EE 排序（信 Spearman，不信绝对预测值）；C24（线粒体通道）在混合域内排序最好。
- **不可信用途**：跨到纯 siRNA 域的绝对值标定和细粒度排序；旧 readme 的 R²≈0.77。
- **重训建议**：若要服务 drug-seq，应用 **siRNA 域数据重训 + 强制分组 CV（按板/按靶）报告**，并附诚实泛化指标，而非沿用化合物域旧权重。需要更多独立板提升泛化。

### 13.6 标签泄漏根因 + 去泄漏系统 benchmark（路线 1 跑通）

**random R² 虚高的根因 = 标签泄漏（两层）：**
1. **主因（视野共享答案）**：264 个影像样本只对应 **88 个唯一 y 值 / 36 个 (板,处理) 组**。Seahorse 是**孔级测量**，但每孔多视野/重复**共享同一个 y 值**（组内 y 的 std 中位数 = 0.00），平均 3 个样本共享 1 个 Seahorse 读数。随机 CV 把它们拆到 train+test = 开卷考试。
2. **板效应**：板间 y 均值 std 38.7 > 板内 16.8，随机 CV 共享同板。

**修复**：`load_vassay_csv(aggregate=True)` 聚合到 (板,处理) 独立单元（264→36）；`sirna_only=True` 只留 siRNA（→18 单元）；`run_cv(pooled=True)` 用 **pooled OOF** 算指标（小样本/LOTO 的正确口径，非 per-fold 平均）。

**⚠ 二次方法学修正**：§13.3 报的 `group_plate R²=−1.3"泛化≈0"` 其实是 **per-fold 平均 R² 的伪影**（每折 scale 不同被平均放大）。改用 **pooled OOF** 后真相更乐观：

| setting | group_treatment Spearman（C1/C12/C14/C24） |
| --- | --- |
| RAW（264，含视野泄漏） | 0.67 / 0.69 / 0.70 / **0.75** |
| AGGREGATED（36，去视野泄漏） | 0.81 / 0.85 / 0.79 / **0.82** |
| siRNA-domain（18，LOTO） | −0.10 / 0.23 / −0.14 / **0.33** |

- **vAssay 对新扰动的排序能力同域内真实且强**（聚合后 Spearman 0.8），泄漏没摧毁它，聚合去视野泄漏后反而更干净。
- **真瓶颈是 siRNA 域样本量**：LOTO 下仅 18 个独立单元，C24 ρ0.33（远超 mean baseline 的 −0.87，但太弱）。
- 脚本：[scripts/vassay_systematic_benchmark.py](../scripts/vassay_systematic_benchmark.py)（raw/aggregated/sirna_domain 三组一键出图）；产物 `output/2026-06-11/vassay_systematic/`。

**路线 1 结论**：去泄漏 pipeline 已跑通；C24 在 siRNA 域有微弱排序信号，但 **18 个独立单元远不足以训生产模型，瓶颈是样本量而非方法**。下一步要么攒更多 siRNA+Seahorse 配对，要么用 drug-seq 全量（~170 靶有影像+TMRM，但仅 16 靶有真实 Seahorse）做半监督/迁移。

---

*vAssay review 完成：建立了防泄漏评估框架 + 去泄漏 pipeline，两次诚实修正（R² 高估是泄漏；group R² 崩盘是 per-fold 伪影，pooled OOF 下排序信号 0.8 真实）；siRNA 域瓶颈定位为样本量（18 单元）。*
