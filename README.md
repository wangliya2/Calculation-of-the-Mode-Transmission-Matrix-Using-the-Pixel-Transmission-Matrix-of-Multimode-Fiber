# MTM 项目（精简运行版）

## 项目简介

本项目用于将多模光纤像素域传输矩阵（PTM）转换为模式域传输矩阵（MTM），并提供：

- 预处理验证（SNR、相位误差）
- MTM 重构与可视化（幅度/相位热力图）
- 跨条件误差分析（MSE、RE、非对角能量）
- 重构后误差降低（ridge 后端 + 可选 MLP 后端）

核心关系：

`H_modes = M_out† · H_pixel · M_in`

---

## 输入与输出

### 输入

- PTM 输入：`data/input_tiff/*.tif(f)`（双通道：实部+虚部）
- 配置参数：`config.py`
- 可选参考 MTM：`--reference-mtm path/to/reference_mtm.npy|.npz`

### 主要输出

- 重构结果：`data/output_mtm/run_*/`
  - `*_mtm.npy`（主结果）
  - `*_mtm.csv`（可选导出）
  - 统计文本与运行日志
- 报告数据：`report/files/`
  - 预处理验证、误差分析、误差降低 CSV/TXT
- 报告图像：`report/report_figures/`（或 `report/figures/`）

---

## 最小运行步骤（推荐）

在项目根目录依次执行：

```bash
python generate_standard_data.py
python make_dummy_ptm.py
python validate_preprocessing.py
python main.py --input-dir data/input_tiff --output-dir data/output_mtm --basis-correction
python run_task3_error_analysis.py
python run_task4_error_reduction.py
python check_acceptance_metrics.py
python generate_report_figures.py
```

说明：

- `run_task3_error_analysis.py` 与 `run_task4_error_reduction.py` 默认读取 `data/output_mtm` 下最新的 `run_*`。
- 若需显式指定参考矩阵，可添加 `--reference-mtm`。

---

## 关键脚本说明

- `main.py`  
  主入口：读取 TIFF PTM，调用预处理与重构，输出 `*_mtm.npy` 与图表/统计。

- `validate_preprocessing.py`  
  预处理验证：输出复场 SNR 与相位误差，结果写入 `report/files/`。

- `run_task3_error_analysis.py`  
  跨条件误差分析：对 `*_mtm.npy` 计算 MSE/RE/非对角能量并汇总。

- `run_task4_error_reduction.py`  
  重构后误差降低：基于参考 MTM 训练并应用校正映射。  
  支持 ridge 后端；有 TensorFlow 时可选 MLP 后端。

- `check_acceptance_metrics.py`  
  统一检查关键指标是否达标（LP、预处理、MTM 结构、误差降低）。

- `generate_report_figures.py`  
  从 `report/files/` 生成论文使用图表。

---

## 常用命令

### 仅重跑重构

```bash
python main.py --input-dir data/input_tiff --output-dir data/output_mtm --basis-correction
```

### 误差分析（显式指定参考）

```bash
python run_task3_error_analysis.py --reference-mtm path/to/reference_mtm.npy
```

### 误差降低（显式指定参考）

```bash
python run_task4_error_reduction.py --reference-mtm path/to/reference_mtm.npy
```

### 清空旧图后重生（PowerShell）

```powershell
if (Test-Path .\report\figures) { Remove-Item .\report\figures\* -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path .\report\report_figures) { Remove-Item .\report\report_figures\* -Recurse -Force -ErrorAction SilentlyContinue }
python generate_report_figures.py
```

---

## 环境依赖（最小）

```bash
pip install numpy scipy matplotlib pandas tifffile opencv-python-headless scikit-image
```

可选（启用 MLP 后端）：

```bash
pip install tensorflow
```

---

## 作者

Wang Liya (2022213560)  
北京邮电大学国际学院 / 伦敦玛丽女王大学联合培养项目


# MTM 项目

**利用多模光纤像素传输矩阵计算模式传输矩阵**

## 项目结构

```
MTM_Project/
├── config.py                    # 项目配置（光纤参数、阈值）
├── main.py                      # MTM流水线主入口
├── generate_standard_data.py    # 生成标准LP模式参考数据
├── make_dummy_ptm.py            # 生成用于测试的模拟PTM数据
├── run_article_mtm.py           # 使用 article_MMF_disorder 仓库 Data/（TM 分片 + conversion_matrices.npz）跑 MTM
├── export_article_reference_mtm.py  # 将仓库 TM_modes_*.npz 导出为参考 MTM（.npy）
├── validate_preprocessing.py    # 预处理验证（SNR + 相位MAE）
├── src/
│   ├── __init__.py
│   ├── lp_theory.py             # LP模式理论（Bessel函数、特征方程）
│   ├── lp_mode_simulation.py    # LP模式仿真与CCC验证
│   ├── data_preprocessing.py    # PTM数据预处理（去噪、相位解包裹）
│   ├── mtm_reconstruction.py    # MTM重建（H_modes = M_out† · H_pixel · M_in）
│   ├── mtm_calculator.py        # 端到端MTM计算流水线
│   ├── error_metrics.py         # MSE和相对误差计算
│   ├── error_analysis.py        # 跨条件MTM误差分析
│   ├── error_reduction.py       # 重构后MTM误差降低（ridge / MLP）
│   └── article_reference_mtm.py # 从 .npy / TM_modes_*.npz 加载参考 MTM
├── data/
│   ├── input_tiff/              # 输入PTM的TIFF文件
│   └── standard_lp_modes/       # 生成的标准LP模式数据
└── report/                      # 生成的报告和图表
```

## 快速开始

### 环境依赖

```bash
pip install numpy scipy matplotlib pandas tifffile opencv-python-headless scikit-image
```

### 第1步：生成标准数据

```bash
python generate_standard_data.py
```

### 第2步：生成模拟PTM（用于测试）

```bash
python make_dummy_ptm.py
```

### 第3步：运行MTM流水线

```bash
python main.py --input-dir data/input_tiff --output-dir data/output_mtm
```

### 使用论文仓库像素域 TM（替代 dummy / TIFF 流水线）

从 [wavefrontshaping/article_MMF_disorder](https://github.com/wavefrontshaping/article_MMF_disorder) 下载 `Data/`，按该仓库 README 将 `TMxx_0.npy` 与 `TMxx_1.npy` 沿第 0 维拼接得到完整像素 TM，再与 `conversion_matrices.npz` 中的 `modes_in` / `modes_out` 一起做 MTM：

```bash
python run_article_mtm.py ^
  --tm-part0 path/to/Data/TM25_0.npy ^
  --tm-part1 path/to/Data/TM25_1.npy ^
  --conversion path/to/Data/conversion_matrices.npz ^
  --output-dir data/output_mtm
```

可选几何校正（**在论文给定的扁平模式矩阵上做搜索**，不会用 LP 理论覆盖 `modes_out`）：

```bash
python run_article_mtm.py --tm-part0 ... --tm-part1 ... --conversion ... --output-dir data/output_mtm --basis-correction
```

若 `modes_in` 每列可 reshape 为与输出像素网格一致的 `(H,W)`，可启用两侧联合校正：

```bash
python run_article_mtm.py --tm-part0 ... --tm-part1 ... --conversion ... --output-dir data/output_mtm --basis-correction --joint-basis-correction
```

说明：该入口直接使用 `H_pixel`（不经 TIFF 预处理）；输出写入 `data/output_mtm/run_*`（`article_TM_mtm.npy`、CSV、统计 txt、热力图 PNG）。

### 论文仓库：像素 TM 与「官方模式域参考」怎么对应？

- **仓库地址（注意拼写）**：[wavefrontshaping/article_MMF_disorder](https://github.com/wavefrontshaping/article_MMF_disorder)（`disorder` 勿写成 `disorde`）。
- **像素域**：`TMxx_0.npy` + `TMxx_1.npy`（`xx` 如 5、17、25、35、50、52）沿第 0 维拼接 → 本项目的 `H_pixel`。
- **模式域参考（文献侧已处理）**：`Data/TM_modes_*.npz` 为 README 所述 **deformation 校正后的 mode-basis 传输矩阵**，可作 `--reference-mtm`，用于误差分析/误差降低与您自己算的 `M_out† H_pixel M_in` 对照。
- **重要**：`TM25` 里的 **25** 与 `TM_modes_5.0` 里的 **5.0** 不是同一个编号体系；后者多与**形变参数（如位移 mm）**对应。请务必对照仓库 **`Data/param.json`** 以及论文 / Supplement（如 *Learning and Avoiding Disorder in Multimode Fibers*, Phys. Rev. X）中的表格，**选用与当前像素 TM 同一实验状态**的 `TM_modes_*.npz`，否则数值对比只能算「差异」，物理上不好解释。
- **导出参考为 `.npy`**（便于版本管理）：  
  `python export_article_reference_mtm.py --input path/to/Data/TM_modes_5.0.npz --output data/reference_mtm_article.npy`  
  误差分析示例：  
  `python run_task3_error_analysis.py --mtm-dir data/output_mtm/latest --reference-mtm path/to/TM_modes_5.0.npz`  
  多数组 `.npz` 可指定：`--reference-npz-key 键名`。

如需启用“模式基误差校正（缩放/平移/旋转优化）”，可使用：

```bash
python main.py --input-dir data/input_tiff --output-dir data/output_mtm --basis-correction
```

若 `M_in` 是空间模式矩阵（形状为 `(H*W, N_modes)`，每列可 reshape 为 `(H,W)`），可进一步启用“**M_out 与 M_in 联合几何校正**”（同一组 scale/shift/rotation 同时作用两侧）：

```bash
python main.py --input-dir data/input_tiff --output-dir data/output_mtm --basis-correction --joint-basis-correction --input-basis file --m-in-path path/to/your_M_in.npy
```

对多份空间模式矩阵做平均（例如多次标定导出的 `M_in`/`M_out`）：

```bash
python average_spatial_mode_matrix.py --inputs run1/M.npy run2/M.npy --output data/mean_M.npy --orthonormalize
```

说明：
- 当前默认光纤参数为直径 `50um`（半径 `25um`），对应论文常用多模光纤规格。
- 工作波长默认 `1.55um`，数值孔径 `NA=0.22`，与参考仓库实验设置一致。
- 模式场仿真空间默认覆盖 `±35um`，用于覆盖纤芯及其包层过渡区域，避免边界截断。

### 第4步：验证预处理

```bash
python validate_preprocessing.py
```

### 第5步：跨条件 MTM 误差分析

```bash
python run_task3_error_analysis.py
```

### 第6步：重构后 MTM 误差降低

```bash
python run_task4_error_reduction.py

说明：
- `run_task3_error_analysis.py` / `run_task4_error_reduction.py` 默认会自动选择 `data/output_mtm` 下最新的 `run_*` 目录。
- 不要直接把 `run_YYYYmmdd_HHMMSS` 当成真实目录名使用。

### 图像重生成

在仓库根目录依次执行（与「任务1→4 + 报告图」一致）：

```bash
python generate_standard_data.py
python -m src.lp_mode_simulation
python make_dummy_ptm.py
python validate_preprocessing.py
python main.py --input-dir data/input_tiff --output-dir data/output_mtm --basis-correction
python run_task3_error_analysis.py
python run_task4_error_reduction.py
python check_acceptance_metrics.py
python generate_report_figures.py
```

说明：`run_task3` / `run_task4` 默认使用 `data/output_mtm` 下**最新**的 `run_*`（内含 `*_mtm.npy`）。若你刚跑完 `main.py`，无需改参数。

说明：`python -m src.lp_mode_simulation` 可能较慢（数分钟量级），与机器性能有关；若已生成过 `report/report_figures` 中的 CCC 相关图，可酌情跳过以节省时间。

#### 自动核对是否达到项目内定量要求

```bash
python check_acceptance_metrics.py
```

检查项：**LP 仿真 CCC**（均值≥0.85、LP01≥0.90）、**预处理**（`validate_preprocessing.py` 当前仅统计默认**高斯**去噪：复场 SNR 提升在 8 模式上**均值**≥−0.35 dB；相对相位误差**各模式 max**≤0.14 rad；见 `check_acceptance_metrics.check_preprocess`）、**直纤 MTM**（增益归一化后 `offdiag_max`≤0.05、`diag_mean` 在 1±0.2）、**误差降低**（摘要中「是否达标」为「是」即 MSE/RE 平均下降≥30%）。脚本退出码：全部通过为 `0`，否则为 `1`。

**为何常出现 MTM 项 FAIL**：默认 `main.py` 使用 **Hadamard 输入基**与 **dummy 正向模型**，得到的 MTM **不是**「理想直纤 + 单位输入」下的近似单位阵；`config.MTMConfig` 阈值更适用于那种演示设定。若要与该阈值对齐，可显式使用单位输入基，例如：  
`python main.py --input-dir data/input_tiff --output-dir data/output_mtm --input-basis identity`  
（是否采用以课程/导师要求为准。）

**预处理项**：高阶 LP 在 20 dB 复噪声下，去噪+展开未必使复 SNR 的**最小值**单调上升；验收采用**跨模式均值**与**有物理依据的相位上界**，与 `validate_preprocessing.py` 报告文字一致。

#### 先清空旧图再生成

```powershell
Remove-Item .\report\figures\* -Recurse -Force
Remove-Item .\report\report_figures\* -Recurse -Force
```



#### PowerShell 一键全重跑

```powershell
if (Test-Path .\report\figures) { Remove-Item .\report\figures\* -Recurse -Force -ErrorAction SilentlyContinue }; if (Test-Path .\report\report_figures) { Remove-Item .\report\report_figures\* -Recurse -Force -ErrorAction SilentlyContinue }; python generate_standard_data.py; python -m src.lp_mode_simulation; python make_dummy_ptm.py; python validate_preprocessing.py; python main.py --input-dir data/input_tiff --output-dir data/output_mtm --basis-correction; python run_task3_error_analysis.py; python run_task4_error_reduction.py; python check_acceptance_metrics.py; python generate_report_figures.py
```

## 模块进度

| 模块 | 描述 | 状态 |
|------|------|------|
| 模式仿真 | LP模式理论学习与仿真 | 已完成 |
| MTM重构 | MTM计算软件开发 | 已完成 |
| 误差分析 | 跨条件 MTM 误差分析 | 已完成 |
| 误差降低 | 重构后 MTM 误差降低 | 已完成 |

## 配置说明

关键参数可在 `config.py` 中修改：

- **FiberParams（光纤参数）**: 纤芯半径、数值孔径NA、工作波长、包层折射率
- **PreprocessConfig（预处理配置）**: 去噪方法和滤波核参数
- **MTMConfig（MTM配置）**: 模式数量和验收阈值

## 项目概述

本项目实现了一个基于Python的软件流水线，用于从实验测量的多模光纤像素传输矩阵（PTM）数据中计算模式传输矩阵（MTM）。核心算法基于以下数学关系：

```
H_modes = M_out† · H_pixel · M_in
```

其中 `H_pixel` 是预处理后的PTM，`M_out` 和 `M_in` 分别是输出和输入LP模式基矢矩阵，`†` 表示厄米共轭（Hermitian conjugate）。

## 毕设对齐：输入输出是什么？误差分析与误差降低在做什么？

### 输入、中间量、输出（两条数据路径）

| 路径 | 你喂给程序的东西（输入） | 中间量 | 程序产出（输出） |
|------|-------------------------|--------|------------------|
| **TIFF 流水线** `main.py` | `data/input_tiff/*.tif(f)`：双通道 TIFF（实部+虚部），表示像素域 PTM | 预处理后得到复矩阵 **`H_pixel`**（形状约 `(输出像素数, 输入模式数)`）；再由 LP 理论或文件得到 **`M_in`/`M_out`** | 每次运行目录 `data/output_mtm/run_*`：`…_mtm.npy`（**MTM = `H_modes`**）、CSV、统计 txt、热力图、`run.log.txt` |
| **论文数据** `run_article_mtm.py` | `TMxx_0.npy` + `TMxx_1.npy` 拼成 **`H_pixel`**；`conversion_matrices.npz` 里的 **`modes_in`/`modes_out`** | 可选几何校正后再代入同一公式 | 同上，主文件名为 `article_TM_mtm.npy` 等 |

公式始终是：**`H_modes = M_out† · H_pixel · M_in`**（模式域传输矩阵）。

### 跨条件 MTM 误差分析（`run_task3_error_analysis.py` + `src/error_analysis.py`）

- **做什么**：在某个 `run_*` 目录里扫描所有 `*_mtm.npy`，把每个重建的 MTM 与一个**参考 MTM** 对比，算 **MSE、相对误差 RE、非对角能量占比** 等，并按文件名里的条件（如 straight/bent、article）汇总到 CSV/TXT。
- **默认参考**：**单位矩阵 `I`**，适合 **dummy「理想直纤」** 这类故事；**论文实测 MTM 一般不会接近 `I`**，此时对比没有物理意义。
- **用论文仓库自带的「模式域」参考（推荐）**：[article_MMF_disorder](https://github.com/wavefrontshaping/article_MMF_disorder) 的 `Data/TM_modes_*.npz` 即 README 中的 **mode-basis 传输矩阵（deformation 校正后）**，可直接作参考，与你自己算的 `M_out† H_pixel M_in` 对比时，应在论文中说明**双方处理链可能不同，属「文献对照」而非严格同一估计量**。
  - 任选与实验条件接近的 `TM_modes_XX.npz`，导出为 `.npy`：  
    `python export_article_reference_mtm.py --input path/to/Data/TM_modes_5.0.npz --output data/reference_mtm_article.npy`  
  - 或误差分析/误差降低脚本直接传 `.npz`：  
    `python run_task3_error_analysis.py --reference-mtm path/to/TM_modes_5.0.npz`  
    若文件内含多个数组，可加 `--reference-npz-key 数组名`；省略时程序会**自动挑选**最像方阵 TM 的数组。
- **其他真值**：若有仿真 MTM，仍可用 `reference_mtm.npy` +  
  `python run_task3_error_analysis.py --reference-mtm path/to/reference_mtm.npy`

### 重构后 MTM 误差降低（`run_task4_error_reduction.py` + `src/error_reduction.py`）

- **做什么**：把多个「重建出的 MTM」当作输入样本，把「参考 MTM」（默认仍是单位阵，或用 `--reference-mtm`）当作标签，训练一个**从向量化 MTM → 校正后 MTM** 的映射。当前实现支持 **ridge-regression backend** 和可选 **TensorFlow MLP backend**；在无 TensorFlow 环境下，ridge 后端也可独立完成误差降低并输出模型文件（`.npz`）。
- **局限**：只有 **1 条** MTM 时无法严肃验证泛化，报告里会自动提示；要满足「减少误差 ≥30%」这类目标，通常需要 **多条不同条件** 的 MTM 或可信参考。
- **命令**：`python run_task4_error_reduction.py --reference-mtm path/to/reference_mtm.npy`（可选）。

## 作者

Wang Liya (2022213560) - 北京邮电大学国际学院 / 伦敦玛丽女王大学联合培养项目
