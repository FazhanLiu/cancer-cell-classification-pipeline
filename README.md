# Cancer Cell Classification Pipeline · PathScope

基于 PyTorch 的病理小图（patch）二分类与可解释性研究项目，支持 ResNet50、ViT-Small、CNN-GNN，以及本机运行的中文病例分析网站。

```text
训练阶段：病理数据 → 固定划分 → 训练 → 验证集选择模型与阈值 → 独立测试
使用阶段：同一病例的多张 patch → 逐图分类 → 三种 XAI → DeepSeek 中文病例报告
```

**网站已经实现，但真实分析需要训练好的 ViT 权重。** ImageNet 预训练权重不能直接替代病理二分类模型；未配置模型时，网站只能保存待分析病例，不会生成随机预测。当前项目属于研究辅助工具，尚未经临床验证，不提供病灶分割或患者诊断。

## 目录

- [本次改造](#本次改造)
- [安装环境](#安装环境)
- [已有模型：启动本机网站](#已有模型启动本机网站)
- [没有模型：准备数据并训练](#没有模型准备数据并训练)
- [选择阈值与独立评估](#选择阈值与独立评估)
- [微调、Focal Loss 与难例挖掘](#微调focal-loss-与难例挖掘)
- [断点续训](#断点续训)
- [模型概率集成](#模型概率集成)
- [XAI 实现与定量比较](#xai-实现与定量比较)
- [独立生成 LLM 报告](#独立生成-llm-报告)
- [目录与产物](#目录与产物)
- [测试与验证范围](#测试与验证范围)
- [常见问题](#常见问题)

## 本次改造

| 部分 | 修改及目的 |
| --- | --- |
| 数据划分 | 新增不可静默覆盖的划分清单，记录 ID、标签、split、参数与指纹；检查重复、缺失文件和集合互斥；支持患者／切片分组 |
| 训练与评估 | 共用训练入口逻辑，训练结束不再自动查看测试集；按验证 AUC 选最佳模型；统一推理预处理与阈值口径 |
| 训练可靠性 | 修正不足一个梯度累积窗口时的归一化；保存优化器、调度器、AMP 和随机状态；区分微调与 epoch 边界恢复 |
| 难例挖掘 | 仅挖掘训练集，复用同一模型；删除被覆盖的 `hard_weight`，明确采样权重与类别权重；记录难例及增强设置 |
| 产物追溯 | 独立实验目录、配置与环境快照、checkpoint／预测／决策来源校验；修复模型工厂遗漏额外参数的问题 |
| XAI | 保留带正负号的原始证据与 LIME 网格系数；增加融合图、分歧图、删除实验、随机基线、稳定性与耗时统计 |
| DeepSeek | 新增官方 API 适配、根目录 `.env` 加载、错误脱敏、失败保存及报告恢复；报告读取融合与单方法证据 |
| 本机网站 | 新增无标签图片推理、批量上传、异步病例任务、解释查看、中文病例报告、历史记录及结果导出 |
| 工程与验证 | 新增 `pyproject.toml`、固定直接依赖版本、CPU CI、合成数据及真实小型 ViT 功能测试 |

## 安装环境

支持 Python **3.10–3.12**。在项目根目录创建并激活虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活。

先在**已激活的虚拟环境**中安装适合设备的 PyTorch。本项目参考版本为 `torch==2.7.0`、`torchvision==0.22.0`；CUDA 安装方式参见 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)。Linux 纯 CPU 环境可先执行：

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
```

安装依赖及项目：

```bash
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps --no-build-isolation -e .
```

`requirements.txt` 固定经过本机测试的直接依赖；`requirements-dev.txt` 额外包含 pytest。每次实验会保存实际依赖版本。CPU 测试不需要 GPU、完整数据集、预训练下载或真实 API Key。

## 已有模型：启动本机网站

### 配置 `.env`

首次使用可复制 `.env.example` 为 `.env`；已有 `.env` 时直接补充字段，保留原有 Key：

```dotenv
DEEPSEEK_API_KEY=填写你的真实Key
DEEPSEEK_MODEL=deepseek-flash
CANCER_MODEL_PATH=outputs/vit_small/实际训练目录/best.pt
CANCER_DECISION_PATH=
CANCER_DEVICE=auto
```

| 变量 | 作用 |
| --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek 官方 API Key，仅由后端读取 |
| `DEEPSEEK_MODEL` | 网站报告模型，默认 `deepseek-flash` |
| `CANCER_MODEL_PATH` | 当前训练流水线生成的、来源可验证的 ViT checkpoint |
| `CANCER_DECISION_PATH` | 可选，指定该模型在验证集上选择的 `decision.json`；留空使用阈值 0.5 |
| `CANCER_DEVICE` | `auto`、`cpu`、`cuda` 或 `mps`；`auto` 有 CUDA 时使用 CUDA，否则 CPU |

网站和独立报告脚本自动读取项目根目录 `.env`；**已有终端环境变量优先，包括空值**。文件修改后重启网站；若终端仍导出了旧值，需要先取消该环境变量。支持引号与注释，禁用变量插值。`.env` 被 Git 忽略，Key 不写入实验配置或报告，保存的错误信息会脱敏。

网站的 Attention Rollout 需要带 CLS token 的 ViT，因此三方法完整流程目前不接受 ResNet50／CNN-GNN；这些模型仍可用于命令行训练与评估。模型通过本机配置加载，网页不接受上传 `.pt` 文件。只加载可信 checkpoint，其完整恢复状态需要 Python pickle。

### 打开网站

```bash
python scripts/serve.py
```

浏览器访问 **http://127.0.0.1:8765**。端口冲突时使用 `python scripts/serve.py --port 8766`。

1. 同一病例选择 **1–20 张已裁剪的组织病理 patch**，可填写去标识化病例备注。
2. 点击“保存并开始分析”；未配置模型时可先“保存待分析”。
3. 查看逐图模型评分、二分类结果、Attention／LIME／遮挡图、融合图及方法分歧。
4. 全部图片处理后，DeepSeek 基于结构化证据生成一份中文病例汇总，可下载 JSON 和 Markdown。

每张图像独立分类，**阳性图片比例、最大模型分数都不作为患者患癌概率**。XAI 解释模型依据，不保证病灶定位；评分距阈值的间隔也不是经过校准的置信度。

### 上传和任务行为

- 支持单帧 PNG、JPG、TIFF、WebP；每张不超过 10 MB，整批不超过 100 MB；最小 16×16，总像素不超过 16,777,216。不支持 WSI、多页 TIFF、DICOM 或其他影像任务。
- 图片经过解码校验，转换为 RGB PNG，以生成的图像 ID 存在本机。推理**不需要训练标签、`train_labels.csv` 或训练数据目录**。
- 任务在后台逐病例处理，网页轮询实际阶段。单图失败不会取消其他图片；成功的结果保留。
- DeepSeek 缺少 Key 或调用失败时不回退到 mock。配置好后可继续分析，跳过已完成图片，重试报告。
- 已完成病例禁止重复提交；相同证据与报告模型的成功报告可复用，避免重复 API 调用。服务中断后标记为可继续状态，不会自动恢复并产生请求。
- 模型、阈值、解释参数或设备发生变化时要求新建病例，避免混合不同分析设置。
- 病例及结果位于 `outputs/web_cases/`，可在非运行状态下从网页删除。

DeepSeek 只接收生成的图像编号、预测和区域解释摘要；不发送原始图像、原文件名或病例备注。网站报告使用 4096 输出 token 上限，独立报告脚本默认 1024。真实调用可能产生费用。

当前网站仅监听 **127.0.0.1**，是单用户、单进程的本机工具；没有医生账号、多用户授权、医院系统对接或公网部署。不要将开发服务器直接暴露到公网。

## 没有模型：准备数据并训练

### 组织数据

使用 Kaggle Histopathologic Cancer Detection 数据，默认布局：

```text
data/histopathologic-cancer-detection/
  train/
    图片ID.tif
  train_labels.csv          # 至少包含 id、label 两列；label 为 0 或 1
```

修改 `configs/vit.yaml` 的 `dataset.root` 为真实目录。也可在启动训练的终端导出 `CANCER_DATA_ROOT`。**训练脚本不自动加载 `.env`**，不要仅把训练数据路径写进 `.env`。显式 `dataset.manifest` 指定划分文件，否则默认为 `<dataset.root>/splits.json`。

### 固化划分并训练 ViT

```bash
python scripts/prepare_data.py --config configs/vit.yaml
python scripts/train.py --config configs/vit.yaml
```

其他模型：

```bash
python scripts/train.py --config configs/resnet50.yaml
python scripts/train.py --config configs/cnn_gnn_resnet50.yaml
```

默认按类别分层划分 **70% 训练／15% 验证／15% 测试**，seed 42。这不是 XAI 融合权重。已有划分文件只校验、不覆盖；标签或划分参数变化时，需要显式指定新 manifest 路径并重新准备。校验 ID 唯一、标签合法、图片存在、集合覆盖与互斥、各集合包含两类。

有患者／切片分组时，在标签 CSV 添加对应列，并配置 `dataset.group_column: slide_id`。分组不能跨集合；分组大小可能使比例近似，准备时从 128 个固定种子候选中按数量比例及类别分布选择。没有分组信息时明确记录 `patch_random_no_group_guarantee`，不能声称患者级独立。指纹覆盖 ID、标签及分组，**不覆盖图像文件字节**，图像数据仍需单独固定版本。

训练按验证集 AUC 保存最佳权重，**不会自动评估测试集**。每次运行打印独立目录，例如 `outputs/vit_small/<UTC时间>-train-<随机后缀>/`；其中 `best.pt` 可供网站使用。当前仓库不包含数据集、训练权重或可核验的完整数据实验成绩。

## 选择阈值与独立评估

把 `RUN` 替换为训练命令实际输出的目录：

```bash
RUN="outputs/vit_small/实际训练目录"
python scripts/threshold_tuning.py --predictions "$RUN/val_predictions.csv"
```

默认最大化验证集 F1；并列选择更高阈值。也可在满足 Recall 目标后最大化 Precision：

```bash
python scripts/threshold_tuning.py --predictions "$RUN/val_predictions.csv" \
  --objective recall_precision --target-recall 0.95 \
  --start 0.05 --stop 0.95 --step 0.01
```

阈值搜索只接受带来源信息的、来源已验证的验证集预测；非法范围、步长、缺少类别或没有可行阈值时明确失败，不再使用固定 `FN < 249` 目标。

使用命令打印的 `decision.json` 路径独立评估测试集：

```bash
DECISION="实际决策目录/decision.json"
python scripts/evaluate.py --checkpoint "$RUN/best.pt" --decision "$DECISION" --device cpu
```

决策文件绑定 checkpoint 内容、划分指纹及预处理。评估、绘图、XAI 和报告使用同一阈值；未提供决策时默认 0.5。指标包含 Accuracy、Precision、Recall、F1、AUC、混淆矩阵和明确命名的 `cross_entropy_unweighted`。

`evaluate.py --split val` 可重新生成验证预测。`--data-root`、`--manifest`，或 `--config` 中的数据及输出位置可用于搬迁同一数据集；模型和预处理以 checkpoint 为准，不能更换划分身份。

测试集只用于最终评估。反复查看测试结果再调整模型、阈值或解释权重仍会污染测试集；来源记录不能证明操作者没有这样做。

## 微调、Focal Loss 与难例挖掘

普通微调只加载权重，重新初始化优化器和调度器：

```bash
python scripts/train.py --config configs/vit.yaml --fine-tune "$RUN/best.pt"
```

Focal Loss 可从通用预训练初始化训练；也可使用已有分类权重微调：

```bash
python scripts/train_vit_focal_loss.py --config configs/vit.yaml \
  --gamma 2 --alpha-negative 0.5 --alpha-positive 0.5 \
  --fine-tune "$RUN/best.pt"
```

没有基础 checkpoint 时省略 `--fine-tune`。`--pretrained-checkpoint` 保留为其别名。

难例挖掘只对训练集做推理，并复用同一模型继续训练：

```bash
python scripts/train_vit_hard_mining.py --config configs/vit.yaml \
  --base-checkpoint "$RUN/best.pt" --decision "$DECISION" \
  --fp-weight 4 --fn-weight 6 --epochs 20 --lr 0.00001
```

普通样本采样权重为 1，默认 FP 为 4、FN 为 6，均需非负且总权重有效。不再保留无独立作用的 `hard_weight`。难例模式默认关闭类别加权；只有显式 `--class-weight` 才与采样加权叠加。

难例增强使用全图翻转、旋转、颜色扰动及 RandAugment，不再默认随机裁剪／擦除。与旧增强配方比较时应视为新的实验。基础模型、挖掘决策、增强参数与难例表及其哈希都会记录；基础模型划分不一致时拒绝训练。

## 断点续训

```bash
# 总计划仍来自配置；仅在第 5 个 epoch 结束后暂停本次执行。
python scripts/train.py --config configs/vit.yaml --stop-after-epoch 5
python scripts/train.py --config configs/vit.yaml --resume outputs/vit_small/中断目录/last.pt
```

恢复模型、优化器、余弦调度器、AMP scaler、最佳指标、早停状态、训练历史、随机状态及 loader／采样生成器。只支持 **epoch 边界**恢复，不支持 batch 中间恢复。续训目录须保留相邻的 `best.pt`、验证预测及其元数据，且最佳 checkpoint 哈希须匹配。

续训须使用原训练模式和相同的训练计划、模型、预处理、worker 数量及设备类型；若首次运行通过 `--epochs`／`--lr` 覆盖配置，恢复时也需要提供相同值。难例及 Focal 模式使用各自脚本的 `--resume`。已完成或早停的计划应改用明确的微调，而不是续训。

梯度累积按实际有效批次及损失归约方式计算，兼顾不足窗口、短 batch 和加权交叉熵。验证损失始终是未加权 CE，不能与 Focal／加权训练损失视为同一目标。

默认 `deterministic: true`，固定随机性，禁用 persistent workers，记录运行环境。CUDA 确定性运行可能需要在启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；不支持的确定性算子会报错。不承诺跨硬件逐位一致；AMP 仅在 CUDA 启用。

旧 checkpoint 可显式 `--allow-legacy` 做仅权重微调或评估；缺少预处理时需额外配置。其后续来源仍标记 `unverified`，不能用于已验证的阈值选择、XAI 比较或网站分析，也不能恢复旧文件缺失的完整训练状态。

## 模型概率集成

创建配置，例如 `configs/my_ensemble.yaml`：

```yaml
run_name: ensemble
output_dir: outputs
models:
  - checkpoint: outputs/resnet50/实际训练目录/best.pt
    weight: 0.5
  - checkpoint: outputs/vit_small/实际训练目录/best.pt
    weight: 0.5
```

```bash
python scripts/ensemble.py --config configs/my_ensemble.yaml --split val
# 在输出的验证预测上选择阈值，再独立评估：
python scripts/ensemble.py --config configs/my_ensemble.yaml --split test \
  --decision 实际集成决策目录/decision.json
```

各模型使用自身 checkpoint 的预处理，必须共用同一划分，按 ID 与标签对齐。权重须有限、非负且总和为正；集成身份绑定模型哈希与权重。配置中可选 `dataset.root`／`dataset.manifest` 搬迁数据路径。

**预测概率集成会改变分类结果，热力图融合不会。** 集成权重同样只能在验证数据上选择。

## XAI 实现与定量比较

对固定验证／测试集做命令行解释：

```bash
python scripts/xai_vit.py --checkpoint "$RUN/best.pt" \
  --predictions 实际评估目录/test_predictions.csv --mode diagnostic

python scripts/xai_vit.py --checkpoint "$RUN/best.pt" \
  --predictions 实际评估目录/test_predictions.csv \
  --mode quantitative --samples 20 --repeats 3
```

命令行评估型 XAI 会校验划分、标签和预测来源；网站使用独立的无标签推理入口，不要求上传图片属于原数据集。解释对象默认是**癌症类别模型评分**：

| 方法 | 实现与含义 |
| --- | --- |
| Attention Rollout | 多头平均、残差连接、逐层注意力矩阵相乘，提取 CLS 到图像 patch 的信息流；类别无关 |
| Grid LIME | 默认 7×7 网格、200 次随机遮挡，均值色填充，用加权 Ridge 回归近似癌症评分；保留原始正负系数 |
| 遮挡敏感性 | 滑动遮挡，比较遮挡前后癌症评分下降；默认窗口 32、步长 16，并覆盖边缘 |
| 融合 | 正向证据分别 Min-Max 归一化后加权，再归一化展示；默认 Attention／LIME／遮挡为 **0.5／0.3／0.2** |

融合权重是可配置的探索性设置，没有最优权重证明；不是区域取并集。输出 `.npz` 原始／带符号证据、LIME 网格系数、归一化图、融合图与分歧图，以及 PNG、区域摘要、参数、来源及耗时。没有区域标注监督，不能把热点、0.7 相对分数面积或方法分歧当作病灶面积、定位准确率或校准不确定性。阴性图像中的 LIME／遮挡仍是癌症评分的正向证据，不代表证明正常的区域。

`diagnostic` 展示 TP／TN／FP／FN 中接近阈值的案例；这些样本不代表总体。`quantitative` 固定按标签分层抽样的 ID，重复改变扰动种子，对比三种单方法、等权融合、当前融合和随机排序。在相同网格、填充值及删除比例下保存癌症评分曲线、删除 AUC、半数删除分数变化、逐图及跨图统计、稳定性和耗时。

删除 AUC 更低只表示该遮挡协议下分数下降更快；遮挡评价可能偏向扰动方法，也可能引入分布外输入，不能据此宣称分类或定位能力提升。目前不自动搜索融合权重。

## 独立生成 LLM 报告

```bash
# 本地模板，仅检查报告流程，不调用大模型。
python scripts/llm_explain_xai.py --input 实际解释目录/xai_explanations.json --provider mock

# 真实 DeepSeek 调用：先用一张图检查。
python scripts/llm_explain_xai.py --input 实际解释目录/xai_explanations.json \
  --provider deepseek --model deepseek-flash --max-samples 1 --timeout 120 --retries 0
```

默认 provider 为 `mock`；**只有独立报告 CLI 的 mock 模式使用固定模板，网站不会自动回退到它**。真实 provider 还支持 `openrouter`、`groq`、`gemini`、`huggingface`，均需显式 `--model` 及对应环境变量 `OPENROUTER_API_KEY`、`GROQ_API_KEY`、`GEMINI_API_KEY`、`HF_TOKEN`。其他适配器没有做真实联网验证，模型和接口可用性须单独确认。

DeepSeek 使用[官方 Chat Completions 接口](https://api-docs.deepseek.com/zh-cn/) `https://api.deepseek.com/chat/completions`，无需额外 OpenAI SDK。请求关闭 thinking，非流式、temperature 0.2；缺少内容或 finish reason 不是 `stop` 时记录失败。模型可用性和费用以服务商为准。

独立报告 CLI 支持 `--env-file /绝对路径/其他.env`。逐样本保存结果与失败状态，`--retries` 控制重试；`--resume-from 旧报告目录` 仅复用证据、提示词、provider、model 指纹匹配的成功报告，重试失败项，并生成新目录。网站则通过病例“继续分析”重试。

LLM 只整理已有预测、单方法／融合区域和分歧，不参与分类、阈值选择或热力图计算。真实标签和 TP／FP 等分组不进入提示词；`--include-review` 只在本地报告附加离线复盘信息。提示词限制无依据的形态及病变推断，但生成文本仍需复核。

## 目录与产物

```text
configs/                    模型、数据与训练配置
scripts/                    数据准备、训练、评估、XAI、报告及网站启动入口
src/cancer_pipeline/
  data.py, models.py        固定划分、共享预处理与模型工厂
  engine.py, training.py    训练、预测、恢复及训练策略
  artifacts.py              配置、指纹、来源与决策产物
  inference.py              不依赖标签或原始数据集的 patch 推理
  xai.py, reports.py         解释算法与独立 LLM 报告
  webapp.py, case_reports.py 病例后台任务与中文病例汇总
  web/                      本地网站模板、样式与交互
outputs/                    运行产物与病例；不提交 Git
tests/                     功能、契约与合成数据集成测试
.github/workflows/tests.yml  CPU 自动检查
```

| 产物 | 含义 |
| --- | --- |
| `splits.json` | ID、标签、分组、split、参数和内容指纹 |
| `config.json`、`environment.json`、`data_provenance.json` | 有效配置、Git 版本及脏状态、依赖环境、划分来源 |
| `best.pt`、`last.pt` | 最佳模型与完整 epoch 边界恢复状态 |
| `val_predictions.csv` 及 `.meta.json` | 对应最佳模型的验证预测及文件／来源哈希 |
| `history.csv`、`metrics.json` | 训练历史与摘要；不会包含自动测试成绩 |
| 独立评估目录 | 预测、`eval_metrics.json`、统一阈值的混淆矩阵与 ROC 图 |
| `decision.json` | 阈值、选择目标、搜索参数与验证来源 |
| XAI 目录 | 数组、图像、样本清单、解释摘要、删除曲线及比较统计 |
| `outputs/web_cases/<case_id>/` | 病例 JSON、逐图 PNG／数组、报告 JSON／Markdown |

实验输出目录包含 UTC 时间和随机后缀，避免覆盖。哈希用于检测意外不一致，不是实验真实性的密码学证明。实际数据、模型权重、`.env`、病例与环境目录均不随源码分发。

## 测试与验证范围

```bash
python -m pytest -q
python -m compileall -q src scripts tests
node --check src/cancer_pipeline/web/assets/app.js
```

Node.js 仅用于可选的前端语法检查，启动网站不依赖 Node.js。

当前本机完整测试结果：**38 项通过**，覆盖：

- 固定／分组划分、重复及缺失图片、来源和阈值不匹配、缺少类别。
- CE、加权／平滑 CE、Focal Loss 的梯度累积，以及普通／加权采样的受控 CPU 续训一致性。
- 旧 checkpoint 隔离、参数转发、统一绘图阈值、无效权重与零热力图。
- 未预训练小型 ViT 的真实前向、三种 XAI、删除实验和无标签批量推理。
- DeepSeek 请求序列化、缺少 Key、不完整响应、错误脱敏、`.env` 优先级及报告恢复。
- 网站上传约束、路径隔离、仅本机访问、病例保存／中断／删除、重复提交、逐图失败与报告重试不重复推理。
- 合成数据下的数据准备、训练、阈值选择、评估、集成、解释和报告命令行链路。

GitHub Actions 配置为 CPU 检查。DeepSeek 测试拦截 HTTP，不产生真实 API 请求或费用；测试小模型不能证明真实癌症分类能力。网站页面布局已浏览器检查，但浏览器扩展权限限制了自动选择本地文件，该段 UI 自动化尚未完成；上传接口及后台完整链路已测试。打包已检查包含网页模板和静态资源。

**尚未验证：**完整数据训练成绩、训练权重的实际临床有效性、病灶定位、GPU／MPS 与跨设备一致性，以及真实 DeepSeek 调用的端到端效果。

## 常见问题

**没有 `best.pt`，能否先打开网站？** 可以保存病例，但分析会在界面和后端同时阻止。先训练 ViT，再配置模型路径。

**模型已训练，为什么网站还显示未就绪？** 检查 `.env` 路径是否真实存在、是否 ViT、是否包含当前流程的来源及预处理信息，以及终端是否覆盖了环境变量。修改后重启服务。

**只有 DeepSeek Key，能否直接识别癌症？** 不能。DeepSeek 接收的是分类和 XAI 摘要，不能替代本项目的图像分类模型。

**报告失败是否丢失图片结果？** 不会。已成功的分类和解释保存到本机；网站继续分析时跳过它们，重试报告。

**阈值和 checkpoint 不匹配怎么办？** 使用该 checkpoint 自己的验证预测重新选择阈值，不要将其他实验的决策文件套用到它上面。

**多个 patch 的阳性数量可以视为患者风险吗？** 当前没有病例级校准或聚合模型，只报告逐图分类统计，不能换算成患者患癌概率。

## 单服务器测试与正式环境

`dev` 对应测试域名 `dev.cancercell.otisliu.com`，`main` 对应正式域名
`cancercell.otisliu.com`。推送和 PR 自动运行 CPU CI；部署在服务器准备完成后通过
`DEPLOY_ENABLED=true` 显式启用。隔离方式、初始化、HTTPS、Secrets 与回滚见
[部署说明](deploy/README.md)。
