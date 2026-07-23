# LAPo — Local Action Bridge with Endpoint Prior

端点先验的局部动作桥：**用未来终点的 latent 作为先验，直接解码出一段动作 chunk**。
基于 **X-VLA**（lerobot 0.4.4 `XVLAPolicy`，Florence2 + DaViT 视觉塔）冻结主干，
包含完整的训练、编排、监控与评估 pipeline（PyTorch + lerobot）。

**核心亮点**

- **Endpoint Prior**：训练时用未来观测帧的 latent 锚定动作轨迹（act-free 监督），
  推理时用学习到的 EndpointPredictor 预测终点——「先定终点，再推动作」；
- **无膨胀捷径**：action-space loss 与 latent 尺度无关 + detached oracle，
  不需要 SIGReg / VICReg / freeze_zt 等正则，latent 幅值零漂移；
- **四阶段课程式训练**：Direct Decoder → Schrödinger Bridge → Endpoint Predictor
  → 联合微调（teacher forcing 渐退），每阶段有明确的训练对象与判断点；
- **多模态动作生成**：IMLE（winner-take-all）+ 薛定谔桥 + PD 反馈力，
  不把多模态示教压成单峰；
- **轻量高效**：可训参数仅 16.4M（X-VLA baseline 的 1/54），
  V100 真机推理 198 ms（比 baseline 快 3.3×）；
- **工程完备**：48h 全自动编排、关键判断点监控告警、oracle/predictor gap 与
  X-VLA baseline 同口径评估、latent 分布分析，105 项单测。

---

## 1. 动机与核心思想

VLA 的常规做法（flow matching / diffusion）把「未来要去哪」压进隐式流场——
看不见、没法当约束、没法检查。LAPo 把未来**显式化**：

> 训练时，用未来第 H 帧观测 `obs_{t+H}` 的 latent `e_t`（oracle endpoint）
> 锚定整段动作轨迹：模型看到「现在在哪」和「一秒后要到哪」，
> 直接桥接出中间的动作序列。
> 推理时真机没有未来帧，换成学习到的 **EndpointPredictor** 预测终点。

endpoint 监督来自**未来观测帧本身**（act-free，不依赖额外标注），
动作监督仍是专家示教。

## 2. 算法详述

### 2.1 符号

| 符号 | 含义 | 维度 |
|---|---|---|
| `z_t` | 当前观测 latent：`encoder(DaViT(obs_t))`，DaViT 冻结、投影头可训 | 192 |
| `e_t` | endpoint latent：`sg(encoder(DaViT(obs_{t+H})))`，detached | 192 |
| `cond` | `ConditionEncoder([z_t, e_t, e_t − z_t])` | 512 |
| `pred` | 动作 chunk（H=30 帧 ≈ 1 秒 @30fps） | `[B, 30, 10]` |
| action | ee6d：xyz(3) + rot6d(6) + gripper(1) | 10 |

### 2.2 数据流（训练，Stage 1）

```
① z_t  = encoder(DaViT(冻结)(obs_t))        ← 可训投影头
② e_t  = sg(encoder(DaViT(obs_{t+H})))      ← oracle endpoint, no_grad
③ cond = ConditionEncoder([z_t, e_t, e_t − z_t])
④ pred = DirectDecoder(cond)                ← [B, H, dim_action]
⑤ loss = action_loss(pred, expert_chunk)    ← action-space, 无膨胀捷径
```

三个关键设计：

- **`rel = e_t − z_t`**：ConditionEncoder 显式编码「到终点的位移方向」，
  decoder 不用自己算。
- **`e_t` 是 detached 的**：encoder 不能通过放大 `z` 的尺度来降 loss——
  action-space loss 与 z 尺度无关，**encoder 无膨胀捷径**，
  因此不需要 SIGReg / VICReg / freeze_zt 这类正则（mag_z 零漂移）。
- **endpoint 利用率可诊断**：用 batch 内 shuffle 的 `e_t` 做对照
  （`loss_with_e` vs `loss_shuffled_e`），直接量化模型到底用没用 endpoint。

### 2.3 动作损失（分通道）

```
L_action = w_xyz · SmoothL1(xyz)           w=1.0
         + w_rot · SmoothL1(rot6d)         w=1.0
         + w_grip · BCE(gripper)           w=2.0   # 稀疏事件，加权
         + w_smooth · MSE(相邻帧差)         w=0.05  # 时序平滑
         + w_disp · SmoothL1(Σxyz delta)   w=0.5   # chunk 总位移 ≈ endpoint 位移
```

### 2.4 四阶段训练

| Stage | endpoint 来源 | decoder | 训练对象（其余冻结） | 目标 |
|---|---|---|---|---|
| 1 | oracle | DirectDecoder | encoder + cond + decoder | 验证条件信息流 |
| 2 | oracle | Schrödinger Bridge | 只训 SB bridge | 多模态动作生成 |
| 3 | predictor | — | 只训 EndpointPredictor | 学会预测终点 |
| 4 | joint（teacher forcing） | SB | predictor 全 lr + bridge ×0.1 | 联合微调，消 oracle-predictor gap |

**Stage 2 — Schrödinger Bridge 动作生成**

- IMLE：从噪声抽 K=4 个候选起点，winner-take-all 认领最近的 mode
  （多模态不压成单峰）；
- 薛定谔力：在专家 chunk 与候选起点之间做三次样条插值得
  `(q_target, v_target, a_target)`，叠加 quartic 噪声包络 `σ(t)`
  （峰值 0.48，两端归零），AccField 学习
  `a = a_target + k_p(q_target − q) + k_d(v_target − v)`（PD 反馈，`k_p=k_d=4`）；
- 推理：从噪声出发 Euler 积分 N=5 步出 clean chunk。

**Stage 3 — EndpointPredictor**

```
e_pred = z_t + delta,  delta = Linear(z_t) + MLP(z_t, lang, progress)
```

- 线性捷径抓主体信号（latent 分布分析：`z_t → delta` 线性 R²≈0.87），
  6 层残差 MLP（隐维 1024）只学剩余非线性修正，head 零初始化起步；
- `lang` = Florence2 图文融合特征（任务意图方向），
  `progress` = proprio（关节状态天然编码轨迹阶段）；
- 损失：`MSE + 0.2·(1 − cos 方向) + 0.1·|幅值差| L1`；
- 最小二乘线性初始化留了开关（`LAPO_LSTSQ_FIT`），但实测 1280 样本时
  线性 R² 仅 0.53（过拟合），默认不用。

**Stage 4 — 联合微调 + teacher forcing**

- 每个样本按概率 `p_oracle` 掷骰子：用 oracle endpoint（detach）还是
  predictor endpoint（带梯度）；
- `p_oracle` 从 0.5 线性退到 0，让 bridge 逐步适应 predictor 的噪声；
- 总损失 = action loss + 0.1 · endpoint 辅助 loss（防 predictor 漂移）；
- 分组学习率：predictor 全 lr，bridge/condition encoder ×0.1（防振荡）。

### 2.5 推理（真机无 oracle）

```
z_t = encoder(DaViT(obs_t))
e_t = EndpointPredictor(z_t, lang(obs_t, 指令), proprio)
cond = ConditionEncoder([z_t, e_t, e_t − z_t])
chunk = SB.sample(cond)        # Euler 积分 5 步（或 DirectDecoder 一步出）
```

## 3. 与 X-VLA 的关系

- **主干直接复用 `lerobot/xvla-base`**：Florence2 VLM 冻结，
  DaViT 视觉塔出 4096 维池化特征喂给可训 encoder；语言侧复用
  Florence2 的图文融合 encoder（Stage 3 predictor 的指令特征）。
- **batch 契约复用 `modeling_xvla`**：`LapoPolicy` 直接用
  `OBS_LANGUAGE_TOKENS / OBS_STATE / ACTION / pad_vector / resize_with_pad`，
  数据管线与 X-VLA ee6d 路径一致（BART tokenizer、双视角图像）。
- **action_space / proprio 维度**从加载的 XVLAPolicy 读取，
  `image_projection` 权重从 xvla-base checkpoint 注入修复。

即：LAPo 是在 X-VLA 冻结主干之上，把动作生成头从 flow matching 换成
「endpoint prior + 局部动作桥」。

## 4. 训练动态与损失设计

### 4.1 监控指标（每步写入 metrics.jsonl）

| 指标 | 含义 | 健康标准 |
|---|---|---|
| `loss_action` | 分通道动作损失（Stage 1/4） | 告警线 0.15（超过视为 bridge 崩） |
| `loss_imle` / `loss_force` | SB 的 IMLE / 薛定谔力损失（Stage 2） | — |
| `ep_mse` / `ep_dir_loss` / `ep_scale_loss` | endpoint 预测误差（Stage 3/4） | 见下方判断点 |
| `mag_z_t` / `mag_e` | latent 幅值（膨胀监控） | `mag_z_t` 阈值 8.0，action-space loss 下零漂移 |
| `endpoint_gain` / `endpoint_delta` | shuffle 对照：endpoint 利用率 | `endpoint_delta > 0` = endpoint 在被利用 |
| `grip_acc` | gripper 二分类准确率 | 稀疏事件的关键指标 |
| `p_oracle` / `oracle_ratio` | Stage 4 teacher forcing 进度 | 0.5 → 0 线性 |

### 4.2 Stage 4 训练判断点（自动监控，`scripts/lapo_stage4_monitor.sh`）

| step | 期望 |
|---|---|
| 500 | warmup 结束，p_oracle 开始衰减 |
| 1000 | `ep_mse < 0.9` |
| 3000 | `ep_mse < 0.6`，action 稳定 |
| 5000 | `ep_mse < 0.4` |
| 10000 | 训练结束（p_oracle → 0） |

异常告警：`action_loss > 0.15`（bridge 崩）、`ep_mse` 连续 3 个判断点
单调上升（predictor 退化）、`mag_z_t` 超阈（latent 膨胀）。

### 4.3 评估口径

- **oracle vs predictor gap**（`scripts/lapo_eval.py`）：同一 checkpoint，
  分别用 oracle endpoint（性能上限）和 predictor endpoint（实际部署）
  算 action loss，gap 越小说明 predictor 越接近「免费的未来信息」；
  同时输出分通道 MSE（xyz/rot/grip）和 SB sample variance（多模态性）。
- **与 X-VLA baseline 同口径对比**（`scripts/lapo_vs_xvla_eval.py`）：
  同数据集、同预处理，对比 LAPo 与 X-VLA flow matching baseline 的动作质量。
- **latent 分布分析**（`scripts/analyze_latent_dist.py`）：R² + PCA，
  验证 latent 空间结构（线性可预测性、无坍缩）。

### 4.4 结果

与 X-VLA flow matching baseline 同数据集、同预处理对比
（`scripts/lapo_vs_xvla_eval.py`，验证集分通道指标）：

| 指标 | LAPo (s4 @2499 步) | X-VLA (afe5 @34999 步) | LAPo 表现 |
|---|---|---|---|
| loss_xyz（位置，SmoothL1） | 0.0531 | 0.0167 | 高 3.2× |
| loss_rot（旋转 rot6d，SmoothL1） | 0.1878 | 0.1632 | 高 1.15× |
| loss_grip（夹爪 BCE） | 0.6801 | 0.6932 | **略优** |
| grip_acc（夹爪准确率） | 98.02% | 96.67% | **+1.35pp** |
| loss_endpoint_disp（端点位移） | 7.436 | 3.887 | 高 1.9× |
| sample_var（采样方差） | 6.57e-4 | 6.29e-5 | 高 10×（多模态设计，见下） |

推理与部署效率（V100 真机实测）：

| 指标 | LAPo | X-VLA | 对比 |
|---|---|---|---|
| 推理延迟（V100 真机） | 198 ms | 660 ms | **快 3.3×** |
| 可训参数 | 16.4M | 879.5M | **小 54×** |

效率来自结构：VLM 主干全冻结，可训部分只有 encoder 投影头 +
ConditionEncoder + decoder/predictor；动作生成是 SB Euler 积分 5 步
（或 DirectDecoder 一步出整段 chunk），比 flow matching 的多步去噪采样便宜。

解读：

- **训练量差 14 倍**（2.5k vs 35k 步）：LAPo 用 ~7% 的步数，旋转通道已追到
  1.15×，夹爪 loss 与准确率两项反超——稀疏关键事件的学习效率更高。
- **xyz / 端点位移的差距**主要来自 predictor endpoint 精度：ep 误差直接
  传导为位置误差，Stage 4 在 p_oracle 退到 0 后仍有收敛空间
  （监控判断点设到 10k 步）。
- **sample_var 高 10× 是设计使然而非纯劣势**：LAPo 用 IMLE winner-take-all
  保留多模态（不同示教路径不压成单峰），flow matching baseline 天然趋于
  低方差。该指标需结合任务多模态性解读。
- 以上为验证集动作损失，**真机成功率评估（rollout）是下一步**。

复现：`scripts/lapo_eval.py`（oracle vs predictor gap）与
`scripts/lapo_vs_xvla_eval.py`（baseline 对比），产出 `lapo_eval_report.json`。

## 5. 仓库结构

```
lapo/
├── lapo/
│   ├── paths.py                 # 存储根解析（$LAPO_HOME）
│   └── train/                   # 配置驱动训练框架
│       ├── config.py / registry.py / strategy.py / engine.py
│       ├── compat.py            # lerobot 0.4.4 兼容层
│       ├── cli.py               # lapo-train CLI
│       ├── policies/
│       │   ├── lapo/            # LAPo: config / model / policy
│       │   ├── sb/              # SB 组件库（Encoder / AccField / IMLE / 样条）
│       │   └── xvla_tokenizer.py
│       ├── strategies/
│       │   ├── lapo.py          # LapoStrategy（分阶段冻结 + 分组 LR + TF 调度）
│       │   ├── sbvla.py         # SB-VLA 策略（VLM 加载 + 视觉投影修复）
│       │   └── xvla_sp.py       # X-VLA soft prompt 差异学习率
│       ├── flows/schrodinger_bridge.py
│       ├── artifacts/ services/ distributed.py env_check.py
├── configs/                     # LAPo 4 个 stage + SB final + X-VLA baseline 配置
├── scripts/                     # 48h 编排 / stage 监控 / oracle vs predictor 评估
└── tests/                       # 105 项单测（核心逻辑无 torch 环境可跑）
```

## 6. 用法

```bash
pip install -e ".[dev]"

lapo-train env                                          # 环境自检
lapo-train train --config configs/openarm_pick_place_green277_lapo_stage1.yaml

# 多卡（torchrun，FSDP；--ddp 切 DDP）
torchrun --nproc_per_node=3 -m lapo.train --config configs/...stage1.yaml

# 全自动 4 阶段（含监控与异常告警）
bash scripts/lapo_48h_full.sh
```

依赖版本约束：lerobot `>=0.4.4,<0.5.0`、transformers `>=4.57.1,<5.0.0`、
torch `>=2.6.0,<2.11.0`。lerobot 0.4.4 的已知问题（draccus `type` 字段、
strict-load shape 过滤、双路径 import）集中在 `lapo/train/compat.py` 处理，
主路径零 hack。
