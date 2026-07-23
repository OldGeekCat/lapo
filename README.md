# LAPo — Local Action Bridge with Endpoint Prior

端点先验的局部动作桥：**用未来终点的 latent 作为先验，直接解码出一段动作 chunk**。
实验基于 **X-VLA**（lerobot 0.4.4 `XVLAPolicy`，Florence2 + DaViT 视觉塔）展开，
代码整理自 `lr` monorepo（分支 `feat/web-redesign-xvla-norm`）的 `lrt` 训练框架。

## LAPo 的解法

### 核心思想

训练时用一个 **oracle endpoint**——未来第 H 帧观测 `obs_{t+H}` 的 latent `e_t`——
锚定整段动作轨迹：模型看到「现在在哪」和「一秒后要到哪」，直接桥接出中间的
动作序列。推理时真机没有未来帧，换成学习到的 **EndpointPredictor** 预测终点。
即「先定终点，再推动作」（predict-then-infer）。

### 数据流（训练，Stage 1）

```
① z_t  = encoder(DaViT(冻结)(obs_t))        ← 可训投影头
② e_t  = sg(encoder(DaViT(obs_{t+H})))      ← oracle endpoint, no_grad
③ cond = ConditionEncoder([z_t, e_t, e_t - z_t])
④ pred = DirectDecoder(cond)                ← [B, H, dim_action]
⑤ loss = action_loss(pred, expert_chunk)    ← action-space, 无膨胀捷径
```

### 关键设计

- **`rel = e_t - z_t`**：ConditionEncoder 显式编码「到终点的位移方向」，
  decoder 不用自己算。
- **`e_t` 是 detached 的**：encoder 不能通过放大 z 来降 loss；
  action-space loss 与 z 的尺度无关 → **encoder 无膨胀捷径**，
  因此不需要 SIGReg / VICReg / freeze_zt 这类正则。
- **endpoint 利用率可诊断**：每个 val 步用 batch 内 shuffle 的 `e_t` 对比
  （`loss_with_e` / `loss_shuffled_e` / `endpoint_gain`），
  直接量化模型到底用没用 endpoint 信息。

### 动作表示（ee6d）

`dim_action = 10` = xyz(3) + rot6d(6) + gripper(1)；`chunk_size = 30`
（30fps ≈ 1 秒窗口）；`horizon = 30`（endpoint 取 1 秒后的观测帧）。
action loss 分通道：xyz/rot6d SmoothL1 + gripper BCE + 时序平滑 + xyz 累计位移。

## 4 阶段训练

| Stage | endpoint 来源 | decoder | 训练对象 | 备注 |
|---|---|---|---|---|
| 1 | oracle (`obs_{t+H}`) | DirectDecoder | encoder + cond + decoder | 验证条件信息流，action loss |
| 2 | oracle | Schrödinger Bridge | 只训 SB bridge | 冻 encoder + cond；IMLE + 薛定谔力 |
| 3 | predictor | — | 只训 EndpointPredictor | 冻其余；MSE + 方向 cos + 幅值 L1 |
| 4 | joint (teacher forcing) | SB | predictor + bridge 联合微调 | p_oracle 0.5→0 渐退；predictor 全 lr、bridge ×0.1 |

- **Stage 2 的 SB 动作生成**：AccField 加速度场 + IMLE（K=4 候选
  winner-take-all）+ 三次样条轨迹目标 + quartic 噪声包络 + PD 反馈力；
  推理时从噪声 Euler 积分 N=5 步出 clean chunk。
- **Stage 3 的 EndpointPredictor**：`e_pred = z_t + delta`，
  `delta = Linear(z_t) + MLP(z_t, 语言特征, proprio progress)`——
  线性捷径抓主体信号（分析显示 z_t→delta 线性 R² 很高），
  6 层残差 MLP 只学非线性修正；progress 用 proprio（关节状态天然编码轨迹阶段）。
- **Stage 4**：每样本按概率 p_oracle 掷骰子用 oracle 还是 predictor 的 endpoint，
  p_oracle 从 0.5 线性退到 0，让 bridge 逐步适应 predictor 的噪声；
  另加 0.1 权重的 endpoint 辅助 loss 防 predictor 漂移。

## 为什么基于 X-VLA

- **主干直接复用 `lerobot/xvla-base`**：Florence2 VLM 冻结，
  DaViT 视觉塔出 4096 维池化特征喂给可训 encoder；语言侧复用
  Florence2 的图文融合 encoder 出指令特征（Stage 3 predictor 用）。
- **batch 契约复用 `modeling_xvla`**：`LapoPolicy` 直接用
  `OBS_LANGUAGE_TOKENS / OBS_STATE / ACTION / pad_vector / resize_with_pad`，
  数据管线与 X-VLA ee6d 路径一致（BART tokenizer、双视角图像）。
- **action_space / proprio 维度**从加载的 XVLAPolicy 读取，
  `image_projection` 权重从 xvla-base checkpoint 注入修复。

即：LAPo 是在 X-VLA 冻结主干之上，把动作生成头从 flow matching 换成
「endpoint prior + 局部动作桥」的实验。

## 仓库结构

```
lapo/
├── lapo/
│   ├── paths.py                 # 存储根解析（$LAPO_HOME）
│   └── train/                   # 训练框架（整理自 lr/lrt）
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
├── configs/                     # LAPo 4 个 stage 的训练配置
├── scripts/                     # 48h 编排 / stage 监控 / oracle vs predictor 评估
└── tests/
```

## 用法

```bash
pip install -e ".[dev]"

lapo-train env                                          # 环境自检
lapo-train train --config configs/openarm_pick_place_green277_lapo_stage1.yaml

# 多卡（torchrun，FSDP；--ddp 切 DDP）
torchrun --nproc_per_node=3 -m lapo.train --config configs/...stage1.yaml
```

依赖与 `lr` 一致：lerobot `>=0.4.4,<0.5.0`、transformers `<5.0.0`、torch `>=2.6.0`。
兼容 hack 集中在 `lapo/train/compat.py`（draccus `type` 字段、strict-load
shape 过滤、双路径 import）。
