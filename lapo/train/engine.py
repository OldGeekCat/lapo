"""TrainingEngine: 串联 strategy/registry/artifacts/compat 的训练骨架。

负责：写静态产物 → 调 strategy.train_loop（默认 standard_loop）→ 收尾产物。
strategy 负责真正的训练决策（build_policy/optimizer/loss/...），engine 提供
基础设施（dataloader 迭代、产物写出、设备搬运）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from lapo.train.artifacts.writer import ArtifactWriter
from lapo.train.config import RunConfig
from lapo.train.strategy import StepContext, TrainStrategy


class TrainingEngine:
    """一次训练 run 的执行器。

    Args:
        cfg: 已解析的 RunConfig。
        strategy: 已实例化的 TrainStrategy。
        run_dir: 产物输出目录（$LAPO_HOME/outputs/<run_id>）。
        registry: Registry 实例（当前仅用于 trait 校验后的引用，不在 engine 内再校验）。
        ds_meta: dataset.meta（写 dataset_info 用）。
        dataloader: 可迭代的 batch 来源。
        device: 训练设备字符串。
    """

    def __init__(self, cfg: RunConfig, *, strategy: TrainStrategy,
                 run_dir: Path, registry: Any, ds_meta: Any,
                 dataloader: Any = None, device: str = "cpu",
                 env_info: Optional[dict] = None,
                 delta_timestamps: Optional[dict] = None,
                 val_dataloader: Any = None):
        self.cfg = cfg
        self.strategy = strategy
        self.run_dir = Path(run_dir)
        self.registry = registry
        self.ds_meta = ds_meta
        self.dataloader = dataloader
        self.device = device
        self.val_dataloader = val_dataloader
        # 多卡下：非 rank0 不写产物（避免冲突），用 no-op writer
        from lapo.train.distributed import is_main_process
        if is_main_process():
            self.writer = ArtifactWriter(self.run_dir)
        else:
            self.writer = _NoOpWriter()
        self._env_info = env_info
        self._delta_timestamps = delta_timestamps
        self._batch_iter: Any = None

    def run(self) -> list[dict]:
        """完整生命周期：写静态产物 → 训练循环 → 收尾。

        异常时把 status 写成 failed 并记录 error，然后 re-raise（不吞异常）。
        """
        self.writer.write_run_json(
            run_id=self.run_dir.name, status="running",
            policy=self.cfg.policy_name, strategy=self.cfg.strategy_name,
            dataset=self.cfg.dataset.repo_id,
            num_steps=self.cfg.training.num_steps, device=self.device,
        )
        self.writer.write_run_config(self.cfg)
        if self._env_info:
            self.writer.write_env_info(self._env_info)
        self.writer.write_dataset_info(
            self.ds_meta, repo_id=self.cfg.dataset.repo_id,
            delta_timestamps=self._delta_timestamps,
        )
        try:
            infos = self.strategy.train_loop(self)
            from lapo.train.artifacts.writer import _now_iso
            self.writer.update_run_json(status="completed", ended_at=_now_iso())
            self.writer.write_diff(baseline_run=None, changes=[])
            return infos
        except Exception as e:
            from lapo.train.artifacts.writer import _now_iso
            self.writer.update_run_json(
                status="failed", error=str(e), ended_at=_now_iso(),
            )
            raise

    def standard_loop(self) -> list[dict]:
        """默认训练循环（TrainStrategy.train_loop 的默认实现）。

        policy = strategy.build_policy → optimizer = build_optimizer →
        每步: next_batch → compute_loss → backward → clip_grad → step →
              on_step_end(可写 ctx.metrics) → append_metrics → should_save。
        结束存 final checkpoint。
        """
        import torch

        policy = self._build_policy()

        # 多卡 wrap：DDP 优先（V100 fp16），其次 FSDP（A100 bf16）。单卡正常搬 device。
        if self.cfg.training.ddp:
            from lapo.train.distributed import wrap_ddp
            policy = wrap_ddp(
                policy, self.device,
                grad_checkpoint=self.cfg.training.grad_checkpoint,
                dtype=self.cfg.training.dtype,
            )
        elif self.cfg.training.fsdp:
            from lapo.train.distributed import wrap_fsdp
            policy = wrap_fsdp(
                policy, self.device,
                sharding=self.cfg.training.fsdp_sharding,
                grad_checkpoint=self.cfg.training.grad_checkpoint,
                dtype="float32",
            )
        else:
            policy.to(self.device)

        opt = self.strategy.build_optimizer(self._unwrap(policy))
        sched = self.strategy.build_scheduler(opt)

        # 断点续训：加载 checkpoint 恢复 model 权重 + 起始 step
        start_step = 0
        resume_from = self.cfg.training.resume_from
        if resume_from:
            start_step = self._load_checkpoint(policy, opt, sched, resume_from)

        # 模型产物（build_policy 后写）
        self.writer.write_model_info(
            policy, policy=self.cfg.policy_name,
            base_model=self.cfg.policy_overrides.get("base_model"),
            param_groups=[
                {"name": g.get("name", f"g{i}"),
                 "lr_scale": g["lr"] / self.cfg.training.lr}
                for i, g in enumerate(opt.param_groups)
            ],
            dtype=self.cfg.training.dtype,
        )
        self.writer.write_model_graph(self._extract_graph(policy),
                                      policy=self.cfg.policy_name)

        infos: list[dict] = []
        grad_accum = self.cfg.training.grad_accumulation_steps
        accum_count = 0  # 独立累积计数器，不依赖 step 取模（resume 安全）
        for step in range(start_step, self.cfg.training.num_steps):
            batch = self._to_device(self._next_batch())
            policy.train()
            loss = self.strategy.compute_loss(policy, batch)
            # 梯度累积：loss 除以累积步数，保证梯度大小一致
            (loss / grad_accum).backward()
            accum_count += 1

            # 只在累积完成时才更新参数
            is_update_step = accum_count >= grad_accum
            is_last = (step + 1) == self.cfg.training.num_steps

            if is_update_step or is_last:
                if self.cfg.training.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        policy.parameters(), self.cfg.training.grad_clip_norm,
                    )
                # on_step_end 移到 zero_grad 前: 此时 grad 已累积完 + clip 后,
                # strategy 可读取各模块 grad norm（诊断用）。step/zero 在 on_step_end 后。
                ctx = StepContext(step=step, policy=policy, optimizer=opt, loss=loss,
                                  batch=batch, metrics={})
                self.strategy.on_step_end(step, ctx)
                opt.step()
                opt.zero_grad()
                if sched:
                    sched.step()
                accum_count = 0  # 重置累积计数器
            else:
                # 非累积完成步: 也调 on_step_end（记 loss/metrics）, 但此时 grad 是中间态
                ctx = StepContext(step=step, policy=policy, optimizer=opt, loss=loss,
                                  batch=batch, metrics={})
                self.strategy.on_step_end(step, ctx)

            # 验证集评估（val_every 步触发一次，用 rank0 避免多卡重复）
            # 多卡下：评估用裸 policy（不经 DDP forward），rank1+ 不参与推理，但必须
            # 在评估前后 barrier 等待——否则 rank1+ 跑进下一轮 backward → DDP all-reduce
            # → 等 rank0 → NCCL timeout SIGABRT（这是提交历史里反复出现的 DDP 崩溃根因）。
            val_every = self.cfg.training.val_every
            if (val_every > 0 and self.val_dataloader is not None
                    and (step + 1) % val_every == 0):
                import torch.distributed as dist
                _is_ddp = dist.is_available() and dist.is_initialized()
                if _is_ddp:
                    dist.barrier()  # 所有 rank 到齐后再开始评估
                from lapo.train.distributed import is_main_process
                if is_main_process():
                    val_metrics = self._validate(policy)
                    ctx.metrics.update(val_metrics)
                if _is_ddp:
                    dist.barrier()  # rank0 评估完才放行 rank1+ 继续训练

            # save checkpoint + action MSE（和 save 同周期，save 后做 MSE 推理）
            # 多卡下同样需要 barrier 包裹评估阶段（原因同上）。
            if self.strategy.should_save(step):
                self.writer.save_checkpoint(self._unwrap(policy), step=step, optimizer=opt)
                import torch.distributed as dist
                _is_ddp = dist.is_available() and dist.is_initialized()
                if _is_ddp:
                    dist.barrier()
                from lapo.train.distributed import is_main_process
                if is_main_process():
                    eval_metrics = self._eval_action_mse(policy)
                    if eval_metrics is not None:
                        ctx.metrics.update(eval_metrics)
                if _is_ddp:
                    dist.barrier()

            log_every = self.cfg.training.log_every
            lr_groups = {g.get("name", f"g{i}"): g["lr"]
                         for i, g in enumerate(opt.param_groups)}
            # 记录 metrics（含验证/save 步的 val_loss + val_action_mse）
            if (step + 1) % log_every == 0 or is_last or ctx.metrics:
                self.writer.append_metrics(
                    step=step + 1, loss=loss.detach(), extra={"lr_groups": lr_groups, **ctx.metrics},
                )

            # 散热休息：每 val_every 步检查 GPU 温度（和验证同周期）
            if val_every > 0 and (step + 1) % val_every == 0:
                self._maybe_cooldown(policy, opt, step)

            self.writer.update_run_json(current_step=step + 1)
            infos.append({"step": step, "loss": loss.item(), "lr_groups": lr_groups})

        # 最终 checkpoint
        self.writer.save_checkpoint(self._unwrap(policy), optimizer=opt)
        return infos

    # ---- 可被子类/mock 覆盖的钩子 ----
    def _build_policy(self) -> Any:
        return self.strategy.build_policy(self.cfg, self.ds_meta)

    def _unwrap(self, policy) -> Any:
        """剥去 FSDP/DDP wrapper，返回裸 module。

        用于 build_optimizer（param name 无前缀）和 save_checkpoint（state_dict
        key 无前缀，lerobot from_pretrained 兼容）。
        """
        from lapo.train.distributed import unwrap
        return unwrap(policy)

    def _load_checkpoint(self, policy, optimizer, scheduler, ckpt_dir) -> int:
        """从 checkpoint 恢复 model 权重，返回起始 step（1-based）。

        加载 model.safetensors 到裸 policy，跳过已训练的 step。
        scheduler 的进度靠 start_step 在 LR lambda 里自然恢复（LambdaLR 按步推进）。
        """
        import json
        import sys
        from pathlib import Path

        ckpt_dir = Path(ckpt_dir)
        ts_path = ckpt_dir / "training_state.json"
        model_path = ckpt_dir / "model.safetensors"

        if not model_path.exists():
            print(f"[lapo.train] ⚠️ checkpoint 不存在: {model_path}，从头训练", file=sys.stderr)
            return 0

        # 加载 model 权重到裸 policy（shape 过滤：跳过维度不匹配的层）
        # 和 compat.py build_policy_for 同逻辑，因为 PyTorch 2.10 的 strict=False
        # 不跳过 shape mismatch，只跳过 missing/unexpected
        raw = self._unwrap(policy)
        from safetensors.torch import load_file
        from lapo.train.compat import load_state_dict_shape_filtered
        state_dict = load_file(str(model_path))
        load_state_dict_shape_filtered(raw, state_dict)

        # 读 training_state 获取 step
        start_step = 0
        if ts_path.exists():
            ts = json.loads(ts_path.read_text())
            start_step = (ts.get("step") or 0) + 1  # 1-based：step_499 → 从 500 开始
            print(f"[lapo.train] 📂 resume from {ckpt_dir.name} (step {start_step - 1} → 从 {start_step} 继续)",
                  file=sys.stderr)

        # 推进 scheduler 到正确位置（LambdaLR 按 optimizer step 计算 lr）。
        # 注意：训练循环里 scheduler 只在 optimizer step 时推进（每 grad_accum 步一次），
        # 所以 resume 时推进次数应是 optimizer step 数（start_step // grad_accum），
        # 而非按数据 step 数——否则 grad_accum>1 时 LR 会被多推进 grad_accum 倍，落点错误。
        if scheduler is not None and start_step > 0:
            n_optim_steps = start_step // max(self.cfg.training.grad_accumulation_steps, 1)
            for _ in range(n_optim_steps):
                scheduler.step()

        return start_step

    def _iter_batches(self):
        """迭代 dataloader。耗尽时重建并（分布式下）调 sampler.set_epoch。"""
        if self._batch_iter is None:
            self._batch_iter = iter(self.dataloader)
            self._epoch = 0
        return self._batch_iter

    def _next_batch(self):
        """取下一个 batch，耗尽时重建迭代器（触发 epoch 切换）。"""
        try:
            return next(self._iter_batches())
        except StopIteration:
            # epoch 耗尽：重建 + 分布式下 set_epoch（保证每 epoch shuffle 不同）
            self._epoch = getattr(self, "_epoch", 0) + 1
            sampler = getattr(self.dataloader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
            self._batch_iter = iter(self.dataloader)
            return next(self._iter_batches())

    def _validate(self, policy, max_batches: int = 10) -> dict:
        """跑一轮验证集 val_loss。

        用 unwrap 后的裸 policy forward（不经 DDP），避免 rank 间 collective
        不匹配导致 NCCL SIGABRT。strategy.compute_loss 传入裸 policy。
        """
        import torch
        raw = self._unwrap(policy)
        raw.eval()
        total_loss, count = 0.0, 0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._to_device(batch)
                loss = self.strategy.compute_loss(raw, batch)
                total_loss += loss.item()
                count += 1
                if count >= max_batches:
                    break
        raw.train()
        return {"val_loss": total_loss / max(count, 1)}

    def _eval_action_mse(self, policy, max_batches: int = 3) -> dict[str, float] | None:
        """在 save checkpoint 之后做 action 评估（和 DDP reducer 解耦）。

        用 unwrap 后的裸 policy + 完全 no_grad，不经过 DDP forward。
        返回分维度指标 dict（写入 metrics，跨版本可比）：
          - val_joints_mse:       关节维 MSE（归一化空间，纯回归精度）
          - val_gripper_acc:      夹爪维准确率（logits→sigmoid≥0.5 判开/合，和 BCE 对齐）
          - val_traj_straight:    轨迹直度（flow 积分路径弯曲度，越低越直，0=完美直线）
          - val_sample_var:       多次采样一致性（同输入 3 次采样的方差，越低越稳）
          - val_chunk_smooth:     chunk 内时序平滑度（二阶差分 L2，越低越平滑）

        注意：夹爪维不参与 MSE（BCE logits 和 0/1 target 做 MSE 无意义）。
        若 policy 无 gripper_idx（非 openarm_gripper action space），回退到全维 MSE。
        """
        import torch
        raw = self._unwrap(policy)
        if not hasattr(raw, "predict_action_chunk"):
            return None
        if self.val_dataloader is None:
            return None
        raw.eval()

        # 检测 action space 是否有 gripper_idx（openarm_gripper / EE6DActionSpace 等）。
        # 注意：EE6DActionSpace.gripper_idx=(9,19) 是双臂 20 维布局，对单臂 10 维数据，
        # 索引 19 会越界。循环内的 guard `gi < min(p.shape[-1], t.shape[-1])` 会跳过它，
        # 但若 acc 分母仍用 len(gripper_idx)=2，指标会被腰斩到 ≤50%（见下方 total_gripper_dims）。
        # joints_idx 保守取 range(8)：对 10 维数据取 dim0~7（漏 dim8，但不越界、不改原行为）。
        gripper_idx = getattr(getattr(raw.model, "action_space", None), "gripper_idx", ())
        has_gripper = len(gripper_idx) > 0
        joints_idx = tuple(i for i in range(8) if i not in set(gripper_idx)) if has_gripper else None

        # 检测是否 XVLA flow-matching head（有 generate_actions 且 model 有 transformer
        # 属性 = SoftPromptedTransformer flow-matching head）。SB head 也有 generate_actions
        # 但内部是薛定谔桥积分，无 transformer 属性 → 不 hook（否则用 FM 逻辑跑 SB 必崩）。
        has_fm_transformer = hasattr(raw.model, "transformer")
        has_generate_actions = has_fm_transformer and hasattr(raw.model, "generate_actions")

        # 检测是否 SmolVLA（有 denoise_step 在 model.model 上）
        smolvla_inner = getattr(raw.model, "model", None)  # VLAFlowMatching
        smolvla_denoise = getattr(smolvla_inner, "denoise_step", None) if smolvla_inner else None
        has_smolvla_denoise = smolvla_denoise is not None

        # --- hook 积分中间步（算 trajectory straightness）---
        _intermediate_actions = []

        # XVLA: hook generate_actions
        _orig_gen = getattr(raw.model, "generate_actions", None) if has_generate_actions else None

        if has_generate_actions:
            def _hooked_generate(self_model, **kwargs):
                """复刻 generate_actions 逻辑但记录每步输出。"""
                steps = kwargs["steps"]
                action_dim = self_model.dim_action
                batch_size = kwargs["input_ids"].shape[0]
                device = kwargs["proprio"].device
                target_dtype = kwargs["proprio"].dtype
                enc = self_model.forward_vlm(kwargs["input_ids"], kwargs["image_input"], kwargs["image_mask"])
                x1 = torch.randn(batch_size, self_model.chunk_size, action_dim, device=device, dtype=target_dtype)
                action = torch.zeros_like(x1)
                mid_actions = []
                for i in range(steps, 0, -1):
                    t = torch.full((batch_size,), i / steps, device=device, dtype=target_dtype)
                    x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
                    proprio_m, x_t_m = self_model.action_space.preprocess(kwargs["proprio"], x_t)
                    action = self_model.transformer(
                        domain_id=kwargs["domain_id"], action_with_noise=x_t_m,
                        proprio=proprio_m, t=t, **enc)
                    mid_actions.append(action.detach().cpu())
                _intermediate_actions.append(mid_actions)
                return self_model.action_space.postprocess(action)

            import types
            raw.model.generate_actions = types.MethodType(_hooked_generate, raw.model)

        elif has_smolvla_denoise:
            # SmolVLA: hook denoise_step 记录每步的 x_t（积分状态）
            _orig_denoise = smolvla_inner.denoise_step
            _smolvla_x_t_history = []

            def _hooked_denoise(self_inner, x_t, **kwargs):
                v_t = _orig_denoise(x_t=x_t, **kwargs)
                _smolvla_x_t_history.append(x_t.detach().cpu())
                _intermediate_actions.append([x.detach().cpu() for x in _smolvla_x_t_history])
                # 清空避免重复累积（每次 predict 会重新填）
                _smolvla_x_t_history.clear()
                _smolvla_x_t_history.append(x_t.detach().cpu())
                return v_t

            import types as _types
            smolvla_inner.denoise_step = _types.MethodType(_hooked_denoise, smolvla_inner)

        total_joints_mse, total_gripper_correct, count = 0.0, 0, 0
        total_gripper_dims = 0  # 实际命中（未越界）的 gripper 维累计数，用作 acc 分母
        total_straight, total_sample_var, total_smooth = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._to_device(batch)
                try:
                    inf_batch = self.strategy.preprocess(dict(batch))
                    true = inf_batch.get("action")

                    # --- 多次采样一致性：同输入采样 3 次 ---
                    preds_multi = []
                    for _ in range(3):
                        p = raw.predict_action_chunk(inf_batch)
                        preds_multi.append(p)
                    pred = preds_multi[0]  # 第一次的用于 mse/acc

                    if true is not None:
                        p = pred.float()
                        t = true.float()
                        if has_gripper and joints_idx is not None:
                            jm = torch.nn.functional.mse_loss(
                                p[..., joints_idx], t[..., joints_idx]).item()
                            total_joints_mse += jm
                            for gi in gripper_idx:
                                if gi < min(p.shape[-1], t.shape[-1]):
                                    pred_open = (torch.sigmoid(p[..., gi]) >= 0.5).float()
                                    true_open = t[..., gi]
                                    total_gripper_correct += (pred_open == true_open).float().mean().item()
                                    total_gripper_dims += 1
                        else:
                            min_dim = min(p.shape[-1], t.shape[-1])
                            total_joints_mse += torch.nn.functional.mse_loss(
                                p[..., :min_dim], t[..., :min_dim]).item()

                    # --- 轨迹直度：积分中间步的路径弯曲度（XVLA + SmolVLA）---
                    # 从最后一次 hook 调用取中间步
                    if _intermediate_actions:
                        mid = _intermediate_actions[-1]  # list of [B,T,D]
                        if len(mid) >= 3:
                            stacked = torch.stack(mid, dim=0)  # [steps, B, T, D]
                            # 用关节维算（跳过 gripper）
                            idx = joints_idx if (joints_idx and has_gripper) else tuple(range(stacked.shape[-1]))
                            traj = stacked[..., idx]  # [steps, B, T, len(idx)]
                            # 路径总长度 vs 起终点直线距离
                            seg_lens = (traj[1:] - traj[:-1]).norm(dim=-1).sum(dim=0)  # [B,T]
                            direct = (traj[-1] - traj[0]).norm(dim=-1).clamp(min=1e-6)  # [B,T]
                            straightness = (seg_lens / direct - 1.0).mean().item()  # 0=直线
                            total_straight += straightness

                    # --- 多次采样一致性：3 次采样的方差 ---
                    if len(preds_multi) >= 2:
                        stack = torch.stack(preds_multi, dim=0)  # [n_samples, B, T, D]
                        idx = joints_idx if (joints_idx and has_gripper) else tuple(range(stack.shape[-1]))
                        sample_var = stack[..., idx].var(dim=0).mean().item()
                        total_sample_var += sample_var

                    # --- chunk 时序平滑度：输出 chunk 的二阶差分 L2 ---
                    p_out = pred[..., (joints_idx if (joints_idx and has_gripper) else slice(None))]
                    if p_out.shape[1] >= 3:
                        accel = p_out[:, 2:] - 2 * p_out[:, 1:-1] + p_out[:, :-2]
                        smoothness = (accel ** 2).mean().item()
                        total_smooth += smoothness

                    count += 1
                except Exception as e:
                    import sys
                    print(f"[lapo.train] ⚠️ action eval 跳过: {e}", file=sys.stderr)
                if count >= max_batches:
                    break

        # 恢复原始方法
        if has_generate_actions and _orig_gen is not None:
            raw.model.generate_actions = _orig_gen
        elif has_smolvla_denoise and _orig_denoise is not None:
            smolvla_inner.denoise_step = _orig_denoise
        raw.train()
        if count == 0:
            return None
        result = {
            "val_joints_mse": total_joints_mse / count,
            "val_sample_var": total_sample_var / count,
            "val_chunk_smooth": total_smooth / count,
        }
        # traj_straight（XVLA + SmolVLA 都能采集到积分中间步）
        if total_straight > 0:
            result["val_traj_straight"] = total_straight / count
        if has_gripper and total_gripper_dims > 0:
            # 分母用实际命中（未越界）的 gripper 维累计数，而不是 len(gripper_idx)：
            # EE6DActionSpace 对单臂数据 dim19 越界，若用原 n_g=2 会把指标腰斩到 ≤50%。
            result["val_gripper_acc"] = total_gripper_correct / total_gripper_dims
        return result

    def _maybe_cooldown(self, policy, optimizer, step) -> None:
        """散热看门狗：任一 GPU 温度 ≥ cooldown_temp 时，先存 checkpoint 再所有 rank 一起 sleep。

        barrier 保证所有 rank 同步进入/退出休息，不触发 NCCL timeout。
        sleep 后温度自然下降，继续训练。
        """
        import sys
        threshold = self.cfg.training.cooldown_temp
        rest_sec = self.cfg.training.cooldown_seconds
        if threshold <= 0 or rest_sec <= 0:
            return

        max_temp = self._max_gpu_temp()

        # 分布式下：rank0 算 max_temp，广播给所有 rank
        is_dist = self.cfg.training.fsdp or self.cfg.training.ddp
        if is_dist:
            import torch.distributed as dist
            if dist.is_initialized():
                import torch
                temp_tensor = torch.tensor(max_temp if dist.get_rank() == 0 else 0,
                                           device=f"cuda:{dist.get_rank()}" if torch.cuda.is_available() else "cpu")
                dist.broadcast(temp_tensor, src=0)
                max_temp = temp_tensor.item()
                if max_temp >= threshold:
                    if dist.get_rank() == 0:
                        print(f"[lapo.train] 🌡️ GPU {max_temp}°C ≥ {threshold}°C，"
                              f"先存 checkpoint 再散热休息 {rest_sec}s", file=sys.stderr)
                        self.writer.save_checkpoint(self._unwrap(policy), step=step, optimizer=optimizer)
                    dist.barrier()  # 所有 rank 到齐（rank0 存完了）
                    import time
                    time.sleep(rest_sec)
                    if dist.get_rank() == 0:
                        print(f"[lapo.train] ❄️ 休息结束，继续训练", file=sys.stderr)
                    dist.barrier()  # 休息完所有 rank 同步继续
        else:
            if max_temp >= threshold:
                import time
                print(f"[lapo.train] 🌡️ GPU {max_temp}°C ≥ {threshold}°C，"
                      f"先存 checkpoint 再散热休息 {rest_sec}s", file=sys.stderr)
                self.writer.save_checkpoint(self._unwrap(policy), step=step, optimizer=optimizer)
                time.sleep(rest_sec)
                print(f"[lapo.train] ❄️ 休息结束，继续训练", file=sys.stderr)

    @staticmethod
    def _max_gpu_temp() -> int:
        """读所有 GPU 最高温度。无 GPU/pynvml 返回 0。"""
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
                pynvml.nvmlInit()
                try:
                    count = pynvml.nvmlDeviceGetCount()
                    temps = []
                    for i in range(count):
                        h = pynvml.nvmlDeviceGetHandleByIndex(i)
                        temps.append(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                    return max(temps) if temps else 0
                finally:
                    pynvml.nvmlShutdown()
        except Exception:
            return 0

    def _to_device(self, batch: dict) -> dict:
        import torch
        dev = self.device
        return {k: (v.to(dev) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}

    def _extract_graph(self, policy: Any) -> dict:
        """优先用 strategy 手写图，否则 forward hook 自动提取。"""
        manual = self.strategy.describe_graph(policy)
        if manual is not None:
            return manual
        from lapo.train.artifacts.graph_extractor import extract_graph
        try:
            sample = next(iter(self.dataloader))
            sample = {k: v[:1] for k, v in sample.items()}
            return extract_graph(policy, sample_input=sample)
        except Exception:
            # 无 dataloader 或提取失败 → 空图（不阻塞训练）
            return {"nodes": [], "edges": [], "frozen": []}


class _NoOpWriter:
    """非 rank0 进程的空 writer：所有写操作静默 no-op。

    FSDP/DDP 下只有 rank0 写产物（run.json/metrics/checkpoint），非 rank0
    用此对象避免多进程写冲突。方法签名对齐 ArtifactWriter（duck typing）。
    """

    def __getattr__(self, name):
        # 任何未显式定义的方法 → 返回一个 no-op callable
        return lambda *a, **kw: None
