"""SBVLAHead —— 组合 encoder + g + SB + 冻结 Florence2。

这是核心 nn.Module，被 SBPolicy 包装暴露给 lerobot 兼容层。

数据流（推理，对齐文档 §2.3）：
  ① h = Florence2(观测) 语言特征池化         ← 冻结，出任务意图
  ② z_t = encoder(DaViT 特征)                ← 可训练，物理 latent
  ③ z_goal = g(z_t, h)                       ← act-free 世界模型（预测一秒后 latent）
  ④ chunk = SB.sample([z_t, z_goal], noise)  ← Euler 积分生成

两阶段训练（forward 按 cfg.phase 分支）：
  phase=1: 训 encoder+g+SIGReg；Loss=‖Δg−Δtar‖+γ·L_SIGReg
  phase=2: 冻 encoder/g；训 SB；Loss=L_IMLE+λ_acc·L_force

f（JumpDynamics）已删除：约束来自 g 的预测一致性，无 L_reach。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from lapo.train.policies.sb.config import SBVLAConfig
from lapo.train.policies.sb.sigreg import SIGReg
from lapo.train.policies.sb.components import Encoder, GoalProposer
from lapo.train.policies.sb.bridge import SchrodingerBridge


class SBVLAHead(nn.Module):
    """SB-VLA 核心 head：Florence2(冻结) + encoder + g + SB。

    Args:
        cfg: SBVLAConfig
        vlm: 已加载的 Florence2ForConditionalGeneration（冻结，从 lerobot/xvla-base 注入）
        action_space: lerobot action space（复用 EE6DActionSpace 的 loss/preprocess/postprocess）
    """

    def __init__(self, cfg: SBVLAConfig, vlm: nn.Module, action_space):
        super().__init__()
        self.cfg = cfg
        self.chunk_size = cfg.chunk_size
        self.dim_action = action_space.dim_action  # model-facing（32）
        self.action_space = action_space

        # ---- 冻结 Florence2（只复用 forward_vlm 出特征）----
        self.vlm = vlm
        self.vlm.eval()
        for p in self.vlm.parameters():
            p.requires_grad_(False)

        # ---- 任务意图 h 的投影（Florence 语言特征 → dim_latent）----
        # Florence 只出语言语义 h，不再出视觉特征（z_t 来自独立 ViT）。
        self.h_proj = nn.Linear(cfg.florence_hidden, cfg.dim_latent)

        # ---- 可训练组件 ----
        # encoder：DaViT(冻结) 特征 → 物理 latent（投影头可训，SIGReg 守护）
        # 输入 = 2 视角 × 2048 维 = 4096（拼接）
        self.encoder = Encoder(
            dim_latent=cfg.dim_latent, dim_davit=4096,
            depth=cfg.enc_depth, heads=cfg.enc_heads, mlp_ratio=cfg.enc_mlp_ratio,
        )
        self.g = GoalProposer(
            dim_latent=cfg.dim_latent, dim_h=cfg.dim_latent,
            depth=cfg.gf_depth, heads=cfg.gf_heads, mlp_ratio=cfg.gf_mlp_ratio,
        )
        self.sb = SchrodingerBridge(self.dim_action, cfg.dim_latent, cfg)

        # ---- SIGReg 守护 latent ----
        self.sigreg = SIGReg(cfg.dim_latent, num_proj=cfg.num_sigreg_proj)

        # 当前训练步（用于 schedule，由 strategy 注入）
        self._train_step = 0
        self._total_steps = 1  # 防除 0，由 strategy 设置

    # ---- Florence2 特征提取（复用 XVLAModel.forward_vlm）----
    def forward_vlm(self, input_ids, image_input, image_mask):
        """复用 XVLA 的 forward_vlm，出 {vlm_features, aux_visual_inputs}."""
        return self.vlm and self.vlm_forward(input_ids, image_input, image_mask)

    def _florence_semantic(self, input_ids, image_input, image_mask):
        """Florence2 只出语义 h（语言+视觉对齐特征）。

        Florence 看图理解场景语义（"绿方块在桌上"），和语言指令（"抓绿方块"）融合。
        但它的视觉特征不为物理预测优化 → 不用它做 z_t，只用它出 h。
        """
        batch_size, num_views = image_input.shape[:2]
        flat_mask = image_mask.view(-1).to(dtype=torch.bool)
        flat_images = image_input.flatten(0, 1)
        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")
        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        tokens_per_view, hidden_dim = valid_feats.shape[1:]
        image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0], inputs_embeds,
        )
        enc_out = self.vlm.language_model.model.encoder(
            attention_mask=attention_mask, inputs_embeds=merged_embeds,
        )[0]
        h = enc_out.mean(dim=1)                   # [B, florence_hidden] 池化
        return self.h_proj(h)                      # [B, dim_latent]

    def _davit_features(self, image_input, image_mask):
        """用 Florence2 DaViT 视觉塔（冻结）提取物理视觉特征 [B, 2048]。

        取两个有效视角，各自过 DaViT → token 池化 → 拼接。
        DaViT 输出 [B, 49, 2048]（49 token × 2048 维），mean(dim=1) 池化成 [B, 2048]。
        """
        b = image_input.shape[0]
        feats = []
        for v in range(image_input.shape[1]):
            if image_mask[:, v].all():
                img = image_input[:, v]                          # [B, 3, H, W]
                raw = self.vlm.vision_tower.forward_features_unpool(img)  # [B, 49, 2048]
                pooled = raw.mean(dim=1)                        # [B, 2048] token 维池化
                feats.append(pooled)
            if len(feats) == 2:
                break
        if len(feats) < 2:
            dim = feats[0].shape[-1] if feats else 2048
            while len(feats) < 2:
                feats.append(torch.zeros(b, dim, device=image_input.device, dtype=image_input.dtype))
        return torch.cat(feats, dim=-1)  # [B, 2*2048]

    # ============================================================
    # 训练 forward
    # ============================================================
    def forward(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,  # noqa: ARG002（SB 不用 domain，保留接口兼容）
        proprio,
        action,
        image_input_tar=None,
        image_mask_tar=None,
    ) -> dict[str, torch.Tensor]:
        """训练 loss（按 cfg.phase 分支）。

        image_input/image_mask:        当前帧 frame_t（z_t 用），[B, n_view, C,H,W]
        image_input_tar/image_mask_tar:num_h 个未来 horizon，已 fold 进 batch 维：
                                       [B*num_h, n_view, C,H,W]（policy 层组装）
                                       None/缺省时退化为当前帧（z_tar=z_t 复制 num_h 次）。
        proprio: [B, dim_proprio] 本体状态（action_space.preprocess 用它清零 gripper）
        action: [B, chunk_size, dim_action] 专家动作块（已 pad）
        return: loss dict（strategy 会 sum）
        """
        target_dtype = next(self.encoder.parameters()).dtype
        image_input = image_input.to(dtype=target_dtype)
        action = action.to(dtype=target_dtype)
        if proprio is not None and torch.is_tensor(proprio):
            proprio = proprio.to(dtype=target_dtype)

        # ① Florence 出语义 h（语言+场景理解，act-free predictor 的方向信号）
        h = self._florence_semantic(input_ids, image_input, image_mask)  # [B, dim_lat]

        # ② z_t = encoder(DaViT 特征) —— 纯物理 latent（DaViT 冻结，投影头可训）
        davit_t = self._davit_features(image_input, image_mask)   # [B, 2048]
        z_t = self.encoder(davit_t)                               # [B, dim_lat]

        # ②' z_tar = encoder(DaViT(frame_{t+hi})) —— num_h 个未来 horizon 的真实 latent
        #    policy 把 num_h 个未来帧 fold 进 batch 维 → 一次 DaViT+encoder 出全部
        num_h = len(self.cfg.horizons)
        b = z_t.shape[0]
        if image_input_tar is not None:
            image_input_tar = image_input_tar.to(dtype=target_dtype)
            mask_tar = image_mask_tar if image_mask_tar is not None else image_mask
            # image_input_tar 期望 [B*num_h, n_view, ...]（policy fold）
            expected_b = b * num_h
            if image_input_tar.shape[0] != expected_b:
                # 兼容：若 policy 没传多帧（单帧 [B, ...]），复制成 num_h 份
                if image_input_tar.shape[0] == b:
                    image_input_tar = image_input_tar.unsqueeze(1).expand(-1, num_h, *([-1]*(image_input_tar.dim()-1))).reshape(expected_b, *image_input_tar.shape[1:])
                    mask_tar = mask_tar.unsqueeze(1).expand(-1, num_h, -1).reshape(expected_b, *mask_tar.shape[1:])
                # 否则交给下游报错
            davit_tar = self._davit_features(image_input_tar, mask_tar)
            z_tar_flat = self.encoder(davit_tar)                 # [B*num_h, dim_lat]
            z_tar = z_tar_flat.view(b, num_h, -1)                # [B, num_h, dim_lat]
        else:
            z_tar = z_t.unsqueeze(1).expand(-1, num_h, -1)       # [B, num_h, dim_lat]

        if self.cfg.phase == 1:
            return self._forward_phase1(z_t, z_tar, h, action, proprio)
        else:
            return self._forward_phase2(z_t, z_tar, h, action, proprio)

    # ---- 阶段1：训 encoder + g + SIGReg + VICReg ----
    def _forward_phase1(self, z_t, z_tar, h, action, proprio):  # noqa: ARG002
        """Phase1 loss。

        use_delta_loss=True（默认, P0/P1）:
          delta-space 监督。target = stopgrad(z_tar - z_t), loss = masked_dir_cos
          + 逐样本 clamp 的 log-ratio + smooth_l1_delta。
          背景: full-space loss 被 z_t 主体淹没（‖z_t‖≫‖Δz‖）→ g 输出反向 Δg
          也能让 L_dir=0.10（假象）。delta-space 避开主体淹没, 直接监督增量。
          stopgrad 切断 g loss → encoder 的 target-moving 通道。
          encoder 的时序结构由 VICReg(z) + delta-VICReg（P1）维护。

        use_delta_loss=False:
          回退到旧 full-space 三分项 loss（ablation 对照）。
        """
        z_tar_sg = z_tar.detach()                    # [B, num_h, dim]
        weights = self.cfg.horizon_weights           # (num_h,)
        eps = 1e-6

        # encoder 正则：SIGReg（LeWM 原版唯一防坍缩项）+ VICReg(z_t)
        # SIGReg 把 z 的完整分布拉向 N(0,I)，既防坍缩也防膨胀。
        # 之前 SIGReg 被创建(self.sigreg)但从未调用 → mag_z 无约束膨胀到 4.3。
        L_sigreg = self.sigreg(z_t)
        L_enc_var, L_enc_cov = self._vicreg(z_t)

        if self.cfg.use_delta_loss:
            L_g, L_dir_sum, L_mag_sum, L_dmse_sum, monitor = self._loss_delta(
                z_t, z_tar, z_tar_sg, h, weights, eps)
        else:
            L_g, L_dir_sum, L_fit_sum, L_div_sum, monitor = self._loss_full(
                z_t, z_tar_sg, h, weights)

        # VICReg（方差 + 协方差），施加在 g 输出上，破跨样本坍缩
        z_goal_seq = monitor["z_goal_seq"]
        L_vic_var, L_vic_cov = self._vicreg(z_goal_seq)

        # 时序增量正则（旧 hinge）：delta-space 模式下作为监控项保留（不进 loss）。
        # P1 用 delta-VICReg 替代它做时序约束。
        with torch.no_grad():
            L_temporal_mon = z_t.new_zeros(())
            for i, hw in enumerate(weights):
                mag_tar_i = (z_tar_sg[:, i] - z_t).pow(2).mean().sqrt()
                L_temporal_mon = L_temporal_mon + hw * torch.relu(self.cfg.tau_temporal - mag_tar_i)

        # P1: delta-VICReg（对 Δz 做每维 std + 去相关, 带梯度回 encoder）
        L_delta_var = z_t.new_zeros(())
        L_delta_cov = z_t.new_zeros(())
        if self.cfg.use_delta_vicreg:
            if not getattr(self, "_dbg_p1_logged", False):
                import logging
                logging.getLogger(__name__).warning(
                    f"[P1-DBG] use_delta_loss={self.cfg.use_delta_loss} "
                    f"use_delta_vicreg={self.cfg.use_delta_vicreg} "
                    f"p0_freeze={self.cfg.p0_freeze_encoder} "
                    f"enc_lr_mult={self.cfg.encoder_lr_mult} "
                    f"gamma_delta={self.cfg.gamma_delta} "
                    f"w_dir_delta={self.cfg.w_dir_delta} w_loss_full={self.cfg.w_loss_full}")
                self._dbg_p1_logged = True
            for i, hw in enumerate(weights):
                delta_z_i = z_tar_sg[:, i] - z_t        # 带梯度回 encoder
                lv, lc = self._vicreg(delta_z_i, gamma=self.cfg.gamma_delta)
                L_delta_var = L_delta_var + hw * lv
                L_delta_cov = L_delta_cov + hw * lc
        elif not getattr(self, "_dbg_p1_logged", False):
            import logging
            logging.getLogger(__name__).warning(
                f"[P1-DBG] use_delta_vicreg=False → delta-VICReg 分支未执行 "
                f"use_delta_loss={self.cfg.use_delta_loss} p0_freeze={self.cfg.p0_freeze_encoder}")
            self._dbg_p1_logged = True

        # ---- 诊断量（no_grad）----
        with torch.no_grad():
            mag_z = z_t.pow(2).mean().sqrt()

            goal_idx = self.cfg.horizons.index(self.cfg.z_goal_horizon)
            # gain_h: h=30 的 h 响应（不变）
            z_goal_no_h = self.g(z_t, torch.zeros_like(h))
            goal_main = z_goal_seq[:, goal_idx]
            gain_h = (z_goal_no_h[:, goal_idx] - goal_main).pow(2).mean() / \
                     (goal_main.pow(2).mean() + 1e-8)

            # horizon-wise 监控（两种 loss 模式通用）
            cos_h_diag = {}
            mag_g_main = z_t.new_zeros(())
            mag_tar_main = z_t.new_zeros(())
            delta_std_diag = {}
            for i, hv in enumerate(self.cfg.horizons):
                cos_h_diag[f"cos_h{hv}"] = F.cosine_similarity(
                    z_goal_seq[:, i], z_tar_sg[:, i], dim=-1).mean()
                # Δz 每维跨样本 std（给 P1 校准 gamma_delta 用）
                dz_i = z_tar_sg[:, i] - z_t              # [B, dim]
                dstd_i = torch.sqrt(dz_i.var(dim=0) + 1e-4)   # [dim]
                delta_std_diag[f"delta_std_h{hv}"] = dstd_i.mean()
            # 主 horizon 汇总
            delta_std_diag["delta_std_main"] = delta_std_diag[f"delta_std_h{self.cfg.z_goal_horizon}"]
            if self.cfg.use_delta_loss:
                mag_g_main = monitor["mag_g_h"][goal_idx]
                mag_tar_main = monitor["mag_tar_h"][goal_idx]
                ratio_g_tar = monitor["ratio_g_tar_h"][goal_idx]
                # delta 诊断指标汇总（horizon-wise 已在 _loss_delta 算好）
                delta_mon = monitor["delta_mon"]
            else:
                mag_g_main = (z_goal_seq[:, goal_idx] - z_t).pow(2).mean().sqrt()
                mag_tar_main = (z_tar_sg[:, goal_idx] - z_t).pow(2).mean().sqrt()
                ratio_g_tar = mag_g_main / (mag_tar_main + 1e-8)
                delta_mon = {}

        out = {
            "loss_g": L_g,
            "loss_sigreg": self.cfg.gamma_sigreg * L_sigreg,
            "loss_enc_var": self.cfg.lambda_vic_var * L_enc_var,
            "loss_enc_cov": self.cfg.lambda_vic_cov * L_enc_cov,
            "loss_vic_var": self.cfg.lambda_vic_var * L_vic_var,
            "loss_vic_cov": self.cfg.lambda_vic_cov * L_vic_cov,
            "loss_temporal": self.cfg.lambda_temporal * L_temporal_mon.detach(),  # 仅监控
            "loss_delta_var": self.cfg.lambda_delta_var * L_delta_var,   # P1
            "loss_delta_cov": self.cfg.lambda_delta_cov * L_delta_cov,   # P1
            "gain_h": gain_h,
            "mag_z": mag_z,
            "mag_g_main": mag_g_main,
            "mag_tar_main": mag_tar_main,
            "ratio_g_tar": ratio_g_tar,
        }
        out.update(cos_h_diag)
        out.update(delta_mon)
        out.update(delta_std_diag)
        if self.cfg.use_delta_loss:
            out["loss_dir_delta"] = L_dir_sum.detach()
            out["loss_mag_ratio"] = L_mag_sum.detach()
            out["loss_delta_mse"] = L_dmse_sum.detach()
        else:
            out["loss_dir"] = L_dir_sum.detach()
            out["loss_fit"] = L_fit_sum.detach()
            out["loss_div"] = L_div_sum.detach()
        return out

    def _loss_delta(self, z_t, z_tar, z_tar_sg, h, weights, eps):
        """delta-space loss（P0/P1 主线）。

        target delta 不回传 encoder: delta_tar = stopgrad(z_tar - z_t)。
        g 的 delta 直接取 raw output（return_delta=True）, 不从 z_goal 反推。
        三项: ① masked 增量 cos（小 target delta 不计方向）② 逐样本 clamp 的 log-ratio
              ③ smooth_l1 delta（低权辅助）。

        delta_loss_freeze_zt=True 时 g 用 z_t.detach()（master 诊断 A: 切断
        delta loss 经 g.net→z_t→encoder 的塌缩通路, 让 delta loss 只训 g）。
        """
        z_g_in = z_t.detach() if self.cfg.delta_loss_freeze_zt else z_t
        z_goal_seq, delta_g_seq = self.g(z_g_in, h, return_delta=True)  # [B,num_h,dim] both
        delta_tar = (z_tar - z_t.unsqueeze(1)).detach()               # stopgrad

        L_g = z_t.new_zeros(())
        L_dir_sum = z_t.new_zeros(())
        L_mag_sum = z_t.new_zeros(())
        L_dmse_sum = z_t.new_zeros(())

        # horizon-wise 诊断（MSE 定义, 单位一致）
        delta_mon = {}
        mag_g_h, mag_tar_h, ratio_g_tar_h = [], [], []

        for i, hw in enumerate(weights):
            dg = delta_g_seq[:, i]          # [B, dim] 带梯度
            dt = delta_tar[:, i]            # [B, dim] detached
            norm_g = dg.norm(dim=-1).clamp_min(eps)
            norm_t = dt.norm(dim=-1).clamp_min(eps)

            # ① 方向: 只在真实 delta 够大时算
            mask = (norm_t > self.cfg.delta_mask_tau).float()
            cos_i = F.cosine_similarity(dg, dt, dim=-1, eps=eps)
            loss_dir_each = (1.0 - cos_i) * mask
            loss_dir = loss_dir_each.sum() / (mask.sum() + eps)

            # ② 幅值: 逐样本 log-ratio + soft clamp（log1p 平滑过渡, 梯度不归零）
            # 旧版 hard clamp(max=C) 在 (log_g-log_t)²>C 时梯度断崖归零:
            #   ratio=0.08 → log²=6.3 > C=4 → 梯度=0, mag 项从 step 1 起就死。
            # soft: C·log1p(diff²/C)。diff²<C 时 ≈ diff²（精确梯度）; diff²>C 时
            #   梯度 = 2·diff/(diff²+C)（衰减但不归零, 始终指向正确方向）。防爆且不死。
            log_g = torch.log(norm_g.clamp_min(self.cfg.delta_mag_floor))
            log_t = torch.log(norm_t.clamp_min(self.cfg.delta_mag_floor))
            diff_sq = (log_g - log_t).pow(2)
            clip = self.cfg.mag_loss_clip
            loss_mag = (clip * torch.log1p(diff_sq / clip)).mean()

            # ③ delta smooth_l1（辅助）
            loss_dmse = F.smooth_l1_loss(dg, dt)

            L_g = L_g + hw * (self.cfg.w_dir_delta * loss_dir
                              + self.cfg.w_mag_ratio * loss_mag
                              + self.cfg.w_delta_mse * loss_dmse)
            L_dir_sum = L_dir_sum + hw * loss_dir
            L_mag_sum = L_mag_sum + hw * loss_mag
            L_dmse_sum = L_dmse_sum + hw * loss_dmse

            # horizon-wise 诊断（no_grad）
            with torch.no_grad():
                mse_pred = ((dg - dt) ** 2).mean(dim=-1)         # [B]
                mse_zero = (dt ** 2).mean(dim=-1)                # [B]
                improve = (mse_zero - mse_pred).mean()
                rel_improve = ((mse_zero - mse_pred) / (mse_zero + eps)).mean()
                cos_delta_h = cos_i.mean()
                mag_ratio_h = norm_g.mean() / (norm_t.mean() + eps)
                mask_ratio = mask.mean()
                hv = self.cfg.horizons[i]
                delta_mon[f"cos_delta_h{hv}"] = cos_delta_h
                delta_mon[f"improve_h{hv}"] = improve
                delta_mon[f"rel_improve_h{hv}"] = rel_improve
                delta_mon[f"mag_ratio_h{hv}"] = mag_ratio_h
                delta_mon[f"mask_ratio_h{hv}"] = mask_ratio
                delta_mon[f"mse_pred_h{hv}"] = mse_pred.mean()
                delta_mon[f"mse_zero_h{hv}"] = mse_zero.mean()
                mag_g_h.append(norm_g.mean())
                mag_tar_h.append(norm_t.mean())
                ratio_g_tar_h.append(mag_ratio_h)

        # 主 horizon（h=30）的汇总判据（最关键）
        goal_idx = self.cfg.horizons.index(self.cfg.z_goal_horizon)
        delta_mon["cos_delta_main"] = delta_mon[f"cos_delta_h{self.cfg.z_goal_horizon}"]
        delta_mon["improve_main"] = delta_mon[f"improve_h{self.cfg.z_goal_horizon}"]
        delta_mon["rel_improve_main"] = delta_mon[f"rel_improve_h{self.cfg.z_goal_horizon}"]

        # full-space MSE 残留（P0 默认 0, ablation 可设 0.01）
        if self.cfg.w_loss_full > 0:
            zg = z_t.unsqueeze(1) + delta_g_seq
            L_full = F.mse_loss(zg, z_tar)
            L_g = L_g + self.cfg.w_loss_full * L_full

        monitor = {
            "z_goal_seq": z_goal_seq,
            "delta_mon": delta_mon,
            "mag_g_h": mag_g_h,
            "mag_tar_h": mag_tar_h,
            "ratio_g_tar_h": ratio_g_tar_h,
        }
        return L_g, L_dir_sum, L_mag_sum, L_dmse_sum, monitor

    def _loss_full(self, z_t, z_tar_sg, h, weights):
        """旧 full-space 三分项 loss（ablation 回退分支）。"""
        z_goal_seq = self.g(z_t, h)
        z_t_sg = z_t.detach()
        L_g = z_t.new_zeros(())
        L_dir_sum = z_t.new_zeros(())
        L_fit_sum = z_t.new_zeros(())
        L_div_sum = z_t.new_zeros(())
        for i, hw in enumerate(weights):
            zg = z_t_sg + (z_goal_seq[:, i] - z_t_sg)
            zt = z_tar_sg[:, i]
            cos_i = F.cosine_similarity(zg, zt, dim=-1).mean()
            l_dir = (1.0 - cos_i)
            l_fit = (zg - zt).pow(2).mean()
            dg = zg - z_t_sg
            dt = zt - z_t_sg
            l_div = (1.0 - F.cosine_similarity(dg, dt, dim=-1).mean())
            L_g = L_g + hw * (self.cfg.w_loss_dir * l_dir
                              + self.cfg.w_loss_fit * l_fit
                              + self.cfg.w_loss_div * l_div)
            L_dir_sum = L_dir_sum + hw * l_dir
            L_fit_sum = L_fit_sum + hw * l_fit
            L_div_sum = L_div_sum + hw * l_div
        monitor = {"z_goal_seq": z_goal_seq}
        return L_g, L_dir_sum, L_fit_sum, L_div_sum, monitor

    def _vicreg(self, z: torch.Tensor, gamma: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """VICReg 方差 + 协方差项（对齐 VICReg 论文 §3）。

        z: [..., dim]（任意前导维度）→ 展平 [N, dim]。
        gamma: std 阈值, 默认用 cfg.gamma_vic; delta-VICReg 传 cfg.gamma_delta。
        return: (var_loss, cov_loss)

        方差项（主力, 恒用 raw z）：每维度跨样本 std < γ → hinge 惩罚。破跨样本坍缩。
          V = mean_j max(0, γ − √(Var(z_j)+ε))
          hinge 形式：std ≥ γ 时梯度为 0（满足就放手，不像 SIGReg 持续施压）。
          ★ var 守 absolute scale, 不归一化 —— 职责与 cov 分离。
        协方差项（去相关, 按 cfg.vic_cov_mode 归一化）：
          raw:       C = ‖off-diag cov(z)‖²_F / d
          sample_l2: C = ‖off-diag cov(z/‖z‖_sample)‖²_F / d（去整体尺度）
          corr:      C = ‖off-diag corr(z)‖²_F / d（per-dim standardize, 真正惩罚相关性；
                     既不奖励缩尺度, 也不受单维尺度差异影响）
        ★ 关键: raw cov 是 scale-variant（z→αz 则 C→α²·C → L→α⁴·L）→ encoder 走捷径
          缩 z 降 cov → mag_z 塌。corr/l2norm 切断此捷径, 保留去相关几何约束。
        """
        d = z.shape[-1]
        flat = z.reshape(-1, d)                      # [N, dim]
        n = flat.shape[0]
        g = self.cfg.gamma_vic if gamma is None else gamma
        # 方差项（raw, 守 absolute std）
        std = torch.sqrt(flat.var(dim=0) + 1e-4)     # [dim] 每维 std
        vic_var = torch.relu(g - std).mean()
        # 协方差项 —— 按 vic_cov_mode 归一化后再算 off-diag Frobenius
        mode = getattr(self.cfg, "vic_cov_mode", "raw")
        if mode == "corr":
            # per-dim standardize: 减均值 + 除 per-dim std → 算出的是相关矩阵
            zc = flat - flat.mean(dim=0, keepdim=True)
            zc = zc / (zc.std(dim=0, keepdim=True, unbiased=False) + 1e-4)
            z_for_cov = zc
        elif mode == "sample_l2":
            # 按样本 L2 归一（保留各维尺度差异, 只去整体向量长度）
            z_for_cov = flat / (flat.norm(dim=-1, keepdim=True) + 1e-4)
        else:  # "raw"
            z_for_cov = flat
        # cov 矩阵 off-diag（统一走 center: corr 已 center 过但重复无副作用; raw/l2 需 center）
        zc = z_for_cov - z_for_cov.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / n                        # [dim, dim]
        cov_off = cov.fill_diagonal_(0.0)            # 去对角（in-place, z 已 reshape 不影响原张量）
        vic_cov = cov_off.pow(2).sum() / d
        return vic_var, vic_cov

    # ---- 阶段2：训 SB ----
    def _forward_phase2(self, z_t, z_tar, h, action, proprio):  # noqa: ARG002
        """Loss = L_IMLE + λ_acc·L_force.

        encoder/g 已冻结（strategy 控制 requires_grad），这里只前向。
        z_goal = g(z_t, h)[:, h=30 index]（act-free 目标锚点，=1秒，SB 到达目标）。
        多 horizon 输出里取主角 h=30；其余 horizon 只在 phase1 当陪练，phase2 不用。
        SB 拿 [z_t, z_goal] 生成 chunk，被 z_goal 钉住终点。
        """
        with torch.no_grad():
            z_goal_seq = self.g(z_t, h)                       # [B, num_h, dim]
            goal_idx = self.cfg.horizons.index(self.cfg.z_goal_horizon)
            z_goal = z_goal_seq[:, goal_idx]                  # [B, dim] = h=30

        # action 预处理（清零 gripper 维）
        action_for_sb = action
        if hasattr(self.action_space, 'preprocess') and proprio is not None:
            _, action_for_sb = self.action_space.preprocess(proprio, action)

        # SB 桥 loss（IMLE 多模态 + 薛定谔力/加速度场）
        sb_loss = self.sb.loss(z_t, z_goal, action_for_sb, k_p=self.cfg.k_p, k_d=self.cfg.k_d)

        return {
            "loss_imle": sb_loss["imle"],
            "loss_force": self.cfg.lambda_acc * sb_loss["force"],
        }

    # ============================================================
    # 推理生成（对齐 XVLAModel.generate_actions 接口）
    # ============================================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        image_input,
        image_mask,
        domain_id,  # noqa: ARG002
        proprio,    # noqa: ARG002
        steps: int,
    ) -> torch.Tensor:
        """推理: 生成动作块。

        推理链路（g 定目标，SB 生成）：
          frame → encoder → z_t
          z_t + h → g → z_goal（act-free 世界模型，预测一秒后 latent）
          [z_t, z_goal] → SB.sample → chunk（Euler 积分，内部平滑）
        return action_space.postprocess(chunk)
        """
        self.eval()
        target_dtype = next(self.encoder.parameters()).dtype
        image_input = image_input.to(dtype=target_dtype)

        # encoder → z_t → g → z_goal(取 h=30 主角) → SB
        h = self._florence_semantic(input_ids, image_input, image_mask)
        davit_t = self._davit_features(image_input, image_mask)
        z_t = self.encoder(davit_t)
        z_goal_seq = self.g(z_t, h)                          # [1, num_h, dim]
        goal_idx = self.cfg.horizons.index(self.cfg.z_goal_horizon)
        z_goal = z_goal_seq[:, goal_idx]                     # [1, dim] = h=30（SB 到达目标）

        # SB 拿 z_goal 直接生成 chunk。z_goal 钉住终点，PD 拉回样条，
        # 不同 noise 收敛到同一 chunk → 推理确定，调一次即可。
        action = self.sb.sample(z_t, z_goal, self.chunk_size, steps)

        return self.action_space.postprocess(action)
