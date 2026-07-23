#!/usr/bin/env python3
"""LAPo (S4 step_2499) vs xvla_afe5 (step_34999) 在同一 val split 上对比。

两个架构用同一套指标 (lapo_action_loss 口径):
  loss_xyz / loss_rot / loss_grip / grip_acc / loss_endpoint_disp / sample_var

保证 apples-to-apples 对比。
"""
import torch, sys, os, json, yaml
import numpy as np
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def lapo_action_loss_uniform(pred, target, dim_action=10):
    """统一口径 action loss (不依赖 LapoConfig)。
    pred/target: [B, H, 10] ee6d 布局: xyz[0:3] + rot6d[3:9] + grip[9]
    """
    import torch.nn.functional as F
    xyz_loss = F.smooth_l1_loss(pred[..., 0:3], target[..., 0:3])
    rot_loss = F.smooth_l1_loss(pred[..., 3:9], target[..., 3:9])
    # grip: 假设两个 policy 输出的是 logit 或概率, 都当 logit 处理
    grip_logit = pred[..., 9]
    grip_target = target[..., 9].clamp(0, 1)
    grip_loss = F.binary_cross_entropy_with_logits(grip_logit, grip_target)
    grip_acc = ((torch.sigmoid(grip_logit) > 0.5).float() == grip_target).float().mean()
    pred_disp = pred[..., 0:3].sum(dim=1)
    tgt_disp = target[..., 0:3].sum(dim=1)
    endpoint_disp_loss = F.smooth_l1_loss(pred_disp, tgt_disp)
    return {
        "loss_xyz": xyz_loss.item(),
        "loss_rot": rot_loss.item(),
        "loss_grip": grip_loss.item(),
        "grip_acc": grip_acc.item(),
        "loss_endpoint_disp": endpoint_disp_loss.item(),
    }


def build_dataloader(cfg):
    """构建 val dataloader (和 lapo_eval 一致的 split)。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads(Path(cfg['dataset']['root'], 'meta/info.json').read_text())
    H = 30; fps = info['fps']
    dt = {}
    for k in info['features']:
        if k.startswith('observation.'):
            dt[k] = [i/fps for i in [0, H]]
    dt['action'] = [i/fps for i in range(H)]
    total_ep = info['total_episodes']
    val_eps = list(range(int(total_ep*0.9), total_ep))
    ds = LeRobotDataset(cfg['dataset']['repo_id'], root=cfg['dataset']['root'],
                        delta_timestamps=dt, episodes=val_eps, tolerance_s=0.05)
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=4), info


# ============================================================
# LAPo 评估
# ============================================================
def eval_lapo(checkpoint, nbatches=8):
    from safetensors.torch import load_file
    from lapo.train.compat import build_policy_for, _resolve_base_model, load_state_dict_shape_filtered
    from lapo.train.services.training import load_registry_with_builtins
    from lapo.train.policies.lapo.config import LapoConfig
    from lapo.train.policies.lapo.model import LapoBridge
    from lapo.train.policies.lapo.policy import LapoPolicy
    from lapo.train.policies.xvla_tokenizer import tokenize_language_xvla
    from lerobot.policies.xvla.modeling_xvla import resize_with_pad, OBS_LANGUAGE_TOKENS, OBS_STATE, ACTION, pad_vector, pad_tensor_along_dim

    cfg_path = str(Path(__file__).resolve().parent.parent / 'configs/openarm_pick_place_green277_lapo_stage4.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    overrides = cfg['policy_overrides']
    rename_map = cfg['dataset']['rename_map']

    registry = load_registry_with_builtins()
    info = json.loads(Path(cfg['dataset']['root'], 'meta/info.json').read_text())
    class Meta:
        fps = info['fps']; features = info['features']
    ds_meta = Meta()

    xvla_overrides = {
        "base_model": overrides['base_model'], "action_mode": "ee6d",
        "max_action_dim": 32, "empty_cameras": 1, "dtype": "float32",
    }
    xvla = build_policy_for("xvla", registry, ds_meta, overrides=xvla_overrides, rename_map=rename_map)
    vlm = xvla.model.vlm
    resolved = _resolve_base_model(overrides['base_model'])
    sd_base = load_file(str(Path(resolved) / 'model.safetensors'))
    if "model.vlm.image_projection" in sd_base:
        vlm.image_projection.data.copy_(sd_base["model.vlm.image_projection"].to(vlm.image_projection.dtype))
    for suf in ["weight","bias"]:
        k = f"model.vlm.image_proj_norm.{suf}"
        if k in sd_base:
            getattr(vlm.image_proj_norm, suf).data.copy_(sd_base[k])

    lapo_cfg = LapoConfig(
        chunk_size=xvla.config.chunk_size, dim_action=xvla.model.action_space.dim_action,
        max_action_dim=32, dim_proprio=xvla.config.max_state_dim,
        florence_hidden=getattr(vlm.config, "projection_dim", 1024),
        action_mode="ee6d", dtype="float32",
    )
    for k, v in overrides.items():
        if hasattr(lapo_cfg, k) and not k.startswith("_"):
            cur = getattr(lapo_cfg, k)
            try:
                if isinstance(cur, bool): setattr(lapo_cfg, k, bool(v))
                elif isinstance(cur, int): setattr(lapo_cfg, k, int(v))
                elif isinstance(cur, float): setattr(lapo_cfg, k, float(v))
                else: setattr(lapo_cfg, k, v)
            except: setattr(lapo_cfg, k, v)

    head = LapoBridge(lapo_cfg, vlm, xvla.model.action_space)
    sd = load_file(str(Path(checkpoint) / 'model.safetensors'))
    load_state_dict_shape_filtered(head, sd)
    print(f"[lapo] checkpoint 加载: {checkpoint}")

    policy = LapoPolicy(lapo_cfg, head, xvla_config=xvla.config).cuda().eval()
    dl, _ = build_dataloader(cfg)

    all_metrics = []
    sample_vars = []
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= nbatches: break
            for orig, new in rename_map.items():
                if orig in batch: batch[new] = batch[orig]
            from lapo.train.policies.xvla_tokenizer import tokenize_language_xvla
            batch = tokenize_language_xvla(batch, rename_map=rename_map)
            input_ids = batch[OBS_LANGUAGE_TOKENS].cuda()
            imgs = []
            for key in ['observation.images.image2', 'observation.images.image3']:
                if key not in batch: continue
                img = batch[key].cuda()
                imgs.append(resize_with_pad(img[:,0], 224, 224))
            img_t = torch.stack(imgs, dim=1).float()
            mask = torch.ones(img_t.shape[0], img_t.shape[1], dtype=torch.bool).cuda()
            proprio = batch[OBS_STATE]
            if proprio.ndim > 2: proprio = proprio[:, -1, :]
            proprio = pad_vector(proprio.cuda(), lapo_cfg.dim_proprio)
            action = batch[ACTION].cuda()
            if action.ndim == 2: action = action.unsqueeze(1)
            action = pad_tensor_along_dim(action, lapo_cfg.chunk_size, dim=1)
            action = pad_vector(action, lapo_cfg.dim_action)
            dtype = next(head.encoder.parameters()).dtype
            img_t = img_t.to(dtype)

            # 两次采样看 sample variance
            preds = []
            for _ in range(2):
                davit_t = head._davit_features(img_t, mask)
                z_t = head.encoder(davit_t)
                h = head._florence_lang(input_ids, img_t, mask)
                progress = proprio if lapo_cfg.use_progress else None
                e_t = head.endpoint_predictor(z_t, h, progress)
                cond = head.condition_encoder(z_t, e_t)
                pred = head.decoder.sample(cond, lapo_cfg.chunk_size, 5)
                preds.append(pred)
            svar = ((preds[0] - preds[1])**2).mean()
            m = lapo_action_loss_uniform(preds[0].float(), action.float())
            all_metrics.append(m)
            sample_vars.append(svar.item())
            print(f"  lapo batch {i}: xyz={m['loss_xyz']:.4f} grip_acc={m['grip_acc']:.3f}")

    out = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    out['sample_var'] = np.mean(sample_vars)
    return out


# ============================================================
# xvla_afe5 评估
# ============================================================
def eval_xvla(run_dir, checkpoint_step, nbatches=8):
    """加载 xvla policy (flow matching), 同一 val split, 同一 loss 口径。"""
    from safetensors.torch import load_file
    from lapo.train.compat import build_policy_for, _resolve_base_model, load_state_dict_shape_filtered
    from lapo.train.services.training import load_registry_with_builtins
    from lerobot.policies.xvla.modeling_xvla import resize_with_pad, OBS_LANGUAGE_TOKENS, OBS_STATE, ACTION, pad_vector, pad_tensor_along_dim

    cfg_path = f'{run_dir}/run_config.yaml'
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    rename_map = cfg['dataset']['rename_map']
    registry = load_registry_with_builtins()
    info = json.loads(Path(cfg['dataset']['root'], 'meta/info.json').read_text())
    class Meta:
        fps = info['fps']; features = info['features']
    ds_meta = Meta()

    overrides = dict(cfg['policy_overrides'])
    xvla_overrides = {
        "base_model": overrides['base_model'], "action_mode": "ee6d",
        "max_action_dim": 32, "empty_cameras": 1, "dtype": "float32",
    }
    policy = build_policy_for("xvla", registry, ds_meta, overrides=xvla_overrides, rename_map=rename_map)
    vlm = policy.model.vlm
    resolved = _resolve_base_model(overrides['base_model'])
    sd_base = load_file(str(Path(resolved) / 'model.safetensors'))
    if "model.vlm.image_projection" in sd_base:
        vlm.image_projection.data.copy_(sd_base["model.vlm.image_projection"].to(vlm.image_projection.dtype))
    for suf in ["weight","bias"]:
        k = f"model.vlm.image_proj_norm.{suf}"
        if k in sd_base:
            getattr(vlm.image_proj_norm, suf).data.copy_(sd_base[k])

    ckpt = f'{run_dir}/checkpoints/step_{checkpoint_step}'
    sd = load_file(f'{ckpt}/model.safetensors')
    load_state_dict_shape_filtered(policy, sd)
    print(f"[xvla] checkpoint 加载: {ckpt}")
    policy = policy.cuda().eval()

    # xvla select_action 对 batch 预处理敏感, 用 batch_size=1 逐样本跑
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    info = json.loads(Path(cfg['dataset']['root'], 'meta/info.json').read_text())
    H = 30; fps = info['fps']
    dt = {}
    for k in info['features']:
        if k.startswith('observation.'):
            dt[k] = [i/fps for i in [0, H]]
    dt['action'] = [i/fps for i in range(H)]
    total_ep = info['total_episodes']
    val_eps = list(range(int(total_ep*0.9), total_ep))
    ds = LeRobotDataset(cfg['dataset']['repo_id'], root=cfg['dataset']['root'],
                        delta_timestamps=dt, episodes=val_eps, tolerance_s=0.05)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    all_metrics = []
    sample_vars = []
    nsamples = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            if nsamples >= nbatches: break
            for orig, new in rename_map.items():
                if orig in batch: batch[new] = batch[orig]
            from lapo.train.policies.xvla_tokenizer import tokenize_language_xvla
            batch = tokenize_language_xvla(batch, rename_map=rename_map)
            input_ids = batch[OBS_LANGUAGE_TOKENS].cuda()
            proprio = batch[OBS_STATE]
            if proprio.ndim > 2: proprio = proprio[:, -1, :]
            proprio = pad_vector(proprio.cuda(), policy.config.max_state_dim)
            action = batch[ACTION].cuda()
            if action.ndim == 2: action = action.unsqueeze(1)
            action = pad_tensor_along_dim(action, policy.config.chunk_size, dim=1)
            action = pad_vector(action, policy.config.max_action_dim)

            # select_action 需要 batch dict, 逐样本 (B=1)
            xbatch = {
                OBS_LANGUAGE_TOKENS: input_ids,
                'observation.images.image2': batch['observation.images.image2'][:, 0].cuda(),
                'observation.images.image3': batch.get('observation.images.image3', batch['observation.images.image2'])[:, 0].cuda(),
                OBS_STATE: proprio,
            }
            preds = []
            for _ in range(2):
                pred = policy.select_action(xbatch)
                preds.append(pred)
            svar = ((preds[0].float() - preds[1].float())**2).mean()
            p = preds[0].float()
            if p.ndim == 2: p = p.unsqueeze(0)
            if p.shape[-1] != 10:
                if p.shape[-1] > 10: p = p[..., :10]
                else: p = pad_vector(p, 10)
            if p.shape[1] != action.shape[1]:
                p = pad_tensor_along_dim(p, action.shape[1], dim=1)
            m = lapo_action_loss_uniform(p, action.float())
            all_metrics.append(m)
            sample_vars.append(svar.item())
            nsamples += 1
            print(f"  xvla sample {nsamples}: xyz={m['loss_xyz']:.4f} rot={m['loss_rot']:.4f} grip_acc={m['grip_acc']:.3f}")

    out = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
    out['sample_var'] = np.mean(sample_vars)
    return out


def main():
    print("="*60)
    print(" LAPo S4 (step_2499) vs xvla_afe5 (step_34999) val 对比")
    print("="*60)

    print("\n[1/2] 评估 LAPo S4 final...")
    lapo_m = eval_lapo('/home/gacii/lr-home/outputs/20260720_1022_lapo_ac45/checkpoints/final', nbatches=8)
    print(f"\nLAPo 汇总: {lapo_m}")

    print("\n[2/2] 评估 xvla_afe5 step_34999...")
    xvla_m = eval_xvla('/home/gacii/lr-home/outputs/20260703_1802_xvla_afe5', 34999, nbatches=8)
    print(f"\nxvla 汇总: {xvla_m}")

    print("\n" + "="*60)
    print(" 最终对比")
    print("="*60)
    print(f"\n{'指标':>18} | {'LAPo S4':>10} | {'xvla_afe5':>10} | {'delta':>10} | {'胜者':>8}")
    print("-"*70)
    for k in ['loss_xyz', 'loss_rot', 'loss_grip', 'grip_acc', 'loss_endpoint_disp', 'sample_var']:
        l = lapo_m[k]; x = xvla_m[k]
        d = l - x
        # loss 类: 越小越好; acc 类: 越大越好
        if k == 'grip_acc':
            winner = 'LAPo' if l > x else 'xvla'
        else:
            winner = 'LAPo' if l < x else 'xvla'
        print(f"{k:>18} | {l:>10.4f} | {x:>10.4f} | {d:>+10.4f} | {winner:>8}")

    report = {'lapo_s4_2499': lapo_m, 'xvla_afe5_34999': xvla_m}
    with open('/home/gacii/lr/lapo_vs_xvla_val.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: /home/gacii/lr/lapo_vs_xvla_val.json")


if __name__ == '__main__':
    main()
