#!/usr/bin/env python3
"""LAPo 全流程评估：对比 oracle endpoint vs predictor endpoint 的动作质量。

Stage 4 完成后运行。输出:
  - loss_with_oracle_e: 用 oracle endpoint 的 action loss (上限)
  - loss_with_pred_e:   用 predictor endpoint 的 action loss (实际部署)
  - gap: 两者差距 (越小越好)
  - action MSE 分通道 (xyz/rot/grip)
  - SB sample variance (多模态性)
"""
import torch, sys, os, json, yaml
import numpy as np
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(Path(__file__).resolve().parent.parent / 'configs/openarm_pick_place_green277_lapo_stage4.yaml'))
    p.add_argument('--checkpoint', required=True, help='Stage 4 final checkpoint dir')
    p.add_argument('--output', default='/home/gacii/lr/lapo_eval_report.json')
    args = p.parse_args()

    from safetensors.torch import load_file
    from lapo.train.compat import build_policy_for, _resolve_base_model
    from lapo.train.services.training import load_registry_with_builtins
    from lapo.train.policies.lapo.config import LapoConfig
    from lapo.train.policies.lapo.model import LapoBridge, lapo_action_loss
    from lapo.train.policies.lapo.policy import LapoPolicy
    from lerobot.policies.xvla.modeling_xvla import resize_with_pad, OBS_LANGUAGE_TOKENS, OBS_STATE, ACTION, pad_vector, pad_tensor_along_dim

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    overrides = cfg['policy_overrides']
    rename_map = cfg['dataset']['rename_map']

    registry = load_registry_with_builtins()
    info = json.loads(Path(cfg['dataset']['root'], 'meta/info.json').read_text())
    class Meta:
        fps = info['fps']
        features = info['features']
    ds_meta = Meta()

    # 构建 policy
    xvla_overrides = {
        "base_model": overrides['base_model'], "action_mode": "ee6d",
        "max_action_dim": 32, "empty_cameras": 1, "dtype": "float32",
    }
    xvla = build_policy_for("xvla", registry, ds_meta, overrides=xvla_overrides, rename_map=rename_map)
    vlm = xvla.model.vlm

    # 修复 image projection
    resolved = _resolve_base_model(overrides['base_model'])
    sd_base = load_file(str(Path(resolved) / 'model.safetensors'))
    if "model.vlm.image_projection" in sd_base:
        vlm.image_projection.data.copy_(sd_base["model.vlm.image_projection"].to(vlm.image_projection.dtype))
    for suf in ["weight","bias"]:
        k = f"model.vlm.image_proj_norm.{suf}"
        if k in sd_base:
            getattr(vlm.image_proj_norm, suf).data.copy_(sd_base[k])

    # 构建 LapoConfig (Stage 4 config)
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
    # 加载 checkpoint
    sd = load_file(str(Path(args.checkpoint) / 'model.safetensors'))
    raw = head
    from lapo.train.compat import load_state_dict_shape_filtered
    load_state_dict_shape_filtered(raw, sd)
    print(f"checkpoint 加载完成: {args.checkpoint}")

    policy = LapoPolicy(lapo_cfg, head, xvla_config=xvla.config)
    policy = policy.cuda().eval()
    head = policy.model

    # dataloader
    H = 30; fps = info['fps']
    dt = {}
    for k in info['features']:
        if k.startswith('observation.'):
            dt[k] = [i/fps for i in [0, H]]
    dt['action'] = [i/fps for i in range(H)]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    # val split: 最后 10% episodes
    total_ep = info['total_episodes']
    val_eps = list(range(int(total_ep*0.9), total_ep))
    ds = LeRobotDataset(cfg['dataset']['repo_id'], root=cfg['dataset']['root'],
                        delta_timestamps=dt, episodes=val_eps, tolerance_s=0.05)
    dl = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)

    # 评估函数
    def eval_batch(batch, use_oracle):
        """返回 (action_loss_dict, pred_actions)."""
        # rename
        for orig, new in rename_map.items():
            if orig in batch:
                batch[new] = batch[orig]
        # 准备输入
        input_ids = batch[OBS_LANGUAGE_TOKENS].cuda()
        imgs_t, imgs_tar = [], []
        for key in ['observation.images.image2', 'observation.images.image3']:
            if key not in batch: continue
            img = batch[key].cuda()
            imgs_t.append(resize_with_pad(img[:,0], 224, 224))
            imgs_tar.append(resize_with_pad(img[:,1], 224, 224))
        img_t = torch.stack(imgs_t, dim=1).float()
        img_tar = torch.stack(imgs_tar, dim=1).float()
        mask = torch.ones(img_t.shape[0], img_t.shape[1], dtype=torch.bool).cuda()
        # proprio
        proprio = batch[OBS_STATE]
        if proprio.ndim > 2: proprio = proprio[:, -1, :]
        proprio = pad_vector(proprio.cuda(), lapo_cfg.dim_proprio)
        # action target
        action = batch[ACTION].cuda()
        if action.ndim == 2: action = action.unsqueeze(1)
        action = pad_tensor_along_dim(action, lapo_cfg.chunk_size, dim=1)
        action = pad_vector(action, lapo_cfg.dim_action)

        dtype = next(head.encoder.parameters()).dtype
        img_t = img_t.to(dtype); img_tar = img_tar.to(dtype)

        with torch.no_grad():
            davit_t = head._davit_features(img_t, mask)
            z_t = head.encoder(davit_t)

            if use_oracle:
                davit_tar = head._davit_features(img_tar, mask)
                e_t = head.encoder(davit_tar)
            else:
                h = head._florence_lang(input_ids, img_t, mask)
                progress = proprio if lapo_cfg.use_progress else None
                e_t = head.endpoint_predictor(z_t, h, progress)

            cond = head.condition_encoder(z_t, e_t)
            pred = head.decoder.sample(cond, lapo_cfg.chunk_size, 5)

        loss, metrics = lapo_action_loss(pred, action, lapo_cfg)
        return metrics, pred.detach().cpu().numpy()

    # 跑评估
    oracle_metrics = []
    pred_metrics = []
    sample_vars = []
    nbatches = 10

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= nbatches: break
            print(f"  eval batch {i}/{nbatches}...")
            om, _ = eval_batch(batch, use_oracle=True)
            pm, pred1 = eval_batch(batch, use_oracle=False)
            # sample variance: 再跑一次 predictor 看 SB 采样差异
            _, pred2 = eval_batch(batch, use_oracle=False)
            svar = ((pred1 - pred2)**2).mean()
            oracle_metrics.append({k: v.item() for k,v in om.items()})
            pred_metrics.append({k: v.item() for k,v in pm.items()})
            sample_vars.append(svar.item())

    # 汇总
    def avg(metrics_list, key):
        vals = [m[key] for m in metrics_list if key in m]
        return sum(vals)/len(vals) if vals else 0

    report = {
        'checkpoint': args.checkpoint,
        'nbatches': nbatches,
        'oracle': {k: avg(oracle_metrics, k) for k in oracle_metrics[0]},
        'predictor': {k: avg(pred_metrics, k) for k in pred_metrics[0]},
        'sample_var': sum(sample_vars)/len(sample_vars),
    }
    # gap
    for k in report['oracle']:
        o = report['oracle'][k]; p = report['predictor'].get(k, 0)
        report.setdefault('gap', {})[k] = p - o

    # 打印
    print(f"\n{'='*60}")
    print(f" LAPo 评估报告")
    print(f"{'='*60}")
    print(f"\n{'指标':>16} | {'oracle':>8} | {'predictor':>8} | {'gap':>8}")
    print('-'*55)
    for k in ['loss_action','loss_xyz','loss_rot','loss_grip','grip_acc','loss_endpoint_disp']:
        o = report['oracle'].get(k, 0); p = report['predictor'].get(k, 0)
        g = report.get('gap', {}).get(k, 0)
        print(f"{k:>16} | {o:>8.4f} | {p:>8.4f} | {g:>+8.4f}")
    print(f"\n{'sample_var':>16} | {report['sample_var']:.6f}")
    print(f"\n{'='*60}")
    gap_action = report.get('gap',{}).get('loss_action', 0)
    if abs(gap_action) < 0.05:
        print(f" ✅ predictor 和 oracle 动作质量接近 (gap={gap_action:+.4f})")
    elif gap_action < 0:
        print(f" 🎉 predictor 比 oracle 更好?! (gap={gap_action:+.4f})")
    else:
        print(f" ⚠️ predictor 比 oracle 差 (gap={gap_action:+.4f})")
    print(f"{'='*60}")

    # 保存
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n报告已保存: {args.output}")

if __name__ == '__main__':
    main()
