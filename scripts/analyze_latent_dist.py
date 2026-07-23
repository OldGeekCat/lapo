"""轻量 latent 分布分析：直接用 dataloader + 冻结 encoder。"""
import torch, sys, os, yaml, json
import numpy as np
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 用空闲的 GPU 2


def main():
    from safetensors.torch import load_file
    from lapo.train.compat import build_policy_for, _resolve_base_model
    from lapo.train.services.training import load_registry_with_builtins

    cfg_path = str(Path(__file__).resolve().parent.parent / 'configs/openarm_pick_place_green277_lapo_stage2.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    overrides = cfg['policy_overrides']
    rename_map = cfg['dataset']['rename_map']

    registry = load_registry_with_builtins()

    # 读 ds_meta
    info = json.loads(Path(cfg['dataset']['root'], 'meta/info.json').read_text())
    class Meta:
        fps = info['fps']
        features = info['features']
    ds_meta = Meta()

    # 构建 xvla policy (拿 vlm)
    xvla_overrides = {
        "base_model": overrides.get("base_model"),
        "action_mode": "ee6d", "max_action_dim": 32,
        "empty_cameras": 1, "dtype": "float32",
    }
    xvla = build_policy_for("xvla", registry, ds_meta, overrides=xvla_overrides, rename_map=rename_map)
    vlm = xvla.model.vlm

    # 修复 image_projection
    resolved = _resolve_base_model(overrides['base_model'])
    sd_ckpt = load_file(str(Path(resolved) / 'model.safetensors'))
    if "model.vlm.image_projection" in sd_ckpt:
        vlm.image_projection.data.copy_(sd_ckpt["model.vlm.image_projection"].to(vlm.image_projection.dtype))
    for suf in ["weight","bias"]:
        k = f"model.vlm.image_proj_norm.{suf}"
        if k in sd_ckpt:
            getattr(vlm.image_proj_norm, suf).data.copy_(sd_ckpt[k])

    # 构建 encoder
    from lapo.train.policies.sb.components import Encoder
    encoder = Encoder(dim_latent=192, dim_davit=4096, depth=4, heads=4, mlp_ratio=4.0)

    # 加载 Stage 2 的 encoder 权重
    s2_sd = load_file("/home/gacii/lr-home/outputs/20260717_0856_lapo_4262/checkpoints/step_2499/model.safetensors")
    enc_sd = {k.replace('model.encoder.',''): v for k,v in s2_sd.items() if k.startswith('model.encoder.')}
    encoder.load_state_dict(enc_sd)
    print(f"encoder 加载: {len(enc_sd)} 个 key")

    # dataloader
    H = 30
    fps = info['fps']
    dt = {}
    for k in info['features']:
        if k.startswith('observation.'):
            dt[k] = [i/fps for i in [0, H]]
    dt['action'] = [i/fps for i in range(H)]

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(cfg['dataset']['repo_id'], root=cfg['dataset']['root'],
                        delta_timestamps=dt, episodes=None, tolerance_s=0.05)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)

    # 特征提取函数
    def davit_features(image_input, image_mask, vlm):
        b = image_input.shape[0]
        feats = []
        for v in range(image_input.shape[1]):
            if image_mask[:, v].all():
                raw = vlm.vision_tower.forward_features_unpool(image_input[:, v])
                feats.append(raw.mean(dim=1))
            if len(feats) == 2: break
        return torch.cat(feats, dim=-1)

    # 把 encoder + vlm 放到 GPU
    device = 'cuda'
    vlm = vlm.to(device).eval()
    encoder = encoder.to(device).eval()

    from lerobot.policies.xvla.modeling_xvla import resize_with_pad

    # 收集 z_t, e_oracle
    z_list, e_list = [], []
    nbatches = 40

    with torch.no_grad():
        for i, batch in enumerate(dl):
            if i >= nbatches: break
            # 取 image2/image3
            for img_key_orig, img_key_new in rename_map.items():
                if img_key_orig in batch:
                    batch[img_key_new] = batch[img_key_orig]

            imgs_t, imgs_tar = [], []
            for key in ['observation.images.image2', 'observation.images.image3']:
                if key not in batch: continue
                img = batch[key]  # [B, 2, C, H, W]
                img_t = img[:, 0].to(device)
                img_tar = img[:, 1].to(device)
                img_t = resize_with_pad(img_t, 224, 224)
                img_tar = resize_with_pad(img_tar, 224, 224)
                imgs_t.append(img_t)
                imgs_tar.append(img_tar)

            img_t = torch.stack(imgs_t, dim=1).to(torch.float32)
            img_tar = torch.stack(imgs_tar, dim=1).to(torch.float32)
            mask = torch.ones(img_t.shape[0], img_t.shape[1], dtype=torch.bool, device=device)

            davit_t = davit_features(img_t, mask, vlm)
            davit_tar = davit_features(img_tar, mask, vlm)
            z_t = encoder(davit_t)
            e_oracle = encoder(davit_tar)

            z_list.append(z_t.cpu().numpy())
            e_list.append(e_oracle.cpu().numpy())
            print(f"  batch {i}: z{z_t.shape} e{e_oracle.shape}")

    Z = np.concatenate(z_list, 0)
    E = np.concatenate(e_list, 0)
    D = E - Z
    N = Z.shape[0]
    print(f"\n收集: {N} 样本, Z{Z.shape}")

    # === 分析 ===
    z_std = Z.std(0); e_std = E.std(0); d_std = D.std(0)
    print(f"\n{'='*60}")
    print(f" 1. 每维 std")
    print(f"{'='*60}")
    print(f"  z_t:   mean={z_std.mean():.3f} min={z_std.min():.3f} max={z_std.max():.3f}")
    print(f"  delta: mean={d_std.mean():.3f} min={d_std.min():.3f} max={d_std.max():.3f}")
    z_eff = (z_std > 0.1).sum()
    d_eff = (d_std > 0.05).sum()
    d_dead = (d_std < 0.01).sum()
    print(f"  z_t 活跃维(std>0.1): {z_eff}/192")
    print(f"  delta 活跃维(std>0.05): {d_eff}/192")
    print(f"  delta 死维(std<0.01): {d_dead}/192")

    print(f"\n{'='*60}")
    print(f" 2. delta PCA (变化向量的有效维度)")
    print(f"{'='*60}")
    cov = np.cov(D.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.maximum(eigvals, 1e-10)
    total = eigvals.sum()
    cumvar = np.cumsum(eigvals) / total
    for pct in [0.5, 0.8, 0.9, 0.95, 0.99]:
        k = np.searchsorted(cumvar, pct) + 1
        print(f"  解释 {pct*100:.0f}% 方差: {k} 维")
    print(f"  top-10 占比: {eigvals[:10].sum()/total*100:.1f}%")
    print(f"  top-30 占比: {eigvals[:30].sum()/total*100:.1f}%")

    print(f"\n{'='*60}")
    print(f" 3. 可预测性 (z_t → delta 的线性 R²)")
    print(f"{'='*60}")
    Z_aug = np.hstack([Z, np.ones((N,1))])
    from numpy.linalg import lstsq
    A,_,_,_ = lstsq(Z_aug, D, rcond=None)
    D_lin = Z_aug @ A
    r2 = 1 - ((D-D_lin)**2).sum() / ((D-D.mean(0))**2).sum()
    print(f"  线性 R²(z_t→delta) = {r2:.3f}  → 解释 {r2*100:.0f}% 方差")
    A_e,_,_,_ = lstsq(Z_aug, E, rcond=None)
    E_lin = Z_aug @ A_e
    r2_e = 1 - ((E-E_lin)**2).sum() / ((E-E.mean(0))**2).sum()
    print(f"  线性 R²(z_t→endpoint) = {r2_e:.3f}  → 解释 {r2_e*100:.0f}% 方差")

    # ★ 保存最小二乘解: delta = W @ z_t + b
    # 用 ridge regression (L2 正则) 防过拟合: (Z^T Z + λI)^-1 Z^T D
    # 256 样本拟合 37k 参数会过拟合 → 需要正则化
    lam = 1.0  # ridge 强度
    ZtZ = Z.T @ Z + lam * np.eye(Z.shape[1])  # [192, 192]
    ZtD = Z.T @ D  # [192, 192]
    W_ridge = np.linalg.solve(ZtZ, ZtD).T  # [192, 192] (delta = W @ z)
    b_ridge = D.mean(0) - W_ridge.T @ Z.mean(0)

    # 验证 ridge 解的尺度
    D_ridge = Z @ W_ridge.T + b_ridge
    r2_ridge = 1 - ((D - D_ridge)**2).sum() / ((D - D.mean(0))**2).sum()
    delta_scale = np.linalg.norm(D_ridge, axis=1).mean()
    z_scale = np.linalg.norm(Z, axis=1).mean()
    print(f"\n  Ridge (λ={lam}) 验证:")
    print(f"    R² = {r2_ridge:.3f}")
    print(f"    ‖delta_pred‖ = {delta_scale:.2f}, ‖z_t‖ = {z_scale:.2f}, 比值 = {delta_scale/z_scale:.2f}x")
    print(f"    W norm = {np.linalg.norm(W_ridge):.2f}")

    # 用 ridge 解保存
    fit = {'weight': torch.from_numpy(W_ridge.astype(np.float32)),
           'bias': torch.from_numpy(b_ridge.astype(np.float32))}
    torch.save(fit, '/tmp/lapo_lstsq_fit.pt')
    print(f"\n  ★ Ridge 解已保存: /tmp/lapo_lstsq_fit.pt")
    print(f"    W{fit['weight'].shape} (norm={np.linalg.norm(W_ridge):.2f}), b{fit['bias'].shape}")

    print(f"\n{'='*60}")
    print(f" 4. 理论上限: 用 z_t 最近邻预测 delta")
    print(f"{'='*60}")
    # leave-one-out NN
    from scipy.spatial.distance import cdist
    if N <= 300:
        dists = cdist(Z, Z)
        np.fill_diagonal(dists, 1e10)
        nn_idx = dists.argmin(axis=1)
        D_nn = D[nn_idx]
        r2_nn = 1 - ((D-D_nn)**2).sum() / ((D-D.mean(0))**2).sum()
        cos_nn = (D*D_nn).sum(1) / (np.linalg.norm(D,axis=1)*np.linalg.norm(D_nn,axis=1)+1e-8)
        print(f"  最近邻 R² = {r2_nn:.3f}")
        print(f"  最近邻 cos = {cos_nn.mean():.3f}")
        print(f"  → 这是非参数方法的上限参考")

    print(f"\n{'='*60}")
    print(f" 结论")
    print(f"{'='*60}")
    print(f"  delta 有效维度: ~{np.searchsorted(cumvar,0.95)+1} 维 (解释95%方差)")
    print(f"  线性可预测: R²={r2:.2f}")
    print(f"  → predictor 的 MSE 下界 ≈ {(1-r2)*d_std.mean()**2 * 192:.1f}")

if __name__ == '__main__':
    main()
