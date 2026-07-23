#!/usr/bin/env bash
# LAPo Stage 2 → Stage 3 串联训练脚本
# Stage 2: SB bridge (oracle endpoint), resume from Stage 1 final checkpoint
# Stage 3: Endpoint predictor, resume from Stage 2 final checkpoint (自动找)
#
# 用法: bash scripts/lapo_stage2_3_run.sh
set -euo pipefail

cd /home/gacii/lr/lrt
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

TORCHRUN=/home/gacii/miniconda3/envs/lr/bin/torchrun
LOG_DIR=/home/gacii/lr
OUTPUTS=/home/gacii/lr-home/outputs
STAGE2_CONFIG=configs/openarm_pick_place_green277_lapo_stage2.yaml
STAGE3_CONFIG=configs/openarm_pick_place_green277_lapo_stage3.yaml

echo "========================================================"
echo " LAPo Stage 2: Schrödinger Bridge (oracle endpoint)"
echo "========================================================"
$TORCHRUN --nproc_per_node=2 -m lapo.train \
  --config $STAGE2_CONFIG --ddp \
  2>&1 | tee $LOG_DIR/train_lapo_stage2.log

echo ""
echo "Stage 2 完成。查找 final checkpoint..."

# 找 Stage 2 的 run 目录（最新的 lapo run，排除 Stage 1 的）
STAGE2_RUN=$(ls -dt $OUTPUTS/20260*_lapo_* 2>/dev/null | head -1)
STAGE2_CKPT="$STAGE2_RUN/checkpoints/final"

if [ ! -d "$STAGE2_CKPT" ]; then
  echo "❌ 找不到 Stage 2 final checkpoint: $STAGE2_CKPT"
  echo "   手动检查后，修改 $STAGE3_CONFIG 的 resume_from 再跑 Stage 3"
  exit 1
fi

echo "✅ Stage 2 checkpoint: $STAGE2_CKPT"

# 动态修改 Stage 3 yaml 的 resume_from
python3 -c "
import yaml
with open('$STAGE3_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['training']['resume_from'] = '$STAGE2_CKPT'
with open('$STAGE3_CONFIG','w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
print(f'Stage 3 resume_from 已设为: $STAGE2_CKPT')
"

echo ""
echo "========================================================"
echo " LAPo Stage 3: Endpoint Predictor"
echo "========================================================"
$TORCHRUN --nproc_per_node=2 -m lapo.train \
  --config $STAGE3_CONFIG --ddp \
  2>&1 | tee $LOG_DIR/train_lapo_stage3.log

echo ""
echo "========================================================"
echo " ✅ LAPo Stage 2 + Stage 3 全部完成"
echo "========================================================"
