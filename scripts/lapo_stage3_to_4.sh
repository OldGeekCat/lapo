#!/usr/bin/env bash
# 等 Stage 3 到 step 5000 存 checkpoint → 停 → 自动启动 Stage 4
set -euo pipefail

cd /home/gacii/lr/lrt
LOG_DIR=/home/gacii/lr
OUTPUTS=/home/gacii/lr-home/outputs
STAGE3_RUN="20260717_1630_lapo_edfc"
STAGE4_CONFIG="configs/openarm_pick_place_green277_lapo_stage4.yaml"
TORCHRUN=/home/gacii/miniconda3/envs/lr/bin/torchrun

# 找 Stage 3 的 torchrun PID
find_stage3_pid() {
    ps -ef | grep "torchrun.*lapo_stage3" | grep -v grep | awk '{print $2}' | head -1
}

echo "[stage3→4] 等待 Stage 3 存 step_5000 checkpoint..."
echo "[stage3→4] Stage 3 run: $STAGE3_RUN"

while true; do
    CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/step_5000/model.safetensors"
    if [ -f "$CKPT" ]; then
        echo "[stage3→4] ✅ 找到 step_5000 checkpoint!"
        break
    fi
    # 也检查是否进程已停（异常退出）
    PID=$(find_stage3_pid)
    if [ -z "$PID" ]; then
        echo "[stage3→4] ⚠️ Stage 3 进程不在了，检查是否有 final checkpoint..."
        FINAL="$OUTPUTS/$STAGE3_RUN/checkpoints/final/model.safetensors"
        if [ -f "$FINAL" ]; then
            echo "[stage3→4] 用 final checkpoint 代替"
            CKPT_DIR="$OUTPUTS/$STAGE3_RUN/checkpoints/final"
            break
        fi
        echo "[stage3→4] ❌ 没有 checkpoint，退出"
        exit 1
    fi
    # 打印进度
    STEPS=$(wc -l < "$OUTPUTS/$STAGE3_RUN/metrics.jsonl" 2>/dev/null || echo 0)
    echo "[stage3→4] Stage 3 还在跑 (~$STEPS steps), 等待 step 5000 checkpoint..."
    sleep 120
done

# 确定 checkpoint 目录
if [ -z "${CKPT_DIR:-}" ]; then
    CKPT_DIR="$OUTPUTS/$STAGE3_RUN/checkpoints/step_5000"
fi
echo "[stage3→4] checkpoint: $CKPT_DIR"

# 停 Stage 3
PID=$(find_stage3_pid)
if [ -n "$PID" ]; then
    echo "[stage3→4] 停 Stage 3 (PID $PID)..."
    kill $PID 2>/dev/null || true
    sleep 3
    # 残留强杀
    REMAINING=$(ps -ef | grep -E "torchrun|lapo.train" | grep -v grep | awk '{print $2}')
    [ -n "$REMAINING" ] && echo "$REMAINING" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# patch Stage 4 yaml 的 resume_from
/home/gacii/miniconda3/envs/lr/bin/python3 -c "
import yaml
with open('$STAGE4_CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['training']['resume_from'] = '$CKPT_DIR'
with open('$STAGE4_CONFIG','w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
print(f'[stage3→4] Stage 4 resume_from = $CKPT_DIR')
"

# 启动 Stage 4
echo "[stage3→4] 启动 Stage 4 (joint fine-tune)..."
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

$TORCHRUN --nproc_per_node=2 -m lapo.train \
    --config $STAGE4_CONFIG --ddp \
    2>&1 | tee $LOG_DIR/train_lapo_stage4.log

echo "[stage3→4] ✅ Stage 4 完成"
