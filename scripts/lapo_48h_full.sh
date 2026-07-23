#!/usr/bin/env bash
# LAPo 48h 完整全自动 orchestrator
# 周五晚启动 → 周一回来看报告
#
# 流程:
#   Phase 1: Stage 3 等 step 5000 checkpoint (~5h)
#   Phase 2: Stage 4 joint fine-tune 10000步 (~11h)
#   Phase 3: Stage 4 评估 (~10min)
#   Phase 4: Stage 4 不同 p_oracle 对比实验 (~8h)
#   Phase 5: 回 Stage 1 大 batch 重训对比 (~8h)
#   Phase 6: 全流程综合评估 + 最终报告 (~15min)
#   总计: ~32h (留 16h 余量给 cooldown/异常)
#
set -euo pipefail

cd /home/gacii/lr/lrt
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOG=/home/gacii/lr/lapo_48h.log
REPORT=/home/gacii/lr/LAPO_FINAL_REPORT.txt
OUTPUTS=/home/gacii/lr-home/outputs
TORCHRUN=/home/gacii/miniconda3/envs/lr/bin/torchrun
PYTHON=/home/gacii/miniconda3/envs/lr/bin/python3

STAGE3_RUN="20260717_1630_lapo_edfc"
S1_RUN="20260715_1739_lapo_4354"
S2_RUN="20260717_0856_lapo_4262"

log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

run_train() {
    local config=$1
    local logfile=$2
    log "启动训练: $config → $logfile"
    CUDA_VISIBLE_DEVICES=0,1 $TORCHRUN --nproc_per_node=2 -m lapo.train \
        --config $config --ddp 2>&1 | tee $logfile
}

find_latest_run() {
    ls -dt $OUTPUTS/20260*_lapo_*/ | head -1
}

find_ckpt() {
    local run=$1
    if [ -d "${run}checkpoints/final" ]; then
        echo "${run}checkpoints/final"
    else
        ls -dt ${run}checkpoints/step_*/ 2>/dev/null | head -1
    fi
}

log "========================================"
log " LAPo 48h 完整全自动流程启动"
log " 预计周一早上完成"
log "========================================"

# ============================================
# Phase 1: 等 Stage 3 到 step 5000
# ============================================
log "Phase 1: 等待 Stage 3 (step 5000 checkpoint)"

while true; do
    CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/step_5000/model.safetensors"
    if [ -f "$CKPT" ]; then
        log "✅ Stage 3 step_5000 就绪"
        S3_CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/step_5000"
        break
    fi
    if ! ps -ef | grep "torchrun.*lapo_stage3" | grep -v grep > /dev/null 2>&1; then
        if [ -f "$OUTPUTS/$STAGE3_RUN/checkpoints/final/model.safetensors" ]; then
            log "用 final checkpoint"
            S3_CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/final"
            break
        fi
        # 也检查 step_5000 是否中途出现
        if [ -f "$CKPT" ]; then
            S3_CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/step_5000"
            break
        fi
        log "❌ Stage 3 异常退出无 checkpoint"
        exit 1
    fi
    sleep 300
done

# 停 Stage 3
PID=$(ps -ef | grep "torchrun.*lapo_stage3" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    log "停 Stage 3 (PID $PID)"
    kill $PID 2>/dev/null || true
    sleep 5
    kill -9 $(ps -ef | grep -E "torchrun|lapo.train" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 3
fi

log "Stage 3 最终: $($PYTHON -c "
import json
rows=[]
with open('$OUTPUTS/$STAGE3_RUN/metrics.jsonl') as f:
    for l in f:
        try:
            d=json.loads(l)
            if 'step' in d: rows.append(d)
        except: pass
r=rows[-1]
print(f'step={r[\"step\"]} cos={1-r.get(\"ep_dir_loss\",0):.3f} mse={r.get(\"ep_mse\",0):.3f}')
")"

# ============================================
# Phase 2: Stage 4 joint fine-tune (10000步)
# ============================================
log "Phase 2: Stage 4 joint (10000步)"

S4_CONFIG="configs/openarm_pick_place_green277_lapo_stage4.yaml"
# 改成 10000 步
$PYTHON -c "
import yaml
with open('$S4_CONFIG') as f: c=yaml.safe_load(f)
c['training']['resume_from']='$S3_CKPT'
c['training']['num_steps']=10000
c['training']['save_every']=2500
c['training']['scheduler_decay_steps']=10000
with open('$S4_CONFIG','w') as f: yaml.dump(c,f,default_flow_style=False,sort_keys=False,allow_unicode=True)
"

run_train "$S4_CONFIG" "/home/gacii/lr/train_lapo_stage4.log"

S4_RUN=$(find_latest_run)
S4_CKPT=$(find_ckpt "$S4_RUN")
log "Stage 4 完成: $S4_RUN, ckpt=$S4_CKPT"
log "Stage 4 最终: $($PYTHON -c "
import json
rows=[]
with open('${S4_RUN}metrics.jsonl') as f:
    for l in f:
        try:
            d=json.loads(l)
            if 'step' in d: rows.append(d)
        except: pass
r=rows[-1]
print(f'step={r[\"step\"]} loss={r.get(\"loss\",0):.4f} p_oracle={r.get(\"p_oracle\",0):.3f}')
")"

# ============================================
# Phase 3: Stage 4 评估
# ============================================
log "Phase 3: Stage 4 评估"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/lapo_eval.py \
    --config "$S4_CONFIG" --checkpoint "$S4_CKPT" \
    --output /home/gacii/lr/lapo_eval_s4.json 2>&1 | tee -a $LOG

# ============================================
# Phase 4: 对比实验 - 不同 p_oracle schedule
# ============================================
log "Phase 4: 对比实验 - p_oracle=0.3 起步 (更早用 predictor)"

S4B_CONFIG="configs/openarm_pick_place_green277_lapo_stage4b.yaml"
cp "$S4_CONFIG" "$S4B_CONFIG"
$PYTHON -c "
import yaml
with open('$S4B_CONFIG') as f: c=yaml.safe_load(f)
c['policy_overrides']['p_oracle_start']=0.3
c['policy_overrides']['p_oracle_end']=0.0
c['training']['resume_from']='$S3_CKPT'
c['training']['num_steps']=8000
c['training']['save_every']=2000
c['training']['scheduler_decay_steps']=8000
with open('$S4B_CONFIG','w') as f: yaml.dump(c,f,default_flow_style=False,sort_keys=False,allow_unicode=True)
"

run_train "$S4B_CONFIG" "/home/gacii/lr/train_lapo_stage4b.log"

S4B_RUN=$(find_latest_run)
S4B_CKPT=$(find_ckpt "$S4B_RUN")
log "Stage 4b 完成: $S4B_RUN"

log "Stage 4b 评估"
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/lapo_eval.py \
    --config "$S4B_CONFIG" --checkpoint "$S4B_CKPT" \
    --output /home/gacii/lr/lapo_eval_s4b.json 2>&1 | tee -a $LOG

# ============================================
# Phase 5: Stage 1 大 batch 重训 (对比 baseline)
# ============================================
log "Phase 5: Stage 1 大 batch (128) 重训对比"

S1B_CONFIG="configs/openarm_pick_place_green277_lapo_stage1_bigbatch.yaml"
cp configs/openarm_pick_place_green277_lapo_stage1.yaml "$S1B_CONFIG"
$PYTHON -c "
import yaml
with open('$S1B_CONFIG') as f: c=yaml.safe_load(f)
c['training']['batch_size']=128
c['training']['grad_accumulation_steps']=2  # 有效 batch 256
c['training']['num_steps']=15000
c['training']['save_every']=5000
c['training']['resume_from']=None
with open('$S1B_CONFIG','w') as f: yaml.dump(c,f,default_flow_style=False,sort_keys=False,allow_unicode=True)
"

run_train "$S1B_CONFIG" "/home/gacii/lr/train_lapo_stage1_bigbatch.log"

S1B_RUN=$(find_latest_run)
log "Stage 1 大 batch 完成: $S1B_RUN"

# ============================================
# Phase 6: 综合评估 + 最终报告
# ============================================
log "Phase 6: 综合评估 + 最终报告"

# 评估所有 stage 的最终 checkpoint
for label_json in \
    "Stage1:/home/gacii/lr/lapo_eval_s1.json:$OUTPUTS/$S1_RUN/checkpoints/final:configs/openarm_pick_place_green277_lapo_stage1.yaml" \
    "Stage1-bigbatch:/home/gacii/lr/lapo_eval_s1b.json:$(find_ckpt $S1B_RUN):$S1B_CONFIG" \
    "Stage2:/home/gacii/lr/lapo_eval_s2.json:$OUTPUTS/$S2_RUN/checkpoints/step_2499:configs/openarm_pick_place_green277_lapo_stage2.yaml" \
    "Stage4:/home/gacii/lr/lapo_eval_s4.json:$S4_CKPT:$S4_CONFIG" \
    "Stage4b:/home/gacii/lr/lapo_eval_s4b.json:$S4B_CKPT:$S4B_CONFIG"; do

    label=$(echo $label_json | cut -d: -f1)
    outpath=$(echo $label_json | cut -d: -f2)
    ckpt=$(echo $label_json | cut -d: -f3)
    config=$(echo $label_json | cut -d: -f4)

    if [ -d "$ckpt" ] && [ ! -f "$outpath" ]; then
        log "评估 $label..."
        CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/lapo_eval.py \
            --config "$config" --checkpoint "$ckpt" --output "$outpath" 2>&1 | tee -a $LOG
    fi
done

# 生成综合报告
log "生成综合报告"
$PYTHON << 'PYEOF' > $REPORT
import json, os

OUTPUTS = "/home/gacii/lr-home/outputs"

print("=" * 70)
print("  LAPo 48h 全流程最终报告")
print("=" * 70)
print()

# 各阶段汇总
evals = [
    ("Stage 1 (Direct)", "/home/gacii/lr/lapo_eval_s1.json"),
    ("Stage 1 大batch", "/home/gacii/lr/lapo_eval_s1b.json"),
    ("Stage 2 (SB)", "/home/gacii/lr/lapo_eval_s2.json"),
    ("Stage 4 (joint p=0.5)", "/home/gacii/lr/lapo_eval_s4.json"),
    ("Stage 4b (joint p=0.3)", "/home/gacii/lr/lapo_eval_s4b.json"),
]

print("【oracle vs predictor 动作质量对比】")
print(f"  {'实验':>20} | {'oracle_loss':>11} | {'pred_loss':>9} | {'gap':>8} | {'grip_acc':>8}")
print("  " + "-" * 65)
for label, path in evals:
    if not os.path.exists(path): continue
    with open(path) as f: ev = json.load(f)
    o = ev['oracle'].get('loss_action', 0)
    p = ev['predictor'].get('loss_action', 0)
    g = p - o
    ga = ev['predictor'].get('grip_acc', 0)
    print(f"  {label:>20} | {o:>11.4f} | {p:>9.4f} | {g:>+8.4f} | {ga:>7.1%}")

print()
print("【分通道动作精度 (predictor endpoint)】")
print(f"  {'实验':>20} | {'xyz':>8} | {'rot':>8} | {'grip':>8} | {'smooth':>8}")
print("  " + "-" * 60)
for label, path in evals:
    if not os.path.exists(path): continue
    with open(path) as f: ev = json.load(f)
    pr = ev['predictor']
    print(f"  {label:>20} | {pr.get('loss_xyz',0):>8.4f} | {pr.get('loss_rot',0):>8.4f} | {pr.get('loss_grip',0):>8.4f} | {pr.get('loss_smooth',0):>8.4f}")

print()
print("【SB 多模态性 (sample variance)】")
for label, path in evals:
    if not os.path.exists(path): continue
    with open(path) as f: ev = json.load(f)
    print(f"  {label:>20}: {ev.get('sample_var', 0):.6f}")

print()
print("=" * 70)
print("  结论: 看 predictor loss 和 oracle loss 的 gap")
print("  gap < 0.05  → predictor 足够好, 可直接部署")
print("  gap 0.05-0.2 → predictor 有损耗但可用")
print("  gap > 0.2   → predictor 需要继续改进")
print("=" * 70)
PYEOF

log "报告: $REPORT"
log "========================================"
log " LAPo 48h 全流程完成!"
log " 看报告: cat $REPORT"
log "========================================"
