#!/usr/bin/env bash
# LAPo 48h 全自动 orchestrator
# 流程: Stage3完成(等step5000) → Stage4(joint 5000步) → 评估 → 汇总报告
# 全自动, 周末回来直接看 /home/gacii/lr/LAPO_FINAL_REPORT.txt
set -euo pipefail

cd /home/gacii/lr/lrt
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

LOG=/home/gacii/lr/lapo_orchestrator.log
REPORT=/home/gacii/lr/LAPO_FINAL_REPORT.txt
OUTPUTS=/home/gacii/lr-home/outputs
TORCHRUN=/home/gacii/miniconda3/envs/lr/bin/torchrun
PYTHON=/home/gacii/miniconda3/envs/lr/bin/python3

STAGE3_RUN="20260717_1630_lapo_edfc"
STAGE4_CONFIG="configs/openarm_pick_place_green277_lapo_stage4.yaml"

log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }

log "========================================"
log " LAPo 48h 全自动流程启动"
log "========================================"

# ============================================
# Phase 1: 等 Stage 3 到 step 5000 checkpoint
# ============================================
log "Phase 1: 等待 Stage 3 (step 5000 checkpoint)"
while true; do
    CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/step_5000/model.safetensors"
    if [ -f "$CKPT" ]; then
        log "✅ Stage 3 step_5000 checkpoint 就绪"
        STAGE3_CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/step_5000"
        break
    fi
    # 检查进程
    if ! ps -ef | grep "torchrun.*lapo_stage3" | grep -v grep > /dev/null; then
        log "⚠️ Stage 3 进程不在, 检查 final checkpoint"
        if [ -f "$OUTPUTS/$STAGE3_RUN/checkpoints/final/model.safetensors" ]; then
            log "用 final checkpoint"
            STAGE3_CKPT="$OUTPUTS/$STAGE3_RUN/checkpoints/final"
            break
        fi
        log "❌ Stage 3 异常退出且无 checkpoint"
        exit 1
    fi
    STEPS=$(wc -l < "$OUTPUTS/$STAGE3_RUN/metrics.jsonl" 2>/dev/null || echo 0)
    log "  Stage 3 在跑 (~$STEPS steps), 等待 step 5000..."
    sleep 300
done

# 停 Stage 3 (如果还在跑)
PID=$(ps -ef | grep "torchrun.*lapo_stage3" | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$PID" ]; then
    log "停 Stage 3 (PID $PID)"
    kill $PID 2>/dev/null || true
    sleep 3
    kill -9 $(ps -ef | grep -E "torchrun|lapo.train" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 2
fi

# 记录 Stage 3 最终状态
log "Stage 3 最终状态:"
$PYTHON -c "
import json
rows=[]
with open('$OUTPUTS/$STAGE3_RUN/metrics.jsonl') as f:
    for l in f:
        try:
            d=json.loads(l)
            if 'step' in d: rows.append(d)
        except: pass
r=rows[-1]
print(f'  step={r[\"step\"]} ep_mse={r.get(\"ep_mse\",0):.3f} cos={1-r.get(\"ep_dir_loss\",0):.3f}')
" | tee -a $LOG

# ============================================
# Phase 2: Stage 4 (joint fine-tune)
# ============================================
log "Phase 2: Stage 4 (joint fine-tune, 5000步)"

# patch resume_from
$PYTHON -c "
import yaml
with open('$STAGE4_CONFIG') as f: cfg=yaml.safe_load(f)
cfg['training']['resume_from']='$STAGE3_CKPT'
with open('$STAGE4_CONFIG','w') as f: yaml.dump(cfg,f,default_flow_style=False,sort_keys=False,allow_unicode=True)
print(f'resume_from=$STAGE3_CKPT')
" | tee -a $LOG

$TORCHRUN --nproc_per_node=2 -m lapo.train --config $STAGE4_CONFIG --ddp 2>&1 | tee -a $LOG

log "Stage 4 训练完成"

# ============================================
# Phase 3: 评估
# ============================================
log "Phase 3: 全流程评估"

# 找 Stage 4 的 final checkpoint
S4_RUN=$(ls -dt $OUTPUTS/20260*_lapo_*/ | head -1)
S4_CKPT="${S4_RUN}checkpoints/final"

if [ ! -d "$S4_CKPT" ]; then
    log "⚠️ 找不到 Stage 4 final checkpoint, 用最新的 step_"
    S4_CKPT=$(ls -dt ${S4_RUN}checkpoints/step_*/ 2>/dev/null | head -1)
fi

log "Stage 4 checkpoint: $S4_CKPT"

# 记录 Stage 4 最终状态
log "Stage 4 最终状态:"
$PYTHON -c "
import json
rows=[]
with open('${S4_RUN}metrics.jsonl') as f:
    for l in f:
        try:
            d=json.loads(l)
            if 'step' in d: rows.append(d)
        except: pass
r=rows[-1]
print(f'  step={r[\"step\"]} loss={r.get(\"loss\",0):.4f} p_oracle={r.get(\"p_oracle\",0):.3f}')
" | tee -a $LOG

# 跑评估
export CUDA_VISIBLE_DEVICES=0
$PYTHON scripts/lapo_eval.py --config $STAGE4_CONFIG --checkpoint $S4_CKPT \
    --output /home/gacii/lr/lapo_eval_report.json 2>&1 | tee -a $LOG

# ============================================
# Phase 4: 汇总报告
# ============================================
log "Phase 4: 生成最终报告"

$PYTHON << 'PYEOF' > $REPORT
import json, os

OUTPUTS = "/home/gacii/lr-home/outputs"
runs = sorted([d for d in os.listdir(OUTPUTS) if '_lapo_' in d])

print("=" * 60)
print("  LAPo 全流程最终报告")
print("=" * 60)
print()

# 各 stage 最佳指标
stages = {
    "Stage 1 (Direct Decoder)": "20260715_1739_lapo_4354",
    "Stage 2 (SB Bridge)": "20260717_0856_lapo_4262",
    "Stage 3 (Predictor)": "20260717_1630_lapo_edfc",
}

for label, run in stages.items():
    path = f"{OUTPUTS}/{run}/metrics.jsonl"
    if not os.path.exists(path):
        continue
    rows = []
    with open(path) as f:
        for l in f:
            try:
                d = json.loads(l)
                if 'step' in d: rows.append(d)
            except: pass
    if not rows: continue
    print(f"【{label}】({run})")
    val_rows = [r for r in rows if isinstance(r.get('val_loss'), float)]
    if val_rows:
        best_val = min(val_rows, key=lambda r: r['val_loss'])
        print(f"  最佳 val_loss: {best_val['val_loss']:.4f} (step {best_val['step']})")
    r = rows[-1]
    for k in ['loss','loss_action','ep_mse','loss_imle','loss_force','mag_z_t']:
        if k in r:
            print(f"  最终 {k}: {r[k]:.4f}")
    print()

# Stage 4
s4_runs = [d for d in runs if d > "20260717_16" and d != "20260717_1630_lapo_edfc"]
if s4_runs:
    s4 = s4_runs[-1]
    path = f"{OUTPUTS}/{s4}/metrics.jsonl"
    if os.path.exists(path):
        rows = []
        with open(path) as f:
            for l in f:
                try:
                    d = json.loads(l)
                    if 'step' in d: rows.append(d)
                except: pass
        if rows:
            print(f"【Stage 4 (Joint Fine-tune)】({s4})")
            val_rows = [r for r in rows if isinstance(r.get('val_loss'), float)]
            if val_rows:
                best_val = min(val_rows, key=lambda r: r['val_loss'])
                print(f"  最佳 val_loss: {best_val['val_loss']:.4f} (step {best_val['step']})")
            r = rows[-1]
            for k in ['loss','loss_action','p_oracle','oracle_ratio','mag_z_t']:
                if k in r:
                    print(f"  最终 {k}: {r[k]}")
            print()

# 评估报告
eval_path = "/home/gacii/lr/lapo_eval_report.json"
if os.path.exists(eval_path):
    with open(eval_path) as f:
        ev = json.load(f)
    print("【评估: oracle vs predictor endpoint】")
    print(f"  {'指标':>16} | {'oracle':>8} | {'predictor':>8} | {'gap':>8}")
    print("  " + "-" * 50)
    for k in ['loss_action','loss_xyz','loss_rot','loss_grip','grip_acc']:
        o = ev['oracle'].get(k, 0); p = ev['predictor'].get(k, 0)
        g = ev.get('gap', {}).get(k, 0)
        print(f"  {k:>16} | {o:>8.4f} | {p:>8.4f} | {g:>+8.4f}")
    print(f"\n  sample_var: {ev.get('sample_var', 0):.6f}")
    print()

print("=" * 60)
print("  全流程完成")
print("=" * 60)
PYEOF

log "报告已生成: $REPORT"
log "========================================"
log " LAPo 48h 全自动流程完成"
log " 周末回来看: $REPORT"
log "========================================"
