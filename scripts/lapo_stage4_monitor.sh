#!/usr/bin/env bash
# Stage 4 joint fine-tune 监控脚本
# 在关键判断点 (500/1000/3000/5000/10000) 输出状态 + 异常告警
#
# 判断逻辑:
#   step 500:   warmup 结束, p_oracle 开始衰减 → 看 ep_mse 是否回升
#   step 1000:  第一个判断点 → ep_mse 应该 < 0.9
#   step 3000:  第二个判断点 → ep_mse 应该 < 0.6, action 稳定
#   step 5000:  中场 → ep_mse 应该 < 0.4
#   step 10000: 结束 → 触发评估
#
# 异常告警条件:
#   - action_loss > 0.15 (bridge 崩)
#   - mag_z_t > 8.0      (latent 膨胀, shortcut 出现)
#   - 连续 3 个 checkpoint ep_mse 单调上升 (predictor 退化)
#
# 用法:
#   ./lapo_stage4_monitor.sh
#   nohup ./lapo_stage4_monitor.sh > /home/gacii/lr/stage4_monitor.log 2>&1 &

set -uo pipefail

RUN_DIR="/home/gacii/lr-home/outputs/20260720_1022_lapo_ac45"
METRICS="$RUN_DIR/metrics.jsonl"
STATUS_FILE="/home/gacii/lr/stage4_status.json"   # 最新状态 (机器可读)
ALERT_FILE="/home/gacii/lr/stage4_alerts.log"     # 告警历史

# 已触发的判断点 (避免重复)
declare -A TRIGGERED

log() {
    echo "[$(date '+%m-%d %H:%M:%S')] $*"
}

alert() {
    echo "[$(date '+%m-%d %H:%M:%S')] 🚨 $*" | tee -a "$ALERT_FILE"
}

# 拿最新一行 metric (python 解析)
get_latest() {
    tail -1 "$METRICS" 2>/dev/null
}

# 拿某 step 附近的窗口 (前后 20 步均值, 用来去抖动)
window_avg() {
    local target=$1
    python3 -c "
import json
target = $target
rows = []
with open('$METRICS') as f:
    for line in f:
        d = json.loads(line)
        if abs(d['step'] - target) <= 30:
            rows.append(d)
if not rows:
    print('{}')
    exit()
import statistics as st
keys = ['loss', 'loss_action', 'ep_mse', 'ep_dir_loss', 'mag_z_t', 'mag_e', 'p_oracle']
out = {'step': rows[-1]['step'], 'n': len(rows)}
for k in keys:
    vals = [r.get(k, 0) for r in rows if k in r]
    if vals:
        out[k] = sum(vals)/len(vals)
print(json.dumps(out))
"
}

# 在判断点输出详细状态
checkpoint_report() {
    local label=$1
    local step=$2
    local data=$3

    log ""
    log "═══════════════════════════════════════════════════"
    log "  📊 $label (step $step)"
    log "═══════════════════════════════════════════════════"

    echo "$data" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
if not d: print('  (无数据)'); exit()
step = d.get('step', '?')
loss = d.get('loss', 0)
action = d.get('loss_action', 0)
ep_mse = d.get('ep_mse', 0)
mag_z = d.get('mag_z_t', 0)
mag_e = d.get('mag_e', 0)
p_or = d.get('p_oracle', 0)
print(f'  step       = {step}')
print(f'  loss_total = {loss:.4f}')
print(f'  action     = {action:.4f}   (阈值 0.15)')
print(f'  ep_mse     = {ep_mse:.4f}   (越小越好)')
print(f'  mag_z_t    = {mag_z:.4f}    (阈值 8.0)')
print(f'  mag_e      = {mag_e:.4f}')
print(f'  p_oracle   = {p_or:.3f}')
"
    log "═══════════════════════════════════════════════════"
}

# 实时告警检测
check_realtime_alerts() {
    local data=$1
    echo "$data" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
if not d: exit()
action = d.get('loss_action', 0)
mag_z = d.get('mag_z_t', 0)
step = d.get('step', 0)
alerts = []
if action > 0.15:
    alerts.append(f'action_loss={action:.4f} > 0.15 (bridge 崩?)')
if mag_z > 8.0:
    alerts.append(f'mag_z_t={mag_z:.2f} > 8.0 (latent 膨胀 shortcut)')
if alerts:
    print(f'step {step}: ' + ' | '.join(alerts))
"
}

# 主循环
log "🚀 Stage 4 监控启动"
log "  run: $RUN_DIR"
log "  metrics: $METRICS"
log "  判断点: 500 / 1000 / 3000 / 5000 / 10000"
log ""

# 判断点定义
declare -a CHECKPOINTS=(500 1000 3000 5000 10000)
declare -a LABELS=("warmup结束,p_oracle开始衰减" "第一判断点:ep_mse应<0.9" "第二判断点:ep_mse应<0.6" "中场:ep_mse应<0.4" "训练结束")

LAST_STEP=0
EPI_MSE_HISTORY=()

while true; do
    LATEST=$(get_latest)
    if [ -z "$LATEST" ]; then
        sleep 60
        continue
    fi

    CUR_STEP=$(echo "$LATEST" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('step',0))")
    CUR_EP=$(echo "$LATEST" | python3 -c "import json,sys; print(round(json.loads(sys.stdin.read()).get('ep_mse',0),4))")

    # 实时告警
    ALERT=$(check_realtime_alerts "$LATEST")
    if [ -n "$ALERT" ]; then
        alert "$ALERT"
    fi

    # 检查是否进程还在
    PID=$(pgrep -f "torchrun.*lapo_stage4" | head -1)
    if [ -z "$PID" ] && [ "$CUR_STEP" -lt 10000 ]; then
        log "⚠️ Stage 4 进程不在了 (step=$CUR_STEP < 10000)"
        log "  可能崩溃或被手动停了,检查日志: /home/gacii/lr/train_lapo_stage4.log"
        # 进程死掉就退出监控
        break
    fi

    # 触发判断点
    for i in "${!CHECKPOINTS[@]}"; do
        CP=${CHECKPOINTS[$i]}
        LABEL=${LABELS[$i]}
        if [ "$CUR_STEP" -ge "$CP" ] && [ -z "${TRIGGERED[$CP]:-}" ]; then
            TRIGGERED[$CP]=1
            DATA=$(window_avg $CP)
            checkpoint_report "$LABEL" "$CP" "$DATA"

            # ep_mse 退化检测 (从 step 1000 开始)
            if [ "$CP" -ge 1000 ]; then
                EP=$(echo "$DATA" | python3 -c "import json,sys; print(round(json.loads(sys.stdin.read()).get('ep_mse',0),4))")
                EPI_MSE_HISTORY+=("$EP")
                if [ "${#EPI_MSE_HISTORY[@]}" -ge 3 ]; then
                    N=${#EPI_MSE_HISTORY[@]}
                    A=${EPI_MSE_HISTORY[$((N-3))]}
                    B=${EPI_MSE_HISTORY[$((N-2))]}
                    C=${EPI_MSE_HISTORY[$((N-1))]}
                    # python 浮点比较
                    INCREASE=$(python3 -c "print(1 if ($A < $B < $C) else 0)")
                    if [ "$INCREASE" = "1" ]; then
                        alert "ep_mse 连续 3 个判断点单调上升: $A → $B → $C (predictor 退化)"
                    fi
                fi
            fi

            # step 10000: 触发评估
            if [ "$CP" = "10000" ]; then
                log ""
                log "🎉 Stage 4 训练完成!"
                log "  checkpoint: $RUN_DIR/checkpoints/final"
                log ""
                log "下一步评估命令:"
                log "  cd /home/gacii/lr/lrt && python3 scripts/lapo_eval.py \\"
                log "    --checkpoint $RUN_DIR/checkpoints/final \\"
                log "    --output /home/gacii/lr/lapo_stage4_eval.json"
                break 2
            fi
        fi
    done

    # 进度提示 (每 10 分钟一次)
    if [ $((CUR_STEP - LAST_STEP)) -ge 60 ] || [ $LAST_STEP -eq 0 ]; then
        ETA_STEPS=$((10000 - CUR_STEP))
        # 估算 ETA: 用最近 100 步的速度
        ETA_MIN=$(python3 -c "
import json
rows = []
with open('$METRICS') as f:
    for line in f:
        rows.append(json.loads(line))
if len(rows) < 2: print('?'); exit()
recent = rows[-100:] if len(rows) > 100 else rows
t0 = recent[0].get('timestamp',''); t1 = recent[-1].get('timestamp','')
from datetime import datetime
try:
    d0 = datetime.fromisoformat(t0); d1 = datetime.fromisoformat(t1)
    dt = (d1-d0).total_seconds()/60
    ds = recent[-1]['step'] - recent[0]['step']
    if ds <= 0 or dt <= 0: print('?'); exit()
    rate = ds / dt  # steps / min
    eta = $ETA_STEPS / rate
    print(f'{eta:.1f}min ({rate:.1f}step/min)')
except: print('?')
")
        log "⏳ step $CUR_STEP/10000 | ep_mse=$CUR_EP | ETA 剩余 ~$ETA_MIN"

        # 写状态文件
        echo "$LATEST" | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
status = {
    'step': d.get('step', 0),
    'total_steps': 10000,
    'progress': round(d.get('step',0)/100, 2),
    'loss': round(d.get('loss',0), 4),
    'action_loss': round(d.get('loss_action',0), 4),
    'ep_mse': round(d.get('ep_mse',0), 4),
    'mag_z_t': round(d.get('mag_z_t',0), 3),
    'p_oracle': round(d.get('p_oracle',0), 3),
    'eta_min_remaining': '$ETA_MIN',
}
with open('$STATUS_FILE', 'w') as f:
    json.dump(status, f, indent=2)
" 2>/dev/null

        LAST_STEP=$CUR_STEP
    fi

    sleep 30
done

log ""
log "监控结束.告警历史见: $ALERT_FILE"
log "最终状态: $STATUS_FILE"
