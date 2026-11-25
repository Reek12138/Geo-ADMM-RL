#!/usr/bin/env bash
set -euo pipefail

trial=$(cd -- "$(dirname -- "$0")" && pwd)
stage1=$1
native_checkout=$2
torch_checkout=$3
frozen_baseline=$4
raw_reference=$5
previous_followup=${6:-}
comparison=${7:-}
arms=(control combined temporal)
decision_budget=600
replay_capacity=8192
training_seed=87109
screen_seed=45123
gate_seed=46209
if [[ -n "$comparison" && -z "$previous_followup" ]]; then
    printf '%s\n' 'The comparison requires a previous followup' >&2
    exit 1
fi
case "$comparison" in
    ''|terminal-replay|correlated-yaw|rollout-router|cross-track|batched-layer|executed-critic|collision-critic|fine-physics|expected-mode-risk|observed-collision|far-conflict|topology-replay|mixed-projection|multi-step-return) ;;
    *)
        printf '%s\n' 'Unknown Stage 1 comparison' >&2
        exit 1
        ;;
esac

test -s "$stage1/development-gate.json"
test -s "$stage1/formal-stage1/comparison.json"
test -s "$stage1/completed.txt"
gate_status() {
    python -c 'import json, sys; print("admit" if json.load(open(sys.argv[1]))["eligible_for_stage2"] else "hold")' "$1"
}
initial_status=$(gate_status "$stage1/development-gate.json")
if [[ $initial_status == admit ]]; then
    printf '%s\n' 'original-stage1-admitted' > "$trial/status.txt"
    exit 0
fi
if [[ -n "$previous_followup" ]]; then
    previous_status=$(cat "$previous_followup/status.txt")
    case "$previous_status" in
        original-stage1-admitted|stage2-completed)
            printf '%s\n' 'previous-stage1-admitted' > "$trial/status.txt"
            exit 0
            ;;
        followup-held-by-development-gate)
            test "$(gate_status "$previous_followup/development-gate.json")" = hold
            ;;
        *)
            printf '%s\n' 'previous-followup-incomplete' > "$trial/status.txt"
            exit 1
            ;;
    esac
    replay_capacity=65536
    decision_budget=450
    if [[ $comparison == multi-step-return ]]; then
        arms=(control multi-step-return)
        training_seed=101079
        screen_seed=71143
        gate_seed=71309
        native_seed=71429
    elif [[ $comparison == mixed-projection ]]; then
        arms=(control mixed-projection)
        training_seed=100129
        screen_seed=70711
        gate_seed=70901
        native_seed=71009
    elif [[ $comparison == topology-replay ]]; then
        arms=(control topology-replay)
        training_seed=99929
        screen_seed=69931
        gate_seed=70001
        native_seed=70111
    elif [[ $comparison == far-conflict ]]; then
        arms=(control far-conflict)
        training_seed=99173
        screen_seed=69109
        gate_seed=69247
        native_seed=69473
    elif [[ $comparison == observed-collision ]]; then
        arms=(control observed-collision)
        decision_budget=466
        training_seed=98299
        screen_seed=68071
        gate_seed=68437
        native_seed=68729
    elif [[ $comparison == expected-mode-risk ]]; then
        arms=(control expected-mode-risk)
        training_seed=97213
        screen_seed=67531
        gate_seed=67801
        native_seed=67943
    elif [[ $comparison == fine-physics ]]; then
        arms=(control fine-physics)
        training_seed=96523
        screen_seed=66701
        gate_seed=67139
        native_seed=67229
    elif [[ $comparison == collision-critic ]]; then
        arms=(control collision-critic)
        training_seed=95003
        screen_seed=65309
        gate_seed=65437
    elif [[ $comparison == executed-critic ]]; then
        arms=(control executed-critic)
        training_seed=94057
        screen_seed=64223
        gate_seed=64451
    elif [[ $comparison == batched-layer ]]; then
        arms=(control batched-layer)
        training_seed=93761
        screen_seed=63617
        gate_seed=63719
    elif [[ $comparison == cross-track ]]; then
        arms=(control cross-track)
        training_seed=93481
        screen_seed=62459
        gate_seed=62731
    elif [[ $comparison == rollout-router ]]; then
        arms=(control rollout-router)
        training_seed=92717
        screen_seed=61703
        gate_seed=61813
    elif [[ $comparison == correlated-yaw ]]; then
        arms=(control correlated-yaw)
        training_seed=92083
        screen_seed=58967
        gate_seed=58741
    elif [[ $comparison == terminal-replay ]]; then
        arms=(control terminal)
        training_seed=91211
        screen_seed=57149
        gate_seed=57601
    else
        arms=(control verified-teacher)
        training_seed=89717
        screen_seed=56503
        gate_seed=56377
        demonstrations=("$trial"/teacher/*.pt)
        test -s "${demonstrations[0]}"
    fi
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$trial/tmp" XDG_CACHE_HOME="$trial/cache" HF_HOME="$trial/cache/hf"
export TORCH_HOME="$trial/cache/torch" PIP_CACHE_DIR="$trial/cache/pip"
export PYTHONPATH="$torch_checkout:$trial/source"
sha256sum --quiet -c "$trial/source-sha256.txt"
sha256sum --quiet -c "$trial/inputs-sha256.txt"
/usr/local/bin/python -m venv --system-site-packages --without-pip "$trial/env"
worker="$trial/env/bin/python"
base="$stage1/run/checkpoint.pt"
cd "$trial/source"
printf '%s\n' 'running-paired-development' > "$trial/status.txt"
sha256sum "$base" > "$trial/original-stage1-sha256.txt"
if [[ -n "$previous_followup" ]]; then
    base=$("$worker" - "$previous_followup/development-gate.json" <<'PY'
import json
from pathlib import Path
import sys
from eval.bluesky_pair import _checksum

report = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(report['checkpoint'])
if _checksum(checkpoint) != report['sha256']:
    raise ValueError('The previous development checkpoint digest changed')
print(checkpoint)
PY
    )
    inherited_arm=$("$worker" -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected"])' \
        "$previous_followup/development-selection.json")
    if [[ -n $comparison && $inherited_arm == verified-teacher ]]; then
        demonstrations=("$previous_followup"/teacher/*.pt)
        test -s "${demonstrations[0]}"
    fi
elif [[ -s "$trial/resume-panel.json" ]]; then
    "$worker" -B -m eval.stage1_resume_selection --panel "$trial/resume-panel.json" --incumbent "$base" \
        --calibration "$stage1/calibration.json" --output "$trial/resume-selection" \
        2>&1 | tee "$trial/resume-selection.log"
    base=$("$worker" -c 'import json, sys; print(json.load(open(sys.argv[1]))["selected"])' \
        "$trial/resume-selection/selection.json")
fi
sha256sum "$base" > "$trial/incoming-sha256.txt"
inherited_options=$("$worker" - "$base" <<'PY'
import json
from pathlib import Path
import sys

config_path = Path(sys.argv[1]).with_name("config.json")
config = json.loads(config_path.read_text()) if config_path.is_file() else {}
print(int(bool(config.get("batched_layer_search"))), int(bool(config.get("executed_action_critic"))),
      float(config.get("cross_track_weight", 0.0)), float(config.get("collision_critic_weight", 0.0)),
      float(config.get("sim_dt", 0.25)), int(bool(config.get("expected_mode_risk"))),
      int(config.get("collision_label_horizon", 0)), config.get("collision_critic_target", "td"),
      int(bool(config.get("topology_replay"))), float(config.get("unprotected_rollout_fraction", 0.0)),
      float(config.get("coordination_weight", 0.0)))
PY
)
read -r inherited_batched inherited_executed inherited_cross_track inherited_collision_weight \
    inherited_sim_dt inherited_expected_mode inherited_collision_horizon inherited_collision_target \
    inherited_topology inherited_unprotected inherited_coordination <<< "$inherited_options"
if [[ $comparison == multi-step-return ]]; then
    return_window=24
    replay_maturity=$return_window
    if ((inherited_collision_horizon > 0)); then
        replay_maturity=$((replay_maturity + inherited_collision_horizon - 1))
    fi
    decision_budget=$((decision_budget + replay_maturity - 4))
elif [[ $comparison == far-conflict || $comparison == topology-replay || $comparison == mixed-projection ]] \
   && ((inherited_collision_horizon > 4)); then
    decision_budget=$((decision_budget + inherited_collision_horizon - 4))
fi
head_on_reference="$frozen_baseline"
if [[ -s "$trial/head-on-reference.txt" ]]; then
    IFS= read -r head_on_reference < "$trial/head-on-reference.txt"
    test -s "$head_on_reference"
fi
limit=$("$worker" - "$base" "$decision_budget" <<'PY'
import sys
import torch

saved = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
print(int(saved['decisions']) + int(sys.argv[2]))
PY
)
ppu-smi -q -d UTILIZATION,MEMORY -l 10 > "$trial/ppu-utilization.log" &
monitor=$!
finish() {
    local code=$?
    kill "$monitor" 2>/dev/null || true
    if ((code != 0)); then
        printf 'failed-paired-development-%s\n' "$code" > "$trial/status.txt"
    fi
}
trap finish EXIT

log_training() {
    local log_path=$1
    shift
    local -a result
    "$@" 2>&1 | tee --output-error=warn "$log_path" || {
        result=("${PIPESTATUS[@]}")
        if ((result[0] != 0)); then
            return "${result[0]}"
        fi
        printf '%s\n' "The training process completed, but writing $log_path reported an error." >&2
    }
}

for arm in "${arms[@]}"; do
    anchor="$frozen_baseline"
    options=(--raw-risk-weight 0.75)
    sim_dt=0.25
    variant=${inherited_arm:-$arm}
    if [[ $variant == combined || $variant == temporal ]]; then
        anchor="$head_on_reference"
        options=(--raw-risk-weight 0.15 --dense-risk --head-on-anchor --safe-navigation-weight 0.35)
    fi
    if [[ $variant == temporal ]]; then
        options+=(--overlap-risk)
    fi
    if [[ $arm == verified-teacher || ${inherited_arm:-} == verified-teacher ]]; then
        options+=(--teacher-data "${demonstrations[@]}" --teacher-weight 0.05)
    fi
    if [[ $arm == terminal || ${inherited_arm:-} == terminal ]]; then
        options+=(--terminal-replay)
    fi
    if [[ $arm == correlated-yaw || ${inherited_arm:-} == correlated-yaw ]]; then
        options+=(--correlated-yaw)
    fi
    if [[ $arm == rollout-router ]]; then
        options+=(--rollout-router-temperature 0.5)
    fi
    if [[ $arm == cross-track ]]; then
        options+=(--cross-track-observation --cross-track-weight 1.0)
    elif [[ $inherited_cross_track != 0.0 ]]; then
        options+=(--cross-track-weight "$inherited_cross_track")
    fi
    if [[ $arm == batched-layer || $inherited_batched == 1 ]]; then
        options+=(--batched-layer-search)
    fi
    if [[ $arm == executed-critic || $comparison == collision-critic || $comparison == observed-collision
          || $comparison == mixed-projection || $inherited_executed == 1 ]]; then
        options+=(--executed-action-critic)
    fi
    if [[ $arm == collision-critic || $comparison == observed-collision ]]; then
        options+=(--collision-critic-weight 40)
    elif [[ $inherited_collision_weight != 0.0 ]]; then
        options+=(--collision-critic-weight "$inherited_collision_weight")
    fi
    if [[ $comparison == fine-physics || $comparison == expected-mode-risk || $comparison == observed-collision
          || $comparison == far-conflict || $comparison == topology-replay || $comparison == mixed-projection
          || $comparison == multi-step-return ]]; then
        options+=(--native-contact-tolerance --event-checks-per-decision 3)
    fi
    if [[ $comparison == expected-mode-risk || $comparison == observed-collision || $comparison == far-conflict
          || $comparison == topology-replay || $comparison == mixed-projection || $comparison == multi-step-return ]]; then
        sim_dt=$inherited_sim_dt
    fi
    if [[ $arm == fine-physics ]]; then
        sim_dt=0.05
    fi
    if [[ $arm == expected-mode-risk || $inherited_expected_mode == 1 ]]; then
        options+=(--expected-mode-risk)
    fi
    if [[ $comparison == observed-collision ]]; then
        options+=(--collision-label-horizon 20)
    fi
    if [[ $arm == observed-collision ]]; then
        options+=(--collision-critic-target observed)
    fi
    if [[ ( $comparison == far-conflict || $comparison == topology-replay || $comparison == mixed-projection
            || $comparison == multi-step-return )
          && $inherited_collision_horizon != 0 ]]; then
        options+=(--collision-label-horizon "$inherited_collision_horizon" --collision-critic-target "$inherited_collision_target")
    fi
    if [[ $arm == far-conflict ]]; then
        options+=(--far-conflict-observation)
    fi
    if [[ $arm == topology-replay || $inherited_topology == 1 ]]; then
        options+=(--topology-replay)
    fi
    if [[ $arm == mixed-projection ]]; then
        options+=(--unprotected-rollout-fraction 0.25)
    elif [[ $inherited_unprotected != 0.0 ]]; then
        options+=(--unprotected-rollout-fraction "$inherited_unprotected")
    fi
    if [[ $comparison == multi-step-return ]]; then
        options+=(--return-window "$return_window")
        if [[ $arm == multi-step-return ]]; then
            options+=(--multi-step-return)
        fi
    fi
    if [[ $inherited_coordination != 0.0 ]]; then
        options+=(--coordination-weight "$inherited_coordination")
    fi
    log_training "$trial/$arm.log" "$worker" -B -m train.bluesky_td3 \
        --plan b --device ppu --envs 512 --aircraft 20 --sim-dt "$sim_dt" --control-dt 6 --decisions "$limit" \
        --updates 12 --batch 64 --inference-batch 256 --replay "$replay_capacity" --checkpoint-every 3000 --log-every 50 \
        --learning-rate 0.00012 --cosine-updates 150000 --seed "$training_seed" --family full_mixture \
        --physics64 --performance-calibration "$stage1/calibration.json" \
        --timed-departures --stratified-replay --admm-k 2 --dense-extra-rounds 2 --directional-heading \
        --shaped-reward --lookahead-weight 0.7 --navigation-weight 1.5 --projection-weight 0.12 \
        --resume "$base" --anchor "$anchor" --anchor-weight 0.65 \
        --output "$trial/$arm" "${options[@]}"
done

for arm in incoming "${arms[@]}"; do
    checkpoint="$trial/$arm/checkpoint.pt"
    if [[ $arm == incoming ]]; then checkpoint=$base; fi
    for mode in projection raw; do
        flags=()
        if [[ $mode == raw ]]; then flags=(--without-projection); fi
        "$worker" -B -m eval.bluesky_stage_gate --checkpoint "$checkpoint" --calibration "$stage1/calibration.json" \
            --output "$trial/$arm-$mode.json" --envs 12 --seed "$screen_seed" "${flags[@]}" \
            2>&1 | tee "$trial/$arm-$mode.log"
    done
done
sha256sum --quiet -c "$trial/incoming-sha256.txt"
sha256sum --quiet -c "$trial/original-stage1-sha256.txt"

arm=$("$worker" - "$trial" "$comparison" "${arms[@]}" <<'PY'
import json
import sys
from pathlib import Path

trial = Path(sys.argv[1])
comparison = sys.argv[2]
scores = {}
for arm in ('incoming', *sys.argv[3:]):
    data = json.loads((trial / f'{arm}-projection.json').read_text())
    counts = data['families']
    for result in counts.values():
        total = sum(result[key] for key in ('arrivals', 'collisions', 'exits', 'timeouts'))
        if total != result['aircraft']:
            raise ValueError(f'Incomplete development counts for {arm}')
    scores[arm] = {
        'arrivals': sum(row['arrivals'] for row in counts.values()),
        'collisions': sum(row['collisions'] for row in counts.values()),
        'head_on': counts['head_on']['arrivals'],
        'action_cost': sum(row['action_cost_per_aircraft'] * row['aircraft'] for row in counts.values())
                       / sum(row['aircraft'] for row in counts.values()),
    }
    raw = json.loads((trial / f'{arm}-raw.json').read_text())
    scores[arm].update(raw_arrival_rate=raw['arrival_rate'], raw_collision_rate=raw['collision_rate'],
                       raw_head_on=raw['families']['head_on']['arrivals'])
incoming = scores['incoming']
head_on_floor = max(incoming['head_on'], 140)
eligible = [name for name, row in scores.items() if row['head_on'] >= head_on_floor
            and row['arrivals'] >= incoming['arrivals'] and row['collisions'] <= incoming['collisions']]
if comparison in ('expected-mode-risk', 'observed-collision', 'far-conflict', 'topology-replay',
                  'mixed-projection', 'multi-step-return'):
    eligible = [name for name in eligible
                if scores[name]['raw_arrival_rate'] >= incoming['raw_arrival_rate']
                and scores[name]['raw_collision_rate'] <= incoming['raw_collision_rate']
                and scores[name]['raw_head_on'] >= incoming['raw_head_on']]
selected = max(eligible, key=lambda name: (scores[name]['arrivals'], -scores[name]['collisions'],
                                          -scores[name]['action_cost'],
                                          scores[name]['raw_arrival_rate'], -scores[name]['raw_collision_rate']),
               default='incoming')
(trial / 'development-selection.json').write_text(json.dumps({
    'scores': scores, 'selected': selected, 'head_on_floor': head_on_floor,
}, indent=2) + '\n')
print(selected)
PY
)
selected="$trial/$arm/checkpoint.pt"
if [[ $arm == incoming ]]; then selected=$base; fi
"$worker" -B -m eval.bluesky_stage_gate --checkpoint "$selected" --calibration "$stage1/calibration.json" \
    --output "$trial/development-gate.json" --envs 32 --seed "$gate_seed" 2>&1 | tee "$trial/development-gate.log"

if [[ -n ${native_seed:-} ]]; then
    PYTHONPATH="$native_checkout:$trial/source" "$worker" -B -m eval.bluesky_pair --device ppu \
        --baseline "$base" --candidates "$trial/control/checkpoint.pt" "$trial/${arms[1]}/checkpoint.pt" \
        --output "$trial/native-development" --seed "$native_seed" --seeds 3 --ablation --ablate-baseline \
        --parallel-pairs 2 \
        --admm-k 2 --dense-extra-rounds 2 --directional-heading --batched-layer-search \
        2>&1 | tee "$trial/native-development.log"
fi

followup_status=$(gate_status "$trial/development-gate.json")
if [[ $followup_status == hold ]]; then
    printf '%s\n' 'followup-held-by-development-gate' > "$trial/status.txt"
    exit 0
fi

export PYTHONPATH="$native_checkout:$trial/source"
"$worker" -B -m eval.bluesky_pair --device ppu --baseline "$frozen_baseline" \
    --candidates "$selected" "$raw_reference" --output "$trial/formal-stage1" \
    --seed 12001 --seeds 5 --admm-k 2 --dense-extra-rounds 2 --directional-heading --ablation \
    2>&1 | tee "$trial/formal-stage1.log"
"$worker" -B -m eval.stage_acceptance --comparison "$trial/formal-stage1/comparison.json" \
    --output "$trial/stage1-acceptance.json"
stage2_arm=${inherited_arm:-$arm}
if [[ $stage2_arm == verified-teacher || $stage2_arm == terminal || $stage2_arm == correlated-yaw
      || $stage2_arm == rollout-router || $stage2_arm == cross-track || $stage2_arm == batched-layer
      || $stage2_arm == executed-critic || $stage2_arm == collision-critic || $stage2_arm == fine-physics
      || $stage2_arm == expected-mode-risk || $stage2_arm == observed-collision || $stage2_arm == far-conflict
      || $stage2_arm == topology-replay || $stage2_arm == mixed-projection || $stage2_arm == multi-step-return ]]; then
    stage2_arm=control
fi
stage2_options=(--stage1-arm "$stage2_arm")
if [[ $arm == verified-teacher || ${inherited_arm:-} == verified-teacher ]]; then
    stage2_options+=(--teacher-data "${demonstrations[@]}" --teacher-weight 0.05)
fi
if [[ $arm == correlated-yaw || ${inherited_arm:-} == correlated-yaw ]]; then
    stage2_options+=(--correlated-yaw)
fi
if [[ $arm == rollout-router ]]; then
    stage2_options+=(--rollout-router-temperature 0.5)
fi
if [[ $arm == cross-track ]]; then
    stage2_options+=(--cross-track-weight 1.0)
elif [[ $inherited_cross_track != 0.0 ]]; then
    stage2_options+=(--cross-track-weight "$inherited_cross_track")
fi
if [[ $arm == batched-layer || $inherited_batched == 1 ]]; then
    stage2_options+=(--batched-layer-search)
fi
if [[ $arm == executed-critic || $comparison == collision-critic || $inherited_executed == 1
      || ( ( $comparison == observed-collision || $comparison == mixed-projection ) && $arm != incoming ) ]]; then
    stage2_options+=(--executed-action-critic)
fi
if [[ $arm == collision-critic || ( $comparison == observed-collision && $arm != incoming ) ]]; then
    stage2_options+=(--collision-critic-weight 40)
elif [[ $inherited_collision_weight != 0.0 ]]; then
    stage2_options+=(--collision-critic-weight "$inherited_collision_weight")
fi
if [[ $arm == expected-mode-risk || $inherited_expected_mode == 1 ]]; then
    stage2_options+=(--expected-mode-risk)
fi
if [[ $arm == observed-collision ]]; then
    stage2_options+=(--collision-label-horizon 20 --collision-critic-target observed)
elif [[ ( $comparison == far-conflict || $comparison == topology-replay || $comparison == mixed-projection
          || $comparison == multi-step-return )
        && $inherited_collision_horizon != 0 ]]; then
    stage2_options+=(--collision-label-horizon "$inherited_collision_horizon"
                    --collision-critic-target "$inherited_collision_target")
fi
if [[ $arm == topology-replay || $inherited_topology == 1 ]]; then
    stage2_options+=(--topology-replay)
fi
if [[ $arm == mixed-projection ]]; then
    stage2_options+=(--unprotected-rollout-fraction 0.25)
elif [[ $inherited_unprotected != 0.0 ]]; then
    stage2_options+=(--unprotected-rollout-fraction "$inherited_unprotected")
fi
if [[ $arm == multi-step-return ]]; then
    stage2_options+=(--return-window "$return_window" --multi-step-return)
fi
if [[ $inherited_coordination != 0.0 ]]; then
    stage2_options+=(--coordination-weight "$inherited_coordination")
fi
"$worker" -B -m train.bluesky_stage2 --gate "$trial/development-gate.json" --checkpoint "$selected" \
    --frozen-baseline "$frozen_baseline" --raw-reference "$raw_reference" --output "$trial/stage2" \
    --episodes 2000 --every-episodes 500 "${stage2_options[@]}" 2>&1 | tee "$trial/stage2.log"
printf '%s\n' 'stage2-completed' > "$trial/status.txt"
