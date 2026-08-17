#!/bin/bash
# Watcher: when fix_recap part DONE.txt appears for all 20 parts,
# dispatch 47m top-up packing to 10 hosts in 2 rounds.
set -euo pipefail
export PYTHONUNBUFFERED=1

LOG=/tmp/watcher_pandas70m_recap_topup.log
exec >>"${LOG}" 2>&1

ts() { date "+[%Y-%m-%d %H:%M:%S]"; }

PIPE=/vlm/yinxie/code/OV2/OV2_public_main/offline_packing
FIX_RECAP_ROOT=/ov21/pandas70m_gt10min_packed_64k_fix_recap
TOPUP_ROOT=/ov21/pandas70m_gt10min_packed_64k_fix_recap_with47m

HOSTS=(172.16.5.32 172.16.5.38 172.16.5.40 172.16.5.41 172.16.5.43 \
       172.16.5.44 172.16.5.45 172.16.5.46 172.16.5.47 172.16.5.49)
declare -A HOST_IMG=(
    ["172.16.5.49"]="llava_megatron:26.05"
)
DEFAULT_IMG="llava_megatron:25.12"

mkdir -p "${TOPUP_ROOT}"

echo "$(ts) === Recap top-up watcher started ==="
echo "$(ts) FIX_RECAP_ROOT=${FIX_RECAP_ROOT}"
echo "$(ts) TOPUP_ROOT=${TOPUP_ROOT}"

# Phase 1: wait for all 20 fix_recap parts DONE
echo "$(ts) Phase 1: wait for all 20 fix_recap parts DONE..."
while true; do
    n_done=0
    for i in $(seq 0 19); do
        [[ -f ${FIX_RECAP_ROOT}/pandas70m_part${i}/DONE.txt ]] && n_done=$((n_done+1))
    done
    if (( n_done == 20 )); then
        echo "$(ts) Phase 1: all 20 fix_recap parts DONE."
        break
    fi
    echo "$(ts)   fix_recap DONE: ${n_done}/20"
    sleep 300
done

launch_part() {
    local idx=$1 host=$2
    local img="${HOST_IMG[$host]:-$DEFAULT_IMG}"
    echo "$(ts)   → ${host} part=${idx} (IMG=${img})"
    ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@${host} \
        "IMG=${img} setsid bash ${PIPE}/tools/run_topup_part_recap.sh ${idx} \
            > /tmp/run_topup_recap_part${idx}.out 2>&1 < /dev/null & disown" || \
        echo "$(ts)   ssh fail on ${host}"
}

# Phase 2: round 1 (parts 0-9)
echo "$(ts) Phase 2 round=1: launch parts 0..9"
for i in 0 1 2 3 4 5 6 7 8 9; do
    h=${HOSTS[$i]}
    launch_part $i $h &
done
wait
echo "$(ts) Phase 2 round=1: all 10 launched."

while true; do
    n=0
    for i in 0 1 2 3 4 5 6 7 8 9; do
        [[ -f ${TOPUP_ROOT}/pandas70m_part${i}/DONE.txt ]] && n=$((n+1))
    done
    if (( n == 10 )); then
        echo "$(ts) Phase 2: round=1 all 10 DONE."
        break
    fi
    echo "$(ts)   round=1: ${n}/10 done"
    sleep 300
done

# Phase 3: round 2 (parts 10-19)
echo "$(ts) Phase 3 round=2: launch parts 10..19"
for j in 0 1 2 3 4 5 6 7 8 9; do
    i=$((10 + j))
    h=${HOSTS[$j]}
    launch_part $i $h &
done
wait
echo "$(ts) Phase 3 round=2: all 10 launched."

while true; do
    n=0
    for i in 10 11 12 13 14 15 16 17 18 19; do
        [[ -f ${TOPUP_ROOT}/pandas70m_part${i}/DONE.txt ]] && n=$((n+1))
    done
    if (( n == 10 )); then
        echo "$(ts) Phase 3: round=2 all 10 DONE."
        break
    fi
    echo "$(ts)   round=2: ${n}/10 done"
    sleep 300
done

# Phase 4: validate
echo "$(ts) Phase 4: validate"
total_tar=0; total_idx=0; total_size=0
for i in $(seq 0 19); do
    d=${TOPUP_ROOT}/pandas70m_part${i}
    n_tar=$(ls $d/webdataset/*.tar 2>/dev/null | wc -l)
    n_idx=$(ls $d/webdataset/*.tar.idx 2>/dev/null | wc -l)
    sz=$(du -sb $d 2>/dev/null | awk '{print $1}')
    total_tar=$((total_tar+n_tar))
    total_idx=$((total_idx+n_idx))
    total_size=$((total_size+sz))
    if (( n_tar == n_idx && n_tar > 0 )); then
        echo "$(ts)   part${i}: tars=${n_tar} idx=${n_idx} size=$(du -sh $d | awk '{print $1}')"
    else
        echo "$(ts)   part${i}: !!! FAIL tars=${n_tar} idx=${n_idx}"
    fi
done
echo "$(ts) Phase 4: total tars=${total_tar} idx=${total_idx} size=$((total_size/1024/1024/1024)) GB"

echo "$(ts) === Recap top-up watcher done ==="
