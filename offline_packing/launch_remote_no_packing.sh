#!/bin/bash
# ============================================================================
# Launch auto_pipe_no_packing.sh in parallel on 9 remote hosts.
#
# Each remote host runs ONE part inside a llava_megatron:25.12 docker
# container. Logs go to /ov21/streaming/180s_wds/logs/<part>_<host>.log.
#
# Allocation (locked in by user):
#   5.18 (this host) -> part0 (already running natively, do not touch)
#   5.32 -> part1   5.38 -> part2   5.40 -> part3
#   5.41 -> part4   5.43 -> part5   5.44 -> part6
#   5.45 -> part7   5.46 -> part8   5.47 -> part9
# ============================================================================

set -euo pipefail

LOG_DIR="/ov21/streaming/180s_wds/logs"
mkdir -p "${LOG_DIR}"

IMAGE="llava_megatron:25.12"
SCRIPT_DIR_HOST="/vlm/yinxie/code/OV2/OV2_public_main/offline_packing"
PARTS_BASENAME_PREFIX="all_180s_96w_done_part"

# host -> part_index
declare -A HOST_TO_PART=(
    ["172.16.5.32"]="01"
    ["172.16.5.38"]="02"
    ["172.16.5.40"]="03"
    ["172.16.5.41"]="04"
    ["172.16.5.43"]="05"
    ["172.16.5.44"]="06"
    ["172.16.5.45"]="07"
    ["172.16.5.46"]="08"
    ["172.16.5.47"]="09"
)

ACTION="${1:-launch}"   # launch | status | tail | stop | logs

container_name() {
    local part_idx="$1"
    echo "no_packing_pipeline_part${part_idx}"
}

launch_one() {
    local ip="$1"
    local part_idx="$2"
    local part_stem="${PARTS_BASENAME_PREFIX}${part_idx}"
    local cname
    cname="$(container_name "${part_idx}")"
    local log_file="${LOG_DIR}/${part_stem}_${ip}.log"

    echo "[${ip}] Launching ${cname} for ${part_stem} ..."
    # Important: --rm so the container disappears once finished.
    # Use bash -lc so the script's #!/bin/bash and PATH work as expected.
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${ip}" "
        # Best-effort cleanup if a stale container exists
        docker rm -f '${cname}' >/dev/null 2>&1 || true

        # Run detached. Mount /vlm and /ov21 (both NFS, shared across hosts).
        docker run -d --rm \
            --name '${cname}' \
            --network host \
            --ipc=host \
            -v /vlm:/vlm \
            -v /ov21:/ov21 \
            -w '${SCRIPT_DIR_HOST}' \
            '${IMAGE}' \
            bash -lc 'bash ${SCRIPT_DIR_HOST}/auto_pipe_no_packing.sh --parts ${part_stem}' \
            >/dev/null

        # Stream logs to the shared log file (background)
        nohup bash -c \"docker logs -f '${cname}' >> '${log_file}' 2>&1\" >/dev/null 2>&1 &
    "
    echo "[${ip}] launched. log=${log_file}"
}

cmd_launch() {
    echo "Launching on 9 remote hosts..."
    for ip in "${!HOST_TO_PART[@]}"; do
        launch_one "${ip}" "${HOST_TO_PART[$ip]}"
    done
    echo ""
    echo "All launched. Use:  bash $0 status   to monitor."
}

cmd_status() {
    echo "=================================================================="
    echo "  Per-host docker container status"
    echo "=================================================================="
    for ip in $(echo "${!HOST_TO_PART[@]}" | tr ' ' '\n' | sort); do
        local part_idx="${HOST_TO_PART[$ip]}"
        local cname
        cname="$(container_name "${part_idx}")"
        local part_stem="${PARTS_BASENAME_PREFIX}${part_idx}"
        local stat
        stat=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${ip}" \
            "docker inspect --format '{{.State.Status}} ({{.State.StartedAt}})' '${cname}' 2>/dev/null || echo 'NOT-FOUND'")
        printf "  %-13s part=%s container=%-30s state=%s\n" "${ip}" "${part_idx}" "${cname}" "${stat}"
        # Show step done flags from shared filesystem
        local sd="/ov21/streaming/180s_wds/${part_stem}/.status"
        local s1=$([[ -f "${sd}/step1.done" ]] && echo "✓" || echo "·")
        local s2=$([[ -f "${sd}/step2.done" ]] && echo "✓" || echo "·")
        local s3=$([[ -f "${sd}/step3.done" ]] && echo "✓" || echo "·")
        local n_tar
        n_tar=$(ls /ov21/streaming/180s_wds/"${part_stem}"/webdataset/*.tar 2>/dev/null | wc -l)
        local size
        size=$(du -sh /ov21/streaming/180s_wds/"${part_stem}"/ 2>/dev/null | cut -f1)
        printf "                steps: s1=%s s2=%s s3=%s tars=%-5d size=%s\n" "${s1}" "${s2}" "${s3}" "${n_tar}" "${size}"
    done
    echo ""
    echo "  5.18 (local)  part=00 (running natively, see /ov21/streaming/180s_wds/run.log)"
    local sd="/ov21/streaming/180s_wds/${PARTS_BASENAME_PREFIX}00/.status"
    local s1=$([[ -f "${sd}/step1.done" ]] && echo "✓" || echo "·")
    local s2=$([[ -f "${sd}/step2.done" ]] && echo "✓" || echo "·")
    local s3=$([[ -f "${sd}/step3.done" ]] && echo "✓" || echo "·")
    local n_tar
    n_tar=$(ls /ov21/streaming/180s_wds/"${PARTS_BASENAME_PREFIX}00"/webdataset/*.tar 2>/dev/null | wc -l)
    local size
    size=$(du -sh /ov21/streaming/180s_wds/"${PARTS_BASENAME_PREFIX}00"/ 2>/dev/null | cut -f1)
    printf "                steps: s1=%s s2=%s s3=%s tars=%-5d size=%s\n" "${s1}" "${s2}" "${s3}" "${n_tar}" "${size}"
}

cmd_tail() {
    # Tail the last lines of every shared log file.
    for ip in $(echo "${!HOST_TO_PART[@]}" | tr ' ' '\n' | sort); do
        local part_idx="${HOST_TO_PART[$ip]}"
        local part_stem="${PARTS_BASENAME_PREFIX}${part_idx}"
        local log_file="${LOG_DIR}/${part_stem}_${ip}.log"
        echo "===== [${ip}] ${part_stem} (${log_file}) ====="
        tail -n 5 "${log_file}" 2>/dev/null || echo "  (no log yet)"
        echo ""
    done
}

cmd_stop() {
    echo "Stopping all remote containers..."
    for ip in "${!HOST_TO_PART[@]}"; do
        local part_idx="${HOST_TO_PART[$ip]}"
        local cname
        cname="$(container_name "${part_idx}")"
        ssh -o BatchMode=yes "${ip}" "docker rm -f '${cname}' >/dev/null 2>&1 && echo '[${ip}] stopped' || echo '[${ip}] no container'"
    done
}

cmd_logs() {
    # List shared log files
    ls -la "${LOG_DIR}/"
}

case "${ACTION}" in
    launch)  cmd_launch ;;
    status)  cmd_status ;;
    tail)    cmd_tail ;;
    stop)    cmd_stop ;;
    logs)    cmd_logs ;;
    *) echo "Usage: $0 {launch|status|tail|stop|logs}"; exit 1 ;;
esac
