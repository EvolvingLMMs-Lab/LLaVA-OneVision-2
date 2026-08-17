#!/bin/bash
# Mixed packing: robobrain(clean/distance/tracer)+47m_subset+vsi590k(video+image) -> 64k webdataset.
# 10 parts across idle hosts. Avoids .26/.22/.18.
set -uo pipefail
IMAGE="llava_megatron:25.12"
SDIR="/workspace/LLaVA-OneVision-2.5/offline_packing"
INPUT_DIR="/data/video_tracking_pointmix_packed_64k/parts"
OUT_BASE="/data/video_tracking_pointmix_packed_64k"
LOG_DIR="${OUT_BASE}/logs"
MODEL="/data/pack_model_cfg/LLaVA-OneVision-2-4B-p16m33-imgproc"
mkdir -p "${LOG_DIR}"
declare -A HOST_PARTS=(
  ["192.0.2.12"]="00 07" ["192.0.2.16"]="01 08" ["192.0.2.27"]="02 09"
  ["192.0.2.47"]="03" ["192.0.2.48"]="04" ["192.0.2.49"]="05" ["192.0.2.50"]="06" )
cname() { echo "pack_track64k_part$1"; }
launch_one() {
  local ip="$1" p="$2"; local cn; cn="$(cname "$p")"
  local inp="${INPUT_DIR}/track_part${p}.jsonl"
  local out="${OUT_BASE}/part${p}"
  local log="${LOG_DIR}/part${p}_${ip}.log"
  echo "[${ip}] launching ${cn} in=${inp}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "${ip}" "
    docker rm -f '${cn}' >/dev/null 2>&1 || true
    docker run -d --rm --name '${cn}' --network host --ipc=host       -e S2_MAX_PROCESSES=48 -e S2_MAX_THREADS_PER_PROC=4       -v /vlm:/vlm -v /data:/data -v /ov2:/ov2 -v /rice_vl:/rice_vl       -w '${SDIR}' '${IMAGE}'       bash -lc 'bash ${SDIR}/auto_pipe.sh -i ${inp} -r /ov2 -o ${out} -L 64000 -M 200 -w 48 --direct-workers 24 -m ${MODEL} --longest-edge 3072 --jpeg-quality 92 --shard-prefix video_tracking_pointmix 2>&1' >/dev/null
    nohup bash -c \"docker logs -f '${cn}' >> '${log}' 2>&1\" >/dev/null 2>&1 &
  "
  echo "[${ip}]   -> log=${log}"
}
ACTION="${1:-launch}"
case "${ACTION}" in
  launch) for ip in "${!HOST_PARTS[@]}"; do for p in ${HOST_PARTS[$ip]}; do launch_one "${ip}" "${p}"; done; done; echo; echo 'All launched.' ;;
  status)
    for ip in $(echo "${!HOST_PARTS[@]}" | tr ' ' '\n' | sort); do for p in ${HOST_PARTS[$ip]}; do
      cn="$(cname "$p")"
      st=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "${ip}" "docker inspect -f '{{.State.Status}}' '${cn}' 2>/dev/null || echo NONE")
      sd="${OUT_BASE}/part${p}/.status"
      s1=$([[ -f "${sd}/step1.done" ]] && echo 1 || echo -)
      s2=$(ls "${sd}"/direct_step2_*.done >/dev/null 2>&1 && echo 2 || echo -)
      s3=$(ls "${sd}"/direct_step3_*.done >/dev/null 2>&1 && echo 3 || echo -)
      s4=$(ls "${sd}"/direct_step4_*.done >/dev/null 2>&1 && echo 4 || echo -)
      nt=$(ls "${OUT_BASE}/part${p}/webdataset"/*.tar 2>/dev/null | wc -l)
      printf '  %-13s part%s state=%-8s steps[%s%s%s%s] tars=%s\n' "${ip}" "${p}" "${st}" "${s1}" "${s2}" "${s3}" "${s4}" "${nt}"
    done; done ;;
  stop) for ip in "${!HOST_PARTS[@]}"; do for p in ${HOST_PARTS[$ip]}; do ssh -o BatchMode=yes "${ip}" "docker rm -f '$(cname $p)' >/dev/null 2>&1 || true"; echo "[${ip}] stopped part${p}"; done; done ;;
esac
