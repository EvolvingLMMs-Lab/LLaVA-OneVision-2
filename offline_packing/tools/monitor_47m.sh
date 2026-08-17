#!/bin/bash
# Quick status snapshot for the 7-host 47m packing run.
declare -A MAP=( [192.0.2.54]=1 [192.0.2.52]=2 [192.0.2.51]=3 [192.0.2.57]=4 [192.0.2.56]=5 [192.0.2.58]=6 [192.0.2.55]=7 )
for ip in "${!MAP[@]}"; do
  idx=${MAP[$ip]}
  printf "=== part%d @ %s ===\n" "$idx" "$ip"
  ssh -o ConnectTimeout=5 "$ip" "
    if [[ -f /data/47m_sft_packed/47m_part${idx}/DONE.txt ]]; then echo '  ✅ DONE'; fi
    echo -n '  CPU/MEM: '; uptime | awk -F'load average:' '{print \$2}'
    echo -n '  last log: '; tail -1 /data/47m_sft_packed/47m_part${idx}/run.log 2>/dev/null | head -c 180; echo
    ls /data/47m_sft_packed/47m_part${idx}/webdataset/*.tar 2>/dev/null | wc -l | awk '{print \"  tar shards: \" \$1}'
  " 2>&1 | grep -v Warning
done
