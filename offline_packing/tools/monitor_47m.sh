#!/bin/bash
# Quick status snapshot for the 7-host 47m packing run.
declare -A MAP=( [172.16.5.114]=1 [172.16.5.111]=2 [172.16.5.109]=3 [172.16.5.162]=4 [172.16.5.161]=5 [172.16.5.163]=6 [172.16.5.154]=7 )
for ip in "${!MAP[@]}"; do
  idx=${MAP[$ip]}
  printf "=== part%d @ %s ===\n" "$idx" "$ip"
  ssh -o ConnectTimeout=5 "$ip" "
    if [[ -f /ov21/47m_sft_packed/47m_part${idx}/DONE.txt ]]; then echo '  ✅ DONE'; fi
    echo -n '  CPU/MEM: '; uptime | awk -F'load average:' '{print \$2}'
    echo -n '  last log: '; tail -1 /ov21/47m_sft_packed/47m_part${idx}/run.log 2>/dev/null | head -c 180; echo
    ls /ov21/47m_sft_packed/47m_part${idx}/webdataset/*.tar 2>/dev/null | wc -l | awk '{print \"  tar shards: \" \$1}'
  " 2>&1 | grep -v Warning
done
