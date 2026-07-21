#!/bin/bash
set -u

# Summarize live Slurm CPU and memory efficiency without relying on seff.
USER_NAME="${USER_NAME:-${USER:-kyang}}"

command -v squeue >/dev/null 2>&1 || {
  echo "squeue is unavailable; run this script on an OzSTAR login node." >&2
  exit 1
}

printf "%-10s %-34s %6s %9s %9s %8s %9s %9s %8s\n" \
  "JOBID" "NAME" "CPUS" "ELAPSED" "AVG_CPU" "CPU_EFF" "MAX_GB" "REQ_GB" "MEM_EFF"

squeue -u "$USER_NAME" -h -t R -o "%i|%j" |
while IFS='|' read -r job_id job_name; do
  usage=$(sacct -j "${job_id}.batch" -n -P \
    --format=JobIDRaw,Elapsed,ElapsedRaw,TotalCPU,AllocCPUS,MaxRSS 2>/dev/null |
    awk -F'|' '$1 ~ /\.batch$/ {print $2 "|" $3 "|" $4 "|" $5 "|" $6; exit}')

  if [[ -z "$usage" ]]; then
    printf "%-10s %-34s %6s %9s %9s %8s %9s %9s %8s\n" \
      "$job_id" "$job_name" "?" "?" "?" "?" "?" "?" "?"
    continue
  fi

  IFS='|' read -r elapsed elapsed_raw total_cpu alloc_cpus max_rss <<< "$usage"

  req_mem=$(scontrol show job -o "$job_id" 2>/dev/null |
    sed -n 's/.*ReqTRES=[^ ]*mem=\([^, ]*\).*/\1/p' | head -1)

  metrics=$(awk -v total_cpu="$total_cpu" \
    -v elapsed_raw="$elapsed_raw" \
    -v alloc_cpus="$alloc_cpus" \
    -v max_rss="$max_rss" \
    -v req_mem="$req_mem" '
    function duration_seconds(value, parts, clock, n, days) {
      sub(/\..*$/, "", value)
      days = 0
      n = split(value, parts, "-")
      if (n == 2) {
        days = parts[1] + 0
        value = parts[2]
      }
      split(value, clock, ":")
      return days * 86400 + clock[1] * 3600 + clock[2] * 60 + clock[3]
    }
    function gib(value, unit, number) {
      if (value == "" || value == "Unknown") return -1
      unit = substr(value, length(value), 1)
      number = substr(value, 1, length(value) - 1) + 0
      if (unit == "K" || unit == "k") return number / 1024 / 1024
      if (unit == "M" || unit == "m") return number / 1024
      if (unit == "G" || unit == "g") return number
      if (unit == "T" || unit == "t") return number * 1024
      return (value + 0) / 1024 / 1024 / 1024
    }
    BEGIN {
      cpu_seconds = duration_seconds(total_cpu)
      average_cpu = elapsed_raw > 0 ? cpu_seconds / elapsed_raw : 0
      cpu_efficiency = elapsed_raw > 0 && alloc_cpus > 0 \
        ? 100 * average_cpu / alloc_cpus : 0
      max_gb = gib(max_rss)
      req_gb = gib(req_mem)
      memory_efficiency = max_gb >= 0 && req_gb > 0 ? 100 * max_gb / req_gb : -1

      printf "%.2f|%.1f|", average_cpu, cpu_efficiency
      if (max_gb >= 0) printf "%.1f|", max_gb
      else printf "?|"
      if (req_gb >= 0) printf "%.1f|", req_gb
      else printf "?|"
      if (memory_efficiency >= 0) printf "%.1f", memory_efficiency
      else printf "?"
    }
  ')

  IFS='|' read -r average_cpu cpu_eff max_gb req_gb mem_eff <<< "$metrics"
  printf "%-10s %-34s %6s %9s %9s %7s%% %9s %9s %7s%%\n" \
    "$job_id" "$job_name" "${alloc_cpus:-?}" "$elapsed" "$average_cpu" \
    "$cpu_eff" "$max_gb" "$req_gb" "$mem_eff"
done
