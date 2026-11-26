#!/usr/bin/env bash
# What a served engine actually leaves free: the host's MemAvailable sampled while a container
# serves, and each container's own cgroup usage. Run it on the Spark next to a serve and it reports
# the minimum it saw, not an average -- a serve's headroom is its worst moment.
#
#   tools/mem_watch.sh [seconds] [interval]     # default 600 s, 1 s
#
# CSV on stdout (sample,epoch,mem_available_gib,mem_free_gib,cached_gib,slab_gib,swap_free_gib,
# top_container,top_container_gib,all_containers_gib); summary on stderr, so `> mem.csv` keeps the
# data and `2>&1 | tail` shows the verdict.
#
# On this box (GB10) the GPU shares the host's memory: `nvidia-smi memory.used` reports [N/A] and
# there is no separate VRAM to add, so MemAvailable is the whole headroom. On a box with its own
# VRAM, MemAvailable alone overstates it -- add the GPU's used memory to the weights' side.
set -uo pipefail

secs=${1:-600}
ival=${2:-1}
end=$(( $(date +%s) + secs ))

gib_kb() { awk -v b="$1" 'BEGIN { printf "%.2f", b / 1048576 }'; }       # /proc/meminfo: kB
gib_b() { awk -v b="$1" 'BEGIN { printf "%.2f", b / 1073741824 }'; }    # cgroup: bytes

# Container memory: `/sys/fs/cgroup/system.slice/docker-*.scope/memory.current` is root-only here,
# so the numbers come from the daemon (`docker stats`), which costs ~1 s per call -- refreshed every
# fifth sample and reused in between.
declare -A cmem
refresh_containers() {
    while IFS=$'\t' read -r name used _; do
        [ -n "$name" ] || continue
        cmem[$name]=$(awk -v u="$used" 'BEGIN {
            split(u, part, " ")
            v = part[1]
            unit = v
            gsub(/[0-9.]+/, "", unit)
            gsub(/[A-Za-z]+/, "", v)
            if (unit == "KiB") v /= 1048576; else if (unit == "MiB") v /= 1024; else if (unit == "TiB") v *= 1024
            printf "%.2f", v
        }')
    done < <(docker stats --no-stream --format '{{.Name}}	{{.MemUsage}}' 2>/dev/null)
}

container_gib() {   # $1 = name -> "name gib"
    echo "${1:-?} ${cmem[${1:-}]:-0}"
}

{
    echo "# containers at start:"
    docker ps --format '{{.Names}} {{.Image}}' 2>/dev/null | sed 's/^/#   /'
    echo "# serve profile (first .env found):"
    grep -hE '^(GLM_(MODEL_DIR|MAX_SEQ|KV|SLOTS|FLASH_DENSE)|GLM_EXL3_PACK)=' "$HOME"/tidelm/.env.* 2>/dev/null | head -8 | sed 's/^/#   /'
    echo "sample,epoch,mem_available_gib,mem_free_gib,cached_gib,slab_gib,swap_free_gib,top_container,top_container_gib,all_containers_gib"
} >&2

min_avail=""; sum_avail=0; n=0; i=0; peak_top=0; peak_name="-"
while [ "$(date +%s)" -lt "$end" ]; do
    read -r avail free cached slab swapfree < <(awk '
        /^MemAvailable:/ { a = $2 } /^MemFree:/ { f = $2 } /^Cached:/ { c = $2 }
        /^SReclaimable:/ { s = $2 } /^SwapFree:/ { w = $2 }
        END { print a + 0, f + 0, c + 0, s + 0, w + 0 }' /proc/meminfo)

    top_name="-"; top=0; tot=0
    if [ $(( i % 5 )) -eq 0 ]; then
        refresh_containers
        mapfile -t names < <(docker ps --format '{{.Names}}' 2>/dev/null)
    fi
    for nm in "${names[@]:-}"; do
        read -r nm g < <(container_gib "$nm")
        tot=$(awk -v a="$tot" -v b="$g" 'BEGIN { printf "%.2f", a + b }')
        if awk -v a="$g" -v b="$top" 'BEGIN { exit !(a > b) }'; then top=$g; top_name=$nm; fi
    done
    if awk -v a="$top" -v b="$peak_top" 'BEGIN { exit !(a > b) }'; then peak_top=$top; peak_name=$top_name; fi

    a=$(gib_kb "$avail")
    echo "$i,$(date +%s),$a,$(gib_kb "$free"),$(gib_kb "$cached"),$(gib_kb "$slab"),$(gib_kb "$swapfree"),$top_name,$top,$tot"

    min_avail=$(awk -v x="$a" -v m="${min_avail:-99999}" 'BEGIN { print (x < m) ? x : m }')
    sum_avail=$(awk -v s="$sum_avail" -v x="$a" 'BEGIN { printf "%.4f", s + x }')
    n=$((n + 1)); i=$((i + 1))
    sleep "$ival"
done

awk -v n="$n" -v sum="$sum_avail" -v min="$min_avail" -v pn="$peak_name" -v pg="$peak_top" 'BEGIN {
    printf "samples %d  MemAvailable min %.2f GiB  mean %.2f GiB   biggest container %s %.2f GiB\n",
           n, min, sum / n, pn, pg
}' >&2
