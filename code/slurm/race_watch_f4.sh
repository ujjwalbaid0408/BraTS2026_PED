#!/usr/bin/env bash
# Watch the two fold-4 extend jobs (rp6b vs B200). The instant one starts
# RUNNING, cancel the other so they never write the same results dir
# concurrently. Exits once a winner is decided (or both are gone).
set -uo pipefail

RP6B_JID="${1:?need rp6b jobid}"
B200_JID="${2:?need b200 jobid}"
LOG="/scratch/ubaid/BraTS2026_PED/slurm_logs/race_f4.log"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
state() { squeue -h -j "$1" -o "%t" 2>/dev/null | tr -d ' '; }

echo "[$(stamp)] race start: rp6b=${RP6B_JID} b200=${B200_JID}" >>"$LOG"

while true; do
    s_rp=$(state "$RP6B_JID")
    s_b2=$(state "$B200_JID")

    # Both gone from queue → nothing to arbitrate.
    if [[ -z "$s_rp" && -z "$s_b2" ]]; then
        echo "[$(stamp)] both jobs left the queue; no action, exiting" >>"$LOG"
        exit 0
    fi

    if [[ "$s_rp" == "R" ]]; then
        echo "[$(stamp)] WINNER rp6b ${RP6B_JID} (R). Cancelling B200 ${B200_JID}" >>"$LOG"
        scancel "$B200_JID" 2>>"$LOG"
        exit 0
    fi
    if [[ "$s_b2" == "R" ]]; then
        echo "[$(stamp)] WINNER B200 ${B200_JID} (R). Cancelling rp6b ${RP6B_JID}" >>"$LOG"
        scancel "$RP6B_JID" 2>>"$LOG"
        exit 0
    fi

    sleep 10
done
