#!/bin/bash
# Must be run as root

# ----- CPU: All clusters to performance + lock min = max -----
for policy in /sys/devices/system/cpu/cpufreq/policy*; do
    echo performance > "$policy/scaling_governor" 2>/dev/null
    cat "$policy/scaling_max_freq" > "$policy/scaling_min_freq" 2>/dev/null
done

# ----- NPU, GPU, DMC: performance + lock min = max -----
for dev in /sys/class/devfreq/*; do
    case "$dev" in
        *.gpu|*.npu|*.dmc|dmc)   # dmc (without dot) also covered
            echo performance > "$dev/governor" 2>/dev/null
            if [ -f "$dev/max_freq" ]; then
                cat "$dev/max_freq" > "$dev/min_freq" 2>/dev/null
            fi
            ;;
    esac
done