#!/usr/bin/env bash
set -euo pipefail

# GPU metrics collection via nvidia-smi polling.
# Usage: collect_metrics.sh start | stop
#
# Writes CSV to /tmp/gpu_metrics.csv in the format:
#   gpu_index, utilization%, memory_used_mib, memory_total_mib, power_draw_w, temperature_c

METRICS_FILE="/tmp/gpu_metrics.csv"
PID_FILE="/tmp/nvidia_smi.pid"

case "${1:-}" in
    start)
        # Check nvidia-smi is available
        if ! command -v nvidia-smi &>/dev/null; then
            echo "WARNING: nvidia-smi not found, GPU metrics will not be collected." >&2
            exit 0
        fi

        # Start polling at 1-second intervals
        nvidia-smi \
            --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits \
            -l 1 > "$METRICS_FILE" 2>/dev/null &

        echo $! > "$PID_FILE"
        echo "nvidia-smi polling started (PID: $(cat $PID_FILE))" >&2
        ;;

    stop)
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID" 2>/dev/null || true
                wait "$PID" 2>/dev/null || true
                echo "nvidia-smi polling stopped." >&2
            fi
            rm -f "$PID_FILE"
        fi

        # Print summary
        if [ -f "$METRICS_FILE" ]; then
            LINES=$(wc -l < "$METRICS_FILE" | tr -d ' ')
            echo "Collected $LINES GPU metric samples." >&2
        fi
        ;;

    *)
        echo "Usage: collect_metrics.sh start|stop" >&2
        exit 1
        ;;
esac
