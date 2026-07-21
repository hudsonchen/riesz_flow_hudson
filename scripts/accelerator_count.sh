#!/usr/bin/env bash

# Print the number of local accelerator devices. Prefer CUDA when it is
# available, then fall back to Intel XPU. Always return at least one process so
# launchers using `set -euo pipefail` can also run on XPU-only or CPU hosts.
accelerator_count() {
    local count=0

    if command -v nvidia-smi >/dev/null 2>&1; then
        count=$(nvidia-smi -L 2>/dev/null | wc -l) || count=0
    fi

    if [ "$count" -lt 1 ]; then
        count=$(python -c 'import torch; xpu = getattr(torch, "xpu", None); print(xpu.device_count() if xpu is not None and xpu.is_available() else 0)' 2>/dev/null) || count=0
    fi

    if ! [[ "$count" =~ ^[0-9]+$ ]] || [ "$count" -lt 1 ]; then
        count=1
    fi

    printf '%s\n' "$count"
}
