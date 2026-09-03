#!/usr/bin/env bash
set -euo pipefail

# The default is the private Tailscale DNS name.  Override only for a
# break-glass connection when the local machine is not on the tailnet.
readonly quant_vm_host="${QUANT_VM_HOST:-free-tier-a1.tail7f5470.ts.net}"

# PostgreSQL remains private on the VM. This exposes it only on localhost here.
ssh -f -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:5433:127.0.0.1:5433 \
  "opc@$quant_vm_host"
