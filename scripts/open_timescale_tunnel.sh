#!/usr/bin/env bash
set -euo pipefail

# PostgreSQL remains private on the VM. This exposes it only on localhost here.
ssh -f -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:5433:127.0.0.1:5433 \
  opc@163.176.128.107
