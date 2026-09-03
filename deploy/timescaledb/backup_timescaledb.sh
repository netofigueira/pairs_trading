#!/usr/bin/env bash
# Create an atomic, portable TimescaleDB backup on the VM and keep 14 daily copies.
set -euo pipefail

readonly backup_dir="${QUANT_BACKUP_DIR:-/var/backups/quant-timescaledb}"
readonly retention_days="${QUANT_BACKUP_RETENTION_DAYS:-14}"
readonly timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
readonly archive="$backup_dir/quantpairs-$timestamp.dump"
readonly partial_archive="$archive.partial"

install -d -m 0700 "$backup_dir"
trap 'rm -f "$partial_archive"' EXIT

docker exec quant-timescaledb pg_dump -U quantpairs -Fc -d quantpairs >"$partial_archive"
docker exec -i quant-timescaledb pg_restore --list <"$partial_archive" >/dev/null
mv "$partial_archive" "$archive"
find "$backup_dir" -type f -name 'quantpairs-*.dump' -mtime "+$retention_days" -delete
printf 'backup=%s bytes=%s\n' "$archive" "$(stat -c '%s' "$archive")"
