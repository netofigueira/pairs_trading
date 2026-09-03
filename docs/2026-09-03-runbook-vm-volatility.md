# Runbook: VM de volatilidade

## Acesso privado

A VM `free-tier-a1` só expõe o n8n pela tailnet. Instale o cliente Tailscale
na máquina local e entre na mesma conta; depois use:

```bash
ssh opc@free-tier-a1.tail7f5470.ts.net
./scripts/open_timescale_tunnel.sh
```

O editor n8n fica em `https://free-tier-a1.tail7f5470.ts.net/`. Não há Funnel
público. O túnel expõe o banco somente em `127.0.0.1:5433` localmente. Em uma
emergência sem Tailscale, o host pode ser substituído explicitamente:

```bash
QUANT_VM_HOST=163.176.128.107 ./scripts/open_timescale_tunnel.sh
```

## Serviços que devem estar ativos

```bash
docker ps
systemctl status quant-timescaledb-backup.timer
tailscale serve status
```

Serviços esperados: `quant-timescaledb`, `quant-collector`,
`quant-dashboard` e `general-n8n`. O workflow `Quant | Hourly option tape
collection` está publicado no n8n e coleta dados públicos da Deribit de hora
em hora.

## Backup e manutenção

O timer `quant-timescaledb-backup.timer` roda diariamente às 03:15 UTC, valida
o dump PostgreSQL e retém 14 cópias em `/home/opc/backups/quant-timescaledb`.
Ele é proteção local: antes de depender de pesquisa contínua, configure cópia
para storage externo.

Para deploy: atualize `/home/opc/quant-pairs`, reconstrua collector/dashboard
com seus Composes e aplique `scripts/migrate_database.py` ou a migration nova
correspondente. Não reescreva migrations já registradas em
`public.schema_migration`.

`hermes-gateway.service` está preservado, mas desativado. Profit Anatomy foi
arquivado em `/home/opc/archives/` e removido da operação da VM.
