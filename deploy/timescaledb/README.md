# TimescaleDB na VM

Este Compose sobe o banco privado da plataforma quant. A porta `5433` é publicada apenas em `127.0.0.1` da VM; não crie Funnel, regra de firewall ou proxy público para ela.

## Primeira subida

Na VM, copie `compose.yaml`, crie um arquivo `.env` com permissão `600` e uma senha longa:

```bash
cd /home/opc/quant-timescaledb
umask 077
openssl rand -base64 36 | tr -d '\n' | sed 's/^/POSTGRES_PASSWORD=/' > .env
docker compose up -d
```

As migrations do projeto devem ser aplicadas em ordem a partir de `db/migrations/`. A tabela `public.schema_migration` registra o estado.

## Operação

```bash
docker compose ps
docker compose logs -f timescaledb
```

Faça backup antes de upgrades de versão e teste a restauração em outro volume. Não use tags mutáveis de imagem; o Compose fixa a combinação TimescaleDB/PostgreSQL.

## Backup diário na VM

`quant-timescaledb-backup.service` gera um dump no formato PostgreSQL custom,
valida o arquivo com `pg_restore --list` e mantém os últimos 14 dias em
`/home/opc/backups/quant-timescaledb`. Instale o service e o timer em
`/etc/systemd/system`, então ative-o com:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now quant-timescaledb-backup.timer
systemctl list-timers quant-timescaledb-backup.timer
```

Esse backup local protege contra erro operacional; antes de depender da VM para
pesquisa contínua, replique os dumps para um storage externo.
