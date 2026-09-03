# Integração n8n ↔ TimescaleDB

O n8n e o banco comunicam-se somente pela rede Docker privada `quant_internal`. A porta do banco não é publicada para a internet.

O arquivo `quant-network.override.yaml` documenta a extensão necessária para o Compose do n8n. No workflow, use o hostname Docker `timescaledb`, porta `5432`, e uma credencial PostgreSQL dedicada de escrita limitada ao schema `market`.

## Padrão de coleta

Cada workflow deve:

1. consultar no banco o último `open_time` persistido para cada símbolo e intervalo;
2. buscar da fonte o cursor menos uma pequena sobreposição (duas velas);
3. inserir com a chave natural da série temporal e `ON CONFLICT DO UPDATE`;
4. registrar falhas, atraso de coleta e número de registros;
5. permanecer desativado até passar por uma execução manual e reconciliação.

O banco é a fonte de verdade para o cursor. Não use estado de execução do n8n como checkpoint exclusivo: ele pode expirar, ser apagado ou ser reexecutado.

## Variáveis de runtime

O workflow de coleta usa `QUANT_COLLECTOR_TOKEN` injetado no container n8n e,
por isso, o Compose deve definir `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`. O token
continua fora do workflow versionado e só trafega na rede Docker privada.

O editor deve ser servido apenas pela tailnet (Tailscale Serve), nunca por
Funnel público. Para importar a agenda de tape, use
`n8n import:workflow --input=/caminho/03-quant-hourly-tape-collection.json`;
o arquivo já vem marcado como ativo.
