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
