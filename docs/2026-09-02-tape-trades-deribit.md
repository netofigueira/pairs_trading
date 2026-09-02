# Tape histórico público de opções Deribit — coletor V1

Data: 2026-09-02  
Estado: coletor implementado e validado em um probe de 2019-07-01; backfill
completo e seleção diária ainda pendentes.

## Fonte e contrato

O host `https://history.deribit.com/api/v2` expõe o tape histórico público no
endpoint `public/get_last_trades_by_currency_and_time`. Um probe de dez minutos
ao redor de 12:00 UTC em 2019-07-01 retornou 11 trades de opções BTC, com
instrumento, timestamp, preço, IV, mark, index price, tamanho e direção.

O endpoint público normal da Deribit guarda apenas 24 horas; este host histórico
é uma fonte separada. A documentação atual do endpoint principal descreve os
campos de trade e o limite de 1.000 registros por resposta, mas não deve ser
confundida com a cobertura temporal do host histórico. [Referência da API
principal](https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency_and_time).

## Coleta congelada

`config/experiment.deribit-history-tape-v1.json` declara BTC, cada dia de
2021-04-01 a 2026-08-18, janela 10:00--14:00 UTC e somente `kind=option`.

```bash
.venv/bin/python scripts/collect_deribit_history_trades.py \
  --start 2021-04-01 --end 2026-08-18 --window-minutes 120
```

O cache diário vai para
`data/market/deribit/history-trades/BTC/option/YYYY-MM-DD/`. Ele é ignorado
pelo Git, idempotente por arquivo e pode ser retomado. Se `has_more=true`, o
coletor divide a janela temporal recursivamente; não avança um cursor de tempo,
pois isso poderia perder negócios distintos no mesmo milissegundo.

## O que o tape permitirá

Ele substitui a entrada circular DVOL do book rolling por prints de opções
realmente executados. A seleção posterior escolherá um call e um put ATM com
vencimento próximo de 14 dias; o preço central será o print observado e os
cenários de venda aplicarão o half-spread calibrado nos 26 books Tardis.

Isso ainda não equivale a um fill próprio: não sabemos a fila, a profundidade
nem se nosso tamanho teria executado. Dias sem ambos os lados do straddle na
janela são falhas de cobertura declaradas, nunca interpolação.
