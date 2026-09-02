# Fase 1 — short straddle delta-hedgeado diariamente (envelope)

Data: 2026-09-02
Estado: experimento executado; envelope sobre marcação sintética, não fills
observados; nada aprovado para dinheiro real.
Plano: [docs/2026-09-02-plano-pipeline-vol.md](2026-09-02-plano-pipeline-vol.md)

## Pergunta

O prêmio de variância sobrevive quando removemos o risco direcional? O bootstrap
mostrou média negativa por trade no straddle não-hedgeado; a hipótese da Fase 1
é que grande parte dessa perda (e da cauda) é delta acumulado, não vol.

## Desenho

Mesmas 22 entradas trimestrais do envelope (pós-DVOL, book mínimo de 0,1
contrato), short at bid sintético, hold até o vencimento e settlement oficial,
idênticos ao gate. A novidade é a perna de hedge:

- **Marcação diária sintética** existente (`build_daily_straddle_marks`):
  Black-76 inverso, IV ancorada na variação do DVOL, basis constante da entrada.
- **Delta em BTC** por diferença central sobre o pricer exato, com o forward
  escalando proporcionalmente ao spot (`straddle_delta_btc`).
- **Hedge no perp inverso**: notional `H = contratos · delta · S²`, arredondado
  a contratos de $10, rebalanceado a cada marcação diária. Taker fee de 0,05%
  sobre o notional negociado; **funding horário real** da Deribit (baixado da
  API pública e cacheado em `data/market/deribit/funding/`).
- Hedge fechado no delivery price junto com o settlement.

```bash
.venv/bin/python scripts/run_delta_hedged_carry.py
```

Artefato: `artifacts/delta-hedged-carry-v1.json` (por trade, com decomposição
diária). Implementação: `src/quant_pairs/delta_hedged_carry.py`.

## Resultado (22 trades, 0,1 contrato por perna)

| Métrica | Não-hedgeado | Hedgeado |
|---|---:|---:|
| P&L total (BTC) | +0,01747 | +0,02421 |
| P&L médio por trade | +0,00079 | +0,00110 |
| Desvio-padrão por trade | 0,00495 | 0,00272 |
| t-stat da média | 0,75 | 1,90 |
| Trades positivos | 14/22 | 17/22 |
| Pior trade | -0,00763 | -0,00710 |
| Std média do P&L diário | 0,00125 | 0,00050 |

Decomposição agregada do book hedgeado: opção +0,01747, trading do hedge
+0,01190, funding -0,00215, fees do hedge -0,00302.

## Leitura

1. **A hipótese da Fase 1 se confirma na direção esperada.** O hedge corta o
   desvio-padrão por trade quase pela metade e o do P&L diário em 2,5×, sem
   destruir a média: fees (+funding) do hedge custaram 0,00517 BTC no agregado,
   menos de 30% do P&L da opção, e a média por trade subiu.
2. **O ganho de trading do hedge (+0,01190) é sorte de amostra, não estrutura.**
   Um hedge delta-neutro tem esperança de trading ~0; o sinal honesto é a
   redução de variância, e a média hedgeada só é comparável porque fees+funding
   não comem o prêmio.
3. **t-stat 1,90 ainda não é significância** (N=22, e as observações usam a
   mesma marcação sintética). Mas compare com 0,75 do não-hedgeado: a mesma
   amostra, com a expressão certa, fica perto de resolver a pergunta que o
   formato não-hedgeado nunca resolveria.
4. **Reconciliação com o bootstrap.** O bootstrap reprova o straddle
   não-hedgeado sob frequência histórica; este resultado não o contradiz,
   porque muda a expressão. O próximo bootstrap (Fase 3) deve reamostrar o book
   hedgeado.

## Limitações declaradas

- Marcas `synthetic_model`: trajetória de IV presa à variação do DVOL e basis
  constante; nada aqui é fill observado de opção.
- Hedge executa no close diário com taker fee; sem slippage intradiária nem
  gaps entre rebalanceamentos (o risco de gap continua real e é assunto da
  margem na Fase 3).
- Sem margem, liquidação ou efeito de portfólio.
- O funding usado é o histórico realizado; posições grandes moveriam o funding.

## Próximo passo (Fase 2 do plano)

Custo e execução: cruzar spread vs postar no mid vs strangle 25-delta, sobre a
mesma população, antes de qualquer aumento de frequência.
