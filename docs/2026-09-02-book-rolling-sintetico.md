# Book rolling diário de short-vol — envelope sintético V1

Data: 2026-09-02  
Estado: mecânica de portfólio executada; **não é backtest de fills nem validação
de alpha**. Não altera o holdout vivo selado.

## Por que esta rodada existe

O pipeline anterior tinha marcação diária, mas somente 22 entradas trimestrais;
a mudança de expressão não criava novas decisões independentes. Este experimento
separa a mecânica necessária para uma política rolling (entradas, posições
sobrepostas, hedge líquido e margem do book) da evidência histórica executável,
que continua esparsa porque só há 26 dias de books Tardis locais.

## Especificação congelada

Configuração: `config/experiment.rolling-volatility-book-v1.json`, commitada
antes da geração do artefato.

- Uma decisão por dia com forecast GARCH corrigido causalmente de 14 dias;
- candidata short quando `forecast² < bid-IV sintética²`;
- straddle ATM inverso, 14 DTE, 0,1 contrato por entrada;
- IV de entrada = DVOL; bid sintética = DVOL menos 1 ponto de IV;
- marcação diária pelo spot e pela variação de DVOL, hedge delta líquido no
  perp e funding fixo na média da Fase 3;
- capacidade dinâmica máxima de 0,5 contrato por BTC de equity; a entrada é
  recusada se ultrapassar o teto do book;
- formação de 2022-05-01 a 2026-08-18, com apenas entradas que liquidam antes
  do fim da janela. O holdout que inicia em 2026-09-03 não é lido nem alterado.

```bash
.venv/bin/python scripts/run_rolling_volatility_book.py
```

## Resultado de mecânica

| Métrica | Resultado |
|---|---:|
| Decisões diárias elegíveis | 1.557 |
| Sinais short sintéticos | 862 |
| Entradas aceitas | 483 |
| Recusadas pelo teto do book | 379 |
| Máximo de posições simultâneas | 8 |
| Máximo de contratos brutos | 0,8 |
| Maior utilização de margem aproximada | 28,2% |
| Dias de violação de margem aproximada | 0 |
| Equity terminal sintética (inicial 1 BTC) | 1,684 BTC |

O máximo bruto de 0,8 contrato não viola a regra: o teto é uma razão dinâmica
de 0,5 contrato por BTC de equity, não 0,5 contrato absoluto. O resultado
financeiro (+0,684 BTC) **não é evidência de retorno esperado**: ele usa a
mesma DVOL tanto como proxy de IV de entrada quanto para a trajetória de IV.

Artefato: `artifacts/rolling-volatility-book-v1.json`.

## O que passa a ser possível

O motor já aceita decisões diárias e mantém o risco agregado corretamente.
Quando houver books densos, a próxima troca é somente da camada de entrada:
substituir a bid-IV sintética por bid/ask, tamanho e fills Tardis/Deribit reais.
Então o mesmo motor poderá medir fill rate, adverse selection, hedge e margem
do book histórico executável.

## Limites que permanecem

- Não há histórico diário de **book** de opções no repositório; os arquivos
  locais Tardis cobrem apenas 26 dias trimestrais. O tape público histórico já
  tem um coletor separado, mas ainda precisa de backfill e seleção de straddle.
  Logo esta rodada não prova execução nem skill do forecast.
- DVOL representa 30 dias, não a IV bid de um straddle de 14 DTE.
- O spot é proxy de forward; funding é constante; margem é aproximação padrão
  e uma violação é reportada, não liquidada pela regra da exchange.
- As entradas se sobrepõem, portanto P&Ls diários não devem ser tratados como
  amostra independente para inferência estatística.
