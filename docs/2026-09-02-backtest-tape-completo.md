# Backtest diário no tape real — 2021-2026 completo

Data: 2026-09-02
Estado: executado sobre o tape integral carregado no banco (3,67M trades,
2021-04-01 a 2026-08-18, 1.966 dias); nada aprovado para dinheiro real.
Antecedente: [backtest 2025](2026-09-02-backtest-tape-2025.md).

```bash
source deploy/timescaledb/quant_ingest.env   # túnel VM
.venv/bin/python scripts/run_tape_backtest.py \
  --start 2021-04-01 --end 2026-08-18 \
  --output artifacts/tape-backtest-full-v1.json
```

## Resultado (gate congelado, spread P50 cruzado, 0,1 contrato)

1.966 decisões: 545 short, 1.073 flat, 318 sem forecast causal (início da
amostra), 30 falhas de cobertura/inversão.

| Conjunto | N | Média/trade (BTC) | t-stat |
|---|---:|---:|---:|
| Shorts sobrepostos | 545 | +0,000698 | 5,70 (**inválido**: trajetórias compartilhadas) |
| Não-sobrepostos (fase única) | 61 | -0,0000004 | -0,00 |
| Não-sobrepostos (21 fases) | ~61/fase | +0,000188 (média das fases) | -0,0 a +0,76 |

Por ano (média/trade, cruzando → postando no mid):

| Ano | N | Cruzando | Postando |
|---|---:|---:|---:|
| 2022 | 51 | +0,00275 | +0,00295 |
| 2023 | 117 | +0,00034 | +0,00045 |
| 2024 | 237 | +0,00073 | +0,00087 |
| 2025 | 85 | -0,00013 | 0,00000 |
| 2026 | 55 | +0,00071 | +0,00083 |

## Leitura

1. **O sinal aponta positivo, mas a significância não fecha.** Média por trade
   positiva em 4 de 5 anos, 64% de trades positivos, +0,38 BTC acumulados em
   545 entradas de 0,1 contrato. Porém o t-stat honesto (janelas independentes,
   varrendo as 21 fases de não-sobreposição) fica entre 0 e 0,76: com ~61
   janelas independentes e desvio de 0,0032, um edge verdadeiro de 0,0002-0,0007
   é indetectável. O dado real diz "pequeno e positivo, não provado".
2. **A discrepância entre pooled (+0,0007) e não-sobreposto (+0,0002) é
   informativa:** o gate dispara em rajadas (237 trades em 2024), e as rajadas
   boas pesam no pooled. Um book real com teto de posições captura parte disso;
   a média por janela independente é o piso conservador.
3. **2022 domina** (+0,14 de +0,38): regime de vol alta pós-crash. Consistente
   com toda a evidência anterior: o prêmio existe quando IV está alta; em vol
   comprimida (2025) o edge líquido é zero.
4. Postar em vez de cruzar segue somando ~+0,00014/trade, uniforme.

## Decisão

- A hipótese sai **viva, mas não provada**: sinal econômico positivo pequeno,
  abaixo do limiar de significância nas janelas independentes.
- Próxima alavanca de inferência (registrada): P&L diário agregado do book
  rolling com erros HAC, que usa toda a amostra sem o desperdício da
  subamostragem não-sobreposta; o motor rolling do Codex serve de base.
- Fase 5 (paper no holdout, teto 0,5 contrato/BTC) continua justificada como
  árbitro: custo zero e acumula a evidência que falta.
- Nenhum capital real; nenhuma regra alterada.

## Limitações

As do [backtest 2025](2026-09-02-backtest-tape-2025.md), mais: os 318 dias
iniciais sem forecast causal (exigência de 365d de treino + 30 targets) excluem
2021, justamente um regime de vol alta que provavelmente favoreceria o short.
