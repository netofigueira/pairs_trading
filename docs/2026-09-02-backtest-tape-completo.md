# Backtest diário no tape real — 2021-2026 completo

Data: 2026-09-02 (revisado no mesmo dia: seleção estritamente pré-decisão,
ver nota no doc de 2025)
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

1.966 decisões: 520 short, 975 flat, 318 sem forecast causal (início da
amostra), 153 sem cobertura de prints pré-decisão ou falha de inversão. A regra
causal estrita custa cobertura (146 dias a mais sem par), o preço correto de
não olhar o futuro.

| Conjunto | N | Média/trade (BTC) | t-stat |
|---|---:|---:|---:|
| Shorts sobrepostos | 520 | +0,000715 | 5,83 (**inválido**: trajetórias compartilhadas) |
| Não-sobrepostos (fase única) | 63 | +0,000235 | +0,64 |

A estrutura por ano se mantém: 2022 domina, 2025 ≈ zero (ver artefato
`tape-backtest-full-v1.json` para a quebra completa).

## Leitura

1. **O sinal aponta positivo, mas a significância por janelas independentes não
   fecha.** Média por trade positiva em 4 de 5 anos, 65% de trades positivos,
   +0,37 BTC acumulados em 520 entradas de 0,1 contrato. O t-stat em janelas
   não-sobrepostas (t=0,64) não tem poder para um edge dessa magnitude; a
   inferência eficiente está no [teste HAC](2026-09-02-inferencia-hac-book.md).
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
2021. O que 2021 teria dado é contrafactual não testado.
