# Forecast diário de volatilidade BTC — V1

Data: 2026-09-02
Estado: benchmark executado e integrado ao dashboard; diagnóstico, sem sinal
operacional ou autorização de capital.

## Pergunta

Modelos simples de retornos do BTC preveem a variância realizada futura melhor
que uma RV rolling, antes de tentarmos escolher long/short-vol nas poucas datas
com opções executáveis?

O forecast é estimado nos retornos do subjacente, não na IV. A comparação
econômica posterior é entre variância implícita e variância realizada prevista:

```text
gap de variância = IV² - E[RV²]
```

Gap positivo pode favorecer short-vol; gap negativo pode favorecer long-vol.
Isso ainda não é ordem porque falta descontar spread, fees, VRP normal, risco de
cauda e conferir a IV bid/ask do contrato negociável.

## Desenho pré-declarado

- BTC-PERPETUAL diário, com close disponível somente em `timestamp + 1 dia`.
- Alvos: média anualizada dos retornos logarítmicos quadráticos nos próximos 14
  e 30 dias.
- Treino mínimo de 365 dias.
- Benchmarks: RV rolling de 30 dias e EWMA com lambda fixo de 0,94.
- GARCH(1,1) zero-mean, expanding-window, reestimado a cada 30 dias.
- Métricas primárias: MSE de variância e QLIKE; menor é melhor.
- Correlação de RV e viés médio são diagnósticos, não critério isolado.
- Resultados diários sobrepostos e amostra com targets não sobrepostos são
  reportados separadamente.

Configuração: `config/experiment.volatility-forecast-v1.json`.

## Resultado em targets não sobrepostos

| Horizonte | Modelo | N | MSE variância | QLIKE | Correlação RV | Forecast médio | RV média futura |
|---|---|---:|---:|---:|---:|---:|---:|
| 14d | rolling | 121 | 0,05917 | -0,2340 | 0,291 | 48,07% | 46,44% |
| 14d | EWMA | 121 | 0,05759 | -0,2773 | 0,317 | 48,21% | 46,44% |
| 14d | GARCH | 121 | **0,05470** | **-0,3638** | 0,280 | 53,98% | 46,44% |
| 30d | rolling | 56 | 0,04211 | -0,2890 | 0,351 | 48,65% | 47,70% |
| 30d | EWMA | 56 | 0,04179 | -0,3157 | 0,326 | 49,68% | 47,70% |
| 30d | GARCH | 56 | **0,03851** | **-0,3597** | 0,312 | 55,67% | 47,70% |

O GARCH vence os dois critérios primários, mas prevê em média 7,5--8 pontos de
vol acima da RV futura. EWMA/rolling têm correlação semelhante ou maior. A
leitura correta é capacidade preditiva modesta com viés de nível; não escolher
GARCH automaticamente nem ajustar parâmetros após olhar a tabela.

## Monitor atual

Último instante comum disponível no artefato: 2026-09-01 08:00 UTC.

| Horizonte | DVOL | Rolling | EWMA | GARCH | DVOL² − GARCH² |
|---|---:|---:|---:|---:|---:|
| 14d | 37,63% | 47,99% | 46,85% | 51,18% | -0,1203 |
| 30d | 37,63% | 47,99% | 46,85% | 52,09% | -0,1297 |

O gap negativo sinaliza apenas um candidato a long-vol. DVOL é um índice de 30
dias e não substitui a IV ask do straddle 14d; além disso, nenhuma faixa de
confiança ou threshold de custo foi calibrada.

## Plataforma e rotina diária

O artefato `artifacts/btc-volatility-forecast-v1.json` alimenta
`/api/v1/volatility/forecast`. A página `/volatility` mostra o snapshot atual e
as métricas não sobrepostas junto do diagnóstico IV--RV e do carry.

```bash
.venv/bin/python scripts/collect_deribit_p0.py --start 2021-01-01T00:00:00Z
.venv/bin/python scripts/forecast_btc_volatility.py
.venv/bin/python scripts/build_volatility_report.py
.venv/bin/uvicorn quant_pairs.dashboard_api:app --reload
```

Ainda não existe agendamento: essa sequência é a rotina manual reproduzível.
Antes de automatizar, o próximo gate deve calibrar o viés do forecast e os
thresholds long/short/flat sem tocar no holdout econômico.
