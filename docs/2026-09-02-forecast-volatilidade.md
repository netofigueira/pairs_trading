# Forecast diário de volatilidade BTC — V1

Data: 2026-09-02
Estado: benchmark, correção causal e comparação estatística executados e
integrados ao dashboard; pesquisa, sem autorização de capital.

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
- Correção de nível expanding-window: razão entre variância realizada e
  prevista, usando somente alvos já encerrados em cada `forecast_at`, mínimo de
  30 alvos e fator limitado a `[0,5; 1,5]`.
- Diebold--Mariano sobre diferencial de QLIKE em targets não sobrepostos, com
  IC95% e referência Student-t. Diferença negativa favorece GARCH.

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

## Correção causal e Diebold--Mariano

Depois da correção de viés aplicada igualmente aos três modelos, o GARCH mantém
o menor QLIKE. Em 14 dias, o diferencial GARCH menos rolling é -0,1324
(IC95% -0,2447 a -0,0201; p=0,021) e contra EWMA é -0,0906
(IC95% -0,1811 a -0,0001; p=0,050). Em 30 dias, os efeitos continuam favoráveis,
mas os intervalos cruzam zero: p=0,078 contra rolling e p=0,082 contra EWMA.

Logo, há evidência moderada no horizonte de 14 dias e apenas evidência sugestiva
em 30 dias. Isto justifica usar 14 dias no gate econômico, não declarar que o
GARCH domina universalmente.

## Monitor atual

Último instante comum disponível no artefato: 2026-09-01 08:00 UTC.

| Horizonte | DVOL | Rolling corr. | EWMA corr. | GARCH corr. | DVOL² − GARCH² corr. |
|---|---:|---:|---:|---:|---:|
| 14d | 37,63% | 47,97% | 46,75% | 46,79% | -0,0773 |
| 30d | 37,63% | 47,71% | 46,48% | 46,69% | -0,0764 |

O gap negativo sinaliza apenas um candidato a long-vol. DVOL é um índice de 30
dias e não substitui a IV ask do straddle 14d. A regra executável usa as IVs de
bid/ask das duas pernas e está documentada no gate econômico separado.

## Plataforma e rotina diária

O artefato `artifacts/btc-volatility-forecast-v1.json` alimenta
`/api/v1/volatility/forecast`. A página `/volatility` mostra o snapshot atual e
as métricas não sobrepostas junto do diagnóstico IV--RV e do carry.

```bash
.venv/bin/python scripts/collect_deribit_p0.py --start 2021-01-01T00:00:00Z
.venv/bin/python scripts/forecast_btc_volatility.py
.venv/bin/python scripts/run_volatility_regime_gate.py
.venv/bin/python scripts/build_volatility_report.py
.venv/bin/uvicorn quant_pairs.dashboard_api:app --reload
```

Ainda não existe agendamento: essa sequência é a rotina manual reproduzível.
Dados a partir de 2026-09-03 estão selados pelo arquivo
`config/holdout.volatility-live-v1.json` e não podem ser usados para trocar
modelo, parâmetros ou regra.
