# Plataforma operacional de pairs trading

## Decisão

Vamos evoluir uma plataforma própria em paralelo ao núcleo quant. Ela será uma interface de **pesquisa e paper trading** até que os controles de risco, dados, custos e validação fora da amostra estejam aprovados. Nenhum componente da interface terá credenciais de exchange ou permissão de enviar ordens nesta fase.

O protótipo legado `coint_app/` demonstra a direção visual desejada (formulário e gráficos Plotly), mas não será promovido como base funcional: ele baixa dados do Yahoo Finance, executa em modo debug e usa a metodologia que a auditoria marcou como inválida para operação.

## Arquitetura alvo

```
Binance e futuros adaptadores públicos
        │
        ▼
TimescaleDB ──► núcleo quant / backtests ──► resultados de pesquisa
        │                                             │
        └──────────────── API de leitura ◄────────────┘
                              │
                              ▼
                    dashboard web (gráficos e alertas)
```

O banco separa os domínios `market`, `research` e `execution`. A tabela `execution.order_intent` foi criada somente como um contrato futuro e possui uma restrição explícita que permite exclusivamente `paper`; não há adaptador de execução nesta etapa.

## MVP do dashboard

1. Visão de saúde: atraso de coleta, cobertura por ativo/timeframe e qualidade de dados.
2. Screener: candidatos por cointegração, estabilidade, liquidez, correlação e custos estimados.
3. Detalhe do par: preços, spread/z-score, beta congelado, bandas, funding e eventos de entrada/saída do backtest.
4. Backtests reproduzíveis: parâmetros, intervalo, versão do código, métricas e lista de trades.
5. Paper blotter: sinais, posições virtuais, P&L realizado/não realizado e limites de risco.

### Primeira visão de pesquisa entregue

A rota `/volatility` é independente do banco operacional e consome o artefato
versionado `artifacts/volatility-research-v1.json` (cerca de 422 KB). Ela não lê
os 6,7 GB de quotes brutos no acesso e contém:

- KPIs das janelas independentes e da cobertura do carry;
- scatter IV/DVOL contra RV futura, com diagonal de calibração;
- série contextual de `IV² - RV²`;
- retornos trimestrais do long straddle carregado ao vencimento;
- cobertura de tamanho e alertas explícitos sobre interpretação.

Os gráficos usam SVG e JavaScript locais, sem dependência de CDN. A API de
leitura correspondente é `/api/v1/volatility/research`.

## Tecnologia e fronteiras

O próximo serviço será uma API Python de leitura sobre o pacote `quant_pairs` e TimescaleDB; o frontend poderá iniciar com Plotly para velocidade, mas com API separada para permitir uma UI web mais completa depois. n8n entra para alertas, agendas de coleta e notificações — não para calcular sinais críticos ou executar ordens.

Antes de qualquer tela de execução real, exigiremos: autenticação, trilha de auditoria, limites de exposição, kill switch, reconciliação de fills, chaves sem saque e período suficiente de paper trading.
