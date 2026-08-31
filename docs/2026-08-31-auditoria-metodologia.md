# Auditoria metodológica — pairs trading por cointegração

Data: 2026-08-31  
Escopo: revisão estática do protótipo legado (`pair_trading.py`, `backtest.py`, `ibov_screener.py` e `pairs_trading.ipynb`). Não representa uma validação de rentabilidade.

## Veredito

O projeto tem a base conceitual correta: estima uma relação linear entre duas séries de preço, testa estacionariedade do resíduo e usa o z-score do spread como sinal de reversão. Isso é a ideia de Engle–Granger.

Entretanto, o backtest atual **não é válido para decidir operar**. Ele contém vazamento fora da amostra, não representa uma carteira hedgeada e não inclui os custos/fricções que determinam o resultado em cripto. A prioridade é reconstruir o motor de pesquisa e backtest antes de conectá-lo a n8n, Telegram ou APIs de exchange.

## O que o legado faz hoje

1. Baixa preços do Yahoo Finance.
2. Ajusta `y = alpha + beta*x` por OLS em uma janela de treino.
3. Aplica ADF ao resíduo e considera o par cointegrado quando `p < 0.05`.
4. Normaliza o spread e abre sinal ao cruzar ±2 desvios-padrão.
5. Fecha perto do centro ou após uma meia-vida estimada.

O notebook tenta fazer janelas de formação de 12 meses e negociação de 3 meses, o que é a direção certa para um walk-forward.

## Problemas críticos

### 1. O hedge ratio é reestimado no período de negociação

Em `backtest.py` e no notebook, o beta/alpha calculado no treino é usado para aprovar a cointegração, mas o resíduo negociado é recalculado por outro OLS sobre toda a janela de teste. Assim, cada sinal conhece preços futuros da própria janela e usa uma relação de hedge que não estaria disponível na entrada.

**Correção:** estimar `alpha`, `beta`, média e desvio apenas no treino; no teste, calcular `spread_t = y_t - alpha_train - beta_train*x_t`. Reestimar somente na próxima data de rebalanceamento.

### 2. O teste de cointegração não usa a estatística correta

O ADF direto no resíduo é uma aproximação comum, mas os valores críticos/p-values padrão do ADF não são os do teste residual de Engle–Granger. O código também não define seleção de defasagens, constante/tendência, nem trata a assimetria de escolher qual ativo é `y`.

**Correção:** usar `statsmodels.tsa.stattools.coint` (Engle–Granger aumentado), guardar estatística, p-value, valores críticos e especificação. Confirmar que cada série é compatível com I(1), e testar estabilidade em janelas sucessivas. Para cestas ou pares muito correlacionados, comparar depois com Johansen/ECM; não é o primeiro passo.

### 3. Seleção múltipla cria falsos pares

O screener testa milhares de combinações e aceita 5% pelo limiar fixo. Mesmo sem relações reais, aparecerão muitos falsos positivos; depois escolher os melhores resultados no mesmo histórico agrava data snooping.

**Correção:** formar pares dentro de grupos econômicos/mercado comparáveis, aplicar FDR (Benjamini–Hochberg) sobre os p-values, congelar a seleção no fim do treino e avaliar em um holdout nunca usado. Para o conjunto completo de estratégias/hiperparâmetros, aplicar bootstrap de Reality Check ou SPA.

### 4. P&L não representa a posição hedgeada

O resultado soma retorno percentual de uma ponta long e uma short como se ambas tivessem o mesmo notional. Isso ignora `beta`, normalização por dólar, gross exposure, capital/margem, leverage, borrow e rebalanceamento. Em alguns ramos do notebook há também atribuição da saída short com o preço da ponta long.

**Correção:** no instante de entrada, definir unidades por notional: por exemplo `q_y = +G/(2*y)` e `q_x = -G/(2*abs(beta)*x)` (ajustando o sinal conforme o spread), registrar cash, P&L diário mark-to-market e retorno sobre gross/notional ou margem explicitamente.

### 5. Execução e saída têm viés e regras incompletas

O sinal é observado no fechamento e a operação usa o mesmo preço; isso é look-ahead operacional. A saída por meia-vida fixa não equivale a uma regra de stop, e o código permite fragilidade de índice/calendário ao somar barras. Não há stop-loss, limite de holding, limite de posições correlacionadas ou kill switch de regime.

**Correção:** sinal em `t`, execução no próximo candle com bid/ask (ou uma hipótese conservadora de slippage); usar state machine de uma posição por par, saída por z-score, stop de z-score/perda, time stop e fechamento ao perder validade estatística.

### 6. Dados e reprodutibilidade ainda são de protótipo

`Close` e `Adj Close` são usados de forma inconsistente; não existe requirements/lockfile, cache de dados, calendário, validação de dados ou testes funcionais. Os testes atuais não correspondem ao retorno real de `get_price_data`, esperam uma coluna diferente da usada pelo código e não cobrem o backtest.

**Correção:** pacote Python com ambiente travado, dados versionados/cacheados, testes unitários para hedge ratio, spread fora da amostra, transições de posição e P&L, além de testes de integração determinísticos.

## Adaptação para cripto

Cripto é adequada para pesquisa de pares, mas muda o problema de execução.

| Camada | Decisão recomendada |
|---|---|
| Universo inicial | Perpétuos lineares e líquidos da mesma exchange; evitar cruzar spot e perp na primeira versão. |
| Dados | Candles e, para execução, bid/ask, trades e funding rate da própria exchange. Nunca usar preço de uma fonte e executar em outra sem modelar basis/latência. |
| Sinal | Trabalhar em log-preços; beta congelado da janela de formação; z-score rolling calculado só com informação passada. |
| Custos | Maker/taker, spread, slippage por volume, funding, borrow quando aplicável, taxas de saque e impacto de rebalanceamento. |
| Risco | Limite de gross/net exposure, concentração por moeda, alavancagem, drawdown diário, desconexão de API e desligamento automático. |
| Operação | Começar com paper trading e reconciliação contra a exchange; só então execução mínima, com chaves sem permissão de saque. |

Uma arquitetura inicial saudável é: **coletor de dados → banco de candles/features → pesquisa/backtest walk-forward → serviço de sinais → paper executor → n8n/Telegram para alertas e comandos**, mantendo a decisão de trade no serviço quant versionado, não em nós Code do n8n.

## Plano de evolução

1. Criar o pacote `pairs/` e um ambiente reprodutível; preservar os scripts legados como referência.
2. Implementar o screener com `coint`, FDR, filtros de liquidez e persistência de resultados por janela.
3. Implementar um backtester event-driven, sem OLS no teste, com posições beta-neutral, custos e funding parametrizados.
4. Validar por walk-forward e holdout final; reportar retorno líquido, Sharpe/Sortino, max drawdown, turnover, hit rate, tempo em posição e sensibilidade aos parâmetros.
5. Integrar dados de uma única exchange e paper trading.
6. Só após aprovação dos gates, expor comandos/alertas via Telegram e usar o n8n como orquestrador.

## Critérios mínimos para avançar a paper trading

- Sem look-ahead em dados, parâmetros, seleção de pares ou execução.
- Custos conservadores e funding incluídos.
- Resultado positivo em vários blocos temporais fora da amostra, não apenas no agregado.
- Robustez a variações razoáveis de janela, threshold e custo.
- Limites de risco e logs/reconciliação implementados.

## Referências

- Engle & Granger (1987), [Co-integration and Error Correction: Representation, Estimation, and Testing](https://ideas.repec.org/a/ecm/emetrp/v55y1987i2p251-76.html).
- statsmodels, [teste `coint` de Engle–Granger aumentado](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html).
- White (2000), [A Reality Check for Data Snooping](https://doi.org/10.1111/1468-0262.00152).
- Hansen (2005), [A Test for Superior Predictive Ability](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569).
