# Roadmap de pesquisa — plataforma quant cripto

Data: 2026-08-31  
Status: pesquisa e priorização; nenhuma técnica foi aprovada para dinheiro real.

## Tese de construção

O objetivo é uma mesa pessoal **market-neutral e auditável**, não um painel que apenas dispara ordens. A plataforma deve separar pesquisa, dados, sinal, risco e execução. Cada ideia entra como hipótese versionada, passa por backtest líquido fora da amostra e paper trading, e só então pode receber capital pequeno.

O primeiro alpha é pairs trading por cointegração. A expansão deve ocorrer por camadas: primeiro retirar vieses e modelar fricção, depois adaptar o hedge e a seleção ao regime, e só então testar modelos mais complexos.

## Técnicas pesquisadas e decisão

| Técnica | O que pode acrescentar | Decisão |
|---|---|---|
| Engle–Granger aumentado + ECM | Base estatística para pares I(1) e velocidade de correção do spread. | **Implementar agora.** |
| Z-score/OU com parâmetros só do treino | Threshold e half-life coerentes com reversão à média, sem olhar o futuro. | **Implementar agora.** |
| FDR + holdout + DSR/Reality Check | Reduz falso positivo ao testar muitos pares, janelas e thresholds. | **Implementar agora.** |
| Backtester event-driven com bid/ask, fee e funding | Mede o P&L que pode existir depois de execução, não o retorno teórico do close. | **Implementar agora.** |
| Kalman filter para beta/alpha dinâmicos | Adapta o hedge ratio gradualmente quando a relação muda. | **Comparar como variante após o baseline.** |
| Matching em grafo | Escolhe uma carteira de pares sem reutilizar o mesmo ativo em vários trades. | **Implementar após o screener.** |
| Regime/structural-break filter | Evita abrir/segurar spread quando cointegração ou liquidez se rompe. | **P&D após baseline.** |
| ML/neural signal | Pode capturar não linearidade, mas eleva radicalmente risco de overfit. | **Fora do MVP.** |
| Arbitragem de funding/basis | Alpha distinto de pairs; usa carry de perp/spot e risco de basis/exchange. | **Trilha separada, depois do paper trading de pares.** |

## 1. Baseline estatístico correto

Para cada janela de formação:

1. Usar preços sincronizados e em log, do mesmo mercado de execução.
2. Aplicar Engle–Granger aumentado (`coint`) e armazenar p-value, estatística, especificação e hedge ratio.
3. Corrigir a família de p-values por FDR; selecionar apenas pares líquidos que sobrevivem à correção.
4. Congelar `alpha`, `beta`, média, escala e parâmetros do spread para a janela de trade seguinte.
5. Calcular o spread fora da amostra, executar somente no próximo candle disponível e registrar cada transição da posição.
6. Reestimar numa cadência pré-definida, sem otimizar a janela olhando o resultado futuro.

O modelo de erro-correção é útil para medir se o spread de fato se corrige e em qual velocidade; cointegração não é sinônimo de uma operação rentável. O teste `coint` do statsmodels implementa o Engle–Granger aumentado e declara como hipótese nula a ausência de cointegração. [Documentação](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.coint.html) · [Engle & Granger (1987)](https://ideas.repec.org/a/ecm/emetrp/v55y1987i2p251-76.html)

## 2. Hedge dinâmico: Kalman como experimento, não como premissa

O beta fixo pode envelhecer rápido em cripto. Um filtro de Kalman modela `alpha_t` e `beta_t` como estados que mudam suavemente, produzindo um spread adaptativo. A comparação honesta será:

- baseline: beta fixo por janela;
- variante K1: Kalman com hiperparâmetros definidos no treino;
- mesmo universo, mesmos custos, mesmos splits e mesma execução.

Só avançamos K1 se melhorar estabilidade líquida fora da amostra e não apenas o Sharpe agregado. Há literatura de pairs como modelo espaço-estado/Kalman, mas ela também alerta que o modelo é uma aproximação; não justifica pular os controles de validação. [Milstein et al. (2022)](https://arxiv.org/abs/2210.15448)

## 3. Seleção de carteira, não só ranking de pares

Selecionar todos os pares com menor p-value frequentemente reutiliza BTC, ETH ou outra moeda várias vezes e transforma uma aparente carteira de arbitragem em uma aposta concentrada. Vamos representar ativos como nós e pares aprovados como arestas ponderadas por qualidade líquida esperada, estabilidade e liquidez. Um matching ponderado cria pares sem ativo compartilhado; depois aplicamos limites de cluster/correlação e risco por exchange.

Esse desenho reduz concentração e turnover em relação ao ranking ingênuo, e deve ser medido contra ele no mesmo backtest. [Qureshi & Zaman (2024)](https://arxiv.org/abs/2403.07998)

## 4. Microestrutura cripto: onde a arbitragem vira ou deixa de virar P&L

O motor precisa operar sobre uma exchange inicialmente, com perpétuos lineares líquidos, e registrar:

- melhor bid/ask e profundidade no instante de decisão;
- maker/taker fee e slippage parametrizado por tamanho;
- funding realizado/esperado e intervalos de funding;
- limites de ordem, preenchimento parcial, latência e rejeições;
- margem, alavancagem, liquidation buffer e disponibilidade do contrato.

Para sinais baseados em candle, o mínimo é sinal em `t` e fill conservador em `t+1`; para a etapa seguinte, usar trades e quotes da própria exchange. Pesquisa específica de cripto mostra que modelar best bid/ask, disponibilidade de tamanho e atraso de execução muda materialmente o resultado de pairs trading. [Tadi & Kortchmeski (2021)](https://arxiv.org/abs/2109.10662)

Funding/basis deve entrar como custo/feature no pairs de perp primeiro. Uma estratégia de cash-and-carry ou cross-exchange é outro book, com inventário, transferência, risco de contraparte e reconciliação próprios; não deve mascarar o P&L do alpha de cointegração.

## 5. Robustez contra data snooping

Cada experimento deve gravar em uma tabela/arquivo imutável: universo, intervalo, versão de dados, número de pares, testes estatísticos, parâmetros tentados, custos, versão do código e resultados. O relatório não pode mostrar só o vencedor.

Gates obrigatórios:

- walk-forward com períodos de formação/trade congelados;
- holdout final intocado;
- FDR na descoberta de pares;
- bootstrap Reality Check ou SPA para a família de variações pesquisadas;
- Probabilistic/Deflated Sharpe Ratio, junto de drawdown, turnover e assimetria;
- teste de sensibilidade: custos maiores, atraso maior, thresholds/janelas próximos e remoção dos melhores pares.

O Reality Check foi proposto para corrigir a seleção entre muitas especificações; o SPA é uma alternativa relacionada. [White (2000)](https://doi.org/10.1111/1468-0262.00152) · [Hansen (2005)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=264569)

## 6. Arquitetura alvo

```
Exchange data → data lake/versionamento → features e screener
                                      ↓
                          backtest/research reports
                                      ↓
                         signal service + risk engine
                                      ↓
                    paper executor → exchange executor
                                      ↓
                 n8n/Telegram: alertas, aprovações, diário
```

O n8n orquestra alertas, coleta agendada, aprovações e rotinas administrativas. Cálculo de sinais, sizing e execução ficam em pacote Python versionado, testado e com logs determinísticos. O agente Telegram pode consultar relatórios e acionar operações permitidas, mas não receber privilégio para transferir fundos ou alterar regras de risco.

## Venue inicial: Binance para dados e paper trading

A conta existente na Binance reduz atrito para iniciar a coleta e o paper trading. A API disponibiliza WebSocket para mercado e para dados privados, e recomenda chaves Ed25519 para Spot. [Binance API](https://developers.binance.com/docs/binance-spot-api-docs)

Isso **não** é uma escolha definitiva para execução real. A escolha deve ser feita por par e por estratégia usando custo total realizado: spread, profundidade, maker/taker fee do nível efetivo da conta, funding, fill ratio, latência, limites de API e disponibilidade regional. A Bybit publica VIP 0 de 0,0200% maker / 0,0550% taker em perpétuos; a tabela de referência da OKX mostra 0,0200% / 0,0500% para o nível 1 — números que podem variar por região e conta. [Bybit](https://www.bybit.com/en/help-center/article/Trading-Fee-Structure) · [OKX](https://www.okx.com/en-us/help/how-to-calculate-the-contract-transaction-fee)

O adaptador de exchange é, portanto, requisito de arquitetura: o backtest armazena um `FeeSchedule` e um modelo de fill por venue; o executor só recebe uma implementação concreta depois que a comparação de paper trading estiver disponível.

## Backlog priorizado

### Fase A — fundação (próxima implementação)

- [ ] Estruturar pacote Python, ambiente travado e dados de candles cacheados.
- [ ] Criar modelo de dados para candles, quotes, funding, instrumentos e trades simulados.
- [ ] Reescrever screener Engle–Granger/FDR, com artefatos por janela.
- [ ] Construir backtester stateful com beta congelado, fills em `t+1`, custos e funding.
- [ ] Criar relatório de robustez e testes automatizados.

### Fase B — qualidade de portfolio

- [ ] Implementar matching ponderado e limites por ativo/cluster.
- [ ] Comparar beta fixo contra Kalman em experimento versionado K1.
- [ ] Adicionar filtros de liquidez, volatilidade e quebra de cointegração.

### Fase C — operação controlada

- [ ] Conectar uma exchange em modo leitura e validar dados/reconciliação.
- [ ] Paper trading contínuo com alertas no Telegram.
- [ ] Criar credenciais de execução sem saque, circuit breakers e aprovação explícita para qualquer ordem real.

## O que não faremos no início

- Otimizar parâmetros até “ficar bonito” no histórico.
- Misturar spot, perp e múltiplas exchanges sem modelar basis e inventário.
- Usar martingale, aumentar mão após perda ou operar sem stop/circuit breaker.
- Dar ao agente Telegram acesso de saque, alteração de risco ou execução irrestrita.
