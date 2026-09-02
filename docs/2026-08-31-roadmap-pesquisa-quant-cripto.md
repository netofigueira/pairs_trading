# Roadmap de pesquisa — plataforma quant cripto

Data: 2026-08-31  
Status: pesquisa e priorização; nenhuma técnica foi aprovada para dinheiro real.

## Tese de construção

O objetivo é uma mesa pessoal **market-neutral e auditável**, não um painel que apenas dispara ordens. A plataforma deve separar pesquisa, dados, sinal, risco e execução. Cada ideia entra como hipótese versionada, passa por backtest líquido fora da amostra e paper trading, e só então pode receber capital pequeno.

O primeiro alpha é pairs trading por cointegração. A expansão deve ocorrer por camadas: primeiro retirar vieses e modelar fricção, depois adaptar o hedge e a seleção ao regime, e só então testar modelos mais complexos.

## Registro de implementação

- **2026-08-31 — base quant:** criado o pacote `quant_pairs`, com modelo Engle–Granger de parâmetros congelados fora da amostra e testes de regressão contra refit/look-ahead.
- **2026-08-31 — dados públicos:** criado coletor read-only Binance USDⓈ-M para klines e funding paginados, normalizados em UTC e persistidos de forma deduplicada em `data/` (fora do Git). Validação real: BTCUSDT 1h, 7 dias, 168 candles e 21 eventos de funding; 6 testes automatizados verdes.
- **2026-08-31 — primeiro e2e do backtester:** motor event-driven criado (fill em t+1, sizing por beta, fee, slippage, funding, stop/time stop) e validado por testes. A sonda BTCUSDT×ETHUSDT em 90d **reprovou** a formação de 30d (`p=0,400940`) e teve -4,02% líquido no diagnóstico forçado; portanto não é candidato. O CLI agora bloqueia por padrão pares com `p >= 0,05`.
- **2026-08-31 — TimescaleDB privado:** migrations criadas para `market`, `research` e `execution`; a VM recebeu TimescaleDB 2.27.1/PostgreSQL 17 com volume persistente, limite de 3 GB/1,25 CPU e porta somente em `127.0.0.1:5433`. A ingestão pública BTCUSDT 1h foi validada ponta a ponta (168 candles e 21 fundings). A interface operacional fica estritamente em pesquisa/paper trading nesta fase.

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

## Protocolo intraday v1 (declarado antes da avaliação)

Para a primeira avaliação, inicialmente estimada em 187 dias disponíveis,
preservaremos candles de 1h e compararemos três tamanhos de formação: 30, 60 e
90 dias. Cada variante
refaz Engle--Granger/FDR e congela os parâmetros antes de negociar os 7 dias
seguintes; o refit é semanal e as janelas OOS não se sobrepõem. Os últimos 30
dias comuns ficam reservados como holdout final e não participam da escolha da
variante. A especificação executável é
[`config/experiment.crypto-usdm-intraday-v1.json`](../config/experiment.crypto-usdm-intraday-v1.json).

Esta comparação é uma pesquisa de especificação, não uma aprovação de alpha:
qualquer variante escolhida nos folds de pesquisa deve ser executada uma única
vez no holdout, com funding e custos conservadores, antes de avançar.

A escolha de manter granularidade intraday é compatível com estudos de pairs em
cripto que formam sinais em dados por minuto, mas a próxima melhoria de
execução deve substituir a hipótese de candle por bid/ask e trades. A literatura
sobre walk-forward intraday também mostra que a escolha de janelas influencia o
resultado, reforçando a reserva de um OOS independente após a comparação das
variantes. [Tadi & Kortchmeski (2021)](https://arxiv.org/abs/2109.10662) ·
[Mroziewicz & Ślepaczuk (2026)](https://arxiv.org/abs/2602.10785)

### Primeira execução — 2026-09-01

O banco continha 8.760 candles comuns fechados de 1h (365 dias), não apenas os
187 dias inicialmente estimados. Foram reservadas as últimas 720 barras (de
2026-08-02 02:00 UTC a 2026-09-01 01:00 UTC) como holdout intocado. No período
de pesquisa anterior, com Engle--Granger/FDR reexecutado em cada fold, o resumo
foi:

| Formação | Folds | Pares aprovados em folds | Trades | P&L líquido | Hit rate |
|---|---:|---:|---:|---:|---:|
| 30 dias | 43 | 57 | 80 | -0,0481 | 45,0% |
| 60 dias | 39 | 63 | 35 | -0,3695 | 42,9% |
| 90 dias | 35 | 83 | 57 | +0,1706 | 56,1% |

A janela de 90 dias é a única candidata a validação adicional; o resultado não
aprova paper trading. O P&L acima soma trades individuais de notional bruto 1,
ainda não representa uma carteira com matching/concentração nem inclui funding.
Antes de tocar no holdout, a especificação V1 será repetida no período de
pesquisa com funding realizado, matching sem ativo compartilhado e três cenários
de custo. Só se sobreviver a esse gate haverá **uma** execução final no holdout,
com métricas de equity curve, drawdown e turnover.

## Próximo experimento — escala de volatilidade e preço executável

O baseline vencedor mantém `alpha`, `beta` e a média do spread congelados nos
90 dias de formação. Antes de comparar hedge dinâmico por Kalman, vamos testar
se a escala fixa do z-score está confundindo mudança recente de volatilidade
com sinal de reversão. O teste não suaviza os preços nem reaplica
Engle--Granger sobre uma série filtrada: isso alteraria a distribuição do teste
e poderia criar estacionariedade espúria.

Variantes declaradas, a avaliar somente no período de pesquisa anterior ao
holdout:

| Variante | Seleção/hedge | Escala do sinal | Execução |
|---|---|---|---|
| V1 | baseline Engle--Granger/FDR, OLS congelado em 90d | desvio da formação | close t+1 + fee/slippage atual |
| V2 | igual a V1 | volatilidade EWMA do spread, calculada até t-1 | igual a V1 |
| V3 | igual a V1 | desvio rolling do spread de 72h, calculado até t-1 | igual a V1 |
| E1 | V1 | desvio da formação | melhor bid/ask t+1 e profundidade disponível |
| E2 | melhor escala entre V2/V3 | escala vencedora | melhor bid/ask t+1 e profundidade disponível |

A média permanece congelada neste experimento para mudar uma dimensão por vez.
Para V2/V3, o spread é sempre calculado com os parâmetros de formação:

```text
spread_t = log(y_t) - alpha_formacao - beta_formacao * log(x_t)
z_t = (spread_t - media_formacao) / vol_estimado_com_dados_ate_t_menos_1
```

O coletor deve persistir best bid, best ask e tamanho disponível, além de
trades. A simulação usará ask ao comprar e bid ao vender; a profundidade e o
volume do sinal determinam o slippage ou preenchimento parcial. Só uma variante
que melhore consistentemente nos folds, sobreviva a custos maiores e mantenha
o resultado no holdout poderá seguir para paper trading.

Há também arquivo histórico público da Binance Vision para USDⓈ-M Futures:
`bookTicker` contém top-of-book (preço e quantidade de melhor bid/ask) e
`bookDepth` contém profundidade agregada. O primeiro é o candidato para
reconstituir fills históricos; o segundo só pode ser usado como feature de
liquidez, não como livro executável. A cobertura de `bookTicker` disponível no
arquivo público é limitada aproximadamente a 2023-05--2024-03, portanto não
cobre o período de pesquisa corrente 2025-09--2026-09. Faremos duas trilhas:
um backtest de microestrutura separado nesse intervalo histórico e coleta
contínua de bookTicker/trades para a operação/paper atual. [Binance Vision —
bookTicker USDⓈ-M](https://data.binance.vision/?prefix=data/futures/um/daily/bookTicker/BTCUSDT/) ·
[repositório público da Binance, discussão de cobertura](https://github.com/binance/binance-public-data/issues/380)

Kalman continua como experimento K1 posterior: ele atualiza `alpha`/`beta` de
forma online e será comparado contra V1/V2/E2 sem alterar o gate de seleção
Engle--Granger/FDR. O sinal de K1 deve usar a inovação prevista antes da
atualização pelo candle atual, com fill somente no candle seguinte.

### Resultado V1/V2/V3 — 2026-09-01

As três escalas foram avaliadas nos mesmos 35 folds de pesquisa, com formação
de 90 dias e o mesmo holdout de 30 dias preservado. V2/V3 receberam o histórico
de spread da formação apenas para inicializar a escala; em cada candle de trade,
a escala foi deslocada para usar dados no máximo até `t-1`.

| Variante | Trades | P&L líquido | P&L médio/trade | Hit rate | Decisão |
|---|---:|---:|---:|---:|---|
| V1 — escala da formação | 57 | +0,1706 | +0,0030 | 56,1% | manter |
| V2 — EWMA 72h | 186 | -0,3586 | -0,0019 | 47,3% | rejeitar |
| V3 — rolling 72h | 216 | -0,3840 | -0,0018 | 44,0% | rejeitar |

As escalas recentes reduziram o limiar efetivo em regimes de baixa volatilidade
e elevaram fortemente o turnover; com fee/slippage atual, não compensaram. O
baseline V1 de 90 dias permanece como única especificação candidata. Não será
feita busca adicional de spans EWMA/rolling: isso ampliaria a família de testes
após observar o resultado. A próxima melhoria é incluir funding no runner e
coletar bid/ask/trades para testar E1 sem tocar no holdout.

### Hipótese H1 — janela do z-score calibrada pela meia-vida

Uma contribuição prática de Ernest Chan para mean reversion é usar a meia-vida
estimada do spread para escolher a escala temporal de média/desvio do sinal, em
vez de escolher uma janela arbitrária. A abordagem aparece em sua discussão de
estratégias de mean reversion e é implementada explicitamente em pesquisa de
pairs cripto: estima-se a velocidade OU na formação e dela se deriva uma janela
rolling/EMA para o z-score. [Chan, *Algorithmic Trading* (2013)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118676998) ·
[Tadi & Kortchmeski (2021)](https://arxiv.org/abs/2109.10662)

H1 é distinta de V2/V3: para cada fold, a janela será derivada **somente** da
meia-vida já estimada na formação daquele par, limitada previamente ao intervalo
de 4--72 barras, e permanecerá fixa durante seus 7 dias OOS. Não haverá busca
de spans depois de ver resultados. A comparação será V1 versus H1 com as mesmas
regras de execução; funding será integrado antes de qualquer avaliação de
holdout. O holdout continua intocado.

#### Resultado H1 — 2026-09-01

H1 foi executada contra V1 nos mesmos 35 folds, com a janela rolling de cada
par arredondada a partir de sua meia-vida da formação. Ela produziu 261 trades,
P&L líquido de **-0,1504** e hit rate de 40,6%, contra 57 trades, **+0,1706** e
56,1% do baseline V1. A adaptação pela meia-vida tornou o sinal excessivamente
reativo e elevou o turnover; H1 está rejeitada e não será levada ao holdout.

A nota de Chan sobre pairs também favorece diversificação e alerta que relações
estatísticas podem falhar por eventos específicos. A extrapolação para cripto é
tratar listagem/delistagem, unlocks, alterações de contrato, incidentes de
venue e mudanças anormais de funding como eventos de exclusão/regime, além de
aplicar o matching sem ativos repetidos já previsto no roadmap. Esta é uma
inferência de desenho de risco, não uma evidência de que os mesmos eventos de
ações se comportem igual em cripto. [Chan, *Pair Trading Stocks* (2007)](https://www.epchan.com/subscription/PairTradingStks.pdf)

### S1 — pairs swing com funding — 2026-09-01

S1 preservou candles de 1h, mas ampliou a formação para 180 dias, o OOS/refit
para 14 dias e o holding máximo para 7 dias. Diferentemente das execuções
anteriores, carregou os eventos históricos de funding de ambas as pernas no
P&L. Em 11 folds de pesquisa, somente 2 pair-folds passaram pelo FDR e houve
2 trades: P&L líquido de **-0,0034** e hit rate de 50%. O resultado é
insuficiente e não positivo; S1 não será levada ao holdout.

Isso não demonstra que todo pairs swing em cripto falha, mas mostra que esta
configuração (universo, formação de 180d e gates atuais) quase não encontra
oportunidades. Não faremos busca retrospectiva de thresholds/janelas para
forçar trades.

### Gate de validação V1 — funding, concentração e custo — 2026-09-01

A candidata V1 intraday, com formação de 90 dias, foi reexecutada no período de
pesquisa de 2025-09-01 02:00 a 2026-08-02 01:00 UTC, preservando integralmente
o holdout final de 30 dias. Em cada um dos 35 folds o runner recalculou
Engle--Granger/FDR; os pares aprovados foram submetidos a matching ponderado
exato, sem reutilizar ativo no mesmo fold. O P&L incluiu funding histórico das
duas pernas. Houve 83 pair-folds aprovados antes do matching e 29 depois dele.

| Cenário | Custo por perna (fee + slippage) | Trades | P&L líquido | P&L médio/trade | Hit rate | Max drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Base | 6 bps (5 + 1) | 24 | -0,0603 | -0,0025 | 41,7% | -0,0851 |
| Stress 1 | 12 bps (10 + 2) | 24 | -0,0890 | -0,0037 | 41,7% | -0,0958 |
| Stress 2 | 18 bps (15 + 3) | 24 | -0,1177 | -0,0049 | 37,5% | -0,1066 |

V1 está **rejeitada**: falha já no cenário-base e piora monotonicamente com
custos. O +0,1706 observado antes não era uma carteira executável: não incluía
funding nem o limite de concentração. O holdout não será usado para esta
hipótese, pois isso transformaria o período protegido em mais uma tentativa de
encontrar resultado favorável. Também não faremos tuning de janela, entrada ou
stop para resgatar V1.

O próximo ramo cripto deve ser separado: carry de funding/basis (perp versus
spot), com inventário, risco de exchange e custos próprios. Antes de execução,
esse ramo precisará de hipótese versionada e walk-forward independente; não
será apenas uma variante de pairs mean-reversion.

### V0 — volatilidade implícita versus realizada — 2026-09-01

Foi aberta uma trilha independente de opções, inspirada no princípio de Euan
Sinclair: a hipótese negociável não é “volatilidade alta/baixa”, mas a diferença
entre a variância implícita paga na opção e a variância que de fato se realiza
até o vencimento. O instrumento exige cadeia de opções **com bid/ask**, hedge
delta no perp e atribuição de P&L por theta, gamma, vega, hedge, spread, fee e
funding. Uma previsão de volatilidade feita só com candles de perp pode ajudar
o sizing, mas não é um teste de volatility trading puro.

O adaptador público da Deribit foi validado em uma sonda BTC: 978 contratos de
opções com bid/ask/mark-IV e 25 barras horárias de DVOL, sem credencial e sem
qualquer capacidade de enviar ordem. As cotações serão armazenadas como
snapshots imutáveis por contrato e timestamp; o histórico de DVOL será
deduplicado por timestamp. A Deribit oferece HTTP JSON-RPC e ambientes de teste
separados, mas a elegibilidade brasileira e KYC serão verificados diretamente
com a venue antes de qualquer conta ou operação. [Deribit API](https://docs.deribit.com/)

Não precisamos aguardar o histórico live para iniciar: fornecedores possuem
quotes e livros históricos da Deribit. Tardis disponibiliza quotes, snapshots e
updates L2; Amberdata também declara histórico de trades e order book desde
2021. Há ainda uma fonte menor, Volar, que declara arquivo denso de BTC de
2021-06 a 2024-09, mas seu trecho posterior é uma superfície modelada e deve ser
separado de dados de exchange. A API pública da Deribit permite consultar o
livro **corrente**, não reconstituir retrospectivamente a cadeia bid/ask. Por
isso: (a) dados de trade/mark podem servir à descoberta da hipótese; (b) só
quotes históricos de exchange podem aprovar o backtest líquido. [Deribit —
order book atual](https://docs.deribit.com/api-reference/market-data/public-get_order_book) ·
[Tardis](https://tardis.dev/) · [Amberdata](https://www.amberdata.io/deribit-market-data) ·
[Volar](https://www.volardata.com/)

O leitor Parquet foi exercitado contra a amostra pública da Volar (ETH,
2026-07-14 12:00--14:00 UTC): 58.030 linhas, 87 snapshots e 674 instrumentos.
Após aplicar os gates de execução, restaram 49.213 quotes e 640 contratos; 8.817
linhas foram recusadas por lado não cotado, preço cruzado, expiração ou fonte
não-live. O arquivo era exclusivamente `live_ws`. Isto valida o schema e os
filtros, mas **não** constitui backtest: duas horas de ETH não bastam para
formar nem avaliar uma hipótese. A próxima aquisição deve ser uma janela
histórica BTC de quotes reais; dados `modeled_surface` permanecem fora de todo
resultado de P&L executável.

A integração autenticada sandbox foi então validada com uma cadeia BTC pontual
de 2026-08-25 12:00 UTC: 100 contratos, timestamp de exchange e fonte
`live_ws`, armazenados em Parquet local. Os 100 passaram no gate de execução.
A sandbox oferece apenas BTC e 14 dias de histórico; ela é suficiente para
validar API, schema e armazenamento, mas não para estimar a relação IV--RV nem
para produzir qualquer conclusão de P&L. A chave fica exclusivamente em `.env`
ignorado pelo Git e nunca é impressa. O cliente HTTP usa identificação explícita
de pesquisa read-only, necessária para não acionar o filtro anti-bot da API.

Em seguida foram coletados 27 snapshots BTC na janela sandbox, de 2026-08-18
23:59 a 2026-08-31 23:59 UTC, com cadência-alvo de 12 horas. Após preencher uma
lacuna observada em 2026-08-29, a mediana entre snapshots foi 12,0001 horas, a
maior lacuna 12,0098 horas e não houve timestamps duplicados. Essa coleta é
intencionalmente pequena: mede a continuidade da API, não o desempenho de uma
estratégia.

Como smoke test do painel IV--RV, selecionamos calls com delta mais próximo de
0,5 e vencimento de 10--16 dias, e comparamos a mark-IV de entrada contra a RV
realizada nos 7 dias subsequentes usando os retornos de 12h. Só **duas** entradas
tiveram tanto contrato elegível quanto 7 dias futuros completos: IV média 38,81%
e RV média 25,11%. O resultado é registrado somente como validação de cálculo;
`n=2` não é estimativa de prêmio, não suporta teste estatístico e não pode
autorizar venda de volatilidade. A próxima amostra para pesquisa precisa cobrir
múltiplos regimes e vencimentos com quotes históricos reais.

#### P0 — calibração pública DVOL contra RV futura — 2026-09-01

O endpoint público de histórico do DVOL foi paginado até o início disponível:
1.988 barras diárias de 2021-03-24 a 2026-09-01. Em paralelo, foram guardadas
2.070 barras diárias de BTC-PERPETUAL. Para cada DVOL no instante `t`, a P0 usa
somente os 30 log-retornos fechados **depois** de `t` para calcular a RV futura
anualizada; janelas incompletas ou com lacuna são excluídas. Isso permite testar
a calibração do índice sem comprar dados pagos e sem reconstruir artificialmente
um livro de opções.

Nos 1.958 desfechos diários sobrepostos, IV média foi 60,48%, RV futura 51,36%
e IV excedeu RV em 73,49% dos casos. Como esses desfechos compartilham quase
todos os retornos, a leitura principal usa 66 janelas de 30 dias não
sobrepostas: IV média 62,31%, RV 51,68%, diferença média de **10,64 pontos de
volatilidade**, IV maior que RV em 75,76% e correlação IV--RV de 0,65. É uma
hipótese de prêmio de variância economicamente plausível, mas ainda **não é
P&L, Sharpe nem aprovação de short-vol**: DVOL é um índice derivado da
superfície, e faltam preço de entrada/saída, spreads por strike, hedge delta,
fees, funding e perdas de cauda. A metodologia oficial caracteriza o DVOL como
uma expectativa anualizada e prospectiva de volatilidade para os próximos 30
dias. [Deribit — metodologia DVOL](https://insights.deribit.com/exchange-updates/dvol-deribit-implied-volatility-index/)

P0 está concluída como gate de descoberta. P1 só poderá converter a hipótese em
backtest quando usar quotes históricos executáveis; Volar sandbox serve para
testar a tubulação, e Tardis/Amberdata continuam candidatos para adquirir a
amostra longa, mediante decisão explícita de custo.

#### P1 — gate de quotes executáveis — iniciado em 2026-09-01

Foi implementado um auditor que só forma um round-trip quando encontra o
**mesmo instrumento** em snapshots distintos: compra hipotética no `ask` de
entrada e venda hipotética no `bid` de saída. Ele não preenche preços ausentes
com mark, não usa uma superfície modelada e não produz P&L enquanto a regra de
hedge e risco não estiver declarada.

Na amostra Volar sandbox atual, há 27 snapshots, 2.412 quotes executáveis e
422 contratos distintos em 13 dias. Para o horizonte de 7 dias, 10 snapshots
de entrada possuem snapshot de saída dentro de uma hora do alvo, produzindo 478
round-trips de contratos idênticos. O gate permanece **reprovado para backtest
histórico**: menos de 180 dias e um único regime não permitem escolher,
validar e reservar OOS uma estratégia. A próxima ação requer autorização de
aquisição de quotes longos (por exemplo, Tardis/Amberdata/Volar plano pago) ou
coleta contínua; somente então P1 definirá a estratégia delta-hedged, custo,
limites de vega/gamma e stress de salto.

Há agora um gate gratuito anterior à aquisição: a Tardis libera os CSVs do
primeiro dia de cada mês. A sonda de `2024-01-01` baixou com sucesso quotes
reais de todas as opções Deribit (133 MB; `ask_price`, `ask_amount`,
`bid_price`, `bid_amount`) e do BTC-PERPETUAL (8,2 MB), ambos provenientes da
captura WebSocket da fonte. O `HEAD` do CDN retornou 404, mas o `GET` do arquivo
funcionou; o coletor usa somente `GET` e escrita atômica. Essas cross-sections
mensais não podem estimar o P&L de manter uma opção por 7--30 dias, pois não
oferecem a cotação de saída entre os dias amostrados. Elas permitem, porém,
validar ponta a ponta o parser, reconstrução de quote, regra ask/bid, hedge
intraday e custos usando dados reais antes de comprar cobertura contínua.

O primeiro round-trip intraday foi executado na cross-section de 2024-01-01,
entre 12:00 e 20:00 UTC. A seleção observável escolheu
BTC-12JAN24-43000-C/P (10,83 DTE), comprou ambos no ask e encerrou os mesmos
contratos no bid. O `options_chain` real (1,87 GB comprimidos) forneceu deltas
de +0,53343 e -0,46657 na entrada. A exposição líquida de +0,06686 BTC foi
neutralizada com -286 contratos do BTC-PERPETUAL, deixando -0,00010 BTC de
residual por arredondamento. O corte as-of usa o timestamp de captura local,
portanto não aceita mensagens recebidas depois do instante de decisão.

Por straddle, opções+hedge tiveram movimento mid-to-mid de +0,003181 BTC,
custo de spread de 0,005001 BTC e fees de 0,001266 BTC. O líquido antes de
funding foi -0,003086 BTC. Como a janela de oito horas exige contabilidade de
funding do perp, o campo de líquido delta-hedged final continua nulo. Isso
valida sincronização, seleção, continuidade, hedge inverso e custos, mas não
mede performance: é uma única data, e uma cross-section intraday não testa o
carry da opção até o vencimento.

Também foi avaliada uma alternativa P1a gratuita: usar o endpoint público
`get_mark_price_history` da própria Deribit como proxy, junto do perp para o
hedge. A hipótese foi **rejeitada antes de implementação**. Embora a
documentação diga que há marks de 5 minutos para um subconjunto de opções do
DVOL, a consulta de produção para uma opção BTC histórica expirada foi aceita
na sintaxe REST, mas devolveu `instrument is not active`. Logo, ela não permite
reconstituir a cadeia de contratos que existia no passado; consultar apenas
instrumentos ainda ativos seria viés de sobrevivência. O endpoint permanece
útil para marcas de contratos ativos/coleta futura, não para P1a histórica.
Esta conclusão é consistente com a limitação documentada a um subconjunto das
opções do DVOL. [Deribit — mark-price history](https://docs.deribit.com/api-reference/market-data/public-get_mark_price_history)

O desenho V0 fica pré-registrado assim:

1. Coletar continuamente BTC e ETH: cadeia, bid/ask, mark-IV, underlying,
   open interest e DVOL; excluir contratos sem ambos os lados cotados.
2. Formar buckets ATM de 7, 14 e 30 dias por tempo até vencimento, sempre com
   a cadeia observada no instante de decisão.
3. Comparar IV com variância realizada **subsequente** no mesmo horizonte;
   treinar e escolher regras somente em folds anteriores.
4. Simular execução no ask para compra e no bid para venda, hedge delta no
   perp no candle seguinte e custos/funding explícitos.
5. Reportar P&L e cauda: drawdown, pior salto, exposição vega/gamma, turnover,
   fill e desempenho por regime. Vender vol que tenha retorno médio positivo e
   perda inaceitável em choque será reprovado.

Não será testada venda sistemática de straddle como atalho para “coletar prêmio”;
o prêmio de variância pode remunerar precisamente o risco de crash. A literatura
de opções de BTC documenta IV, variância realizada e prêmio de variância, mas
não elimina o risco de cauda ou prova rentabilidade após execução. [Alexander &
Imeraj (2020)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3383734) ·
[Sinclair, *Volatility Trading*](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118662724)

### Fase A — fundação (próxima implementação)

- [x] Estruturar pacote Python, ambiente travado e dados de candles cacheados.
- [x] Criar modelo de dados para candles, funding, instrumentos e trades simulados.
- [x] Reescrever screener Engle–Granger/FDR, com artefatos por janela.
- [x] Construir backtester stateful com beta congelado, fills em `t+1`, custos e funding.
- [ ] Criar relatório de robustez e testes automatizados.

### Fase B — qualidade de portfolio

- [x] Implementar matching ponderado sem ativos repetidos por fold.
- [ ] Implementar limites complementares por cluster/correlação.
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
