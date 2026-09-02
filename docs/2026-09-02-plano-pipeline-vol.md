# Plano do pipeline de volatilidade — da hipótese ao capital

Data: 2026-09-02
Estado: plano aprovado para execução em fases; nada aprovado para dinheiro real.

## Diagnóstico que motiva o plano

O prêmio de variância existe no dado (P0: IV média 62,31% vs RV futura 51,68%,
IV > RV em 75,76% das 66 janelas), mas a expressão testada até aqui, short
straddle ATM 14 DTE sem hedge, tem média por trade negativa no bootstrap de
frequência histórica. A leitura não é "short-vol não funciona"; é que a
implementação desperdiça o prêmio em três lugares:

1. **Custo de execução.** Cruzar o spread nas duas pontas consome 3--7% do
   prêmio por perna (calibração Tardis), fatal para um prêmio bruto de ~10
   pontos de vol.
2. **Path dependence sem hedge.** O straddle não hedgeado mistura prêmio de vol
   com risco direcional. A cauda de 16,78× do bootstrap é majoritariamente
   delta acumulado, não vol.
3. **Incondicionalidade.** O gate GARCH congelado já mostrou que o lado short
   condicionado (6/6 positivos) supera vender sempre; o lado long perde.

## Fases

### Fase 1 — mudar a variável de pesquisa: P&L diário delta-hedgeado

Reconstruir, para as mesmas entradas trimestrais do envelope, o P&L do short
straddle **delta-hedgeado diariamente com o perp**, usando a marcação sintética
diária já existente (Black-76 inverso, IV ancorada no DVOL). O hedge é
rebalanceado a cada marcação, com taker fee do perp e funding horário real.

Por que primeiro: isola o prêmio de variância do ruído direcional e converte
observações trimestrais em séries diárias, dando poder estatístico que N=18
nunca terá. Responde em uma rodada se o prêmio sobrevive ao hedge.

Critério de saída: comparação hedgeado × não-hedgeado por trade (média,
dispersão, cauda). Se o hedge não reduzir substancialmente a variância sem
destruir a média, a hipótese short-vol nesta expressão morre aqui.

### Fase 2 — custo e execução como cidadão de primeira classe

Medir quanto do prêmio sobrevive sob três políticas de fill: cruzar o spread
(atual), postar no mid com fill parcial, e strikes 25-delta (strangle), que têm
spread relativo menor e prêmio de skew. Hipótese de mesa: a diferença entre
cruzar e postar decide sozinha a viabilidade. Em Deribit, o desenho padrão de
quem colhe VRP é ser maker na opção e hedgear no perp.

### Fase 3 — sizing, margem e probabilidade de ruína

Com a distribuição condicional por trade (bootstrap + Fase 1), calcular fração
de Kelly com desconto por incerteza de parâmetro (meia-Kelly ou menos), simular
a margem de portfólio da Deribit ao longo da trajetória e a barreira de
liquidação. Só aqui nasce a probabilidade de ruína de verdade. Regra prática: o
pior cenário do bootstrap com stop (~5× o crédito) tem que caber na margem sem
liquidação forçada.

### Fase 4 — gate como filtro short/flat, não sinal bidirecional

Manter o GARCH congelado (refit mensal, update diário, correção causal de
viés). Política: `forecast < IV_bid` → vende hedgeado; caso contrário, flat.
O lado long fica fora até existir pesquisa própria (timing de evento ou skew
barato — outra trilha). Filtro adicional barato e testável sem tocar no
holdout: não vender com DVOL no decil inferior do histórico (vol comprimida
tem a pior assimetria para short).

### Fase 5 — paper trading no holdout vivo, depois capital pequeno

Rodar a política short/flat hedgeada no holdout que começa em 2026-09-03
(`config/holdout.volatility-live-v1.json`), em paper, por 2 a 3 meses, com o
monitor diário existente. Promoção a capital real exige: média positiva no
paper, margem simulada nunca violada e nenhuma mudança de regra durante o
período.

## O que deliberadamente não faremos agora

- Consertar o lado long do gate (outra pesquisa, outra fonte de edge).
- Adicionar features ao GARCH antes de resolver expressão e custo — o gargalo
  não é forecast, é implementação.
- Aumentar frequência de entrada antes da Fase 2, porque frequência multiplica
  exatamente o custo que hoje mata a média.

## Invariantes de método (herdados e mantidos)

- Holdout fisicamente protegido; dado novo nunca reescreve regra antiga.
- Regras pré-declaradas antes de ver o resultado; nomeação honesta dos
  artefatos (envelope ≠ fill observado; perda por trade ≠ ruína).
- Artefatos versionados e reproduzíveis (seed fixa, SHA estável).
