# Fase 3 — sizing, margem e ruína do book hedgeado (bootstrap)

Data: 2026-09-02
Estado: experimento executado; conclusão metodológica importante; nada aprovado
para dinheiro real.
Plano: [docs/2026-09-02-plano-pipeline-vol.md](2026-09-02-plano-pipeline-vol.md)

## Desenho

Duas camadas, ambas em `src/quant_pairs/hedged_book_bootstrap.py`:

1. **Trade hedgeado sob bootstrap.** As mesmas trajetórias conjuntas
   `(retorno BTC, ΔDVOL)` do block bootstrap (blocos de 4 dias, 2.000
   trajetórias por entrada, seeds da rodada anterior), agora reprecificadas
   como book **delta-hedgeado diário**: hedge no perp com taker fee, funding
   constante na média horária realizada (8,9e-6 ≈ 8% a.a.) e exigência de
   margem de manutenção aproximada da Deribit (0,075 + mark por opção vendida,
   0,5% do notional do perp) acompanhada dia a dia. Entrada no bid cruzado
   (piso conservador da Fase 2).
2. **Sequências de capital.** 4.000 sequências de 26 trades i.i.d. do pool,
   sizing fracionário (contratos ∝ equity), barreira de liquidação (equity
   marcada < margem exigida → fechamento forçado com penalidade de 25% do
   crédito) e ruína definida como equity ≤ 50% do capital inicial.

```bash
.venv/bin/python scripts/run_hedged_book_sizing.py
```

Artefato: `artifacts/hedged-book-sizing-v1.json`.

## Resultado bruto

Pool de 44.000 trajetórias (22 entradas × 2.000): **média por trade negativa**,
-0,0031 BTC/contrato, P(perda) 52%, P05 -0,077, pior -0,324. O Kelly resultante
é ≈ 0 (crescimento esperado de log-riqueza negativo em qualquer tamanho): sob
esta distribuição, **não existe tamanho seguro porque não há edge a dimensionar**.

Na grade de sizing (mecânica de margem, condicional a operar):

| Contratos/BTC | P(ruína) | P(liquidação) | P(DD>30%) | Mediana terminal |
|---:|---:|---:|---:|---:|
| 0,25 | 0 | 0 | 0 | 0,979 |
| 0,5 | 0 | 0 | 0,9% | 0,958 |
| 1,0 | 1,0% | 0,03% | 25,7% | 0,909 |
| 2,0 | 25,0% | 33,0% | 77,6% | 0,766 |
| 4,0 | 99,9% | 99,9% | 100% | 0,459 |

## A leitura que importa: o null estava errado para a pergunta

A decomposição por entrada mostra que a perda **não é custo de hedge** (fees +
funding custam só -0,0014 do total de -0,0031). O P&L bruto por entrada é quase
perfeitamente monotônico na IV de entrada: entradas com IV acima de ~54%
ganham; abaixo, perdem. E 54% é exatamente a vol diária anualizada do histórico
completo 2021-2026 que o bootstrap reamostra **incondicionalmente** contra
todas as entradas.

Ou seja: o bootstrap incondicional responde à pergunta "e se a vol futura for
sempre a média histórica?" — e nesse mundo, vender IV de 35-50% (todas as
entradas de 2023 em diante) perde por construção. Isso não é uma propriedade da
estratégia; é a negação da premissa dela.

O ponto fica explícito no subconjunto do gate congelado: as 6 datas `short`
(GARCH previu variância abaixo da IV de bid) tiveram 6/6 resultados positivos
no realizado, mas média **-0,0052** sob o bootstrap incondicional, porque o
null joga fora justamente a informação do forecast. Conclusão estrutural:

> **O edge alegado desta estratégia não é "carry de vol sempre existe"; é
> "o GARCH distingue regimes de vol futura". Um null que assume vol
> imprevisível reprova a estratégia por definição, não por evidência.**

Isso ecoa a lição do teste de gap (stress adversarial ≠ frequência): cada null
responde uma pergunta; é preciso nomear qual.

## O que a Fase 3 entrega apesar disso

1. **Mecânica de margem e teto operacional.** Mesmo no null pessimista, com
   ≤0,5 contrato por BTC de equity a barreira de liquidação nunca dispara e o
   drawdown >30% é raro. A margem começa a morder em ~2 contratos/BTC e é
   suicida em 4. **Teto declarado para o paper trading da Fase 5: 0,5
   contrato/BTC-equity, com meia-banda de segurança em 0,25.**
2. **Custos de hedge quantificados no bootstrap:** -0,0014 BTC/contrato-trade,
   consistente com o realizado da Fase 1. O hedge não é o problema.
3. **Critério de promoção afinado:** o que o holdout vivo precisa demonstrar
   não é P&L positivo genérico, e sim **skill do forecast** (o lado short do
   gate ganhando do null incondicional). Sem isso, o Kelly correto é zero.

## Próxima iteração metodológica (registrada, não executada)

Bootstrap **condicionado a regime**: reamostrar blocos de períodos com nível de
DVOL comparável ao da entrada (ou simular RV da distribuição preditiva do
GARCH), preservando a informação que a estratégia usa. Só esse null separa
"forecast tem skill" de "carry incondicional". Alternativa mais limpa e já em
curso: o holdout vivo a partir de 2026-09-03.

## Limitações declaradas

- Underlying aproximado pelo forward nas trajetórias.
- Funding constante; margem padrão (não-portfólio), conservadora para book
  hedgeado; penalidade de liquidação constante e crua (25% do crédito).
- Trades i.i.d. entre entradas: sem persistência de regime entre trades na
  camada de capital.
