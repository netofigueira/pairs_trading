# Piloto trimestral — straddle BTC carregado ao vencimento

Data da execução: 2026-09-01

## Pergunta

Uma compra sistemática de straddle BTC, com entrada executável no primeiro dia
de cada trimestre e vencimento próximo de 14 DTE, apresentou vantagem suficiente
para justificar a aquisição dos arquivos históricos de Greeks?

## Desenho pré-declarado

- 26 entradas trimestrais de 2020-04-01 a 2026-07-01, às 12:00 UTC.
- Primeiro o vencimento mais próximo de 14 DTE dentro de 7--30 dias; depois o
  strike ATM com call e put executáveis.
- Compra das duas pernas no ask Tardis com quote de no máximo cinco minutos.
- Hold até o vencimento e payoff pelo delivery price oficial da Deribit.
- Fees de entrada e liquidação explícitas.
- Sem hedge; nenhum `options_chain` adicional foi baixado.

Configurações: `config/experiment.tardis-carry-quarterly-v1.json` e
`config/experiment.tardis-carry-quarterly-v1-min-size.json`.

## Resultado principal — um contrato por perna

- 26 datas tentadas; 22 executáveis e quatro recusadas por tamanho insuficiente
  no ask do par ATM.
- 7 de 22 resultados positivos (31,82%).
- P&L total nas 22 posições não sobrepostas: -0,177873 BTC.
- P&L médio: -0,008085 BTC; mediano: -0,009361 BTC.
- Retorno médio sobre o prêmio: -7,62%; mediano: -9,91%.
- Melhor P&L: +0,111339 BTC (2021-01-01).
- Pior P&L: -0,087906 BTC (2022-10-01).

## Sensibilidade de cobertura — 0,1 contrato

O tamanho mínimo foi aplicado somente às quatro datas recusadas, após ser
registrado em configuração separada. Todas passaram. Com as 26 observações
comparadas por retorno sobre prêmio:

- 8 de 26 positivas (30,77%).
- retorno médio sobre o prêmio: -8,45%; mediano: -12,68%;
- melhor retorno: +139,78%; pior retorno: -93,52%.

O tamanho muda a cobertura e o P&L absoluto, mas não o retorno percentual dos
casos já executáveis, pois payoff e custos usados aqui escalam linearmente.

## Decisão do gate

**Reprovada a compra sistemática de straddle como estratégia standalone neste
piloto.** As perdas foram frequentes e a média permaneceu negativa, apesar de
poucas caudas positivas grandes. Isso é coerente com a hipótese de prêmio de
variância, mas não aprova venda de volatilidade: short straddle adiciona margem,
liquidação e perdas de cauda que este experimento long-only não modela.

Não baixar os `options_chain` de todos os trimestres neste estágio. Os quotes
brutos ocupam 6,7 GB; o resultado completo versionado ocupa poucos kilobytes.

## Limitações

- amostra pequena, esparsa e restrita ao primeiro dia de cada trimestre;
- nenhuma separação treino/validação/holdout;
- estratégia sem hedge, portanto mistura volatilidade e exposição direcional
  desenvolvida ao longo do caminho;
- fees constantes, sem auditoria de cada tabela histórica da venue;
- retorno sobre prêmio não representa retorno sobre capital de uma estratégia
  short-vol nem considera margem de venda.
