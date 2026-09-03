# Inferência HAC no P&L diário do book — tape 2021-2026

Data: 2026-09-02 (revisado no mesmo dia: rerodado após a correção causal da
seleção de prints — somente `traded_at <= 12:00`, último print por perna, idade
máxima 2h)
Estado: executado; evidência promissora com ressalvas fortes abaixo; nada
aprovado para dinheiro real.
Antecedente: [backtest completo](2026-09-02-backtest-tape-completo.md).

## Por que este teste

O backtest completo deixou um impasse: 545 trades sobrepostos com média
positiva, mas t inválido; subamostras independentes sem poder (t≤0,76 jogando
fora 90% do dado). A solução padrão é testar a **média do P&L diário do book
agregado** (posições sobrepostas somam) com erros Newey-West, que corrigem a
dependência serial da sobreposição sem descartar observações.

```bash
source deploy/timescaledb/quant_ingest.env
.venv/bin/python scripts/run_tape_book_hac.py
```

Artefato: `artifacts/tape-book-hac-v1.json`. Dias flat contam como zero: o
teste é da média diária incondicional, que é o que o capital vivencia.

## Resultado

| Métrica | Valor |
|---|---:|
| Trades no book | 520 (0,1 contrato cada, sem teto) |
| Dias na série | 1.606 |
| Máximo de posições simultâneas | 19 |
| P&L total | +0,3718 BTC (bate com o pooled do backtest) |
| Média diária | +0,000232 BTC |
| **t Newey-West** (lags 14 / 21 / 28) | **2,20 / 2,25 / 2,32** |

## Leitura

1. **O t cruza 2 de forma estável nos três lags, agora com seleção
   estritamente causal.** A leitura correta continua modesta: as entradas são
   prints de terceiros com haircut de spread, e marcação e hedge seguem
   sintéticos; isto é um envelope sobre preços reais de entrada, não um
   backtest de fills. A média diária de +0,000232 BTC equivale a ~8,5% ao ano
   sobre 1 BTC nesse sizing sem teto.
2. **Ressalva de método, dita com todas as letras:** este teste não foi
   pré-declarado; ele foi escolhido depois de vermos o impasse do backtest
   (embora seja o teste canônico para esse impasse, e nenhuma regra da
   estratégia tenha sido tocada). t≈2 com escolha de teste a posteriori vale
   menos que t≈2 pré-declarado. O holdout vivo continua sendo o único juiz
   totalmente limpo.
3. **Concentração continua sendo o risco:** 2022 responde por +0,14 do total e
   2025 é zero. O edge é condicional a regime de vol alta; o book passa anos
   ganhando pouco e concentra o resultado em janelas.
4. **Este teste não mede a política que irá ao paper.** O book sem teto chegou
   a 19 posições (1,9 contratos brutos); a política da Fase 5 tem teto de 0,5
   contrato/BTC e recusaria parte das rajadas, exatamente as janelas que
   concentram o retorno. O rerun com teto no motor rolling é pré-requisito para
   citar este número como expectativa do paper.

## Sensibilidade de lags e o teste pré-declarado com teto

Ressalvas do review incorporadas em rerodadas:

**Lags estendidos (book sem teto):** t continua estável e até sobe com lags
maiores — 2,20 (14), 2,25 (21), 2,32 (28), 2,34 (30), 2,40 (42), 2,52 (60).
A dependência serial além de um mês não desfaz o resultado sem teto.

**Teto operacional (experimento pré-declarado
`config/experiment.tape-book-cap-v1.json`, commitado antes da execução):**
entradas recusadas quando os contratos brutos abertos excederiam 0,5 (capital
estático de 1 BTC, uma aproximação do teto dinâmico). Se a equity cair, o
cap estático permite mais risco; se subir, bloqueia mais. Resultado:

| Métrica | Sem teto | Com teto 0,5 |
|---|---:|---:|
| Trades aceitos | 520 | 264 (256 recusados) |
| P&L total (BTC) | +0,372 | +0,095 |
| Média diária (BTC) | +0,000232 | +0,000059 (~2,1% a.a.) |
| t NW (14→60) | 2,20→2,52 | **1,05→1,21** |

**O critério pré-declarado (t≥2 em todos os lags) FALHOU.** O teto recusa
metade das entradas, justamente as rajadas que concentram o retorno, e o que
sobra não é estatisticamente distinguível de zero. Conclusão honesta: o edge
histórico demonstrável vive na frequência plena das rajadas, que o sizing
operacional atual não consegue capturar. Ou o teto sobe com capital/margem
dimensionados para as rajadas (pesquisa da Fase 3 revisitada, com
pré-declaração), ou a expectativa do paper é a linha com teto: positiva,
pequena e não provada.

Artefato: `artifacts/tape-book-hac-cap-v1.json`.

## Decisão

- Estado honesto: **promissora sem teto (t=2,2-2,5, seleção causal), não
  demonstrada no sizing operacional (t≈1,1 com teto de 0,5 contrato/BTC,
  critério pré-declarado falhou)**. Marcação sintética e prints de terceiros
  em ambos os casos. Justifica a Fase 5 como árbitro; não autoriza capital.
- Fase 5 (paper no holdout, iniciado 2026-09-03, teto 0,5 contrato/BTC,
  execução postada) passa a ser o critério de promoção. Nada mais será ajustado
  no histórico: qualquer novo teste retrospectivo nesta trilha precisa de
  pré-declaração explícita.

## Limitações

- Mesmas aproximações de marcação do backtest (marks sintéticos diários, index
  como forward, funding constante, prints ≠ fills nossos).
- HAC corrige dependência até o lag escolhido; choques de regime mais longos
  que ~1 mês não são capturados.
- O artefato sem teto não inclui margem; o experimento capped limita contratos
  brutos estáticos, mas ainda não reproduz o cap dinâmico por equity e margem
  da política paper.
