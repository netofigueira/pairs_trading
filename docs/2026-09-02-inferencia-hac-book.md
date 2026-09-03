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

## Decisão

- Estado honesto: **evidência promissora** — edge positivo pequeno, dependente
  de regime, t≈2,2-2,3 no teste eficiente com seleção causal, porém teste
  escolhido a posteriori, marcação sintética e sizing diferente da política do
  paper. Justifica a Fase 5; não prova a estratégia nem autoriza capital.
- Fase 5 (paper no holdout, iniciado 2026-09-03, teto 0,5 contrato/BTC,
  execução postada) passa a ser o critério de promoção. Nada mais será ajustado
  no histórico: qualquer novo teste retrospectivo nesta trilha precisa de
  pré-declaração explícita.

## Limitações

- Mesmas aproximações de marcação do backtest (marks sintéticos diários, index
  como forward, funding constante, prints ≠ fills nossos).
- HAC corrige dependência até o lag escolhido; choques de regime mais longos
  que ~1 mês não são capturados.
- Sem teto de posições nem margem nesta série (mecânica testada à parte nas
  Fases 3 e no motor rolling).
