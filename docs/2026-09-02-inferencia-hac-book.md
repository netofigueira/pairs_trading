# Inferência HAC no P&L diário do book — tape 2021-2026

Data: 2026-09-02
Estado: executado; primeiro resultado com preços reais a cruzar t=2; ressalvas
importantes abaixo; nada aprovado para dinheiro real.
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
| Trades no book | 545 (0,1 contrato cada, sem teto) |
| Dias na série | 1.620 |
| Máximo de posições simultâneas | 19 |
| P&L total | +0,3805 BTC (bate com o pooled do backtest) |
| Média diária | +0,000235 BTC |
| **t Newey-West** (lags 14 / 21 / 28) | **2,00 / 2,05 / 2,13** |

## Leitura

1. **Primeira vez que a evidência com preços reais cruza o limiar convencional
   de significância**, e de forma estável nos três lags. A média diária de
   +0,000235 BTC com 19 posições máximas equivale a ~8,6% ao ano sobre 1 BTC de
   capital nesse sizing sem teto.
2. **Ressalva de método, dita com todas as letras:** este teste não foi
   pré-declarado; ele foi escolhido depois de vermos o impasse do backtest
   (embora seja o teste canônico para esse impasse, e nenhuma regra da
   estratégia tenha sido tocada). t≈2 com escolha de teste a posteriori vale
   menos que t≈2 pré-declarado. O holdout vivo continua sendo o único juiz
   totalmente limpo.
3. **Concentração continua sendo o risco:** 2022 responde por +0,14 do total e
   2025 é zero. O edge é condicional a regime de vol alta; o book passa anos
   ganhando pouco e concentra o resultado em janelas.
4. Book sem teto chegou a 19 posições (1,9 contratos brutos); com o teto de 0,5
   contrato/BTC da Fase 3, parte das rajadas seria recusada, o retorno cai e o
   risco também. O rerun com teto pertence ao motor rolling (integração já
   mapeada).

## Decisão

- Evidência histórica final desta rodada: **edge positivo pequeno, dependente
  de regime, t≈2 no teste eficiente, não pré-declarado**. Suficiente para
  justificar a Fase 5 com convicção; insuficiente para capital real.
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
