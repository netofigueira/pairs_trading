# Distribuição de perda por block bootstrap — short straddle BTC

Data: 2026-09-02
Estado: experimento executado; resultado é **distribuição de perda condicional a
um trade**, não probabilidade de ruína; nada aprovado para dinheiro real.

## Contexto e por que este teste

O teste de gap anterior era *stress adversarial*: forçava um evento de cauda em
100% das posições, no pior dia escolhido ex post. Isso reprova o stop diário como
proteção de cauda, mas não pode refutar estatisticamente a hipótese short-vol,
porque não usa a frequência histórica real do evento.

Este experimento troca o adversarial por frequência. Reamostra a trajetória
conjunta `(retorno BTC, variação de DVOL)` do próprio histórico via **moving-block
bootstrap**, preservando clustering de volatilidade e a correlação retorno–vol
(observada em cerca de -0,14 no período). Cada trajetória simulada tem o **DTE
real** do contrato observado, é reprecificada diariamente por Black-76 inverso e
recomprada no ask sintético.

Cuidados de método (evitam auto-engano):

- **Não somar gaps à trajetória real** — isso duplicaria retornos. A trajetória é
  reamostrada por inteiro, não somada.
- **Nomeação honesta.** O resultado é distribuição de perda por trade e
  probabilidade de exceder múltiplos do crédito. **Não é probabilidade de ruína**:
  ruína exige capital, sizing, margem e barreira de liquidação, fora do escopo
  desta rodada.
- **Regras pré-declaradas.** As três regras de saída e o hold-to-expiry são as
  mesmas do envelope; nenhuma regra nova foi criada após ver o resultado.

## Desenho

- Fonte: `build_joint_history` alinha close diário de BTC-PERPETUAL e DVOL
  (1.986 dias conjuntos, 2021-03 a 2026-08).
- `sample_block_paths`: blocos de 4 dias, com reposição, concatenados até o DTE.
- `simulate_trade_losses`: Black-76 inverso vetorizado; casa com o preço escalar
  de `inverse_options.inverse_option_price` até 1e-12, incluindo intrínseco em
  `t=0`. Taxa Deribit `min(0,0003; 0,125·prêmio)`.
- 10.000 trajetórias por entrada, seed fixa (`20260902`) e offset estável
  `YYYYMMDD`, spreads P50/P90/P95. Duas execuções em processos separados
  produziram o mesmo SHA-256 do artefato.
- Cobertura: 22/26. As 4 anteriores ao DVOL seguem excluídas, como no envelope.

```bash
.venv/bin/python scripts/run_short_straddle_bootstrap.py \
  --n-paths 10000 --block-size 4 --spread-scenarios p50,p90,p95 \
  --output artifacts/short-straddle-bootstrap-v1.json
```

## Resultado (spread P50; perdas em múltiplos do crédito)

| Regra | Média/crédito | P(perda) | P(>1×) | P(>2×) | P(>5×) | ES95 | ES99 | Pior |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hold até vencimento | -14,31% | 43,78% | 16,01% | 5,86% | 0,385% | 3,25× | 5,12× | 16,78× |
| tp25 / stop2 / 3 DTE | -9,88% | 32,04% | 16,48% | 1,43% | 0,00045% | 1,89× | 2,61× | 5,02× |
| tp50 / stop1,5 / 7 DTE | -8,88% | 39,47% | 4,96% | 0,37% | 0% | 1,38× | 2,02× | 4,20× |
| tp50 / stop2 / 3 DTE | -11,33% | 40,48% | 18,22% | 1,60% | 0,00045% | 1,94× | 2,65× | 5,02× |

P90 e P95 deslocam tudo levemente para pior, sem mudar a ordenação — o spread não
é o fator dominante. Média de P&L é negativa em todas as regras.

## Leitura

Três fatos novos, que o ponto-estimativa e o gap não mostravam:

1. **A distribuição por trade é de média negativa em todas as regras.** Sob
   frequência histórica reamostrada, não há prêmio líquido a capturar nesta
   configuração (ATM, 14 DTE, marcação diária) — o resultado positivo do
   envelope no cenário central era específico das 22 datas observadas, não uma
   propriedade da distribuição.

2. **O stop corta a cauda extrema mas não o prejuízo esperado.** As regras com
   stop cortam a pior perda de 16,78× para 4,20--5,02× e tornam perdas acima de
   5× quase ausentes nesta simulação, porém não tornam a média positiva. É o padrão clássico: o stop
   converte poucas perdas catastróficas em muitas perdas médias.

3. **A cauda do hold é grave.** 5,86% de chance de perder mais que 2× o crédito e
   ES99 de 5,12× num único trade — inaceitável sem dimensionamento e margem
   explícitos.

## Conclusão honesta e escopo

Esta rodada **enfraquece a hipótese short-vol** muito mais do que o gap: sob a
frequência histórica reamostrada, a média por trade é negativa e a cauda é
pesada. Ainda assim, o que está medido é distribuição por trade, não ruína.

O que falta antes de uma decisão final de reprovar/aprovar:

- Margem Deribit e barreira de liquidação (converte perda de recompra em
  liquidação forçada, que pode ser pior que o stop).
- Sequência de trades com capital e sizing para estimar drawdown e, aí sim,
  ruína.
- Sensibilidade a block-size (3–5 dias) e a strikes não-ATM.

Não faz sentido procurar novas regras de take-profit: seria overfit sobre 22
datas. O próximo experimento útil é margem + sequência de trades.

Artefato: `artifacts/short-straddle-bootstrap-v1.json` (cerca de 207 KB, com
resumos agregados e por entrada).
