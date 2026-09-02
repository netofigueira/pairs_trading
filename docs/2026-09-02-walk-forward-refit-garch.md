# Walk-forward e frequência de refit do GARCH

Data: 2026-09-02  
Decisão: manter reestimação dos parâmetros a cada 30 dias.

## O que já era walk-forward

O forecast V1 já avança a origem diariamente em expanding-window: em cada data,
usa somente retornos disponíveis até aquele instante e prevê os próximos 14 ou
30 dias. A otimização dos parâmetros GARCH, porém, era executada a cada 30 dias;
entre refits, o estado condicional era atualizado com cada novo retorno.

O challenger desta rodada reestima todo o GARCH a cada origem. Também incluímos
o alvo one-step-ahead lembrado na discussão. O horizonte de um dia é diagnóstico
porque o retorno quadrático diário é um proxy muito ruidoso de variância. A
escolha operacional continua baseada em 14 dias, próximo do DTE negociado.

## Protocolo

- Janela de treino expanding, mínimo de 365 dias.
- Mesma especificação GARCH(1,1), zero-mean, e mesma correção causal de viés.
- Challenger: refit a cada dia; incumbent: refit a cada 30 dias.
- Comparações estritamente pareadas.
- Horizonte 14d avaliado em targets não sobrepostos.
- Promoção exige QLIKE menor, MSE de variância não pior e IC95% do diferencial
  de QLIKE inteiramente abaixo de zero.
- Dados a partir de 2026-09-03 não foram tocados e seguem como holdout.

## Resultado

| Horizonte | Cadência | N | MSE variância | QLIKE | Correlação RV |
|---|---|---:|---:|---:|---:|
| 1d | refit diário | 1.674 | 0,31522 | -0,42533 | 0,211 |
| 1d | refit 30d | 1.674 | 0,31858 | -0,42217 | 0,188 |
| 14d | refit diário | 117 | 0,05592 | -0,37743 | 0,289 |
| 14d | refit 30d | 117 | 0,05646 | **-0,37772** | 0,272 |

No one-step, o diferencial de QLIKE do refit diário contra o mensal foi
-0,00316, IC95% [-0,01553; 0,00922], p=0,617. Em 14 dias, foi +0,00029,
IC95% [-0,01041; 0,01100], p=0,957. O sinal muda entre os horizontes e ambos os
intervalos incluem zero: não há evidência de ganho pela reestimação diária.

## Consequência para a plataforma

Continuamos produzindo uma nova previsão todos os dias. Apenas a otimização dos
parâmetros permanece mensal, o que reduz custo e variação de estimação sem perda
detectável. O experimento completo leva cerca de 146 segundos nesta máquina; o
runner operacional atual continua em aproximadamente 12 segundos.

Reprodução:

```bash
.venv/bin/python scripts/compare_garch_refit_cadence.py
```

Artefato: `artifacts/garch-refit-cadence-v1.json`.
