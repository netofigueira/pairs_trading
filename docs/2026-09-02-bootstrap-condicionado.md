# Bootstrap condicionado a regime — o null que a estratégia merece

Data: 2026-09-02
Estado: experimento executado com desenho pré-declarado
(`config/experiment.conditioned-bootstrap-v1.json`); critério de sucesso
atendido; nada aprovado para dinheiro real.
Plano: [docs/2026-09-02-plano-pipeline-vol.md](2026-09-02-plano-pipeline-vol.md)

## Por que este teste

A Fase 3 mostrou que o bootstrap incondicional reprova a estratégia por
construção: ele reamostra a vol média do histórico (~54%) contra toda entrada,
apagando a informação que o gate GARCH alega ler. O teste justo para a
afirmação "o forecast distingue regimes" é um null que **preserva o regime
observado na entrada** e deixa o resto aleatório.

## Regra pré-declarada (congelada antes de rodar)

Mesma máquina da Fase 3 (book hedgeado, bid cruzado, funding médio realizado,
margem padrão), com uma mudança no sorteio: **um bloco só pode começar em dia
cujo DVOL disponível na véspera esteja a ≤10 pontos do DVOL da entrada**
(alarga de 5 em 5 pontos se houver menos de 100 inícios elegíveis; a tolerância
final fica registrada no artefato — nas 22 entradas, nenhuma precisou alargar).
Critério de sucesso declarado: média pooled positiva das entradas `short` do
gate congelado. Falha devolveria a estratégia à pesquisa de forecast.

```bash
.venv/bin/python scripts/run_conditioned_bootstrap.py
```

Artefato: `artifacts/conditioned-bootstrap-v1.json`.

## Resultado

| Pool | Trajetórias | Média (BTC/contrato) | P(perda) |
|---|---:|---:|---:|
| gate `short` (6 entradas) | 12.000 | **+0,00139** | 42,7% |
| gate `long` (12 entradas) | 24.000 | -0,00449 | 50,8% |
| todas (22) | 44.000 | -0,00124 | 46,7% |

O critério passou. E a estrutura é a esperada se o gate tem conteúdo: short
positivo, long negativo (consistente com o realizado, em que o lado long também
perdia), agregado próximo de zero.

Por entrada short: 4/6 positivas sob o null condicionado (2024-01 contribui
mais, +0,0169); as duas negativas (2023-10 e 2026-07) são justamente as em que
a vol da amostra condicionada ainda ficou **acima** da IV de entrada (0,373 vs
0,315; 0,448 vs 0,419). Isso revela um conservadorismo estrutural do desenho:
condicionar pelo **nível do DVOL** herda o prêmio DVOL>RV (~10 pontos no P0),
então o null simula vol futura mais alta do que a que historicamente se
realizou naquele regime. Mesmo assim o pool short é positivo.

## Sizing sobre o pool short (a política de referência)

Kelly no pool short: 1,46 contratos/BTC; meia-Kelly 0,73. Grade:

| Contratos/BTC | P(ruína) | P(liquidação) | P(DD>30%) | Mediana terminal | P05 |
|---:|---:|---:|---:|---:|---:|
| 0,25 | 0 | 0 | 0 | 1,009 | 0,944 |
| 0,5 | 0 | 0 | 0 | 1,019 | 0,889 |
| 0,73 (½K) | 0 | 0 | 0,4% | 1,027 | 0,837 |
| 1,0 | 0 | 0,2% | 2,8% | 1,033 | 0,794 |
| 1,46 (K) | 0,4% | 2,6% | 16,7% | 1,034 | 0,691 |
| 2,0 | 4,1% | 26,6% | 40,3% | 1,019 | 0,544 |

A mediana terminal quase não melhora acima de 1 contrato/BTC enquanto a cauda
piora rápido — o clássico platô do Kelly. O teto de 0,5 contrato/BTC declarado
na Fase 3 se mantém: fica bem abaixo da meia-Kelly, com P05 de 0,889 e
drawdown>30% nulo na simulação.

## Decisão

1. O histórico agora contém evidência (fraca, 6 entradas, mas sob null
   pré-declarado e conservador) de que o lado short do gate reflete skill do
   forecast, não carry incondicional.
2. **Seguir para a Fase 5**: paper trading da política short/flat hedgeada no
   holdout vivo (início 2026-09-03), execução postada com piso cruzado, teto
   de 0,5 contrato/BTC-equity. O holdout continua sendo o árbitro final.
3. O lado long do gate permanece reprovado em todos os testes; segue fora.

## Limitações declaradas

- 6 entradas short: as 12.000 trajetórias compartilham 6 pontos de partida;
  a diversidade de entrada é pequena e o resultado é dominável por uma data.
- Condicionamento só no início de cada bloco; a dinâmica intra-bloco pode sair
  do regime (é desejável: permite o choque de vol dentro do trade).
- Mesmas aproximações da Fase 3 (forward como proxy do spot, funding constante,
  margem padrão, penalidade de liquidação crua).
