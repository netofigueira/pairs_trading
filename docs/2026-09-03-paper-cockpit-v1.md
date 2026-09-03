# Cockpit de paper trading de volatilidade — V1

## Objetivo

Mostrar a operação do holdout selado como ela realmente ocorreu: qualidade do
dado, decisão short/flat, risco aberto e P&L marcado. A tela nunca recalcula o
modelo nem cria uma decisão.

## Fonte de verdade

Cada execução diária grava somente novas linhas em `research.paper_run`,
`paper_decision`, `paper_position` e `paper_mark`. Toda decisão bloqueada por
ausência de quote executável ou forecast também é registrada. Isso diferencia
"flat por regra" de "não operou porque o dado não era suficiente".

## Painel `/paper`

1. **Agora:** último run, atraso de input, ação, razão e IV/forecast usados.
2. **Risco aberto:** cada perna, contratos, preço de entrada e última margem
   estimada.
3. **P&L:** realizado e não realizado, sem esconder custos ou dias flat.
4. **Auditoria:** `as_of`, timestamps dos inputs, configuração e SHA do código.

Na VM, o painel é servido apenas dentro da tailnet em
`https://free-tier-a1.tail7f5470.ts.net:8443/paper`.

## Invariantes

- A regra congelada só permite `short` ou `flat`; long permanece fora.
- Ausência ou atraso de quote executável produz `blocked`, não um preço
  sintético nem uma operação inferida.
- O dashboard é leitura. n8n apenas orquestra os jobs Python que escrevem os
  registros append-only.
