# Arquitetura da plataforma de volatilidade — do tape ao dashboard vivo

Data: 2026-09-02
Estado: desenho aprovado; nenhum código de produção nesta rodada. Nada
autorizado para dinheiro real; a plataforma permanece read-only e sem
credenciais de exchange (herda as fronteiras de
[plataforma operacional](2026-08-31-plataforma-operacional.md)).

## Motivação

A carga histórica do tape público da Deribit em `market.option_trade`
(migration `0002`, hypertable por `traded_at`, com `iv`, `mark_price`,
`index_price`, direção e proveniência idempotente por arquivo) transforma em
**query viva** o dado que os estudos de volatilidade hoje consomem offline e
publicam como JSON estático. Ao mesmo tempo, a Fase 5 (paper no holdout,
início 2026-09-03) não tem nenhuma tela que a observe. Este documento desenha
como fechar esse loop sem violar os invariantes de método.

## Princípio: agregação vive no banco, modelo vive no Python

A pergunta "para reconstruir os JSON, modelamos os cálculos no banco?" tem duas
respostas, conforme a natureza do cálculo.

### Grupo A — agregação pura sobre o tape → banco

Cobertura, volume por strike/expiry, superfície de IV dos prints reais e a
marcação do straddle ATM são `SELECT ... GROUP BY` sobre `market.option_trade`.

- Para séries recomputáveis por janela de tempo (volume diário, IV média por
  bucket), usar **continuous aggregate** do TimescaleDB
  (`CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous)`): refresca
  incrementalmente apenas os chunks novos, sem recomputar o histórico. É o
  análogo nativo do que um job manual faria, e é o desenho pensado para
  hypertable.
- Para o que é leve e precisa refletir o tape do instante (marcação do dia,
  último print por instrumento), uma **view comum** basta.

Nenhuma dessas exige job externo: o banco se mantém sozinho.

### Grupo B — modelo estatístico → job Python que persiste em `research.*`

GARCH, EWMA, forecast de volatilidade, HAC/Newey-West e bootstrap são
econometria (`arch`, `statsmodels`). Não são expressáveis em SQL de forma
auditável, e reimplementá-los em plpgsql seria criar uma segunda fonte de
verdade sem teste. O fluxo correto:

```
market.option_trade (DB)
        │  SQL de leitura
        ▼
  job Python (quant_pairs)  ── usa o código já testado
        │  grava uma linha por run
        ▼
  research.*  (tabela versionada: run_id, as_of, config, code_version, metrics)
        │
        ▼
  API lê a tabela  ──►  dashboard
```

O cálculo continua em Python; muda o destino: em vez de sobrescrever um `.json`
em `artifacts/`, cada execução vira uma **linha versionada**. Ganhos: histórico
(cada run é uma observação, não um overwrite), reprodutibilidade explícita
(`config` + `code_version` na própria linha) e consulta viva. O invariante
"artefatos versionados e reproduzíveis, SHA estável" é preservado — o artefato
deixa de ser arquivo e passa a ser registro.

### Regra de bolso

| Cálculo | Onde | Mecanismo |
|---|---|---|
| Cobertura / volume / IV agregada | Banco | continuous aggregate |
| Marcação straddle ATM do dia | Banco | view comum |
| Forecast (GARCH/EWMA) | Python → banco | job → `research.volatility_forecast` |
| Gate de regime | Python → banco | job → `research.volatility_regime` |
| HAC / bootstrap do book | Python → banco | job → `research.book_inference` |

## Esboço de schema `research.*` (a detalhar na implementação)

Contrato comum a todas as tabelas de resultado de modelo:

- `run_id` (PK), `as_of` (data de referência do cálculo), `created_at`
- `config` JSONB (parâmetros pré-declarados), `code_version` (git SHA)
- colunas de métrica específicas do modelo
- `research.*` é append-only; nada sobrescreve run anterior (mesmo invariante
  do holdout: dado novo não reescreve regra/resultado antigo).

Tabelas previstas: `volatility_forecast`, `volatility_regime`,
`book_inference` (HAC/bootstrap), e para a Fase 5 uma família de paper:
`paper_mark` (marcação diária), `paper_position` (posições virtuais vs. teto),
`paper_pnl` (P&L realizado/não realizado). A tabela `execution.order_intent`
já existe como contrato futuro restrito a `paper`.

## Orquestração: n8n comanda, o Python computa

Decisão (refina o README de n8n): **n8n é orquestrador, não computação.** A
computação de coleta já existe testada em Python (`deribit.py`,
`load_deribit_history_trades.py`); reescrevê-la em nós JS duplicaria a fonte de
verdade sem teste, CI ou versionamento. O padrão escolhido:

```
n8n (schedule + retry + alerta)
   │  HTTP POST
   ▼
quant-collector (endpoint, ex. POST /ingest/deribit-tape?since=...)
   │  roda o código Python testado
   ▼
market.option_trade  (ON CONFLICT, cursor = banco é fonte de verdade)
```

n8n orquestra a **cadeia diária inteira** da Fase 5, cada passo sendo uma
chamada a um endpoint do collector:

```
coletar tape → marcar straddle ATM → rodar forecast → gravar research.*
            → checar teto 0,5/BTC e margem → alertar
```

Isso satisfaz o README ("n8n para agendas, alertas e notificações, não para
sinais críticos") e a visão de n8n como orquestrador do pipeline: ele é o
relógio e o painel de alarme; o cálculo mora onde é testável.

Alternativas consideradas e por que não: (a) tudo no n8n com HTTP+INSERT nos
nós — aceitável para candles (uma linha, sem paginação), ruim para tape
(paginação, normalização de IV/greeks, volume); (b) n8n via Execute Command no
container — acopla n8n ao filesystem da imagem. A opção HTTP mantém o
desacoplamento.

## Lacuna crítica a resolver antes da Fase 5 render dado

A ingestão contínua do tape Deribit **não existe hoje**. `market.option_trade`
veio de carga histórica pontual; os dois workflows n8n da VM estão inativos
(`active: None`) e cobrem apenas candles Binance. Sem tape fresco diário
chegando, a Fase 5 não tem o que marcar. O primeiro entregável operacional é,
portanto, o endpoint de ingestão diária do tape no collector + o workflow n8n
que o agenda — não uma tela.

## Sequência sugerida (a confirmar)

1. Endpoint de ingestão diária do tape no collector + workflow n8n que agenda e
   alerta (destrava a Fase 5).
2. Continuous aggregates de cobertura/volume/IV + view de marcação do dia.
3. Painel `/paper` lendo essas views + `research.paper_*` (observa a Fase 5).
4. Migração incremental dos jobs de modelo (forecast, regime, HAC) de JSON para
   `research.*`, começando pelo forecast.

## Fronteiras mantidas

- Read-only, sem credenciais de exchange, sem envio de ordem nesta fase.
- Holdout fisicamente protegido; `research.*` é append-only.
- Regras pré-declaradas antes do resultado; `config` + `code_version` em cada
  run.
