# Tech Challenge — Fase 1

Pipeline end-to-end de ML para previsão de churn em telecomunicações — MLP com PyTorch, baselines Scikit-Learn, rastreamento com MLflow e API de inferência com FastAPI. Tech Challenge Fase 1 · PÓS TECH FIAP.

## Contexto do problema

Em telecom, churn representa o cancelamento de clientes. Antecipar esse comportamento ajuda o negócio a:

- reduzir perda de receita;
- priorizar ações de retenção;
- melhorar a experiência do cliente com decisões orientadas por dados.

## Estrutura do repositório

```text
tech-challenge-fase-1/
├── src/
│   ├── data/       # Importação e parse de dados
│   ├── features/   # Limpeza, feature engineering e seleção
│   ├── eda/        # Funções auxiliares para exploração
│   ├── training/   # Split, treino e tuning
│   ├── inference/  # Predição desacoplada da camada web
│   ├── schemas/    # Schemas de validação (Pandera/Pydantic)
│   ├── api/        # Camada FastAPI
│   └── pipelines/  # Scripts de execução/orquestração
├── data/           # Dados do projeto
├── models/         # Artefatos de modelo
├── tests/          # Testes automatizados
├── notebooks/      # Exploração e análises
└── docs/           # Documentação complementar
```

### Regra de pipelines

- Notebooks são para exploração.
- Artefatos finais devem ser gerados por scripts parametrizáveis em `src/pipelines/`, executados via terminal.

## Sincronização do Ambiente

Para sincronizar o ambiente incluindo pacotes de desenolvimento utilizando o uv, use:

```bash
uv sync
```

Para sincronizar sem dependências de desenolvimento, útil para CI e contâiners docker, use:
```bash
uv sync --no-dev
```
- Artefatos finais (modelos, bases finais, tracking) devem ser gerados por scripts em `src/pipelines/` via terminal.

## Instalação (setup de ambiente)

> 🚧 **Placeholder:** setup ainda em construção e será formalizado nas issues de configuração do projeto.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
```

## Sincronização do Ambiente

Para sincronizar o ambiente incluindo pacotes de desenvolvimento utilizando o uv, use:

```bash
uv sync
```

Para sincronizar sem dependências de desenvolvimento, útil para CI e containers Docker, use:

```bash
uv sync --no-dev
```

## Stack e versões definidas

Referência consolidada com base nas decisões já registradas nas PRs em andamento.

- **Python:** `>=3.13`
- **Dependências base:**
  - `fastapi>=0.135.2`
  - `pandas>=2.0.0,<3.0.0`
  - `scikit-learn>=1.8.0`
  - `mlflow>=2.0.0`
- **Dependências de desenvolvimento:**
  - `ruff>=0.15.8`
  - `mypy>=1.19.1`
  - `pytest>=9.0.2`
  - `pytest-cov>=7.1.0`
  - `pre-commit>=4.5.1`

### Qualidade e testes (padrões iniciais)

- lint e formatação com **Ruff**;
- tipagem estática com **MyPy**;
- testes com **Pytest**;
- cobertura mínima alvo em `src`: **80%**.

## Configuração de Variáveis de Ambiente

Para configuração local, copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Preencha no `.env` as variáveis essenciais para começar:

```bash
ENVIRONMENT=development
DEBUG=true
API_TOKEN=seu_token_seguro_aqui
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=tech-challenge-fase-1
```

> 🔒 Segurança: nunca versionar o arquivo `.env` com credenciais reais.

## Comandos básicos

Comandos disponíveis no momento (validação local):

```bash
# verificar branch atual
git branch --show-current

# ver mudanças locais
git status
```

Comandos planejados (placeholder):

```bash
make lint
make test
```

> 🚧 **Placeholder:** os comandos `make` serão ativados quando `Makefile` e pipeline de qualidade estiverem formalizados.

## Roadmap / Próximos passos

Resumo conciso do cronograma de execução (issue de controle):

- **Etapa 0 (26/03 → 06/04):** setup de repositório e base técnica;
- **Etapa 1 (07/04 → 20/04):** entendimento dos dados (EDA) e baselines;
- **Etapa 2 (21/04 → 30/04):** modelagem com MLP (PyTorch);
- **Etapa 3 (24/04 → 04/05):** engenharia (API, testes, integração);
- **Etapa 4 (01/05 → 05/05):** documentação final e entrega.

Marcos críticos:

- **04/05/2026:** gravação do vídeo STAR;
- **05/05/2026:** entrega final.

## Boas práticas e módulos previstos (baseado na Issue #3)

> ✅ **Status:** esta seção descreve direcionadores e componentes **previstos** para evolução do projeto.

Boas práticas de engenharia previstas:

- TDD desde o início dos módulos críticos;
- padronização de lint/format (ruff);
- tipagem estática gradual (mypy em CI + pyright no IDE);
- organização por pipeline reproduzível (ambiente declarativo e containerização);
- documentação contínua de decisões técnicas e do fluxo de execução.

Módulos e componentes previstos:

- **Qualidade e automação:** pre-commit, testes automatizados e cobertura;
- **ML pipeline:** `sklearn.Pipeline`, rastreamento com MLflow e serialização de artefatos;
- **API e contratos:** FastAPI + Pydantic v2 para inferência e validação de entradas;
- **Operação:** Docker multi-stage, variáveis de ambiente e CI com GitHub Actions;
- **Suporte ao ciclo de dados:** uso opcional de DVC e organização de notebooks (com possibilidade de jupytext).

## Autores

- Eduardo Pereira (@eduardonunesp)
- Bruno Fructuoso (@BrunoFructuoso)
- Fernando Failla Foschiani (@FernandoFailla)
- Rafael (@gabipasse)
- Ygor Martinelli (@ygormartinelli)

## Links úteis

- Roadmap e controle de execução: https://github.com/G13-MLE/tech-challenge-fase-1/issues/7
- Boas práticas de desenvolvimento (Issue #3): https://github.com/G13-MLE/tech-challenge-fase-1/issues/3
- Kickoff Planning (Miro): https://miro.com/app/board/uXjVGt4ginw=/
