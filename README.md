# tech-challenge-fase-1

Estrutura inicial de pastas do projeto.

## Estrutura

```text
tech-challenge-fase-1/
├── src/
│   ├── data/
│   ├── features/
│   ├── eda/
│   ├── training/
│   ├── inference/
│   ├── schemas/
│   ├── api/
│   └── pipelines/
├── data/
├── models/
├── tests/
├── notebooks/
└── docs/
```

## Regra de execução (pipelines)

- Não gerar artefatos finais (modelos treinados, bases finais, tracking de experimento) a partir de notebooks.
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

## Como usar o MLflow

### Iniciar

```bash
docker-compose -f docker/docker-compose.yml up --build
```

### Acessar

- MLflow UI: http://localhost:5000
