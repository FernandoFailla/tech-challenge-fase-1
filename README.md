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

## Comandos Disponíveis (Makefile)

O projeto inclui um Makefile com comandos essenciais para desenvolvimento:

### Setup

```bash
make setup        # Configurar ambiente (uv sync + pre-commit)
```

### Docker (MLflow)

```bash
make docker-up    # Iniciar MLflow em background (requer .env)
make docker-down  # Parar todos os containers MLflow
```

### Desenvolvimento

```bash
make test         # Rodar testes com cobertura
make lint         # Verificar código com ruff
make format       # Formatar código com ruff
```

### Ajuda

```bash
make help         # Mostrar todos os comandos disponíveis
```

### Acessar

- MLflow UI: http://localhost:5000
