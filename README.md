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

## Regra de execucao (pipelines)

- Nao gerar artefatos finais (modelos treinados, bases finais, tracking de experimento) a partir de notebooks.
- Notebooks sao para exploracao.
- Artefatos finais devem ser gerados por scripts parametrizaveis em `src/pipelines/`, executados via terminal.

---

## IMPORTANTE: Ambiente Virtual

**Sempre utilize um ambiente virtual (venv) para isolar as dependencias do projeto.**

O projeto suporta duas formas de gerenciar o ambiente virtual:

### Opcao 1: Com uv (Recomendado)

O [uv](https://docs.astral.sh/uv/) e um gerenciador de pacotes extremamente rapido (escrito em Rust) que gerencia automaticamente o ambiente virtual e dependencias.

**Instalar o uv:**

Linux/macOS:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Ou com pip:
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
