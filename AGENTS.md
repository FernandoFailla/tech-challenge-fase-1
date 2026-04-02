# AGENTS.md - Diretrizes de Codificação para TechChallenge1ChurnTelco

Este arquivo fornece informações essenciais para agentes de IA que trabalham neste projeto de ML.

## Visão Geral do Projeto

Este é um projeto de machine learning em Python 3.12+ para o Tech Challenge da pós-graduação MLE da FIAP. Utiliza MLflow para rastreamento de experimentos, FastAPI para serviço e scikit-learn para modelagem.

## Filosofia Central: Simplicidade em Primeiro Lugar

**O código deve ser sempre simples, conciso e funcional.**

Ao adicionar ou alterar qualquer coisa:
- **Prefira simplicidade** - Evite engenharia excessiva e complexidade desnecessária
- **Faça o mínimo** - Planeje escrever apenas o necessário para resolver o problema
- **Seja direto** - Sem enfeites, sem abstrações desnecessárias, sem recursos "só por precaução"
- **Função sobre forma** - Código funcionando vence arquitetura perfeita
- **Menos é mais** - Menos linhas, menos arquivos, menos dependências = melhor
- **Questione cada adição** - Isso realmente precisa ser adicionado? Pode ser mais simples?

**Exemplos do que evitar:**
- Hierarquias de classes complexas quando funções bastam
- Camadas de abstração que não fornecem valor imediato
- "Frameworks" genéricos para tarefas simples
- Preparação para cenários hipotéticos futuros
- Código boilerplate que não adiciona funcionalidade

**Em caso de dúvida, escolha a opção mais simples.**

## Comandos de Build/Lint/Test

Todos os comandos usam `uv` como gerenciador de pacotes:

```bash
# Configuração do ambiente
make setup                    # Instalar deps e hooks do pre-commit
uv sync                       # Instalar todas as dependências
uv sync --no-dev             # Apenas produção (CI/Docker)

# Testes
make test                     # Executar todos os testes com cobertura
uv run pytest tests/ -v      # Executar todos os testes (verbose)
uv run pytest tests/test_file.py -v              # Arquivo de teste único
uv run pytest tests/test_file.py::test_func -v   # Função de teste única
uv run pytest -m fast -v     # Executar apenas testes rápidos
uv run pytest -m slow -v     # Executar apenas testes de integração

# Requisitos de cobertura: 80% mínimo em src/
# Configuração de cobertura está em pyproject.toml

# Linting e Formatação
make lint                    # Verificar código com ruff
make format                  # Formatar código com ruff
uv run ruff check .          # Verificação manual de lint
uv run ruff check . --fix    # Auto-corrigir problemas
uv run ruff format .         # Formatação manual

# Verificação de Tipos
uv run mypy src/             # Verificar tipos (modo estrito habilitado)

# Pre-commit
uv run pre-commit run --all-files     # Executar todos os hooks
uv run pre-commit run ruff --all-files # Executar hook específico

# MLflow (Docker)
make docker-up               # Iniciar MLflow + PostgreSQL + MinIO
docker-compose -f docker/docker-compose.yml up -d
make docker-down             # Parar todos os containers
```

## Diretrizes de Estilo de Código

### Versão do Python e Imports
- Use recursos do Python 3.12+ (requires-python = ">=3.12,<3.14")
- Sempre use `from __future__ import annotations` no topo
- Ordem de imports: stdlib → terceiros → locais (imposto pela regra I do ruff)
- Use `TYPE_CHECKING` para imports necessários apenas para type hints

### Type Hints (Estrito)
- **Todas as funções devem ter type hints** (`disallow_untyped_defs = true`)
- Use sintaxe moderna: `list[str]`, `dict[str, Any]`, `str | None`
- Use `NDArray[Any]` de `numpy.typing` para tipos de array
- Use Protocols para definições de interfaces

### Formatação
- Comprimento de linha: **79 caracteres** (padrão PEP 8)
- Use ruff tanto para linting quanto para formatação
- Regras do Ruff habilitadas: I (imports), F (Pyflakes), E/W (pycodestyle), PL (pylint), PT (pytest)

### Convenções de Nomenclatura
- Classes: `PascalCase` (ex: `ExperimentRunner`, `ModelConfig`)
- Funções/variáveis: `snake_case` (ex: `run_experiment`, `model_name`)
- Constantes: `UPPER_CASE` (ex: `HTTP_OK`, `DEFAULT_PORT`)
- Privadas: `_leading_underscore` para uso interno
- Enums: Use `auto()` para valores quando apropriado

### Estrutura do Código
- Use `@dataclass` para objetos de configuração (prefira `frozen=True`)
- Use `@dataclass` para entidades com `field(default_factory=list)` para padrões mutáveis
- Use Protocols para injeção de dependência e interfaces
- Prefira métodos estáticos em classes de treino/utilitárias
- Organize imports: stdlib, terceiros, locais com linhas em branco entre eles

### Tratamento de Erros
- Use exceções específicas quando possível
- Use `try/except` com context managers
- Registre erros apropriadamente usando o módulo logging
- Para erros esperados: `except SpecificException as e:`
- Para captura geral (use com moderação): `except Exception as e:  # noqa: BLE001`
- Retorne status booleano para funções de validação

### Documentação
- Todos os módulos precisam de docstrings com aspas triplas
- Todas as funções públicas precisam de docstrings
- Use docstrings no estilo Google ou padrão
- Adicione type hints em vez de documentar tipos nas docstrings

### Testes
- Arquivos de teste: `tests/test_*.py`
- Marcadores de teste: `@pytest.mark.fast` para testes rápidos, `@pytest.mark.slow` para integração
- Cobertura mínima: 80% em `src/`
- Use fixtures do pytest para setup compartilhado
- Mock serviços externos (MLflow, bancos de dados) em testes unitários

### Integração MLflow
- Carregue config do ambiente usando `MLflowConfig.from_env()`
- Use `python-dotenv` para suporte a arquivo `.env`
- Sempre use context managers: `with mlflow.start_run():`
- Registre params com `mlflow.log_param()`, métricas com `mlflow.log_metric()`
- Registre modelos com flavor apropriado: `mlflow.sklearn.log_model()`

### Regras de Estrutura do Projeto
- **SEM artefatos em notebooks** - notebooks são apenas para exploração
- Artefatos finais (modelos, datasets) vão em `src/pipelines/` como scripts parametrizados
- Dados vão em `data/`, modelos em `models/`, docs em `docs/`
- Código fonte organizado: `api/`, `data/`, `features/`, `training/`, `inference/`, `pipelines/`, `schemas/`

### Gerenciamento de Ambiente
- Use arquivo `.env` para configuração local (copie de `.env.example`)
- Nunca commite arquivos `.env`
- Use `python-dotenv` para carregar variáveis de ambiente
- Docker Compose usa arquivo `.env` automaticamente

### Dependências
- Deps de produção: listadas em `[project] dependencies`
- Deps de dev: listadas em `[dependency-groups] dev`
- Use `uv add <package>` para adicionar dependências de produção
- Use `uv add --dev <package>` para adicionar dependências de dev
- Arquivo de lock `uv.lock` deve ser commitado

### Fluxo de Trabalho Git
- Hooks do pre-commit rodam automaticamente no commit
- CI corrige automaticamente PRs com pre-commit
- Use mensagens de commit convencionais

## Arquitetura do Projeto

### Estrutura de Diretórios
```
src/
├── api/           # Aplicação FastAPI e endpoints
├── data/          # Carregamento e validação de dados
├── eda/           # Ferramentas de análise exploratória de dados
├── features/      # Pipelines de engenharia de features
├── inference/     # Serviço de modelo e predição
├── pipelines/     # Pipelines de treino end-to-end
├── schemas/       # Modelos Pydantic e contratos de dados
└── training/      # Treino e avaliação de modelo
data/              # Datasets brutos e processados
models/            # Artefatos de modelo salvos
tests/             # Testes unitários e de integração
notebooks/         # Notebooks Jupyter para exploração
docs/              # Documentação e relatórios
```

### Padrões de Design Principais
- **Configuration as Code**: Use dataclasses para todos os objetos de configuração
- **Protocol-based Interfaces**: Defina contratos com typing.Protocol
- **Static Methods**: Prefira `@staticmethod` para funções utilitárias
- **Context Managers**: Use instruções `with` para gerenciamento de recursos
- **Immutable Data**: Use `frozen=True` em dataclasses quando possível
