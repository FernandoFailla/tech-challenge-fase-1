# Makefile para o TechChallenge Fase 1
# Comandos essenciais para desenvolvimento

.PHONY: setup test lint format help docker-up docker-down

# Verifica se o arquivo .env existe
CHECK_ENV := $(shell test -f .env && echo 1 || echo 0)
ifeq ($(CHECK_ENV),0)
  ENV_ERROR = @echo "❌ ERRO: Arquivo .env não encontrado!" && echo "👉 Copie .env.example para .env:" && echo "   cp .env.example .env" && echo "" && exit 1
endif

# Help padrao
help:
	@echo "Tech Challenge Fase 1 - Comandos Disponiveis"
	@echo ""
	@echo "Setup:"
	@echo "  make setup      - Configurar ambiente (uv sync + pre-commit)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up  - Iniciar MLflow em background (requer .env)"
	@echo "  make docker-down - Parar todos os containers MLflow"
	@echo ""
	@echo "Desenvolvimento:"
	@echo "  make test       - Rodar testes"
	@echo "  make lint       - Verificar codigo com ruff"
	@echo "  make format     - Formatar codigo com ruff"
	@echo ""

# Setup inicial
setup:
	@echo "Configurando ambiente..."
	uv sync
	uv run pre-commit install
	@echo "Setup concluido!"

# Testes
test:
	@echo "Executando testes..."
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Verificar codigo
lint:
	@echo "Verificando codigo com ruff..."
	uv run ruff check .

# Formatar codigo
format:
	@echo "Formatando codigo com ruff..."
	uv run ruff format .

# Iniciar Docker em background
docker-up:
	$(ENV_ERROR)
	@echo "🐳 Iniciando MLflow em background..."
	docker-compose -f docker/docker-compose.yml --env-file .env up -d
	@echo "✅ MLflow iniciado! Acesse http://localhost:$$(grep -E '^MLFLOW_PORT=' .env | cut -d '=' -f2) para usar."

# Parar Docker
docker-down:
	@echo "🛑 Parando containers MLflow..."
	docker-compose -f docker/docker-compose.yml --env-file .env down
	@echo "✅ Containers parados!"
