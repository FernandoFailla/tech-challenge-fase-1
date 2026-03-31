#!/usr/bin/env python3
<<<<<<< HEAD
<<<<<<< HEAD
"""Script para validar a configuração do MLflow.

Verifica se o servidor está acessível e funcionando corretamente.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.NullHandler()],
)
logger = logging.getLogger(__name__)

# Console Rich para output bonito
console = Console()


@dataclass
class CheckResult:
    """Resultado de uma verificação."""

    name: str
    status: bool
    message: str = ""
    details: list[str] = field(default_factory=list)


class MLflowValidator:
    """Validador do ambiente MLflow."""

    def __init__(self) -> None:
        """Inicializa o validador com variáveis de ambiente."""
        load_dotenv()
        self.tracking_uri: str = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.port: str = os.getenv("MLFLOW_PORT", "5000")
        self.compose_file: Path = Path("docker/docker-compose.yml")

    async def _check_docker(self) -> CheckResult:
        """Verifica se os containers Docker estão rodando."""
        console.print("\n[blue]🔍 Verificando containers Docker...[/blue]")

        if not self.compose_file.exists():
            return CheckResult(
                name="Docker Containers",
                status=False,
                message="Arquivo docker-compose.yml não encontrado",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker-compose",
                "-f",
                str(self.compose_file),
                "ps",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=10.0
            )

            output = stdout.decode()
            if "mlflow-server" in output:
                return CheckResult(
                    name="Docker Containers",
                    status=True,
                    message="✅ Container mlflow-server encontrado",
                )

            return CheckResult(
                name="Docker Containers",
                status=False,
                message="❌ Container mlflow-server não encontrado",
            )

        except subprocess.CalledProcessError as e:
            return CheckResult(
                name="Docker Containers",
                status=False,
                message=f"❌ Erro ao verificar containers: {e}",
            )
        except asyncio.TimeoutError:
            return CheckResult(
                name="Docker Containers",
                status=False,
                message="❌ Timeout ao verificar containers",
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="Docker Containers",
                status=False,
                message=f"❌ Erro inesperado: {e}",
            )

    async def _check_health(self, client: httpx.AsyncClient) -> CheckResult:
        """Verifica se o servidor MLflow está respondendo."""
        console.print(
            f"\n[blue]🔍 Verificando MLflow em {self.tracking_uri}...[/blue]"
        )

        try:
            response = await client.get(
                f"{self.tracking_uri}/health",
                timeout=5.0,
            )

            if response.status_code == 200:
                return CheckResult(
                    name="MLflow Health",
                    status=True,
                    message="✅ MLflow server está respondendo",
                )

            return CheckResult(
                name="MLflow Health",
                status=False,
                message=f"⚠️ MLflow retornou status {response.status_code}",
            )

        except httpx.ConnectError:
            return CheckResult(
                name="MLflow Health",
                status=False,
                message=f"❌ Não foi possível conectar ao MLflow em {self.tracking_uri}",
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="MLflow Health",
                status=False,
                message=f"❌ Erro: {e}",
            )

    async def _check_api(self, client: httpx.AsyncClient) -> CheckResult:
        """Verifica se a API do MLflow está funcionando."""
        console.print("\n[blue]🔍 Verificando API MLflow...[/blue]")

        try:
            response = await client.get(
                f"{self.tracking_uri}/api/2.0/mlflow/experiments/list",
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                experiments: Sequence[dict] = data.get("experiments", [])
                return CheckResult(
                    name="MLflow API",
                    status=True,
                    message="✅ API respondendo",
                    details=[
                        f"📊 {len(experiments)} experimento(s) encontrado(s)"
                    ],
                )

            return CheckResult(
                name="MLflow API",
                status=False,
                message=f"⚠️ API retornou status {response.status_code}",
            )

        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="MLflow API",
                status=False,
                message=f"❌ Erro ao acessar API: {e}",
            )

    async def _check_minio(self, client: httpx.AsyncClient) -> CheckResult:
        """Verifica se o MinIO está acessível."""
        console.print("\n[blue]🔍 Verificando MinIO...[/blue]")

        try:
            response = await client.get(
                "http://localhost:9000/minio/health/live",
                timeout=5.0,
            )

            if response.status_code == 200:
                return CheckResult(
                    name="MinIO",
                    status=True,
                    message="✅ MinIO está respondendo",
                    details=["🌐 Console: http://localhost:9001"],
                )

            return CheckResult(
                name="MinIO",
                status=False,
                message=f"⚠️ MinIO retornou status {response.status_code}",
            )

        except httpx.ConnectError:
            return CheckResult(
                name="MinIO",
                status=False,
                message="⚠️ MinIO não está rodando (pode estar usando versão simples)",
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="MinIO",
                status=False,
                message=f"⚠️ Erro ao verificar MinIO: {e}",
            )

    async def _check_python_client(self) -> CheckResult:
        """Testa a conexão via cliente Python MLflow."""
        console.print("\n[blue]🔍 Testando cliente Python MLflow...[/blue]")

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow_client = MlflowClient()

            # Listar experimentos
            experiments = mlflow_client.search_experiments()
            exp_count = len(experiments)

            details = [f"📊 {exp_count} experimento(s) disponível(eis)"]

            # Criar experimento de teste
            exp_name = "test_validation"
            try:
                exp_id = mlflow_client.create_experiment(exp_name)
                details.append("✅ Experimento de teste criado")

                mlflow_client.delete_experiment(exp_id)
                details.append("✅ Experimento de teste removido")
            except Exception as e:  # noqa: BLE001
                details.append(
                    f"⚠️ Não foi possível criar experimento de teste: {e}"
                )

            return CheckResult(
                name="Python Client",
                status=True,
                message="✅ Cliente Python conectado com sucesso",
                details=details,
            )

        except ImportError:
            return CheckResult(
                name="Python Client",
                status=False,
                message="❌ MLflow não está instalado no Python",
                details=["💡 Instale com: pip install mlflow"],
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="Python Client",
                status=False,
                message=f"❌ Erro no cliente Python: {e}",
            )

    async def run_all_checks(self) -> list[CheckResult]:
        """Executa todas as verificações em paralelo."""
        async with httpx.AsyncClient() as client:
            # Rodar checks independentes em paralelo
            tasks = [
                self._check_docker(),
                self._check_health(client),
                self._check_api(client),
                self._check_minio(client),
            ]

            # Executar checks HTTP em paralelo
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Tratar exceções
            processed_results: list[CheckResult] = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append(
                        CheckResult(
                            name="Unknown",
                            status=False,
                            message=f"❌ Exceção: {result}",
                        )
                    )
                else:
                    processed_results.append(result)

            # Python client precisa ser sequencial (pode ter conflitos)
            python_result = await self._check_python_client()
            processed_results.append(python_result)

            return processed_results

    def display_results(self, results: list[CheckResult]) -> int:
        """Exibe os resultados em formato de tabela."""
        # Painel de resumo
        passed = sum(1 for r in results if r.status)
        failed = len(results) - passed

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", style="bold")
        table.add_column("Check", style="cyan")
        table.add_column("Mensagem")

        for result in results:
            status_icon = "✅" if result.status else "❌"
            status_color = "green" if result.status else "red"
            table.add_row(
                f"[{status_color}]{status_icon}[/{status_color}]",
                result.name,
                result.message,
            )

            for detail in result.details:
                table.add_row("", "", f"   {detail}")

        console.print("\n")
        console.print(
            Panel(table, title="📋 Resumo da Validação", border_style="blue")
        )

        # Status final
        if failed == 0:
            console.print(
                Panel(
                    f"🎉 Ambiente MLflow está totalmente funcional!\n\n"
                    f"🔗 Acesse a interface em: {self.tracking_uri}",
                    title="Sucesso",
                    border_style="green",
                )
            )
            return 0

        console.print(
            Panel(
                Text(
                    f"⚠️  {failed} verificação(ões) falharam.\n\n"
                    "💡 Dicas de solução:\n"
                    "   1. Verifique se o Docker está rodando\n"
                    "   2. Execute: ./mlflow.sh start\n"
                    "   3. Aguarde alguns segundos e tente novamente\n"
                    "   4. Verifique os logs: docker-compose logs -f",
                    justify="left",
                ),
                title="Atenção",
                border_style="yellow",
            )
        )
        return 1


async def main() -> int:
    """Função principal de validação."""
    console.print(
        Panel.fit(
            "[bold blue]🔧 Validação do Ambiente MLflow[/bold blue]",
            border_style="blue",
        )
    )

    validator = MLflowValidator()
    console.print(f"\n[cyan]📍 MLflow URI:[/cyan] {validator.tracking_uri}")

    results = await validator.run_all_checks()
    return validator.display_results(results)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
=======
"""
Script para validar a configuração do MLflow
Verifica se o servidor está acessível e funcionando corretamente
=======
"""Script para validar a configuração do MLflow.

Verifica se o servidor está acessível e funcionando corretamente.
>>>>>>> ac79aee (feat: moderniza exemplos MLflow com código type-safe e UI melhorada (#45))
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Sequence

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.NullHandler()],
)
logger = logging.getLogger(__name__)

# Console Rich para output bonito
console = Console()


@dataclass
class CheckResult:
    """Resultado de uma verificação."""

    name: str
    status: bool
    message: str = ""
    details: list[str] = field(default_factory=list)


class MLflowValidator:
    """Validador do ambiente MLflow."""

    def __init__(self) -> None:
        """Inicializa o validador com variáveis de ambiente."""
        load_dotenv()
        self.tracking_uri: str = os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.port: str = os.getenv("MLFLOW_PORT", "5000")
        self.compose_file: Path = Path("docker/docker-compose.yml")

    async def _check_docker(self) -> CheckResult:
        """Verifica se os containers Docker estão rodando."""
        console.print("\n[blue]🔍 Verificando containers Docker...[/blue]")

        if not self.compose_file.exists():
            return CheckResult(
                name="Docker Containers",
                status=False,
                message="Arquivo docker-compose.yml não encontrado",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker-compose",
                "-f",
                str(self.compose_file),
                "ps",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=10.0
            )

            output = stdout.decode()
            if "mlflow-server" in output:
                return CheckResult(
                    name="Docker Containers",
                    status=True,
                    message="✅ Container mlflow-server encontrado",
                )

            return CheckResult(
                name="Docker Containers",
                status=False,
                message="❌ Container mlflow-server não encontrado",
            )

        except subprocess.CalledProcessError as e:
            return CheckResult(
                name="Docker Containers",
                status=False,
                message=f"❌ Erro ao verificar containers: {e}",
            )
        except asyncio.TimeoutError:
            return CheckResult(
                name="Docker Containers",
                status=False,
                message="❌ Timeout ao verificar containers",
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="Docker Containers",
                status=False,
                message=f"❌ Erro inesperado: {e}",
            )

    async def _check_health(self, client: httpx.AsyncClient) -> CheckResult:
        """Verifica se o servidor MLflow está respondendo."""
        console.print(
            f"\n[blue]🔍 Verificando MLflow em {self.tracking_uri}...[/blue]"
        )

        try:
            response = await client.get(
                f"{self.tracking_uri}/health",
                timeout=5.0,
            )

            if response.status_code == 200:
                return CheckResult(
                    name="MLflow Health",
                    status=True,
                    message="✅ MLflow server está respondendo",
                )

            return CheckResult(
                name="MLflow Health",
                status=False,
                message=f"⚠️ MLflow retornou status {response.status_code}",
            )

        except httpx.ConnectError:
            return CheckResult(
                name="MLflow Health",
                status=False,
                message=f"❌ Não foi possível conectar ao MLflow em {self.tracking_uri}",
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="MLflow Health",
                status=False,
                message=f"❌ Erro: {e}",
            )

    async def _check_api(self, client: httpx.AsyncClient) -> CheckResult:
        """Verifica se a API do MLflow está funcionando."""
        console.print("\n[blue]🔍 Verificando API MLflow...[/blue]")

        try:
            response = await client.get(
                f"{self.tracking_uri}/api/2.0/mlflow/experiments/list",
                timeout=5.0,
            )

            if response.status_code == 200:
                data = response.json()
                experiments: Sequence[dict] = data.get("experiments", [])
                return CheckResult(
                    name="MLflow API",
                    status=True,
                    message="✅ API respondendo",
                    details=[
                        f"📊 {len(experiments)} experimento(s) encontrado(s)"
                    ],
                )

            return CheckResult(
                name="MLflow API",
                status=False,
                message=f"⚠️ API retornou status {response.status_code}",
            )

        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="MLflow API",
                status=False,
                message=f"❌ Erro ao acessar API: {e}",
            )

    async def _check_minio(self, client: httpx.AsyncClient) -> CheckResult:
        """Verifica se o MinIO está acessível."""
        console.print("\n[blue]🔍 Verificando MinIO...[/blue]")

        try:
            response = await client.get(
                "http://localhost:9000/minio/health/live",
                timeout=5.0,
            )

            if response.status_code == 200:
                return CheckResult(
                    name="MinIO",
                    status=True,
                    message="✅ MinIO está respondendo",
                    details=["🌐 Console: http://localhost:9001"],
                )

            return CheckResult(
                name="MinIO",
                status=False,
                message=f"⚠️ MinIO retornou status {response.status_code}",
            )

        except httpx.ConnectError:
            return CheckResult(
                name="MinIO",
                status=False,
                message="⚠️ MinIO não está rodando (pode estar usando versão simples)",
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="MinIO",
                status=False,
                message=f"⚠️ Erro ao verificar MinIO: {e}",
            )

    async def _check_python_client(self) -> CheckResult:
        """Testa a conexão via cliente Python MLflow."""
        console.print("\n[blue]🔍 Testando cliente Python MLflow...[/blue]")

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow_client = MlflowClient()

            # Listar experimentos
            experiments = mlflow_client.search_experiments()
            exp_count = len(experiments)

            details = [f"📊 {exp_count} experimento(s) disponível(eis)"]

            # Criar experimento de teste
            exp_name = "test_validation"
            try:
                exp_id = mlflow_client.create_experiment(exp_name)
                details.append("✅ Experimento de teste criado")

                mlflow_client.delete_experiment(exp_id)
                details.append("✅ Experimento de teste removido")
            except Exception as e:  # noqa: BLE001
                details.append(
                    f"⚠️ Não foi possível criar experimento de teste: {e}"
                )

            return CheckResult(
                name="Python Client",
                status=True,
                message="✅ Cliente Python conectado com sucesso",
                details=details,
            )

        except ImportError:
            return CheckResult(
                name="Python Client",
                status=False,
                message="❌ MLflow não está instalado no Python",
                details=["💡 Instale com: pip install mlflow"],
            )
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                name="Python Client",
                status=False,
                message=f"❌ Erro no cliente Python: {e}",
            )

    async def run_all_checks(self) -> list[CheckResult]:
        """Executa todas as verificações em paralelo."""
        async with httpx.AsyncClient() as client:
            # Rodar checks independentes em paralelo
            tasks = [
                self._check_docker(),
                self._check_health(client),
                self._check_api(client),
                self._check_minio(client),
            ]

            # Executar checks HTTP em paralelo
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Tratar exceções
            processed_results: list[CheckResult] = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append(
                        CheckResult(
                            name="Unknown",
                            status=False,
                            message=f"❌ Exceção: {result}",
                        )
                    )
                else:
                    processed_results.append(result)

            # Python client precisa ser sequencial (pode ter conflitos)
            python_result = await self._check_python_client()
            processed_results.append(python_result)

            return processed_results

    def display_results(self, results: list[CheckResult]) -> int:
        """Exibe os resultados em formato de tabela."""
        # Painel de resumo
        passed = sum(1 for r in results if r.status)
        failed = len(results) - passed

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Status", style="bold")
        table.add_column("Check", style="cyan")
        table.add_column("Mensagem")

        for result in results:
            status_icon = "✅" if result.status else "❌"
            status_color = "green" if result.status else "red"
            table.add_row(
                f"[{status_color}]{status_icon}[/{status_color}]",
                result.name,
                result.message,
            )

            for detail in result.details:
                table.add_row("", "", f"   {detail}")

        console.print("\n")
        console.print(
            Panel(table, title="📋 Resumo da Validação", border_style="blue")
        )

        # Status final
        if failed == 0:
            console.print(
                Panel(
                    f"🎉 Ambiente MLflow está totalmente funcional!\n\n"
                    f"🔗 Acesse a interface em: {self.tracking_uri}",
                    title="Sucesso",
                    border_style="green",
                )
            )
            return 0

        console.print(
            Panel(
                Text(
                    f"⚠️  {failed} verificação(ões) falharam.\n\n"
                    "💡 Dicas de solução:\n"
                    "   1. Verifique se o Docker está rodando\n"
                    "   2. Execute: ./mlflow.sh start\n"
                    "   3. Aguarde alguns segundos e tente novamente\n"
                    "   4. Verifique os logs: docker-compose logs -f",
                    justify="left",
                ),
                title="Atenção",
                border_style="yellow",
            )
        )
        return 1


async def main() -> int:
    """Função principal de validação."""
    console.print(
        Panel.fit(
            "[bold blue]🔧 Validação do Ambiente MLflow[/bold blue]",
            border_style="blue",
        )
    )

    validator = MLflowValidator()
    console.print(f"\n[cyan]📍 MLflow URI:[/cyan] {validator.tracking_uri}")

    results = await validator.run_all_checks()
    return validator.display_results(results)


if __name__ == "__main__":
<<<<<<< HEAD
    sys.exit(main())
>>>>>>> c2c42ab (feat: implementa setup local de MLflow com PostgreSQL e MinIO (#45))
=======
    sys.exit(asyncio.run(main()))
>>>>>>> ac79aee (feat: moderniza exemplos MLflow com código type-safe e UI melhorada (#45))
