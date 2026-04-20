"""Funções utilitárias comuns para pipelines de ML.

Este módulo fornece helpers para padrões repetidos em pipelines
como carregamento de configuração, obtenção de nomes de experimentos,
e versões de datasets.
"""

from __future__ import annotations

import os

from src.data.versioning import get_dataset_version_from_dvc


def load_dotenv_silent() -> None:
    """Carrega variáveis de ambiente do arquivo .env silenciosamente.

    Não levanta erro se o pacote python-dotenv não estiver instalado
    ou se o arquivo .env não existir. Útil para scripts que podem
    rodar em ambientes sem dotenv.
    """
    try:
        # nosec: B404 - lazy import necessário para fallback gracioso
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv()  # nosec: B108
    except ImportError:
        # python-dotenv não instalado - usa variáveis de ambiente existentes
        pass


def get_experiment_name(
    cli_arg: str | None,
    env_var_name: str,
    default_name: str,
) -> str:
    """Obtém nome do experimento com prioridade: CLI > env var > default.

    Args:
        cli_arg: Valor passado via argumento de linha de comando.
            None se não fornecido.
        env_var_name: Nome da variável de ambiente específica do modelo
            (ex: "MLFLOW_MLP_EXPERIMENT_NAME").
        default_name: Nome padrão se nem CLI nem env var forem fornecidos.

    Returns:
        Nome do experimento a ser usado

    Example:
        >>> name = get_experiment_name(
        ...     "mlp-v2", "MLFLOW_MLP_EXPERIMENT_NAME", "mlp"
        ... )
        >>> print(name)
        mlp-v2
    """
    if cli_arg:
        return cli_arg

    env_value = os.getenv(env_var_name)
    if env_value:
        return env_value

    return default_name


def safe_get_dataset_version() -> str:
    """Obtém versão do dataset via DVC com fallback gracioso.

    Tenta obter a versão do dataset via DVC. Se falhar (DVC não
    configurado, arquivo não encontrado, etc.), retorna "unknown".

    Returns:
        String com versão do dataset (hash Git de 8 chars) ou "unknown"

    Example:
        >>> version = safe_get_dataset_version()
        >>> print(version)
        abc123de
    """
    try:
        return get_dataset_version_from_dvc()
    except (FileNotFoundError, ValueError):
        return "unknown"
