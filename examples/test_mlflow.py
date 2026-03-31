#!/usr/bin/env python3
"""
Script para validar a configuração do MLflow
Verifica se o servidor está acessível e funcionando corretamente
"""

import os
import sys
import requests
import time
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5000")


def check_docker_containers():
    """Verifica se os containers Docker estão rodando"""
    import subprocess

    print("🔍 Verificando containers Docker...")
    try:
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True,
            check=True,
        )

        if "mlflow-server" in result.stdout:
            print("✅ Container mlflow-server encontrado")
            return True
        else:
            print("❌ Container mlflow-server não encontrado")
            return False
    except Exception as e:
        print(f"❌ Erro ao verificar containers: {e}")
        return False


def check_mlflow_health():
    """Verifica se o servidor MLflow está respondendo"""
    print(f"\n🔍 Verificando MLflow em {MLFLOW_TRACKING_URI}...")

    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=5)
        if response.status_code == 200:
            print("✅ MLflow server está respondendo")
            return True
        else:
            print(f"⚠️  MLflow retornou status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(
            f"❌ Não foi possível conectar ao MLflow em {MLFLOW_TRACKING_URI}"
        )
        return False
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False


def check_mlflow_api():
    """Verifica se a API do MLflow está funcionando"""
    print("\n🔍 Verificando API MLflow...")

    try:
        # Listar experimentos
        response = requests.get(
            f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list", timeout=5
        )

        if response.status_code == 200:
            data = response.json()
            experiments = data.get("experiments", [])
            print(f"✅ API respondendo")
            print(f"   📊 {len(experiments)} experimento(s) encontrado(s)")
            return True
        else:
            print(f"⚠️  API retornou status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erro ao acessar API: {e}")
        return False


def check_minio():
    """Verifica se o MinIO está acessível (se usando versão completa)"""
    print("\n🔍 Verificando MinIO...")

    try:
        response = requests.get(
            "http://localhost:9000/minio/health/live", timeout=5
        )
        if response.status_code == 200:
            print("✅ MinIO está respondendo")
            print("   🌐 Console: http://localhost:9001")
            return True
        else:
            print(f"⚠️  MinIO retornou status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️  MinIO não está rodando (pode estar usando versão simples)")
        return False
    except Exception as e:
        print(f"⚠️  Erro ao verificar MinIO: {e}")
        return False


def test_mlflow_python_client():
    """Testa a conexão via cliente Python"""
    print("\n🔍 Testando cliente Python MLflow...")

    try:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Tentar obter lista de experimentos
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()

        print(f"✅ Cliente Python conectado com sucesso")
        print(f"   📊 {len(experiments)} experimento(s) disponível(eis)")

        # Criar um experimento de teste
        exp_name = "test_validation"
        try:
            client.create_experiment(exp_name)
            print(f"✅ Experimento de teste criado")
            client.delete_experiment(
                client.get_experiment_by_name(exp_name).experiment_id
            )
            print(f"✅ Experimento de teste removido")
        except Exception as e:
            print(f"⚠️  Não foi possível criar experimento de teste: {e}")

        return True
    except ImportError:
        print("❌ MLflow não está instalado no Python")
        print("   💡 Instale com: pip install mlflow")
        return False
    except Exception as e:
        print(f"❌ Erro no cliente Python: {e}")
        return False


def main():
    """Função principal de validação"""
    print("=" * 60)
    print("🔧 Validação do Ambiente MLflow")
    print("=" * 60)
    print(f"\n📍 MLflow URI: {MLFLOW_TRACKING_URI}")
    print("")

    checks = []

    # Verificar containers
    checks.append(("Docker Containers", check_docker_containers()))

    # Verificar saúde do MLflow
    checks.append(("MLflow Health", check_mlflow_health()))

    # Verificar API
    checks.append(("MLflow API", check_mlflow_api()))

    # Verificar MinIO
    checks.append(("MinIO", check_minio()))

    # Verificar cliente Python
    checks.append(("Python Client", test_mlflow_python_client()))

    # Resumo
    print("\n" + "=" * 60)
    print("📋 Resumo da Validação")
    print("=" * 60)

    passed = 0
    failed = 0

    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("")
    print(f"Resultado: {passed}/{len(checks)} verificações passaram")

    if failed == 0:
        print("\n🎉 Ambiente MLflow está totalmente funcional!")
        print(f"\n🔗 Acesse a interface em: {MLFLOW_TRACKING_URI}")
        return 0
    else:
        print("\n⚠️  Algumas verificações falharam.")
        print("\n💡 Dicas de solução:")
        print("   1. Verifique se o Docker está rodando")
        print("   2. Execute: ./mlflow.sh start")
        print("   3. Aguarde alguns segundos e tente novamente")
        print("   4. Verifique os logs: docker-compose logs -f")
        return 1


if __name__ == "__main__":
    sys.exit(main())
