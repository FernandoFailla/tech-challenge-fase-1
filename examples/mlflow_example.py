#!/usr/bin/env python3
"""
Exemplo de uso do MLflow com tracking server local
Este script demonstra como usar o MLflow para tracking de experimentos
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")

# Carregar variáveis de ambiente (se existir .env)
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ Variáveis de ambiente carregadas do .env")
except ImportError:
    print("⚠️  python-dotenv não instalado. Usando variáveis padrão.")

# Configuração do MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME", "tech-challenge-demo"
)

print(f"\n🔌 Conectando ao MLflow: {MLFLOW_TRACKING_URI}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Configurar S3/MinIO para artifacts (se estiver usando versão completa)
if os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    os.environ["AWS_ACCESS_KEY_ID"] = os.getenv(
        "AWS_ACCESS_KEY_ID", "minioadmin"
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv(
        "AWS_SECRET_ACCESS_KEY", "minioadmin"
    )
    print("✅ Configurado S3 endpoint para artifacts")

# Criar/Setar experimento
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"📊 Experimento: {MLFLOW_EXPERIMENT_NAME}\n")


def train_and_log_model(
    model, model_name, dataset_name, X_train, X_test, y_train, y_test, params
):
    """Treina um modelo e loga no MLflow"""

    with mlflow.start_run(run_name=f"{model_name}_{dataset_name}"):
        print(f"🚀 Iniciando run: {model_name} - {dataset_name}")

        # Log de parâmetros
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("dataset", dataset_name)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Treinamento
        model.fit(X_train, y_train)

        # Predições
        y_pred = model.predict(X_test)

        # Métricas
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

        # Log de métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log de modelo
        mlflow.sklearn.log_model(model, "model")

        # Log de artifact (exemplo)
        # mlflow.log_artifact("caminho/para/arquivo.txt")

        print(f"   ✅ Accuracy: {metrics['accuracy']:.4f}")
        print(f"   ✅ Run ID: {mlflow.active_run().info.run_id}")

        return metrics


def main():
    """Função principal - demonstração completa"""

    # Dataset 1: Iris
    print("=" * 60)
    print("📊 Dataset: Iris")
    print("=" * 60)
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42
    )

    # Experimento 1: Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    rf_model = RandomForestClassifier(**rf_params)
    train_and_log_model(
        rf_model,
        "RandomForest",
        "Iris",
        X_train_i,
        X_test_i,
        y_train_i,
        y_test_i,
        rf_params,
    )

    # Experimento 2: Gradient Boosting
    gb_params = {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
    gb_model = GradientBoostingClassifier(**gb_params)
    train_and_log_model(
        gb_model,
        "GradientBoosting",
        "Iris",
        X_train_i,
        X_test_i,
        y_train_i,
        y_test_i,
        gb_params,
    )

    # Dataset 2: Wine
    print("\n" + "=" * 60)
    print("📊 Dataset: Wine")
    print("=" * 60)
    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_wine, y_wine, test_size=0.2, random_state=42
    )

    # Experimento 3: Random Forest com diferentes parâmetros
    rf_params2 = {"n_estimators": 200, "max_depth": 10, "random_state": 42}
    rf_model2 = RandomForestClassifier(**rf_params2)
    train_and_log_model(
        rf_model2,
        "RandomForest",
        "Wine",
        X_train_w,
        X_test_w,
        y_train_w,
        y_test_w,
        rf_params2,
    )

    print("\n" + "=" * 60)
    print("✅ Experimentos concluídos!")
    print("=" * 60)
    print(f"\n🔗 Acesse a interface do MLflow:")
    print(f"   {MLFLOW_TRACKING_URI}")
    print(f"\n📊 Lá você pode:")
    print("   • Comparar métricas entre runs")
    print("   • Visualizar parâmetros")
    print("   • Baixar modelos treinados")
    print("   • Registrar modelos para produção")
    print("")


if __name__ == "__main__":
    # Verificar conexão com MLflow
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        print(f"✅ Conexão com MLflow estabelecida!")
        print(f"   Experiments disponíveis: {len(experiments)}")
    except Exception as e:
        print(f"❌ Erro ao conectar ao MLflow: {e}")
        print(f"\n💡 Certifique-se de que o servidor está rodando:")
        print(f"   ./mlflow.sh start")
        exit(1)

    main()
