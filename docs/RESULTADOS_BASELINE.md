# Resultados dos Modelos Baseline - Churn Telco

Este documento consolida os resultados dos modelos baseline treinados
para predicao de churn no dataset Telco Customer Churn.

## Dataset

- **Fonte:** WA_Fn-UseC_-Telco-Customer-Churn.csv
- **Amostras:** ~7.043 clientes
- **Taxa de churn:** ~26,5% (classe positiva desbalanceada)
- **Features:** 30 apos preprocessamento (one-hot encoding)
- **Split:** 80% treino / 20% teste, estratificado, seed=42

---

## Modelos Treinados

### 1. Dummy Classifier (Baseline Ingenuo)

O DummyClassifier serve como linha de base minima. Qualquer modelo
supervisionado deve superar esses resultados.

| Estrategia | Acuracia | Precisao | Recall | F1-Score | ROC-AUC | PR-AUC |
|-----------|----------|----------|--------|----------|---------|--------|
| most_frequent | 0,734 | 0,000 | 0,000 | 0,000 | 0,500 | 0,266 |
| stratified | 0,615 | 0,277 | 0,278 | 0,277 | 0,507 | 0,269 |
| uniform | 0,522 | 0,288 | 0,543 | 0,376 | 0,500 | 0,266 |

**Observacoes:**
- `most_frequent`: Sempre prediz classe majoritaria (No churn). Acuracia
  alta mas inutil para retenacao (precision/recall = 0).
- `stratified`: Prediz proporcionalmente a distribuicao de classes.
- `uniform`: Prediz aleatoriamente entre as classes.

**MLflow:** [tech-challenge-dummy-baseline](http://localhost:5000/#/experiments/1)

### 2. Logistic Regression (Baseline Estatistico)

Regressao logistica com preprocessamento one-hot encoding e
StandardScaler (fit apenas no treino, sem data leakage).

**Validacao Cruzada (5 folds):**

| Metrica | Media | Desvio Padrao |
|---------|-------|---------------|
| Acuracia | 0,802 | 0,011 |
| Precisao | 0,653 | 0,028 |
| Recall | 0,548 | 0,024 |
| F1-Score | 0,596 | 0,024 |
| ROC-AUC | 0,846 | 0,005 |

**Conjunto de Teste:**

| Metrica | Valor |
|---------|-------|
| Acuracia | 0,804 |
| Precisao | 0,648 |
| Recall | 0,575 |
| F1-Score | 0,609 |
| ROC-AUC | 0,836 |
| PR-AUC | 0,621 |
| Brier Score | 0,140 |

**Top 5 Features por Coeficiente (importancia):**

| Feature | Coeficiente | Impacto |
|---------|-------------|---------|
| tenure | -1,348 | Menor risco (clientes antigos) |
| MonthlyCharges | -0,852 | Menor risco (cobrancas menores) |
| InternetService_Fiber optic | +0,728 | Maior risco |
| TotalCharges | +0,639 | Maior risco |
| Contract_Two year | -0,603 | Menor risco (contrato longo) |

**MLflow:** [tech-challenge-logistic-regression](http://localhost:5000/#/experiments/3)

### 3. MLP - Multi-Layer Perceptron (Baseline Neural)

#### 3.1 MLP Baseline Original

Rede neural feedforward com arquitetura (128, 64, 32), dropout 0,3,
batch normalization, treinada com Adam e early stopping.

**Conjunto de Teste:**

| Metrica | Valor |
|---------|-------|
| Acuracia | 0,792 |
| Precisao | 0,609 |
| Recall | 0,604 |
| F1-Score | 0,607 |
| ROC-AUC | 0,831 |
| PR-AUC | 0,615 |
| Brier Score | 0,143 |

#### 3.2 MLP Tunado (Optuna)

Hiperparametros otimizados via 30 trials com Optuna, maximizando PR-AUC.
Espaço de busca expandido: arquiteturas (64,32) a (512,256,128),
dropout 0,0-0,6, lr 5e-5 a 5e-2, weight_decay 1e-6 a 1e-2,
batch_size 16-128, early_stopping 3-20, otimizadores adam/sgd,
max_epochs 100-200.

**Melhor Configuracao (Trial 24):**

| Parametro | Valor |
|-----------|-------|
| hidden_dims | (256, 128, 64, 32) |
| dropout_rate | 0,10 |
| lr | 0,0339 |
| weight_decay | 0,00238 |
| batch_size | 128 |
| early_stopping_patience | 19 |
| use_batch_norm | True |
| optimizer | adam |
| max_epochs | 200 |
| scheduler_patience | 6 |

**Conjunto de Teste - MLP Tunado:**

| Metrica | Valor |
|---------|-------|
| Acuracia | 0,800 |
| Precisao | 0,639 |
| Recall | 0,567 |
| F1-Score | 0,601 |
| ROC-AUC | 0,835 |
| **PR-AUC** | **0,632** |
| Brier Score | 0,140 |

**Metricas de Validacao (ultima epoca do trial):**

| Metrica | Valor |
|---------|-------|
| Loss treino | 0,404 |
| Loss validacao | 0,424 |
| F1 validacao | 0,588 |
| ROC-AUC validacao | 0,838 |

**Analise de Custo (FN=R$500, FP=R$50):**

| Cenario | Custo Total |
|---------|-------------|
| Threshold padrao (0,5) | R$ 81.250 |
| Threshold otimo (0,05) | R$ 36.900 |

**Bandas de Risco:**

| Banda | % da Populacao | Taxa de Churn Real | Captura de Churners |
|-------|----------------|--------------------|---------------------|
| Low (p < 0,30) | 57,5% | 9,8% | - |
| Medium (0,30-0,60) | 26,8% | 39,0% | 39,3% (High) |
| High (p > 0,60) | 15,7% | 67,0% | 78,9% (Med+High) |

**Precision@k (top-k por probabilidade):**

| k | Precision@k | Recall@k |
|---|-------------|----------|
| 70 | 68,6% | 12,8% |
| 100 | 69,0% | 18,4% |
| 250 | 66,8% | 44,7% |
| 500 | 54,0% | 72,2% |

**MLflow:** [tech-challenge-mlp](http://localhost:5000/#/experiments/2)

---

## Tabela Comparativa Consolidada

| Metrica | Dummy (best) | Logistic Regression | MLP Original | **MLP Tunado** |
|---------|--------------|---------------------|--------------|----------------|
| Acuracia | 0,615 | **0,804** | 0,792 | 0,800 |
| Precisao | 0,288 | **0,648** | 0,609 | 0,639 |
| Recall | 0,543 | **0,575** | 0,604 | 0,567 |
| F1-Score | 0,376 | **0,609** | 0,607 | 0,601 |
| ROC-AUC | 0,507 | **0,836** | 0,831 | 0,835 |
| PR-AUC | 0,269 | 0,621 | 0,615 | **0,632** |
| Brier Score | 0,266 | **0,140** | 0,143 | 0,140 |

> Best = melhor valor entre os modelos. Dummy best = stratified
> (unico com alguma capacidade discriminativa > aleatorio).
> **MLP Tunado** superou Logistic Regression em PR-AUC (metrica mais
> relevante para datasets desbalanceados).

---

## Analise de Desempenho

### Capacidade Discriminativa

- **ROC-AUC:** Logistic Regression (0,836) ainda lidera, seguida de
  perto pelo MLP Tunado (0,835). Ambos bem acima do minimo util (0,7).
- **PR-AUC:** **MLP Tunado (0,632) superou Logistic Regression (0,621)**
  e MLP Original (0,615). PR-AUC e a metrica mais relevante para
  datasets desbalanceados (~26,5% churn), pois foca na capacidade de
  encontrar positivos sem muitos falsos alarmes.

### Impacto do Tuning

O tuning com Optuna trouxe ganhos significativos:

| Metrica | MLP Original | MLP Tunado | Delta |
|---------|------------|------------|-------|
| PR-AUC | 0,615 | **0,632** | **+2,8%** |
| ROC-AUC | 0,831 | **0,835** | +0,5% |
| Brier Score | 0,143 | **0,140** | -2,1% |
| Acuracia | 0,792 | **0,800** | +1,0% |

Principais mudancas na configuracao:
- Arquitetura mais profunda: (256, 128, 64, 32) vs (128, 64, 32)
- Learning rate maior: 0,034 vs 0,001 (convergencia mais rapida)
- Batch size maior: 128 vs 64 (gradientes mais estaveis)
- Early stopping mais tolerante: patience=19 vs 5 (evita parada prematura)
- Dropout menor: 0,1 vs 0,3 (menos regularizacao, modelo mais expressivo)
- Weight decay ativo: 0,00238 (regularizacao L2 substitui parte do dropout)

### Calibracao

- **Brier Score:** MLP Tunado (0,140) empatou com Logistic Regression.
- **ECE (MLP):** ~0,036 — probabilidades refletem frequencias observadas.

### Trade-off Custo-Negocio

- O MLP Tunado mantem a capacidade de otimizacao de threshold.
- O recall do MLP Tunado (0,567) esta proximo da Logistic (0,575),
  mas com PR-AUC superior, indicando melhor ranking dos churners.

### Generalizacao

- **MLP Tunado:** Val AUC 0,838, Test AUC 0,835 — gap pequeno (0,003),
  indicando boa generalizacao apesar da arquitetura maior.

---

## Insights sobre os Dados

1. **Desbalanceamento:** ~26,5% de churn. Modelos devem priorizar
   recall ou PR-AUC em vez de acuracia pura.

2. **Fatores de risco identificados (Logistic Regression):**
   - Clientes com **Fiber optic** tem risco elevado (+0,73)
   - **Contratos curtos** (Month-to-month) vs **Two year** (-0,60)
   - **Baixa permanencia (tenure)** e o maior fator protetor (-1,35)
   - Servicos adicionais (StreamingTV, StreamingMovies, MultipleLines)
     associados a maior risco (possivel proxy para pacote premium)

3. **Valor da segmentacao por probabilidade:**
   - 15,7% dos clientes na banda High tem 67% de taxa real de churn
   - Priorizando os top 100 clientes por probabilidade, 69% de fato
     churnam (Precision@100)

4. **Oportunidade de otimizacao:**
   - Ajustar threshold de decisao para minimizar custo total de negocio
   - Investigar por que MonthlyCharges tem coeficiente negativo (pode
     indicar confounding com tenure/contract type)

---

## Benchmark para Modelos Futuros

| Criterio | Baseline Atual | Meta Proxima Etapa |
|----------|---------------|--------------------|
| ROC-AUC | 0,835-0,836 | > 0,85 |
| PR-AUC | 0,632 | > 0,65 |
| F1-Score | 0,601-0,609 | > 0,62 |
| Brier Score | 0,140 | < 0,13 |
| Precision@100 | 69% | > 75% |

**Proximos passos recomendados:**
- Feature engineering (interacoes tenure x contract, clusterizacao de
  servicos)
- Testar pos_weight dinamico e class_weight balanced como alternativas
  ao custo de negocio
- Avaliar ensemble (Logistic + MLP stacking / blending)
- Coletar custos reais de negocio para refinar analise de threshold
  (issue #19)
- Experimentar regularizacao adicional (dropout maior, weight decay
  mais agressivo) para reduzir Brier Score

---

## Referencias MLflow

| Experimento | ID | Link Local |
|-------------|----|------------|
| Dummy Baseline | 1 | http://localhost:5000/#/experiments/1 |
| Logistic Regression | 3 | http://localhost:5000/#/experiments/3 |
| MLP | 2 | http://localhost:5000/#/experiments/2 |

> Nota: Os links assumem MLflow rodando localmente na porta padrao
> (5000). Ajuste conforme sua configuracao em `.env`.
