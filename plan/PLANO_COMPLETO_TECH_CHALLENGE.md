# Plano Completo - Tech Challenge Fase 1

## Guia Passo a Passo para o Time

**Projeto:** Pipeline end-to-end de ML para previsão de churn em telecomunicações  
**Time:** Grupo 13  
**Tech Leads:** Eduardo (DevOps) e Fernando (ML)  
**Dataset:** Telco Customer Churn (IBM)

---

## 1. Levantamento de Skills da Equipe

### 1.1 Template de Pesquisa

**Formato:** Google Forms ou Planilha compartilhada

**Campos obrigatórios:**
- Nome do membro
- Email
- Timestamp
- 21 habilidades com escala 0-3:
  - **0** = Sem experiência
  - **1** = Iniciante (já vi/teoria)
  - **2** = Intermediário (já usei em projetos)
  - **3** = Avançado (posso ensinar outros)

**Lista completa de skills:**

1. Git/GitHub e estrutura de projeto
2. Python project tooling (pyproject.toml, ruff, Makefile)
3. Jupyter Notebooks
4. Docker
5. Pandas / manipulação de dados
6. EDA (análise exploratória)
7. Entendimento de negócio / ML Canvas
8. scikit-learn (baselines, pipelines)
9. Métricas ML (AUC-ROC, PR-AUC, F1, FP/FN)
10. PyTorch (construção de MLP)
11. Conceitos: arquitetura, loss, training loop, early stopping
12. MLflow (experiment tracking)
13. pytest (unit, schema, smoke tests)
14. Pandera (schema validation)
15. Refatoração em módulos Python (src/)
16. FastAPI (endpoints /predict e /health)
17. Pydantic (validação de dados)
18. Logging estruturado
19. Documentação de arquitetura de deploy
20. Vídeo no formato STAR
21. Cloud deploy (AWS/Azure/GCP) — opcional

### 1.2 Processo de Coleta

1. **Criar formulário** (Google Forms)
2. **Enviar para todos** via Discord/email
3. **Prazo:** 24-48 horas
4. **Lembrete:** A cada 12h no Discord
5. **Consolidar** em planilha CSV

### 1.3 Análise das Respostas

**Checklist de validação:**
- [ ] Todos os 4 membros responderam?
- [ ] Nenhum campo em branco?
- [ ] Valores entre 0-3 apenas?
- [ ] Nomes padronizados?

**Identificação de gaps:**
- Calcular média por skill
- Destacar skills com média < 1.0 (crítico)
- Identificar mentores (skill ≥ 2)

---

## 2. Distribuição de Tarefas

### 2.1 Definição de Tech Leads

**Critérios:** Membros com maior cobertura de skills críticas

| Tech Lead | Área de Domínio | Responsabilidade |
|-----------|-----------------|------------------|
| Fernando | ML/Modelagem | Arquitetura PyTorch, métricas, EDA, FastAPI (code review) |
| Eduardo | DevOps/Qualidade | Git, Docker, infraestrutura, pytest, documentação, deploy |

### 2.2 Regras de Pair Programming

**Quando usar:**
- Skill gap < 1.0 para mais de 2 membros
- Tecnologia crítica para o projeto
- Primeira vez do membro com a tecnologia

**Estrutura do par:**
- **Mentor:** Skill ≥ 2 na área
- **Aprendiz:** Skill 0-1
- **Duração:** Mínimo 2h por sessão
- **Ferramenta:** VS Code Live Share, Discord screen share

### 2.3 Template de Alocação

```
SPRINT [X]: [NOME]
├── Responsável Principal: [Nome]
├── Apoio: [Nome] (se houver)
├── Pair Programming:
│   ├── Mentor: [Nome]
│   └── Aprendiz: [Nome]
├── Entregáveis:
│   ├── [Item 1]
│   ├── [Item 2]
│   └── [Item 3]
└── Critérios de Aceitação:
    ├── [Critério 1]
    └── [Critério 2]
```

---

## 3. Execução das Sprints

### Sprint 0: Setup (Dias 1-2)

**Objetivo:** Preparar ambiente de desenvolvimento

**Entregáveis:**

1. **Estrutura de pastas:**
   ```
   projeto/
   ├── src/
   ├── data/
   ├── models/
   ├── tests/
   ├── notebooks/
   ├── docs/
   └── plan/
   ```

2. **Arquivos de configuração:**
   - `pyproject.toml` (dependências, ruff, pytest)
   - `.gitignore` (Python, ML, IDE)
   - `Makefile` (targets: install, lint, test, run)
   - `README.md` inicial

3. **Dataset baixado e validado**

**Checklist:**
- [ ] Python 3.9+ instalado
- [ ] Virtual environment criado
- [ ] Dependências instaladas (`make install`)
- [ ] Git repo inicializado
- [ ] Primeiro commit realizado
- [ ] Dataset no diretório `data/`

**Rituais:**
- Standup: Definir quem faz o quê
- Review: Validar estrutura juntos

---

### Sprint 1: EDA & ML Canvas (Dias 3-5)

**Objetivo:** Entender dados e definir estratégia de negócio

**Entregáveis:**

1. **Notebook EDA** (`notebooks/01_eda.ipynb`):
   - Análise de volume e qualidade
   - Distribuição de features
   - Tratamento de missing values
   - Visualizações

2. **ML Canvas preenchido:**
   - Stakeholders identificados
   - Problema de negócio definido
   - Métricas de sucesso
   - SLOs estabelecidos

3. **Baselines treinados:**
   - DummyClassifier (estratificado)
   - LogisticRegression
   - Métricas calculadas: AUC-ROC, PR-AUC, F1
   - Registrados no MLflow

**Checklist:**
- [ ] EDA executa sem erros
- [ ] Todos os gráficos salvos
- [ ] ML Canvas revisado pelo time
- [ ] Baselines com métricas comparáveis
- [ ] Experimentos visíveis no MLflow UI

**Rituais:**
- Daily: 15min no Discord (o que fiz, o que vou fazer, bloqueios)
- Review: Apresentar EDA para o time

---

### Sprint 2: Modelagem com MLP (Dias 6-10)

**Objetivo:** Construir e treinar rede neural

**Entregáveis:**

1. **Modelo MLP em PyTorch:**
   - Arquitetura definida (camadas, ativações)
   - Loss function apropriada
   - Training loop
   - Early stopping implementado

2. **Comparação de modelos:**
   - Tabela comparativa (baselines + MLP)
   - Análise de trade-off FP vs FN
   - Gráficos de learning curves

3. **Artefatos no MLflow:**
   - Parâmetros do modelo
   - Métricas de treino/validação/teste
   - Modelo serializado (.pt ou .pkl)
   - Requirements usados

**Checklist:**
- [ ] MLP treina sem erros
- [ ] Early stopping funciona
- [ ] Métricas >= baselines
- [ ] Overfitting controlado
- [ ] Artefatos versionados no MLflow

**Rituais:**
- Daily: Foco em resolver bugs de treinamento
- Pair sessions: 2h PyTorch (todos participam)
- Review: Demo do modelo funcionando

---

### Sprint 3: API & Testes (Dias 11-14)

**Objetivo:** Produzir código de qualidade com API funcional

**Entregáveis:**

1. **Código refatorado:**
   - Estrutura `src/` organizada
   - Módulos separados (data, model, api, utils)
   - Pipelines sklearn reprodutíveis

2. **API FastAPI:**
   - Endpoint POST `/predict` (input JSON, output predição)
   - Endpoint GET `/health` (status da API)
   - Validação Pydantic dos inputs
   - Logging estruturado (JSON)
   - Middleware de latência

3. **Testes automatizados** (mínimo 3):
   - Testes unitários (funções críticas)
   - Testes de schema (Pandera)
   - Smoke test (API sobe e responde)

**Checklist:**
- [ ] `make test` passa todos os testes
- [ ] `make lint` não reporta erros críticos
- [ ] API responde em < 200ms (local)
- [ ] Logs estruturados funcionando
- [ ] Pipeline reproduz resultado da sprint 2

**Rituais:**
- Daily: Status dos testes
- Pair sessions: FastAPI (Fernando + 1 aprendiz)
- Review: Testar API localmente juntos

---

### Sprint 4: Documentação & Entrega (Dias 15-17)

**Objetivo:** Documentar e preparar entrega final

**Entregáveis:**

1. **Model Card** (`docs/model_card.md`):
   - Performance em dados de teste
   - Limitações identificadas
   - Vieses e fairness
   - Cenários de falha

2. **Documentação de arquitetura** (`docs/architecture.md`):
   - Diagrama de fluxo (batch vs real-time)
   - Plano de monitoramento
   - Métricas e alertas
   - Playbook de incidentes

3. **README final:**
   - Descrição do projeto
   - Instruções de setup
   - Como rodar localmente
   - Como fazer deploy

4. **Vídeo STAR** (5 minutos):
   - **S**ituation: Contexto do problema
   - **T**ask: O que precisava ser feito
   - **A**ction: O que fizemos
   - **R**esult: Resultados alcançados

5. **(Opcional) Deploy cloud:**
   - API acessível publicamente
   - URL documentada

**Checklist:**
- [ ] Toda documentação revisada
- [ ] README claro e completo
- [ ] Vídeo gravado e revisado
- [ ] Todos os artefatos no GitHub
- [ ] (Opcional) Deploy testado

**Rituais:**
- Daily: Revisão de documentos
- Review geral: Todo time valida entrega
- Retrospectiva: Lições aprendidas

---

## 4. Checkpoints e Validações

### Checkpoints por Sprint

**Antes de iniciar cada sprint:**
- [ ] Sprint anterior foi aceita?
- [ ] Issues criadas no GitHub?
- [ ] Responsáveis definidos?
- [ ] Pares de programação agendados?

**Durante a sprint:**
- [ ] Daily realizada?
- [ ] Bloqueios identificados e resolvidos?
- [ ] Commits frequentes (mínimo 1/dia)?

**Ao final da sprint:**
- [ ] Todos os entregáveis prontos?
- [ ] Checklist de qualidade passou?
- [ ] Code review realizado?
- [ ] Documentação atualizada?

### Critérios de Aceitação por Etapa

**Setup:**
- Repo clonável e rodável em qualquer máquina
- `make install` funciona
- Estrutura de pastas padronizada

**EDA:**
- Notebook executável
- Insights claros sobre os dados
- Decisões de pré-processamento documentadas

**Modelagem:**
- Modelo salvo e versionado
- Métricas reprodutíveis
- Overfitting não crítico

**API:**
- Testes passando
- API documentada (OpenAPI/Swagger)
- Logging funcional

**Documentação:**
- Model Card completo
- Instruções claras
- Vídeo com qualidade adequada

### Quando Pedir Ajuda aos Tech Leads

**Problemas técnicos:**
- Erro que não resolve em 30 min
- Dúvida arquitetural
- Conflito de merge complexo

**Problemas de gestão:**
- Membro não responde em 24h
- Entrega em risco
- Necessidade de redistribuir tarefas

**Comunicar via:**
- Discord (canal #tech-help ou DM)
- Marcar nos comentários da issue
- Tag `@tech-lead` no GitHub

---

## 5. FAQ (Perguntas Frequentes)

### Sobre o Projeto

**Q: Qual é o objetivo do Tech Challenge?**  
A: Construir um pipeline end-to-end de ML para previsão de churn em telecomunicações, desde a análise de dados até a API de predição deployada.

**Q: Quais são as entregas obrigatórias?**  
A: Repositório GitHub com código funcional, modelos treinados, API, testes, documentação e vídeo STAR. Deploy em cloud é bônus.

**Q: Qual o dataset usado?**  
A: Telco Customer Churn da IBM (disponível no Kaggle).

### Sobre Skills

**Q: Não tenho experiência em PyTorch. Vou conseguir?**  
A: Sim! O pair programming nas sprints 2 e 3 vai te ajudar. Fernando vai mentorar e todos vão aprender juntos.

**Q: Como funciona o nívelamento de skills?**  
A: Membros com mais experiência (skill ≥ 2) fazem pair programming com quem está começando (skill 0-1). Todos aprendem e entregam juntos.

**Q: O que acontece se eu não preencher a pesquisa de skills?**  
A: O planejamento fica incompleto e pode afetar a distribuição de tarefas. Preencha assim que possível!

### Sobre Pair Programming

**Q: Como são formados os pares?**  
A: Baseado na pesquisa de skills: mentor (skill ≥ 2) + aprendiz (skill 0-1). Os tech leads supervisionam.

**Q: Quanto tempo dura uma sessão de pair?**  
A: Mínimo 1 hora, mas pode ser mais se necessário. Agende com antecedência.

**Q: Posso fazer pair programming sozinho se souber a tecnologia?**  
A: Sim, mas é recomendado fazer code review com outro membro mesmo assim.

### Sobre GitHub e Issues

**Q: Quando criar uma issue?**  
A: Para cada tarefa significativa: bugs, novas features, dúvidas técnicas, documentação. Fique a vontade para criar issues, elas irão passar pelo processo de grooming antes de serem de fato aceitas para Ready.

**Q: Como nomear as issues?**  
A: Use prefixos: `[EDA]`, `[MODEL]`, `[API]`, `[TEST]`, `[DOC]`, `[BUG]`. Não é obrigatório mas ajuda, pode usar também as labels do GitHub

**Q: Quando fazer commit?**  
A: Faça commits pequenos e frequentes (mínimo 1 por dia de trabalho). Mensagens claras: "feat: adiciona baseline logistic regression".

### Sobre Rituais

**Q: Quais são os rituais obrigatórios?**  
A: Daily (15min), Sprint Planning (início), Sprint Review (final), Retrospectiva (após sprint 4). Pode ser feito assíncrono também.

**Q: Onde acontecem as reuniões?**  
A: No Discord do grupo. Agende com antecedência. Ou assíncrono, para assuntos menos complexos, ou pontuais.

**Q: O que levar para o Status Report?**  
A: O que você fez desde o último report, o que vai fazer até o próximo, e se tem algum bloqueio. É interessante reportar com antecedência máxima, caso esteja bloqueado em algo.

### Sobre Entregas

**Q: E se eu não conseguir entregar no prazo?**  
A: Comunique imediatamente no Discord. Os tech leads podem redistribuir ou ajustar escopo. Não deixe atrasar, ou ficar bloqueado por muito tempo.

**Q: Posso entregar antes do prazo?**  
A: Sim! Quanto antes melhor, assim sobra tempo para revisão e ajustes.

**Q: Como saber se minha entrega está boa?**  
A: Use os checklists de cada sprint. Se passou em todos os itens, está ótimo! e se tem aval do lead da tarefa.

### Sobre Tecnologias

**Q: Posso usar outras bibliotecas além das especificadas?**  
A: Pode, mas mantenha as obrigatórias: PyTorch, FastAPI, scikit-learn, MLflow, pytest, Pandera.

**Q: Preciso usar Docker?**  
A: Altamente recomendado para garantir reprodutibilidade. Eduardo pode ajudar.

**Q: E se eu tiver problema com alguma instalação?**  
A: Pergunte no canal #tech-help no Discord. Alguém do time vai ajudar.

### Sobre Avaliação

**Q: Como o projeto é avaliado?**  
A: Por critérios com pesos: Código (20%), Rede Neural (25%), Pipeline (15%), API (15%), Documentação (10%), Vídeo STAR (10%), Deploy (5% bônus).

**Q: O que é o vídeo STAR?**  
A: Apresentação de 5 minutos no formato: Situação, Tarefa, Ação, Resultado. Grave mostrando tela e explicando o projeto.

**Q: O deploy em cloud é obrigatório?**  
A: Não, mas vale 5% da nota. Se fizer, use AWS, Azure ou GCP.

---

## 6. Referências e Recursos

### Documentação Oficial

- **PyTorch:** https://pytorch.org/docs/
- **FastAPI:** https://fastapi.tiangolo.com/
- **scikit-learn:** https://scikit-learn.org/stable/
- **MLflow:** https://mlflow.org/docs/latest/index.html
- **pytest:** https://docs.pytest.org/
- **Pandera:** https://pandera.readthedocs.io/

### Templates Úteis

- **ML Canvas:** https://www.madewithml.com/courses/mlops/product-design/
- **Model Card:** https://modelcards.withgoogle.com/about
- **Cookiecutter Data Science:** https://drivendata.github.io/cookiecutter-data-science/

### Comunidades

- Discord ML Ops Brasil
- PyTorch Forums
- FastAPI Discord

---

## 7. Contatos e Responsáveis

| Papel | Responsável | Discord/GitHub |
|-------|-------------|----------------|
| Tech Lead ML | Fernando | @fernando |
| Tech Lead DevOps | Eduardo | @eduardonunesp |
| Time | Bruno | @bruno |
| Time | Ygor | @ygor |

**Canal Discord:** `#tech-challenge`

---

## Próximos Passos Imediatos

1. [ ] Todos preencherem a pesquisa de skills
2. [ ] Tech leads consolidarem dados
3. [ ] Criar issues para Sprint 0
4. [ ] Agendar primeira reunião de kickoff

---

*Documento criado em: Março 2026*  
*Última atualização: [data da última modificação]*
