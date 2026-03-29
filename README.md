## Sincronização do Ambiente

Para sincronizar o ambiente incluindo pacotes de desenolvimento utilizando o uv, use:

```bash
uv sync
```

Para sincronizar sem dependências de desenolvimento, útil para CI e contâiners docker, use:
```bash
uv sync --no-dev
```
