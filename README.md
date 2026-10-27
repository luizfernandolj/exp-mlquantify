# exp-mlquantify

Experimentos de comparacao entre bibliotecas de quantificacao, referentes a
Etapa 4 da monografia: a biblioteca proposta (`mlquantify`) e comparada com
`QuaPy`, `QuantificationLib` e `qunfold` em erro de quantificacao, custo
computacional e esforco de uso.

```bash
.\setup_env.ps1                                    # ambiente (Python 3.11)
.venv\Scripts\python build_catalog.py              # catalogo e selecao
.venv\Scripts\python run_experiments.py --run-id final
.venv\Scripts\python aggregate_results.py --run-id final
```

Para dividir o trabalho entre varias maquinas, retomar uma execucao
interrompida, rodar os metodos ELM e neurais mais tarde ou entender as
colunas dos resultados, veja **[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Organizacao

```
datasets/               conjuntos de dados brutos
descricao/              catalogo, selecao e cobertura de metodos
expq/                   codigo dos experimentos
scripts/q3_usage/       um script por biblioteca, para a medida de esforco de uso
results/<run-id>/       resultados (summary.csv e as tabelas Q1, Q2 e Q3)
logs/                   log de cada execucao
```
