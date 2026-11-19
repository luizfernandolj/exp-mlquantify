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

## Parametros alinhados

Os quantificadores rodam com os hiperparametros padrao de cada biblioteca. A
excecao sao os casos em que o padrao faria o par medir parametrizacao em vez de
implementacao: ali o valor e alinhado, do lado que diverge da definicao do
metodo ou da escolha comum as demais. Tudo declarado em `expq/registry.py`.

| biblioteca | metodo | parametro | valor | motivo |
| --- | --- | --- | --- | --- |
| `mlquantify` | EMQ | `max_iter` | 1000 | o padrao da biblioteca e 100 e trunca o EM antes de convergir; 1000 e o valor da `QuaPy` e da `QuantificationLib` |
| `mlquantify` | DyS | `bins_size` | 8 | o padrao varre `[2, 4, ..., 20, 30]`; 8 e o valor da `QuaPy` |
| `mlquantify` | HDy8 | `bins_size` | 8 | os 8 intervalos fixos da `QuantificationLib`, para o par medir implementacao e nao parametrizacao |
| `mlquantify` | HDx | `bins_size` | `[8]` | o padrao varre `[2, 4, ..., 20, 30]`; 8 e o valor das outras tres |
| `QuaPy` | DyS | `divergence` | `topsoe` | o padrao e Hellinger; Topsoe e a divergencia da definicao do metodo |
| `QuaPy` | DyS | `n_bins` | 8 | mesmo numero de intervalos das demais |
| `QuaPy` | EDy | `distance` | `euclidean` | o padrao da `QuaPy` e manhattan |
| `QuaPy` | EDx | `distance` | `euclidean` | o padrao da `QuaPy` e manhattan |
| `QuantificationLib` | EDy | `distance` | euclidiana | o padrao da biblioteca e manhattan |
| `qunfold` | EMQ | `max_iter` | 1000 | mesmo limite de iteracoes das demais |
| `qunfold` | EMQ | `tol` | 1e-4 | ver a ressalva abaixo |

Alem da tabela, o protocolo comum escreve em todo metodo que aceite o
argumento: `cv = 5` dobras, `random_state = 42`, `stratified` e `shuffle`
ligados. O classificador base e a regressao logistica do scikit-learn em todas
as bibliotecas.

**Ressalva da tolerancia do `qunfold`.** Igualar o numero nao iguala o
criterio: a `qunfold` compara iteracoes pela norma L2 e as outras pelo erro
absoluto medio, o que num vetor binario e um fator raiz de dois. O valor
proprio dela seria 1e-6, e afrouxar para 1e-4 a faz parar antes, elevando a
diferenca contra a `mlquantify` de 0,000127 para 0,000422. Fica em 1e-4 porque
e o valor com que os resultados existentes foram medidos.

## Organizacao

```
datasets/               conjuntos de dados brutos
descricao/              catalogo, selecao e cobertura de metodos
expq/                   codigo dos experimentos
scripts/q3_usage/       um script por biblioteca, para a medida de esforco de uso
results/<run-id>/       resultados (summary.csv e as tabelas Q1, Q2 e Q3)
logs/                   log de cada execucao
```
