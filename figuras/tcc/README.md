# Figuras do capítulo de resultados da monografia

Três figuras, em `.png`, `.svg` e `.pdf`, mais os csv que as sustentam e o
texto das seções Q1 e Q2 que foi para o Overleaf.

```bash
python figuras/tcc/gerar.py
```

| arquivo | onde entra |
| --- | --- |
| `q1-equivalencia` | Q1, teste de equivalência do erro em AE |
| `q2-tempo` | Q2, razão do tempo total, somadas as duas fases |
| `q2-memoria` | Q2, razão do pico de memória, o maior entre as duas fases |
| `q2-fases.csv` | a tabela de razão mediana por fase |
| `secoes-q1-q2.tex` | o texto das duas seções e das considerações finais |
| `conclusoes.tex` | o capítulo de conclusões e trabalhos futuros |
| `resumo.tex` | o resumo e as palavras-chave |
| `introducao.tex` | o capítulo de introdução |
| `apendice-conjuntos.tex` | o apêndice com os 110 conjuntos, gerado por `gerar_apendice.py` |

## Por que os números diferem dos do painel

**Estas figuras usam 110 conjuntos de dados e a execução tem 142.** O
recorte é o que a monografia define e sai de duas listas em `descricao/`,
lidas pelo `recorte_analise.py`:

| lista | o que sai | quantos |
| --- | --- | --- |
| `conjuntos_fora_do_criterio.txt` | passam de dez classes e entraram na coleção depois da seleção | 10 |
| `conjuntos_minoria_abaixo_de_10.txt` | classe minoritária com menos de 10% das linhas | 22 |

A primeira lista é de coleção e não de critério. Os 26 arquivos a mais
renderam 10 conjuntos selecionados, todos multiclasse: dois do Kaggle, seis
do OpenML (o `amazon-commerce-reviews` e cinco variantes dele), o
`Walking.csv` e o `letter.csv`.

A segunda é de critério. O catálogo corta classe rara por um piso absoluto
de 30 instâncias, enquanto o texto define a minoria em 10% das linhas, e o
corte relativo entra na análise, e não no `build_catalog.py`, para não
renumerar o `indice` de `datasets_selecionados.csv`, que o `--range` e o
`--shard` usam. Restam 97 binários e 13 multiclasse.

O corte vale pelo nome do conjunto, de propósito: uma reagregação a partir
de `parts/` reescreve o `summary.csv` e devolveria em silêncio o que o
recorte já tinha tirado. As linhas dos 22 ficam guardadas em
`summary_minoria_abaixo_de_10.csv`, ao lado do summary da execução.

**O painel aplica o mesmo recorte**, no `carregar()` do
`painel_bibliotecas.py`, para que os dois não divirjam. O MS2 contra a
`QuaPy` é o único par sensível a ele: sem o corte de minoria o intervalo
ficava dentro da margem por um fio, em $[-0{,}00996, -0{,}00379]$, e com os
110 escapa dela por $0{,}0002$, em $[-0{,}01020, -0{,}00379]$. É a única
exceção entre as 39 nos dois casos, uma vez pela borda de dentro e outra
pela de fora.


## Outras diferenças em relação ao painel

* **O custo vem só da `final-pareado`**, sem a execução otimizada
  `custo-final5`, para que o resultado corresponda ao método descrito no
  capítulo anterior.
* **As duas fases entram juntas.** O tempo soma treinamento e estimação,
  que são trabalho sequencial. A memória toma o maior dos dois picos,
  porque o processo libera a primeira fase antes de iniciar a segunda.
* **O texto das figuras tem acento e vírgula decimal**, ao contrário do
  restante do repositório, porque rótulo de figura é texto de leitura.
