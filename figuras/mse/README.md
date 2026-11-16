# As mesmas figuras em MSE

Comando:

```bash
python painel_bibliotecas.py --run-id final-pareado --custo-run custo-final5 \
    --erro mse --margem 0.0016 --out-dir C:/Users/luiz_/exp-mlquantify-run
```

**So duas figuras mudam.** A medida de erro entra na equivalencia (01) e na
concordancia (02); a razao de custo, os perfis e o ranking medem tempo e
memoria e nao sabem qual erro esta sendo usado. Copiar as outras aqui seria
guardar o mesmo arquivo duas vezes, entao elas ficam so na pasta de cima.

## A margem: de onde vem o 0,0016

Este e o unico numero que nao se transporta de graca entre as duas metricas,
e merece ser conferido por quem escreve o texto.

Em AE a margem e **0,01**, um ponto percentual de prevalencia, e ela esta na
propria unidade da grandeza estimada: a diferenca de erro entre duas
bibliotecas vale um ponto de prevalencia. O MSE esta na unidade **ao
quadrado**, entao o mesmo deslocamento nao vale 0,01.

A transposicao usada parte do nivel tipico de erro. Se o AE sai de `e` para
`e + 0,01`, o SE sai de `e^2` para `(e + 0,01)^2`, isto e, move `2e * 0,01`.
Com o AE tipico medido na `final-pareado` -- a mediana entre os 39 pares e
**0,0822** -- isso da `2 * 0,0822 * 0,01 = 0,0016`. E a resposta a pergunta
"quanto o MSE anda quando o erro anda um ponto percentual, no nivel em que
estes metodos operam".

**O que acontece com as alternativas**, medido:

| margem | IC na faixa | compativel | irrelevante | inconclusivo | diferente |
| --- | --- | --- | --- | --- | --- |
| 0,01 (a de AE, **errada**) | 39 de 39 | 39 | 0 | 0 | 0 |
| 0,0001 (`0,01^2`) | 23 de 39 | 16 | 5 | 10 | 8 |
| **0,0016 (usada)** | **34 de 39** | 25 | 9 | 1 | 4 |
| 0,003 | 36 de 39 | 25 | 11 | 1 | 2 |

A margem de AE aplicada ao MSE **declara tudo equivalente**, que e o pior
erro possivel num teste de equivalencia: ele deixa de separar qualquer
coisa e a afirmacao fica vazia. O `0,01^2` e a transposicao ingenua e erra
para o outro lado -- ela equivale a exigir um ponto percentual quando o erro
ja e quase zero, e deixa 10 pares inconclusivos.

Com 0,0016 o quadro fica proximo do de AE (que da 38 de 39 na faixa), o que
e o esperado: as duas medidas descrevem os mesmos dados. As diferencas de
veredito estao nos pares em que o erro tem cauda, porque o MSE pesa o
quadrado e a cauda conta mais.

## O que muda em relacao ao MAE

Os pares que saem da faixa sao outros, e isso e informacao e nao ruido:
`HDx` e `T50` contra a quapy, `MS2` contra a quapy e `PWKAC` contra a
quantificationlib. O `MS` fica inconclusivo, com intervalo largo dos dois
lados. Sao os pares em que a diferenca nao esta no conjunto tipico e sim
nos conjuntos em que uma das duas erra muito -- exatamente o que o quadrado
amplifica e o valor absoluto dilui.
