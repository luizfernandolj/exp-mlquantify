# As mesmas figuras em MSE

Comando:

```bash
python painel_bibliotecas.py --run-id final-pareado --custo-run custo-final5 \
    --erro mse --out-dir C:/Users/luiz_/exp-mlquantify-run
```

**So duas figuras mudam.** A medida de erro entra na equivalencia (01) e na
concordancia (02); a razao de custo, os perfis e o ranking medem tempo e
memoria e nao sabem qual erro esta sendo usado. Copiar as outras aqui seria
guardar o mesmo arquivo duas vezes, entao elas ficam so na pasta de cima.

## A margem: 0,0001, e de onde ela vem

**Nao ha padrao de literatura para a margem de equivalencia de um erro
quadratico.** O que existe e orientacao de metodo, e ela e sempre a mesma:
fixar a margem **antes** de olhar os dados e dizer de onde ela vem. Em
quantificacao a questao mal aparece, porque a literatura testa diferenca e
nao equivalencia.

Entao a margem sai da **definicao da medida**, e nao dos resultados:

> Uma estimativa errada em exatamente um ponto percentual em toda classe
> tem `AE = 0,01` e `SE = 0,01^2 = 0,0001`.

E a mesma decisao -- "um ponto percentual de prevalencia nao muda nada" --
escrita na unidade de cada medida. No codigo isso esta literal:
`MARGENS = {"mae": MARGEM, "mse": MARGEM ** 2}`, e o `--margem` so existe
para quem quiser outra.

**MRAE e NKLD ficam de fora de proposito.** Nao ha transposicao honesta de
um ponto percentual para um erro relativo nem para uma divergencia, entao o
script **recusa** rodar o teste com elas sem um `--margem` explicito, em vez
de usar um numero qualquer.

### O que foi descartado, e por que

Duas outras margens foram consideradas e descartadas, e vale registrar para
que ninguem as reinvente:

* **0,01, a mesma de AE.** Padroniza pelo numero em vez de pela decisao, e
  o resultado e vazio: os 39 pares caem dentro da faixa, porque 0,01 e
  setenta por cento do MSE tipico. Um teste de equivalencia que aprova tudo
  nao afirma nada;
* **0,0016, tirada do nivel de erro observado.** Transpoe pela derivada
  (`2e * 0,01` com o AE tipico de 0,0822) e da um resultado confortavel --
  34 dos 39 dentro da faixa, parecido com o de AE. Foi descartada por ser
  **circular**: o dado que o teste julga entraria na escolha da margem que o
  julga, e o numero so vale para este catalogo de conjuntos.

A margem pela definicao e mais exigente que as duas, e isso aparece no
resultado: **23 dos 39 pares com o IC inteiro na faixa**, contra 38 em AE,
com 16 compativeis, 5 com diferenca irrelevante, 8 diferentes e **10
inconclusivos**.

Os dez inconclusivos sao a resposta honesta, e nao um defeito: com uma
margem estreita, nao ha evidencia suficiente para afirmar equivalencia nem
para apontar diferenca. O erro que se quer evitar e o oposto -- declarar
equivalencia por causa de uma margem larga demais.

## O que muda em relacao ao MAE

Os pares que escapam da faixa sao outros, e isso e informacao: a diferenca
nao esta no conjunto tipico, esta nos conjuntos em que uma das duas erra
muito. O quadrado amplifica essa cauda e o valor absoluto a dilui -- e um
argumento para reportar as duas medidas, nao para escolher uma.
