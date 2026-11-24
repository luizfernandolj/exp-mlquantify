# Figuras da comparacao entre bibliotecas

Sete figuras, cada uma em `.png` (para olhar), `.svg` (para o artigo, que e
vetorial e reescalavel) e `.pdf` (para LaTeX).

Todas saem do `painel_bibliotecas.py`, na raiz do repositorio. O erro vem da
execucao `final-pareado`, que e a oficial; o custo vem da `custo-final5` nos
12 metodos que ela mediu e da `final-pareado` nos demais. Para regerar:

```bash
python painel_bibliotecas.py --run-id final-pareado \
    --custo-run custo-final5 --out-dir C:/Users/luiz_/exp-mlquantify-run
```

As figuras nascem em `results/custo-final5/` com o prefixo
`painel_bibliotecas_`; esta pasta e uma copia com nomes descritivos.

Em `dados/` fica o csv que sustenta cada figura, com o mesmo numero. Os
pequenos trazem o resultado ja agregado (um por par metodo x biblioteca) e
os grandes trazem a observacao crua, uma por (par, conjunto de dados):

| csv | uma linha por | colunas |
| --- | --- | --- |
| `01-equivalencia-erro` | par metodo x biblioteca | `hl`, `ic_baixo`, `ic_alto`, os p do TOST e do Wilcoxon com e sem Holm, `veredito` |
| `02-concordancia-erro` | par x conjunto | os dois erros, a diferenca e o nivel (a media dos dois) |
| `03-razao-tempo`, `04-razao-memoria` | par metodo x biblioteca | `razao`, `ic_baixo`, `ic_alto`, `ganha`, `perde` |
| `05-perfil-tempo`, `06-perfil-memoria` | (metodo, conjunto, biblioteca) | o custo bruto, de onde a curva sai |
| `07-ranking-pares-*` | par metodo x biblioteca | `posto_medio`, `posto_mediano`, `posto_p25`, `posto_p75` |

| arquivo | o que responde |
| --- | --- |
| `01-equivalencia-erro` | as implementacoes do mesmo metodo concordam dentro de +-0,01 de MAE? |
| `02-concordancia-erro` | a discordancia e constante, ou cresce com a dificuldade do conjunto? |
| `03-razao-tempo` | quanto a mlquantify cobra de tempo em relacao a cada rival? |
| `04-razao-memoria` | o mesmo, em memoria |
| `05-perfil-tempo` | em que fracao dos problemas cada biblioteca fica dentro de um fator da mais barata? |
| `06-perfil-memoria` | o mesmo, em memoria |
| `07-ranking-pares` | qual par metodo+biblioteca e o melhor, em tempo e em memoria? |

Todas se leem **impressas em preto e branco**: a forma do marcador diz a
biblioteca nas florestas, o tracejado diz a biblioteca nos perfis, a
hachura diz a biblioteca no ranking, e o preenchimento do marcador diz o
veredito. A cor e redundante com isso.

## As variantes

Quatro figuras -- a 01, a 03, a 04 e a 07 -- saem tambem em duas variantes,
em subpastas:

* **`deitadas/`** troca os eixos. Na 01, na 03 e na 04 os metodos passam
  para o eixo x e o valor para o y, virando figuras largas e baixas em vez
  de colunas altas; na 07 as caixas deitam e os nomes dos pares ficam
  legiveis a esquerda, sem a rotacao de 90 graus. Serve quando a figura tem
  de caber na largura de uma coluna de texto.
* **`por-biblioteca/`** mostra uma rival de cada vez, em vez de todas
  sobrepostas. Na 07, que compara os pares entre si, **o posto continua
  sendo o do ranking inteiro**: a figura de cada biblioteca responde onde
  os pares dela caem entre os 68, e nao como eles se ordenam entre si.

As variantes por biblioteca saem na orientacao padrao. Para deita-las
tambem, basta acrescentar `deitado=True` na funcao `variantes()` do
`painel_bibliotecas.py`. Para nao gerar nenhuma delas, `--sem-variantes`.

Em **`tcc/`** estao as tres figuras do capitulo de resultados da
monografia, que usam **110 conjuntos de dados** em vez de 142. O recorte
segue os criterios de selecao descritos no texto, sai das duas listas do
`recorte_analise.py` e o README de la explica a diferenca. O painel passou
a aplicar o mesmo recorte, para que os dois nao divirjam.

Em **`mse/`** estao as duas figuras que dependem da medida de erro -- a 01 e
a 02 -- refeitas em MSE, com a margem de equivalencia reescolhida. O README
de la explica de onde sai o numero. As figuras de custo nao estao repetidas
ali porque nao dependem da medida de erro.

## 01 -- equivalencia do erro (TOST com Wilcoxon)

Uma linha por par metodo x biblioteca. O ponto e a diferenca mediana de MAE
(Hodges-Lehmann, `mlquantify - outra`) e a barra e o intervalo de confianca
de 95% que acompanha o Wilcoxon. A faixa verde tracejada e a margem de
equivalencia de +-0,01 -- um ponto percentual de prevalencia, abaixo do qual
a diferenca nao muda decisao nenhuma.

**Marcador cheio: o intervalo inteiro cabe na faixa**, isto e, a
compatibilidade esta *demonstrada*, e nao apenas nao refutada. Marcador
vazado: o intervalo escapa. O teste e o TOST (dois Wilcoxon unilaterais
sobre `d + 0,01` e `d - 0,01`, com `p = max`), corrigido por Holm sobre os
39 pares.

A unidade e o conjunto de dados: os quatro tamanhos de amostra do mesmo
conjunto entram pela media, porque nao sao observacoes independentes.

## 02 -- concordancia (Bland-Altman)

A 01 resume cada par num ponto; esta abre os conjuntos. No eixo x o nivel do
erro (a media das duas bibliotecas) e no y a diferenca entre elas. O eixo y
e linear dentro da faixa de equivalencia e logaritmico fora dela, porque as
diferencas se amontoam perto de zero e vao a 0,3.

A linha cheia e a diferenca mediana e as tracejadas sao os percentis 2,5 e
97,5 -- os limites de concordancia, em versao distribuicao-livre. O `rho` em
cada painel e a correlacao de Spearman entre o tamanho da discordancia e o
nivel do erro: positivo alto quer dizer que as duas so divergem nos
conjuntos dificeis.

## 03 e 04 -- razao de custo

Mesma gramatica da 01, com o objetivo invertido: aqui interessa ficar **a
esquerda** da linha vertical em 1. O ponto e a razao mediana entre conjuntos
(Hodges-Lehmann no log2, trazido de volta para razao) e a barra e o IC de
95%; marcador cheio e o par em que o intervalo inteiro fica de um lado de 1,
isto e, em que a vantagem e firme.

A escala e logaritmica em base 2 porque so nela "o dobro" e "a metade" ficam
a mesma distancia de 1.

## 05 e 06 -- perfil de desempenho (Dolan-More)

Media e mediana sao dominadas por poucos problemas caros; o perfil nao. Cada
problema e um par (metodo, conjunto); o eixo x e o fator em relacao a
biblioteca mais barata **naquele problema** e o y e a fracao de problemas
dentro desse fator. O valor em 1 e a fracao de problemas em que a biblioteca
e a mais barata de todas, e onde a curva chega ao topo e o pior caso dela.

Tres paineis pareados, mais um quarto com as quatro curvas nos tres metodos
que todas implementam -- esse quarto e ilustracao do formato canonico, nao
resultado: 404 problemas em tres metodos e um recorte enviesado do catalogo.

## 07 -- ranking dos pares metodo x biblioteca

As outras comparam a mlquantify contra uma rival de cada vez. Esta junta as
duas variaveis num tratamento so -- o par `metodo/biblioteca` -- e ordena
todos ao mesmo tempo. O bloco e o conjunto de dados: dentro dele os 68 pares
sao ordenados, o posto 1 fica com o melhor, e a caixa e a distribuicao dos
postos daquele par entre os 131 conjuntos.

Caixa **baixa e curta** e bom e constante; **baixa e comprida** e bom em
media e instavel. As duas fases entram juntas: o tempo **soma**, porque e
trabalho sequencial, e a memoria e o **maior** dos dois picos, porque as
fases sao sequenciais e somar inventaria uma ocupacao que nunca existiu ao
mesmo tempo.

Ordenar exige bloco completo, entao os pares medidos em menos de 95% dos
conjuntos ficam de fora, nomeados na saida do script.

## Ressalvas que valem para todas

* **O posto nao diz por quanto** (figura 07): uma vitoria por 1% e uma por
  100x valem o mesmo posto. O tamanho da diferenca esta na 03, na 04 e nos
  perfis.
* **Estar dentro da faixa e uma afirmacao; nao conseguir separar nao e.**
  Equivalencia (01) afirma igualdade dentro de uma margem. As demais figuras
  nao fazem essa afirmacao.
* **Os numeros de limiar** (`T50`, `MAX`, `X`, `MS`, `MS2`) nas figuras de
  custo vem da `custo-final5`, que rodou com `score_edges="unique"`. O
  padrao voltou depois para `"fixed"` e a remedicao esta pendente.
