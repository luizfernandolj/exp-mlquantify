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

## Por que os números diferem dos do painel

**Estas figuras usam 132 conjuntos de dados e as da pasta de cima usam
142.** A diferença é deliberada e vale só para a monografia.

O capítulo de método descreve um funil que parte de 176 arquivos e aplica
quatro critérios de seleção, chegando a 132 conjuntos, com 108 binários e
24 multiclasse. O catálogo que alimentou a execução parte de 202 arquivos,
com as mesmas quatro regras e a mesma minoria mínima de 30, e chega a 142.

A diferença está na coleção e não nos critérios. Os 26 arquivos a mais
renderam 10 conjuntos selecionados, todos multiclasse, o que explica os 34
multiclasse da execução contra os 24 do texto. São eles dois do Kaggle,
seis do OpenML (o `amazon-commerce-reviews` e cinco variantes dele), o
`Walking.csv` e o `letter.csv`. A lista está em
`descricao/conjuntos_fora_do_criterio.txt` e o `gerar.py` a exclui.

**O painel do experimento continua com os 142**, por decisão de quem
conduz o estudo. Ele descreve a execução como ela foi, enquanto estas
figuras descrevem o recorte que o texto define.

Nenhuma conclusão depende disso. O erro mantém 38 dos 39 pares dentro da
faixa nos dois recortes, com a mesma exceção. O que se move são as razões
de custo, de 0,983 para 0,967 no tempo e de 0,800 para 0,816 na memória,
além do PWKAC contra a `QuantificationLib`, que cai de 3,6 para 3,4 por
sair os dois conjuntos de mais classes.

## Outras diferenças em relação ao painel

* **O custo vem só da `final-pareado`**, sem a execução otimizada
  `custo-final5`, para que o resultado corresponda ao método descrito no
  capítulo anterior.
* **As duas fases entram juntas.** O tempo soma treinamento e estimação,
  que são trabalho sequencial. A memória toma o maior dos dois picos,
  porque o processo libera a primeira fase antes de iniciar a segunda.
* **O texto das figuras tem acento e vírgula decimal**, ao contrário do
  restante do repositório, porque rótulo de figura é texto de leitura.
