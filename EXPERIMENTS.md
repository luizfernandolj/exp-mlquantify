# Experimentos de comparacao entre bibliotecas de quantificacao

Implementacao da Etapa 4 da monografia (Secao 4.3.4): a biblioteca proposta
(`mlquantify`) e comparada com `QuaPy`, `QuantificationLib` e `qunfold` em
erro de quantificacao, custo computacional e esforco de uso.

---

## 1. Preparar o ambiente

Python **3.11** (as bibliotecas comparadas ainda nao acompanham versoes mais
novas).

```powershell
.\setup_env.ps1
```

```bash
./setup_env.sh
```

O script cria `.venv/`, instala o `requirements.txt` e imprime a versao de
cada biblioteca. Duas armadilhas ja estao resolvidas no arquivo de
dependencias:

- o pacote `qunfold` do PyPI e **outro projeto** (unfolding quantico); a
  biblioteca de quantificacao de Bunse so existe no GitHub e e instalada de
  la;
- o `scikit-learn` esta fixado em `1.5.2`, porque a `QuantificationLib`
  chama `cross_val_predict` com o argumento `fit_params`, removido na versao
  1.6. Todas as bibliotecas recebem a mesma versao, entao a comparacao
  continua justa.

Para que todas as maquinas tenham **exatamente** o mesmo ambiente, use o
arquivo de versoes travadas em vez do `requirements.txt`:

```bash
.venv/Scripts/pip install -r requirements.lock.txt
```

---

## 2. Montar o catalogo (uma vez)

```bash
.venv/Scripts/python build_catalog.py --methods-table
```

Percorre `datasets/`, mede os atributos de cada arquivo e aplica os quatro
criterios de selecao da Secao 4.3.3, em sequencia:

| Criterio | O que remove |
| --- | --- |
| arquivos auxiliares | particoes de treino/teste/validacao e submissoes |
| classe unica | arquivos sem prevalencia a estimar |
| duplicatas | copias do mesmo conjunto |
| classes raras | classe minoritaria com menos de 30 instancias |

Gera tres arquivos em `descricao/`:

- `catalogo_experimentos.csv` — todos os arquivos, com os atributos medidos
  e o motivo de cada exclusao;
- `datasets_selecionados.csv` — os selecionados, com uma coluna `indice`
  deterministica. **Esse indice e o que `--range` e `--shard` usam**, entao
  o arquivo precisa ser o mesmo em todas as maquinas (gere uma vez e copie,
  ou mantenha no repositorio);
- `cobertura_metodos.csv` — quais metodos cada biblioteca implementa.

Opcoes uteis:

```bash
# criterio de duplicata da monografia (tamanho + distribuicao de classes)
python build_catalog.py --duplicate-key shape_dist

# outro corte de classe minoritaria
python build_catalog.py --min-minority 10
```

---

## 3. Rodar

```bash
.venv/Scripts/python run_experiments.py --run-id final
```

Usa todos os nucleos. O que cada job faz, por conjunto de dados:

1. divisao estratificada 50/50 entre treino e teste;
2. protocolo UPP sobre a parte de teste: `m` em {10, 50, 100, 500}, 100
   vetores de prevalencia por tamanho, totalizando **400 amostras**. As
   amostras sao geradas **uma unica vez**, guardadas como indices em
   `.cache/samples/` e reusadas por todas as bibliotecas;
3. cada metodo e treinado com uma regressao logistica nova, de semente fixa,
   e estima a prevalencia das 400 amostras;
4. sao registrados AE, RAE, SE e NKLD por amostra, mais tempo e pico de
   memoria.

As medidas de tempo sao repetidas tres vezes e resumidas pela mediana; o
pico de memoria e medido em uma repeticao separada, porque o `tracemalloc`
altera o tempo. Cada processo roda com **uma thread** (`OMP_NUM_THREADS=1`),
de modo que o paralelismo fique entre jobs e os tempos nao dependam do BLAS.

### Acompanhar

Uma barra geral e uma barra por trabalhador, mostrando o conjunto, a
biblioteca e o metodo em andamento:

```
Combinacoes:  34%|███▍      | 3542/10290 [1:12:03<2:17:31, 1.22comb/s] ok=3498 erro=44
  [ 12/30 ] mlquantify        KDEyML     1489_phoneme.csv
  [  7/20 ] quapy             EMQ        44_spambase.csv
  [  3/12 ] quantificationlib HDy        BNG.csv
  [  5/8  ] qunfold           EDy        letter.csv
```

O detalhe de cada combinacao vai para `logs/<run-id>__<node>.log`; o
terminal fica reservado para as barras.

---

## 4. Dividir entre varias maquinas

Todas as maquinas usam o **mesmo `--run-id`** e um **`--node` diferente**.
Cada uma grava em sua propria pasta, entao nao ha conflito.

### Por fatia em rodizio (recomendado)

Distribui conjuntos grandes e pequenos de forma parecida entre as maquinas:

```bash
# maquina 1
python run_experiments.py --run-id final --node pc1 --shard 1/3
# maquina 2
python run_experiments.py --run-id final --node pc2 --shard 2/3
# maquina 3
python run_experiments.py --run-id final --node pc3 --shard 3/3
```

### Por faixa de indice

O catalogo esta ordenado por tamanho, entao as faixas finais sao bem mais
pesadas. Use quando as maquinas tiverem capacidades diferentes:

```bash
python run_experiments.py --run-id final --node forte  --range 120:999
python run_experiments.py --run-id final --node medio  --range 60:120
python run_experiments.py --run-id final --node fraco  --range 0:60
```

### Por biblioteca, ou por conjunto especifico

```bash
python run_experiments.py --run-id final --node pc1 --libraries mlquantify,quapy
python run_experiments.py --run-id final --node pc2 --libraries quantificationlib,qunfold

python run_experiments.py --run-id final --node pc1 --datasets phoneme,spambase
python run_experiments.py --run-id final --node pc2 --types multiclasse
```

### Reunir no final

Copie de cada maquina as duas pastas abaixo para a mesma arvore:

```
results/final/parts/<node>/
results/final/ledger/<node>/
```

e rode uma unica vez:

```bash
python aggregate_results.py --run-id final
```

---

## 5. Retomar e acompanhar

O **ledger** (`results/<run-id>/ledger/<node>/*.jsonl`) e a fonte de verdade
sobre o andamento: uma linha JSON por combinacao, gravada assim que ela
termina. Saber quanto falta, ou retomar de onde parou, nao depende de ler
nenhum arquivo de resultado.

```bash
# andamento, lido so do ledger
python run_experiments.py --run-id final --status

# retomar: basta repetir o mesmo comando, o que ja esta no ledger e pulado
python run_experiments.py --run-id final --node pc1 --shard 1/3

# refazer tudo do zero
python run_experiments.py --run-id final --fresh
```

Metodos que falharam tambem ficam registrados e **nao sao repetidos** — uma
falha por limitacao da biblioteca e um resultado, nao um erro de execucao.
Para tentar de novo, apague o ledger ou use `--fresh`.

---

## 6. Metodos deixados para depois

Os quantificadores da familia ELM (`SVM(Q)`, `SVM(KLD)`, `SVM(NKLD)`,
`SVM(AE)`, `SVM(RAE)`) e os neurais (`QuaNet`, `HistNetQ`, `GMNet`) ainda
nao estao na versao publicada da `mlquantify` e **nao entram na execucao
padrao**. Ja estao registrados e podem ser rodados sozinhos quando o branch
estiver pronto:

```bash
# ELM: na QuaPy exige o binario externo do SVM-perf (variavel SVMPERF_HOME)
python run_experiments.py --run-id elm --groups elm --libraries mlquantify,quapy

# neurais: exigem PyTorch
python run_experiments.py --run-id neural --groups neural --libraries mlquantify,quapy
```

Outros grupos fora do padrao:

```bash
# metaquantificadores (Ensembles, Bootstrap)
python run_experiments.py --run-id meta --groups meta

# README da QuaPy: so se aplica a dados textuais com atributos binarios
python run_experiments.py --run-id texto --groups text --libraries quapy
```

Enquanto nao forem executados, esses metodos aparecem no `summary.csv` com
`status = indisponivel` e o motivo em `mensagem_erro`.

---

## 7. O que sai

Em `results/<run-id>/`:

| Arquivo | Conteudo |
| --- | --- |
| `summary.csv` | **o arquivo principal**: uma linha por (conjunto, biblioteca, metodo, tamanho de amostra) |
| `per_sample.csv` | uma linha por amostra, com a prevalencia real e a estimada |
| `q1_erro.csv` | erro pareado entre a mlquantify e cada biblioteca de referencia |
| `q2_custo.csv` | tempo e memoria, com a razao sobre a mlquantify |
| `q3_uso.csv` | configuracoes manuais, taxa de sucesso sem ajustes e linhas de codigo |
| `testes_estatisticos.csv` | Friedman por metodo e por medida, com os ranks medios |
| `nemenyi/*.csv` | Nemenyi onde o Friedman acusou diferenca (alfa = 0,05) |
| `visao_geral.csv` | contagem por biblioteca e situacao |
| `amostras_descartadas.csv` | uma linha por combinacao com amostra descartada |

### As respostas de Q1, Q2 e Q3

A agregacao para em uma linha por metodo e biblioteca. Quem responde as tres
perguntas em tabela de tamanho de texto e o `analyze_results.py`:

```bash
python aggregate_results.py --run-id final --out-dir ... --no-samples
python analyze_results.py --run-id final --out-dir ...
python analyze_results.py --run-id final --out-dir ... --erro mrae
```

Ele grava em `results/<run-id>/analise/`:

| Arquivo | Conteudo |
| --- | --- |
| `respostas.md` | as respostas escritas, com os numeros no lugar |
| `q1_equivalencia.*` | concordancia entre implementacoes, por biblioteca e tamanho |
| `q1_discordancias.*` | os metodos que mais discordam -- as excecoes de Q1 |
| `q1_testes.*` | resumo dos testes, por medida e conjunto de bibliotecas |
| `q2_resumo.*` | razao de custo, com quartis |
| `q3_resumo.*` | esforco de uso |
| `figuras/*.png` | uma figura por pergunta |

Cada tabela sai em `.csv`, `.md` e `.tex` (com `booktabs`, pronta para colar).

Duas escolhas que mudam a leitura dos numeros:

* **A unidade e o caso pareado.** Para Q1, um caso e um conjunto, um metodo e
  um tamanho de amostra em que as duas bibliotecas estimaram; para Q2, um
  conjunto e um metodo. Media sobre casos pareados nao depende de quais
  conjuntos cada biblioteca conseguiu rodar.
* **Os postos medios so se comparam dentro do mesmo conjunto de bibliotecas.**
  Um metodo que existe em tres tem posto de 1 a 3, e um que existe em duas tem
  posto de 1 a 2; por isso o `q1_testes.csv` separa as linhas por
  `bibliotecas` em vez de misturar tudo em uma media por medida.

Pode rodar com a execucao em andamento: o topo do `respostas.md` diz quantas
combinacoes das planejadas entraram na conta.

### Colunas do `summary.csv`

**Identificacao** — `dataset_id`, `tipo_dataset`, `n_instancias`,
`n_atributos`, `n_classes`, `razao_desbalanceamento`, `n_treino`, `n_teste`,
`biblioteca`, `versao_biblioteca`, `metodo`, `familia`, `classe`, `aliases`.

**Protocolo** — `protocolo`, `tamanho_amostra`, `n_amostras` (quantas
amostras deste tamanho entraram nas medidas), `n_amostras_descartadas`
(quantas deste tamanho foram descartadas), `amostras_descartadas_total`
(descartadas na combinacao inteira), `motivo_descarte`, `n_prevalencias`,
`semente`, `proporcao_teste`, `protocolo_escores` (sob qual protocolo de
geracao dos escores de treinamento a linha foi medida; vazia nas linhas
gravadas antes do alinhamento da secao 11).

**Q1, erro** — `mae`, `mrae`, `mse`, `nkld_medio`, mais desvio padrao,
mediana, quartis e maximo de AE e RAE, e a faixa de prevalencia coberta
(`prev_real_min`, `prev_real_max`).

**Q2, custo** — `tempo_treino_s`, `tempo_estimacao_s` (deste tamanho de
amostra), `tempo_estimacao_total_s` (todas as amostras),
`tempo_estimacao_por_amostra_s`, `memoria_treino_mb`,
`memoria_estimacao_mb`, `repeticoes_custo`.

**Q3, esforco de uso** — `configuracoes_manuais`,
`usou_decomposicao_manual`, `forma_direta_ok` (a biblioteca aceitou o uso
direto, sem ajuste) e `erro_forma_direta`.

**Situacao** — `status` (`ok`, `erro`, `timeout`, `indisponivel`),
`tipo_erro`, `mensagem_erro`. Uma combinacao com `status = ok` e
`amostras_descartadas_total > 0` foi concluida com menos de 400 amostras;
veja *Amostras invalidas* na secao 9.

### Nomes equivalentes de metodo

Bibliotecas diferentes dao nomes diferentes a mesma formulacao. O `GACC` da
`mlquantify` corresponde ao `ACC` das outras tres, que ja sao multiclasse
nativas; o `GPACC` corresponde ao `PACC`. A agregacao usa essa equivalencia
(coluna `aliases`) para que a comparacao por nome nao acabe comparando
formulacoes distintas. No `summary.csv` o nome e sempre o da execucao; nas
tabelas Q1 e Q2 as linhas aparecem tambem sob o nome equivalente, com
`via_alias = True`.

---

## 8. Esforco de uso (Q3)

Em `scripts/q3_usage/` ha um script por biblioteca que conduz o mesmo
experimento sobre um conjunto de dados, do carregamento ao calculo do erro,
seguindo apenas a documentacao oficial de cada uma. As instrucoes que
existem so para adaptar o metodo ou os dados ao formato da biblioteca estao
marcadas com `# MANUAL:`.

```bash
python count_loc.py
```

Formata os quatro scripts com o Black (88 colunas) e conta as linhas
logicas com o modulo `tokenize`, desconsiderando linhas em branco,
comentarios e docstrings. O resultado vai para
`descricao/q3_linhas_de_codigo.csv` e entra no `q3_uso.csv`.

---

## 9. Quanto tempo leva

A execucao completa sao **142 conjuntos x 70 combinacoes = 9.940
combinacoes**. Com o protocolo padrao, cada combinacao faz 4 treinamentos e
1.600 estimacoes (3 repeticoes de tempo, mais uma repeticao so para medir
memoria).

Medidas feitas neste repositorio, somando as quatro bibliotecas:

| Conjunto | Instancias | CPU por conjunto |
| --- | --- | --- |
| `iris.2`, `wine`, `sonar` | 150 a 208 | ~11 min |
| `space_ga`, `yeast_data` | ~3.100 | ~48 min |

O custo cresce devagar com o tamanho do conjunto (cerca de `n^0,5`), porque
**o que domina sao as 1.600 estimacoes por metodo**, feitas sobre amostras
de no maximo 500 instancias; so o treinamento cresce com `n`. Extrapolando
para a colecao inteira:

- **cerca de 220 horas de CPU** no total;
- **13 horas de relogio** em uma maquina de 20 nucleos;
- divididas em tres maquinas parecidas, **4 a 5 horas** em cada.

A maior parte do custo esta em poucos metodos: `KDEyHD`, `HDy` com
decomposicao e `PACC` da qunfold passam de 2 minutos por conjunto cada.

### Reduzir o tempo

O tamanho dos conjuntos **nao** e o lever principal: limitar a 50.000
instancias so tira 15% do total. O que corta de verdade e o numero de
repeticoes:

```bash
# ~4x mais rapido (55 h de CPU, ~3 h em 20 nucleos): uma repeticao de tempo
# e sem medir memoria. Os numeros de erro (Q1) ficam identicos, porque sao
# deterministicos; so as medidas de custo (Q2) perdem a mediana de 3.
python run_experiments.py --run-id piloto --cost-repeats 1 --no-memory

# metade das amostras: 50 vetores de prevalencia em vez de 100
python run_experiments.py --run-id piloto --n-prevalences 50
```

Uma estrategia que funciona bem: rodar tudo com `--cost-repeats 1
--no-memory` para ter o Q1 em poucas horas, e depois uma execucao separada,
com outro `--run-id` e o protocolo completo, so nos metodos e conjuntos que
interessam para o Q2.

### Conjuntos muito grandes e o tempo limite

Os oito conjuntos maiores concentram boa parte do custo:

| Conjunto | Instancias | CPU estimada |
| --- | --- | --- |
| `Dermatology.csv` | 1.000.000 | ~16 h |
| `healthcare.csv` | 318.438 | ~9 h |
| `phphQEck0.csv` | 245.057 | ~8 h |
| `Mosquitoes.csv` | 158.249 | ~6 h |

Cada um deles vira quatro jobs (um por biblioteca), e **um job desses passa
do `--job-timeout` padrao de 2 horas**. O job e encerrado, os metodos que
ja terminaram ficam gravados e o resto continua pendente no ledger. Para
fechar esses conjuntos, rode de novo sem limite — so o que falta sera
refeito:

```bash
python run_experiments.py --run-id final --job-timeout 0 --range 130:999
```

Alternativamente, limite esses conjuntos desde o inicio:

```bash
python run_experiments.py --run-id final --max-rows 100000
```

#### O que o `--method-timeout` nao cobre

O prazo por metodo e verificado em um unico ponto: no laco que estima as
400 amostras, entre uma amostra e a proxima. **O ajuste fica fora dele**, e
isso tem duas consequencias diferentes, dependendo de o `fit` devolver ou
nao.

Quando o `fit` e lento mas termina, o prazo e cobrado tarde: a primeira
verificacao acontece antes da amostra 0, quando o orcamento ja foi todo
gasto no ajuste. A linha sai como `timeout`, so que acima do limite. Na
`Dermatology.csv` o `KDEyML` e o `KDEyHD` da QuaPy registraram
`tempo limite excedido apos 0 amostras` com **2 441 s** e **2 496 s** de
parede contra um orcamento de 1 800 s. A medicao e honesta — o metodo
realmente nao estimou nenhuma amostra —, mas o corte nao e no ponto pedido.

Quando o `fit` nao devolve, **o prazo nunca e cobrado.** O `MMD` da qunfold
no `Mosquitoes.csv` foi tentado tres vezes e em nenhuma delas chegou a
sair do ajuste: 8 h na primeira (morto pelo limite de job), 9 h 47 na
segunda (a execucao morreu antes) e 15 h na terceira, tambem morto pelo
limite de job — **cerca de 33 h de CPU sem uma unica linha no ledger.** O
unico limite que o alcanca e o `--job-timeout`, que mata o job inteiro e
leva com ele os metodos que vinham depois: no caso da qunfold, o `GHDx` e o
`EDx`, que estao atras do `MMD` na ordem do registro e nunca foram
tentados. Para chegar neles e preciso excluir o `MMD` da passada:

```bash
python run_experiments.py --run-id final-alinhado --skip-methods MMD
```

Com `cost_repeats = 3` o ajuste e feito tres vezes, mais uma quarta para
medir memoria, de modo que a parte descoberta nao e pequena para os metodos
que montam matriz de distancia.

O relogio e o de CPU, e isso e proposital — a maquina foi suspensa tres
vezes nesta execucao, e um relogio de parede teria reprovado metodos
baratos por isso. O avesso da escolha e que o tempo de CPU tambem para
enquanto o processo espera disco: sob paginacao pesada a contagem do prazo
congela junto com o processo que ela deveria interromper.

Observado em 05/11/2026: as 07:18 o `EDy` da mlquantify no `healthcare.csv`
comecou e a execucao inteira — processo pai e trabalhadores — morreu sem
escrever nenhuma linha, sem dump e sem evento do sistema. A maquina seguiu
de pe por mais de tres horas depois disso. Nao ha medicao que prove a
causa. O que se sabe: o job do `MMD` da qunfold estava no ar desde as 21:31
da noite anterior, preso no ajuste, e numa tentativa posterior esse mesmo
trabalhador foi medido com **pico de 10,4 GB de memoria residente e 17
milhoes de faltas de pagina** numa maquina de 15,7 GB. Tres trabalhadores,
um deles nesse estado, e um `EDy` comecando a montar matriz de distancia e
um cenario compativel com esgotamento do limite de confirmacao de memoria,
no qual nem registrar em log e possivel. Compativel, nao comprovado.

A correcao seria passar o prazo tambem ao `fit`, cobrando-o num processo
separado em vez de dentro do laco de amostras. Ficou de fora de proposito
enquanto a execucao `final-alinhado` estava aberta: mudar o criterio de
corte no meio tornaria heterogeneas as medidas de uma mesma execucao.

### Um nucleo por trabalhador

Cada trabalhador e **preso a um nucleo logico** (`pin_cpu`, em
`config.py`). Isso nao e so arrumacao: o JAX, usado pela qunfold, **ignora**
`OMP_NUM_THREADS` e abre um pool proprio de cerca de 70 threads por
processo. Sem a afinidade, 20 trabalhadores viravam mais de 600 threads
disputando 20 nucleos, o que multiplicou por cinco o tempo medido da
qunfold e contaminou tambem o tempo das outras bibliotecas.

Medido em maquina ociosa, um `predict` da qunfold leva ~220 ms sem
afinidade e ~250 ms com afinidade (12% de diferenca); sob a disputa, passava
de 1.100 ms. A afinidade tambem alinha a medicao com a monografia, que pede
medidas "em um unico processo, sem paralelismo".

### Pasta de saida fora de pasta sincronizada

Se o repositorio estiver no OneDrive, Dropbox ou similar, grave os
resultados fora dele: o sincronizador mantem os arquivos abertos enquanto
envia, o que trava a gravacao dos parciais.

```bash
python run_experiments.py --run-id final --out-dir C:\Users\voce\exp-run
python aggregate_results.py --run-id final --out-dir C:\Users\voce\exp-run
```

### Se a maquina dormir no meio

O relogio do Windows continua correndo durante a suspensao. Sem tratamento,
ao acordar todos os jobs em andamento pareceriam ter estourado o tempo
limite de uma vez, e suas medidas de tempo ficariam sem sentido. Tres
defesas:

- o tempo limite por metodo e contado em **tempo de CPU**
  (`time.process_time`), que nao avanca enquanto a maquina dorme;
- o escalonador detecta saltos de relogio maiores que um minuto e desconta
  o salto do inicio dos jobs em andamento, em vez de encerra-los;
- cada combinacao grava `cpu_total_s` junto do tempo de relogio; se os dois
  divergirem em mais de 300 s, a linha sai com `medida_suspeita = True` e a
  agregacao a exclui do Q2 (o erro continua valendo para o Q1).

Para execucoes longas, vale impedir o sono por ociosidade. Se mesmo assim
uma suspensao contaminar parte do ledger, `clean_ledger.py` encontra as
janelas paradas e remove apenas as combinacoes que as cruzaram:

```bash
python clean_ledger.py --run-id final --out-dir C:\Users\voce\exp-run
python clean_ledger.py --run-id final --out-dir C:\Users\voce\exp-run --apply
```

Sem `--apply` ele so mostra o que faria. As combinacoes removidas sao
refeitas na proxima execucao; o resto e preservado.

Ao fim da execucao `final-alinhado` o filtro marcou **405 das 38 965 linhas
do `summary.csv` (1,04%)**, atingindo **264 das 9 904 combinacoes (2,7%)**;
192 delas caiam em casos pareados de custo e sairam do Q2.

| Biblioteca | Linhas | Suspeitas | Taxa | Tempo medio por combinacao |
| --- | --- | --- | --- | --- |
| qunfold | 4 387 | 87 | **1,98%** | 103,5 s |
| QuantificationLib | 6 642 | 74 | 1,11% | 16,7 s |
| mlquantify | 16 824 | 152 | 0,90% | 12,4 s |
| QuaPy | 11 112 | 92 | 0,83% | 15,6 s |

Por conjunto: `fashion-mnist` 65, `healthcare` 58, `Dermatology` 41,
`phphQEck0` 24, `Walking` 21, `Mosquitoes` 20 -- os trabalhos longos, que
sao os unicos expostos a uma suspensao. A qunfold, que custa seis a oito
vezes mais por combinacao que as outras tres, tem o dobro da taxa. Entre
as outras tres a taxa **nao** ordena pelo tempo medio -- a QuaPy custa mais
que a mlquantify e e marcada menos --, entao a leitura honesta e so esta: a
marca segue a exposicao a uma janela de suspensao, nao o metodo. Se as
marcas estivessem distribuidas por igual entre conjuntos baratos e caros,
seria sinal de que o criterio corta medida boa; nao e o que se ve.

### Rodizio entre bibliotecas

Os jobs pendentes sao reordenados em rodizio por biblioteca antes de
comecar. Sem isso, uma execucao retomada pode deixar o que sobrou
concentrado em uma unica biblioteca e colocar todos os trabalhadores nela
ao mesmo tempo -- o que, no caso da qunfold, significa nenhuma conclusao
por horas e memoria estourada, porque cada processo dela carrega o JAX.

### Se precisar interromper

Os trabalhadores checam o processo principal entre um metodo e outro e
encerram junto com ele. Ainda assim, se a execucao for morta de fora no
meio de um metodo longo, pode sobrar algum processo orfao, consumindo
nucleo e segurando os arquivos do ledger. Para conferir e limpar:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*spawn_main*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

### Memoria

Cada processo trabalhador usa cerca de **260 MB** so de bibliotecas (numpy,
scikit-learn e, no caso da qunfold, jax). Com 20 processos sao cerca de
**5,2 GB** antes de qualquer dado. Em uma maquina de 16 GB com navegador
aberto isso ja fica apertado.

O custo real, no entanto, nao vem das bibliotecas: vem do conjunto de
dados, e **cresce com o tamanho dele**. O catalogo esta ordenado por numero
de instancias, entao o numero de trabalhadores que funciona no comeco da
execucao nao funciona no fim. Na nossa execucao, 14 trabalhadores
atravessaram 91% do catalogo sem problema e so entao a maquina (15,7 GB)
ficou sem memoria, ja na cauda, onde estao conjuntos de 100 mil a 1 milhao
de instancias.

Quando isso acontece o estrago nao fica contido em um trabalhador:

* alocacoes banais, de poucos MiB, comecam a falhar com `MemoryError` em
  metodos que nao tem nenhum problema de escala -- foi assim que 19
  combinacoes da QuaPy falharam todas no mesmo minuto;
* chamadas triviais do proprio escalonador passam a levantar `OSError`
  (no Windows, `WinError 1450`, "recursos insuficientes do sistema"). Era
  por aqui que a execucao morria: o escalonador caia, os trabalhadores
  continuavam em seus jobs, e por horas ninguem lancava os jobs restantes.

O escalonador agora tolera essas falhas (`MAX_FALHAS_DE_SISTEMA` em
`expq/parallel.py`): espera um segundo e tenta de novo, porque trabalhador
que termina devolve memoria. Mas isso e rede de protecao, nao solucao.

**A cauda do catalogo pede menos trabalhadores que o comeco.** Rode em duas
passadas, separando os conjuntos gigantes:

```bash
GIGANTES=Dermatology,healthcare,phphQEck0,Mosquitoes,Walking,Covertype,star_classification,fashion-mnist,blog_feedback

# os 133 conjuntos normais, com folga de paralelismo
python run_experiments.py --run-id final -j 6 --skip-datasets $GIGANTES

# os 9 gigantes, um por vez em cada biblioteca
python run_experiments.py --run-id final -j 3 --datasets $GIGANTES
```

As duas passadas somam o catalogo inteiro, e o ledger cuida de nao repetir
nada. Alternativa, se o tempo importar mais que a fidelidade do conjunto:
`--max-rows 50000`, que subamostra os gigantes de forma estratificada.

#### Separar falta de memoria de limitacao do metodo

As duas coisas aparecem no ledger como `MemoryError`, e so uma delas e
resultado. O que separa e o **tamanho da alocacao**:

| alocacao pedida | o que e | o que fazer |
|---|---|---|
| 1 MiB a 46 MiB | a maquina estava sem memoria | refazer |
| 3,6 GiB a 70 GiB | o metodo montou matriz de distancia entre todas as instancias | e resultado, fica |

Nao houve nada entre 46 MiB e 3,6 GiB, entao o corte e seguro. O
`clean_pressure.py` faz essa separacao, remove do ledger so o primeiro caso
e deixa o segundo como medida valida:

```bash
python clean_pressure.py --run-id final --out-dir ... # simula
python clean_pressure.py --run-id final --out-dir ... --apply
```

As medidas de tempo tiradas durante a pressao de memoria continuam no
ledger: a paginacao infla o tempo de relogio, mas a coluna `medida_suspeita`
marca os casos em que relogio e CPU divergem demais, e a agregacao de Q2 os
descarta.

### Amostras invalidas

O protocolo UPP gera **400 amostras** por combinacao. Algumas vezes um
metodo nao consegue estimar uma delas: devolve um vetor sem massa nenhuma,
ou a propria biblioteca recusa a saida. Isso nao e o metodo falhando, e uma
amostra em que ele nao achou resposta.

Por isso a amostra invalida e **descartada sozinha**, e a combinacao segue
com as demais. O limite e `MAX_INVALID_FRACTION`, 5% das amostras, em
`expq/runner.py`: acima disso nao se trata mais de amostra isolada -- o
metodo nao esta estimando nada -- e a combinacao e reprovada com
`TooManyInvalidSamples`. Falta de memoria e tempo limite nao entram nessa
conta: dizem respeito a medicao inteira e reprovam a combinacao na hora.

Nao importa quem recusou a estimativa, nos ou a biblioteca. A QuaPy checa o
vetor por conta propria e levanta a excecao dentro do `predict`; para a
comparacao isso e a mesma coisa que o nosso `normalize_prevalence` recusar a
saida.

O limite separa os dois casos por evidencia, nao por suposicao. O `SMM`
multiclasse da QuaPy ilustra bem:

| conjunto | amostras invalidas | resultado |
|---|---|---|
| `wine.csv` (3 classes) | 1 de 400 | concluida com 399 amostras |
| `IRIS.csv` (3 classes) | 37 de 400 (9,3%) | reprovada |

Onde houve descarte, o `summary.csv` traz `n_amostras_descartadas`,
`amostras_descartadas_total` e `motivo_descarte`, a `visao_geral.csv` traz a
coluna `ok_com_descarte`, e a agregacao grava
`amostras_descartadas.csv` com uma linha por combinacao afetada.

A politica antiga reprovava a combinacao na primeira amostra invalida, o que
jogava fora as outras 399 medidas e subestimava a cobertura das bibliotecas
afetadas. Para refazer o que foi reprovado assim:

```bash
python clean_invalid.py --run-id final --out-dir ... # simula
python clean_invalid.py --run-id final --out-dir ... --apply
python run_experiments.py --run-id final -j 6       # refaz so o que saiu
```

O ledger guarda a mensagem do erro, nao o tipo da excecao, e a mensagem do
`TooManyInvalidSamples` repete o motivo mais frequente entre as amostras
descartadas -- entao ela casa com os mesmos textos que o script procura. O
`clean_invalid.py` reconhece esse caso e **nao** marca a combinacao para
refazer: ela passou dos 5% e foi reprovada com razao, refazer daria o mesmo
resultado. Aconteceu uma vez na execucao final, com o `ACC` da mlquantify em
`news_popularity_data.csv` (39 644 instancias, 4 classes): **169 das 400
amostras** com estimativa de soma nula, 42% do protocolo.

### Coluna nova com a execucao em andamento

O `_append_rows` grava o cabecalho so quando cria o arquivo. Se uma coluna e
acrescentada ao `summary.csv` no meio da execucao, o parcial de uma
combinacao medida antes continua com o cabecalho antigo, e as linhas novas
entram com mais campos do que ele declara -- o pandas recusa o arquivo e a
agregacao final para.

Nada se perde: o numero de campos da linha diz sob qual esquema ela foi
gravada. O `_realign` le cada linha pela posicao e escolhe o esquema por esse
tamanho, usando `SUMMARY_ANTERIORES` em `expq/runner.py`. **Toda coluna nova
precisa ser registrada ali**, no inicio da tupla; linha de tamanho que nenhum
esquema explica interrompe a gravacao em vez de virar valor trocado em
silencio.

O `_realign` corrige o parcial que ainda vai receber linha nova. Para os que
ja estao completos:

```bash
python clean_schema.py --run-id final --out-dir ... # simula
python clean_schema.py --run-id final --out-dir ... --apply
```

Ele deixa de lado o parcial cuja combinacao ainda tem metodo pendente, para
nao disputar o arquivo com um trabalhador vivo, e por padrao so toca nos que
estao com linhas de tamanhos diferentes. Parcial de cabecalho antigo mas
coerente o pandas le sem erro, com as colunas novas vazias; `--todos`
reescreve esses tambem.

Na execucao final isso aconteceu duas vezes: com `cpu_total_s` e
`medida_suspeita` (25/10) e com as tres colunas de descarte (27/10). A
primeira deixou 1 239 linhas em 50 parciais com o cabecalho defasado. As
colunas de metrica e de tempo nao foram afetadas -- as colunas novas entraram
perto do fim, entao o deslocamento atingiu so `python`, `sklearn` e
`threads`, e o realinhamento devolve os valores aos seus lugares.

---

## 10. Ajustes do protocolo

```bash
# protocolo reduzido, para uma primeira passada rapida
python run_experiments.py --run-id piloto \
    --sample-sizes 50,100 --n-prevalences 20 --cost-repeats 1 --no-memory

# limitar conjuntos muito grandes por subamostragem estratificada
python run_experiments.py --run-id final --max-rows 50000

# um metodo so, em todas as bibliotecas
python run_experiments.py --run-id acc --methods ACC,GACC

# ver o plano sem executar
python run_experiments.py --run-id final --shard 1/3 --dry-run
```

Opcoes principais:

| Opcao | Para que serve |
| --- | --- |
| `--run-id` | nome da execucao; o mesmo em todas as maquinas |
| `--node` | nome desta maquina (padrao: hostname) |
| `--shard I/N`, `--range INI:FIM` | divisao do trabalho |
| `--datasets`, `--skip-datasets`, `--types` | filtro de conjuntos |
| `--libraries`, `--groups`, `--methods` | filtro de metodos |
| `--sample-sizes`, `--n-prevalences` | protocolo UPP |
| `--cost-repeats`, `--no-memory` | medidas de custo |
| `-j`, `--jobs` | processos simultaneos |
| `--job-timeout`, `--method-timeout` | limites em segundos |
| `--max-rows` | subamostra conjuntos grandes |
| `--fresh` | ignora o ledger |
| `--status`, `--dry-run` | inspecao |

---

## 11. Limitacoes das bibliotecas encontradas na montagem

Observacoes registradas ao adaptar cada biblioteca, que sustentam a
discussao das questoes Q1 e Q3. As versoes sao as do `requirements.txt`.

| Biblioteca | O que acontece |
| --- | --- |
| QuaPy | metodos binarios (`HDy`, `DyS`, `SMM`, `T50`, `X`, `MAX`, `MS`, `MS2`) interrompem a execucao em problemas multiclasse e precisam de `newOneVsAll` |
| QuaPy | mesmo com a decomposicao, o `SMM` multiclasse produz prevalencias negativas e a propria biblioteca aborta; fica registrado como `erro` |
| QuaPy | o `README` so aceita matriz binaria de atributos, entao nao se aplica a dados tabulares (grupo `text`) |
| QuantificationLib | metodos que estimam a distribuicao de treinamento exigem `CV_estimator` e a separacao entre `estimator_train` e `estimator_test` |
| QuantificationLib | o `OneVsRestQuantifier` exige, alem da decomposicao, que os dois estimadores sejam instancias de `OneVsRestClassifier` |
| QuantificationLib | chama `cross_val_predict` com `fit_params`, removido no scikit-learn 1.6; por isso a versao esta fixada em 1.5.2 |
| QuantificationLib | o `EoQ` 0.1.2 quebra na forma documentada e, na forma que executa, devolve sempre a mesma estimativa; marcado como `indisponivel` |
| qunfold | o pacote `qunfold` do PyPI e outro projeto; a biblioteca de quantificacao so existe no GitHub |
| qunfold | metodos agregativos exigem `CVClassifier`, e `HDy`/`HDx` exigem `n_bins`, que nao tem valor padrao |
| qunfold | nao oferece protocolos de avaliacao nem metricas de erro: ambos precisam ser escritos no script |
| mlquantify 0.5.1 | aplica a decomposicao multiclasse sozinha, sem configuracao; nao exigiu nenhum ajuste |
| mlquantify 0.5.1 | nao traz `PACC` como classe propria: o equivalente e o `GPACC`, que coincide com o `PACC` da QuaPy |
| mlquantify 0.5.1 | ainda sem os metodos ELM e neurais (outro branch), por isso o grupo padrao os deixa de fora |
| QuaPy | nao expoe o embaralhamento das dobras: `val_split` inteiro cai em `cross_val_predict(cv=5)`, que usa `StratifiedKFold(shuffle=False)` |
| qunfold | todo metodo sorteia o ponto de partida do otimizador e, sem `seed`, nao repete os numeros entre execucoes |
| mlquantify 0.5.1 | o `MLPE` exige um classificador que a propria documentacao diz nao influenciar a estimativa, e valida-o cruzadamente: 6 treinamentos jogados fora por combinacao (ver abaixo) |

### Os parametros nao eram os mesmos entre as bibliotecas

As quatro resolvem a mesma etapa -- obter predicoes fora da amostra para
treinar a funcao de agregacao -- e cada uma a resolvia com parametros
proprios. Medido nas versoes instaladas, antes do alinhamento:

| Biblioteca | Dobras | Estratifica | Embaralha | Semente |
| --- | --- | --- | --- | --- |
| mlquantify | 5 | sim | **nao** | **nao** |
| QuaPy | 5 | sim | **nao** | **nao** |
| QuantificationLib | 5 | sim | sim | sim |
| qunfold | 5 | sim | sim | sim |

O numero de dobras ja coincidia, e nao por acaso: na mlquantify `cv=None`
cai em cinco (`base_aggregative.py`, `if cv is None: cv = 5`) e na QuaPy
`val_split=5` e o padrao das classes que usam predicoes fora da amostra. O
que nao coincidia era o **embaralhamento** -- e com ele a dependencia da
ordem das linhas do arquivo -- e a **semente**.

Duas consequencias, as duas medidas:

**A ordem do arquivo entrava na conta.** A mlquantify e a QuaPy dividiam em
dobras sem embaralhar, sobre uma particao de treinamento que o `split`
devolvia ordenada (`np.sort`). Em arquivo agrupado por classe ou coletado em
ordem temporal, as dobras dessas duas seguiam essa ordem enquanto as das
outras duas eram sorteadas. Nao e ruido: e um protocolo diferente.

**Tres bibliotecas tinham partes sem semente.** A pior nao e a mais obvia:
todo metodo da qunfold sorteia o ponto de partida do otimizador
(`methods/__init__.py`, `_rand_x0(np.random.RandomState(seed), ...)`), de
modo que, sem `seed`, cada uma das 400 amostras partia de um ponto
diferente. A KDEy da QuaPy faz Monte Carlo com `random_state=None`. Cinco
execucoes da mesma celula, faixa maxima por classe:

| Metodo | Antes | Depois |
| --- | --- | --- |
| `quapy.KDEyHD` | 4,9 x 10^-3 | 0 |
| `qunfold.EDx` | 7,9 x 10^-4 | 0 |
| `qunfold.EDy` | 4,4 x 10^-5 | 0 |
| `qunfold.GHDy` | 1,4 x 10^-5 | 0 |
| `qunfold.MMD` | 7,6 x 10^-6 | 0 |
| `qunfold.GACC` | 3,6 x 10^-8 | 0 |
| `mlquantify.GACC`, `mlquantify.HDy`, `quapy.ACC` | 0 | 0 |

O ruido da `KDEyHD` entre execucoes era **maior que o limiar de 10^-3** com
que Q1 conta casos equivalentes: a mesma celula rodada duas vezes podia
mudar de lado.

#### O que passou a ser escrito

Toda construcao documentada (`build_adapted`) recebe os mesmos valores, nos
nomes de cada biblioteca: `cv`/`val_split` = 5, `stratified` = True,
`shuffle` = True, `random_state`/`seed` = a semente da execucao. O filtro e
a assinatura do construtor (`_protocolo` em `expq/registry.py`), e nao uma
lista de excecoes por metodo: quem nao tem o parametro nao recebe nada.

Com isso a mlquantify, a QuantificationLib e a qunfold passam a montar o
mesmo `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` -- dobras
identicas, nao equivalentes. A QuaPy e a excecao, porque nao tem a chave do
embaralhamento; por isso a particao de treinamento agora sai embaralhada do
proprio `split` (`expq/protocol.py`), uma vez e com semente: sobre linhas em
ordem aleatoria, dobras sem embaralhamento sao dobras sorteadas, e nenhuma
biblioteca depende mais da ordem do arquivo. A particao de teste continua
ordenada, senao o cache de amostras do UPP perderia validade.

Fora do construtor sobra o que as bibliotecas leem de estado global:
`_semear` fixa o `np.random` e o `quapy.environ['_R_SEED']` antes de cada
celula.

Uma ressalva de leitura para Q3: isto e protocolo do experimento, nao
configuracao exigida pela biblioteca. Vale igual para as quatro e **nao**
entra em `configuracoes_manuais`. A forma direta (`build_naive`), que e a
sonda de Q3, continua sem nenhum desses valores -- ela existe justamente
para medir o padrao da biblioteca.

Cada linha do `summary.csv` passa a dizer sob qual protocolo foi medida, na
coluna `protocolo_escores`. Linhas gravadas antes desta mudanca ficam com a
coluna vazia.

#### Quanto os numeros andam

A mudanca de dobras desloca as estimativas, e mais do que o ruido que ela
remove. Mesma celula, protocolo antigo contra novo, diferenca maxima por
classe:

| Metodo | Deslocamento |
| --- | --- |
| `mlquantify.GACC` | 1,6 x 10^-2 |
| `mlquantify.HDy` | 1,5 x 10^-2 |
| `quapy.KDEyHD` | 3,0 x 10^-3 |
| `qunfold.EDx` | 1,2 x 10^-4 |
| `quapy.ACC` | 0 |

Os 1,6 x 10^-2 da mlquantify passam do limiar mais largo de Q1 (10^-2), de
modo que resultados medidos antes e depois **nao se comparam**. Qualquer
execucao anterior a esta mudanca precisa ser refeita.

Verificado de ponta a ponta em duas execucoes completas do harness sobre
`wine.csv` e `wine.1.csv`, sete metodos, quatro bibliotecas: as 46 linhas do
`summary.csv` sairam identicas em `mae`, `mrae`, `mse` e `nkld_medio`. E o
`GACC` da mlquantify e o da qunfold passaram a coincidir digito a digito
(0,04400 e 0,03016), o que antes nao acontecia.

#### Os parametros do metodo

O alinhamento acima cobre a geracao dos escores. Faltavam os parametros do
proprio metodo, que ficaram assim.

**Distancia do `EDx` e do `EDy`.** A QuaPy usava `manhattan` e as outras
tres `euclidean`. Passou a ser `euclidean` nas quatro. Na
QuantificationLib o parametro exige um callable, nao uma string, daqui o
`_euclidean` em `expq/registry.py`. O `EDx` ficou **identico nas quatro**
(MAE 0,1103 em `wine.csv`, 0,0870 em `wine.1.csv`); antes a QuaPy dava
0,076 contra 0,087 das outras.

O `EDy` nao fechou: a mlquantify e a QuantificationLib coincidem (0,0168 e
0,0490), a QuaPy da 0,0232 e a qunfold 0,0183. Com a distancia e as dobras
agora iguais, o que resta e **formulacao**, nao configuracao -- e esse e
justamente o tipo de diferenca que Q1 deve medir.

**Escores do `EMQ`.** A QuantificationLib era a unica a passar os escores
por validacao cruzada; a mlquantify, a QuaPy (`val_split=None`) e a qunfold
leem o classificador treinado na particao inteira. O `estimator_train` da
QuantificationLib passou a ser o classificador simples (`train_cv=False`).
As quatro agora dao o mesmo numero (0,0172 e 0,0527). A classe continua
exigindo o argumento, entao `configuracoes_manuais` nao muda.

**Intervalos do `HDx`.** A mlquantify varria `[2, 4, ..., 20, 30]`; as
outras tres fixam 8. Passou a fixar 8 tambem -- em lista, `[8]`, porque a
classe valida `bins_size` como array-like e recusa inteiro. Custa quase
nada a ela: MAE 0,1795 para 0,1802 em `wine.csv` e 0,2371 para 0,2440 em
`wine.1.csv`. O que sobra e diferenca de metodo: 0,244 na mlquantify contra
0,049 na QuaPy em `wine.1.csv` nao e mais efeito de configuracao.

**Intervalos e divergencia do `DyS`.** A mlquantify varria
`[2, 4, ..., 20, 30]` com Topsoe; a QuaPy fixava 8 com Hellinger. Agora as
duas usam 8 intervalos e Topsoe, que e a divergencia da definicao do
metodo. A troca de divergencia quase nao mexe no resultado da QuaPy (AE
0,2626 para 0,2654 em `wine.csv`): o que a afunda ali e a decomposicao
um-contra-todos, nao a divergencia.

#### O `HDy` nao da para igualar, e o motivo importa

Era o que parecia pesar mais, e acabou sendo o contrario do esperado.
Medido:

| Biblioteca | Intervalos | Agregacao |
| --- | --- | --- |
| mlquantify | `[10, 20, ..., 110]` | mediana das estimativas |
| QuaPy | `[10, 20, ..., 110]` | mediana das estimativas |
| QuantificationLib | 8, largura igual | estimativa unica |

**A mlquantify e a QuaPy ja estavam iguais**, e nas duas isso e a definicao
publicada do metodo: Gonzalez-Castro et al. (2013) varrem de 10 a 110 em
passos de 10 e tomam a mediana das onze estimativas. Nenhuma das duas
precisou de ajuste.

Nenhuma das duas tambem *aceita* ajuste. A QuaPy fixa
`self.bins = np.linspace(10, 110, 11)` dentro do `aggregation_fit`, sem
parametro no construtor -- a classe `DMy`, que tem `nbins`, nao serve de
substituta porque e multiclasse nativa, ou seja, calcula a formulacao
generalizada e nao o `HDy` binario. E o `n_bins` da QuantificationLib e
escalar: nao ha como pedir a varredura.

Entao a coluna `HDy` da QuantificationLib e uma **variante de intervalo
fixo**, nao o mesmo metodo com outro parametro, e fica registrado assim nas
notas do metodo. Em dados binarios, onde a decomposicao nao entra, a
mlquantify e a QuaPy caem juntas (0,1200 e 0,1187 em `wine.1.csv`) e a
QuantificationLib se separa (0,0825) -- o efeito do intervalo fixo, isolado.

O mesmo vale para o `GHDy` e o `GHDx`: a mlquantify nao expoe o numero de
intervalos nessas duas classes e a qunfold exige, entao as duas colunas nao
sao comparaveis no parametro.

Com protocolo e parametros alinhados, duas execucoes completas do harness
sobre `wine.csv` e `wine.1.csv` -- onze metodos, quatro bibliotecas, 64
combinacoes -- sairam identicas em `mae`, `mrae`, `mse` e `nkld_medio`, sem
nenhum erro.

### O custo do MLPE na mlquantify

O `MLPE` e a linha de base trivial: sob a hipotese de nenhum desvio, a
estimativa de maxima verossimilhanca da prevalencia de teste e a propria
prevalencia de treinamento, de modo que o metodo ignora a amostra de teste.
Nas duas bibliotecas que o tem, ele devolve o mesmo numero -- a diferenca
maxima de MAE em 134 conjuntos e 5,6 x 10^-17, o epsilon da maquina. O custo
de chegar nele nao e o mesmo.

| Medida (mediana) | mlquantify | QuaPy | Razao |
| --- | --- | --- | --- |
| tempo de treinamento | 0,2752 s | 0,0002 s | **1 376x** |
| tempo de estimacao (400 amostras) | 0,0076 s | 0,0040 s | 1,9x |
| tempo total | 0,2828 s | 0,0042 s | **67x** |
| memoria no treinamento | 0,854 MB | 0,029 MB | 30x |

O `predict` nao e o problema: ele devolve `train_priors_` direto, e as 400
amostras custam 7,6 ms. **Tudo esta no `fit`.** A classe fica na hierarquia
agregativa (`SoftPredictionMixin`, `AggregativeMixin`,
`LikelihoodComposeQuantifier`) e passa pelo caminho comum de gerar escores
fora da amostra. Contado com um classificador instrumentado: **6
treinamentos e 5 chamadas a `predict_proba` por combinacao**, cinco dobras
mais o reajuste na particao inteira, e nada disso entra na conta. Com `cv=2`
sao 3 treinamentos, o que confirma que o custo e a validacao cruzada.

A propria documentacao da classe diz o que acontece: o `estimator` e
"kept for API compatibility with the aggregative quantifiers; it does **not**
influence the estimate", e `cv` e descrito como as dobras usadas ao treinar
"the (unused) estimator".

Nao e artefato do harness. `MLPE()` sem classificador e recusado:

    InvalidParameterError: The quantifier MLPE requires an estimator.

Entao nao ha como registrar o metodo sem pagar a validacao cruzada, e a
medida fica como esta: o numero que o experimento reporta e o custo real de
usar a linha de base trivial da biblioteca. A QuaPy resolve o mesmo problema
contando rotulos, sem classificador.

E o unico lugar do experimento em que a mlquantify perde por tres ordens de
grandeza, e perde no metodo mais simples que existe. Vale como item de
correcao na biblioteca -- tornar o `estimator` opcional no `MLPE`, ou nao
cruzar validacao quando ele nao e usado -- e nao como ajuste do protocolo.

Duas ressalvas que vieram depois, da medicao da correcao (PR #6, no fim
desta secao):

* **As tres ordens de grandeza sao do tempo de treinamento, e esse numero
  e fragil.** O denominador -- o treino do `MLPE` da QuaPy -- fica em
  0,0001 a 0,0002 s, na resolucao do relogio, e o mesmo dado da 1 376x,
  2 294x ou 1 307x conforme o estimador escolhido. O numero robusto e o
  composto de treino mais estimacao: **27,4x**. A conclusao de que a
  mlquantify e mais cara no metodo mais simples se sustenta; a de que e
  por tres ordens de grandeza nao, e o texto deve liderar pelo composto.
* **A correcao existe e foi medida.** Ajustando o estimador uma vez em vez
  de validar cruzado, o `fit` cai de 6 treinamentos para 1, o erro nao
  muda, e o composto contra a QuaPy cai de 27,4x para 9,9x.

### O `random_state` do `solve_simplex` na mlquantify

Encontrado ao explicar por que a qunfold custa o que custa, comparando os
solucionadores das duas. **Fica anotado e nao foi consertado**, por decisao de
quem conduz o experimento: o efeito medido esta abaixo do que a analise
reporta, e mexer na biblioteca agora invalidaria as medicoes.

**O que acontece.** O `minimize_prevalence` da mlquantify despacha por
estrutura: com duas classes vai para `solve_binary`, que no modo `bounded`
(o padrao) e uma busca escalar em [0, 1] e **nao tem parametro `x0`**; com
mais classes vai para `solve_simplex`, que resolve por SLSQP com
`bounds=[(0,1)]*n` e a restricao de soma. E ali esta a surpresa: o `x0` do
SLSQP e `1/n` **quando `random_state` e None**, e um ponto aleatorio do
simplex quando nao e. O experimento passa `random_state=42` a toda classe
que o aceite (ver `_protocolo` em `registry.py`), entao as combinacoes
multiclasse partem de um ponto arbitrario, nao do uniforme.

Nao e aleatoriedade por amostra: `_random_simplex_start(42, n)` refaz o
`RandomState` a partir de 42 em cada chamada, de modo que e **um ponto fixo
arbitrario**, igual nas 400 amostras. A execucao se repete.

**Quanto isso muda, medido.** Isolando o solucionador -- mesmo `M`, mesmo `q`,
so o `x0` diferente, com dados sinteticos:

| objetivo | n | sacos que mudaram | dif. mediana | dif. maxima |
| --- | --- | --- | --- | --- |
| minimos quadrados | 5 | 79,0% | 2,5e-6 | 1,3e-5 |
| Hellinger | 5 | 5,0% | 2,1e-9 | 6,9e-6 |
| minimos quadrados | 50 | 100% | 6,3e-5 | **2,6e-4** |
| Hellinger | 50 | 85,0% | 2,6e-6 | 1,3e-5 |

O pior caso -- 50 classes, que 6 conjuntos do catalogo tem -- move a
estimativa em no maximo 2,6e-4, **uma ordem de grandeza abaixo do limiar de
1e-3** com que a Q1 chama duas bibliotecas de iguais. Nao vira veredito
nenhum. E o alcance e menor do que parece: 108 dos 142 conjuntos (76,1%) sao
binarios e vao pelo caminho que ignora o `x0`; so os 34 multiclasse chegam
ao `solve_simplex`.

**E o que os numeros mostram e tolerancia, nao dois otimos.** O objetivo
convexo move *mais*, o que a primeira vista e ao contrario: minimos quadrados
tem otimo unico, as duas partidas chegam nele, e o que sobra e a folga de
parada do SLSQP -- maior ali porque a superficie e mais plana perto do otimo.
O Hellinger normalizado tem bacia mais afiada e as duas partidas caem dentro
de 2e-9 na maioria dos sacos. Nenhum sinal de multimodalidade morder.

**O custo.** 1,594 para 1,851 ms por resolucao nos minimos quadrados com
n=5 (+16%); 13,1 para 13,6 ms no Hellinger (+3,6%); com n=50, 1,00x e 1,12x.
Alguns por cento do tempo de estimacao multiclasse num quarto dos conjuntos,
invisivel dentro das medianas da Q2.

**Por que ainda assim vale consertar depois.** Nao pelos numeros, pela
semantica: na convencao do scikit-learn o `random_state` **semeia**
aleatoriedade que ja existe, e aqui ele a **introduz**. Quem passa
`random_state=42` para ter reprodutibilidade recebe uma partida pior
condicionada do que quem nao passa nada -- o parametro pune o usuario
cuidadoso. E o padrao descartado e o melhor: `1/n` e o prior desinformado e
fica perto das solucoes tipicas, enquanto o ponto aleatorio esta a 0,477 em
L1 dele com n=50, que e exatamente por que custa mais iteracoes. O conserto
seria `x0` uniforme sempre, `x0` exposto para quem quiser escolher, e
reinicios aleatorios -- se um dia forem desejados, para sondar
multimodalidade -- com parametro proprio em vez de pegar carona no
`random_state`.

**Uma ressalva sobre o reparo final.** O `solve_simplex` termina com
`np.clip(prevalence, 0, None)` e renormaliza. Isso e arremate numerico depois
de um solucionador que ja impos os limites e a igualdade -- o SLSQP satisfaz
restricoes dentro da tolerancia dele, nao exatamente --, e nao se confunde
com o `clip` da quapy, que e a unica coisa entre uma solucao sem restricao e
o simplex.

### Limitacoes de escala observadas na execucao

As anteriores aparecem ao escrever o codigo. Estas so aparecem ao rodar, e
quase todas sao a mesma coisa vista de angulos diferentes: **o metodo monta
uma estrutura quadratica no numero de instancias ou de atributos.** Nenhuma
e erro de adaptacao; o conjunto apenas passou do ponto em que a formulacao
se sustenta. O limite por metodo e de 30 min de tempo de CPU.

| Biblioteca | Metodo | O que acontece |
| --- | --- | --- |
| QuantificationLib | `SORD` | nao termina as 400 amostras em 30 min em **nenhum** conjunto acima de ~10 mil instancias |
| QuantificationLib | `HDx` | `list index out of range` em dados de alta dimensionalidade; em dois casos devolveu um unico valor onde cabiam varias prevalencias |
| qunfold | `GHDx` | nao converge em 30 min em dados de alta dimensionalidade (`gina_agnostic` com 970 atributos, `madelon` com 500) |
| QuaPy, QuantificationLib, qunfold | `EDx`, `EDy`, `MMD` | pedem a matriz de distancia entre todas as instancias: 70 GiB em um conjunto de 245 mil instancias |
| QuaPy, mlquantify | `KDEyCS` | mesma causa, 18 GiB no mesmo conjunto |
| QuaPy | `SMM` | mesmo com `newOneVsAll`, produz prevalencias negativas ou que nao somam 1 em problemas multiclasse |
| QuaPy, QuantificationLib | `EDx` | `constraints are inconsistent, no solution` quando a matriz de distancia degenera |
| mlquantify | metodos de limiar (`CDE`, `HDy`, `MS`, `T50`, `X`, `MAX`, `MS2`, `ACC`, `TAC`) | devolvem um vetor inteiramente nulo em amostras pequenas de problemas multiclasse (ver abaixo) |

#### As 36 combinacoes que ficaram sem medida

A execucao `final-alinhado` fechou com **9 904 das 9 940 combinacoes
planejadas (99,6%)**. As 36 que faltam tem uma causa unica, e nao e o
metodo: **todas sao cauda de um trabalho que estourou o limite de job.**

Os 12 pares (conjunto, biblioteca) com pendencia sao 12 dos 27 pares que
gravaram a sentinela `__job__`, e em todos os 12 o que falta e um trecho
**contiguo** que vai de alguma posicao da ordem do registro ate a ultima.
Nao ha buraco no meio de nenhuma lista.

| Conjunto | Inst. | Atrib. | Cl. | Biblioteca | Faltam | A partir de |
| --- | --- | --- | --- | --- | --- | --- |
| `dataset_1457_amazon-commerce-reviews` | 1 500 | 10 000 | 50 | mlquantify | 12 de 30 | `SORD` (19) |
| `dataset_1457_amazon-commerce-reviews` | 1 500 | 10 000 | 50 | QuaPy | 5 de 20 | `KDEyHD` (16) |
| `dataset_1457_amazon-commerce-reviews` | 1 500 | 10 000 | 50 | qunfold | 3 de 8 | `MMD` (6) |
| `dataset_44478` a `dataset_44482` (5x) | 1 500 | 100 | 50 | qunfold | 2 de 8 cada | `GHDx` (7) |
| `music_genre` | 17 996 | 16 | 11 | qunfold | 3 de 8 | `MMD` (6) |
| `Mosquitoes` | 158 249 | 53 | 6 | qunfold | 1 de 8 | `MMD` (6) |
| `Mfeat` | 2 000 | 6 | 10 | qunfold | 1 de 8 | `EDx` (8) |
| `click-prediction` | 39 948 | 11 | 2 | qunfold | 1 de 8 | `EDx` (8) |

O que faz cada trabalho demorar varia -- 10 mil atributos em um caso, 50
classes com 1 500 instancias em outro, 158 mil instancias no terceiro -- e
**19 das 36 estao na qunfold**, a biblioteca mais cara por combinacao
(103,5 s de media contra 12 a 17 s das outras tres). Nao e, portanto, uma
limitacao de dimensionalidade: os cinco conjuntos `44478`-`44482` tem 100
atributos, e o `Mfeat` tem 6.

Isso expoe um vies de medicao que vale declarar. O `specs_for` respeita
apenas a ordem do registro: nao ha reordenacao por custo, nem retomada do
ponto em que o trabalho parou dentro da lista. Quando o limite de job
dispara, **quem fica sem medida e sempre o fim da lista**, de modo que um
metodo registrado por ultimo esta sistematicamente mais exposto do que um
registrado no inicio. Uma execucao retomada fecha a cauda, porque o ledger
pula o que ja foi medido -- salvo quando o metodo que travou nao devolve
nunca, e ai a cauda atras dele fica bloqueada: foi o que o `MMD` da qunfold
fez em `Mosquitoes`, em tres tentativas e cerca de 33 h de CPU sem gravar
uma linha. O contorno usado aqui foi tirar o metodo da frente com
`--skip-methods`, nao mexer na ordem.

#### O vetor nulo da mlquantify

Verificado em `IRIS.csv` (3 classes), reproduzindo o protocolo: o `CDE`
devolve `[0, 0, 0]` em **1** das 400 amostras e o `T50` em **2**. Em ambos
os casos a amostra tem `m = 10`, o menor tamanho do protocolo; nenhuma
amostra de 50, 100 ou 500 instancias foi afetada. Nao sao prevalencias
negativas: a decomposicao um-contra-resto estima zero para todas as
classes, e nao sobra nada para normalizar.

O mecanismo foi rastreado em `news_popularity_data.csv` (39 644 instancias,
4 classes), onde o `ACC` e o `TAC` da mlquantify devolvem o vetor nulo na
maior parte das amostras pequenas. O numero depende da execucao, e vale
registrar as tres, porque a diferenca entre elas diz algo:

| Execucao | O que muda | `ACC` e `TAC` |
| --- | --- | --- |
| `final` | antes do alinhamento do protocolo de escores | 169 de 400 nulas |
| `sens_escalado` | classificador padronizado | 134 de 400 nulas |
| `final-alinhado` | a execucao de referencia | 130 de 400 nulas |

Na `final-alinhado` as duas combinacoes **nao aparecem** em
`amostras_descartadas.csv`, e nao e porque pararam de anular: com 130
invalidas contra o limite de 20 (`MAX_INVALID_FRACTION = 0.05`), a
combinacao inteira e reprovada e sai com `status = erro`. O descarte
parcial e para amostra isolada; isto nao e isolado.

A forma do problema por tamanho de amostra, reproduzindo a combinacao com o
limite desarmado (uma thread de BLAS e nucleo fixo, como na execucao -- ver
a ressalva adiante, que nao e detalhe):

| m | `ACC` e `TAC` | `T50` |
| --- | --- | --- |
| 10 | 79 de 100 nulas | 7 de 100 |
| 50 | 27 de 100 | 0 de 100 |
| 100 | 21 de 100 | 2 de 100 |
| 500 | **3 de 100** | 0 de 100 |

O `ACC` e o `TAC` dao o mesmo resultado nos quatro tamanhos, ate a sexta
casa do MAE: nao sao dois achados, e um medido duas vezes. O efeito cai com
o tamanho da amostra, como esperado, mas **nao desaparece**: ainda ha 3% de
amostras anuladas em `m = 500`. Os totais fecham com o ledger -- 130, 130 e
9 --, e o `T50` fecha tambem no padrao por tamanho (`10:7|100:2`) e nos
quatro valores de MAE.

**Essa reproducao exige o mesmo arranjo de threads da execucao, e isso vale
como condicao de reproducao, nao como metadado.** Reproduzindo a mesma
combinacao com o BLAS livre nos 20 nucleos -- que e o que acontece ao
chamar o `run_job` direto, sem passar pelo `limit_threads` do `expq/cli.py`
e pelo `pin_to_core` do `expq/parallel.py` -- saem 135 nulas em vez de 130,
e o `T50` sai com 11 e com outro padrao (`10:9|50:1|100:1`). Nao e ruido de
contagem: o padrao por tamanho muda, ou seja a estimativa de cada amostra
muda.

A causa esta no classificador deste conjunto, e so vale onde ele se
comporta assim: como a regressao logistica **nao converge** em 1 000
iteracoes (ver abaixo), o solver e interrompido no meio, e o resultado de
um solver iterativo interrompido depende da ordem em que as somas sao
reduzidas -- que muda com o numero de threads do BLAS. Coeficientes
ligeiramente diferentes deslocam os escores, e outras amostras passam a
cruzar o ponto em que o truncamento zera todas as classes. Com o mesmo
numero de threads o resultado e deterministico: duas execucoes livres
deram 135 identico, e a execucao de uma thread com nucleo fixo reproduz o
ledger exatamente.

A primeira consequencia e operacional: quem quiser conferir um numero desta
execucao precisa de `OMP_NUM_THREADS=1` e nucleo fixo, nao basta a semente
-- e o `summary.csv` ja grava a coluna `threads` por isso.

#### Em quantos conjuntos o classificador nao converge

A segunda consequencia depende da extensao, e ela foi medida. Ajustando o
mesmo classificador que o harness entrega as bibliotecas
(`make_classifier_factory`, mesma particao, mesma semente, uma thread e
nucleo fixo) em cada um dos 142 conjuntos, e lendo `n_iter_` contra o teto
de 1 000:

```
104 de 142 conjuntos convergem
 38 de 142 batem o teto  (26,8%)
```

O `ConvergenceWarning` do scikit-learn concorda exatamente com o criterio
`n_iter_ >= 1000`: 104 sem aviso e 38 com, sem um unico caso cruzado. E
projetando nas medidas, **2 653 das 9 904 (26,8%) assentam em um
classificador que foi interrompido antes de convergir** -- nao e um caso
isolado do `news_popularity`, e um quarto do experimento.

O que prediz a nao convergencia nao e o tamanho, e nao e so a magnitude:

| Magnitude maxima do atributo | Conjuntos | Batem o teto |
| --- | --- | --- |
| <= 1 | 9 | 0 |
| 1 a 10 | 34 | 0 |
| 10 a 1 000 | 51 | 9 (17,6%) |
| 1 000 a 100 000 | 29 | 16 (55,2%) |
| > 100 000 | 17 | 13 (76,5%) |

**Escala sem padronizacao e necessaria, mas nao suficiente.** Nenhum dos 43
conjuntos com atributos ate 10 bate o teto -- nem o `electricity-normalized`,
com 45 312 instancias --, o que torna a padronizacao um remedio plausivel.
Mas magnitude grande nao condena: o `star_classification`, com atributos
chegando a 1,4 x 10^19, **converge em 23 iteracoes**, enquanto o
`Dermatology`, com maximo de 82, bate o teto. Entre os onze conjuntos acima
de 50 mil instancias, tres convergem em menos de 40 iteracoes e oito batem
o teto. Ou seja: o que decide e o condicionamento do problema, nao o
tamanho nem a escala isoladamente, e nao ha regra simples para prever quais
conjuntos estao expostos -- a lista tem de ser medida, como aqui.

Fica entao a limitacao, agora com numero: um quarto das medidas depende do
arranjo de threads para ser reproduzido bit a bit. **O que nao foi medido e
se essa sensibilidade se cancela na diferenca entre bibliotecas**, que e a
unidade de analise de Q1. O argumento de que deveria cancelar existe -- a
configuracao do classificador e a mesma para as quatro --, mas e so um
argumento: cada biblioteca reajusta o proprio classificador internamente
(a mlquantify por um-contra-resto, a QuaPy pela validacao cruzada dela), de
modo que o deslocamento nao e necessariamente identico entre elas. Medir
isso pede reexecutar uma celula nas quatro bibliotecas com os dois arranjos
de thread e comparar as diferencas pareadas, o que nao foi feito.

Os dois sao binarios na mlquantify, e o multiclasse passa por
um-contra-resto: cada sub-quantificador ajusta a propria regressao
logistica binaria e corrige a contagem com

```
prevalencia_k = clip((cc_k - fpr_k) / (tpr_k - fpr_k), 0, 1)
```

Medido neste conjunto:

| classe | prevalencia | TPR | FPR | TPR-FPR | % predito positivo |
|---|---|---|---|---|---|
| 0 | 0,3134 | 0,0282 | 0,0121 | 0,0160 | 1,66% |
| 1 | 0,4125 | 0,0051 | 0,0031 | 0,0020 | 0,40% |
| 2 | 0,1462 | 0,0866 | 0,0802 | 0,0064 | 0,05% |
| 3 | 0,1279 | 0,0856 | 0,0845 | 0,0011 | 0,23% |

As quatro taxas somam 2,3%, nao 100%: os binarios sao independentes, nao
uma particao por `argmax`. Quase nunca preveem a classe positiva, entao na
amostra `cc_k = 0` para todas as classes, `cc_k - fpr_k` fica negativo nas
quatro, **cada uma e truncada em zero antes de combinar**, e sobra
`[0, 0, 0, 0]`. O `normalize_prevalences` da biblioteca faz
`if total > 0: arr /= total`, de modo que o vetor nulo passa sem erro: o
metodo devolve zeros em silencio.

O corte e da decomposicao, nao do conjunto. No mesmo ajuste, o `GACC` --
a formulacao generalizada, multiclasse por construcao -- da **0 de 400**
nulos, e o `ACC` da QuaPy e o da QuantificationLib concluem normalmente.
Entre os de limiar, `MAX`, `MS`, `MS2` e `X` tambem concluem: escolhem um
limiar com separacao maior, onde `cc_k` nao chega a zerar.

O classificador deste conjunto ajuda a anular, mas nao e a causa. O
`classifier_pipeline` e `plain` (Secao 10), sem padronizacao, e aqui os
atributos vao de -1 a 843 000: a regressao logistica **nao converge** em
1 000 iteracoes e termina com acuracia de 0,4251 contra 0,4125 da classe
majoritaria. Com `--scaled-classifier` ela converge em 0,4 s em vez de
6,9 s e sobe para 0,4676. Refazendo so estas combinacoes assim
(`--run-id sens_escalado`):

| metodo | sem padronizar | com padronizar |
|---|---|---|
| `ACC` | 169 de 400 nulas | **134 de 400 nulas** |
| `TAC` | 169 de 400 nulas | **134 de 400 nulas** |
| `T50` | 24 de 400 nulas | 0 nulas, MAE 0,231 a 0,177 |
| `GACC` | 0 nulas, MAE 0,286 a 0,282 | 0 nulas, MAE 0,277 a 0,243 |

Ou seja: padronizar conserta o `T50`, melhora um pouco o erro de todos --
e **nao resolve o `ACC` nem o `TAC`**, que continuam anulando um terco das
amostras com um classificador convergido. O truncamento antes da combinacao
e o que basta para zerar o vetor; o classificador ruim so aumenta a conta.
Como ele e o mesmo para as quatro bibliotecas, a comparacao continua justa:
a diferenca e que as outras tres degradam e a mlquantify anula.

O alcance disto na execucao de referencia e maior do que o caso rastreado.
Contando no ledger as combinacoes reprovadas com motivo de soma nula:

```
20 combinacoes perdidas inteiras, todas mlquantify
  = 20 dos 96 erros da execucao (21%)
  = 20 dos 27 erros da mlquantify (74%)
CDE 10 | ACC 3 | TAC 3 | MS 2 | MS2 1 | HDy 1
```

As piores sao `news_popularity | CDE` com 378 de 400 nulas, `yeast_data |
CDE` com 347 e `customer_segmentation | CDE` com 219. Alem dessas, outras
41 combinacoes sobreviveram com descarte parcial, somando 216 amostras
perdidas -- tambem todas da mlquantify, contra zero das outras tres.

Isso tem efeito direto em Q3. A taxa de conclusao da mlquantify e 0,9868, e
**tres quartos dos erros que a derrubam sao este unico defeito**; sem ele a
taxa iria para cerca de 0,9915. Ou seja, a biblioteca esta sendo penalizada
na metrica de esforco de uso por um caso que a propria normalizacao dela
resolveria -- ver a subsecao final desta secao.

#### O `HDx` da mlquantify nao acompanha a amostra

E o unico lugar da execucao em que duas bibliotecas concordam entre si e
discordam da mlquantify por um fator grande. Em `mlquantify 0.5.1`, sobre
os casos pareados das 142 bases:

| Comparada | Casos | AE mlquantify | AE da outra | Dif. abs. media | Iguais ate 1e-3 | mlq. melhor |
| --- | --- | --- | --- | --- | --- | --- |
| QuantificationLib | 524 | **0,2561** | 0,0986 | 0,1600 | 3,6% | 8,2% |
| QuaPy | 564 | **0,2505** | 0,1053 | 0,1519 | 1,8% | 13,7% |

A QuantificationLib e a QuaPy caem em 0,0986 e 0,1053 -- concordam entre si
--, a mlquantify fica 2,5x acima das duas, e e melhor em apenas 8% e 14%
dos casos. Nao e a familia dos metodos sobre atributos crus:

| Metodo | Comparada | Dif. abs. media | Iguais ate 1e-3 |
| --- | --- | --- | --- |
| `EDx` | QuaPy | 0,0014 | 95,9% |
| `EDx` | QuantificationLib | 0,0024 | 95,2% |
| `EDx` | qunfold | 0,0026 | 94,1% |
| `GHDx` | qunfold | 0,0063 | 23,9% |
| `HDx` | QuaPy | **0,1519** | 1,8% |
| `HDx` | QuantificationLib | **0,1600** | 3,6% |

O `GHDx` merece a ressalva: 23,9% de igualdade ate 1e-3 e baixo, mas a
diferenca media dele e de 0,0063 -- ele concorda em magnitude e difere no
digito fino, que e o esperado de duas buscas numericas distintas sobre as
mesmas bordas. O `HDx` difere em magnitude, que e outra coisa.

A causa esta identificada e corrigida em `coenlab/mlquantify#9`: o `HDx`
monta histograma sobre os atributos crus, mas herdou o intervalo dos irmaos
de espaco de escores (`range=(0, 1)` com `bin_edges="fixed"`), que e o
certo para `HDy` e `DyS`, cuja entrada e uma posteriori, e errado para
valor de atributo. O `np.histogram` descarta o que cai fora do intervalo,
entao em dados nao padronizados a maior parte de cada atributo era jogada
fora e **a estimativa parava de depender da amostra de teste**: virava um
predito quase constante. O `GHDx` nao foi afetado porque ajusta as
proprias bordas -- o que explica por que ele concorda aqui e o `HDx` nao.

Duas consequencias para o texto. A primeira: o numero de `HDx` da
mlquantify medido aqui **e o da versao publicada, com o defeito**. Quem
repetir a execucao com a correcao aplicada obtem outro valor, e a tabela
acima precisa dizer a versao. A segunda e metodologica. A PR mede o
defeito em dados sinteticos, com seis atributos e prevalencia conhecida;
esta execucao o mede em 142 conjuntos e 1 088 casos pareados, sem ter sido
montada para isso. Duas implementacoes independentes concordando entre si
e discordando da terceira e o tipo de sinal que um experimento de
equivalencia produz de graca, e vale mais do que qualquer uma das tres
comparada a um valor de referencia unico.

Dois contrastes valem a discussao de Q1:

* **mlquantify e QuaPy completam `HDx` onde as outras duas nao completam
  -- mas concluir nao e acertar.** Em `gina_agnostic` (970 atributos), o
  `HDx` da mlquantify termina em 4.815 s e o da QuaPy em 1.042 s, enquanto
  a QuantificationLib erra e a qunfold estoura o tempo. O da mlquantify
  conclui com a estimativa errada, pelo motivo da subsecao anterior, de
  modo que o contraste e de cobertura, nao de qualidade.
* **A formulacao generalizada da mlquantify e mais barata que a classica
  dela mesma.** No mesmo conjunto, `GHDx` leva 194 s contra 4.815 s do
  `HDx` -- uma ordem de grandeza. Em `312_scene`, 65 s contra 1.254 s.

### O que esta corrigido e nao entrou na medicao

Tudo nesta secao foi medido contra a `mlquantify 0.5.1` instalada do PyPI
(instalacao normal, nao editavel: a venv do experimento nao ve o checkout de
desenvolvimento). Parte do que a secao relata tem correcao escrita, revisada
e fundida na `master` da biblioteca; as PRs #9 e #10 foram medidas na
execucao `sens_corrigido` e a #6 na `sens_pr6`, no fim desta secao.

| PR | O que muda | Afeta | Esta na 0.5.1? |
| --- | --- | --- | --- |
| #9 `HDx` | bordas do histograma por atributo | **erro do `HDx`**: MAE de 0,2505 para 0,1005; desacordo pareado de 0,1600 para 0,0055 | nao |
| #10 vetor nulo | `normalize_prevalences` devolve a uniforme no lugar de zeros | **cobertura de 11 metodos**: as 20 combinacoes perdidas e as 216 amostras descartadas voltam; erro quase nao muda | nao |
| #6 `MLPE` | ajusta o estimador uma vez em vez de validar cruzado | **custo do `MLPE`** (os 1 376x e 67x acima) | nao |
| #8 `CDE` | so docstring, nenhuma linha de codigo | nada | nao |
| #7 CI | so o workflow de teste | nada | nao |

**Estado das correcoes em 09/11/2026.** As cinco foram revisadas e fundidas
na `master` de `coenlab/mlquantify`, cada uma como um commit de squash: #7
em `540db346`, #10 em `51d43830`, #6 em `ce89bad1`, #8 em `68e23a64` e #9
em `8340e071`. A revisao achou dois defeitos anteriores de documentacao, que
foram para PRs proprias (`#11`, a entrada de `bin_edges` que faltava no
`HistogramRepresentation`, e `#13`, o rotulo da secao v0.3.0 no CHANGELOG), e
as mudancas foram registradas no CHANGELOG em `#12`. Isso nao muda nada do
que esta medido aqui: a coluna acima continua valendo, porque a `master` nao
e uma versao publicada e a medicao usou a 0.5.1 do PyPI.

Duas verificacoes que sustentam a comparacao, e que vale registrar porque o
harness depende da biblioteca que ele avalia:

* **A geracao de amostras nao mudou.** O protocolo vem da propria
  mlquantify (`from mlquantify.model_selection import UPP`, em
  `expq/protocol.py`). O diff de `model_selection/_protocol.py` entre a
  0.5.1 e o branch de desenvolvimento toca apenas o `apply_protocol`; o
  `BaseProtocol.split`, o `APP` e o `UPP` ficam intocados, e o harness usa
  so o `UPP(...).split()`, calculando as metricas por conta propria. As 400
  amostras por combinacao sao as mesmas.
* **Nada publicado depois da 0.5.1 altera comportamento.** O diff de
  `v0.5.1` para `v0.5.2b3` em `mlquantify/` e vazio. As tres betas da
  `v0.5.2` estao marcadas na `master`, e as correcoes acima entraram depois
  da `v0.5.2b3`: uma `v0.5.2` final passa a carregar o `HDx` com bordas por
  atributo e o vetor nulo corrigido. A medicao descreve a 0.5.1, e a analise
  de sensibilidade adiante antecipa essa release.

E uma ausencia, que nao e correcao: **o grupo `elm` nao existe na 0.5.1.**
O pacote instalado nao tem o modulo; o branch de desenvolvimento adiciona
`elm/_classes.py` e `elm/_svmperf.py` do zero, e mais 732 linhas em
`neural/_classes.py`. Os metodos da Secao 6 nao estao apenas deixados para
depois por custo: na versao publicada eles nao existem.

Qualquer reexecucao com as correcoes aplicadas deve usar um `--run-id`
proprio e entrar no texto como analise de sensibilidade, nunca sobrescrever
a `final-alinhado`: as 9 904 medidas dela foram tomadas com a 0.5.1, e
trocar a versao no meio tornaria a execucao heterogenea.

#### A execucao de sensibilidade `sens_corrigido`

As PRs #9 e #10 foram aplicadas e medidas. A execucao rodou de
2026-11-06 14:53 a 2026-11-07 02:38, com 217 combinacoes: 216 `ok`, 1
`timeout`, 22,3 h de CPU acumuladas.

**O arranjo.** As duas correcoes foram aplicadas **sobre a `v0.5.1`**, nao
sobre o branch de desenvolvimento. Isso e deliberado: pelo branch, a
diferenca medida incluiria os modulos novos e as remocoes no
`apply_protocol`, e nao seria atribuivel as correcoes. A biblioteca
corrigida foi para uma venv separada (`.venv-corrigido`); a venv do
experimento nao foi tocada, e a `final-alinhado` segue reproduzivel.

O `pip install` do fonte falha nesta maquina: a mlquantify compila a
extensao Cython `matching/_histogram_sweep` e nao ha compilador C
disponivel. O contorno foi instalar a `0.5.1` do PyPI, cujo wheel traz o
`.pyd` pronto, e aplicar os patches nos `.py` instalados. Antes disso foi
conferido que esses `.py` sao identicos aos da tag `v0.5.1`: diferem
apenas em CRLF contra LF, mesmo numero de linhas, nenhuma diferenca de
conteudo.

**O controle, que e o que autoriza ler o resto.** Doze combinacoes de
`GACC`, `CC` e `PCC` -- metodos que nenhuma das correcoes toca -- rodaram
no mesmo arranjo. Nas 48 linhas pareadas, a diferenca maxima contra a
`final-alinhado` e exatamente `0.000e+00` em MAE, MRAE, MSE e NKLD. O
arranjo reproduz **o erro** da execucao de referencia exatamente, de modo
que as diferencas de erro abaixo sao das correcoes e nao do ambiente. O
custo nao reproduz exatamente, e nao poderia: ver a ressalva na
`final-corrigido`, adiante.

**PR #9, o `HDx`.** Medido nos 142 conjuntos; 564 linhas que as duas
execucoes mediram:

| | antes | depois |
| --- | --- | --- |
| MAE | 0,2505 | **0,1005** |
| MRAE | 1,3877 | **0,6052** |
| MSE | 0,1216 | **0,0269** |

Melhora em 516 das 564 linhas, e em todos os tamanhos de amostra: de
0,2944 para 0,1751 em m=10 e de 0,2180 para **0,0552** em m=500.

No pareado contra as outras bibliotecas, que e a unidade da Q1:

| | antes | depois |
| --- | --- | --- |
| contra quantificationlib (524 casos) | 0,1600 | **0,0055** |
| contra quapy (564 casos) | 0,1519 | **0,0137** |

A media da mlquantify cai de 0,2561 para 0,1024 contra os 0,0986 da
quantificationlib, e de 0,2505 para 0,1005 contra os 0,1053 da quapy. A
concordancia ate 0,01 sobe de 13,2% para 84,9% contra a quantificationlib
e de 13,8% para 63,7% contra a quapy. **O `HDx` deixa de ser o pior
desacordo da Q1** -- sai da lista das cinco maiores diferencas, que passa
a ser encabecada pelo `HDy` contra a quantificationlib (0,0436). O que a
`final-alinhado` registrou como desacordo entre implementacoes era o
defeito das bordas do histograma.

Esse `HDy` na cabeca da lista tambem nao durou: a subsecao da
`final-separado` mostra que o `HDy` da quantificationlib e uma variante de
intervalo fixo, e nao a mesma coisa que o `HDy` da mlquantify e da quapy.
Com ele e o `KMM` fora do pareamento, a lista passa a ser encabecada pelo
`MS` contra a quapy (0,0384).

**PR #10, o vetor nulo.** As 20 combinacoes que a `final-alinhado` perdeu
inteiras com `TooManyInvalidSamples` **completam todas**: `CDE` 10, `ACC`
3, `TAC` 3, `MS` 2, `MS2` 1, `HDy` 1. O descarte parcial tambem zera: as
216 amostras descartadas em 41 combinacoes caem para **nenhuma**, nos onze
metodos afetados.

O efeito no erro, porem, e pequeno. Nas linhas que as duas execucoes
mediram, o MAE anda na terceira ou quarta decimal -- `ACC` 0,2036 para
0,2027, `MS` 0,1709 para 0,1698, `CDE` 0,1779 para 0,1778 -- porque o que
muda e a inclusao das amostras antes descartadas, nao o calculo das que
ja entravam. **A correcao do vetor nulo e um ganho de cobertura, nao de
erro**, e o texto nao deve apresenta-la como melhora de acuracia.

A taxa de conclusao da mlquantify na Q3 sobe de 0,9868 para **0,9915**
(4 212 de 4 248), com as 20 combinacoes recuperadas.

**O unico `timeout`.** `HDx` em `dataset_1457_amazon-commerce-reviews`
(10 000 atributos, e o `HDx` monta histograma por atributo). Nao e
regressao: no ledger da `final-alinhado` esse par mlquantify x `HDx`
tambem nao tem medida -- esta entre as 36 que o limite de job cortou. A
cobertura do `HDx` e 141 de 142 nas duas execucoes.

#### A execucao `final-corrigido`: Q1, Q2 e Q3 com as correcoes

Os numeros acima sao da comparacao direta. A tabela oficial exige a
`analysis.py`, que expande cada metodo pelos seus apelidos
(`expand_aliases`) e por isso pareia mais casos que um recalculo a mao --
7 688 contra a quantificationlib, nao os 5 984 de um merge simples. Para
obter os numeros com as correcoes foi montada a execucao
`final-corrigido`; os numeros oficiais terminam na `final-separado`, duas
subsecoes adiante, que separa dois pares de metodos que nao eram o mesmo
metodo.

**Como ela foi montada, e o que nao foi tocado.** Partiu da
`final-alinhado` e substituiu **apenas as 202 combinacoes em que a
correcao de fato morde**: as 141 de `HDx` e as 61 da PR #10 -- as 20 que
tinham dado `TooManyInvalidSamples` e as 41 que tinham descarte parcial
por soma nula. Todo o resto veio da referencia, inclusive os 12 controles.
Isso e deliberado: substituir uma linha cujo erro nao muda troca a medida
de **custo** por uma tomada em outra sessao, e injetaria ruido de
cronometragem na Q2 sem ganho nenhum.

A verificacao confirma o corte: as tres outras bibliotecas ficam
**identicas linha a linha**, e os 18 metodos da mlquantify que as
correcoes nao tocam tambem -- `GACC`, `CC` e `PCC` entre eles. Nos 12
metodos tocados mudam exatamente as 57 linhas que tinham descarte, mais as
80 linhas das 20 combinacoes recuperadas. A cobertura nao muda: 9 904
medidas, as mesmas 36 faltando.

**Uma distincao que o controle revelou e que vale registrar.** O erro
reproduz exatamente entre as duas sessoes -- `0.000e+00` em MAE, MRAE, MSE
e NKLD. O **custo nao reproduz**, e nao tem como: nas 16 linhas de cada
controle o tempo varia com razao mediana de 1,000 mas diferenca de ate
9 s de treino. Medida de erro e deterministica dado o pareamento de
amostras; medida de tempo nao e. Qualquer afirmacao de reprodutibilidade
bit a bit vale para o erro, nunca para o custo.

**Q1.** O `HDx` sai da lista de desacordos e os agregados melhoram:

| | referencia | corrigido |
| --- | --- | --- |
| quantificationlib, dif. media | 0,0195 | **0,0090** |
| quantificationlib, iguais ate 0,01 | 80,7% | **85,4%** |
| quantificationlib, correlacao | 0,8653 | **0,9628** |
| quapy, dif. media | 0,0155 | **0,0091** |
| quapy, iguais ate 0,01 | 80,6% | **82,8%** |
| quapy, correlacao | 0,9066 | **0,9665** |
| qunfold | 0,0097 | 0,0097 (nao implementa `HDx`) |

A maior diferenca absoluta cai de 0,5742 para 0,4537 contra a
quantificationlib e de 0,5747 para 0,4127 contra a quapy. O numero de
metodos que acusam diferenca nos testes **nao muda**: 13 de 28 (46,4%). A
correcao aproxima as medias sem alterar o veredito dos testes de
hipotese, e o texto nao deve sugerir o contrario.

A nova lista dos cinco piores desacordos nao tem mais `HDx`: `HDy` contra
quantificationlib (0,0434), `MMD` contra qunfold (0,0428), `MS` contra
quapy (0,0384), `PWK` contra quantificationlib (0,0355), `MS2` contra
quapy (0,0203). Nenhum chega a um decimo do que o `HDx` media antes.

Dos dois primeiros se descobriu depois que nao eram desacordo de
implementacao, e sim metodo diferente de cada lado; a execucao
`final-separado`, adiante, os separa e e de onde saem os numeros oficiais.

**Q2, e um achado que nao era esperado.** A correcao do `HDx` **tambem o
deixa mais barato**: razao mediana de 0,425 no tempo de estimacao e 0,652
no de treino, contra a propria medida anterior. Nao e ruido de sessao --
nos controles essa razao e 1,000. Pedir as bordas ajustadas custa menos
que o caminho que o `HDx` tomava. Com isso o tempo de treinamento da
mlquantify contra a quantificationlib vai de 0,90x para **0,88x** (mais
barata em 77% dos casos, contra 73%) e o de estimacao de 1,33x para
1,31x. As demais razoes andam na segunda decimal ou menos.

Vale a ressalva de escopo: das 6 088 linhas de custo dos metodos da
PR #10, 156 (2,6%) foram re-medidas, com razao mediana de 1,064 no treino
e 1,002 na estimacao -- simetrico, sem vies. O deslocamento da Q2 vem do
`HDx`, nao da re-medicao.

**Q3.** A taxa de conclusao da mlquantify vai de 0,9868 para **0,9915**
(4 212 de 4 248). As outras tres bibliotecas nao mudam, e a mlquantify
segue a unica com taxa de configuracao manual igual a de conclusao.

A `final-corrigido` esta em `results/final-corrigido/`, com as tabelas em
`analise/`. As linhas guardam o `run_id` de origem, de modo que da para
separar o que veio da `final-alinhado` do que veio da `sens_corrigido`.
A `final-alinhado` nao foi alterada.

#### A execucao `final-separado`: dois metodos que nao sao o mesmo metodo

**A regra e anterior aos dois casos, e isso importa para ler o que vem
depois.** O `registry.py` ja a enunciava para as formulacoes generalizadas
da qunfold: "sao formulacoes distintas e **nao devem ser comparadas entre
si**". Por isso `qunfold.HDy`, `qunfold.HDx`, `qunfold.ACC` e
`qunfold.PACC` estao registrados como `GHDy`, `GHDx`, `GACC` e `GPACC`,
sem apelido para a versao sem o G. Faltava aplica-la a dois pares que
estavam casados pelo nome.

**O `HDy` da quantificationlib.** A nota do registro ja dizia, antes de
qualquer medicao, que "esta coluna e uma variante de intervalo fixo". A
classe confirma: `n_bins=8`, `bin_strategy=equal_width`, os padroes dela.
A definicao do metodo varre intervalos e toma a mediana das estimativas,
que e o que a mlquantify e a quapy fazem. Sao dois estimadores. Passa a
`HDyFix`.

**O `MMD` da qunfold.** `qunfold.KMM` e kernel mean matching com nucleo de
energia; o `MMD` da mlquantify e `MMD_RKHS` com `kernel='rbf'`. Mesmo
arcabouco, nucleo diferente. Passa a `KMM`. Esta era uma escolha
deliberada -- o registro trazia a nota "KMM com nucleo de energia" --, e
foi revista.

**O que decide cada caso nao e a leitura do codigo, e a assimetria.** Se a
discordancia viesse de semente ou de particao, ela seria simetrica: a
mlquantify seria melhor em cerca de metade dos casos.

| par | dif. abs. media | mlquantify melhor | p-valor | leitura |
| --- | --- | --- | --- | --- |
| `HDy` vs qlib | 0,0434 | **15,2%** de 560 | 1,4e-29 | enviesado: estimador diferente |
| `MMD` vs qunfold | 0,0428 | **51,2%** de 508 | **0,133** | simetrico: metodo diferente |
| `MS` vs quapy | 0,0384 | 56,5% | **0,419** | magnitude sem sistematicidade |
| `MS2` vs quapy | 0,0203 | 68,3% | 5,9e-08 | em aberto |
| `PWK` vs qlib | 0,0355 | 42,9% | 6,6e-06 | em aberto |

**E a semente esta descartada por um controle, nao por argumento.** `CC`,
`PCC` e `MLPE` concordam em **100%** dos casos ate 1e-3, com p = 1,0. Os
tres leem o classificador direto -- o `CC` conta predicoes duras, o `PCC`
media posteriores. Se o classificador diferisse entre as bibliotecas por
semente, por dobra de validacao cruzada ou por ajuste, eles nao poderiam
ser identicos em 568 casos. Somado a padronizacao que o `registry.py` ja
documenta (`cv=5`, estratificado, embaralhado, `random_state=42` passados
a cada biblioteca no nome dela), e aos 12 controles que reproduziram o
erro em `0.000e+00`, nao sobra espaco para atribuir a discordancia a
aleatoriedade.

**Q1, pelas tabelas da `final-separado`:**

| | `final-corrigido` | `final-separado` |
| --- | --- | --- |
| quantificationlib | 7 688 casos, 0,0090, 85,4%, corr 0,9628 | 7 128 casos, **0,0063**, **89,7%**, corr **0,9767** |
| quapy | 12 140 casos, 0,0091, 82,8%, corr 0,9665 | **identico em tudo** |
| qunfold | 4 352 casos, 0,0097, 81,8%, corr 0,9702 | 3 844 casos, **0,0053**, **88,5%**, corr **0,9863** |
| testes em AE | 13 de 28 (46,4%) | 12 de 27 (44,4%) |

A quapy identica nos tres digitos e o controle da operacao: confirma que a
renomeacao alcancou so o que devia. Nos testes sai apenas o `HDy`; o `KMM`
ja nao era significativo, entao ele deixa o denominador e nao o numerador.
Os postos do grupo mlquantify/qlib/quapy vao de 2,08 / 1,79 / 2,12 para
2,03 / 1,91 / 2,06, porque o `HDy` saia justamente do grupo em que a
mlquantify estava pior.

A lista dos piores desacordos fica: `MS` contra quapy (0,0384), `PWK`
contra quantificationlib (0,0355), `MS2` contra quapy (0,0203), `ACC`
contra quapy (0,0190) e `DyS` contra quapy (0,0180). Os tres primeiros sao
os registrados sem nota de diferenca **e** acusados pelo teste: sao os
candidatos com a forma do `HDx`.

**Q2 passa a ter uma medida de tempo que lidera, e ela inverte a
manchete.** O custo de usar a biblioteca e um treinamento mais a estimacao
das 400 amostras, somados **antes** de parear -- a razao de uma soma nao e a
soma das razoes. Essa medida entrou no `CUSTOS` da `analysis.py` na frente
das duas fases, que ficaram: separadas elas carregam um resultado que o
total esconde.

| | treinamento | estimacao | treino + estimacao |
| --- | --- | --- | --- |
| quantificationlib | 0,87x (77%) | 1,41x (36%) | **1,12x (40%)** |
| quapy | 0,95x (64%) | 1,03x (46%) | **0,96x (56%)** |
| qunfold | 1,08x (33%) | 0,02x (100%) | **0,02x (100%)** |

Entre parenteses, a fracao dos casos em que a mlquantify saiu mais barata.
De ponta a ponta ela e **mais cara** que a quantificationlib, 1,12x e mais
barata em so 40% dos casos, enquanto o treinamento isolado dava 0,87x em
77%. A estimacao domina a soma, porque o treinamento e um unico ajuste e a
estimacao sao 400 amostras. Reportar so o tempo de treinamento favorecia a
proposta sem dizer isso.

E a razao de 0,02x contra a qunfold mostra por que as fases nao foram
colapsadas: num tempo unico ela aparece, mas o 1,08x do treinamento contra a
mesma biblioteca desaparece, e e o unico lugar em que a mlquantify perde no
treino.

**Q2 anda contra a proposta tambem na comparacao entre as duas execucoes, e
isso e evidencia a favor da separacao.**
Contra a qunfold o tempo de treinamento vai de 1,05x para **1,08x**, e a
mlquantify passa de mais barata em 41% para 33% dos casos; a memoria de
treino vai de 0,99x para 1,00x. O `KMM` nao usa classificador e era um
caso barato que saiu do pareamento. Contra a quantificationlib o
treinamento vai de 0,88x para 0,87x. A quapy nao muda. **Q3** nao muda em
nada: a renomeacao nao altera conclusao nem configuracao manual.

**A ressalva que precisa estar aqui, e nao em nota de rodape.** Os numeros
de Q1 melhoram para a proposta -- 30% contra a quantificationlib e 45%
contra a qunfold. Quem ler depois vai cobrar a justificativa, e com razao.
O que a sustenta e a ordem dos fatos: a regra estava escrita e aplicada
contra a qunfold antes destes dois casos; a nota do `HDy` da qlib
descrevia a variante de intervalo fixo antes de qualquer medicao; e a
assimetria distingue os dois casos sem depender de nenhuma escolha. Nada
disso foi decidido depois de ver o resultado. Mas a Q2 andando para o
outro lado e o que mostra que o corte nao foi escolhido pelo efeito.

A `final-separado` esta em `results/final-separado/`, com as tabelas em
`analise/`, gerada pelos proprios `aggregate_results.py` e
`analyze_results.py`. Partiu das parciais da `final-corrigido`, que ja
contem a substituicao das 202 combinacoes, com o nome trocado em 565
linhas de `HDy` da qlib, 520 de `MMD` da qunfold e 281 entradas de ledger.
A `final-corrigido` e a `final-alinhado` nao foram alteradas.

> **A separacao do `HDy` foi desfeita depois**, na `final-ajustado`, por
> decisao de quem conduz o experimento: o `HDy` da quantificationlib volta a
> ser par do `HDy` da mlquantify. A separacao do `MMD`/`KMM` da qunfold
> continua valendo. O que esta escrito aqui sobre o intervalo fixo segue
> valendo como descricao do metodo -- o que mudou foi a decisao de compara-lo
> mesmo assim, e o custo disso em Q1 esta medido adiante.

#### A PR #6: o custo do `MLPE` medido

Medida na execucao `sens_pr6`: `MLPE` nos 142 conjuntos mais 12 de
controle, 154 combinacoes, todas `ok`, 0,80 h de CPU em 30 min. A
correcao foi aplicada sobre a `v0.5.1` numa venv propria (`.venv-pr6`),
**sem** as PRs #9 e #10, para que o efeito fosse atribuivel a ela so. Os
dois simbolos que a PR importa (`_get_estimator` e
`_get_estimator_function`) existem na 0.5.1, e o patch aplica limpo.

**O mecanismo, contado antes de medir.** Com um classificador
instrumentado, a 0.5.1 limpa faz **6 treinamentos e 5 `predict_proba`**
por `fit` -- cinco dobras mais o reajuste, o que confirma a contagem que
esta na Secao "O custo do MLPE". Com a PR #6 faz **1 e 1**. A estimativa
nao muda (`0.000e+00` de diferenca), segue igual a prevalencia de treino,
e `estimator_` continua existindo depois do `fit`.

Nas 560 linhas pareadas, o erro e **identico** em MAE, MRAE, MSE e NKLD
(`0.000e+00`), como tem de ser: a correcao mexe no custo de chegar ao
numero, nao no numero.

**Uma correcao de sessao que esta medida precisou.** Os 12 controles
mostraram que esta sessao roda **1,26x mais rapido** que a da
`final-alinhado`: a razao de tempo nos controles e 0,793 (96 razoes),
nao 1,000 como fora na `sens_corrigido`. Parte do ganho bruto e a maquina.
As razoes abaixo estao divididas por esse fator; a de memoria nao, porque
memoria nao depende da velocidade da sessao.

| Medida (mediana das razoes) | bruta | calibrada | ganho |
| --- | --- | --- | --- |
| tempo de treinamento | 0,230 | **0,291** | 3,4x |
| tempo de estimacao | 1,011 | -- | inalterado |
| treino + estimacao | 0,377 | **0,476** | 2,1x |
| memoria no treinamento | 0,311 | 0,311 | 3,2x |

O tempo de estimacao nao muda, o que era esperado: o `predict` devolve
`train_priors_` e nunca foi o problema.

**Contra a quapy**, em mediana das razoes caso a caso, calibrado: treino
mais estimacao cai de **27,4x para 9,9x** e a memoria de treino de 24,0x
para 6,9x.

**E aqui uma ressalva sobre o numero mais citado desta secao.** O `1 376x`
de tempo de treinamento e fragil, porque o denominador -- o treino do
`MLPE` da QuaPy -- fica em 0,0001 a 0,0002 s, na resolucao do relogio.
Dependendo do estimador, a mesma execucao de referencia da 1 376x (razao
das medianas, como a tabela original), 2 294x (razao das medianas no
subconjunto de 139 conjuntos) ou 1 307x (mediana das razoes). As tres
descrevem o mesmo dado. O numero robusto e o composto: **treino mais
estimacao, 27,4x na referencia e 9,9x com a correcao**. O texto deve
liderar com ele e tratar o `1 376x` como ilustracao, nao como medida.

Com isso, a conclusao qualitativa da secao sobrevive mas enfraquece: a
mlquantify continua mais caro que a QuaPy no metodo mais simples que
existe, e por uma ordem de grandeza, nao por tres.

**Por que a PR #6 nao entrou na `final-corrigido`.** As medidas de custo
desta sessao estao 1,26x deslocadas, e injetar linhas assim na Q2
enviesaria o agregado -- exatamente o que a montagem da `final-corrigido`
evitou ao nao substituir linhas cujo erro nao muda. O efeito da PR #6
fica registrado aqui, como analise de sensibilidade. A `final-corrigido` e
a `final-separado` seguem com o `MLPE` medido na 0.5.1; a `final-mlpe`,
adiante, adota a correcao.

#### A execucao `final-mlpe`: o `MLPE` corrigido nos numeros oficiais

A ressalva acima foi apresentada e dispensada: as linhas de custo do `MLPE`
passam a vir do relogio desta sessao, e nao da sessao em que o resto foi
medido. Fica registrado que e isso, e so isso, que a calibracao media.

Medido na `.venv-pr6` -- 0.5.1 do PyPI com a PR #6 aplicada nos arquivos
instalados, sem a #9 e a #10, que nao tocam o `MLPE`. 142 de 142
combinacoes, **todas `ok`**, 0,66 h de CPU em 26 minutos. A substituicao
trocou **apenas** as linhas de `mlquantify` + `MLPE`, em 142 parciais e 142
entradas de ledger; todo o resto veio da `final-separado`.

| | 0.5.1 | com a PR #6 | quapy | razao antes | razao depois |
| --- | --- | --- | --- | --- | --- |
| tempo de treinamento | 0,2690 s | **0,0571 s** | 0,0001 s | 1 307x | **347x** |
| tempo de estimacao | 0,0091 s | 0,0102 s | 0,0066 s | 1,35x | 1,32x |
| memoria no treinamento | 0,8873 MB | **0,1853 MB** | 0,0310 MB | 24,0x | **6,9x** |
| memoria na estimacao | 0,2299 MB | 0,1935 MB | 0,0702 MB | 2,57x | 1,90x |

Treinamento 4,7x mais rapido e memoria de treino 4,8x menor. No tempo
combinado, que e por onde a Q2 julga o custo, a razao contra a quapy cai de
**27,4x para 8,51x**. A estimacao nao muda, como tem de ser: o `predict`
devolve `train_priors_` e nunca foi o problema.

**E houve ganho de cobertura, que nao estava previsto.** Na
`final-separado` o `MLPE` tinha 560 linhas `ok` e **2 `timeout`**: o
`fashion-mnist.csv` e o `Dermatology.csv` nao completavam dentro do limite
de 1800 s por metodo, justamente porque o caminho antigo ajustava o
estimador seis vezes. Com a correcao os dois completam, e o `MLPE` fica com
568 linhas, todas `ok`. Consequencias:

* Q1 contra a quapy vai de 12 140 para **12 148** casos pareados, e o erro
  segue identico ao da quapy em todo caso (diferenca `0,000000`) -- a media
  de AE anda de 0,2609 para 0,2593 porque o conjunto de casos cresceu, nao
  porque alguma estimativa mudou;
* a taxa de conclusao da mlquantify vai de 0,9915 para **0,9920**;
* os `timeout` do ledger caem de 121 para 119.

As razoes agregadas de Q2 praticamente nao se movem -- o `MLPE` e um metodo
entre os 22 pareados com a quapy, e as medianas sao sobre milhares de casos.
O que muda e o metodo, nao o agregado.

A `final-mlpe` esta em `results/final-mlpe/`. A `final-separado`, a
`final-corrigido` e a `final-alinhado` nao foram alteradas.

#### A execucao `final-mlpe-na`: o `MLPE` deixa de ser agregativo

A tabela acima deixou uma pergunta em aberto: corrigido, o `MLPE` ainda
custava **347x** o da quapy no treinamento. A resposta e que sobrava
exatamente **um ajuste de classificador**, e a prova e o `CC`, que treina um
classificador e mais nada:

| metodo | mlquantify | quapy | razao |
| --- | --- | --- | --- |
| `CC` | 0,0523 s | 0,0549 s | 1,0x |
| `MLPE` (com a PR #6) | 0,0571 s | 0,0001 s | 347x |

O `MLPE` custava o que o `CC` custa. E o denominador era ~zero porque **a
quapy nao treina classificador nenhum**: o
`MaximumLikelihoodPrevalenceEstimation` vive em `non_aggregative.py`, nao
recebe estimador, e o `fit` tem duas linhas que contam rotulos. Os 347x nao
eram lentidao, eram duas bibliotecas implementando o mesmo metodo como
objetos de natureza diferente.

**A mudanca** (`coenlab/mlquantify#14`) faz o `MLPE` da mlquantify ser do
mesmo tipo: a classe perde o `SoftPredictionMixin` e o `AggregativeMixin`,
passa `representation=None` e `aggregative=False`, e saem da assinatura o
`estimator`, o `cv`, o `stratified` e o `shuffle`. Com `aggregative=False` o
`_uses_estimator_predictions()` fica `False`, e o `fit` toma o caminho que le
o `train_priors_` dos rotulos. Nada e treinado, e nao ha mais `estimator_`.

Os parametros que sobraram sao exclusivos de palavra-chave de proposito: com
posicionais, um `MLPE(LogisticRegression())` escrito para a assinatura antiga
ligaria o classificador ao `solver` e rodaria malconfigurado em silencio.
Quem achou isso foi o proprio teste, que falhou com `DID NOT RAISE`.

No harness bastou `needs_classifier=False`: o `_mlq` passa a construir
`cls(**prot)`, e o `_protocolo` filtra pela assinatura, entao `cv`,
`stratified` e `shuffle` simplesmente deixam de ser escritos.

**Medido.** 142 de 142 combinacoes, todas `ok`, em **0,014 h de CPU** --
50 segundos, contra 0,66 h da versao com a PR #6. Tres minutos de relogio
contra vinte e seis.

| | com a PR #6 | nao agregativo | quapy | razao antes | razao agora |
| --- | --- | --- | --- | --- | --- |
| tempo de treinamento | 0,0571 s | **0,0006 s** | 0,0001 s | 347x | **4,99x** |
| tempo de estimacao | 0,0102 s | 0,0092 s | 0,0066 s | 1,32x | 1,30x |
| memoria no treinamento | 0,1853 MB | **0,0691 MB** | 0,0310 MB | 6,90x | **2,04x** |
| memoria na estimacao | 0,1935 MB | 0,0705 MB | 0,0702 MB | 1,90x | **1,00x** |

No tempo combinado a razao contra a quapy vai de 8,51x para **1,38x** -- e,
contando desde a 0.5.1, de 27,4x para 1,38x. A memoria de estimacao fica
**igual** a da quapy. O que sobra de 4,99x no treinamento sao 0,5 ms: a
mlquantify ainda passa pelo `validate_data` e pela canalizacao do `compose`,
e a quapy nao.

**O erro nao muda, e isso foi conferido e nao suposto.** `0,25928454` nas
duas execucoes, contra os mesmos `0,25928454` da quapy, com diferenca de
`7,8e-19`. No `q1_testes` o que se move sao os postos medios das medidas de
**custo**, na terceira casa; nenhuma linha de erro muda.

A `final-mlpe-na` esta em `results/final-mlpe-na/`. As execucoes anteriores
nao foram alteradas.

#### A execucao `final-pares`: o apelido que contava duplo

Montar a aba "O que e comparado" do painel exigiu dizer, por extenso, qual
metodo de cada biblioteca entra em Q1 contra qual metodo da mlquantify. Foi
assim que apareceu uma duplicacao.

O `expand_aliases` acrescenta um apelido **apenas quando a biblioteca nao tem
metodo proprio com aquele nome**. O apelido `PACC`/`GPACC` estava declarado
dos dois lados -- a mlquantify dizia `GPACC -> PACC`, a quapy e a
quantificationlib diziam `PACC -> GPACC`. Resultado: a mlquantify ganhava
`PACC`, as outras ganhavam `GPACC`, e o **mesmo par se materializava sob os
dois nomes**. Conferido por hash das medicoes: as 564 medicoes aparecem
identicas nas duas linhas, com MAE de 0,083856 contra 0,090430 em ambas.

| | com a duplicacao | sem ela |
| --- | --- | --- |
| quantificationlib | 7 128 casos (13 metodos) | **6 564 (12)** |
| quapy | 12 148 (22) | **11 584 (21)** |
| qunfold | 3 844 (7) | 3 844 (7), intacta |

Eram **1 128 casos repetidos**: 7,9% do total contra a quantificationlib e
4,6% contra a quapy. A qunfold intacta e o controle da operacao.

**E o conserto anda contra a proposta**, o que e o que mostra que ele nao foi
escolhido pelo efeito: a diferenca absoluta media contra a quapy piora de
0,0091 para **0,0094**, a concordancia cai de 55,4% para 55,1% ate 1e-3 e de
82,8% para 82,4% ate 1e-2, e a correlacao de 0,9665 para 0,9657. O par
duplicado concordava melhor que a media, e o peso em dobro favorecia a
proposta.

**Qual lado tirar nao era indiferente, e isso so apareceu medindo.** Tirando
o apelido da quapy e da quantificationlib a duplicacao tambem acabava, mas o
bloco de teste do `PACC` passava a comparar **quantificationlib contra
quapy**, sem a mlquantify -- uma comparacao que nao e a pergunta de Q1 e que
ainda assim entrava na contagem, levando os testes de 12 para 13 de 27
(44,4% para 48,1%). Tirando o apelido do lado da mlquantify, os 27 blocos
seguem todos ancorados na proposta, o `PACC` compara mlquantify,
quantificationlib e quapy, o `GPACC` compara mlquantify e qunfold, e a
contagem fica em **12 de 27**, como era. Foi esse o lado escolhido.

**O `ACC -> GACC` das outras duas continua, e nao duplica.** A mlquantify tem
`ACC` e `GACC` como metodos distintos, entao os apelidos produzem duas
comparacoes diferentes contra o mesmo metodo da outra biblioteca, e nao duas
copias da mesma. Conferido: contra a mesma medicao da quantificationlib
(0,125630), a mlquantify da 0,130248 pelo `ACC` e 0,116548 pelo `GACC`.

> **Esse apelido foi retirado depois**, na `final-ajustado`: o `AC` da
> quantificationlib e o `ACC` da quapy sao a formulacao generalizada e passam
> a ser comparados so com o `GACC`. Nao era duplicacao, como esta dito aqui;
> eram duas comparacoes, e uma delas comparava formulacoes diferentes.

Nenhuma medicao mudou nesta execucao -- so parou de ser contada duas vezes.
A `final-pares` esta em `results/final-pares/`, e as anteriores nao foram
alteradas.

#### A execucao `final-ajustado`: tres pares revistos

Tres mudancas de pareamento, pedidas por quem conduz o experimento, todas
sobre quem e comparado com quem -- nenhuma medicao foi refeita.

| biblioteca | metodo | era par de | passa a ser par de |
| --- | --- | --- | --- |
| quantificationlib | `baselines.ac.AC` | `ACC` **e** `GACC` | so `GACC` |
| quapy | `aggregative.ACC` | `ACC` **e** `GACC` | so `GACC` |
| quantificationlib | `multiclass.df.HDy` | nada (`HDyFix`) | `HDy` |

**As duas primeiras.** O `AC` da quantificationlib e o `ACC` da quapy
corrigem a contagem com a matriz de confusao estimada sobre o treino, para
qualquer numero de classes: e a formulacao generalizada, que na mlquantify se
chama `GACC`. O `ACC` da mlquantify e a versao binaria. Comparar a mesma
medicao das outras com os dois de uma vez media um estimador contra dois
estimadores diferentes.

**A terceira anda contra a proposta**, e por isso vale dizer de onde veio: e
o inverso da separacao registrada na `final-separado`. A nota do registro
sobre o intervalo fixo continua no `registry.py`; o que mudou foi a decisao
de comparar os dois mesmo assim.

**O pareamento vem dos nomes gravados nas parciais, nao dos canonicos do
registry.** O `registry.py` so decide os apelidos; quem casa um metodo da
outra biblioteca com um da mlquantify e a igualdade de nome no `summary.csv`.
Trocar o canonico no registro, sozinho, nao reaponta nada -- fez o oposto do
pedido na primeira tentativa, tirando o `GACC` em vez de renomear o `ACC` e
deixando o `HDyFix` orfao. A mudanca foi feita renomeando a medida: 562
linhas de `ACC` da qlib, 565 de `HDyFix` da qlib, 565 de `ACC` da quapy e 426
entradas de ledger. O ledger ficou identico no resto (9 709 `ok`, 76 `erro`,
119 `timeout`) e a cobertura tambem (9 904 de 9 940).

**Q1, pelas tabelas das duas execucoes:**

| | `final-pares` | `final-ajustado` |
| --- | --- | --- |
| qlib: casos (metodos) | 6 564 (12) | 6 564 (12) |
| qlib: dif. abs. media | 0,0063 | **0,0094** |
| qlib: iguais ate 1e-3 | 81,6% | **75,3%** |
| qlib: iguais ate 1e-2 | 89,5% | **84,7%** |
| qlib: correlacao | 0,9776 | **0,9599** |
| qlib: dif. maxima | 0,3155 | **0,4537** |
| quapy: casos (metodos) | 11 584 (21) | **11 024 (20)** |
| quapy: dif. abs. media | 0,0094 | **0,0089** |
| quapy: iguais ate 1e-3 | 55,1% | **55,7%** |
| quapy: correlacao | 0,9657 | **0,9685** |
| qunfold | 3 844 (7) | 3 844 (7), intacta |
| testes com diferenca | 12 de 27 | 12 de 26 |

A contagem da quantificationlib nao muda porque as duas trocas se cancelam:
sai o par `ACC` (560 casos) e entra o par `HDy` (560). **O que muda e a
qualidade**, e muda muito, porque os dois pares sao os extremos opostos da
tabela:

| par que sai ou entra | casos | dif. abs. media | iguais ate 1e-3 | mlquantify melhor |
| --- | --- | --- | --- | --- |
| `ACC` vs qlib (sai) | 560 | 0,0076 | 78,0% | -- |
| `ACC` vs quapy (sai) | 560 | **0,0190** | 42,5% | -- |
| `HDy` vs qlib (entra) | 560 | **0,0434** | **4,6%** | **15,2%** |
| `GACC` vs qlib (fica) | 560 | 0,0100 | 82,5% | 50,2% |
| `GACC` vs quapy (fica) | 564 | 0,0056 | 51,8% | 46,1% |

O `HDy` vs qlib e o pior par do experimento inteiro, e o lado melhor e o da
quantificationlib: 0,0907 contra 0,1178 de MAE. Concorda em 4,6% dos casos
ate 1e-3 e em 31,1% ate 1e-2, e a mlquantify so ganha em 15,2% deles --
assimetria que, pela leitura da `final-separado`, e o que distingue
estimador diferente de ruido de semente. **Ele sozinho responde por quase
toda a piora da quantificationlib nesta execucao.**

Contra a quapy o efeito e o oposto: o par que sai era o de pior concordancia
da biblioteca (0,0190 de diferenca media, 42,5% ate 1e-3), e tirar um
estimador binario comparado com um generalizado melhora os numeros da
proposta. **As tres mudancas juntas, entao, andam para os dois lados** --
contra a proposta na quantificationlib, a favor na quapy --, o que e o que
mostra que nao foram escolhidas pelo efeito.

**Uma perda a registrar: o `ACC` da mlquantify deixa de ser comparado com
qualquer coisa.** Era o unico metodo da proposta cujos dois pares eram o
`ACC` da qlib e o da quapy; sem eles, nenhuma das tres bibliotecas tem
metodo com esse nome, e ele sai de Q1 por completo. Os testes caem de 27
para 26 blocos por isso. A versao generalizada, `GACC`, segue comparada com
as tres.

A `final-ajustado` esta em `results/final-ajustado/`, e as anteriores nao
foram alteradas.

#### A execucao `final-emq`: EM contra otimizacao numerica

A qunfold tem **dois** metodos de verossimilhanca e a propria biblioteca os
distingue pela referencia: o `ExpectationMaximizer` e Saerens et al. (2002),
o mesmo algoritmo EM do `EMQ` das outras tres; o `LikelihoodMaximizer` e
Alexandari et al. (2020), que minimiza a log-verossimilhanca negativa
regularizada com `trust-ncg` e Hessiana por autodiferenciacao. O registro
pareava o `EMQ` com o **segundo**.

Consertado, o par passa a ser o `ExpectationMaximizer`, e o
`LikelihoodMaximizer` ganha nome canonico proprio, `MLQ`, no grupo `core` --
e um metodo que o usuario da qunfold recebe e conta para Q3, so nao tem par,
porque a mlquantify nao expoe um maximizador de verossimilhanca nomeado.

| | par antigo (`LikelihoodMaximizer`) | par certo (`ExpectationMaximizer`) |
| --- | --- | --- |
| diferenca absoluta media | 0,005949 | **0,000127** |
| iguais ate 1e-3 | 80,3% | **99,6%** |
| estimacao, mediana | 102,6 s | **5,19 s** |

A diferenca cai **47 vezes**. E o custo: os 357x que a leitura anterior
atribuiu a arquitetura da qunfold eram artefato do par errado -- contra os
0,3397 s da mlquantify o numero real e **15,3x**, e o que sobra e o JAX
rodando um laco EM sobre arranjos minusculos sem `jit`, nao um otimizador.

As 568 linhas gravadas como `qunfold | EMQ` foram medidas com o
`LikelihoodMaximizer`; a execucao derivada as renomeia para `MLQ` e recebe o
`EMQ` novo, medido em 142 de 142 sem falha. Nada foi descartado.

#### A execucao de sensibilidade `sens-hdy`: a varredura do `HDy`

Mediu o `HDy` da quantificationlib com a varredura de intervalos que a
definicao do metodo pede (10 a 110, mediana das 11 estimativas) em vez do
`n_bins=8` fixo, para separar parametrizacao de implementacao. **Nao serve**,
e o motivo e o achado:

* 131 `ok`, 7 `timeout`, 4 `erro`, contra 141 `ok` e 1 `timeout` do `HDy` de
  intervalo fixo nos mesmos 142 conjuntos. A varredura **quebra 10 conjuntos
  que a versao fixa resolve**, e nao ao acaso: sao os de muitas classes.
* O `erro` e da biblioteca. Em `optimization.py:132` a qlib faz
  `list(problem.solution.primal_vars.values())[0]` sem conferir o status do
  cvxpy: quando o `solve` nao acha solucao, `primal_vars` vem vazio e o
  acesso estoura. Dispara com amostra de 10 itens e 110 intervalos sobre 10
  classes, onde o histograma de teste e quase todo zero.

Por isso o alinhamento foi feito no sentido inverso -- o `HDy8`, com os 8
intervalos da qlib dos dois lados --, e o `HDy` varrido segue como par da
quapy, que fixa a mesma varredura no codigo e nao aceita parametro.

**Um defeito do proprio arcabouco apareceu aqui.** O `Mfeat` teve **1 falha
em 400 amostras**, abaixo do limite de 5%, e mesmo assim a combinacao inteira
foi perdida com a mensagem crua da biblioteca: o `_measure_memory` repetia as
400 estimativas **sem protecao por amostra**, enquanto o `_predict_all`
tolera ate `MAX_INVALID_FRACTION`. Consertado -- as estimativas da medicao de
memoria agora sao protegidas uma a uma e, passado o limite, devolve-se `nan`
no custo em vez de derrubar a combinacao. Conferido que isso nao corrompeu os
numeros oficiais: dos 52 erros de mensagem crua da `final-emq`, todos tem
causa genuina (asseroes da propria quapy no `SMM`, alocacoes de 173 GiB).

#### A execucao de sensibilidade `sens-limiar`: a grade de limiares

A `evaluate_thresholds` da mlquantify tem dois modos. O padrao,
`score_edges="fixed"`, varre `np.linspace(0, 1, 101)`; o outro varre
`np.unique(probabilities)`, que e o que a quapy faz e o que Forman define. O
modo **nao esta exposto no quantificador**: o `_adjust` chama a funcao sem o
argumento. Esta execucao mede os cinco metodos de limiar com a varredura
exata. Resultado: 684 `ok` e 12 `timeout` de 710 combinacoes,
com os 142 conjuntos cobertos.

| metodo | grade de 101 | varredura exata | fator |
| --- | --- | --- | --- |
| `MS` | 0,0386 (6,6% ate 1e-3) | **0,0046 (51,8%)** | **0,12x** |
| `T50` | 0,0173 (20,1%) | **0,0098 (37,6%)** | 0,56x |
| `MS2` | 0,0203 (13,5%) | 0,0168 (53,7%) | 0,83x |
| `X` | 0,0080 (40,9%) | 0,0066 (52,6%) | 0,82x |
| `MAX` | 0,0086 (33,6%) | 0,0088 (38,1%) | 1,03x |

O conserto fecha o vao **na proporcao em que cada criterio depende do
conjunto de candidatos**, que e o que torna o argumento causal e nao
correlacional: o `MS` toma a mediana sobre **todos** os limiares -- o
conjunto *e* a resposta -- e cai 8 vezes, encostando no piso de ruido do par
(~0,006); o `T50` precisa acertar tpr = 0,5 e cai pela metade; o `MAX`, que
otimiza um criterio de otimo largo, **nao se mexe**. A concordancia ate 1e-3
melhora nos cinco, inclusive onde a media nao muda: a grade introduz uma
quantizacao que afeta todos um pouco e o `MS` muito.

**E a grade nao e descuido, e escala.** A varredura exata e O(*n*) no tamanho
do treino contra O(1) da grade: no `Dermatology`, com 500 mil linhas de
treino, sao 500 mil limiares candidatos em vez de 101. Os 12 `timeout` caem
todos nos maiores conjuntos, com o limite de 1 800 s por metodo.

**A grade continua sendo o numero oficial**, por decisao de quem conduz o
estudo, e o motivo e a premissa do experimento: ele mede as bibliotecas como
o usuario as recebe. O `score_edges` nao esta exposto no quantificador, entao
um usuario da mlquantify nao consegue pedir a varredura exata; adotar o
numero dela seria reportar um valor que ninguem reproduz com a biblioteca
publicada. Esta execucao serve para dizer, com medida, **quanto da
discordancia e parametrizacao e nao implementacao**, que e um resultado mais
forte do que um numero melhor.

Para registro, o que a adocao daria: a concordancia da quapy ate 0,01 subiria
de 83,2% para 88,3% e a diferenca media cairia de 0,0087 para 0,0064, com o
`MS` indo de 0,0384 (38,0% dos casos) para 0,0049 (87,9%).

> **Correcao.** Numa leitura anterior desta execucao eu afirmei que a forma
> exata "simplesmente nao roda nessa escala". Estava errado, e o erro foi de
> metodo: eu havia **interrompido** a execucao em 665 de 710 combinacoes,
> com os maiores conjuntos ainda na fila, e conclui da interrupcao em vez de
> da observacao. Retomada, ela completou 684 de 710 e cobriu os 142
> conjuntos. A varredura exata e muito mais cara, e isso e real; mas ela
> termina em 96% das combinacoes.


#### A execucao `final-pareado`: os numeros oficiais com os pares certos

Junta tudo o que a auditoria de pares produziu. Tres renomeacoes do lado da
quantificationlib, porque o pareamento vem dos nomes gravados nas parciais e
nao dos canonicos do registro, e quatro substituicoes de medida:

| renomeado | por que |
| --- | --- |
| `GACC` -> `TAC` | o `AC` dela e a forma fechada de Forman no limiar padrao, que e o `TAC`; o `GACC` resolve regressao restrita sempre |
| `PWK` -> `PWKAC` | e AC sobre o classificador PWK, nao contagem pura |
| `HDy` -> `HDy8` | 8 intervalos fixos, para parear com o `HDy8` da mlquantify |

| substituido | de onde | por que |
| --- | --- | --- |
| mlquantify `PWKAC`, `HDy8` | `pares-novos` | os pares novos, medidos |
| mlquantify `EMQ` | `emq-alinhado` | `max_iter` de 100 para 1000 |
| qunfold `EMQ` | `emq-alinhado` | 1e-6/100 para 1e-4/1000 |

**O efeito em Q1**, contra a `final-emq`:

| | antes | depois |
| --- | --- | --- |
| qlib: diferenca absoluta media | 0,0094 | **0,0047** |
| qlib: iguais ate 1e-3 | 75,3% | **82,3%** |
| qlib: iguais ate 1e-2 | 84,7% | **91,4%** |
| qlib: correlacao | 0,9599 | **0,9834** |
| quapy: diferenca absoluta media | 0,0089 | 0,0087 |
| qunfold: diferenca absoluta media | 0,0045 | 0,0045 |

A quantificationlib muda de patamar: a discordancia cai pela metade e a
correlacao vai de 0,96 para 0,98. A quapy mal se move, o que e esperado --
so o `EMQ` dela mudou de par. A qunfold fica igual.

**E a leitura e outra.** Antes, a quantificationlib parecia a biblioteca que
mais discorda da proposta. Com os pares certos ela e a que **mais concorda**
em concordancia pratica (91,4% dos casos ate 0,01, contra 83,2% da quapy).
O que estava sendo medido como discordancia de implementacao era, em tres
metodos, discordancia de identidade de metodo.

**Dois metodos da mlquantify ficam sem par, e isso esta correto:** o `ACC`
binario, porque nenhuma das outras tres expoe um metodo com esse nome depois
das correcoes -- mas o gemeo dele, o `TAC`, esta pareado, e os dois sao bit a
bit identicos nos 564 casos; e o `PWK` sem ajuste, porque o da
quantificationlib e ajustado. Do outro lado, o `MLQ` da qunfold tambem fica
sem par.

**Uma pendencia registrada.** A tolerancia do `EMQ` da qunfold deveria voltar
para 1e-6, pelo motivo explicado na secao anterior -- igualar o numero nao
iguala o criterio, porque ela compara iteracoes pela norma L2 e as outras por
erro absoluto medio. A remedicao nao pode ser feita nesta maquina enquanto o
Smart App Control bloquear o `jaxlib`; conferido que nem reinstalar nem
instalar fora do OneDrive resolve, ao contrario do `pandas`, que voltou com
um `--force-reinstall` da mesma versao. Os dois valores em disputa, 0,000127
e 0,000422, estao uma ordem de grandeza abaixo do limiar de 1e-3 da Q1.

A `final-pareado` esta em `results/final-pareado/`, e as anteriores nao foram
alteradas.

### O que seria preciso para os erros baterem

Auditoria dos 39 pares da `final-emq`, em resposta a pergunta direta de
quanto falta para a correspondencia ser exata.

**Batem a 1e-6 apenas tres:** `CC`, `PCC` e `MLPE` -- e sao exatamente os
tres que **nao consomem escores fora da amostra**. Os dois primeiros contam
com o classificador treinado na particao inteira; o terceiro ignora o teste.

> **Passaram a ser quatro.** Depois que o `max_iter` do `EMQ` da mlquantify
> foi alinhado em 1000, o `EMQ` entrou nessa lista: 0,0000 de diferenca e
> 100% dos casos ate 1e-3 contra a quapy e contra a quantificationlib. Ele
> tambem nao usa validacao cruzada -- a classe nem aceita o parametro --, de
> modo que a regra se mantem: batem exatamente os metodos que nao dependem
> de escores fora da amostra, e o que impedia o `EMQ` era so o criterio de
> parada.

**O piso de todo o resto sao as dobras da validacao cruzada**, e ele nao e
removivel por parametro: a quapy nao expoe o embaralhamento (`val_split`
inteiro cai em `StratifiedKFold(shuffle=False)`) enquanto a mlquantify
embaralha com `random_state`. Dobras diferentes dao escores diferentes, e
dai em diante tudo difere.

**Provado entregando os mesmos escores as quatro.** Ajustando o classificador
uma vez e passando os mesmos `predict_proba` pelos pontos de entrada que
aceitam predicoes prontas -- `aggregate` na mlquantify, `aggregation_fit` e
`aggregate` na quapy, `predictions_train=` e `predictions_test=` na qlib,
classificador fixo na qunfold --, o `EMQ` da:

| conjunto | quapy | quantificationlib | qunfold |
| --- | --- | --- | --- |
| banana | **1,1e-16** | **1,1e-16** | 5,39e-03 |
| german | **0** | **0** | 5,84e-05 |
| breast-cancer | 5,9e-10 | 5,9e-10 | 6,70e-06 |

Tres das quatro ficam **bit a bit identicas**. A discordancia de 0,0046 dos
numeros oficiais era inteiramente a particao das dobras.

**O residuo da qunfold e o criterio de parada, nao precisao.** A hipotese
inicial era o float32 do JAX; ligando `jax_enable_x64` o residuo nao se move
(5,390e-03 para 5,388e-03). A causa e a norma: a qunfold compara iteracoes
pela **norma L2**, a qlib por erro absoluto medio, e a mlquantify expoe isso
num parametro `criteria` (padrao MAE) -- num vetor binario e um fator raiz de
2. Com `tol=1e-10` o residuo cai de 5,39e-03 para **7,47e-06**: todas
convergem ao mesmo ponto fixo e so param em lugares diferentes da
aproximacao. O tamanho do residuo acompanha a lentidao da convergencia.

**Parametros a anotar na mlquantify**, achados comparando os padroes dos
construtores par a par:

| o que | mlquantify | as outras | efeito |
| --- | --- | --- | --- |
| `EMQ`: `max_iter` | 100 | 1000 (quapy e qlib) | **truncava o EM**: no banana a estimativa muda 0,144 entre 100 e 1000 |
| `EMQ`: `tol` | 1e-4 | qunfold 1e-6 | para mais cedo que a qunfold |
| `on_calib_error` | `'backup'` | quapy `'raise'` | cai em reserva **em silencio** onde a quapy levanta erro |
| `score_edges` | `"fixed"`, nao exposto | quapy varre os escores | corrige o `MS` em 8x |

E uma armadilha de nomenclatura: **`bin_strategy` significa coisas diferentes
nas duas bibliotecas** -- na mlquantify e como agregar a varredura
(`median`/`mean`), na qlib e como calcular as bordas
(`equal_width`/`equal_count`). Quem comparar sem ler conclui errado.

**Conferidos e corretos:** a largura de banda do `KDEy` (0,1 nas duas), as
bordas do histograma (`bin_edges="fixed"` com `range=(0,1)` na mlquantify e
`equal_width` com `att_range=[0,1]` na qlib sao a mesma coisa), e os
intervalos de `DyS` e `HDx`, ja alinhados pelo registro.

**Implementacao, anotada e nao consertada**, por decisao de quem conduz o
experimento:

* a **qlib troca de metodo entre binario e multiclasse** no `AC` e no `PAC`.
  No binario e a forma fechada de Forman; na multiclasse ela **inverte a
  matriz de confusao transposta** (`linalg.inv(self.cm_.T)`), corta os
  negativos e renormaliza, caindo em minimizacao de distancia so no `except
  LinAlgError`. Medido: batem a 1e-6 em **93,8%** dos casos binarios e em
  **0,8%** dos multiclasse. E ate no binario ha uma troca escondida -- com
  `|cm_[1,1] - cm_[0,1]| <= 0,001` ela devolve a contagem sem ajuste, isto e,
  vira `CC`; a mlquantify so faz isso quando `tpr - fpr` e exatamente zero;
* o `HDy` e o `DyS` da **quapy degradam com o numero de classes** no
  um-contra-resto: no binario concordam no piso de ruido (0,0085 e 0,0060) e
  na multiclasse divergem 5,0x e 9,5x. Nao e a decomposicao nem a
  normalizacao -- as duas bibliotecas ajustam um quantificador binario por
  classe, e normalizar as estimativas cruas do mesmo jeito nao muda o vao --,
  e sim a estimativa binaria por classe: com 10 classes as cruas da quapy
  somam 0,53 contra 0,94 da mlquantify;
* o `solve_hd` da qlib nao confere o status do cvxpy antes de indexar.


### O icamento da ROC: o que rendeu, medido

A familia de limiar da mlquantify 0.5.1 recalculava a curva ROC **a cada
estimativa**: as taxas saem dos escores de validacao cruzada, que dependem
so do treino, mas eram refeitas dentro do `predict`. O icamento move esse
calculo para o `fit`, onde ele pertence. A pergunta desta secao e dupla:
quanto o custo caiu, e se o resultado continua o mesmo.

**Continua o mesmo.** Teste direto, mesmo `split` e mesmo classificador,
com o overlay chamado em `score_edges="fixed"` para isolar o icamento da
mudanca de semantica que o acompanhava: 18 combinacoes (`T50`, `X`, `MAX`,
`MS`, `MS2`, `TAC` por binario, 3 classes e 5 classes), 25 amostras cada,
**17 das 18 identicas bit a bit**.

A unica que difere e o `MS` em 5 classes, com `max |dif|` de 9,0e-3 numa
classe. A causa nao e o cache: a `MS._adjust` da 0.5.1 mantinha **todos** os
limiares e, nos degenerados, usava a contagem crua --
`np.where(denom == 0, cc, np.clip((cc - fprs) / safe, 0, 1))` --, enquanto o
overlay filtra `tprs != fprs` antes de escolher, de modo que a mediana sai
de um conjunto menor. E defensavel, e e o que a quapy faz com o seu
`discard`, mas **nao e identidade**, e so aparece quando a ROC tem limiar
degenerado, comum no um-contra-resto de classe rara. O `MS2` nao e afetado
porque o filtro proprio dele (`tprs - fprs > 0,25`) ja excluia esses
limiares.

**Custo.** Execucao `custo-final5` (overlay sobre a 0.5.1, so
`--libraries mlquantify`, 12 metodos, 133 conjuntos), pareada contra a
`final-pareado` por (conjunto, metodo, tamanho de amostra): 3 104 pares,
nenhum orfao. Razao do tempo de estimacao, bruta e corrigida pelo controle:

| metodo | estimacao bruta | corrigida | treino corrigido | memoria |
| --- | --- | --- | --- | --- |
| `T50` | 0,575 | **0,449** | 0,708 | 0,78 |
| `X` | 0,611 | **0,476** | 0,758 | 0,77 |
| `TAC` | 0,707 | **0,552** | 0,773 | 0,80 |
| `MAX` | 0,712 | **0,556** | 0,751 | 0,79 |
| `MS2` | 0,881 | 0,687 | 0,688 | **1,60** |
| `MS` | 0,931 | 0,726 | 0,843 | **1,67** |
| `CC` (controle) | 1,298 | 1,013 | 0,965 | -- |
| `PCC` (controle) | 1,266 | 0,988 | 1,035 | -- |

Leitura: a estimacao dos metodos de limiar cai a cerca de **metade**, e o
treino **nao sobe** -- o custo movido para o `fit` e menor que o que saiu do
`predict`, porque a ROC passa a ser calculada uma vez em vez de 400. `MS` e
`MS2` ganham menos porque nesta execucao a varredura era `"unique"`, que
custa mais por ajuste, e sao os dois que **sobem em memoria** (1,6x), pelo
`thresholds_` agora guardado por classe no um-contra-resto. E uma troca
explicita de memoria por tempo, e so eles a pagam.

Duas ressalvas sobre esses numeros, as duas necessarias:

1. **A correcao pelo controle e um limite superior.** `CC` e `PCC` tem
   codigo identico nas duas execucoes e ainda assim mediram diferente, o que
   mostra que o tempo absoluto entre elas nao e comparavel. O fator de
   correcao, porem, depende de como se agrega: 1,281 pela razao das medias
   (dominada pelos conjuntos grandes), mas **1,039 pela mediana par a par**.
   Pela mediana as duas execucoes ficam a 4% uma da outra, e entao o ganho
   real fica proximo da coluna bruta.
2. **Os numeros de limiar sao de `score_edges="unique"`.** O padrao voltou
   depois para `"fixed"`, que e o da 0.5.1; a remedicao exige a maquina em
   repouso e nao rodou. Com `"fixed"` o icamento e identidade (exceto o `MS`
   acima) e o ganho tende a ser maior, nao menor, porque a varredura fica
   mais barata.

Fora da familia de limiar, a mesma execucao carrega outras duas mudancas,
que **nao** sao o icamento e nao devem ser creditadas a ele: a forma fechada
do minimo quadrado (`GPACC` 0,090, `GACC` 0,129, `FM` 0,182 -- fator de 5 a
11) e a reescrita da representacao de histograma (`HDx` 0,432 na estimacao,
com os 528 pares de erro **identicos**).


### Por que a correspondencia entre as bibliotecas nunca chega a 100%

A subsecao anterior auditou a `final-emq`. Esta refaz a conta sobre a
`final-pareado`, com as quatro bibliotecas, e separa o que e **limitacao das
outras** do que e **diferenca de formulacao** -- a distincao importa porque
so a segunda e irredutivel.

#### O fato de partida: tres metodos, e sempre os mesmos tres

Fracao de pares (conjunto x tamanho de amostra) em que o erro da mlquantify
e **exatamente** igual ao da outra biblioteca:

| | 100% identicos | parcialmente | 0% identicos |
| --- | --- | --- | --- |
| quapy | `CC`, `PCC`, `MLPE` | `EMQ` 25%, `X` 2,9%, `GACC` 2,5%, `MS2` 2,2%, `MAX` 2,0% | `EDy`, `DyS`, `EDx`, `HDx`, `MS`, `PACC`, `SMM` |
| quantificationlib | `CC`, `PCC` | `TAC` 76%, `EMQ` 31% | `EDx`, `PACC`, `HDy8`, `HDx`, `SORD` |
| qunfold | nenhum | `GACC` 2,1% | todos os demais |

**`CC`, `PCC` e `MLPE` sao exatamente os tres metodos que nao consomem
predicoes fora da amostra** (o `EMQ` entra neles quando o `max_iter` esta
alinhado, como a subsecao anterior registra). Todo o resto depende de
escores de validacao cruzada, e e ai que a correspondencia se perde.

A causa e conhecida e e **limitacao da quapy**: ela nao expoe o
embaralhamento das dobras (`val_split` inteiro cai em
`StratifiedKFold(shuffle=False)`), entao o protocolo embaralha as linhas uma
vez no `split` para compensar. O resultado sao dobras sorteadas
**equivalentes, nao identicas** -- escores diferentes, TPR/FPR diferentes,
estimativa diferente, com o mesmo algoritmo. Nao e removivel por parametro.

#### O criterio para separar as duas causas

Dobras diferentes produzem **ruido sem vies**; formulacao diferente produz
**deslocamento enviesado**. O painel de equivalencia
(`painel_bibliotecas.py`) mostra os dois regimes lado a lado:

* **ruido**: `KDEyHD`, `KDEyCS`, `EDy`, `SMM`, `X` contra a quapy --
  diferenca mediana +0,0000 e veredito "compativel", com os 95% centrais
  dos conjuntos espalhados por +-0,01 a +-0,05;
* **vies**: `T50` -0,0038, `MS2` -0,0077 e `HDx` -0,0047 contra a quapy;
  `PWKAC` +0,0047 e `HDx` +0,0010 contra a quantificationlib.

So o segundo grupo pede explicacao de metodo. O primeiro e o piso de ruido
do protocolo, e encolhe com o numero de conjuntos, nao com o conserto de
nenhuma biblioteca.

#### Familia de limiar contra a quapy: o conjunto de candidatos

A quapy varre `np.unique(decision_scores)` em `_threshold_optim.py` -- um
candidato por escore distinto da validacao cruzada, milhares deles. A
mlquantify, com `score_edges="fixed"`, varre `np.linspace(0, 1, 101)`.
Mesma politica de escolha, listas de candidatos completamente diferentes. A
execucao `sens-limiar`, acima, ja mede o tamanho disso: adotar a varredura
exata derruba a distancia ate a quapy em fator de 8 no `MS` e pela metade no
`T50`, e **nao mexe** no `MAX`.

Essa ordem e a assinatura da causa, e e o que permite afirma-la: o `T50`
procura o limiar cujo TPR chega mais perto de 0,5, e varrendo os escores
observados quase sempre existe um ponto com TPR ~ 0,5, enquanto numa grade
de 101 pontos raramente existe; o `MS` tira a mediana das estimativas
**sobre os candidatos**, de modo que mudar quantos candidatos existem muda a
mediana direto; `MAX` e `X` procuram um extremo da curva ROC, de otimo
largo, que a grade aproxima bem.

O residuo de ~0,003 que sobra tem duas partes: os pontos degenerados, que a
quapy descarta (`tpr - fpr == 0`) antes de a politica ver os candidatos e a
0.5.1 nao descartava, ja alinhado pelo overlay; e as dobras.

Isto **nao e limitacao de nenhuma das duas**: sao duas leituras do mesmo
algoritmo. Vale registrar a consequencia de manter o padrao `"fixed"`: a
mlquantify fica identica a 0.5.1 e **mais distante da quapy**. As duas
coisas sao verdade ao mesmo tempo.

#### quantificationlib: quatro causas distintas

| metodo | causa | de quem |
| --- | --- | --- |
| `TAC` | definicao de AC multiclasse | formulacao |
| `HDx` | objetivo da otimizacao | formulacao |
| `HDy8` | `n_bins` unico em vez de varredura | limitacao da qlib |
| `PWKAC` | taxas por reinsercao | limitacao da qlib |
| `EMQ` | tolerancia de convergencia | nenhuma das duas |

**`TAC` e o caso mais limpo:** 99,1% de identidade bit a bit nos conjuntos
binarios (n=432) e **0%** nos multiclasse (n=128, `|dif|` 0,0300). O `AC` da
qlib e a forma fechada de Forman no binario e bate; no multiclasse ela troca
para inversao da matriz de confusao enquanto a mlquantify faz
um-contra-todos. Todo o +0,0046 do `TAC` vem dai, e nao e defeito de nenhuma
das duas: sao duas definicoes de "AC multiclasse".

**`HDx` nao envolve classificador nenhum**, entao aqui nao cabe alegar
dobras. O `DFX` da qlib concatena os histogramas de todos os atributos e
resolve **uma otimizacao conjunta** sobre o simplex; a mlquantify estima por
atributo e agrega **pela mediana entre atributos**, que e o que
Gonzalez-Castro et al. (2013) descrevem. Mesmo nome, objetivos diferentes.

#### `PWK`: nenhuma das duas segue o artigo

Achado de literatura, nao so de implementacao. O artigo de Barranquero,
Gonzalez, Diez e del Coz (2013), *Pattern Recognition* 46(2):472-482, esta
atras de paywall, mas a revisao dos proprios autores (Gonzalez, Castano,
Chawla, del Coz, *A Review on Quantification Learning*, secao 7.2) e
inequivoca: o metodo e **contagem ajustada**, e as taxas se estimam por
**leave-one-out**, justamente porque no kNN a matriz de distancias sai uma
vez so e serve para todas as dobras.

| aspecto | artigo | quantificationlib | mlquantify 0.5.1 |
| --- | --- | --- | --- |
| forma | AC | **AC** | `class PWK(CC)`: contagem pura |
| TPR/FPR | leave-one-out | reinsercao | 5 dobras (fora da amostra, mas nao LOO) |
| pesos | (42) e (43) | (43) | (42) com `alpha`, exposto |

A reinsercao da qlib e literal -- `predictions_train =
self.knn_estimator_.predict(X)` com `X` sendo o proprio treino, em
`multiclass/knn.py`. Num kNN e o pior caso de resubstituicao, porque o ponto
consultado e vizinho de si mesmo a distancia zero, e com o peso do `PWK`,
se ele for da minoria, esse voto vale quatro dos outros.

Medido em 12 conjuntos binarios reais, contra a verdade (o TPR/FPR do
classificador ajustado, na particao de teste):

| estimativa | mais perto em | desvio medio |
| --- | --- | --- |
| leave-one-out | **6 de 12** | **0,0769** |
| reinsercao (qlib) | 5 de 12 | 0,0903 |
| 5 dobras (mlquantify) | 1 de 12 | 0,0877 |

O LOO ganha, como o artigo preve, mas por pouco na media -- e a media
esconde o que importa. A reinsercao e otima nos conjuntos faceis e **desaba
nos dificeis**, sempre para o lado otimista: em `haberman` o TPR verdadeiro
e 0,415 e ela estima 0,675; em `sonar`, 0,729 contra 0,857. Sao exatamente
os conjuntos em que o ajuste tem trabalho a fazer. E o vies que se espera de
resubstituicao, e explica o perfil ja anotado no `registry.py` -- o erro da
qlib caindo com o tamanho da amostra enquanto o da mlquantify fica plano --,
agora com a causa.

Os pesos das duas formulas sao **exatamente proporcionais no binario**
(conferido numericamente), que e o caso de que o artigo trata, entao o voto
ponderado e identico; com 3 classes divergem, e as duas sao extrapolacoes
para fora do que o artigo define.

#### Resumo: o que e limitacao das outras bibliotecas

| limitacao | efeito medido |
| --- | --- |
| quapy nao expoe o embaralhamento das dobras | piso de ruido em **todo** metodo que usa validacao cruzada |
| qlib estima as taxas do `PWK` por reinsercao | vies otimista nos conjuntos dificeis (0,675 contra 0,415 em `haberman`) |
| qlib fixa `n_bins=8` no `HDy8` em vez de varrer | diferenca sistematica detectavel (+0,0004), dentro da margem, mas com os 95% centrais indo a +0,066 |
| qlib troca de metodo entre binario e multiclasse no `AC`/`PAC` | `TAC` com 99,1% de identidade no binario e 0% no multiclasse |
| qunfold compara iteracoes pela norma L2, nao por MAE | residuo de 5,4e-3 no `EMQ`, que cai a 7,5e-6 com `tol=1e-10` |
| qunfold sorteia o ponto de partida sem semente | nao repete os numeros entre execucoes |

E o que **nao** e limitacao de ninguem, e por isso nunca vai a 100%: as duas
definicoes de AC multiclasse, os dois objetivos do `HDx`, e os dois
conjuntos de candidatos da familia de limiar. Sao escolhas de formulacao,
cada uma defensavel, feitas por bibliotecas diferentes sobre o mesmo nome de
metodo.

### As figuras do painel de comparacao

O `painel_bibliotecas.py`, na raiz, compara a mlquantify contra cada outra
biblioteca **no mesmo metodo**, pareado por conjunto e tamanho de amostra,
usando o `expand_aliases` do `expq.aggregate` para casar `PACC`/`GPACC` e
`ACC`/`GACC` como a analise oficial casa. Ele imprime as tabelas, grava os
csv e monta um html de um arquivo so:

```bash
python painel_bibliotecas.py --run-id final-pareado \
    --custo-run custo-final5 --out-dir C:/Users/luiz_/exp-mlquantify-run
```

Sao quatro figuras, e cada uma existe porque responde uma pergunta que as
outras nao respondem:

| figura | pergunta | forma escolhida |
| --- | --- | --- |
| `_equivalencia` | as implementacoes concordam dentro de +-0,01? | floresta: ponto no Hodges-Lehmann, barra no IC de 95% |
| `_concordancia` | a discordancia e constante ou cresce com a dificuldade? | Bland-Altman, y simetrico-logaritmico com o joelho na margem |
| `_tempo` / `_memoria` | quanto a mlquantify cobra no conjunto tipico? | floresta da razao, em log2 |
| `_perfil_tempo` / `_perfil_memoria` | em que fracao dos problemas cada uma fica dentro de um fator? | perfil de desempenho de Dolan-More |

Todas se leem **impressas em preto e branco**, que e um requisito e nao um
enfeite: a forma do marcador diz a biblioteca nas florestas, o tracejado
diz a biblioteca nos perfis, e o preenchimento do marcador diz o veredito.
A cor e redundante com isso, e os tons foram escolhidos com luminancias
distintas.

#### O que o perfil de desempenho mostrou, e as outras nao mostravam

O perfil e o grafico que Dolan e More (2002) propuseram para benchmark de
otimizadores, pela mesma razao que aqui vale: media e mediana sao dominadas
por poucos problemas caros. Cada problema e um par (metodo, conjunto), com
os quatro tamanhos de amostra entrando pela mediana; o eixo x e o fator em
relacao a biblioteca mais barata **naquele problema** e o y e a fracao de
problemas dentro do fator.

Tempo de estimacao, com a `custo-final5` aplicada nos 12 metodos que ela
mediu:

| | mais barata (tau=1) | ate 2x | ate 4x | ate 8x | pior caso |
| --- | --- | --- | --- | --- | --- |
| mlquantify (contra a quapy) | 42,5% | 77,9% | **93,3%** | **99,5%** | 42x |
| quapy | **57,5%** | 76,1% | 84,3% | 88,4% | 28x |
| mlquantify (contra a qlib) | 44,5% | 64,8% | 81,3% | **89,0%** | 523x |
| quantificationlib | **55,5%** | **76,3%** | **86,8%** | 87,1% | 116x |
| mlquantify (contra a qunfold) | **99,7%** | **99,9%** | **99,9%** | **99,9%** | 8x |
| qunfold | 0,3% | 2,0% | 6,7% | 15,3% | 735x |

**As curvas se cruzam, e e esse o achado.** Contra a quapy, a quapy e a mais
barata em 57,5% dos problemas -- ela ganha mais vezes --, mas a partir de
tau ~ 2 a mlquantify cobre mais, e em tau = 8 a distancia e de 11 pontos
(99,5% contra 88,4%). Contra a quantificationlib o cruzamento vem mais
tarde, perto de tau = 8. A floresta de razao nao podia dizer isso: ela
resume cada par numa mediana, e a mediana fica do lado de quem ganha mais
vezes.

**Mas o pior caso continua sendo da mlquantify nos dois**, e o perfil diz
isso na mesma figura: 42x contra os 28x da quapy, 523x contra os 116x da
qlib. As duas leituras sao verdadeiras e se completam -- a mlquantify e
mais consistente no miolo e tem a cauda mais longa.

Os casos da cauda nao estao espalhados, estao concentrados, e a
concentracao nomeia a causa:

* **contra a quantificationlib, sao conjuntos com muitas classes.** As seis
  piores razoes sao `PACC` (523x), `TAC` (309x), `PWKAC` (173x) e `FM`
  (153x) em `Walking.csv`, que tem **22 classes**, mais `PACC` em
  `amazon-commerce-reviews` (10 classes, 274x) e `PWKAC` em `letter.csv`
  (26 classes, 80x). E o custo do um-contra-resto da mlquantify, que ajusta
  um quantificador por classe onde a qlib resolve uma vez so;
* **contra a quapy, sao conjuntos grandes com a familia de limiar.** As
  seis piores sao `MAX`, `T50` e `X` em `phphQEck0.csv` (245 mil
  instancias) e `Mosquitoes.csv` (158 mil), entre 33x e 42x. Vale lembrar
  que esses numeros ja sao os da `custo-final5`, isto e, **depois** do
  icamento da ROC, e com a varredura `"unique"`, que e a mais cara das
  duas; com `score_edges="fixed"` a cauda deve encolher, e a remedicao esta
  pendente.

Na memoria nao ha cruzamento: a mlquantify e a mais barata em 87,0% dos
problemas contra a quantificationlib, 60,9% contra a quapy e 92,8% contra a
qunfold.

**Ressalva sobre o quarto painel.** O perfil canonico, com as quatro curvas
sobre o mesmo conjunto de problemas, exige que todas tenham medido o mesmo
par (metodo, conjunto), e so **tres metodos existem nas quatro bibliotecas**
(`EDx`, `EDy`, `EMQ`), o que deixa 404 problemas. Nesse recorte a quapy
domina sem ambiguidade (59,7% em tau=1 e 99,0% ate 2x, com pior caso de
apenas 2x). O painel serve como ilustracao do formato canonico; a amostra e
pequena e e um recorte enviesado do catalogo, entao nao sustenta conclusao
geral. Os tres paineis pareados, que usam todos os problemas que cada par
tem em comum, e que sustentam.

---

## 12. Organizacao do codigo

```
expq/
  config.py      parametros do protocolo e caminhos
  catalog.py     varredura de datasets/ e os quatro criterios de selecao
  data.py        leitura e preparacao comum (com cache)
  protocol.py    divisao 50/50 e geracao das amostras do UPP (com cache)
  metrics.py     AE, RAE, SE e NKLD, implementados uma unica vez
  registry.py    de que classe e cada metodo, em cada biblioteca
  runner.py      execucao de um job, com tempo, memoria e tratamento de erro
  parallel.py    escalonador em processos, com tempo limite e barras
  ledger.py      registro de andamento e log
  aggregate.py   reuniao dos parciais, Q1/Q2/Q3 e testes estatisticos
  loc_count.py   contagem de linhas de codigo (Q3)
```

Na raiz, alem de `run_experiments.py` e `aggregate_results.py`:

```
build_catalog.py    monta o catalogo e aplica os criterios de selecao
count_loc.py        contagem de linhas de codigo dos scripts de uso (Q3)
clean_ledger.py     apaga medidas contaminadas por suspensao da maquina
clean_pressure.py   apaga falhas de alocacao causadas por falta de RAM
clean_invalid.py    apaga combinacoes reprovadas por amostra invalida isolada
clean_schema.py     realinha parciais gravados antes de uma coluna nova existir
analyze_results.py  responde Q1, Q2 e Q3 a partir das tabelas agregadas
```

Os tres `clean_*.py` sao a mesma ideia: apagam do ledger so as combinacoes
que precisam ser refeitas, preservam o resto, e sem `--apply` apenas mostram
o que fariam.

Para acrescentar um metodo, basta uma linha em `registry.py` com o nome
canonico e a classe que o implementa.
