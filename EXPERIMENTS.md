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
4 classes), onde o `ACC` e o `TAC` da mlquantify devolvem o vetor nulo em
**180 das 400 amostras** -- 81% das de `m = 10` e ainda 23% das de
`m = 500`. Os dois sao binarios na mlquantify, e o multiclasse passa por
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
| `ACC` | 180 de 400 nulas | **134 de 400 nulas** |
| `TAC` | 180 de 400 nulas | **134 de 400 nulas** |
| `T50` | 24 de 400 nulas | 0 nulas, MAE 0,231 a 0,177 |
| `GACC` | 0 nulas, MAE 0,286 a 0,282 | 0 nulas, MAE 0,277 a 0,243 |

Ou seja: padronizar conserta o `T50`, melhora um pouco o erro de todos --
e **nao resolve o `ACC` nem o `TAC`**, que continuam anulando um terco das
amostras com um classificador convergido. O truncamento antes da combinacao
e o que basta para zerar o vetor; o classificador ruim so aumenta a conta.
Como ele e o mesmo para as quatro bibliotecas, a comparacao continua justa:
a diferenca e que as outras tres degradam e a mlquantify anula.

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
