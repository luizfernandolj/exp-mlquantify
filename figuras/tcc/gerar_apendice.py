"""Gera o apendice com a tabela dos 110 conjuntos selecionados."""
import csv
import io
import sys

BASE = "C:/Users/luiz_/OneDrive/Codigos/exp-mlquantify/"
sys.path.insert(0, BASE)
import recorte_analise  # noqa: E402  precisa do BASE no sys.path
SAIDA = BASE + "figuras/tcc/apendice-conjuntos.tex"

#: Marca da coluna de multiclasse, com simbolo em vez de letra.
MARCA = {True: r"$\checkmark$", False: r"$\times$"}

#: De onde veio cada conjunto, no nome usado no capitulo de metodo. Fica
#: registrado aqui ainda que a tabela nao traga mais a coluna de origem.
ORIGENS = {
    "datasets_isa": "Maletzke",
    "datasets_jp/schumacher": "Schumacher",
    "datasets_jp/schumacher/first-order-final": "Schumacher",
    "datasets_jp/ours": "Grupo",
    "datasets_jp/kaggle": "Kaggle",
    "datasets_jp/quapy_data": "QuaPy",
    "datasets_jp/openml": "OpenML",
    "datasets_jp/uci": "UCI",
}

CABECALHO = r"""%\renewcommand{\appendixname}{Anexo} %trocar Apendice por Anexo
\appendix
\chapter{Conjuntos de Dados Utilizados}
\label{ap:conjuntos}

Os 110 conjuntos de dados que restaram após os critérios de seleção da
Seção~\ref{subsec:etapa_dados} são listados na Tabela~\ref{tab:ap_conjuntos}. De cada um
são informados o número de atributos, o número de instâncias, o número de classes, a
razão de desbalanceamento e a indicação de problema multiclasse.

O símbolo $n$ corresponde ao total de instâncias do conjunto, antes da divisão entre
treinamento e teste descrita na Seção~\ref{subsec:etapa_avaliacao}. O símbolo $k$
corresponde ao número de classes distintas, com $k = 2$ nos problemas binários. A razão
de desbalanceamento é o número de instâncias da classe majoritária dividido pelo da
minoritária, de modo que o valor 1 indica classes de mesmo tamanho. Os conjuntos
aparecem com os binários primeiro e, dentro de cada grupo, em ordem alfabética.

\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{lrrrrc}
    \caption{Conjuntos de dados utilizados na avaliação experimental.}
    \label{tab:ap_conjuntos}\\
    \hline
    \textbf{Conjunto} & \textbf{Atrib.} & \textbf{$n$} & \textbf{$k$} & \textbf{Desbal.} & \textbf{Multi.} \\
    \hline
    \endfirsthead
    \caption[]{Conjuntos de dados utilizados na avaliação experimental (continuação).}\\
    \hline
    \textbf{Conjunto} & \textbf{Atrib.} & \textbf{$n$} & \textbf{$k$} & \textbf{Desbal.} & \textbf{Multi.} \\
    \hline
    \endhead
    \hline
    \endfoot
"""

RODAPE = r"""    \hline
\end{longtable}

{\footnotesize Fonte: elaborado pelo autor.}
\normalsize
"""


def milhar(v):
    return f"{int(float(v)):,}".replace(",", ".")


def virgula(v, casas=2):
    return f"{float(v):.{casas}f}".replace(".", ",")


def escapar(nome):
    return nome.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


fora = recorte_analise.excluidos()
linhas = list(csv.DictReader(io.open(BASE + "descricao/datasets_selecionados.csv",
                                     encoding="utf-8")))
sel = [l for l in linhas if l["dataset_id"] not in fora]
# binarios primeiro, como na tabela resumo do capitulo de metodo, e dentro de
# cada grupo em ordem alfabetica
sel.sort(key=lambda l: (int(float(l["n_classes"])) > 2,
                        l["nome_sem_extensao"].lower()))

corpo = []
for l in sel:
    corpo.append("    " + " & ".join([
        r"\texttt{" + escapar(l["nome_sem_extensao"]) + "}",
        milhar(l["n_atributos"]),
        milhar(l["n_instancias"]),
        l["n_classes"],
        virgula(l["razao_desbalanceamento"]),
        MARCA[int(float(l["n_classes"])) > 2],
    ]) + r" \\")

io.open(SAIDA, "w", encoding="utf-8", newline="\n").write(
    CABECALHO + "\n".join(corpo) + "\n" + RODAPE)
print(f"{len(corpo)} conjuntos em {SAIDA}")
