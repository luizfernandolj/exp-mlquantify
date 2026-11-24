"""Recorte dos conjuntos de dados que entram nas analises do capitulo 5.

Duas listas em ``descricao/`` guardam, uma por criterio de selecao, os nomes
dos conjuntos que ficam de fora:

``conjuntos_fora_do_criterio.txt``
    entraram na colecao depois da selecao e passam de dez classes;
``conjuntos_minoria_abaixo_de_10.txt``
    classe minoritaria com menos de 10% das linhas.

O corte vale pelo nome do conjunto, e nao pelo que esta gravado no
``summary.csv``, de proposito: uma reagregacao a partir de ``parts/``
reescreve o summary e devolveria em silencio o que o recorte ja tinha
tirado. As linhas dos conjuntos cortados por minoria ficam guardadas em
``summary_minoria_abaixo_de_10.csv``, ao lado do summary da execucao.

Quem le o summary chama ``excluidos()`` ou ``filtrar()``. O
``painel_bibliotecas.carregar`` ja aplica o recorte, entao os scripts que
passam por ele nao precisam filtrar de novo.
"""

from __future__ import annotations

import io
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DESCRICAO = RAIZ / "descricao"

#: motivo do corte -> arquivo com um dataset_id por linha
LISTAS = {
    "classes": "conjuntos_fora_do_criterio.txt",
    "minoria": "conjuntos_minoria_abaixo_de_10.txt",
}


def ler(motivo: str) -> set[str]:
    """Os conjuntos de uma lista, pelo motivo do corte."""
    caminho = DESCRICAO / LISTAS[motivo]
    if not caminho.exists():
        raise SystemExit(f"nao achei {caminho}")
    with io.open(caminho, encoding="utf-8") as f:
        return {linha.strip() for linha in f if linha.strip()}


def excluidos(motivos=None) -> set[str]:
    """Todos os conjuntos fora do recorte, ou so os dos motivos pedidos."""
    return set().union(*(ler(m) for m in (motivos or LISTAS)))


def filtrar(df, coluna: str = "dataset_id", motivos=None):
    """Tira do dataframe as linhas dos conjuntos fora do recorte."""
    return df[~df[coluna].isin(excluidos(motivos))].copy()


def resumo(motivos=None) -> str:
    """Quantos saem por motivo, para a saida dos scripts."""
    return ", ".join(f"{len(ler(m))} por {m}" for m in (motivos or LISTAS))
