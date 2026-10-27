"""Medidas de erro de quantificacao (Secao 3.5.2).

Implementadas aqui, e nao importadas de uma das bibliotecas comparadas,
para que a mesma formula seja aplicada as quatro.
"""

from __future__ import annotations

import numpy as np


class PrevalenceError(ValueError):
    """A saida da biblioteca nao e um vetor de prevalencias valido."""


class PrevalenceShapeError(PrevalenceError):
    """Numero de prevalencias diferente do numero de classes.

    E falha do metodo, nao da amostra: vale para todas as amostras igualmente.
    """


class PrevalenceValueError(PrevalenceError):
    """O vetor tem o formato certo, mas nao ha o que normalizar.

    Acontece quando a estimativa vem toda nula ou com valores nao finitos. A
    chamada da biblioteca funcionou; a estimativa daquela amostra e que nao
    serve.
    """


def smooth(p: np.ndarray, eps: float) -> np.ndarray:
    """Suavizacao aditiva usada no RAE e no NKLD quando ha prevalencia zero."""
    return (p + eps) / (1.0 + eps * p.shape[-1])


def ae(true_p: np.ndarray, pred_p: np.ndarray) -> float:
    """Erro absoluto medio sobre as classes."""
    return float(np.mean(np.abs(np.asarray(pred_p) - np.asarray(true_p))))


def rae(true_p: np.ndarray, pred_p: np.ndarray, eps: float) -> float:
    """Erro absoluto relativo, com suavizacao ``eps = 1 / (2 * m)``."""
    t = smooth(np.asarray(true_p, dtype=float), eps)
    p = smooth(np.asarray(pred_p, dtype=float), eps)
    return float(np.mean(np.abs(p - t) / t))


def se(true_p: np.ndarray, pred_p: np.ndarray) -> float:
    """Erro quadratico medio sobre as classes."""
    return float(np.mean((np.asarray(pred_p) - np.asarray(true_p)) ** 2))


def nkld(true_p: np.ndarray, pred_p: np.ndarray, eps: float) -> float:
    """Divergencia de Kullback-Leibler normalizada."""
    t = smooth(np.asarray(true_p, dtype=float), eps)
    p = smooth(np.asarray(pred_p, dtype=float), eps)
    kld = float(np.sum(t * np.log(t / p)))
    e = np.exp(kld)
    return float(2.0 * e / (1.0 + e) - 1.0)


def eps_for(sample_size: int) -> float:
    """Fator de suavizacao recomendado para uma amostra de ``m`` instancias."""
    return 1.0 / (2.0 * sample_size)


def all_metrics(true_p: np.ndarray, pred_p: np.ndarray, sample_size: int) -> dict:
    e = eps_for(sample_size)
    return {"ae": ae(true_p, pred_p), "rae": rae(true_p, pred_p, e),
            "se": se(true_p, pred_p), "nkld": nkld(true_p, pred_p, e)}


def prevalence_from_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Vetor de prevalencias observado em um conjunto de rotulos."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    return counts / counts.sum()


def normalize_prevalence(p, n_classes: int) -> np.ndarray:
    """Converte a saida de uma biblioteca em um vetor valido no simplexo.

    Cobre as diferencas de formato entre as bibliotecas (dicionario, lista,
    matriz 1xC) e rejeita saidas que nao sejam um vetor de prevalencias.
    """
    if isinstance(p, dict):
        p = [p[k] for k in sorted(p)]
    arr = np.asarray(p, dtype=float).ravel()
    if arr.size != n_classes:
        raise PrevalenceShapeError(
            f"esperado vetor de {n_classes} prevalencias, recebido {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise PrevalenceValueError("vetor de prevalencias com valores nao finitos")
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0:
        raise PrevalenceValueError("vetor de prevalencias com soma nula")
    return arr / total
