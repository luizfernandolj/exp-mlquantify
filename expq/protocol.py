"""Divisao treino/teste e geracao das amostras do protocolo UPP.

As amostras sao geradas uma unica vez por conjunto de dados e armazenadas
como indices, de modo que todas as bibliotecas recebam exatamente as mesmas
amostras de teste (Secao 4.3.4, passo 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class Samples:
    """Amostras do protocolo, achatadas em um unico vetor de indices."""

    indices: np.ndarray  # concatenacao dos indices de todas as amostras
    offsets: np.ndarray  # offsets[i]:offsets[i+1] delimita a amostra i
    sizes: np.ndarray    # tamanho m de cada amostra

    def __len__(self) -> int:
        return len(self.sizes)

    def get(self, i: int) -> np.ndarray:
        return self.indices[self.offsets[i]:self.offsets[i + 1]]

    def by_size(self, m: int) -> list[int]:
        return [i for i in range(len(self)) if self.sizes[i] == m]


def split(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int):
    """Divisao estratificada em duas partes (50/50 por padrao).

    A particao de treinamento sai em ordem aleatoria, nao na ordem do
    arquivo, e isso faz parte do protocolo comum: a quapy nao expoe o
    embaralhamento das dobras da validacao cruzada -- ``val_split`` inteiro
    cai em ``StratifiedKFold(shuffle=False)`` --, de modo que, com as linhas
    na ordem do arquivo, as dobras dela seguiriam essa ordem enquanto as das
    outras tres seriam sorteadas. Embaralhar aqui, uma vez e com semente,
    tira a ordem do arquivo da conta para todas (expq/registry.py).

    A particao de teste continua ordenada: as amostras do UPP sao indices
    sobre ela, e mexer na ordem invalidaria o cache de amostras.
    """
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=random_state,
                              stratify=y, shuffle=True)
    return tr, np.sort(te)


def make_samples(y_test: np.ndarray, sample_sizes, n_prevalences: int,
                 random_state: int) -> Samples:
    """Gera as amostras do UPP sobre a particao de teste.

    Usa o protocolo implementado na biblioteca proposta; as amostras
    resultantes sao indices sobre ``y_test`` e nada mais da biblioteca
    participa da geracao.
    """
    from mlquantify.model_selection import UPP

    upp = UPP(batch_size=list(sample_sizes), n_prevalences=int(n_prevalences),
              random_state=random_state)
    dummy_X = np.zeros((len(y_test), 1), dtype=np.float64)

    chunks, sizes = [], []
    for idx in upp.split(dummy_X, y_test):
        idx = np.asarray(idx, dtype=np.int64).ravel()
        chunks.append(idx)
        sizes.append(len(idx))

    if not chunks:
        raise RuntimeError("protocolo UPP nao gerou nenhuma amostra")

    offsets = np.zeros(len(chunks) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    return Samples(indices=np.concatenate(chunks), offsets=offsets,
                   sizes=np.asarray(sizes, dtype=np.int64))


def load_or_make(cache_dir: Path, dataset_id: str, y_test: np.ndarray,
                 sample_sizes, n_prevalences: int, random_state: int) -> Samples:
    """Le as amostras do cache ou as gera e guarda."""
    safe = dataset_id.replace("/", "__").replace("\\", "__")
    tag = "-".join(str(s) for s in sample_sizes)
    path = Path(cache_dir) / f"{safe}__m{tag}__n{n_prevalences}__s{random_state}.npz"

    if path.exists():
        try:
            d = np.load(path, allow_pickle=False)
            return Samples(indices=d["indices"], offsets=d["offsets"], sizes=d["sizes"])
        except Exception:
            path.unlink(missing_ok=True)

    s = make_samples(y_test, sample_sizes, n_prevalences, random_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, indices=s.indices, offsets=s.offsets, sizes=s.sizes)
    return s
