"""Leitura e preparacao dos conjuntos de dados.

Preparacao comum descrita na Secao 4.3.3: atributos categoricos convertidos
em representacao numerica, rotulos mapeados para inteiros consecutivos e
valores ausentes imputados pela mediana (numericos) ou pela moda
(categoricos). O resultado e guardado em cache para que as quatro
bibliotecas recebam exatamente a mesma matriz.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Nomes aceitos para a coluna de rotulos, em ordem de preferencia.
TARGET_CANDIDATES = ("target", "class", "Class", "label", "Label", "y", "classe")


@dataclass
class Dataset:
    dataset_id: str
    X: np.ndarray
    y: np.ndarray
    target_column: str
    n_instances: int
    n_features: int
    n_classes: int
    class_counts: dict
    imbalance_ratio: float
    n_categorical: int
    n_missing: int
    fingerprint: str

    @property
    def dataset_type(self) -> str:
        return "binario" if self.n_classes == 2 else "multiclasse"


def find_target_column(df: pd.DataFrame) -> str:
    """Coluna de rotulos: nome conhecido, senao a ultima coluna."""
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name
    return df.columns[-1]


def read_csv(path: Path) -> pd.DataFrame:
    """Leitura tolerante a separador e codificacao."""
    for kwargs in ({}, {"sep": ";"}, {"encoding": "latin-1"},
                   {"sep": ";", "encoding": "latin-1"}):
        try:
            df = pd.read_csv(path, low_memory=False, **kwargs)
        except Exception:
            continue
        if df.shape[1] > 1:
            return df
    return pd.read_csv(path, low_memory=False)


def prepare(df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Aplica a preparacao comum e devolve (X, y, n_categoricas, n_ausentes)."""
    df = df.loc[:, ~df.columns.duplicated()]
    y_raw = df[target_column]
    X_df = df.drop(columns=[target_column])

    # Colunas sem nenhum valor nao acrescentam informacao.
    X_df = X_df.dropna(axis=1, how="all")

    keep = y_raw.notna().to_numpy()
    X_df, y_raw = X_df.loc[keep], y_raw.loc[keep]

    n_missing = int(X_df.isna().to_numpy().sum())

    num_cols = X_df.select_dtypes(include=[np.number]).columns
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    # Imputacao antes da codificacao: mediana nos numericos, moda nos demais.
    for c in num_cols:
        if X_df[c].isna().any():
            X_df[c] = X_df[c].fillna(X_df[c].median())
    for c in cat_cols:
        if X_df[c].isna().any():
            mode = X_df[c].mode(dropna=True)
            X_df[c] = X_df[c].fillna(mode.iloc[0] if len(mode) else "__ausente__")

    # Atributos categoricos em representacao numerica (codificacao ordinal).
    for c in cat_cols:
        X_df[c] = pd.factorize(X_df[c].astype(str))[0]

    X = X_df.to_numpy(dtype=np.float64, na_value=0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Rotulos em inteiros consecutivos.
    y = pd.factorize(y_raw.astype(str), sort=True)[0].astype(np.int64)

    return X, y, len(cat_cols), n_missing


def fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    """Resumo do conteudo, usado para detectar arquivos duplicados."""
    h = hashlib.md5()
    h.update(np.asarray(X.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(np.round(X, 6)).tobytes())
    h.update(np.ascontiguousarray(np.sort(np.bincount(y))).tobytes())
    return h.hexdigest()


def load_dataset(path: Path, dataset_id: str, cache_dir: Path | None = None,
                 max_rows: int | None = None, random_state: int = 42) -> Dataset:
    """Carrega, prepara e opcionalmente guarda em cache um conjunto de dados."""
    cache_file = None
    if cache_dir is not None:
        safe = dataset_id.replace("/", "__").replace("\\", "__")
        suffix = f"__max{max_rows}" if max_rows else ""
        cache_file = Path(cache_dir) / f"{safe}{suffix}.npz"
        if cache_file.exists():
            try:
                return _from_cache(cache_file, dataset_id)
            except Exception:
                cache_file.unlink(missing_ok=True)

    df = read_csv(Path(path))
    target_column = find_target_column(df)
    X, y, n_cat, n_missing = prepare(df, target_column)

    if max_rows is not None and len(y) > max_rows:
        X, y = _stratified_subsample(X, y, max_rows, random_state)

    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    values = np.array(list(counts.values()), dtype=float)
    ds = Dataset(
        dataset_id=dataset_id, X=X, y=y, target_column=target_column,
        n_instances=int(len(y)), n_features=int(X.shape[1]), n_classes=len(counts),
        class_counts=counts,
        imbalance_ratio=float(values.max() / values.min()) if len(values) else float("nan"),
        n_categorical=n_cat, n_missing=n_missing, fingerprint=fingerprint(X, y),
    )

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, X=ds.X, y=ds.y,
                            meta=np.array([ds.target_column, str(ds.n_categorical),
                                           str(ds.n_missing), ds.fingerprint]))
    return ds


def _from_cache(cache_file: Path, dataset_id: str) -> Dataset:
    data = np.load(cache_file, allow_pickle=False)
    X, y = data["X"], data["y"]
    target_column, n_cat, n_missing, fp = (str(v) for v in data["meta"])
    counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
    values = np.array(list(counts.values()), dtype=float)
    return Dataset(dataset_id=dataset_id, X=X, y=y, target_column=target_column,
                   n_instances=int(len(y)), n_features=int(X.shape[1]),
                   n_classes=len(counts), class_counts=counts,
                   imbalance_ratio=float(values.max() / values.min()),
                   n_categorical=int(n_cat), n_missing=int(n_missing), fingerprint=fp)


def _stratified_subsample(X, y, max_rows, random_state):
    rng = np.random.default_rng(random_state)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = max(2, int(round(len(idx) * max_rows / len(y))))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]
