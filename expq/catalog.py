"""Catalogacao e selecao dos conjuntos de dados (Etapa 3, Secao 4.3.3).

Percorre ``datasets/``, inspeciona cada arquivo e aplica, em sequencia, os
quatro criterios de selecao da monografia:

1. arquivos auxiliares (particoes de treino/teste/validacao e submissoes);
2. classe unica;
3. duplicatas;
4. classes raras (classe minoritaria com menos de ``min_minority`` instancias).

Gera dois arquivos em ``descricao/``:

``catalogo_experimentos.csv``
    todos os arquivos encontrados, com os atributos medidos e o motivo da
    eventual exclusao;
``datasets_selecionados.csv``
    apenas os selecionados, em ordem deterministica. Essa ordem e o indice
    usado por ``--dataset-range`` para dividir o trabalho entre maquinas.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .config import Config
from .data import load_dataset

log = logging.getLogger(__name__)

#: Padroes de nome que indicam arquivo auxiliar e nao um conjunto completo.
#: Comparados com o nome do arquivo *sem a extensao*, para que ``train.csv``
#: e ``fashion-mnist_train.csv`` sejam igualmente reconhecidos.
AUX_PATTERNS = (
    r"(^|[_\-])train([_\-]|$)", r"(^|[_\-])test([_\-]|$)",
    r"(^|[_\-])valid(ation)?([_\-]|$)", r"(^|[_\-])val([_\-]|$)",
    r"(^|[_\-])dev([_\-]|$)", r"sample[_\-]?submission", r"(^|[_\-])submission",
    r"(^|[_\-])solution", r"(^|[_\-])gold", r"(^|[_\-])split[_\-]?\d+",
    r"data[_\-]?dictionary",
)
AUX_RE = re.compile("|".join(AUX_PATTERNS), re.IGNORECASE)


def discover(cfg: Config) -> list[Path]:
    """Todos os .csv sob ``datasets/``, em ordem deterministica."""
    paths = sorted(cfg.datasets_dir.rglob("*.csv"), key=lambda p: str(p).lower())
    return [p for p in paths if "__pycache__" not in p.parts]


def inspect_all(cfg: Config) -> pd.DataFrame:
    """Le cada arquivo e mede os atributos usados pelos criterios de selecao."""
    rows = []
    paths = discover(cfg)
    for path in tqdm(paths, desc="Inspecionando conjuntos", unit="arq"):
        rel = path.relative_to(cfg.root).as_posix()
        row = {
            "dataset_id": rel,
            "nome_arquivo": path.name,
            "nome_sem_extensao": path.stem,
            "origem": path.parent.relative_to(cfg.datasets_dir).as_posix() or ".",
            "tamanho_mb": round(path.stat().st_size / 1e6, 4),
        }
        try:
            ds = load_dataset(path, rel, cache_dir=cfg.cache_dir / "prepared",
                              max_rows=cfg.max_rows, random_state=cfg.random_state)
            row.update({
                "erro": "",
                "coluna_classe": ds.target_column,
                "n_instancias": ds.n_instances,
                "n_atributos": ds.n_features,
                "n_classes": ds.n_classes,
                "n_categoricas": ds.n_categorical,
                "n_ausentes": ds.n_missing,
                "classe_minoritaria_qtd": min(ds.class_counts.values()),
                "classe_majoritaria_qtd": max(ds.class_counts.values()),
                "razao_desbalanceamento": round(ds.imbalance_ratio, 4),
                "distribuicao_classes": str(sorted(ds.class_counts.values())),
                "fingerprint": ds.fingerprint,
                "tipo_dataset": ds.dataset_type,
            })
        except Exception as exc:  # arquivo ilegivel nao entra na colecao
            row.update({"erro": f"{type(exc).__name__}: {exc}", "n_classes": -1})
            log.warning("falha ao ler %s: %s", rel, exc)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_selection(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Aplica os quatro criterios, registrando o motivo de cada exclusao."""
    df = df.copy()
    df["selecionado"] = True
    df["motivo_exclusao"] = ""

    def drop(mask: pd.Series, reason: str) -> None:
        mask = mask & df["selecionado"]
        df.loc[mask, "selecionado"] = False
        df.loc[mask, "motivo_exclusao"] = reason

    # 0. arquivos que nao puderam ser lidos
    drop(df["erro"].astype(str).str.len() > 0, "ilegivel")

    # 1. arquivos auxiliares
    drop(df["nome_sem_extensao"].str.contains(AUX_RE, na=False), "arquivo_auxiliar")

    # 2. classe unica
    drop(df["n_classes"].fillna(-1) < 2, "classe_unica")

    # 3. duplicatas
    if cfg.duplicate_key == "content":
        key_cols = ["fingerprint"]
    else:  # criterio da monografia: mesmo tamanho e mesma distribuicao
        key_cols = ["n_instancias", "distribuicao_classes"]
    ativos = df[df["selecionado"]]
    dup = ativos.duplicated(subset=key_cols, keep="first")
    drop(pd.Series(dup, index=ativos.index).reindex(df.index, fill_value=False), "duplicata")

    # 4. classes raras
    drop(df["classe_minoritaria_qtd"].fillna(0) < cfg.min_minority, "classe_rara")

    return df


def build(cfg: Config) -> pd.DataFrame:
    """Monta o catalogo completo e o arquivo de selecionados."""
    cfg.make_dirs()
    df = inspect_all(cfg)
    df = apply_selection(df, cfg)

    df = df.sort_values("dataset_id").reset_index(drop=True)
    df.to_csv(cfg.full_catalog, index=False, encoding="utf-8")

    sel = df[df["selecionado"]].copy()
    sel = sel.sort_values(["tipo_dataset", "n_instancias", "dataset_id"]).reset_index(drop=True)
    sel.insert(0, "indice", range(len(sel)))
    sel.to_csv(cfg.selected_catalog, index=False, encoding="utf-8")

    _report(df, sel)
    return sel


def _report(df: pd.DataFrame, sel: pd.DataFrame) -> None:
    print(f"\nArquivos encontrados: {len(df)}")
    excl = df[~df["selecionado"]]["motivo_exclusao"].value_counts()
    for reason, n in excl.items():
        print(f"  removidos por {reason}: {n}")
    print(f"Selecionados: {len(sel)}")
    print(sel["tipo_dataset"].value_counts().to_string())
    if len(sel):
        for col in ("n_instancias", "n_atributos", "n_classes", "razao_desbalanceamento"):
            s = sel.groupby("tipo_dataset")[col].agg(["min", "max", "mean", "std"])
            print(f"\n{col}:\n{s.round(2).to_string()}")


def load_selected(cfg: Config) -> pd.DataFrame:
    """Le ``datasets_selecionados.csv`` e aplica os filtros da linha de comando."""
    if not cfg.selected_catalog.exists():
        raise FileNotFoundError(
            f"{cfg.selected_catalog} nao existe. Rode primeiro: python build_catalog.py"
        )
    sel = pd.read_csv(cfg.selected_catalog)

    if cfg.dataset_types:
        sel = sel[sel["tipo_dataset"].isin(cfg.dataset_types)]
    if cfg.only_datasets:
        pat = "|".join(re.escape(s) for s in cfg.only_datasets)
        sel = sel[sel["dataset_id"].str.contains(pat, case=False, na=False)]
    if cfg.skip_datasets:
        pat = "|".join(re.escape(s) for s in cfg.skip_datasets)
        sel = sel[~sel["dataset_id"].str.contains(pat, case=False, na=False)]
    if cfg.dataset_range is not None:
        lo, hi = cfg.dataset_range
        sel = sel[(sel["indice"] >= lo) & (sel["indice"] < hi)]
    if cfg.shard is not None:
        i, total = cfg.shard
        sel = sel[sel["indice"] % total == (i - 1) % total]

    return sel.reset_index(drop=True)
