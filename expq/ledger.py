"""Registro de andamento e log da execucao.

O *ledger* e a fonte de verdade sobre o que ja foi executado: uma linha JSON
por combinacao (conjunto de dados, biblioteca, metodo), acrescentada assim
que a combinacao termina. Retomar uma execucao interrompida, ou saber quanto
falta, depende apenas desses arquivos -- nenhum arquivo de resultado precisa
ser lido.

Cada processo escreve em seu proprio arquivo (``ledger/<node>/<pid>.jsonl``),
de modo que varios trabalhadores, e varias maquinas, nunca disputam o mesmo
arquivo. O log legivel (``logs/<run_id>__<node>.log``) e escrito apenas pelo
processo principal.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

from .config import Config

_HANDLE = None


def _handle(cfg: Config):
    """Arquivo de ledger deste processo, aberto uma unica vez."""
    global _HANDLE
    if _HANDLE is None or _HANDLE.closed:
        cfg.ledger_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.ledger_dir / f"{os.getpid()}.jsonl"
        _HANDLE = path.open("a", encoding="utf-8")
    return _HANDLE


def ledger_entry(cfg: Config, **fields) -> None:
    """Acrescenta uma linha ao ledger e a descarrega imediatamente."""
    fields.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    fields.setdefault("run_id", cfg.run_id)
    fields.setdefault("node", cfg.node)
    fh = _handle(cfg)
    fh.write(json.dumps(fields, ensure_ascii=False) + "\n")
    fh.flush()


def close_ledger() -> None:
    global _HANDLE
    if _HANDLE is not None and not _HANDLE.closed:
        _HANDLE.close()
    _HANDLE = None


def read_entries(cfg: Config, all_nodes: bool = True) -> list[dict]:
    """Le o ledger de toda a execucao (ou apenas o desta maquina)."""
    root = cfg.run_dir / "ledger"
    if not root.exists():
        return []
    paths = root.glob("*/*.jsonl") if all_nodes else (cfg.ledger_dir).glob("*.jsonl")
    entries = []
    for path in sorted(paths):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # linha truncada por uma interrupcao
    return entries


def load_done(cfg: Config, all_nodes: bool = True) -> set[str]:
    """Chaves ``dataset|biblioteca|metodo`` ja concluidas.

    Entradas com ``status`` de erro tambem contam como concluidas: repetir
    um metodo que falhou por limitacao da biblioteca so gastaria tempo. Para
    reexecuta-las, apague o ledger ou use ``--fresh``.
    """
    done = set()
    for e in read_entries(cfg, all_nodes):
        if {"dataset_id", "library", "method"} <= e.keys():
            done.add(f"{e['dataset_id']}|{e['library']}|{e['method']}")
    return done


def status_report(cfg: Config) -> dict:
    """Resumo do andamento, obtido apenas do ledger."""
    entries = read_entries(cfg)
    by_status = Counter(e.get("status", "?") for e in entries)
    by_library = Counter(e.get("library", "?") for e in entries)
    datasets = {e.get("dataset_id") for e in entries}
    seconds = sum(float(e.get("seconds", 0) or 0) for e in entries)
    return {
        "combinacoes_registradas": len(entries),
        "conjuntos_tocados": len(datasets),
        "por_status": dict(by_status),
        "por_biblioteca": dict(by_library),
        "tempo_acumulado_s": round(seconds, 1),
        "tempo_acumulado_h": round(seconds / 3600, 2),
    }


def setup_logging(cfg: Config, verbose: bool = False) -> logging.Logger:
    """Log em arquivo (detalhado) e no terminal (apenas avisos).

    O terminal fica reservado para as barras de progresso; o que acontece em
    cada combinacao vai para ``logs/``.
    """
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("expq")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(cfg.log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG if verbose else logging.WARNING)
    sh.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(sh)

    return logger


def silence_warnings() -> None:
    """Silencia os avisos que as bibliotecas emitem em volume durante o loop."""
    import warnings

    warnings.filterwarnings("ignore")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    try:
        from sklearn.exceptions import ConvergenceWarning

        warnings.simplefilter("ignore", ConvergenceWarning)
    except Exception:
        pass
