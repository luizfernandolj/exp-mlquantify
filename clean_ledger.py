#!/usr/bin/env python
"""Remove do ledger as combinacoes medidas durante uma suspensao da maquina.

Quando o computador dorme, o relogio salta horas. Combinacoes que estavam
sendo medidas nesse momento ficam com tempos sem sentido, e o escalonador
antigo ainda as encerrava como se tivessem estourado o tempo limite.

Este script encontra as suspensoes pelos intervalos sem atividade no
ledger, marca como contaminada toda combinacao cujo periodo de execucao
cruza uma suspensao (alem das encerradas por tempo limite), e as remove do
ledger e dos arquivos parciais. Na proxima execucao elas serao refeitas; o
que estava limpo e preservado.

    python clean_ledger.py --run-id final --out-dir C:\\Users\\voce\\exp-run
    python clean_ledger.py --run-id final --out-dir ... --apply

O detector de lacunas confunde um metodo longo, que passa meia hora sem
gravar nada, com uma suspensao. Quando o log de energia do Windows da as
suspensoes reais, passe-as com ``--janela INICIO FIM`` (repetivel), e so
elas contam. ``--refazer-metodo`` e ``--refazer-dataset`` removem tambem
combinacoes limpas, para refaze-las depois de mudar o codigo ou quando a
maquina mediu em condicoes diferentes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

#: Intervalo sem nenhuma combinacao registrada que caracteriza suspensao.
GAP = timedelta(minutes=5)

#: Situacoes que, sozinhas, condenam a combinacao.
BAD_STATUS = {"timeout", "timeout_job"}


def read_ledger(ledger_dir: Path):
    entries = []
    for path in sorted(ledger_dir.glob("*/*.jsonl")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append((path, n, json.loads(line)))
            except json.JSONDecodeError:
                continue
    return entries


def find_gaps(entries):
    """Janelas sem atividade, que correspondem as suspensoes da maquina."""
    stamps = sorted(datetime.fromisoformat(e["ts"]) for _, _, e in entries if "ts" in e)
    gaps = []
    for a, b in zip(stamps, stamps[1:]):
        if b - a > GAP:
            gaps.append((a, b))
    return gaps


def is_contaminated(entry, gaps, metodos=(), datasets=()):
    """A combinacao cruzou uma suspensao, ou foi encerrada por tempo limite."""
    if entry.get("method") in metodos:
        return f"refazer o metodo {entry['method']}"
    nome = Path(entry.get("dataset_id", "")).name
    for d in datasets:
        if nome.startswith(d):
            return f"refazer o conjunto {d}"
    if entry.get("status") in BAD_STATUS:
        return "encerrada por tempo limite falso"
    ts = entry.get("ts")
    if not ts:
        return None
    end = datetime.fromisoformat(ts)
    start = end - timedelta(seconds=float(entry.get("seconds", 0) or 0))
    for a, b in gaps:
        if start < b and end > a:  # a execucao cruza a janela de suspensao
            return f"medida cruzou a suspensao de {a:%d/%m %H:%M} a {b:%d/%m %H:%M}"
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="final")
    p.add_argument("--out-dir", default=None,
                   help="pasta usada na execucao, se fora do repositorio")
    p.add_argument("--janela", nargs=2, action="append", metavar=("INICIO", "FIM"),
                   help="suspensao real, em ISO (2026-11-12T15:25:00); repetivel. "
                        "Se dada, substitui a deteccao por lacunas")
    p.add_argument("--refazer-metodo", action="append", default=[],
                   help="remove todas as combinacoes deste metodo; repetivel")
    p.add_argument("--refazer-dataset", action="append", default=[],
                   help="remove as combinacoes do conjunto cujo nome comeca "
                        "assim; repetivel")
    p.add_argument("--apply", action="store_true",
                   help="grava as mudancas; sem isso apenas mostra o que faria")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from expq.config import Config

    cfg = Config(run_id=args.run_id)
    if args.out_dir:
        cfg.set_out_dir(args.out_dir)

    ledger_root = cfg.run_dir / "ledger"
    if not ledger_root.exists():
        print(f"ledger nao encontrado em {ledger_root}")
        return 1

    entries = read_ledger(ledger_root)
    if args.janela:
        gaps = [(datetime.fromisoformat(a), datetime.fromisoformat(b))
                for a, b in args.janela]
    else:
        gaps = find_gaps(entries)

    print(f"combinacoes no ledger: {len(entries)}")
    origem = "informadas" if args.janela else "detectadas"
    print(f"\nsuspensoes {origem}: {len(gaps)}")
    for a, b in gaps:
        print(f"  {a:%d/%m %H:%M:%S} -> {b:%d/%m %H:%M:%S}   "
              f"({(b - a).total_seconds() / 3600:.1f} h parado)")

    bad, keep = [], []
    for path, n, e in entries:
        reason = is_contaminated(e, gaps, args.refazer_metodo, args.refazer_dataset)
        (bad if reason else keep).append((path, n, e, reason))

    print(f"\ncontaminadas : {len(bad)}")
    print(f"preservadas  : {len(keep)}")

    by_reason = {}
    for _, _, _, reason in bad:
        by_reason[reason] = by_reason.get(reason, 0) + 1
    for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {reason}")

    if not args.apply:
        print("\n(simulacao; use --apply para gravar)")
        return 0

    # 1) reescreve o ledger apenas com as entradas preservadas
    por_arquivo: dict[Path, list] = {}
    for path, _, e, _ in keep:
        por_arquivo.setdefault(path, []).append(e)
    for path in {p for p, _, _ in entries}:
        linhas = por_arquivo.get(path, [])
        path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in linhas),
            encoding="utf-8")

    # 2) remove dos parciais as linhas das combinacoes contaminadas
    alvo = {(e["dataset_id"], e["library"], e["method"])
            for _, _, e, _ in bad if "method" in e}
    import pandas as pd

    removidas = 0
    for csv in sorted((cfg.run_dir / "parts").rglob("*.csv")):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty or "metodo" not in df.columns:
            continue
        mask = df.apply(
            lambda r: (r["dataset_id"], r["biblioteca"], r["metodo"]) in alvo, axis=1)
        if mask.any():
            removidas += int(mask.sum())
            df[~mask].to_csv(csv, index=False, encoding="utf-8")

    print(f"\nledger reescrito; {removidas} linhas removidas dos parciais")
    print("as combinacoes removidas serao refeitas na proxima execucao")
    return 0


if __name__ == "__main__":
    sys.exit(main())
