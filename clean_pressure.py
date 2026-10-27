#!/usr/bin/env python
"""Remove do ledger as falhas de alocacao causadas pela maquina, nao pelo metodo.

Em 27/10/2026 a execucao chegou a cauda do catalogo, onde estao os conjuntos
maiores, com 14 trabalhadores em uma maquina de 15,7 GB. A memoria acabou e o
escalonador morreu com ``WinError 1450`` (recursos insuficientes do sistema).

No meio disso dois tipos bem diferentes de ``MemoryError`` foram registrados:

* Alocacoes enormes -- dezenas de GiB -- que nenhuma maquina razoavel
  atenderia. Vem de metodos que montam matrizes de distancia entre todas as
  instancias (EDx, EDy, KDEy, HDx) sobre conjuntos de dezenas de milhares de
  instancias. Essas falhas sao uma limitacao real do metodo e ficam.
* Alocacoes de poucos MiB que falharam apenas porque nao havia mais memoria
  livre no momento. Nao dizem nada sobre o metodo e tem de ser refeitas.

O corte entre os dois casos e o tamanho da alocacao: o maior caso de pressao
observado pediu 45,8 MiB e o menor caso legitimo pediu 3,64 GiB, entao
qualquer limite nessa faixa separa os dois. O padrao e 256 MiB.

    python clean_pressure.py --run-id final --out-dir C:\\Users\\voce\\exp-run
    python clean_pressure.py --run-id final --out-dir ... --apply
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Alocacoes abaixo deste tamanho, em MiB, so podem ter falhado por falta de
#: memoria livre na maquina.
LIMITE_MIB = 256.0

UNIDADES = {"KiB": 1 / 1024, "MiB": 1.0, "GiB": 1024.0, "TiB": 1024.0 * 1024}

ALOCACAO = re.compile(r"Unable to allocate ([\d.]+) (KiB|MiB|GiB|TiB)")


def tamanho_pedido(mensagem: str) -> float | None:
    """Tamanho da alocacao que falhou, em MiB, ou ``None`` se nao for isso."""
    m = ALOCACAO.search(mensagem or "")
    if not m:
        return None
    return float(m.group(1)) * UNIDADES[m.group(2)]


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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="final")
    p.add_argument("--out-dir", default=None,
                   help="pasta usada na execucao, se fora do repositorio")
    p.add_argument("--limite-mib", type=float, default=LIMITE_MIB,
                   help=f"alocacoes menores que isto sao pressao (padrao {LIMITE_MIB:.0f})")
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
    pressao, legitimas = [], []
    for path, n, e in entries:
        if e.get("status") != "erro":
            continue
        mib = tamanho_pedido(str(e.get("error", "")))
        if mib is None:
            continue
        (pressao if mib < args.limite_mib else legitimas).append((path, n, e, mib))

    print(f"combinacoes no ledger      : {len(entries)}")
    print(f"falhas de alocacao         : {len(pressao) + len(legitimas)}")
    print(f"  pressao de memoria       : {len(pressao)}  (serao refeitas)")
    print(f"  limitacao do metodo      : {len(legitimas)}  (ficam no ledger)")

    print("\n=== a refazer (a maquina ficou sem memoria) ===")
    for _, _, e, mib in sorted(pressao, key=lambda t: (t[2]["dataset_id"], t[2]["library"])):
        print(f"  {mib:>8.1f} MiB  {e['dataset_id'].split('/')[-1][:38]:<38} "
              f"{e['library']:<18} {e['method']}")

    print("\n=== preservadas (o metodo pediu o impossivel) ===")
    for _, _, e, mib in sorted(legitimas, key=lambda t: -t[3]):
        print(f"  {mib / 1024:>8.1f} GiB  {e['dataset_id'].split('/')[-1][:38]:<38} "
              f"{e['library']:<18} {e['method']}")

    if not args.apply:
        print("\n(simulacao; use --apply para gravar)")
        return 0

    descartar = {id(e) for _, _, e, _ in pressao}
    por_arquivo: dict[Path, list] = {}
    for path, _, e in entries:
        if id(e) not in descartar:
            por_arquivo.setdefault(path, []).append(e)
    for path in {p for p, _, _ in entries}:
        linhas = por_arquivo.get(path, [])
        path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in linhas),
            encoding="utf-8")

    alvo = {(e["dataset_id"], e["library"], e["method"]) for _, _, e, _ in pressao}
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
