#!/usr/bin/env python
"""Remove do ledger as combinacoes reprovadas por uma amostra isolada.

Ate 27/10/2026 o ``_predict_all`` levantava a excecao na primeira amostra
cuja estimativa nao fosse um vetor de prevalencias valido. Como o protocolo
UPP gera 400 amostras por combinacao, **uma amostra ruim descartava as
outras 399**, todas validas, e a combinacao inteira ia para `erro`.

Isso subestimava a cobertura das bibliotecas afetadas, em especial a propria
mlquantify: os metodos de limiar dela devolvem um vetor nulo em amostras de
10 instancias de problemas multiclasse, em 1 ou 2 das 400 amostras.

A politica agora e descartar so a amostra invalida e reprovar a combinacao
apenas acima de ``MAX_INVALID_FRACTION`` (5%). Este script apaga do ledger
as combinacoes que foram reprovadas pela politica antiga, para que sejam
refeitas pela nova.

    python clean_invalid.py --run-id final --out-dir C:\\Users\\voce\\exp-run
    python clean_invalid.py --run-id final --out-dir ... --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

#: Mensagens de erro que a politica nova nao reprova mais, porque indicam
#: uma estimativa invalida em alguma amostra e nao uma falha do metodo.
MOTIVOS = (
    "vetor de prevalencias com soma nula",
    "vetor de prevalencias com valores nao finitos",
    # QuaPy aborta por conta propria, pelo mesmo motivo
    "some prevalence vectors contain negative numbers",
    "some prevalence vectors do not sum up to",
)

#: Trecho que so aparece na mensagem do ``TooManyInvalidSamples``, a reprovacao
#: da politica nova. Ela repete o motivo mais frequente entre as amostras
#: descartadas, entao casa com ``MOTIVOS`` -- mas essa combinacao passou do
#: limite de 5% e foi reprovada com razao: refazer daria o mesmo resultado.
JA_PELA_POLITICA_NOVA = "amostras com estimativa invalida"


def afetada(entry: dict) -> str | None:
    """O motivo, se a combinacao caiu por estimativa invalida de amostra."""
    if entry.get("status") != "erro":
        return None
    msg = str(entry.get("error", ""))
    if JA_PELA_POLITICA_NOVA in msg:
        return None
    for motivo in MOTIVOS:
        if motivo in msg:
            return motivo
    return None


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
    alvo = [(path, n, e, motivo) for path, n, e in entries
            if (motivo := afetada(e))]

    print(f"combinacoes no ledger : {len(entries)}")
    print(f"a refazer             : {len(alvo)}")

    print("\n=== por biblioteca e metodo ===")
    for (lib, met), n in sorted(Counter((e["library"], e["method"])
                                        for _, _, e, _ in alvo).items(),
                                key=lambda kv: (-kv[1], kv[0])):
        print(f"  {n:>4}  {lib:<18} {met}")

    print("\n=== por motivo ===")
    for motivo, n in Counter(m for _, _, _, m in alvo).most_common():
        print(f"  {n:>4}  {motivo}")

    if not args.apply:
        print("\n(simulacao; use --apply para gravar)")
        return 0

    descartar = {id(e) for _, _, e, _ in alvo}
    por_arquivo: dict[Path, list] = {}
    for path, _, e in entries:
        if id(e) not in descartar:
            por_arquivo.setdefault(path, []).append(e)
    for path in {p for p, _, _ in entries}:
        linhas = por_arquivo.get(path, [])
        path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in linhas),
            encoding="utf-8")

    chaves = {(e["dataset_id"], e["library"], e["method"]) for _, _, e, _ in alvo}
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
            lambda r: (r["dataset_id"], r["biblioteca"], r["metodo"]) in chaves, axis=1)
        if mask.any():
            removidas += int(mask.sum())
            df[~mask].to_csv(csv, index=False, encoding="utf-8")

    print(f"\nledger reescrito; {removidas} linhas removidas dos parciais")
    print("rode run_experiments.py de novo para refazer essas combinacoes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
