#!/usr/bin/env python
"""Realinha os parciais gravados antes de uma coluna nova existir.

O ``_append_rows`` grava o cabecalho so quando cria o arquivo. Quando uma
coluna e acrescentada ao `summary.csv` com a execucao em andamento, o parcial
de uma combinacao medida antes continua com o cabecalho antigo, e as linhas
novas entram com mais campos do que ele declara. O arquivo fica com linhas de
dois tamanhos: o pandas recusa a leitura e a agregacao final para.

Os valores nao se perderam -- o numero de campos da linha diz sob qual
esquema ela foi gravada. Este script le cada linha pela posicao, escolhe o
esquema pelo tamanho e reescreve o arquivo com as colunas atuais. O
``_realign`` do runner faz o mesmo quando uma combinacao nova e acrescentada
ao parcial; este script serve para os parciais que ja estao completos e que
nenhum trabalhador vai tocar de novo.

Para nao disputar o arquivo com um trabalhador vivo, o script deixa de lado
os parciais cuja combinacao ainda tem metodo pendente no ledger.

    python clean_schema.py --run-id final --out-dir C:\\Users\\voce\\exp-run
    python clean_schema.py --run-id final --out-dir ... --apply
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path


def tamanhos(path: Path) -> tuple[int, Counter]:
    """Tamanho do cabecalho e contagem dos tamanhos das linhas."""
    with path.open(newline="", encoding="utf-8") as fh:
        linhas = list(csv.reader(fh))
    if not linhas:
        return 0, Counter()
    return len(linhas[0]), Counter(len(l) for l in linhas[1:])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="final")
    p.add_argument("--out-dir", default=None,
                   help="pasta usada na execucao, se fora do repositorio")
    p.add_argument("--apply", action="store_true",
                   help="grava as mudancas; sem isso apenas mostra o que faria")
    p.add_argument("--todos", action="store_true",
                   help="reescreve tambem os parciais de cabecalho antigo mas "
                        "coerente, que o pandas le sem erro")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from expq.config import Config
    from expq.ledger import load_done
    from expq.catalog import load_selected
    from expq.registry import specs_for
    from expq.runner import (SUMMARY_ANTERIORES, SUMMARY_COLUMNS, _esquemas,
                             _realign)

    cfg = Config(run_id=args.run_id)
    if args.out_dir:
        cfg.set_out_dir(args.out_dir)

    parts = cfg.run_dir / "parts"
    if not parts.exists():
        print(f"parciais nao encontrados em {parts}")
        return 1

    esquemas = _esquemas(SUMMARY_COLUMNS, SUMMARY_ANTERIORES)
    print(f"esquemas conhecidos do summary: {sorted(esquemas)} colunas")

    # combinacoes que ainda faltam: o parcial delas pode receber linha nova
    done = load_done(cfg)
    ocupados = set()
    for row in load_selected(cfg).to_dict("records"):
        for lib in cfg.libraries:
            for s in specs_for(lib, groups=cfg.groups, methods=cfg.methods,
                               skip=cfg.skip_methods):
                if f"{row['dataset_id']}|{lib}|{s.canonical}" not in done:
                    ocupados.add((row["dataset_id"], lib))

    def combinacao(path: Path) -> tuple[str, str]:
        """Conjunto e biblioteca a que o parcial pertence, pelo seu nome."""
        with path.open(newline="", encoding="utf-8") as fh:
            linha = next(csv.DictReader(fh), {})
        return linha.get("dataset_id", ""), linha.get("biblioteca", "")

    alvo, pulados, desconhecidos, antigos = [], [], [], []
    for path in sorted(parts.rglob("*.summary.csv")):
        n_cab, linhas = tamanhos(path)
        if not linhas:
            continue
        fora = {t: n for t, n in linhas.items() if t not in esquemas}
        if n_cab not in esquemas or fora:
            desconhecidos.append((path, n_cab, dict(linhas)))
            continue
        # o pandas so recusa o arquivo cuja linha nao bate com o cabecalho;
        # cabecalho antigo e coerente le sem erro, com as colunas novas vazias
        misto = set(linhas) != {n_cab}
        if not misto and n_cab == len(SUMMARY_COLUMNS):
            continue
        if not misto and not args.todos:
            antigos.append((path, n_cab, dict(linhas)))
        elif combinacao(path) in ocupados:
            pulados.append((path, n_cab, dict(linhas)))
        else:
            alvo.append((path, n_cab, dict(linhas)))

    print(f"\nparciais a realinhar        : {len(alvo)}")
    print(f"parciais com job pendente   : {len(pulados)} (deixados para o runner)")
    print(f"parciais fora dos esquemas  : {len(desconhecidos)}")
    if antigos:
        print(f"parciais antigos coerentes  : {len(antigos)} (o pandas le; use "
              f"--todos para reescrever)")

    for titulo, lista in (("a realinhar", alvo), ("com job pendente", pulados),
                          ("fora dos esquemas", desconhecidos)):
        if not lista:
            continue
        print(f"\n=== {titulo} ===")
        for path, n_cab, linhas in lista[:8]:
            print(f"  {path.name[:70]}")
            print(f"     cabecalho {n_cab} colunas | linhas {linhas}")
        if len(lista) > 8:
            print(f"  ... e outros {len(lista) - 8}")

    if desconhecidos:
        print("\nATENCAO: ha linha de tamanho que nenhum esquema explica.")
        print("Registre a coluna nova em SUMMARY_ANTERIORES antes de continuar.")
        return 2

    if not args.apply:
        print("\n(simulacao; use --apply para gravar)")
        return 0

    linhas_antes = sum(sum(l.values()) for _, _, l in alvo)
    for path, _, _ in alvo:
        _realign(path, SUMMARY_COLUMNS, SUMMARY_ANTERIORES)
    linhas_depois = sum(sum(tamanhos(p)[1].values()) for p, _, _ in alvo)
    print(f"\n{len(alvo)} parciais reescritos | linhas antes {linhas_antes}, "
          f"depois {linhas_depois}")
    if linhas_antes != linhas_depois:
        print("ATENCAO: o numero de linhas mudou; confira os arquivos.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
