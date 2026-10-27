#!/usr/bin/env python
"""Reune os resultados parciais e produz as tabelas finais.

Para juntar varias maquinas, copie as pastas ``results/<run_id>/parts/<node>``
e ``results/<run_id>/ledger/<node>`` de cada uma para a mesma arvore e rode
este script uma unica vez.
"""

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="run01")
    p.add_argument("--out-dir", default=None, metavar="PASTA",
                   help="pasta usada na execucao, se diferente do repositorio")
    p.add_argument("--no-samples", action="store_true",
                   help="nao grava o per_sample.csv (util quando fica grande)")
    args = p.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    from expq.aggregate import run
    from expq.config import Config
    from expq.ledger import status_report

    cfg = Config(run_id=args.run_id)
    if args.out_dir:
        cfg.set_out_dir(args.out_dir)
    run(cfg, write_samples=not args.no_samples)

    import json

    print("\nandamento no ledger:")
    print(json.dumps(status_report(cfg), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
