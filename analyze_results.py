#!/usr/bin/env python
"""Responde Q1, Q2 e Q3 a partir das tabelas agregadas.

Le o que o ``aggregate_results.py`` gravou e escreve, em
``results/<run-id>/analise/``, as tabelas pequenas que entram no texto (em
csv, md e tex), as figuras e um ``respostas.md`` com os numeros no lugar.

Pode rodar com a execucao em andamento: o cabecalho das saidas diz quantas
combinacoes das planejadas entraram na conta. Rode de novo ao final para
atualizar tudo.

    python aggregate_results.py --run-id final --no-samples   # antes
    python analyze_results.py --run-id final
    python analyze_results.py --run-id final --erro mrae      # a mesma analise em RAE
"""

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="final")
    p.add_argument("--out-dir", default=None, metavar="PASTA",
                   help="pasta usada na execucao, se fora do repositorio")
    p.add_argument("--erro", default="mae",
                   choices=("mae", "mrae", "mse", "nkld_medio"),
                   help="medida de erro usada em Q1 (padrao: mae)")
    p.add_argument("--no-figures", action="store_true",
                   help="nao gera as figuras, so as tabelas")
    args = p.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    from expq.analysis import run
    from expq.config import Config

    cfg = Config(run_id=args.run_id)
    if args.out_dir:
        cfg.set_out_dir(args.out_dir)
    run(cfg, com_figuras=not args.no_figures, erro=args.erro)
    return 0


if __name__ == "__main__":
    sys.exit(main())
