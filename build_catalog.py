#!/usr/bin/env python
"""Monta o catalogo dos conjuntos de dados e aplica os criterios de selecao.

Gera ``descricao/catalogo_experimentos.csv`` (todos os arquivos, com o motivo
de cada exclusao) e ``descricao/datasets_selecionados.csv`` (os selecionados,
em ordem deterministica). Essa ordem e o indice usado por ``--range`` para
dividir o trabalho entre maquinas, entao o arquivo deve ser o mesmo em todas.
"""

import argparse
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-minority", type=int, default=30,
                   help="tamanho minimo da classe minoritaria (padrao: 30)")
    p.add_argument("--duplicate-key", choices=("content", "shape_dist"),
                   default="content",
                   help="como identificar duplicatas: pelo conteudo (padrao) "
                        "ou por tamanho e distribuicao de classes")
    p.add_argument("--max-rows", type=int, default=None,
                   help="subamostra estratificada de conjuntos muito grandes")
    p.add_argument("--methods-table", action="store_true",
                   help="grava tambem a tabela de cobertura de metodos")
    args = p.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    from expq.catalog import build
    from expq.config import Config

    cfg = Config(min_minority=args.min_minority, duplicate_key=args.duplicate_key,
                 max_rows=args.max_rows)
    build(cfg)
    print(f"\ncatalogo completo : {cfg.full_catalog}")
    print(f"selecionados      : {cfg.selected_catalog}")

    if args.methods_table:
        from expq.registry import coverage_table

        path = cfg.catalog_dir / "cobertura_metodos.csv"
        coverage_table().to_csv(path, index=False, encoding="utf-8")
        print(f"cobertura         : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
