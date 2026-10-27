#!/usr/bin/env python
"""Conta linhas de codigo e configuracoes manuais dos scripts da questao Q3.

Formata os scripts com o Black (limite de 88 caracteres) antes de contar,
para que a medida nao dependa do estilo de escrita, e grava o resultado em
``descricao/q3_linhas_de_codigo.csv``.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts" / "q3_usage"


def main() -> int:
    if "--no-format" not in sys.argv:
        try:
            subprocess.run([sys.executable, "-m", "black", "--line-length", "88",
                            "--quiet", str(SCRIPTS)], check=True)
        except Exception as exc:
            print(f"aviso: Black nao foi executado ({exc})")

    from expq.config import Config
    from expq.loc_count import run

    cfg = Config()
    out = cfg.catalog_dir / "q3_linhas_de_codigo.csv"
    df = run(SCRIPTS, out)
    print(df.to_string(index=False))
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
