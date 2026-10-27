"""Medida de esforco de uso (Q3): linhas de codigo e configuracoes manuais.

Em ``scripts/q3_usage/`` ha um script por biblioteca que conduz o mesmo
experimento sobre um conjunto de dados, do carregamento ao calculo do erro,
seguindo apenas a documentacao oficial de cada uma. Todos sao formatados
pelo Black com limite de 88 caracteres, para que a contagem nao dependa do
estilo de escrita.

As linhas sao contadas pelo modulo ``tokenize``, desconsiderando linhas em
branco, comentarios e docstrings. As configuracoes manuais sao as instrucoes
que existem apenas para adaptar o metodo ou os dados ao formato da
biblioteca, e nao ao fluxo comum (leitura, divisao, treinamento, geracao de
amostras e estimacao); no script elas sao marcadas com ``# MANUAL:``.
"""

from __future__ import annotations

import io
import tokenize
from pathlib import Path

import pandas as pd


def count_logical_lines(source: str) -> int:
    """Linhas logicas de codigo, sem linhas em branco, comentarios ou docstrings."""
    lines: set[int] = set()
    prev_type = tokenize.INDENT
    readline = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                        tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER,
                        tokenize.ENCODING):
            prev_type = tok.type
            continue
        if tok.type == tokenize.STRING and prev_type in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT):
            prev_type = tok.type  # docstring solta: nao conta
            continue
        lines.update(range(tok.start[0], tok.end[0] + 1))
        prev_type = tok.type
    return len(lines)


def count_manual(source: str) -> int:
    """Instrucoes marcadas como configuracao manual."""
    return sum(1 for line in source.splitlines() if "# MANUAL:" in line)


def analyse(path: Path) -> dict:
    source = path.read_text(encoding="utf-8")
    return {
        "arquivo": path.name,
        "linhas_de_codigo": count_logical_lines(source),
        "configuracoes_manuais": count_manual(source),
        "linhas_totais": len(source.splitlines()),
    }


def run(scripts_dir: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(scripts_dir.glob("*.py")):
        library = path.stem.replace("_script", "")
        rows.append({"biblioteca": library, **analyse(path)})
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return df
