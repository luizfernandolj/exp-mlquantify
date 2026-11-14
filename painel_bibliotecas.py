#!/usr/bin/env python
"""Painel de comparacao: a mlquantify contra cada outra biblioteca, por metodo.

Toda linha deste painel e um par: um metodo da mlquantify contra o mesmo
metodo de outra biblioteca, no mesmo conjunto e no mesmo tamanho de amostra.
Nunca compara biblioteca com biblioteca em geral, nem metodo com metodo
diferente -- a pergunta e sempre "neste metodo, a minha ganha da dela?".

O pareamento usa o ``expand_aliases`` do ``expq.aggregate``, o mesmo da
analise oficial, para que ``PACC``/``GPACC`` e ``ACC``/``GACC`` casem como
casam lah, sem duplicar par.

Escreve ``painel_bibliotecas.html`` (um arquivo, sem dependencia externa) e
imprime as mesmas tabelas no terminal.

    python painel_bibliotecas.py --run-id final-pareado \
        --out-dir C:/Users/luiz_/exp-mlquantify-run
    python painel_bibliotecas.py --run-id final-pareado --custo-run custo-final5
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MINHA = "mlquantify"

#: (coluna, rotulo curto, rotulo longo). Em todas, menor e melhor.
CUSTOS = [
    ("tempo_treino_s", "treino", "tempo de treinamento"),
    ("tempo_estimacao_total_s", "estimacao", "tempo de estimacao"),
    ("memoria_treino_mb", "mem. treino", "memoria no treinamento"),
    ("memoria_estimacao_mb", "mem. estim.", "memoria na estimacao"),
]

CHAVE = ["dataset_id", "metodo", "tamanho_amostra"]

#: Abaixo disto o erro e tratado como identico (ruido de ponto flutuante nao
#: chega perto de 1e-3; essa e a mesma faixa que o q1_erro.csv usa).
TOL_EMPATE = 1e-3


# ---------------------------------------------------------------------------
# leitura e pareamento
# ---------------------------------------------------------------------------

def carregar(run_dir: Path) -> pd.DataFrame:
    caminho = run_dir / "summary.csv"
    if not caminho.exists():
        raise SystemExit(f"nao achei {caminho}")
    df = pd.read_csv(caminho, low_memory=False)
    from expq.aggregate import expand_aliases

    df = expand_aliases(df)
    return df[df["status"] == "ok"].copy()


def parear(ok: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (par de medicao, biblioteca rival).

    Cada linha traz as colunas da mlquantify com sufixo ``_minha`` e as da
    rival com ``_dela``.
    """
    colunas = CHAVE + ["biblioteca", "n_classes", "n_atributos", "n_instancias",
                       "mae", "mrae"] + [c for c, _, _ in CUSTOS]
    colunas = [c for c in colunas if c in ok.columns]
    base = ok[colunas].copy()
    minha = base[base["biblioteca"] == MINHA].drop(columns="biblioteca")
    outras = base[base["biblioteca"] != MINHA]
    if minha.empty or outras.empty:
        raise SystemExit("faltam linhas da mlquantify ou das outras bibliotecas")
    # a mesma combinacao pode aparecer repetida (execucao reiniciada); fica a
    # mediana, que e o que o aggregate tambem faz
    numericas = [c for c in minha.columns if c not in CHAVE]
    minha = minha.groupby(CHAVE, as_index=False)[numericas].median()
    pares = outras.merge(minha, on=CHAVE, suffixes=("_dela", "_minha"))
    return pares


# ---------------------------------------------------------------------------
# resumo por metodo e biblioteca
# ---------------------------------------------------------------------------

def resumir(pares: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for (metodo, lib), sub in pares.groupby(["metodo", "biblioteca"]):
        linha = {"metodo": metodo, "biblioteca": lib, "n_pares": len(sub),
                 "n_conjuntos": sub["dataset_id"].nunique()}
        dif = sub["mae_minha"] - sub["mae_dela"]
        linha["mae_minha"] = sub["mae_minha"].mean()
        linha["mae_dela"] = sub["mae_dela"].mean()
        linha["dif_mae"] = dif.mean()
        linha["dif_mae_max"] = dif.abs().max()
        linha["empate_erro"] = float((dif.abs() < TOL_EMPATE).mean())
        linha["venci_erro"] = float((dif < -TOL_EMPATE).mean())
        linha["perdi_erro"] = float((dif > TOL_EMPATE).mean())
        if "mrae_minha" in sub:
            linha["mrae_minha"] = sub["mrae_minha"].mean()
            linha["mrae_dela"] = sub["mrae_dela"].mean()
        for col, curto, _ in CUSTOS:
            a, b = f"{col}_minha", f"{col}_dela"
            if a not in sub or b not in sub:
                continue
            valido = sub[(sub[b] > 0) & sub[a].notna() & sub[b].notna()]
            if valido.empty:
                linha[f"razao_{curto}"] = np.nan
                linha[f"venci_{curto}"] = np.nan
                continue
            razao = valido[a] / valido[b]
            linha[f"razao_{curto}"] = float(razao.median())
            linha[f"venci_{curto}"] = float((razao < 1.0).mean())
        linhas.append(linha)
    out = pd.DataFrame(linhas)
    return out.sort_values(["metodo", "biblioteca"]).reset_index(drop=True)


def veredito(linha: pd.Series) -> str:
    """Uma palavra por par: onde a mlquantify esta nesse metodo."""
    erro_ok = abs(linha["dif_mae"]) < TOL_EMPATE
    pior_erro = linha["dif_mae"] >= TOL_EMPATE
    r = linha.get("razao_estimacao", np.nan)
    caro = (r == r) and r > 1.10
    barato = (r == r) and r < 0.90
    if pior_erro:
        return "erro pior"
    if caro:
        return "mais lenta" if erro_ok else "mais lenta, erro melhor"
    if barato:
        return "ganha" if erro_ok else "mais rapida, erro melhor"
    return "empata" if erro_ok else "erro melhor"


def _classificar(res: pd.DataFrame):
    """Separa os pares em custa-mais, erra-mais e ganha-nos-dois.

    Um par pode estar nas duas listas de perda; as duas perguntas sao
    independentes, e juntar as duas em um numero so esconderia qual delas
    esta valendo.
    """
    caro = res["razao_estimacao"] > 1.10
    errada = res["dif_mae"] > TOL_EMPATE
    ganha = (res["razao_estimacao"] < 0.90) & (res["dif_mae"] <= TOL_EMPATE)
    return res[caro], res[errada], res[ganha]


# ---------------------------------------------------------------------------
# saida no terminal
# ---------------------------------------------------------------------------

def _pct(x) -> str:
    return "  -  " if x != x else f"{100 * x:4.0f}%"


def _num(x, casas=3) -> str:
    return "   -  " if x != x else f"{x:.{casas}f}"


def imprimir(resumo: pd.DataFrame, pares: pd.DataFrame) -> None:
    libs = sorted(resumo["biblioteca"].unique())
    print()
    print("=" * 78)
    print("A MLQUANTIFY CONTRA CADA OUTRA BIBLIOTECA, METODO POR METODO")
    print("=" * 78)
    print(f"{len(pares):,} pares de medicao  |  "
          f"{resumo['metodo'].nunique()} metodos  |  rivais: {', '.join(libs)}")
    print("razao < 1 = a mlquantify custa menos; dif_mae < 0 = erra menos")

    for lib in libs:
        sub = resumo[resumo["biblioteca"] == lib].copy()
        sub["veredito"] = sub.apply(veredito, axis=1)
        print()
        print("-" * 78)
        print(f"contra {lib}  ({len(sub)} metodos em comum)")
        print("-" * 78)
        print(f"{'metodo':<9}{'pares':>6}{'mae minha':>11}{'mae dela':>10}"
              f"{'dif':>9}{'=erro':>7}{'treino':>8}{'estim':>8}{'mem.es':>8}"
              f"  veredito")
        for _, r in sub.sort_values("razao_estimacao", ascending=False).iterrows():
            print(f"{r['metodo']:<9}{r['n_pares']:>6}"
                  f"{_num(r['mae_minha'], 4):>11}{_num(r['mae_dela'], 4):>10}"
                  f"{r['dif_mae']:>9.1e}{_pct(r['empate_erro']):>7}"
                  f"{_num(r.get('razao_treino'), 2):>8}"
                  f"{_num(r.get('razao_estimacao'), 2):>8}"
                  f"{_num(r.get('razao_mem. estim.'), 2):>8}"
                  f"  {r['veredito']}")

    res = resumo.copy()
    res["veredito"] = res.apply(veredito, axis=1)
    caras, erradas, ganhos = _classificar(res)

    print()
    print("=" * 78)
    print("ONDE A MLQUANTIFY AINDA PERDE  (o que vale atacar)")
    print("=" * 78)
    print(f"-- custa mais que a outra na estimacao ({len(caras)})")
    for _, r in caras.sort_values("razao_estimacao", ascending=False).iterrows():
        print(f"  {r['metodo']:<8} x {r['biblioteca']:<18} "
              f"estimacao {_num(r.get('razao_estimacao'), 2)}x  "
              f"treino {_num(r.get('razao_treino'), 2)}x  "
              f"dif_mae {r['dif_mae']:+.1e}")
    if caras.empty:
        print("  nenhum")
    print(f"-- erra mais que a outra ({len(erradas)})")
    for _, r in erradas.sort_values("dif_mae", ascending=False).iterrows():
        print(f"  {r['metodo']:<8} x {r['biblioteca']:<18} "
              f"dif_mae {r['dif_mae']:+.1e}  max {r['dif_mae_max']:.1e}  "
              f"estimacao {_num(r.get('razao_estimacao'), 2)}x")
    if erradas.empty:
        print("  nenhum")

    print()
    print(f"em {len(res)} pares metodo x biblioteca: ganha em {len(ganhos)}, "
          f"erra mais em {len(erradas)}, custa mais em {len(caras)}, "
          f"o resto empata")


# ---------------------------------------------------------------------------
# compatibilidade de erro: teste de equivalencia pareado
# ---------------------------------------------------------------------------
#
# A pergunta nao e "qual biblioteca e a melhor" -- ranking nao responde
# compatibilidade, e um ranking sempre ordena, mesmo quando a diferenca e
# irrelevante. A pergunta e "no metodo X, o erro da mlquantify e o mesmo da
# biblioteca Y?". Isso e uma hipotese de equivalencia, e o teste proprio
# para ela e o TOST (two one-sided tests): a equivalencia e que vai para a
# hipotese alternativa, entao um p pequeno **afirma** que as duas erram
# igual, dentro de uma margem declarada. Um teste comum de diferenca nunca
# afirma isso: nao rejeitar e so falta de evidencia, que com n pequeno
# acontece mesmo quando a diferenca e grande.
#
# Desenho:
#   * unidade = conjunto de dados. O erro de um conjunto nos quatro tamanhos
#     de amostra nao sao observacoes independentes, entao entram pela media
#     e cada conjunto conta uma vez.
#   * pareado: a diferenca e calculada dentro do conjunto, o que remove a
#     dificuldade do conjunto da conta.
#   * sem suposicao de normalidade: Wilcoxon dos postos sinalizados, com o
#     estimador de Hodges-Lehmann e seu intervalo de confianca, que e o
#     intervalo que acompanha esse teste.
#   * multiplicidade: Holm sobre todos os pares metodo x biblioteca, nos
#     dois testes.

#: Margem de equivalencia em AE. Um ponto percentual de prevalencia: abaixo
#: disso a diferenca nao muda decisao nenhuma tomada com a estimativa.
MARGEM = 0.01


def _walsh(d: np.ndarray) -> np.ndarray:
    """Medias de Walsh, (d_i + d_j)/2 com i <= j."""
    soma = d[:, None] + d[None, :]
    i, j = np.triu_indices(len(d))
    return np.sort(soma[i, j] / 2.0)


def hodges_lehmann(d: np.ndarray) -> tuple[float, float, float]:
    """Estimador de Hodges-Lehmann da diferenca mediana e seu IC de 95%.

    O IC vem das medias de Walsh ordenadas, cortadas no valor critico de
    Wilcoxon (aproximacao normal com correcao de continuidade, valida no n
    desta analise, que passa de 100 conjuntos).
    """
    from scipy.stats import norm

    n = len(d)
    w = _walsh(d)
    hl = float(np.median(w))
    if n < 6:
        return hl, np.nan, np.nan
    z = norm.ppf(0.975)
    k = n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24) - 0.5
    k = int(np.floor(k))
    if k < 0:
        return hl, float(w[0]), float(w[-1])
    k = min(k, len(w) - 1)
    return hl, float(w[k]), float(w[len(w) - 1 - k])


def _holm(ps: np.ndarray) -> np.ndarray:
    """Holm (1979): controla a familia de testes sem supor independencia."""
    ps = np.asarray(ps, dtype=float)
    ordem = np.argsort(ps)
    m = len(ps)
    ajust = np.empty(m)
    corrente = 0.0
    for posicao, idx in enumerate(ordem):
        corrente = max(corrente, (m - posicao) * ps[idx])
        ajust[idx] = min(1.0, corrente)
    return ajust


def diferencas(pares: pd.DataFrame, erro: str = "mae") -> pd.DataFrame:
    """Uma diferenca de erro por (metodo, biblioteca, conjunto)."""
    por_conjunto = pares.groupby(
        ["metodo", "biblioteca", "dataset_id"], as_index=False
    )[[f"{erro}_minha", f"{erro}_dela"]].mean()
    por_conjunto["dif"] = (por_conjunto[f"{erro}_minha"]
                           - por_conjunto[f"{erro}_dela"])
    return por_conjunto


def testar(pares: pd.DataFrame, erro: str = "mae",
           margem: float = MARGEM, alfa: float = 0.05) -> pd.DataFrame:
    from scipy.stats import wilcoxon

    dif = diferencas(pares, erro)
    linhas = []
    for (metodo, lib), sub in dif.groupby(["metodo", "biblioteca"]):
        d = sub["dif"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if len(d) < 6:
            continue
        hl, baixo, alto = hodges_lehmann(d)
        # diferenca (bilateral). Com todas as diferencas exatamente zero o
        # Wilcoxon nao tem posto nenhum para somar; ai p = 1 por construcao.
        if np.allclose(d, 0.0):
            p_dif = 1.0
            p_tost = 0.0
        else:
            p_dif = float(wilcoxon(d, zero_method="wilcox",
                                   alternative="two-sided").pvalue)
            p_baixo = float(wilcoxon(d + margem, zero_method="wilcox",
                                     alternative="greater").pvalue)
            p_alto = float(wilcoxon(d - margem, zero_method="wilcox",
                                    alternative="less").pvalue)
            p_tost = max(p_baixo, p_alto)
        linhas.append({
            "metodo": metodo, "biblioteca": lib, "n_conjuntos": len(d),
            "hl": hl, "ic_baixo": baixo, "ic_alto": alto,
            "p10": float(np.percentile(d, 2.5)),
            "p90": float(np.percentile(d, 97.5)),
            "dif_max_abs": float(np.abs(d).max()),
            "dif_media": float(d.mean()),
            "p_diferenca": p_dif, "p_equivalencia": p_tost,
        })
    out = pd.DataFrame(linhas)
    if out.empty:
        return out
    out["p_diferenca_holm"] = _holm(out["p_diferenca"])
    out["p_equivalencia_holm"] = _holm(out["p_equivalencia"])
    out["ic_na_faixa"] = ((out["ic_baixo"] > -margem)
                          & (out["ic_alto"] < margem))
    # criterio mais severo: a faixa conter a variabilidade entre conjuntos,
    # e nao so a incerteza da diferenca central
    out["espalhamento_na_faixa"] = ((out["p10"] > -margem)
                                    & (out["p90"] < margem))
    out["tudo_na_faixa"] = out["dif_max_abs"] < margem
    out["veredito"] = [
        _veredito_teste(r, alfa) for _, r in out.iterrows()]
    return out.sort_values("hl", key=lambda s: s.abs(),
                           ascending=False).reset_index(drop=True)


def _veredito_teste(r: pd.Series, alfa: float) -> str:
    equivale = r["p_equivalencia_holm"] < alfa
    difere = r["p_diferenca_holm"] < alfa
    if equivale and difere:
        return "diferenca irrelevante"
    if equivale:
        return "compativel"
    if difere:
        return "diferente"
    return "inconclusivo"


def imprimir_teste(teste: pd.DataFrame, margem: float, erro: str) -> None:
    print()
    print("=" * 86)
    print(f"COMPATIBILIDADE DE {erro.upper()}: EQUIVALENCIA PAREADA POR "
          f"CONJUNTO (TOST, margem {margem:g})")
    print("=" * 86)
    print("HL = diferenca mediana (minha - dela); IC de 95% que acompanha o "
          "Wilcoxon")
    print("p_eq = TOST corrigido por Holm; p_dif = Wilcoxon bilateral "
          "corrigido por Holm")
    print()
    print(f"{'metodo':<9}{'biblioteca':<19}{'n':>4}{'HL':>10}"
          f"{'IC 95%':>22}{'p_eq':>9}{'p_dif':>9}  veredito")
    for _, r in teste.iterrows():
        ic = f"[{r['ic_baixo']:+.4f}, {r['ic_alto']:+.4f}]"
        print(f"{r['metodo']:<9}{r['biblioteca']:<19}{r['n_conjuntos']:>4}"
              f"{r['hl']:>+10.4f}{ic:>22}"
              f"{r['p_equivalencia_holm']:>9.1e}{r['p_diferenca_holm']:>9.1e}"
              f"  {r['veredito']}")
    print()
    contagem = teste["veredito"].value_counts()
    for nome in ("compativel", "diferenca irrelevante", "inconclusivo",
                 "diferente"):
        if nome in contagem:
            print(f"  {nome:<24}{contagem[nome]:>3} de {len(teste)}")
    print(f"\n  dentro de +-{margem:g}:")
    print(f"    o IC de 95% da diferenca   "
          f"{int(teste['ic_na_faixa'].sum()):>3} de {len(teste)} pares")
    print(f"    95% centrais dos conjuntos "
          f"{int(teste['espalhamento_na_faixa'].sum()):>3} de {len(teste)}"
          f"  (criterio severo)")
    print(f"    todos os conjuntos         "
          f"{int(teste['tudo_na_faixa'].sum()):>3} de {len(teste)}"
          f"  (sem nenhuma excecao)")


# ---------------------------------------------------------------------------
# grafico de floresta
# ---------------------------------------------------------------------------

#: Uma cor por biblioteca rival, legiveis em fundo escuro.
CORES = {
    "quapy": "#7aa2f7",
    "quantificationlib": "#e0af68",
    "qunfold": "#9ece6a",
}
COR_EXTRA = ["#bb9af7", "#f7768e", "#7dcfff"]


def figura(teste: pd.DataFrame, margem: float, erro: str,
           destino: Path) -> str:
    """Floresta: um ponto por par, cor por biblioteca, faixa de equivalencia.

    Devolve o svg como texto, para embutir no painel.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fg, fraco, fundo = "#edeae4", "#9a968e", "#1d1c1a"
    libs = sorted(teste["biblioteca"].unique())
    cores = dict(CORES)
    for i, lib in enumerate(l for l in libs if l not in cores):
        cores[lib] = COR_EXTRA[i % len(COR_EXTRA)]

    # metodos ordenados pela maior discrepancia, do topo para baixo
    ordem = (teste.assign(abs_hl=teste["hl"].abs())
             .groupby("metodo")["abs_hl"].max()
             .sort_values(ascending=False).index.tolist())
    passo = 1.0
    desloc = 0.24
    ys, rotulos = {}, []
    y = 0.0
    for metodo in ordem:
        sub = teste[teste["metodo"] == metodo]
        n = len(sub)
        for k, lib in enumerate(sorted(sub["biblioteca"])):
            ys[(metodo, lib)] = y + (k - (n - 1) / 2) * desloc
        rotulos.append((y, metodo))
        y -= passo

    altura = max(4.0, 0.46 * len(ordem) + 2.2)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(10.6, altura), dpi=120, sharey=True,
        gridspec_kw={"width_ratios": [2.1, 1.0], "wspace": 0.06})
    fig.patch.set_facecolor(fundo)

    limite = max(margem * 1.9, float(
        np.nanmax(np.abs(teste[["ic_baixo", "ic_alto"]].to_numpy()))) * 1.25)
    largo = float(np.nanmax(np.abs(teste[["p10", "p90"]].to_numpy()))) * 1.06

    for eixo in (ax, ax2):
        eixo.set_facecolor(fundo)
        eixo.axvspan(-margem, margem, color="#5bbf95", alpha=0.13, lw=0,
                     zorder=0)
        for x in (-margem, margem):
            eixo.axvline(x, color="#5bbf95", alpha=0.5, lw=1.0, ls="--",
                         zorder=1)
        eixo.axvline(0, color=fraco, lw=0.9, zorder=1)
        eixo.tick_params(colors=fraco, labelsize=8.5)
        for lado in ("top", "right", "left"):
            eixo.spines[lado].set_visible(False)
        eixo.spines["bottom"].set_color(fraco)
        eixo.grid(axis="x", color=fg, alpha=0.07, lw=0.8)
        eixo.set_axisbelow(True)

    for _, r in teste.iterrows():
        yy = ys[(r["metodo"], r["biblioteca"])]
        cor = cores[r["biblioteca"]]
        dentro = bool(r["ic_na_faixa"])
        marca = "o" if dentro else "D"
        # esquerda: a regiao da decisao, com o intervalo de confianca
        ax.plot([max(r["p10"], -limite), min(r["p90"], limite)], [yy, yy],
                color=cor, lw=1.0, alpha=0.35, solid_capstyle="butt", zorder=2)
        for valor, seta in ((r["p10"], "<"), (r["p90"], ">")):
            if abs(valor) > limite:
                ax.plot([np.sign(valor) * limite], [yy], marker=seta, ms=3.4,
                        color=cor, alpha=0.5, zorder=2)
        ax.plot([r["ic_baixo"], r["ic_alto"]], [yy, yy], color=cor, lw=3.2,
                solid_capstyle="round", zorder=3)
        ax.plot([r["hl"]], [yy], marker=marca, ms=5.2, color=cor, mec=fundo,
                mew=0.9, zorder=4)
        # direita: a mesma linha em escala cheia, para ver a variabilidade
        ax2.plot([r["p10"], r["p90"]], [yy, yy], color=cor, lw=1.0,
                 alpha=0.45, solid_capstyle="round", zorder=2)
        ax2.plot([r["hl"]], [yy], marker=marca, ms=3.6, color=cor, mec=fundo,
                 mew=0.7, zorder=4)

    ax.set_xlim(-limite, limite)
    ax2.set_xlim(-largo, largo)
    ax.set_yticks([p for p, _ in rotulos])
    ax.set_yticklabels([m for _, m in rotulos], color=fg, fontsize=9)
    ax.set_ylim(y + passo / 2 - 0.3, 0.6)
    ax.set_xlabel(f"diferenca de {erro.upper()} por conjunto  "
                  f"(mlquantify - outra biblioteca)", color=fg, fontsize=9.5)
    ax2.set_xlabel("escala cheia: 95% centrais dos conjuntos",
                   color=fraco, fontsize=8.5)
    ax2.tick_params(axis="y", length=0)
    ax.tick_params(axis="y", length=0)

    legenda = [Line2D([], [], color=cores[l], lw=3, label=l) for l in libs]
    legenda += [
        Line2D([], [], color=fraco, lw=3, label="IC 95% da diferenca"),
        Line2D([], [], color=fraco, lw=1, alpha=0.5,
               label="95% centrais dos conjuntos"),
        Line2D([], [], color=fraco, marker="o", ls="", ms=5,
               label="IC inteiro na faixa"),
        Line2D([], [], color=fraco, marker="D", ls="", ms=5,
               label="IC sai da faixa"),
    ]
    leg = ax.legend(handles=legenda, loc="lower left",
                    bbox_to_anchor=(0.0, 1.015), frameon=False,
                    fontsize=8, ncols=4, labelcolor=fg,
                    columnspacing=1.4, handlelength=1.8)

    ax.set_title(f"Faixa de equivalencia +-{margem:g} de {erro.upper()}",
                 color=fg, fontsize=11, loc="left", pad=44)
    fig.tight_layout()
    fig.savefig(destino, facecolor=fundo)
    svg = destino.with_suffix(".svg")
    fig.savefig(svg, facecolor=fundo, format="svg")
    plt.close(fig)
    texto = svg.read_text(encoding="utf-8")
    return texto[texto.index("<svg"):]


# ---------------------------------------------------------------------------
# painel em html
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#141412;--fg:#edeae4;--fraco:#9a968e;--linha:#2e2c28;--card:#1d1c1a;
--bom:#5bbf95;--ruim:#e0825f;--neutro:#8a8679;--realce:#242321}
:root[data-theme="light"]{--bg:#fbfbfa;--fg:#1a1915;--fraco:#6b6862;
--linha:#e4e1da;--card:#ffffff;--bom:#1a7f5a;--ruim:#b3401f;--neutro:#8a8679;
--realce:#f4f1ea}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.env{max-width:1100px;margin:0 auto;padding:40px 16px 72px}
h1{font-size:1.55rem;margin:0 0 .3rem;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:2.6rem 0 .5rem;letter-spacing:-.01em}
p.sub{color:var(--fraco);margin:.2rem 0 0;max-width:62ch}
.fichas{display:flex;flex-wrap:wrap;gap:10px;margin:22px 0 0}
.ficha{background:var(--card);border:1px solid var(--linha);border-radius:10px;
padding:12px 16px;min-width:130px}
.ficha b{display:block;font-size:1.5rem;font-weight:600;letter-spacing:-.02em}
.ficha span{color:var(--fraco);font-size:.78rem;text-transform:uppercase;
letter-spacing:.06em}
table{width:100%;border-collapse:collapse;font-size:.86rem;
font-variant-numeric:tabular-nums}
th,td{padding:7px 9px;border-bottom:1px solid var(--linha);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{color:var(--fraco);font-weight:500;font-size:.74rem;
text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}
tbody tr:hover{background:var(--realce)}
.bom{color:var(--bom)}.ruim{color:var(--ruim)}.neutro{color:var(--neutro)}
.barra{position:relative;display:block;height:16px;border-radius:3px;
background:var(--realce);overflow:hidden}
.barra i{position:absolute;top:0;bottom:0;left:50%;display:block}
.nota{color:var(--fraco);font-size:.82rem;max-width:72ch;margin:.6rem 0 0}
code{background:var(--realce);padding:1px 5px;border-radius:4px;font-size:.85em}
figure{margin:18px 0 0;background:var(--card);border:1px solid var(--linha);
border-radius:10px;padding:14px 10px 6px;overflow-x:auto}
figure svg{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{color:var(--fraco);font-size:.8rem;padding:8px 8px 2px;max-width:76ch}
@media(max-width:700px){.env{padding:24px 16px 48px}table{font-size:.78rem}}
"""


def _cls_razao(r) -> str:
    if r != r:
        return "neutro"
    if r < 0.9:
        return "bom"
    if r > 1.1:
        return "ruim"
    return "neutro"


def _barra(r) -> str:
    """Barra ancorada em 1: para a esquerda e mais barata, direita mais cara."""
    if r != r:
        return '<span class="neutro">-</span>'
    lg = float(np.clip(np.log2(r), -2, 2)) / 2  # -1..1
    cor = "var(--bom)" if lg < 0 else "var(--ruim)"
    largura = abs(lg) * 48
    borda = "left:50%" if lg >= 0 else f"left:{50 - largura}%"
    return (f'<span class="barra" title="{r:.3f}x">'
            f'<i style="{borda};width:{largura}%;background:{cor}"></i></span>')


def _tabela_teste(teste: pd.DataFrame, margem: float) -> str:
    cabeca = ("<tr><th>metodo</th><th>biblioteca</th><th>conjuntos</th>"
              "<th>dif. mediana</th><th>IC 95%</th><th>80% central</th>"
              "<th>p equivalencia</th><th>p diferenca</th>"
              "<th>veredito</th></tr>")
    corpo = []
    for _, r in teste.iterrows():
        cls = {"compativel": "bom", "diferenca irrelevante": "bom",
               "diferente": "ruim", "inconclusivo": "neutro"}[r["veredito"]]
        corpo.append(
            f"<tr><td><b>{html.escape(str(r['metodo']))}</b></td>"
            f"<td>{html.escape(str(r['biblioteca']))}</td>"
            f"<td>{r['n_conjuntos']}</td>"
            f"<td>{r['hl']:+.4f}</td>"
            f"<td>[{r['ic_baixo']:+.4f}, {r['ic_alto']:+.4f}]</td>"
            f"<td>[{r['p10']:+.3f}, {r['p90']:+.3f}]</td>"
            f"<td>{r['p_equivalencia_holm']:.1e}</td>"
            f"<td>{r['p_diferenca_holm']:.1e}</td>"
            f'<td class="{cls}">{html.escape(r["veredito"])}</td></tr>')
    return (f'<table><thead>{cabeca}</thead><tbody>'
            f'{"".join(corpo)}</tbody></table>')


def html_painel(resumo: pd.DataFrame, pares: pd.DataFrame, run_id: str,
                custo_run: str | None, teste: pd.DataFrame | None = None,
                svg: str | None = None, margem: float = MARGEM,
                erro: str = "mae") -> str:
    res = resumo.copy()
    res["veredito"] = res.apply(veredito, axis=1)
    caras, erradas, ganhos = _classificar(res)
    libs = sorted(res["biblioteca"].unique())

    def fichas() -> str:
        itens = [
            (f"{len(res)}", "pares metodo x lib"),
            (f"{len(ganhos)}", "ganha nos dois"),
            (f"{len(caras)}", "custa mais"),
            (f"{len(erradas)}", "erra mais"),
            (f"{len(pares):,}".replace(",", " "), "pares de medicao"),
        ]
        return "".join(f'<div class="ficha"><b>{v}</b><span>{k}</span></div>'
                       for v, k in itens)

    def tabela(sub: pd.DataFrame) -> str:
        cabeca = ("<tr><th>metodo</th><th>pares</th><th>AE minha</th>"
                  "<th>AE dela</th><th>dif</th><th>erro igual</th>"
                  "<th>treino</th><th>estimacao</th><th></th>"
                  "<th>mem. estim.</th><th>veredito</th></tr>")
        corpo = []
        for _, r in sub.sort_values("razao_estimacao",
                                    ascending=False).iterrows():
            dif = r["dif_mae"]
            cls_dif = ("bom" if dif < -TOL_EMPATE
                       else "ruim" if dif > TOL_EMPATE else "neutro")
            corpo.append(
                f"<tr><td><b>{html.escape(str(r['metodo']))}</b></td>"
                f"<td>{r['n_pares']}</td>"
                f"<td>{_num(r['mae_minha'], 4)}</td>"
                f"<td>{_num(r['mae_dela'], 4)}</td>"
                f'<td class="{cls_dif}">{dif:+.1e}</td>'
                f"<td>{_pct(r['empate_erro']).strip()}</td>"
                f'<td class="{_cls_razao(r.get("razao_treino"))}">'
                f'{_num(r.get("razao_treino"), 2)}</td>'
                f'<td class="{_cls_razao(r.get("razao_estimacao"))}">'
                f'{_num(r.get("razao_estimacao"), 2)}</td>'
                f'<td style="width:110px">{_barra(r.get("razao_estimacao"))}</td>'
                f'<td class="{_cls_razao(r.get("razao_mem. estim."))}">'
                f'{_num(r.get("razao_mem. estim."), 2)}</td>'
                f"<td>{html.escape(r['veredito'])}</td></tr>")
        return (f'<table><thead>{cabeca}</thead><tbody>'
                f'{"".join(corpo)}</tbody></table>')

    secoes = []
    for lib in libs:
        sub = res[res["biblioteca"] == lib]
        secoes.append(
            f"<h2>contra {html.escape(lib)} "
            f'<span class="neutro">({len(sub)} metodos em comum)</span></h2>'
            + tabela(sub))

    def lista(sub: pd.DataFrame, por: str, crescente: bool) -> str:
        if sub.empty:
            return '<p class="nota">Nenhum par.</p>'
        itens = "".join(
            f"<li><b>{html.escape(r['metodo'])}</b> contra "
            f"{html.escape(r['biblioteca'])} &mdash; estimacao "
            f"{_num(r.get('razao_estimacao'), 2)}x, treino "
            f"{_num(r.get('razao_treino'), 2)}x, dif de AE "
            f"{r['dif_mae']:+.1e} (maxima {r['dif_mae_max']:.1e})</li>"
            for _, r in sub.sort_values(por, ascending=crescente).iterrows())
        return f"<ul>{itens}</ul>"

    alvo = (f"<h2>Custa mais que a outra na estimacao "
            f'<span class="neutro">({len(caras)})</span></h2>'
            + lista(caras, "razao_estimacao", False)
            + f"<h2>Erra mais que a outra "
              f'<span class="neutro">({len(erradas)})</span></h2>'
            + lista(erradas, "dif_mae", False))

    bloco_teste = ""
    if teste is not None and not teste.empty:
        contagem = teste["veredito"].value_counts()
        resumo_vereditos = ", ".join(
            f"{int(contagem[n])} {n}" for n in
            ("compativel", "diferenca irrelevante", "inconclusivo",
             "diferente") if n in contagem)
        bloco_teste = f"""
<h2>Compatibilidade de {erro.upper()}: teste de equivalencia</h2>
<p class="sub">A pergunta e se o erro da mlquantify e o <i>mesmo</i> da outra
biblioteca naquele metodo, e nao qual delas fica na frente. Equivalencia nao
se demonstra deixando de rejeitar um teste de diferenca, entao a hipotese
alternativa aqui e a propria equivalencia: TOST (dois testes unilaterais de
Wilcoxon) com margem de {margem:g} em {erro.upper()}, pareado por conjunto
de dados, com Holm sobre os {len(teste)} pares. Um p de equivalencia pequeno
<b>afirma</b> que as duas erram igual dentro da margem. Resultado:
{resumo_vereditos}.</p>
<figure>{svg or ""}<figcaption>Cada ponto e a diferenca mediana de
{erro.upper()} (Hodges-Lehmann) entre a mlquantify e a outra biblioteca, no
mesmo metodo. A barra grossa e o intervalo de confianca de 95% dessa
diferenca; a linha fina cobre os 80% centrais dos conjuntos, para mostrar a
variabilidade de onde o intervalo saiu. A faixa verde e a margem de
equivalencia: quando o intervalo de 95% cabe inteiro nela (circulo), a
compatibilidade esta demonstrada, nao apenas nao refutada. Diamante marca o
intervalo que escapa da faixa.</figcaption></figure>
{_tabela_teste(teste, margem)}
<p class="nota">Unidade de analise: o conjunto de dados. O erro do mesmo
conjunto nos quatro tamanhos de amostra nao sao observacoes independentes,
entao entram pela media e cada conjunto conta uma vez. O teste e pareado
dentro do conjunto, o que tira a dificuldade do conjunto da conta, e nao
supoe normalidade. &quot;Diferenca irrelevante&quot; e o par em que os dois
testes rejeitam: existe diferenca sistematica, e ela cabe dentro da margem.
&quot;Inconclusivo&quot; e falta de evidencia nos dois sentidos, nao
igualdade.</p>"""

    nota_custo = ""
    if custo_run:
        nota_custo = (f' O custo da mlquantify vem de <code>'
                      f'{html.escape(custo_run)}</code> nos metodos medidos la,'
                      f' e de <code>{html.escape(run_id)}</code> nos outros.')

    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>mlquantify contra as outras</title><style>{CSS}</style></head>
<body><div class="env">
<h1>A mlquantify contra as outras bibliotecas</h1>
<p class="sub">Cada linha e um metodo da mlquantify contra o mesmo metodo de
outra biblioteca, pareado por conjunto e tamanho de amostra. Razao abaixo de
1 significa que a mlquantify custa menos; diferenca de AE negativa significa
que erra menos. Execucao <code>{html.escape(run_id)}</code>.{nota_custo}</p>
<div class="fichas">{fichas()}</div>
{bloco_teste}
{alvo}
{"".join(secoes)}
<p class="nota">O erro e tratado como identico abaixo de
{TOL_EMPATE:g} de diferenca em AE. As razoes de custo sao a mediana da razao
par a par, nao a razao das medianas, para que um conjunto grande nao domine o
numero. Os pares em que uma das bibliotecas nao terminou ficam de fora.</p>
</div></body></html>"""


# ---------------------------------------------------------------------------

#: Metodos que nao foram tocados em nenhuma das duas execucoes. A razao deles
#: entre as duas mede a maquina, nao o codigo, e serve de controle.
CONTROLES = ["CC", "PCC"]


def aplicar_custo_novo(pares: pd.DataFrame, run_dir_novo: Path,
                       normalizar: bool = True) -> pd.DataFrame:
    """Troca o custo da mlquantify pelo da execucao otimizada, onde existir.

    As duas execucoes nao rodaram na mesma maquina em repouso, entao a razao
    bruta entre elas carrega o estado da maquina. Os metodos de controle
    (``CC`` e ``PCC``), identicos nas duas, dao esse fator; o custo novo e
    dividido por ele antes de entrar no painel. Sem isso, uma execucao mais
    lenta por fora apaga parte do ganho -- ou inventa um.
    """
    novo = carregar(run_dir_novo)
    novo = novo[novo["biblioteca"] == MINHA]
    cols = [c for c, _, _ in CUSTOS if c in novo.columns]
    novo = novo.groupby(CHAVE, as_index=False)[cols].median()
    novo = novo.rename(columns={c: f"{c}_novo" for c in cols})
    pares = pares.merge(novo, on=CHAVE, how="left")

    fatores = {c: 1.0 for c in cols}
    if normalizar:
        unico = pares.drop_duplicates(subset=CHAVE)
        ctrl = unico[unico["metodo"].isin(CONTROLES)]
        print("fator de maquina pelos controles "
              f"({', '.join(CONTROLES)}, {len(ctrl)} pares):")
        for c in cols:
            r = (ctrl[f"{c}_novo"] / ctrl[f"{c}_minha"]).replace(
                [np.inf, -np.inf], np.nan).dropna()
            if len(r) >= 20:
                fatores[c] = float(r.median())
            print(f"  {c:<26} {fatores[c]:.3f}"
                  f"{'' if len(r) >= 20 else '  (poucos pares, nao aplicado)'}")

    for c in cols:
        novo_corrigido = pares[f"{c}_novo"] / fatores[c]
        pares[f"{c}_minha"] = novo_corrigido.fillna(pares[f"{c}_minha"])
        pares = pares.drop(columns=f"{c}_novo")
    return pares


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", default="final-pareado")
    p.add_argument("--out-dir", default=None, metavar="PASTA",
                   help="pasta de resultados, se fora do repositorio")
    p.add_argument("--custo-run", default=None, metavar="RUN",
                   help="execucao de onde tirar o custo da mlquantify "
                        "(para ver o painel com a otimizacao aplicada)")
    p.add_argument("--sem-controle", action="store_true",
                   help="nao normaliza o custo novo pelos metodos de controle")
    p.add_argument("--html", default=None, metavar="ARQUIVO",
                   help="onde gravar o painel (padrao: na pasta da execucao)")
    p.add_argument("--csv", default=None, metavar="ARQUIVO",
                   help="grava tambem o resumo em csv")
    p.add_argument("--erro", default="mae", choices=("mae", "mrae"),
                   help="medida de erro do teste (padrao: mae)")
    p.add_argument("--margem", type=float, default=MARGEM, metavar="X",
                   help=f"margem de equivalencia do TOST (padrao: {MARGEM})")
    p.add_argument("--sem-teste", action="store_true",
                   help="so o painel de custo, sem o teste e o grafico")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raiz = Path(args.out_dir).resolve() if args.out_dir else Path("results")
    if args.out_dir:
        raiz = raiz / "results"
    run_dir = raiz / args.run_id

    ok = carregar(run_dir)
    pares = parear(ok)
    if args.custo_run:
        pares = aplicar_custo_novo(pares, raiz / args.custo_run,
                                   normalizar=not args.sem_controle)
    resumo = resumir(pares)

    imprimir(resumo, pares)

    destino = Path(args.html) if args.html else run_dir / "painel_bibliotecas.html"
    destino.parent.mkdir(parents=True, exist_ok=True)

    teste, svg = None, None
    if not args.sem_teste:
        teste = testar(pares, erro=args.erro, margem=args.margem)
        if not teste.empty:
            imprimir_teste(teste, args.margem, args.erro)
            png = destino.with_name(destino.stem + "_equivalencia.png")
            svg = figura(teste, args.margem, args.erro, png)
            print(f"\ngrafico: {png}")
            teste.to_csv(destino.with_name(destino.stem + "_equivalencia.csv"),
                         index=False)

    destino.write_text(
        html_painel(resumo, pares, args.run_id, args.custo_run,
                    teste=teste, svg=svg, margem=args.margem,
                    erro=args.erro),
        encoding="utf-8")
    print(f"painel: {destino}")
    if args.csv:
        resumo.to_csv(args.csv, index=False)
        print(f"resumo: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
