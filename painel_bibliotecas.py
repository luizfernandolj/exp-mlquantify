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

import recorte_analise

MINHA = "mlquantify"

#: (coluna, rotulo curto, rotulo longo). Em todas, menor e melhor.
CUSTOS = [
    ("tempo_treino_s", "treino", "tempo de treinamento"),
    ("tempo_estimacao_total_s", "estimacao", "tempo de estimacao"),
    ("memoria_treino_mb", "mem. treino", "memoria no treinamento"),
    ("memoria_estimacao_mb", "mem. estim.", "memoria na estimacao"),
]

CHAVE = ["dataset_id", "metodo", "tamanho_amostra"]

#: As medidas de erro que a execucao ja gravou, do nome curto para a coluna
#: do summary. Qualquer outra exige rodar de novo: o vetor de prevalencia
#: estimada por amostra nao fica na execucao oficial.
ERROS = {"mae": "mae", "mrae": "mrae", "mse": "mse", "nkld": "nkld_medio"}

#: Abaixo disto o erro e tratado como identico (ruido de ponto flutuante nao
#: chega perto de 1e-3; essa e a mesma faixa que o q1_erro.csv usa).
TOL_EMPATE = 1e-3


# ---------------------------------------------------------------------------
# leitura e pareamento
# ---------------------------------------------------------------------------

def carregar(run_dir: Path) -> pd.DataFrame:
    """Le o summary da execucao, ja sem os conjuntos fora do recorte."""
    caminho = run_dir / "summary.csv"
    if not caminho.exists():
        raise SystemExit(f"nao achei {caminho}")
    df = pd.read_csv(caminho, low_memory=False)
    from expq.aggregate import expand_aliases

    df = expand_aliases(df)
    ok = df[df["status"] == "ok"].copy()
    fora = recorte_analise.excluidos()
    n = int(ok["dataset_id"].isin(fora).sum())
    if n:
        ok = ok[~ok["dataset_id"].isin(fora)].copy()
        print(f"recorte do capitulo: {ok['dataset_id'].nunique()} conjuntos"
              f" ({recorte_analise.resumo()}), {n} linhas de fora")
    return ok


def parear(ok: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (par de medicao, biblioteca rival).

    Cada linha traz as colunas da mlquantify com sufixo ``_minha`` e as da
    rival com ``_dela``.
    """
    colunas = (CHAVE + ["biblioteca", "n_classes", "n_atributos",
                        "n_instancias"] + list(ERROS.values())
               + [c for c, _, _ in CUSTOS])
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
    # o nome curto e o que o resto do script usa: nkld_medio vira nkld
    for curto, coluna in ERROS.items():
        if curto == coluna:
            continue
        pares = pares.rename(columns={f"{coluna}_minha": f"{curto}_minha",
                                      f"{coluna}_dela": f"{curto}_dela"})
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

#: A mesma decisao, escrita na unidade de cada medida. Nao ha padrao de
#: literatura para a margem de um erro quadratico -- a orientacao corrente e
#: so fixa-la antes de olhar os dados e dizer de onde ela vem --, entao ela
#: sai da **definicao da medida** e nao dos resultados: uma estimativa
#: errada em exatamente um ponto percentual em toda classe tem AE = 0,01 e
#: SE = 0,01^2. Derivar a margem do erro observado seria circular, porque o
#: dado que ela julga entraria na escolha dela.
#:
#: MRAE e NKLD ficam de fora de proposito: nao ha transposicao honesta de um
#: ponto percentual para um erro relativo nem para uma divergencia, entao
#: quem usa essas medidas tem de escolher a margem e dizer por que.
MARGENS = {"mae": MARGEM, "mse": MARGEM ** 2}


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
# custo: razao pareada, com a mesma gramatica do teste de erro
# ---------------------------------------------------------------------------
#
# A razao de custo e tratada no logaritmo, porque so no log a razao e
# simetrica: 2x mais caro e 0,5x mais barato ficam a mesma distancia de 1.
# A mediana e o intervalo saem do mesmo estimador de Hodges-Lehmann usado no
# erro, e voltam para a escala de razao no fim.

def testar_custo(pares: pd.DataFrame, coluna: str) -> pd.DataFrame:
    linhas = []
    a, b = f"{coluna}_minha", f"{coluna}_dela"
    if a not in pares or b not in pares:
        return pd.DataFrame()
    for (metodo, lib), sub in pares.groupby(["metodo", "biblioteca"]):
        v = sub[(sub[a] > 0) & (sub[b] > 0)]
        # uma observacao por conjunto: os quatro tamanhos de amostra do mesmo
        # conjunto nao sao independentes
        v = v.groupby("dataset_id")[[a, b]].median()
        if len(v) < 6:
            continue
        d = np.log2(v[a].to_numpy() / v[b].to_numpy())
        d = d[np.isfinite(d)]
        if len(d) < 6:
            continue
        hl, baixo, alto = hodges_lehmann(d)
        linhas.append({
            "metodo": metodo, "biblioteca": lib, "n_conjuntos": len(d),
            "razao": float(2.0 ** hl),
            "ic_baixo": float(2.0 ** baixo), "ic_alto": float(2.0 ** alto),
            "ganha": bool(2.0 ** alto < 1.0),
            "perde": bool(2.0 ** baixo > 1.0),
        })
    out = pd.DataFrame(linhas)
    if out.empty:
        return out
    return out.sort_values("razao").reset_index(drop=True)


def imprimir_custo(tabela: pd.DataFrame, titulo: str) -> None:
    ganha = int(tabela["ganha"].sum())
    perde = int(tabela["perde"].sum())
    print()
    print("=" * 70)
    print(f"{titulo.upper()}: RAZAO MLQUANTIFY / OUTRA, POR CONJUNTO")
    print("=" * 70)
    print(f"{'metodo':<9}{'biblioteca':<19}{'n':>4}{'razao':>9}{'IC 95%':>20}")
    for _, r in tabela.iterrows():
        ic = f"[{r['ic_baixo']:.2f}, {r['ic_alto']:.2f}]"
        print(f"{r['metodo']:<9}{r['biblioteca']:<19}{r['n_conjuntos']:>4}"
              f"{r['razao']:>9.2f}{ic:>20}")
    print(f"\n  IC inteiro abaixo de 1 (vantagem firme): {ganha} de "
          f"{len(tabela)}   acima de 1: {perde}")


# ---------------------------------------------------------------------------
# figuras
# ---------------------------------------------------------------------------

#: Uma cor por biblioteca rival. Distinguiveis tambem em escala de cinza e
#: para quem nao separa vermelho de verde.
CORES = {
    "quapy": "#2b6cb0",
    "quantificationlib": "#b7791f",
    "qunfold": "#1d5c46",
}
COR_EXTRA = ["#6b46c1", "#9b2c2c", "#2c7a7b"]

#: Uma forma por biblioteca. A figura tem de se ler impressa em preto e
#: branco: a cor e redundante com a forma, e os tres tons foram escolhidos
#: com luminancias distintas (aprox. 96, 129 e 71 em cinza de 0 a 255).
MARCAS = {
    "quapy": "o",
    "quantificationlib": "s",
    "qunfold": "^",
}
MARCA_EXTRA = ["v", "P", "X"]

TINTA = "#1a1a1a"
FRACO = "#6b6b6b"
GRADE = "#d9d9d9"
VERDE = "#2f7a5a"
FUNDO = "#ffffff"


def _virgula(ax, eixo: str) -> None:
    """Separador decimal em virgula, como manda o texto em portugues."""
    from matplotlib.ticker import FuncFormatter

    formatador = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if eixo == "y" else ax.xaxis).set_major_formatter(formatador)


def _estilo(ax, *, fonte=13):
    ax.set_facecolor(FUNDO)
    ax.tick_params(colors=TINTA, labelsize=fonte, length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRADE)
    ax.grid(axis="x", color=GRADE, lw=0.7)
    ax.set_axisbelow(True)


def _posicoes(tabela: pd.DataFrame, ordem, desloc=0.26):
    """y de cada par, agrupado por metodo, uma linha por biblioteca."""
    ys, rotulos, y = {}, [], 0.0
    for metodo in ordem:
        sub = tabela[tabela["metodo"] == metodo]
        n = len(sub)
        for k, lib in enumerate(sorted(sub["biblioteca"])):
            ys[(metodo, lib)] = y + (k - (n - 1) / 2) * desloc
        rotulos.append((y, metodo))
        y -= 1.0
    return ys, rotulos, y


def _cores_de(tabela):
    cores = dict(CORES)
    extras = [l for l in sorted(tabela["biblioteca"].unique()) if l not in cores]
    for i, lib in enumerate(extras):
        cores[lib] = COR_EXTRA[i % len(COR_EXTRA)]
    return cores


def _marcas_de(tabela):
    marcas = dict(MARCAS)
    extras = [l for l in sorted(tabela["biblioteca"].unique())
              if l not in marcas]
    for i, lib in enumerate(extras):
        marcas[lib] = MARCA_EXTRA[i % len(MARCA_EXTRA)]
    return marcas


def _legenda_em_duas_linhas(fig, ax, marcadores, chaves, fonte=15):
    """As bibliotecas em cima, a chave do desenho embaixo.

    Sao duas legendas, e nao uma em ``ncols``: a legenda do matplotlib
    alinha por coluna, e a coluna fica com a largura do item mais largo, o
    que empurra a segunda linha para a direita. A de baixo e desenhada
    primeiro porque e dela que sai a altura onde a de cima se ancora.

    As duas sao centradas no desenho, e nao no eixo. Os rotulos do eixo y
    ficam fora da caixa do eixo e so alargam o desenho para a esquerda, de
    modo que o meio do eixo nao e o meio do que o recorte ``tight`` grava
    no arquivo: ancorar a legenda em ``0.5`` do eixo a joga para a direita.
    O meio sai do ``tightbbox``, medido antes de haver legenda.
    """
    comum = dict(loc="lower center", frameon=False, fontsize=fonte,
                 handletextpad=0.5, handlelength=1.8, borderpad=0.0)
    fig.canvas.draw()
    desenho = (ax.get_tightbbox(fig.canvas.get_renderer())
               .transformed(ax.transAxes.inverted()))
    meio = (desenho.x0 + desenho.x1) / 2.0

    leg_chave = ax.legend(handles=chaves, bbox_to_anchor=(meio, 1.0),
                          ncols=len(chaves), columnspacing=1.8, **comum)
    ax.add_artist(leg_chave)

    fig.canvas.draw()
    caixa = leg_chave.get_window_extent().transformed(ax.transAxes.inverted())
    leg_libs = ax.legend(handles=marcadores,
                         bbox_to_anchor=(meio, caixa.y1 + 0.002),
                         ncols=len(marcadores), columnspacing=2.4, **comum)
    return [leg_chave, leg_libs]


def _marcadores_de(cores, formas, libs, ms=11):
    from matplotlib.lines import Line2D

    return [Line2D([], [], color=cores[l], lw=0, marker=formas[l], ms=ms,
                   label=l) for l in libs]


def variantes(libs) -> list:
    """(sufixo, argumentos) das versoes extras de uma figura.

    Duas coisas diferentes, que a figura aceita separadamente: a **deitada**
    troca os eixos, para caber na largura de uma coluna de texto; a **por
    biblioteca** mostra uma rival de cada vez, em vez de todas sobrepostas.
    As por biblioteca saem na orientacao padrao; para deita-las tambem,
    basta acrescentar ``deitado=True`` aqui.
    """
    return ([("deitada", {"deitado": True})]
            + [(lib, {"so_libs": [lib]}) for lib in libs])


def _salvar(fig, destino: Path, extras=()) -> str:
    # extras: artistas fora do eixo (legendas presas com add_artist) que o
    # recorte "tight" so inclui se forem citados aqui
    # lista vazia tambem sobrescreveria o padrao do matplotlib
    extras = list(extras) or None
    fig.savefig(destino, facecolor=FUNDO, bbox_inches="tight",
                bbox_extra_artists=extras)
    for ext in (".svg", ".pdf"):
        fig.savefig(destino.with_suffix(ext), facecolor=FUNDO,
                    bbox_inches="tight", bbox_extra_artists=extras)
    texto = destino.with_suffix(".svg").read_text(encoding="utf-8")
    return texto[texto.index("<svg"):]


def figura_equivalencia(teste: pd.DataFrame, margem: float, erro: str,
                        destino: Path, deitado: bool = False,
                        so_libs: list | None = None) -> str:
    """Floresta do teste de equivalencia, uma escala so.

    ``deitado`` troca os eixos: por padrao cada metodo e uma linha e o valor
    corre na horizontal, que e a forma classica da floresta e a que aguenta
    muitos metodos; deitado, os metodos vao para o eixo x e o valor para o
    y, que e a forma que cabe na largura de uma coluna de texto.

    ``so_libs`` restringe a uma ou mais bibliotecas rivais, para a figura de
    cada par em vez de todas sobrepostas.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if so_libs:
        teste = teste[teste["biblioteca"].isin(so_libs)]
        if teste.empty:
            return ""
    cores = _cores_de(teste)
    formas = _marcas_de(teste)
    libs = sorted(teste["biblioteca"].unique())
    ordem = (teste.assign(a=teste["hl"].abs()).groupby("metodo")["a"].max()
             .sort_values(ascending=False).index.tolist())
    ys, rotulos, fim = _posicoes(teste, ordem)

    # A escala para logo depois do dado: so a faixa e o intervalo mais largo
    # precisam caber. Margem maior so cria coluna vazia dos dois lados.
    limite = max(margem * 1.12,
                 float(np.nanmax(np.abs(
                     teste[["ic_baixo", "ic_alto"]].to_numpy()))) * 1.06)

    comprimento = 0.50 * len(ordem) + 2.8
    fig, ax = plt.subplots(dpi=200, figsize=(comprimento, 8.4) if deitado
                           else (10.0, comprimento))
    fig.patch.set_facecolor(FUNDO)
    _estilo(ax, fonte=16)
    if deitado:
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRADE, lw=0.7)

    # faixa de equivalencia e linha do zero, no eixo do valor
    faixa = ax.axhspan if deitado else ax.axvspan
    linha_valor = ax.axhline if deitado else ax.axvline
    linha_pos = ax.axvline if deitado else ax.axhline
    banda_pos = ax.axvspan if deitado else ax.axhspan

    # Como a faixa ocupa quase toda a escala, quem marca o limite sao as
    # duas linhas tracejadas; o preenchimento fica so como fundo.
    faixa(-margem, margem, color=VERDE, alpha=0.07, lw=0, zorder=0)
    for lado in (-margem, margem):
        linha_valor(lado, color=VERDE, lw=1.8, ls=(0, (5, 4)), zorder=2)
    linha_valor(0, color=FRACO, lw=1.2, zorder=1)

    # faixas alternadas levam o olho do nome do metodo ate o ponto; a de
    # quem escapa da faixa fica tingida
    fora = set(teste.loc[~teste["ic_na_faixa"].astype(bool), "metodo"])
    for i, (p, metodo) in enumerate(rotulos):
        if metodo in fora:
            banda_pos(p - 0.5, p + 0.5, color="#fbeaea", lw=0, zorder=-1)
        elif i % 2 == 0:
            banda_pos(p - 0.5, p + 0.5, color="#f4f4f4", lw=0, zorder=-1)

    # A forma diz qual biblioteca e o preenchimento diz o veredito, para que
    # nada dependa da cor: impressa em cinza a figura continua separavel.
    for _, r in teste.iterrows():
        p = ys[(r["metodo"], r["biblioteca"])]
        cor = cores[r["biblioteca"]]
        forma = formas[r["biblioteca"]]
        dentro = bool(r["ic_na_faixa"])
        extremos, meio = [r["ic_baixo"], r["ic_alto"]], [r["hl"]]
        barra = ([p, p], extremos) if deitado else (extremos, [p, p])
        ponto = ([p], meio) if deitado else (meio, [p])
        ax.plot(*barra, color=cor, lw=3.0, solid_capstyle="round", zorder=3)
        ax.plot(*ponto, marker=forma, ms=10.0 if forma != "o" else 9.5,
                color=cor if dentro else FUNDO, mec=cor, mew=2.4, zorder=4)

    posicoes = [p for p, _ in rotulos]
    nomes = [m for _, m in rotulos]
    rotulo_valor = (f"diferença de {erro.upper()} "
                    f"(mlquantify $-$ outra biblioteca)")
    if deitado:
        ax.set_ylim(-limite, limite)
        # o primeiro metodo a esquerda: as posicoes descem a partir de zero
        ax.set_xlim(1.05, fim + 0.5)
        ax.set_xticks(posicoes)
        ax.set_xticklabels(nomes, fontsize=14, color=TINTA, rotation=90)
        ax.set_ylabel(rotulo_valor, fontsize=17, color=TINTA, labelpad=12)
        _virgula(ax, "y")
        # x em fracao do eixo e y no valor: o texto fica preso a borda
        # esquerda sem depender da escala
        for texto, alvo, va in (("erra mais $\\uparrow$", limite * 0.97, "top"),
                                ("$\\downarrow$ mlquantify erra menos",
                                 -limite * 0.97, "bottom")):
            ax.annotate(texto, xy=(0.012, alvo), fontsize=15, color=FRACO,
                        ha="left", va=va, xycoords=ax.get_yaxis_transform(),
                        bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2))
    else:
        ax.set_xlim(-limite, limite)
        ax.set_ylim(fim + 0.5, 1.05)
        ax.set_yticks(posicoes)
        ax.set_yticklabels(nomes, fontsize=16, color=TINTA)
        ax.set_xlabel(rotulo_valor, fontsize=17, color=TINTA, labelpad=12)
        _virgula(ax, "x")
        ax.annotate("$\\leftarrow$ mlquantify erra menos",
                    xy=(-limite * 0.97, 0.72), fontsize=15, color=FRACO,
                    ha="left", va="center",
                    bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2))
        ax.annotate("erra mais $\\rightarrow$", xy=(limite * 0.97, 0.72),
                    fontsize=15, color=FRACO, ha="right", va="center",
                    bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2))

    # Duas legendas empilhadas, as duas centradas no desenho: uma so legenda
    # em ncols alinha por coluna e joga a segunda linha para a direita. A de
    # baixo e desenhada primeiro para dar a altura da de cima.
    chaves = [
        Line2D([], [], color=TINTA, lw=0, marker="o", ms=10,
               label="cheio: IC dentro da faixa"),
        Line2D([], [], color=TINTA, lw=0, marker="o", ms=10, mfc=FUNDO,
               mew=2.4, label="vazado: IC fora"),
        Line2D([], [], color=TINTA, lw=3.0, label="IC de 95%"),
        Line2D([], [], color=VERDE, lw=1.8, ls=(0, (5, 4)),
               label="faixa $\\pm$" + f"{margem:g}".replace(".", ",")),
    ]
    legendas = _legenda_em_duas_linhas(
        fig, ax, _marcadores_de(cores, formas, libs), chaves)
    return _salvar(fig, destino, extras=legendas)


def figura_bland_altman(dif: pd.DataFrame, teste: pd.DataFrame, margem: float,
                        erro: str, destino: Path, quantos: int = 8) -> str:
    """Concordancia conjunto a conjunto: a diferenca contra o nivel do erro.

    O grafico de equivalencia resume cada par num ponto e num intervalo.
    Este responde outra pergunta, que aquele nao responde: a discordancia e
    constante ou cresce com a dificuldade do conjunto? E o grafico de
    Bland-Altman, da concordancia entre dois instrumentos de medida, com
    duas adaptacoes. O eixo y e simetrico-logaritmico com o joelho na
    propria margem, porque as diferencas se amontoam em 1e-4 e vao a 0,3 --
    dentro da faixa o eixo e linear, fora dela e logaritmico. E os limites
    de concordancia sao os percentis 2,5 e 97,5 em vez de media +- 1,96
    desvios, porque a cauda e pesada e o resto da analise tambem e
    distribuicao-livre.

    Mostra os ``quantos`` pares de maior espalhamento: onde as duas
    bibliotecas concordam nao ha o que ver.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter

    dados = dif.copy()
    dados["nivel"] = (dados[f"{erro}_minha"] + dados[f"{erro}_dela"]) / 2.0
    cores = _cores_de(dados)
    formas = _marcas_de(dados)

    # os pares em que ha discordancia para olhar, pelos 95% centrais
    def espalhamento(s):
        return float(np.percentile(s, 97.5) - np.percentile(s, 2.5))

    ordem = (dados.groupby(["metodo", "biblioteca"])["dif"]
             .agg(espalhamento).sort_values(ascending=False))
    escolhidos = list(ordem.index[:quantos])
    if not escolhidos:
        return ""

    colunas = 4 if len(escolhidos) > 3 else len(escolhidos)
    linhas = int(np.ceil(len(escolhidos) / colunas))
    fig, eixos = plt.subplots(linhas, colunas, dpi=200, squeeze=False,
                              sharex=True, sharey=True,
                              figsize=(3.9 * colunas, 3.3 * linhas + 1.2))
    fig.patch.set_facecolor(FUNDO)

    limite = max(float(dados.loc[
        dados.set_index(["metodo", "biblioteca"]).index.isin(escolhidos),
        "dif"].abs().max()) * 1.6, margem * 4)
    nivel_max = float(dados["nivel"].max()) * 1.04

    chave = teste.set_index(["metodo", "biblioteca"]) if teste is not None \
        else None

    for i, eixo in enumerate(eixos.ravel()):
        if i >= len(escolhidos):
            eixo.set_visible(False)
            continue
        metodo, lib = escolhidos[i]
        sub = dados[(dados["metodo"] == metodo)
                    & (dados["biblioteca"] == lib)]
        cor = cores[lib]
        _estilo(eixo, fonte=13)
        eixo.grid(axis="y", color=GRADE, lw=0.7)

        eixo.axhspan(-margem, margem, color=VERDE, alpha=0.08, lw=0, zorder=0)
        eixo.axhline(0, color=FRACO, lw=1.0, zorder=1)

        baixo = float(np.percentile(sub["dif"], 2.5))
        alto = float(np.percentile(sub["dif"], 97.5))
        centro = (float(chave.loc[(metodo, lib), "hl"])
                  if chave is not None and (metodo, lib) in chave.index
                  else float(np.median(sub["dif"])))
        for lado in (baixo, alto):
            eixo.axhline(lado, color=cor, lw=1.4, ls=(0, (5, 4)), zorder=2)
        eixo.axhline(centro, color=cor, lw=2.0, zorder=2)

        eixo.plot(sub["nivel"], sub["dif"], ls="", marker=formas[lib],
                  ms=5.5, mfc="none", mec=cor, mew=1.2, alpha=0.75, zorder=3)

        # a pergunta do grafico, em numero: a discordancia acompanha o nivel?
        rho = float(pd.Series(sub["dif"].abs().to_numpy()).corr(
            pd.Series(sub["nivel"].to_numpy()), method="spearman"))
        eixo.set_title(f"{metodo} contra {lib}", fontsize=13.5, color=TINTA,
                       pad=8)
        eixo.annotate(f"n={len(sub)}   rho={rho:+.2f}",
                      xy=(0.97, 0.03), xycoords="axes fraction",
                      fontsize=11.5, color=FRACO, ha="right", va="bottom",
                      bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2),
                      zorder=5)

    marcas = [t for t in (-0.3, -0.1, -0.03, -0.01, 0, 0.01, 0.03, 0.1, 0.3)
              if abs(t) <= limite]
    for eixo in eixos.ravel():
        if not eixo.get_visible():
            continue
        eixo.set_yscale("symlog", linthresh=margem, linscale=1.1)
        eixo.set_ylim(-limite, limite)
        eixo.set_xlim(0, nivel_max)
        eixo.yaxis.set_major_locator(FixedLocator(marcas))
        eixo.set_yticklabels([f"{t:g}" for t in marcas], fontsize=12.5)
        eixo.yaxis.set_minor_formatter(NullFormatter())

    for eixo in eixos[-1]:
        if eixo.get_visible():
            eixo.set_xlabel(f"nível do {erro.upper()}, média das duas",
                            fontsize=13, color=TINTA, labelpad=8)
    for linha in eixos:
        linha[0].set_ylabel(f"diferença de {erro.upper()}\n"
                            f"(mlquantify $-$ outra)",
                            fontsize=13, color=TINTA, labelpad=8)

    chaves = [
        Line2D([], [], color=TINTA, lw=0, marker="o", ms=7, mfc="none",
               mew=1.2, label="um conjunto de dados"),
        Line2D([], [], color=TINTA, lw=2.0, label="diferença mediana"),
        Line2D([], [], color=TINTA, lw=1.4, ls=(0, (5, 4)),
               label="95% centrais dos conjuntos"),
        Line2D([], [], color=VERDE, lw=6, alpha=0.3,
               label=f"faixa $\\pm${margem:g}"),
    ]
    leg = fig.legend(handles=chaves, loc="lower left", frameon=False,
                     fontsize=13, ncols=len(chaves), handletextpad=0.5,
                     handlelength=1.8, borderpad=0.0,
                     bbox_to_anchor=(0.06, 0.985), bbox_transform=fig.transFigure)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _salvar(fig, destino, extras=[leg])


#: Cor e tracejado da propria mlquantify nas figuras em que ela e uma curva
#: entre outras, e nao o referencial. O tracejado e redundante com a cor,
#: pela mesma razao das formas: a figura tem de sair num jornal em cinza.
COR_MINHA = "#6b46c1"
TRACOS = {
    "mlquantify": "-",
    "quapy": (0, (6, 3)),
    "quantificationlib": (0, (1.5, 2)),
    "qunfold": (0, (7, 2, 1.5, 2)),
}


def _custo_longo(pares: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Uma linha por (metodo, conjunto, biblioteca) com o custo.

    Reconstroi o formato longo a partir da tabela pareada, para que o custo
    da mlquantify seja o mesmo que as outras figuras usam -- inclusive com a
    execucao otimizada por cima, quando houver.
    """
    a, b = f"{coluna}_minha", f"{coluna}_dela"
    if a not in pares or b not in pares:
        return pd.DataFrame()
    chave = ["metodo", "dataset_id"]
    # os quatro tamanhos de amostra do mesmo conjunto nao sao observacoes
    # independentes: entram pela mediana, como no teste de custo
    delas = (pares[pares[b] > 0].groupby(chave + ["biblioteca"], as_index=False)
             [b].median().rename(columns={b: "custo"}))
    minha = (pares[pares[a] > 0].groupby(chave, as_index=False)[a].median()
             .rename(columns={a: "custo"}))
    minha["biblioteca"] = MINHA
    return pd.concat([delas, minha], ignore_index=True)


def _perfil(largura: pd.DataFrame, libs: list[str]):
    """Perfil de Dolan-More: fracao dos problemas dentro de um fator tau.

    ``largura`` tem um problema por linha e uma biblioteca por coluna. So
    entram os problemas em que todas as bibliotecas da lista mediram.
    """
    sub = largura[libs].dropna()
    sub = sub[(sub > 0).all(axis=1)]
    if sub.empty:
        return {}, 0
    melhor = sub.min(axis=1)
    return {lib: np.sort((sub[lib] / melhor).to_numpy()) for lib in libs}, \
        len(sub)


def figura_perfil(pares: pd.DataFrame, coluna: str, destino: Path) -> str:
    """Perfil de desempenho de custo, um painel por biblioteca rival.

    Media e mediana respondem "quanto custa no conjunto tipico"; esta figura
    responde outra coisa: **em que fracao dos problemas cada biblioteca fica
    dentro de um fator tau da mais barata**. E o grafico que Dolan e More
    propuseram para benchmark de otimizadores, justamente porque uns poucos
    problemas caros dominam qualquer media. A leitura e direta: a curva que
    sobe mais rapido vence, o valor em tau=1 e a fracao de problemas em que
    a biblioteca e a mais barata de todas, e onde a curva chega a 1 e o pior
    caso dela.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter

    longo = _custo_longo(pares, coluna)
    if longo.empty:
        return ""
    largura = longo.pivot_table(index=["metodo", "dataset_id"],
                                columns="biblioteca", values="custo")
    rivais = [l for l in sorted(largura.columns) if l != MINHA]
    if not rivais:
        return ""
    cores = dict(_cores_de(longo), **{MINHA: COR_MINHA})

    # tres paineis pareados, mais o perfil de quatro curvas nos metodos que
    # todas implementam
    comuns = [MINHA] + rivais
    paineis = [([MINHA, r], f"contra a {r}") for r in rivais]
    if len(comuns) > 2 and not largura[comuns].dropna().empty:
        paineis.append((comuns, "todas, nos metodos que as quatro tem"))

    fig, eixos = plt.subplots(1, len(paineis), dpi=200, squeeze=False,
                              sharey=True,
                              figsize=(4.3 * len(paineis), 4.4))
    fig.patch.set_facecolor(FUNDO)

    limite = 1.0
    curvas = []
    for libs, _ in paineis:
        razoes, n = _perfil(largura, libs)
        curvas.append((razoes, n))
        for v in razoes.values():
            if len(v):
                limite = max(limite, float(v[-1]))
    limite = min(limite, 2.0 ** 12) * 1.15

    for eixo, (libs, rotulo), (razoes, n) in zip(eixos[0], paineis, curvas):
        _estilo(eixo, fonte=13)
        eixo.grid(axis="y", color=GRADE, lw=0.7)
        eixo.set_xscale("log", base=2)
        for lib in libs:
            v = razoes.get(lib)
            if v is None or not len(v):
                continue
            # degrau: a fracao so sobe quando um problema entra
            x = np.concatenate(([1.0], v, [limite]))
            y = np.concatenate(([0.0], np.arange(1, len(v) + 1) / len(v),
                                [1.0]))
            eixo.step(x, y, where="post", color=cores[lib], lw=2.6,
                      ls=TRACOS.get(lib, "-"),
                      label=lib, zorder=3,
                      solid_capstyle="round")
        eixo.set_title(rotulo, fontsize=13.5, color=TINTA, pad=8)
        eixo.annotate(f"{n} problemas", xy=(0.97, 0.04),
                      xycoords="axes fraction", fontsize=11.5, color=FRACO,
                      ha="right", va="bottom",
                      bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2))
        eixo.set_xlabel("fator $\\tau$ em relação à mais barata",
                        fontsize=13, color=TINTA, labelpad=8)

    marcas = [2.0 ** e for e in range(0, 13) if 2.0 ** e <= limite]
    for eixo in eixos[0]:
        eixo.set_xlim(1.0, limite)
        eixo.set_ylim(0, 1.02)
        eixo.xaxis.set_major_locator(FixedLocator(marcas))
        eixo.set_xticklabels([f"{m:g}" for m in marcas], fontsize=12.5)
        eixo.xaxis.set_minor_formatter(NullFormatter())
    eixos[0][0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    eixos[0][0].set_yticklabels(["0", "25%", "50%", "75%", "100%"],
                                fontsize=12.5)
    eixos[0][0].set_ylabel("fração dos problemas", fontsize=13, color=TINTA,
                           labelpad=8)

    chaves = [Line2D([], [], color=cores[l], lw=2.6, ls=TRACOS.get(l, "-"),
                     label=l) for l in [MINHA] + rivais]
    leg = fig.legend(handles=chaves, loc="lower left", frameon=False,
                     fontsize=13, ncols=len(chaves), handletextpad=0.6,
                     handlelength=2.8, borderpad=0.0,
                     bbox_to_anchor=(0.06, 0.975),
                     bbox_transform=fig.transFigure, title=None)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _salvar(fig, destino, extras=[leg])


# ---------------------------------------------------------------------------
# ranking dos pares metodo x biblioteca
# ---------------------------------------------------------------------------
#
# As outras figuras comparam a mlquantify contra uma rival de cada vez, no
# mesmo metodo. Esta junta as duas variaveis num tratamento so -- o par
# ``metodo/biblioteca`` -- e ordena todos ao mesmo tempo, que e a pergunta
# de quem vai escolher o que usar: nao "a implementacao concorda", e sim
# "qual par entrega mais barato, ou erra menos".
#
# O bloco e o conjunto de dados. Dentro de cada conjunto os pares sao
# ordenados, o posto 1 fica com o melhor, e a caixa mostra a distribuicao
# dos postos daquele par entre os conjuntos. Um par com caixa baixa e curta
# e bom e constante; caixa baixa e comprida e bom em media e instavel, que e
# uma informacao que a media de erro esconde.

#: Hachura por biblioteca, para a figura se ler impressa em cinza. A cor
#: continua, redundante.
HACHURAS = {
    "mlquantify": "",
    "quapy": "///",
    "quantificationlib": "...",
    "qunfold": "xxx",
}


def _longo_coluna(ok: pd.DataFrame, pares: pd.DataFrame, coluna: str):
    """Uma linha por (par, conjunto), com o custo corrigido quando houver.

    Parte da tabela crua, que tem **todos** os pares -- inclusive os metodos
    que so uma biblioteca implementa, que a tabela pareada descarta. Depois,
    se a execucao de custo otimizado foi aplicada, sobrescreve as linhas da
    mlquantify com os valores dela.
    """
    if coluna not in ok.columns:
        return pd.DataFrame()
    base = ok[ok[coluna].notna()].copy()
    # os quatro tamanhos de amostra do mesmo conjunto nao sao observacoes
    # independentes: entram pela mediana
    base = (base.groupby(["metodo", "biblioteca", "dataset_id"],
                         as_index=False)[coluna].median()
            .rename(columns={coluna: "valor"}))

    a = f"{coluna}_minha"
    if pares is not None and a in pares.columns:
        minha = (pares[pares[a].notna()]
                 .groupby(["metodo", "dataset_id"], as_index=False)[a].median()
                 .rename(columns={a: "valor"}))
        minha["biblioteca"] = MINHA
        chave = ["metodo", "biblioteca", "dataset_id"]
        base = (minha.set_index(chave)
                .combine_first(base.set_index(chave)).reset_index())

    base["par"] = base["metodo"] + "/" + base["biblioteca"]
    return base


def _longo_total(ok: pd.DataFrame, pares: pd.DataFrame, colunas, juntar):
    """Junta as fases numa medida so: ``soma`` para tempo, ``max`` para pico.

    Tempo de treino e de estimacao **somam**, porque sao trabalho que
    acontece um depois do outro. Memoria **nao soma**: as duas fases sao
    sequenciais e o processo libera a primeira antes da segunda, de modo que
    o pico do programa inteiro e o **maior** dos dois picos, nao a soma --
    somar inventaria uma ocupacao que nunca existiu ao mesmo tempo.

    So entra o par que tem as duas fases medidas no conjunto: uma soma com
    metade das parcelas seria menor por ausencia, nao por eficiencia.
    """
    if isinstance(colunas, str):
        colunas = (colunas,)
    chave = ["metodo", "biblioteca", "dataset_id", "par"]
    junto = None
    for i, coluna in enumerate(colunas):
        parte = _longo_coluna(ok, pares, coluna)
        if parte.empty:
            return pd.DataFrame()
        parte = parte[chave + ["valor"]].rename(columns={"valor": f"v{i}"})
        junto = parte if junto is None else junto.merge(parte, on=chave)
    valores = [c for c in junto.columns if c.startswith("v")]
    junto["valor"] = (junto[valores].sum(axis=1) if juntar == "soma"
                      else junto[valores].max(axis=1))
    return junto.drop(columns=valores)


def postos_dos_pares(ok: pd.DataFrame, pares: pd.DataFrame, colunas,
                     juntar: str = "soma", piso: float = 0.95):
    """Posto de cada par (metodo, biblioteca) dentro de cada conjunto.

    Ordenar exige **bloco completo**: um conjunto em que falte um par daria
    a ele um posto melhor so por ausencia. Como a cobertura e quase cheia --
    2,3% de celulas vazias --, a perda e pequena: o piso de 95% mantem 68
    dos 74 pares e 131 dos 142 conjuntos. Quem sai sao os pares que falham
    muito, e eles saem **nomeados** no resumo impresso, nao em silencio.
    """
    longo = _longo_total(ok, pares, colunas, juntar)
    if longo.empty:
        return None, [], 0
    largura = longo.pivot_table(index="dataset_id", columns="par",
                                values="valor")
    cobertura = largura.notna().mean()
    fora = sorted(cobertura[cobertura < piso].index)
    largura = largura[[c for c in largura.columns if c not in fora]].dropna()
    if largura.empty or largura.shape[1] < 2:
        return None, fora, 0
    # posto 1 para o menor valor: erro e custo sao menor-melhor
    return largura.rank(axis=1, ascending=True), fora, len(largura)


def imprimir_ranking(postos: pd.DataFrame, fora: list, n: int,
                     titulo: str) -> None:
    if postos is None:
        return
    medios = postos.mean().sort_values()
    print()
    print("=" * 86)
    print(f"RANKING DE {titulo.upper()}: PARES METODO x BIBLIOTECA "
          f"({len(medios)} pares, {n} conjuntos)")
    print("=" * 86)
    print("posto medio, 1 = melhor; bloco = conjunto de dados")
    print("\n  os dez melhores:")
    for nome, posto in medios.head(10).items():
        print(f"    {posto:>6.2f}  {nome}")
    print("\n  os cinco piores:")
    for nome, posto in medios.tail(5).items():
        print(f"    {posto:>6.2f}  {nome}")
    if fora:
        print(f"\n  fora do ranking por cobertura abaixo do piso: "
              f"{', '.join(fora)}")


def figura_ranking(ok: pd.DataFrame, pares: pd.DataFrame, medidas: list,
                   destino: Path, piso: float = 0.95,
                   deitado: bool = False, so_libs: list | None = None) -> str:
    """Caixas do posto de cada par metodo x biblioteca, uma linha por medida.

    A caixa e a distribuicao dos postos entre os conjuntos de dados, e os
    pares vao ordenados pelo posto mediano. A cor e a hachura dizem a
    biblioteca.

    ``deitado`` deita as caixas, o que deixa os nomes dos pares legiveis a
    esquerda em vez de girados 90 graus. ``so_libs`` mostra so os pares de
    uma biblioteca -- **o posto continua sendo o do ranking inteiro**, de
    modo que a figura responde onde os pares dela caem entre todos, e nao
    como eles se ordenam entre si.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    painel = []
    for colunas, titulo, juntar in medidas:
        postos, fora, n = postos_dos_pares(ok, pares, colunas, juntar, piso)
        if postos is None:
            continue
        if so_libs:
            # filtra depois de ordenar: o posto e o do ranking inteiro
            fica = [c for c in postos.columns if c.split("/")[1] in so_libs]
            if not fica:
                continue
            postos = postos[fica]
        painel.append((titulo, postos, n))
    if not painel:
        return ""

    libs = sorted({p.split("/")[1] for _, postos, _ in painel
                   for p in postos.columns})
    cores = dict(_cores_de(pd.DataFrame({"biblioteca": libs})),
                 **{MINHA: COR_MINHA})

    quantos = max(len(postos.columns) for _, postos, _ in painel)
    comprimento = max(7.0, 0.26 * quantos)
    fig, eixos = plt.subplots(
        1 if deitado else len(painel), len(painel) if deitado else 1,
        squeeze=False, dpi=200,
        figsize=((5.6 * len(painel), comprimento) if deitado
                 else (comprimento, 3.6 * len(painel) + 0.8)))
    fig.patch.set_facecolor(FUNDO)

    for eixo, (titulo, postos, n) in zip(eixos.ravel(), painel):
        ordem = postos.median().sort_values().index.tolist()
        dados = [postos[c].to_numpy() for c in ordem]
        _estilo(eixo, fonte=11)
        eixo.grid(axis="y", visible=deitado, color=GRADE, lw=0.7)
        eixo.grid(axis="x", visible=not deitado, color=GRADE, lw=0.7)
        if deitado:
            eixo.grid(axis="x", color=GRADE, lw=0.7)

        caixas = eixo.boxplot(dados, patch_artist=True, widths=0.68,
                              vert=not deitado,
                              flierprops=dict(marker="o", ms=2.0,
                                              mfc=FRACO, mec="none",
                                              alpha=0.6),
                              medianprops=dict(color=TINTA, lw=1.4),
                              whiskerprops=dict(color=FRACO, lw=1.0),
                              capprops=dict(color=FRACO, lw=1.0))
        for caixa, nome in zip(caixas["boxes"], ordem):
            lib = nome.split("/")[1]
            caixa.set(facecolor=cores[lib], alpha=0.85, lw=0.8,
                      edgecolor=TINTA, hatch=HACHURAS.get(lib, ""))

        nomes = [c.split("/")[0] for c in ordem]
        if deitado:
            eixo.set_yticks(range(1, len(ordem) + 1))
            eixo.set_yticklabels(nomes, fontsize=9.5, color=TINTA)
            # o primeiro colocado no alto
            eixo.set_ylim(len(ordem) + 0.6, 0.4)
            eixo.set_xlim(0.5, quantos + 0.5)
            eixo.set_xlabel("posto (1 = melhor)", fontsize=12, color=TINTA,
                            labelpad=6)
        else:
            eixo.set_xticks(range(1, len(ordem) + 1))
            eixo.set_xticklabels(nomes, rotation=90, fontsize=9.5,
                                 color=TINTA)
            eixo.set_xlim(0.4, len(ordem) + 0.6)
            eixo.set_ylim(0.5, quantos + 0.5)
            eixo.set_ylabel("posto (1 = melhor)", fontsize=12, color=TINTA,
                            labelpad=6)
        eixo.annotate(f"{titulo}  ·  {len(ordem)} pares em {n} conjuntos",
                      xy=(0.996 if deitado else 0.004, 0.988 if deitado
                          else 0.965),
                      xycoords="axes fraction", fontsize=13, color=TINTA,
                      ha="right" if deitado else "left", va="top",
                      bbox=dict(facecolor=FUNDO, edgecolor="none", pad=3))

    chaves = [Patch(facecolor=cores[l], edgecolor=TINTA, lw=0.8, alpha=0.85,
                    hatch=HACHURAS.get(l, ""), label=l) for l in libs]
    leg = fig.legend(handles=chaves, loc="lower left", frameon=False,
                     fontsize=12.5, ncols=len(chaves), handletextpad=0.6,
                     columnspacing=2.2, borderpad=0.0,
                     bbox_to_anchor=(0.01, 0.995),
                     bbox_transform=fig.transFigure)
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=2.2)
    return _salvar(fig, destino, extras=[leg])


def figura_custo(tabela: pd.DataFrame, titulo: str, destino: Path,
                 deitado: bool = False, so_libs: list | None = None) -> str:
    """Razao de custo por par, em escala logaritmica, com IC de 95%.

    ``deitado`` e ``so_libs`` tem o mesmo sentido da figura de equivalencia:
    trocar os eixos e restringir a uma biblioteca rival.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullFormatter

    if so_libs:
        tabela = tabela[tabela["biblioteca"].isin(so_libs)]
        if tabela.empty:
            return ""
    cores = _cores_de(tabela)
    formas = _marcas_de(tabela)
    libs = sorted(tabela["biblioteca"].unique())
    ordem = (tabela.groupby("metodo")["razao"].min()
             .sort_values().index.tolist())
    ys, rotulos, fim = _posicoes(tabela, ordem)

    # limites assimetricos: a figura nao precisa de espaco vazio do lado
    # onde nenhum par chega
    esq = min(float(np.nanmin(tabela["ic_baixo"])) / 1.3, 1 / 1.6)
    dir_ = max(float(np.nanmax(tabela["ic_alto"])) * 1.3, 1.6)

    # mesma caixa e mesma tipografia da figura de erro
    comprimento = 0.50 * len(ordem) + 2.8
    fig, ax = plt.subplots(dpi=200, figsize=(comprimento, 8.4) if deitado
                           else (10.0, comprimento))
    fig.patch.set_facecolor(FUNDO)
    _estilo(ax, fonte=16)
    if deitado:
        ax.set_yscale("log", base=2)
        ax.grid(axis="x", visible=False)
        ax.grid(axis="y", color=GRADE, lw=0.7)
    else:
        ax.set_xscale("log", base=2)

    faixa = ax.axhspan if deitado else ax.axvspan
    linha_valor = ax.axhline if deitado else ax.axvline
    banda_pos = ax.axvspan if deitado else ax.axhspan

    faixa(esq, 1.0, color=VERDE, alpha=0.07, lw=0, zorder=0)
    linha_valor(1.0, color=FRACO, lw=1.6, zorder=1)

    # faixas alternadas levam o olho do nome do metodo ate o ponto
    for i, (pos, _) in enumerate(rotulos):
        if i % 2 == 0:
            banda_pos(pos - 0.5, pos + 0.5, color="#f4f4f4", lw=0, zorder=-1)

    for _, r in tabela.iterrows():
        p = ys[(r["metodo"], r["biblioteca"])]
        cor = cores[r["biblioteca"]]
        # cheio quando o intervalo fica inteiro de um lado de 1, vazado
        # quando cruza: o lado ja esta dito pela posicao
        firme = bool(r["ganha"]) or bool(r["perde"])
        extremos, meio = [r["ic_baixo"], r["ic_alto"]], [r["razao"]]
        barra = ([p, p], extremos) if deitado else (extremos, [p, p])
        ponto = ([p], meio) if deitado else (meio, [p])
        # mesma gramatica da figura de erro: a forma diz a biblioteca
        ax.plot(*barra, color=cor, lw=3.0, solid_capstyle="round", zorder=3)
        ax.plot(*ponto, marker=formas[r["biblioteca"]], ms=10.0,
                color=cor if firme else FUNDO, mec=cor, mew=2.4, zorder=4)

    marcas = [m for m in (2.0 ** e for e in range(-8, 9))
              if esq <= m <= dir_]
    nomes_marcas = [f"1/{int(round(1 / m))}" if m < 1 else f"{int(round(m))}"
                    for m in marcas]
    posicoes = [p for p, _ in rotulos]
    nomes = [m for _, m in rotulos]
    rotulo_valor = f"{titulo} da mlquantify, em múltiplos da outra biblioteca"

    if deitado:
        ax.yaxis.set_major_locator(FixedLocator(marcas))
        ax.set_yticklabels(nomes_marcas, fontsize=16)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(esq, dir_)
        ax.set_xlim(1.05, fim + 0.5)
        ax.set_xticks(posicoes)
        ax.set_xticklabels(nomes, fontsize=14, color=TINTA, rotation=90)
        ax.set_ylabel(rotulo_valor, fontsize=17, color=TINTA, labelpad=12)
        for texto, alvo, va in ((r"mais cara $\uparrow$", dir_ * 0.96, "top"),
                                (r"$\downarrow$ mlquantify mais barata",
                                 esq * 1.04, "bottom")):
            ax.annotate(texto, xy=(0.012, alvo), fontsize=15, color=FRACO,
                        ha="left", va=va, xycoords=ax.get_yaxis_transform(),
                        bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2))
    else:
        ax.xaxis.set_major_locator(FixedLocator(marcas))
        ax.set_xticklabels(nomes_marcas, fontsize=16)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(esq, dir_)
        ax.set_ylim(fim + 0.5, 1.05)
        ax.set_yticks(posicoes)
        ax.set_yticklabels(nomes, fontsize=16, color=TINTA)
        ax.set_xlabel(rotulo_valor, fontsize=17, color=TINTA, labelpad=12)
        ax.annotate(r"$\leftarrow$ mlquantify mais barata",
                    xy=(esq * 1.04, 0.72), fontsize=15, color=FRACO,
                    ha="left", va="center",
                    bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2))
        ax.annotate(r"mais cara $\rightarrow$", xy=(dir_ * 0.96, 0.72),
                    fontsize=15, color=FRACO, ha="right", va="center",
                    bbox=dict(facecolor=FUNDO, edgecolor="none", pad=2))

    chaves = [
        Line2D([], [], color=TINTA, lw=0, marker="o", ms=10,
               label="cheio: IC todo de um lado de 1"),
        Line2D([], [], color=TINTA, lw=0, marker="o", ms=10, mfc=FUNDO,
               mew=2.4, label="vazado: IC cruza 1"),
        Line2D([], [], color=TINTA, lw=3.0, label="IC de 95%"),
    ]
    legendas = _legenda_em_duas_linhas(
        fig, ax, _marcadores_de(cores, formas, libs), chaves)
    return _salvar(fig, destino, extras=legendas)


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
                erro: str = "mae", svg_custo: list | None = None,
                svg_ba: str | None = None,
                svg_rank: str | None = None) -> str:
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
<figure>{svg or ""}<figcaption>Cada linha e um metodo contra uma
biblioteca. O ponto e a diferenca mediana de {erro.upper()}
(Hodges-Lehmann) e a barra e o intervalo de confianca de 95% dessa
diferenca. A faixa verde e a margem de equivalencia: quando o intervalo
cabe inteiro nela (marcador cheio), a compatibilidade esta demonstrada, e
nao apenas nao refutada; o marcador vazado marca o intervalo que escapa. A
forma do marcador diz a biblioteca, para a figura se ler tambem impressa
em preto e branco.</figcaption></figure>
{_tabela_teste(teste, margem)}
<p class="nota">Unidade de analise: o conjunto de dados. O erro do mesmo
conjunto nos quatro tamanhos de amostra nao sao observacoes independentes,
entao entram pela media e cada conjunto conta uma vez. O teste e pareado
dentro do conjunto, o que tira a dificuldade do conjunto da conta, e nao
supoe normalidade. &quot;Diferenca irrelevante&quot; e o par em que os dois
testes rejeitam: existe diferenca sistematica, e ela cabe dentro da margem.
&quot;Inconclusivo&quot; e falta de evidencia nos dois sentidos, nao
igualdade.</p>"""

    if svg_ba:
        bloco_teste += f"""
<h2>Onde a discordancia aparece</h2>
<p class="sub">O grafico acima resume cada par num ponto. Este abre os
conjuntos e responde o que aquele nao responde: a discordancia e constante
ou cresce com a dificuldade? Cada circulo e um conjunto de dados; no eixo x
esta o nivel do erro (a media das duas bibliotecas) e no y a diferenca
entre elas. O eixo y e linear dentro da faixa de equivalencia e logaritmico
fora dela, porque as diferencas se amontoam perto de zero. O
<code>rho</code> em cada painel e a correlacao de Spearman entre o tamanho
da discordancia e o nivel do erro: positivo alto quer dizer que as duas so
divergem nos conjuntos dificeis.</p>
<figure>{svg_ba}<figcaption>Os pares de maior espalhamento, que sao os
unicos em que ha o que ver. A linha cheia e a
diferenca mediana (Hodges-Lehmann, a mesma do grafico anterior) e as
tracejadas sao os percentis 2,5 e 97,5 dos conjuntos -- os limites de
concordancia, em versao distribuicao-livre, porque a cauda e pesada demais
para media mais ou menos 1,96 desvios.</figcaption></figure>"""

    if svg_custo:
        bloco_teste += f"""
<h2>Custo: quanto a mlquantify cobra em relacao a outra</h2>
<p class="sub">Mesma leitura do grafico acima, agora com o objetivo
invertido: aqui interessa ficar <b>a esquerda</b> de 1. A escala e
logaritmica, porque so nela 2x mais caro e metade do custo ficam a mesma
distancia de 1. Cada observacao e um conjunto de dados, pareado entre as
duas bibliotecas.</p>
{"".join(f'<figure>{s}<figcaption>{html.escape(c)}</figcaption></figure>'
         for s, c in svg_custo)}"""

    if svg_rank:
        bloco_teste += f"""
<h2>Ranking dos pares metodo x biblioteca</h2>
<p class="sub">As figuras acima comparam a mlquantify contra uma rival de
cada vez, no mesmo metodo. Esta junta as duas variaveis num tratamento so
-- o par <code>metodo/biblioteca</code> -- e ordena todos ao mesmo tempo,
que e a pergunta de quem vai escolher o que usar. Cada bloco e um conjunto
de dados: dentro dele os pares sao ordenados, o posto 1 fica com o melhor,
e a caixa mostra a distribuicao dos postos daquele par entre os conjuntos.
Cor e hachura dizem a biblioteca.</p>
<figure>{svg_rank}<figcaption>Caixa <b>baixa e curta</b> e bom e constante;
baixa e comprida e bom em media e instavel -- informacao que a media de
erro esconde. Os pares vao ordenados pelo posto mediano. Ordenar exige
bloco completo, entao os pares que nao foram medidos em pelo menos 95% dos
conjuntos ficam de fora, e saem nomeados na saida do
script.</figcaption></figure>
<p class="nota">O posto e <b>relativo ao bloco</b>: diz quem foi melhor
naquele conjunto, nunca por quanto -- uma vitoria por 1% e uma por 100x
valem o mesmo posto. Para o tamanho da diferenca servem as florestas de
razao e os perfis de desempenho. E o conjunto de pares comparados nao e
neutro: o posto de cada par depende de quais outros entraram, de modo que
o ranking de erro e o de custo nao se somam num numero unico.</p>"""

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
    p.add_argument("--erro", default="mae", choices=tuple(ERROS),
                   help="medida de erro do teste (padrao: mae)")
    p.add_argument("--sem-variantes", action="store_true",
                   help="nao gera as versoes deitada e por biblioteca das "
                        "figuras de equivalencia, razao e ranking")
    p.add_argument("--piso-cobertura", type=float, default=0.95, metavar="F",
                   help="fracao minima de conjuntos que um par metodo x "
                        "biblioteca precisa ter para entrar no ranking "
                        "(padrao: 0,95)")
    p.add_argument("--pares-concordancia", type=int, default=8, metavar="N",
                   help="quantos pares entram no grafico de concordancia, "
                        "do maior espalhamento para o menor (padrao: 8)")
    p.add_argument("--margem", type=float, default=None, metavar="X",
                   help="margem de equivalencia do TOST; sem ela, a da "
                        "medida de erro escolhida (MAE 0,01; MSE 0,0001)")
    p.add_argument("--sem-teste", action="store_true",
                   help="so o painel de custo, sem o teste e o grafico")
    args = p.parse_args()

    if args.margem is None:
        if args.erro not in MARGENS:
            raise SystemExit(
                f"nao ha margem definida para {args.erro.upper()}: um ponto "
                f"percentual de prevalencia nao se transpoe para um erro "
                f"relativo nem para uma divergencia.\n"
                f"Escolha uma com --margem, antes de olhar os dados, e diga "
                f"no texto de onde ela vem.")
        args.margem = MARGENS[args.erro]

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

    teste, svg, svg_ba, svg_custo, svg_rank = None, None, None, [], None
    if not args.sem_teste:
        teste = testar(pares, erro=args.erro, margem=args.margem)
        if not teste.empty:
            imprimir_teste(teste, args.margem, args.erro)
            png = destino.with_name(destino.stem + "_equivalencia.png")
            svg = figura_equivalencia(teste, args.margem, args.erro, png)
            print(f"\ngrafico: {png}")
            teste.to_csv(destino.with_name(destino.stem + "_equivalencia.csv"),
                         index=False)
            for sufixo, extra in (() if args.sem_variantes else
                                  variantes(sorted(teste["biblioteca"]
                                                   .unique()))):
                figura_equivalencia(
                    teste, args.margem, args.erro,
                    destino.with_name(
                        f"{destino.stem}_equivalencia_{sufixo}.png"), **extra)

            png = destino.with_name(destino.stem + "_concordancia.png")
            dif = diferencas(pares, args.erro)
            svg_ba = figura_bland_altman(
                dif, teste, args.margem, args.erro, png,
                quantos=args.pares_concordancia)
            if svg_ba:
                print(f"grafico: {png}")
            # os pontos da figura, um por (par, conjunto)
            dif = dif.assign(nivel=(dif[f"{args.erro}_minha"]
                                    + dif[f"{args.erro}_dela"]) / 2.0)
            dif.to_csv(destino.with_name(destino.stem + "_concordancia.csv"),
                       index=False)

        for coluna, nome, legenda in (
            ("tempo_estimacao_total_s", "tempo",
             "Tempo de estimacao: o total das 400 estimativas de cada "
             "combinacao. O ponto e a razao mediana entre conjuntos e a "
             "barra e o intervalo de confianca de 95%; circulo cheio marca "
             "o par em que o intervalo inteiro fica abaixo de 1, isto e, "
             "em que a vantagem da mlquantify e estatisticamente firme."),
            ("memoria_estimacao_mb", "memoria",
             "Memoria na estimacao, maximo alocado alem da linha de base. "
             "Mesma leitura do grafico do tempo."),
        ):
            legenda_perfil = (
                "Perfil de desempenho (Dolan-More) de {titulo}. Cada problema "
                "e um par (metodo, conjunto de dados); o eixo x e o fator em "
                "relacao a biblioteca mais barata naquele problema e o y e a "
                "fracao de problemas dentro desse fator. O valor em 1 e a "
                "fracao de problemas em que a biblioteca e a mais barata de "
                "todas, e onde a curva chega ao topo e o pior caso dela. "
                "Serve porque media e mediana sao dominadas por poucos "
                "problemas caros, e esta leitura nao e.")
            tabela = testar_custo(pares, coluna)
            if tabela.empty:
                continue
            titulo = {"tempo": "Tempo de estimacao",
                      "memoria": "Memoria na estimacao"}[nome]
            png = destino.with_name(f"{destino.stem}_{nome}.png")
            svg_custo.append((figura_custo(tabela, titulo, png), legenda))
            tabela.to_csv(destino.with_name(f"{destino.stem}_{nome}.csv"),
                          index=False)
            for sufixo, extra in (() if args.sem_variantes else
                                  variantes(sorted(tabela["biblioteca"]
                                                   .unique()))):
                figura_custo(tabela, titulo,
                             destino.with_name(
                                 f"{destino.stem}_{nome}_{sufixo}.png"),
                             **extra)
            print(f"grafico: {png}")
            imprimir_custo(tabela, titulo)

            png = destino.with_name(f"{destino.stem}_perfil_{nome}.png")
            svg_perfil = figura_perfil(pares, coluna, png)
            _custo_longo(pares, coluna).to_csv(
                destino.with_name(f"{destino.stem}_perfil_{nome}.csv"),
                index=False)
            if svg_perfil:
                svg_custo.append((svg_perfil, legenda_perfil.format(
                    titulo=titulo.lower())))
                print(f"grafico: {png}")

        # o grafico carrega so o custo, e com as duas fases juntas: o
        # tempo soma, a memoria e o pico, que e o maior dos dois e nao a soma
        medidas = [(("tempo_treino_s", "tempo_estimacao_total_s"),
                    "Tempo total (treino + estimacao)", "soma"),
                   (("memoria_treino_mb", "memoria_estimacao_mb"),
                    "Memoria (pico entre treino e estimacao)", "max")]
        for colunas, titulo, juntar in medidas:
            postos, fora, n_blocos = postos_dos_pares(
                ok, pares, colunas, juntar, args.piso_cobertura)
            imprimir_ranking(postos, fora, n_blocos, titulo)
            if postos is not None:
                resumo_postos = pd.DataFrame({
                    "par": postos.columns,
                    "metodo": [c.split("/")[0] for c in postos.columns],
                    "biblioteca": [c.split("/")[1] for c in postos.columns],
                    "posto_medio": postos.mean().to_numpy(),
                    "posto_mediano": postos.median().to_numpy(),
                    "posto_p25": postos.quantile(0.25).to_numpy(),
                    "posto_p75": postos.quantile(0.75).to_numpy(),
                    "n_conjuntos": n_blocos,
                }).sort_values("posto_mediano")
                curto = colunas[0].split("_")[0]
                resumo_postos.to_csv(
                    destino.with_name(f"{destino.stem}_ranking_{curto}.csv"),
                    index=False)
        # o erro nao entra no grafico, mas o ranking dele sai impresso
        imprimir_ranking(*postos_dos_pares(ok, pares, args.erro, "soma",
                                           args.piso_cobertura),
                         args.erro.upper())
        png = destino.with_name(destino.stem + "_ranking.png")
        svg_rank = figura_ranking(ok, pares, medidas, png,
                                  args.piso_cobertura)
        if svg_rank:
            print(f"grafico: {png}")
        if not args.sem_variantes:
            todas = sorted(ok["biblioteca"].unique())
            for sufixo, extra in variantes(todas):
                figura_ranking(ok, pares, medidas,
                               destino.with_name(
                                   f"{destino.stem}_ranking_{sufixo}.png"),
                               args.piso_cobertura, **extra)
            print(f"variantes: deitada e uma por biblioteca, "
                  f"em {destino.parent}")

    destino.write_text(
        html_painel(resumo, pares, args.run_id, args.custo_run,
                    teste=teste, svg=svg, margem=args.margem,
                    erro=args.erro, svg_custo=svg_custo, svg_ba=svg_ba,
                    svg_rank=svg_rank),
        encoding="utf-8")
    print(f"painel: {destino}")
    if args.csv:
        resumo.to_csv(args.csv, index=False)
        print(f"resumo: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
