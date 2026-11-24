"""Q1 em paineis: a floresta do teste de equivalencia, uma por biblioteca,
deitada (metodos no eixo horizontal, diferenca no vertical).

Refaz o teste do painel_bibliotecas.py (diferenca pareada por conjunto,
estimador de Hodges-Lehmann e seu IC de 95%) sobre o recorte de 110
conjuntos que o gerar.py usa, sem pandas, para o AE e para o RAE, e desenha
um grafico por biblioteca de referencia para entrar como subfiguras no
LaTeX: a QuaPy (20 metodos) em cima e a QuantificationLib (12) e a qunfold
(7) lado a lado embaixo, que somam quase a mesma largura.

Todos os graficos de uma medida tem a mesma escala vertical e a mesma
largura por metodo, com margens fixas em polegadas. Incluidos no LaTeX com o
mesmo ``scale``, as tres escalas coincidem.

O marcador cheio marca o par sem diferenca estatistica e o vazado o par que
difere. Uma legenda no topo do grafico da QuaPy, que fica em cima no LaTeX,
explica os elementos e de qual teste cada um vem (TOST ou Wilcoxon).
No AE a decisao e a do TOST contra a margem de 0,01 (tracejado): o
marcador fica cheio quando o intervalo cabe inteiro dentro dela. No RAE nao
ha margem declarada -- um ponto percentual nao se transpoe para um erro
relativo --, entao a decisao e a do teste de Wilcoxon, com a correcao de
Holm entre os 39 pares: cheio quando nao ha diferenca (p ajustado >= 0,05).
O intervalo contendo zero nao serve de criterio, porque pares identicos tem
intervalo [0, 0] e seriam marcados como diferentes.
"""
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

PASTA = Path(__file__).resolve().parent
RAIZ = Path("C:/Users/luiz_/OneDrive/Codigos/exp-mlquantify")
sys.path.insert(0, str(RAIZ))
import recorte_analise  # noqa: E402  precisa do RAIZ no sys.path
SUMMARY = Path("C:/Users/luiz_/exp-mlquantify-run/results/final-pareado/summary.csv")
OFICIAIS = SUMMARY.with_name("painel_bibliotecas_equivalencia.csv")
MARGEM = 0.01

LIBS = [("quapy", "QuaPy"), ("quantificationlib", "QuantificationLib"),
        ("qunfold", "qunfold")]
CORES = {"quapy": "#2b6cb0", "quantificationlib": "#b7791f", "qunfold": "#1d5c46"}
MARCAS = {"quapy": "o", "quantificationlib": "s", "qunfold": "^"}
TINTA, FRACO, GRADE = "#1a1a1a", "#6b6b6b", "#d9d9d9"
VERDE, FUNDO = "#2f7a5a", "#ffffff"

#: (coluna do summary, prefixo do arquivo, rotulo do eixo, tem margem)
MEDIDAS = [("mae", "q1-equivalencia", "Diferença de MAE", True),
           ("mrae", "q1-rae", "Diferença de MRAE", False)]


# ------------------------------------------------------------------ dados
def ler_linhas():
    fora = recorte_analise.excluidos()
    linhas = []
    with io.open(SUMMARY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and r["dataset_id"] not in fora:
                linhas.append(r)
    # nomes equivalentes do metodo, como o expand_aliases
    existentes = {(r["biblioteca"], r["metodo"]) for r in linhas}
    for r in list(linhas):
        for alias in filter(None, (a.strip() for a in r.get("aliases", "")
                                   .replace(";", ",").replace("|", ",")
                                   .split(","))):
            if (r["biblioteca"], alias) not in existentes:
                linhas.append(dict(r, metodo=alias))
    return linhas


def diferencas(linhas, coluna):
    """Uma diferenca da medida por (metodo, biblioteca, conjunto)."""
    minha, outras = defaultdict(list), defaultdict(list)
    for r in linhas:
        if r[coluna] in ("", "nan"):
            continue
        chave = (r["dataset_id"], r["metodo"], r["tamanho_amostra"])
        if r["biblioteca"] == "mlquantify":
            minha[chave].append(float(r[coluna]))
        else:
            outras[(r["biblioteca"],) + chave].append(float(r[coluna]))
    por_conjunto = defaultdict(list)
    for (lib, ds, met, tam), vals in outras.items():
        m = minha.get((ds, met, tam))
        if m:  # mediana das repeticoes, media dos tamanhos de amostra
            por_conjunto[(met, lib, ds)].append(np.median(m) - np.median(vals))
    por_par = defaultdict(list)
    for (met, lib, _), d in por_conjunto.items():
        por_par[(met, lib)].append(float(np.mean(d)))
    with io.open(OFICIAIS, newline="", encoding="utf-8") as f:
        oficiais = {(r["metodo"], r["biblioteca"]) for r in csv.DictReader(f)}
    return {k: np.asarray(v) for k, v in por_par.items() if k in oficiais}


def hodges_lehmann(d):
    """Mesmo calculo do painel_bibliotecas.hodges_lehmann."""
    n = len(d)
    soma = d[:, None] + d[None, :]
    i, j = np.triu_indices(n)
    w = np.sort(soma[i, j] / 2.0)
    hl = float(np.median(w))
    z = norm.ppf(0.975)
    k = int(np.floor(n * (n + 1) / 4
                     - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24) - 0.5))
    if k < 0:
        return hl, float(w[0]), float(w[-1])
    k = min(k, len(w) - 1)
    return hl, float(w[k]), float(w[len(w) - 1 - k])


def holm(p):
    p = np.asarray(p, dtype=float)
    ajust, acum = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        acum = max(acum, (len(p) - r) * p[i])
        ajust[i] = min(1.0, acum)
    return ajust


def testar(linhas, coluna):
    from scipy.stats import wilcoxon
    teste = {}
    for par, d in diferencas(linhas, coluna).items():
        d = d[np.isfinite(d)]
        hl, baixo, alto = hodges_lehmann(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        teste[par] = dict(hl=hl, baixo=baixo, alto=alto, n=len(d), p=p,
                          dentro=(baixo > -MARGEM) and (alto < MARGEM),
                          contem_zero=bool(baixo < 0 < alto))
    # decisao do Wilcoxon com a correcao de Holm entre os 39 pares
    pares = list(teste)
    for par, ph in zip(pares, holm([teste[k]["p"] for k in pares])):
        teste[par]["p_holm"] = ph
        teste[par]["sem_diferenca"] = bool(ph >= 0.05)
    return teste


# ------------------------------------------------------------------ figura
COLUNA = 0.25                         # polegadas por metodo, em todos
ESQ, DIR, BAIXO, CIMA = 0.88, 0.08, 0.74, 0.08
ALTURA = 1.95                         # altura total de cada grafico
LEGENDA = 0.62                        # faixa extra no topo do grafico da QuaPy
virgula = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")
                        .replace("-", "\u2212"))


def legenda(fig, ax, com_margem):
    """Explica o que cada elemento mostra e de qual teste ele vem."""
    neutro = "#4a4a4a"
    def ponto(cheio):
        return Line2D([], [], ls="none", marker="o", ms=5.5, mew=1.3,
                      mec=neutro, color=neutro if cheio else FUNDO)
    barra = Line2D([], [], ls="none", marker="|", ms=11, mew=1.6, color=neutro)
    if com_margem:
        itens = [
            (ponto(True), "Equivalente"),
            (ponto(False), "Não equivalente"),
            ((Patch(color=VERDE, alpha=0.12, lw=0),
              Line2D([], [], color=VERDE, lw=1.1, ls=(0, (4, 3)))),
             "Margem de equivalência (\u00b10,01)"),
            (barra, "Intervalo de confiança de 95%"),
        ]
    else:
        itens = [
            (ponto(True), "Sem diferença"),
            (ponto(False), "Com diferença"),
            (barra, "Intervalo de confiança de 95%"),
        ]
    caixa = ax.get_position()
    fig.legend([h for h, _ in itens], [t for _, t in itens], ncol=2 if len(itens) == 4 else len(itens),
               loc="lower center", frameon=False, fontsize=8,
               labelcolor=TINTA, handlelength=1.8, columnspacing=1.2,
               bbox_to_anchor=(0.5, caixa.y1 + 0.015),
               borderaxespad=0.2)


def desenhar(teste, prefixo, rotulo, com_margem):
    extremo = max(max(abs(t["baixo"]), abs(t["alto"])) for t in teste.values())
    limite = max(MARGEM * 1.12, extremo * 1.06) if com_margem else extremo * 1.1
    for lib, nome in LIBS:
        # a mesma ordem nas duas medidas: pelo |HL| do AE, guardada abaixo
        da_lib = sorted(((m, t) for (m, l), t in teste.items() if l == lib),
                        key=lambda mt: ORDEM[(mt[0], lib)])
        n = len(da_lib)
        larg = ESQ + DIR + COLUNA * n
        # a legenda vale para os tres graficos e entra so no da QuaPy, que
        # fica em cima; a altura do eixo continua a mesma nos tres
        extra = LEGENDA if lib == "quapy" else 0.0
        alt = ALTURA + extra
        fig = plt.figure(figsize=(larg, alt), dpi=200)
        fig.patch.set_facecolor(FUNDO)
        ax = fig.add_axes([ESQ / larg, BAIXO / alt,
                           (larg - ESQ - DIR) / larg,
                           (ALTURA - BAIXO - CIMA) / alt])
        if com_margem:
            ax.axhspan(-MARGEM, MARGEM, color=VERDE, alpha=0.07, lw=0, zorder=0)
            for lado in (-MARGEM, MARGEM):
                ax.axhline(lado, color=VERDE, lw=1.1, ls=(0, (4, 3)), zorder=2)
        ax.axhline(0, color=FRACO, lw=0.8, zorder=1)

        cor, forma = CORES[lib], MARCAS[lib]
        for i, (metodo, t) in enumerate(da_lib):
            # com margem, difere quem sai dela (TOST); sem margem, difere
            # quem tem o intervalo todo de um lado do zero
            fora = not (t["dentro"] if com_margem else t["sem_diferenca"])
            if fora:
                ax.axvspan(i - 0.5, i + 0.5, color="#fbeaea", lw=0, zorder=-1)
            elif i % 2 == 0:
                ax.axvspan(i - 0.5, i + 0.5, color="#f4f4f4", lw=0, zorder=-1)
            ax.plot([i, i], [t["baixo"], t["alto"]], color=cor, lw=1.6,
                    solid_capstyle="round", zorder=3)
            ax.plot([i], [t["hl"]], marker=forma, ms=5.5, mew=1.3, mec=cor,
                    color=FUNDO if fora else cor, zorder=4)

        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(-limite, limite)
        ax.set_xticks(range(n), [m for m, _ in da_lib], rotation=90)
        if com_margem:
            ax.set_yticks([-MARGEM, -MARGEM / 2, 0, MARGEM / 2, MARGEM])
        ax.yaxis.set_major_formatter(virgula)
        ax.tick_params(colors=TINTA, labelsize=10, length=0)
        for lado in ("top", "right", "bottom"):
            ax.spines[lado].set_visible(False)
        ax.spines["left"].set_color(GRADE)
        ax.grid(axis="y", color=GRADE, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_ylabel(rotulo, fontsize=9.5, color=TINTA, labelpad=3)
        if extra:
            legenda(fig, ax, com_margem)

        for ext in ("png", "pdf", "svg"):
            fig.savefig(PASTA / f"{prefixo}-{lib}.{ext}", facecolor=FUNDO)
        plt.close(fig)
        print(f"  {nome}: {n} metodos, largura {larg:.2f} pol")


linhas = ler_linhas()
ae = testar(linhas, "mae")
ORDEM = {}
for lib, _ in LIBS:
    for pos, (m, _) in enumerate(sorted(
            ((m, t) for (m, l), t in ae.items() if l == lib),
            key=lambda mt: -abs(mt[1]["hl"]))):
        ORDEM[(m, lib)] = pos

for coluna, prefixo, rotulo, com_margem in MEDIDAS:
    teste = ae if coluna == "mae" else testar(linhas, coluna)
    hls = np.array([abs(t["hl"]) for t in teste.values()])
    zero = sum(t["baixo"] < 0 < t["alto"] for t in teste.values())  # estrito, como o gerar.py
    print(f"{coluna}: pares {len(teste)}, |HL| mediano {np.median(hls):.4f}, "
          f"maximo {hls.max():.4f}, IC contendo zero {zero}, "
          f"IC na margem {sum(t['dentro'] for t in teste.values())}, "
          f"sem diferenca pelo Wilcoxon com Holm "
          f"{sum(t['sem_diferenca'] for t in teste.values())}")
    desenhar(teste, prefixo, rotulo, com_margem)
