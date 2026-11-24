"""Q2: razao de tempo contra razao de memoria, um grafico por biblioteca
de referencia, para entrar como subfiguras (a), (b) e (c) no LaTeX.

Cada ponto e um metodo comparado com aquela biblioteca, com a razao mediana
do tempo total no eixo x e a do pico de memoria no eixo y, as duas em log2 e
lidas dos csv que o gerar.py ja grava (recorte de 110 conjuntos). As linhas
em 1 dividem cada painel em quatro quadrantes: no de baixo a esquerda a
mlquantify e mais rapida e ocupa menos memoria; no de cima a direita, o
contrario. Os tres paineis dividem os mesmos eixos, para que a distancia
entre as bibliotecas se leia direto.

Aqui nao ha a marca cheia e vazada das outras figuras. Cada ponto carrega
duas medidas e uma marca so nao da conta de duas decisoes: nenhum dos 39
pares tem os dois intervalos contendo 1, entao preencher pelo conjunto
vazaria o grafico inteiro, e tracejar o intervalo da medida que nao difere
nao se distingue da barra cortada pelo marcador. Quem leva a decisao por
ponto e a figura das fases, onde cada ponto e uma medida so. O script
imprime a contagem para o texto: 6 pares sem diferenca no tempo e 1 na
memoria.

Usa so csv e matplotlib, sem pandas.
"""
import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PASTA = Path(__file__).resolve().parent

#: Ordem dos paineis e nome exibido de cada biblioteca.
LIBS = [("quapy", "QuaPy"), ("quantificationlib", "QuantificationLib"),
        ("qunfold", "qunfold")]
CORES = {"quapy": "#2b6cb0", "quantificationlib": "#b7791f", "qunfold": "#1d5c46"}
MARCAS = {"quapy": "o", "quantificationlib": "s", "qunfold": "^"}
TINTA, FRACO, GRADE = "#1a1a1a", "#6b6b6b", "#d9d9d9"
VERDE_CLARO, VERMELHO_CLARO = "#e8f2ec", "#f6e9e7"

def espalhar(fig, ax, rotulos, pontos, obstaculos=None, passos=600):
    """Afasta rotulos que se sobrepoem entre si ou cobrem um ponto.

    Trabalha em pixels: a cada passo, cada par de caixas que se cruza e
    empurrado em sentidos opostos, sobretudo na vertical, e cada caixa que
    cobre um marcador e empurrada para longe dele. Rotulos que terminam longe
    do ponto ganham uma linha de chamada.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    desloc = [np.array(r.xyann, dtype=float) for r in rotulos]
    escala = fig.dpi / 72.0  # pontos -> pixels
    pts = [ax.transData.transform(p) for p in pontos]
    fixos = [f.get_window_extent(rend).padded(4) for f in (obstaculos or [])]
    area = ax.get_window_extent(rend)
    for _ in range(passos):
        # folga de alguns pixels, para nomes vizinhos nao encostarem
        caixas = [r.get_window_extent(rend).padded(3) for r in rotulos]
        mexeu = False
        for i in range(len(rotulos)):
            bi = caixas[i]
            for j in range(i + 1, len(rotulos)):
                bj = caixas[j]
                if not bi.overlaps(bj):
                    continue
                sobre_y = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                sentido = 1.0 if (bi.y0 + bi.y1) >= (bj.y0 + bj.y1) else -1.0
                passo = (sobre_y / 2 + 0.5) / escala
                desloc[i][1] += sentido * passo
                desloc[j][1] -= sentido * passo
                lado = 1.0 if (bi.x0 + bi.x1) >= (bj.x0 + bj.x1) else -1.0
                desloc[i][0] += lado * 0.4
                desloc[j][0] -= lado * 0.4
                mexeu = True
            for k, (px, py) in enumerate(pts):
                if bi.x0 - 4 < px < bi.x1 + 4 and bi.y0 - 4 < py < bi.y1 + 4:
                    desloc[i][1] += (1.5 if (bi.y0 + bi.y1) / 2 >= py else -1.5)
                    mexeu = True
            # caixas fixas, como os contadores dos cantos
            for bo in fixos:
                if bi.overlaps(bo):
                    cy = (bo.y0 + bo.y1) / 2
                    desloc[i][1] += (-1.5 if cy > area.y0 + area.height / 2
                                     else 1.5)
                    desloc[i][0] += (-1.5 if (bo.x0 + bo.x1) / 2
                                     > area.x0 + area.width / 2 else 1.5)
                    mexeu = True
            # dentro da area do grafico
            if bi.x1 > area.x1 - 2:
                desloc[i][0] -= (bi.x1 - area.x1 + 2) / escala
                mexeu = True
            if bi.x0 < area.x0 + 2:
                desloc[i][0] += (area.x0 + 2 - bi.x0) / escala
                mexeu = True
            if bi.y1 > area.y1 - 2:
                desloc[i][1] -= (bi.y1 - area.y1 + 2) / escala
                mexeu = True
            if bi.y0 < area.y0 + 2:
                desloc[i][1] += (area.y0 + 2 - bi.y0) / escala
                mexeu = True
        for r, d in zip(rotulos, desloc):
            r.xyann = tuple(d)
        if not mexeu:
            break
    for r, d in zip(rotulos, desloc):
        if np.hypot(*d) > 10:
            ax.annotate("", xy=r.xy, xytext=d, textcoords="offset points",
                        arrowprops=dict(arrowstyle="-", color=FRACO, lw=0.45,
                                        shrinkA=0, shrinkB=3), zorder=3)


def ler(nome):
    with io.open(PASTA / nome, encoding="utf-8") as f:
        return {(r["metodo"], r["biblioteca"]): r for r in csv.DictReader(f)}


def l2(v):
    return np.log2(float(v))


def igual(r):
    """Sem diferenca estatistica: o intervalo da razao contem 1."""
    return r["ganha"] != "True" and r["perde"] != "True"


def fracao(e):
    """Rotulo de potencia de 2 como no grafico de razao: 1/8, 1/2, 1, 2."""
    e = int(round(e))
    return str(2 ** e) if e >= 0 else f"1/{2 ** -e}"


tempo, memoria = ler("q2-tempo.csv"), ler("q2-memoria.csv")
pares = sorted(set(tempo) & set(memoria))

# y comum aos tres paineis; x proprio de cada um, porque so a qunfold chega
# a 1/64 e o eixo comum espremeria os outros dois perto de 1
y_min = min(l2(memoria[p]["ic_baixo"]) for p in pares) - 0.3
y_max = max(l2(memoria[p]["ic_alto"]) for p in pares) + 0.45
yt = np.arange(np.ceil(y_min), np.floor(y_max) + 1)

caixa = dict(boxstyle="round,pad=0.3", fc="white", ec=GRADE, lw=0.8)

# Um arquivo por biblioteca. As letras (a), (b) e (c) e o nome da biblioteca
# entram pelas subfiguras no LaTeX, entao o grafico nao leva titulo.
for lib, nome in LIBS:
    fig, ax = plt.subplots(figsize=(3.6, 3.25), dpi=200)
    da_lib = [p for p in pares if p[1] == lib]
    x_min = min(0.0, min(l2(tempo[p]["ic_baixo"]) for p in da_lib)) - 0.7
    x_max = max(0.0, max(l2(tempo[p]["ic_alto"]) for p in da_lib)) + 1.0
    xt = np.arange(np.ceil(x_min), np.floor(x_max) + 1)
    if len(xt) > 7:  # de dois em dois expoentes, sem perder o 1
        xt = xt[(xt % 2 == 0)]

    ax.fill_between([x_min, 0], y_min, 0, color=VERDE_CLARO, zorder=0)
    ax.fill_between([0, x_max], 0, y_max, color=VERMELHO_CLARO, zorder=0)
    ax.axhline(0, color=FRACO, lw=0.9, zorder=1)
    ax.axvline(0, color=FRACO, lw=0.9, zorder=1)

    cor = CORES[lib]
    contagem = {"ll": 0, "lr": 0, "ul": 0, "ur": 0}
    rotulos, pontos = [], []
    for p in da_lib:
        t, mem = tempo[p], memoria[p]
        x, y = l2(t["razao"]), l2(mem["razao"])
        ax.plot([l2(t["ic_baixo"]), l2(t["ic_alto"])], [y, y], color=cor,
                lw=0.9, alpha=0.5, zorder=2)
        ax.plot([x, x], [l2(mem["ic_baixo"]), l2(mem["ic_alto"])], color=cor,
                lw=0.9, alpha=0.5, zorder=2)
        ax.scatter(x, y, s=26, marker=MARCAS[lib], color=cor,
                   edgecolor="white", linewidth=0.6, zorder=3)
        contagem[("u" if y > 0 else "l") + ("r" if x > 0 else "l")] += 1
        rotulos.append(ax.annotate(p[0], (x, y), xytext=(3, 3),
                                   textcoords="offset points", fontsize=7,
                                   color=TINTA, zorder=4))
        pontos.append((x, y))

    cantos = []
    for chave, cx, cy, ha, va in (("ll", x_min, y_min, "left", "bottom"),
                                  ("lr", x_max, y_min, "right", "bottom"),
                                  ("ul", x_min, y_max, "left", "top"),
                                  ("ur", x_max, y_max, "right", "top")):
        cantos.append(ax.text(cx + (0.1 if ha == "left" else -0.1),
                cy + (0.06 if va == "bottom" else -0.06),
                str(contagem[chave]), ha=ha, va=va, fontsize=8,
                color=FRACO, bbox=caixa, zorder=5))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(xt, [fracao(e) for e in xt])
    ax.set_yticks(yt, [fracao(e) for e in yt])
    ax.grid(color=GRADE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA, labelsize=8, length=0)
    ax.set_xlabel("Tempo", fontsize=9, color=TINTA, labelpad=4)
    ax.set_ylabel("Pico de memória", fontsize=9, color=TINTA, labelpad=4)

    fig.tight_layout()
    espalhar(fig, ax, rotulos, pontos, obstaculos=cantos)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(PASTA / f"q2-tempo-memoria-{lib}.{ext}",
                    bbox_inches="tight")
    plt.close(fig)
    print(f"{nome}: {len(da_lib)} metodos, quadrantes {contagem}, "
          f"sem diferenca no tempo {sum(igual(tempo[p]) for p in da_lib)}, "
          f"na memoria {sum(igual(memoria[p]) for p in da_lib)}")
