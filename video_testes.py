"""Video explicando os testes estatisticos do painel de bibliotecas.

Desenha, quadro a quadro, a cadeia que o painel usa para decidir se duas
bibliotecas empatam no erro e quem e mais barata:

    diferenca pareada por conjunto  ->  Hodges-Lehmann  ->  intervalo de
    Wilcoxon  ->  TOST  ->  Holm  ->  razao de custo em log2

Nada aqui recalcula a analise: os numeros exibidos sao os que o
``painel_bibliotecas.py`` produziu na execucao final-pareado, e a amostra
que aparece na tela e um recorte real das diferencas de HDy8 contra a
quantificationlib e de MS2 contra a quapy.

Uso::

    python video_testes.py --saida video/testes_estatisticos.mp4

Precisa de um ffmpeg: usa o ``imageio-ffmpeg`` se estiver instalado, senao
o ffmpeg do PATH.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# paleta, a mesma das figuras do painel
# ---------------------------------------------------------------------------

FUNDO = "#ffffff"
TINTA = "#1a1a1a"
FRACO = "#6b6b6b"
GRADE = "#d9d9d9"
TRILHO = "#9aa0a6"

MINHA = "#6b46c1"          # mlquantify
QUAPY = "#2b6cb0"
QLIB = "#b7791f"
QUNFOLD = "#2f7a5a"
VERDE = "#2f7a5a"          # faixa de equivalencia
VERMELHO = "#9b2c2c"
ROSA = "#fbeaea"
CINZA = "#f0f0f0"

LARG, ALT = 16.0, 9.0      # sistema de coordenadas da cena
FPS = 30

# ---------------------------------------------------------------------------
# dados reais, recortados da execucao final-pareado
# ---------------------------------------------------------------------------

#: 14 conjuntos sorteados entre os 135 de HDy8 x quantificationlib.
HDY8 = np.array([-0.02469, -0.00154, -0.00025, -0.00021, -0.00010, -0.00004,
                 0.00008, 0.00022, 0.00045, 0.00227, 0.00569, 0.00990,
                 0.01471, 0.16564])

#: os 9 do meio, usados na parte que precisa caber na tela.
NOVE = HDY8[1:10]

#: 14 conjuntos sorteados entre os 138 de MS2 x quapy.
MS2 = np.array([-0.06334, -0.03924, -0.02910, -0.02757, -0.00428, -0.00389,
                -0.00340, -0.00294, -0.00269, -0.00238, -0.00045, -0.00009,
                0.00025, 0.00198])

MARGEM = 0.01

# numeros da analise inteira, para as legendas
HDY8_N, HDY8_HL, HDY8_IC = 135, 0.00039, (0.00023, 0.00095)
HDY8_MEDIA = 0.00837
MS2_N, MS2_HL, MS2_IC = 138, -0.00770, (-0.01070, -0.00449)
PARES, DENTRO = 39, 38
TEMPO_GANHA, MEM_GANHA = 18, 28


# ---------------------------------------------------------------------------
# utilidades de animacao
# ---------------------------------------------------------------------------

def suave(x: float) -> float:
    """Aceleracao e freada nas pontas (smoothstep)."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def trecho(t: float, inicio: float, fim: float) -> float:
    """Progresso de 0 a 1 entre dois instantes da cena."""
    if fim <= inicio:
        return 1.0 if t >= fim else 0.0
    return min(max((t - inicio) / (fim - inicio), 0.0), 1.0)


def surge(t: float, inicio: float, dur: float = 0.5) -> float:
    """Opacidade de quem entra em cena em ``inicio``."""
    return suave(trecho(t, inicio, inicio + dur))


def some(t: float, inicio: float, dur: float = 0.5) -> float:
    """Opacidade de quem sai de cena em ``inicio``."""
    return 1.0 - suave(trecho(t, inicio, inicio + dur))


def entre(a, b, u: float):
    """Interpolacao linear entre dois pontos."""
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def texto(ax, x, y, s, *, fs=26, cor=TINTA, alfa=1.0, ha="center",
          va="center", peso="normal", fundo=None):
    if alfa <= 0.01:
        return None
    caixa = None
    if fundo:
        caixa = dict(facecolor=fundo, edgecolor="none", pad=3.0, alpha=alfa)
    return ax.text(x, y, s, fontsize=fs, color=cor, alpha=min(alfa, 1.0),
                   ha=ha, va=va, fontweight=peso, bbox=caixa, zorder=6)


def peca(ax, s, destino, origem, u, **kw):
    """Um pedaco de equacao que voa da figura ate o seu lugar."""
    if u <= 0:
        return
    x, y = entre(origem, destino, suave(min(u, 1.0)))
    texto(ax, x, y, s, alfa=min(1.0, u * 2.2), **kw)


def titulo(ax, s, alfa, *, y=8.25, fs=40):
    texto(ax, LARG / 2, y, s, fs=fs, cor=TINTA, alfa=alfa, peso="bold")


def legenda(ax, s, alfa, *, y=0.72, fs=25, cor=FRACO):
    texto(ax, LARG / 2, y, s, fs=fs, cor=cor, alfa=alfa)


def cena_nova(ax):
    ax.clear()
    ax.set_xlim(0, LARG)
    ax.set_ylim(0, ALT)
    ax.set_facecolor(FUNDO)
    ax.set_xticks([])
    ax.set_yticks([])
    for lado in ax.spines.values():
        lado.set_visible(False)


class Escala:
    """Converte valor de diferenca em posicao horizontal na cena."""

    def __init__(self, vmin, vmax, x0=1.6, x1=14.4):
        self.vmin, self.vmax, self.x0, self.x1 = vmin, vmax, x0, x1

    def __call__(self, v):
        u = (v - self.vmin) / (self.vmax - self.vmin)
        return self.x0 + u * (self.x1 - self.x0)


def trilho(ax, esc, y, marcas, *, alfa=1.0, fmt="{:+.3f}", fs=18, zero=True):
    """Uma reta numerada onde os dados pousam."""
    if alfa <= 0.01:
        return
    ax.plot([esc.x0, esc.x1], [y, y], color=TRILHO, lw=2.0, alpha=alfa,
            zorder=2, solid_capstyle="round")
    for v in marcas:
        x = esc(v)
        ax.plot([x, x], [y - 0.1, y + 0.1], color=TRILHO, lw=2.0, alpha=alfa,
                zorder=2)
        rotulo = "0" if (zero and abs(v) < 1e-12) else fmt.format(v)
        texto(ax, x, y - 0.42, rotulo, fs=fs, cor=FRACO, alfa=alfa)


def ponto(ax, x, y, cor, *, tam=170, alfa=1.0, vazado=False, forma="o",
          borda=2.4, z=5):
    if alfa <= 0.01:
        return
    if vazado:
        ax.scatter([x], [y], s=tam, marker=forma, facecolor=FUNDO,
                   edgecolor=cor, linewidths=borda, alpha=alfa, zorder=z)
    else:
        ax.scatter([x], [y], s=tam, marker=forma, color=cor, alpha=alfa,
                   zorder=z)


def barra_ic(ax, esc, y, baixo, alto, centro, cor, *, alfa=1.0, vazado=False,
             lw=6.0):
    if alfa <= 0.01:
        return
    ax.plot([esc(baixo), esc(alto)], [y, y], color=cor, lw=lw, alpha=alfa,
            solid_capstyle="round", zorder=4)
    ponto(ax, esc(centro), y, cor, tam=260, alfa=alfa, vazado=vazado,
          forma="D" if vazado else "o")


def faixa(ax, esc, y0, y1, margem, alfa=1.0, *, tracejado=True):
    if alfa <= 0.01:
        return
    ax.add_patch(plt.Rectangle((esc(-margem), y0),
                               esc(margem) - esc(-margem), y1 - y0,
                               facecolor=VERDE, alpha=0.07 * alfa, lw=0,
                               zorder=0))
    if tracejado:
        for lado in (-margem, margem):
            ax.plot([esc(lado), esc(lado)], [y0, y1], color=VERDE, lw=2.4,
                    ls=(0, (5, 4)), alpha=alfa, zorder=2)


# ---------------------------------------------------------------------------
# cena 1 - abertura
# ---------------------------------------------------------------------------

def cena_abertura(ax, t):
    cena_nova(ax)
    a = surge(t, 0.3, 0.9)
    titulo(ax, "Como o painel decide se duas bibliotecas empatam", a, y=5.9,
           fs=44)
    texto(ax, LARG / 2, 4.9, "os testes estatisticos, em duas perguntas",
          fs=28, cor=FRACO, alfa=surge(t, 1.2, 0.8))

    u = suave(trecho(t, 1.6, 2.6))
    ax.plot([LARG / 2 - 5.2 * u, LARG / 2 + 5.2 * u], [5.35, 5.35],
            color=MINHA, lw=3.0, alpha=0.85, solid_capstyle="round")

    texto(ax, 4.4, 3.4, "1. o erro e o mesmo?", fs=30, cor=TINTA,
          alfa=surge(t, 2.6, 0.7))
    texto(ax, 11.6, 3.4, "2. o custo e menor?", fs=30, cor=TINTA,
          alfa=surge(t, 3.2, 0.7))
    texto(ax, 4.4, 2.6, "Wilcoxon pareado, Hodges-Lehmann, TOST, Holm",
          fs=21, cor=FRACO, alfa=surge(t, 3.0, 0.7))
    texto(ax, 11.6, 2.6, "a mesma conta, na escala log2",
          fs=21, cor=FRACO, alfa=surge(t, 3.6, 0.7))


# ---------------------------------------------------------------------------
# cena 2 - de onde vem cada diferenca
# ---------------------------------------------------------------------------

def cena_diferenca(ax, t):
    cena_nova(ax)
    titulo(ax, "De onde vem cada numero", surge(t, 0.2))

    # --- os conjuntos de dados, um circulo cada ---
    n = 14
    xs = np.linspace(2.0, 14.0, n)
    for i, x in enumerate(xs):
        a = surge(t, 0.8 + i * 0.07, 0.4)
        ponto(ax, x, 7.3, FRACO if i else MINHA, tam=150, alfa=a * 0.5)
    legenda(ax, "cada circulo e um conjunto de dados; sao 142 no total",
            surge(t, 1.0) * some(t, 3.4, 0.6))

    # --- os quatro tamanhos de amostra viram uma media ---
    a_tam = surge(t, 3.4, 0.6) * some(t, 6.0, 0.6)
    if a_tam > 0.01:
        u = suave(trecho(t, 4.4, 5.6))
        alvo = (xs[0], 7.3)
        for k, m in enumerate((10, 50, 100, 500)):
            orig = (3.2 + k * 1.5, 5.0)
            x, y = entre(orig, alvo, u)
            ax.add_patch(plt.Rectangle((x - 0.22, y - 0.22), 0.44, 0.44,
                                       facecolor=MINHA, alpha=a_tam * (1 - u * 0.8),
                                       lw=0, zorder=4))
            texto(ax, orig[0], 4.4, f"m={m}", fs=18, cor=FRACO,
                  alfa=a_tam * (1 - u))
        legenda(ax, "os quatro tamanhos de amostra do mesmo conjunto "
                    "entram como uma media so", a_tam)

    # --- as duas barras de erro e a subtracao ---
    a_bar = surge(t, 6.0, 0.6) * some(t, 10.6, 0.6)
    if a_bar > 0.01:
        h_minha, h_dela = 1.9, 1.55
        for x, h, cor, nome in ((5.0, h_minha, MINHA, "mlquantify"),
                                (7.2, h_dela, QLIB, "outra biblioteca")):
            cresce = suave(trecho(t, 6.2, 7.0))
            ax.add_patch(plt.Rectangle((x - 0.55, 2.9), 1.1, h * cresce,
                                       facecolor=cor, alpha=a_bar * 0.9, lw=0,
                                       zorder=4))
            texto(ax, x, 2.45, nome, fs=19, cor=FRACO, alfa=a_bar)
        texto(ax, 6.1, 5.3, "erro (MAE) naquele conjunto", fs=21, cor=FRACO,
              alfa=a_bar)

        u = trecho(t, 7.4, 8.6)
        peca(ax, r"$d_i \;=\;$", (9.4, 4.3), (6.1, 3.6), u, fs=34)
        peca(ax, r"$\mathrm{MAE}_{\,\mathrm{mlquantify}}$", (11.3, 4.3),
             (5.0, 3.6), u, fs=30, cor=MINHA)
        peca(ax, r"$-$", (12.9, 4.3), (6.1, 3.6), u, fs=34)
        peca(ax, r"$\mathrm{MAE}_{\,\mathrm{outra}}$", (14.1, 4.3),
             (7.2, 3.6), u, fs=30, cor=QLIB)
        if t > 8.8:
            legenda(ax, "a diferenca de erro, conjunto a conjunto: "
                        "e so isso que entra no teste", surge(t, 8.8))

    # --- a chuva de diferencas no trilho ---
    a_tri = surge(t, 10.6, 0.7)
    esc = Escala(-0.035, 0.175)
    trilho(ax, esc, 4.6, [-0.02, 0.0, 0.05, 0.10, 0.15], alfa=a_tri)
    texto(ax, LARG / 2, 5.6, "as 14 diferencas, uma por conjunto",
          fs=23, cor=FRACO, alfa=a_tri)
    for i, d in enumerate(HDY8):
        u = trecho(t, 10.9 + i * 0.09, 11.6 + i * 0.09)
        if u <= 0:
            continue
        y = 7.3 + (4.6 - 7.3) * suave(u)
        ponto(ax, esc(d), y, QLIB, tam=190, alfa=a_tri * 0.85)
    if t > 13.0:
        legenda(ax, "uma amostra real: HDy8 contra a quantificationlib",
                surge(t, 13.0))


# ---------------------------------------------------------------------------
# cena 3 - a media mente
# ---------------------------------------------------------------------------

def cena_media(ax, t):
    cena_nova(ax)
    titulo(ax, "A media se deixa levar", surge(t, 0.2))

    esc = Escala(-0.035, 0.175)
    trilho(ax, esc, 4.4, [-0.02, 0.0, 0.05, 0.10, 0.15], alfa=surge(t, 0.4))
    for d in HDY8:
        ponto(ax, esc(d), 4.4, QLIB, tam=190, alfa=surge(t, 0.5) * 0.85)

    # o conjunto que puxa tudo
    a_out = surge(t, 1.6, 0.6)
    if a_out > 0.01:
        x = esc(HDY8[-1])
        ax.annotate("", xy=(x, 4.9), xytext=(x, 6.1),
                    arrowprops=dict(arrowstyle="-|>", color=VERMELHO, lw=2.6,
                                    alpha=a_out))
        texto(ax, x, 6.5, "um unico conjunto", fs=23, cor=VERMELHO,
              alfa=a_out)
        texto(ax, x, 6.0, "+0,166", fs=21, cor=VERMELHO, alfa=a_out)

    # a media caminha ate o lugar errado
    a_m = surge(t, 3.0, 0.6)
    if a_m > 0.01:
        u = suave(trecho(t, 3.2, 4.6))
        x = esc(0.0) + (esc(HDY8_MEDIA) - esc(0.0)) * u
        ax.plot([x, x], [3.5, 4.0], color=VERMELHO, lw=3.0, alpha=a_m)
        ponto(ax, x, 3.35, VERMELHO, tam=240, alfa=a_m, forma="v")
        texto(ax, x, 2.9, "media", fs=22, cor=VERMELHO, alfa=a_m)
        texto(ax, x, 2.45, f"{HDY8_MEDIA:+.5f}".replace(".", ","), fs=20,
              cor=VERMELHO, alfa=a_m)

    a_h = surge(t, 5.4, 0.6)
    if a_h > 0.01:
        x = esc(HDY8_HL)
        ax.plot([x, x], [4.8, 5.3], color=MINHA, lw=3.0, alpha=a_h)
        ponto(ax, x, 5.45, MINHA, tam=240, alfa=a_h)
        texto(ax, x - 0.3, 5.9, "diferenca tipica", fs=22, cor=MINHA,
              alfa=a_h, ha="right")
        texto(ax, x - 0.3, 5.45, f"{HDY8_HL:+.5f}".replace(".", ","), fs=20,
              cor=MINHA, alfa=a_h, ha="right")

    legenda(ax, "a media diz que a mlquantify erra 21 vezes mais do que "
                "realmente erra", surge(t, 4.8) * some(t, 8.4, 0.6))
    legenda(ax, "por isso nada aqui usa media: o teste trabalha com postos "
                "e medianas", surge(t, 8.8), cor=TINTA)


# ---------------------------------------------------------------------------
# cena 4 - Hodges-Lehmann
# ---------------------------------------------------------------------------

def _walsh(d):
    soma = d[:, None] + d[None, :]
    i, j = np.triu_indices(len(d))
    return np.sort(soma[i, j] / 2.0)


PARES_NOVE = [(i, j) for i in range(len(NOVE)) for j in range(i, len(NOVE))]
WALSH_NOVE = _walsh(NOVE)


def cena_hl(ax, t):
    cena_nova(ax)
    titulo(ax, "A diferenca tipica", surge(t, 0.2))

    esc = Escala(-0.0022, 0.0030, x0=2.0, x1=14.0)
    trilho(ax, esc, 6.4, [-0.002, -0.001, 0.0, 0.001, 0.002, 0.003],
           alfa=surge(t, 0.4), fs=17)
    for d in NOVE:
        ponto(ax, esc(d), 6.4, QLIB, tam=200, alfa=surge(t, 0.5) * 0.9)
    texto(ax, LARG / 2, 7.3, "9 conjuntos, para caber na tela", fs=20,
          cor=FRACO, alfa=surge(t, 0.6) * some(t, 4.0, 0.6))

    # as medias de dois a dois
    a_par = surge(t, 1.6, 0.5)
    por_par = 0.085
    if a_par > 0.01:
        for k, (i, j) in enumerate(PARES_NOVE):
            u = trecho(t, 1.8 + k * por_par, 2.5 + k * por_par)
            if u <= 0:
                continue
            xi, xj = esc(NOVE[i]), esc(NOVE[j])
            meio = (NOVE[i] + NOVE[j]) / 2.0
            if u < 1.0:
                ax.plot([xi, xj], [6.4, 6.4], color=QLIB, lw=2.0,
                        alpha=(1 - u) * 0.5, zorder=3)
            y = 6.4 + (4.3 - 6.4) * suave(u)
            ponto(ax, esc(meio), y, FRACO, tam=46, alfa=0.55 * a_par)

    fim_pares = 1.8 + len(PARES_NOVE) * por_par + 0.7
    trilho(ax, esc, 4.3, [-0.002, -0.001, 0.0, 0.001, 0.002, 0.003],
           alfa=surge(t, fim_pares - 0.6), fs=17)

    # a equacao se monta a partir do desenho
    u = trecho(t, fim_pares, fim_pares + 1.3)
    peca(ax, r"$\hat{\Delta}$", (4.9, 2.6), (esc(NOVE[4]), 6.4), u, fs=40,
         cor=MINHA)
    peca(ax, r"$=\;\mathrm{mediana}$", (6.9, 2.6), (8.0, 4.3), u, fs=32)
    peca(ax, r"$_{\,i \leq j}$", (8.5, 2.35), (8.0, 4.3), u, fs=26, cor=FRACO)
    peca(ax, r"$\dfrac{d_i + d_j}{2}$", (9.9, 2.6), (8.0, 4.3), u, fs=34)

    # a mediana das medias de Walsh
    a_med = surge(t, fim_pares + 1.6, 0.6)
    if a_med > 0.01:
        hl = float(np.median(WALSH_NOVE))
        ponto(ax, esc(hl), 4.3, MINHA, tam=360, alfa=a_med)
        ax.plot([esc(hl), esc(hl)], [4.3, 5.2], color=MINHA, lw=2.6,
                alpha=a_med)
        texto(ax, esc(hl), 5.5, "a do meio", fs=22, cor=MINHA, alfa=a_med,
              fundo=FUNDO)

    legenda(ax, "a media de cada par de conjuntos, inclusive de um conjunto "
                "com ele mesmo", surge(t, 2.0) * some(t, fim_pares, 0.6))
    legenda(ax, "o estimador de Hodges-Lehmann: a mediana de todas essas "
                "medias", surge(t, fim_pares + 1.8), cor=TINTA)


# ---------------------------------------------------------------------------
# cena 5 - o intervalo de confianca
# ---------------------------------------------------------------------------

def cena_ic(ax, t):
    cena_nova(ax)
    titulo(ax, "Ate onde essa diferenca pode estar", surge(t, 0.2))

    esc = Escala(-0.0022, 0.0030, x0=2.0, x1=14.0)
    trilho(ax, esc, 5.6, [-0.002, -0.001, 0.0, 0.001, 0.002, 0.003],
           alfa=surge(t, 0.3), fs=17)

    n = len(NOVE)
    w = WALSH_NOVE
    k = int(np.floor(n * (n + 1) / 4
                     - 1.959964 * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
                     - 0.5))
    k = max(k, 0)
    baixo, alto = float(w[k]), float(w[len(w) - 1 - k])
    hl = float(np.median(w))

    for idx, v in enumerate(w):
        a = surge(t, 0.5, 0.5)
        dentro = k <= idx <= len(w) - 1 - k
        ponto(ax, esc(v), 5.6, FRACO if dentro else GRADE, tam=46,
              alfa=a * (0.6 if dentro else 0.4))

    # as duas tesouras cortam as pontas
    a_corte = surge(t, 1.4, 0.6)
    if a_corte > 0.01:
        u = suave(trecho(t, 1.6, 2.8))
        for lado, v in ((-1, baixo), (1, alto)):
            x0 = esc(w[0] if lado < 0 else w[-1])
            x = x0 + (esc(v) - x0) * u
            ax.plot([x, x], [5.1, 6.1], color=VERMELHO, lw=2.4, alpha=a_corte)
        texto(ax, LARG / 2, 6.6, "corta as pontas no valor critico de Wilcoxon",
              fs=22, cor=VERMELHO, alfa=a_corte * some(t, 5.4, 0.6))

    u = trecho(t, 3.0, 4.2)
    peca(ax, r"$k \;=\; \dfrac{n(n+1)}{4}\;-\;"
             r"1{,}96\sqrt{\dfrac{n(n+1)(2n+1)}{24}}\;-\;\dfrac{1}{2}$",
         (LARG / 2, 3.6), (LARG / 2, 5.6), u, fs=30)
    texto(ax, LARG / 2, 2.85, "a k-esima media de Walsh de cada ponta",
          fs=21, cor=FRACO, alfa=surge(t, 4.2))

    # a barra do grafico
    a_bar = surge(t, 5.6, 0.7)
    if a_bar > 0.01:
        u = suave(trecho(t, 5.8, 6.8))
        meio = (baixo + alto) / 2
        barra_ic(ax, esc, 1.9, meio + (baixo - meio) * u,
                 meio + (alto - meio) * u, hl, MINHA, alfa=a_bar)
        texto(ax, LARG / 2, 1.15,
              "e exatamente a barra que aparece no grafico", fs=23,
              cor=TINTA, alfa=surge(t, 6.9))

    legenda(ax, f"no painel, com os {HDY8_N} conjuntos de HDy8: "
                f"{HDY8_HL:+.5f} com intervalo "
                f"[{HDY8_IC[0]:+.5f}; {HDY8_IC[1]:+.5f}]".replace(".", ","),
            surge(t, 8.4), y=0.4, fs=22)


# ---------------------------------------------------------------------------
# cena 6 - TOST
# ---------------------------------------------------------------------------

def cena_tost(ax, t):
    cena_nova(ax)
    titulo(ax, "Provar que a diferenca e pequena", surge(t, 0.2), y=8.5)

    legenda(ax, "nenhum teste prova que dois erros sao iguais; da para provar "
                "que a diferenca nao cabe fora de uma faixa",
            surge(t, 0.6) * some(t, 4.6, 0.6))

    esc = Escala(-0.016, 0.016, x0=2.0, x1=14.0)
    a_eixo = surge(t, 1.4, 0.6)
    trilho(ax, esc, 4.6, [-0.01, 0.0, 0.01], alfa=a_eixo, fmt="{:+.2f}")
    faixa(ax, esc, 2.6, 7.0, MARGEM, alfa=surge(t, 1.8, 0.6))
    texto(ax, esc(0.0), 7.3, "faixa de equivalencia  $\\pm$0,01", fs=23,
          cor=VERDE, alfa=surge(t, 2.0, 0.6))

    # as duas hipoteses nas bordas
    a_h = surge(t, 2.6, 0.6)
    quebra = (trecho(t, 6.2, 7.0), trecho(t, 6.8, 7.6))
    for lado, (nome, eq, u) in zip(
            (-1, 1),
            ((r"$H_0^-$", r"$\Delta \leq -\delta$", quebra[0]),
             (r"$H_0^+$", r"$\Delta \geq +\delta$", quebra[1]))):
        viva = 1.0 - suave(u)
        if a_h * viva <= 0.01:
            continue
        x0 = esc(lado * MARGEM)
        larg = (esc.x1 - x0) if lado > 0 else (x0 - esc.x0)
        ax.add_patch(plt.Rectangle((x0 if lado > 0 else esc.x0, 2.6),
                                   larg, 4.4, facecolor=TINTA,
                                   alpha=0.10 * a_h * viva, lw=0, zorder=1))
        cx = (x0 + (esc.x1 if lado > 0 else esc.x0)) / 2
        texto(ax, cx, 6.2, nome, fs=32, cor=TINTA, alfa=a_h * viva)
        texto(ax, cx, 5.6, eq, fs=26, cor=FRACO, alfa=a_h * viva)

    # o intervalo do HDy8 cai dentro
    a_bar = surge(t, 4.6, 0.6) * some(t, 9.4, 0.6)
    if a_bar > 0.01:
        barra_ic(ax, esc, 3.6, HDY8_IC[0], HDY8_IC[1], HDY8_HL, QLIB,
                 alfa=a_bar)
        texto(ax, esc(HDY8_HL), 3.1, "HDy8", fs=21, cor=QLIB, alfa=a_bar)
        texto(ax, esc(HDY8_HL), 4.3, "nesta escala o intervalo inteiro "
                                     "cabe no ponto", fs=20, cor=FRACO,
              alfa=a_bar)

    # os dois testes de uma ponta so
    u = trecho(t, 5.0, 6.2)
    peca(ax, r"$p^- = \mathrm{Wilcoxon}(d + \delta)$", (4.6, 1.75),
         (esc(-MARGEM), 3.6), u, fs=25)
    peca(ax, r"$p^+ = \mathrm{Wilcoxon}(d - \delta)$", (11.4, 1.75),
         (esc(MARGEM), 3.6), u, fs=25)
    peca(ax, r"$p = \max(p^-,\, p^+)$", (LARG / 2, 1.05), (LARG / 2, 1.75),
         trecho(t, 7.6, 8.4), fs=28, cor=TINTA)

    if t > 7.8:
        texto(ax, LARG / 2, 7.8,
              "as duas hipoteses caem: a diferenca esta dentro da faixa",
              fs=24, cor=VERDE, alfa=surge(t, 7.8) * some(t, 9.4, 0.6))

    # o contraexemplo
    a_ms2 = surge(t, 9.8, 0.7)
    if a_ms2 > 0.01:
        ax.add_patch(plt.Rectangle((esc.x0, 2.6), esc(-MARGEM) - esc.x0, 4.4,
                                   facecolor=VERMELHO, alpha=0.10 * a_ms2,
                                   lw=0, zorder=1))
        texto(ax, (esc.x0 + esc(-MARGEM)) / 2, 6.2, r"$H_0^-$", fs=32,
              cor=VERMELHO, alfa=a_ms2)
        barra_ic(ax, esc, 3.6, MS2_IC[0], MS2_IC[1], MS2_HL, QUAPY,
                 alfa=a_ms2, vazado=True)
        texto(ax, esc(MS2_HL), 3.1, "MS2", fs=21, cor=QUAPY, alfa=a_ms2)
        texto(ax, LARG / 2, 7.8, "o MS2 escapa pela esquerda: nao da para "
                                 "dizer que empata", fs=24, cor=VERMELHO,
              alfa=surge(t, 10.6))

    legenda(ax, f"{DENTRO} dos {PARES} pares cabem na faixa",
            surge(t, 11.8), y=0.3, fs=24, cor=TINTA)


# ---------------------------------------------------------------------------
# cena 7 - Holm
# ---------------------------------------------------------------------------

def cena_holm(ax, t):
    cena_nova(ax)
    titulo(ax, "Trinta e nove testes de uma vez", surge(t, 0.2))
    legenda(ax, "testar 39 vezes a 5% produz quase dois falsos positivos "
                "so por sorte", surge(t, 0.6) * some(t, 4.2, 0.6))

    m = PARES
    alfa = 0.05
    rng = np.random.default_rng(3)
    ps = np.sort(np.concatenate([rng.uniform(1e-4, 0.01, 31),
                                 rng.uniform(0.01, 0.6, m - 31)]))

    # altura = p-valor, em escala logaritmica: so nela o criterio de Holm,
    # que divide por um numero diferente a cada posto, vira uma reta
    def altura(p):
        u = (np.log10(max(float(p), 1e-4)) + 4.0) / 4.0
        return 2.9 + u * 4.2

    xs = np.linspace(2.4, 13.6, m)
    for v, rot in ((1e-4, "0,0001"), (0.01, "0,01"), (1.0, "1")):
        a = surge(t, 0.8, 0.6)
        ax.plot([2.1, 13.9], [altura(v)] * 2, color="#f0f0f0", lw=1.4,
                alpha=a, zorder=0)
        texto(ax, 1.95, altura(v), rot, fs=17, cor=GRADE, alfa=a, ha="right")
    texto(ax, 1.95, 7.6, "p-valor", fs=19, cor=FRACO, alfa=surge(t, 0.8),
          ha="left")

    # entram embaralhados e se ordenam
    u_ord = suave(trecho(t, 2.6, 4.0))
    embaralhado = rng.permutation(m)
    criterio = [alfa / (m - k) for k in range(m)]
    for k in range(m):
        a = surge(t, 1.0 + k * 0.03, 0.35)
        if a <= 0.01:
            continue
        xi = xs[embaralhado[k]]
        x = xi + (xs[k] - xi) * u_ord
        passa = ps[k] <= criterio[k]
        marcado = suave(trecho(t, 6.4 + k * 0.02, 6.9 + k * 0.02))
        cor = TINTA if passa else GRADE
        ax.add_patch(plt.Rectangle((x - 0.13, altura(ps[k]) - 0.13), 0.26,
                                   0.26, facecolor=cor if marcado else FRACO,
                                   alpha=a * (0.85 if marcado else 0.5), lw=0,
                                   zorder=4))
    texto(ax, LARG / 2, 8.0, "os 39 p-valores, do menor para o maior",
          fs=22, cor=FRACO, alfa=surge(t, 4.1) * some(t, 9.6, 0.6))

    # a reta do criterio
    a_esc = surge(t, 4.6, 0.6)
    if a_esc > 0.01:
        u = suave(trecho(t, 4.8, 6.2))
        alturas = [altura(c) for c in criterio]
        corte = int(np.ceil(m * u))
        if corte > 1:
            ax.plot(xs[:corte], alturas[:corte], color=VERMELHO, lw=3.0,
                    alpha=a_esc, solid_capstyle="round", zorder=3)
        texto(ax, 13.9, altura(criterio[-1]) + 0.35, "criterio de Holm",
              fs=21, cor=VERMELHO, alfa=a_esc * u, ha="right")

    u = trecho(t, 6.6, 7.8)
    peca(ax, r"$p_{(k)} \;\leq\; \dfrac{\alpha}{m - k + 1}$",
         (LARG / 2, 1.85), (LARG / 2, 4.0), u, fs=34)
    texto(ax, LARG / 2, 1.0,
          "o primeiro teste enfrenta 39 vezes mais exigencia; "
          "o ultimo, nenhuma", fs=21, cor=FRACO, alfa=surge(t, 7.9))

    legenda(ax, "assim o conjunto das 39 conclusoes carrega 5% de risco, "
                "nao cada uma", surge(t, 9.8), y=0.45, cor=TINTA)


# ---------------------------------------------------------------------------
# cena 8 - o custo em log2
# ---------------------------------------------------------------------------

def cena_custo(ax, t):
    cena_nova(ax)
    titulo(ax, "No custo, a pergunta e outra", surge(t, 0.2))
    legenda(ax, "nao a diferenca, mas quantas vezes: razao entre o custo da "
                "mlquantify e o da outra", surge(t, 0.6) * some(t, 4.4, 0.6))

    u_log = suave(trecho(t, 4.8, 6.4))

    def posicao(r):
        """Linear no comeco, log2 depois: a escala se transforma na tela."""
        lin = 2.0 + (r / 4.0) * 12.0
        lg = 8.0 + np.log2(r) * 2.6
        return lin + (lg - lin) * u_log

    y = 5.9
    a = surge(t, 1.6, 0.6)
    ax.plot([2.0, 14.0], [y, y], color=TRILHO, lw=2.0, alpha=a,
            solid_capstyle="round", zorder=2)
    for r, rot in ((0.25, "1/4"), (0.5, "1/2"), (1.0, "1"), (2.0, "2"),
                   (4.0, "4")):
        x = posicao(r)
        ax.plot([x, x], [y - 0.1, y + 0.1], color=TRILHO, lw=2.0, alpha=a)
        texto(ax, x, y - 0.45, rot, fs=22, cor=FRACO, alfa=a)

    # metade e dobro, com a mesma importancia
    a_set = surge(t, 2.4, 0.6)
    if a_set > 0.01:
        for r, cor, nome in ((0.5, VERDE, "metade do custo"),
                             (2.0, VERMELHO, "o dobro")):
            x, x1 = posicao(r), posicao(1.0)
            ax.annotate("", xy=(x, y + 0.75), xytext=(x1, y + 0.75),
                        arrowprops=dict(arrowstyle="-|>", color=cor, lw=2.8,
                                        alpha=a_set))
            texto(ax, (x + x1) / 2, y + 1.2, nome, fs=21, cor=cor, alfa=a_set)
            ponto(ax, x, y, cor, tam=220, alfa=a_set)

    texto(ax, LARG / 2, 3.5, "na escala direta, dobrar parece maior do que "
                             "cair pela metade", fs=24, cor=TINTA,
          alfa=surge(t, 3.4) * some(t, 4.8, 0.5))
    texto(ax, LARG / 2, 3.5, "em log2, as duas coisas pesam igual",
          fs=24, cor=TINTA, alfa=surge(t, 6.5))

    u = trecho(t, 6.9, 8.1)
    peca(ax, r"$\hat{\Delta} = \mathrm{HL}\!\left(\log_2 "
             r"\dfrac{c_{\,\mathrm{mlquantify}}}{c_{\,\mathrm{outra}}}\right)$",
         (5.0, 2.1), (6.0, 5.2), u, fs=30)
    peca(ax, r"$\hat{r} = 2^{\hat{\Delta}}$", (12.0, 2.1), (10.0, 5.2), u,
         fs=32, cor=MINHA)
    texto(ax, LARG / 2, 1.3, "o mesmo estimador e o mesmo intervalo da "
                             "parte do erro, so que no logaritmo", fs=22,
          cor=FRACO, alfa=surge(t, 8.3))

    legenda(ax, f"tempo: vantagem firme em {TEMPO_GANHA} dos {PARES} pares; "
                f"memoria, em {MEM_GANHA}", surge(t, 10.4), y=0.35, fs=24,
            cor=TINTA)


# ---------------------------------------------------------------------------
# cena 9 - como ler o grafico
# ---------------------------------------------------------------------------

LINHAS_FINAIS = [
    ("MS2", QUAPY, MS2_HL, MS2_IC[0], MS2_IC[1], False),
    ("HDy8", QLIB, HDY8_HL, HDY8_IC[0], HDY8_IC[1], True),
    ("GHDy", QUNFOLD, 0.00185, 0.00118, 0.00281, True),
    ("CC", QUAPY, 0.00002, -0.00004, 0.00009, True),
]


def cena_ler(ax, t):
    cena_nova(ax)
    titulo(ax, "Como ler o grafico", surge(t, 0.2))

    esc = Escala(-0.013, 0.013, x0=3.4, x1=14.0)
    ys = [6.3, 5.3, 4.3, 3.3]

    faixa(ax, esc, 2.7, 6.9, MARGEM, alfa=surge(t, 0.5))
    trilho(ax, esc, 2.3, [-0.01, 0.0, 0.01], alfa=surge(t, 0.4),
           fmt="{:+.2f}")

    for i, ((nome, cor, hl, b, a_, dentro), y) in enumerate(
            zip(LINHAS_FINAIS, ys)):
        al = surge(t, 0.8 + i * 0.35, 0.5)
        if al <= 0.01:
            continue
        if not dentro:
            ax.add_patch(plt.Rectangle((esc.x0 - 1.9, y - 0.45),
                                       esc.x1 - esc.x0 + 1.9, 0.9,
                                       facecolor=ROSA, alpha=al, lw=0,
                                       zorder=0))
        texto(ax, esc.x0 - 0.4, y, nome, fs=24, cor=TINTA, alfa=al,
              ha="right")
        barra_ic(ax, esc, y, b, a_, hl, cor, alfa=al, vazado=not dentro,
                 lw=5.0)

    # as chamadas para cada elemento
    def chamada(x_alvo, y_alvo, x_txt, y_txt, s, u, cor=TINTA):
        if u <= 0:
            return
        a = suave(min(u, 1.0))
        ax.annotate("", xy=(x_alvo, y_alvo), xytext=(x_txt, y_txt),
                    arrowprops=dict(arrowstyle="-", color=FRACO, lw=1.6,
                                    alpha=a * 0.8,
                                    connectionstyle="arc3,rad=0.15"))
        texto(ax, x_txt, y_txt, s, fs=21, cor=cor, alfa=a, fundo=FUNDO)

    chamada(esc(HDY8_HL), 5.3, 11.4, 7.85,
            "ponto: a diferenca tipica", trecho(t, 2.6, 3.4))
    chamada(esc(0.00281) + 0.15, 4.3, 12.4, 3.7,
            "barra: o intervalo de 95%", trecho(t, 3.6, 4.4))
    chamada(esc(MARGEM), 3.0, 11.6, 1.3,
            "faixa: sem efeito pratico", trecho(t, 4.6, 5.4), cor=VERDE)
    chamada(esc(MS2_HL), 6.3, 4.2, 7.85,
            "vazado: a barra escapa", trecho(t, 5.6, 6.4), cor=VERMELHO)

    legenda(ax, "dentro da faixa, as duas bibliotecas empatam na pratica; "
                "fora, nao da para dizer isso", surge(t, 7.0), cor=TINTA)


# ---------------------------------------------------------------------------
# cena 10 - fecho
# ---------------------------------------------------------------------------

def cena_fecho(ax, t):
    cena_nova(ax)
    titulo(ax, "Em uma frase", surge(t, 0.2), y=7.2)

    texto(ax, LARG / 2, 5.5,
          f"No erro: empate em {DENTRO} dos {PARES} pares.", fs=34,
          cor=TINTA, alfa=surge(t, 0.9))
    texto(ax, LARG / 2, 4.6,
          f"No custo: vantagem firme em {TEMPO_GANHA} de {PARES} no tempo",
          fs=34, cor=TINTA, alfa=surge(t, 1.8))
    texto(ax, LARG / 2, 3.9, f"e em {MEM_GANHA} de {PARES} na memoria.",
          fs=34, cor=TINTA, alfa=surge(t, 2.1))

    texto(ax, LARG / 2, 2.4,
          "Wilcoxon pareado  $\\cdot$  Hodges-Lehmann  $\\cdot$  "
          "TOST  $\\cdot$  Holm  $\\cdot$  razao em log2",
          fs=24, cor=FRACO, alfa=surge(t, 3.0))

    u = suave(trecho(t, 3.4, 4.4))
    ax.plot([LARG / 2 - 5.0 * u, LARG / 2 + 5.0 * u], [1.9, 1.9],
            color=MINHA, lw=3.0, alpha=0.85, solid_capstyle="round")


# ---------------------------------------------------------------------------
# montagem
# ---------------------------------------------------------------------------

CENAS = [
    (7.0, cena_abertura, None),
    (15.5, cena_diferenca, "1 / 6  a diferenca pareada"),
    (11.5, cena_media, "2 / 6  por que nao a media"),
    (0.0, cena_hl, "3 / 6  Hodges-Lehmann"),
    (10.5, cena_ic, "4 / 6  o intervalo de Wilcoxon"),
    (13.5, cena_tost, "5 / 6  TOST"),
    (11.5, cena_holm, "6 / 6  Holm"),
    (12.0, cena_custo, "o custo, em log2"),
    (9.0, cena_ler, "o grafico"),
    (6.5, cena_fecho, None),
]

# a cena das medias de Walsh dura o que a chuva de pares precisar
CENAS[3] = (1.8 + len(PARES_NOVE) * 0.085 + 4.2, cena_hl, CENAS[3][2])

TOTAL = sum(d for d, _, _ in CENAS)


def moldura(ax, segundo: float, rotulo) -> None:
    """Fio de progresso e nome da etapa: a cena nunca fica sem borda."""
    if rotulo:
        texto(ax, 0.7, 8.62, rotulo, fs=19, cor=GRADE, ha="left")
    ax.plot([0.7, 15.3], [0.17, 0.17], color="#ededed", lw=3.0,
            solid_capstyle="round", zorder=1)
    u = min(max(segundo / TOTAL, 0.0), 1.0)
    if u > 0:
        ax.plot([0.7, 0.7 + 14.6 * u], [0.17, 0.17], color=MINHA, lw=3.0,
                alpha=0.55, solid_capstyle="round", zorder=2)


def desenhar(ax, segundo: float) -> None:
    acumulado = 0.0
    for dur, func, rotulo in CENAS:
        if segundo < acumulado + dur or dur <= 0:
            func(ax, segundo - acumulado)
            moldura(ax, segundo, rotulo)
            return
        acumulado += dur
    CENAS[-1][1](ax, CENAS[-1][0])
    moldura(ax, segundo, CENAS[-1][2])


def encoder() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        achado = shutil.which("ffmpeg")
        if achado:
            return achado
    raise SystemExit("nao achei ffmpeg; instale o pacote imageio-ffmpeg")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--saida", default="video/testes_estatisticos.mp4")
    p.add_argument("--fps", type=int, default=FPS)
    p.add_argument("--quadro", type=float, default=None, metavar="SEGUNDO",
                   help="grava um png daquele instante, para conferir")
    args = p.parse_args()

    global plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F811

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(FUNDO)
    ax = fig.add_axes([0, 0, 1, 1])

    total = TOTAL

    if args.quadro is not None:
        desenhar(ax, args.quadro)
        destino = Path(args.saida).with_suffix(".png")
        destino.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destino, facecolor=FUNDO)
        print(f"quadro em {destino}")
        return 0

    saida = Path(args.saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    larg, altura = fig.canvas.get_width_height()

    cmd = [encoder(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgba",
           "-s", f"{larg}x{altura}", "-r", str(args.fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "medium", str(saida)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    n = int(round(total * args.fps))
    print(f"{total:.1f} s, {n} quadros, {larg}x{altura}")
    for i in range(n):
        desenhar(ax, i / args.fps)
        fig.canvas.draw()
        proc.stdin.write(bytes(fig.canvas.buffer_rgba()))
        if i % (args.fps * 5) == 0:
            print(f"  {i / args.fps:5.1f} s", flush=True)
    proc.stdin.close()
    proc.wait()
    print(f"video em {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
