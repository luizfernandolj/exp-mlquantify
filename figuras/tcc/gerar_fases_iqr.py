"""Tabela de fases do Q2 com intervalo interquartil.

Refaz, sem pandas, a razao de custo de cada par metodo x biblioteca como o
painel_bibliotecas.testar_custo (mediana por conjunto sobre os tamanhos de
amostra, log2 da razao mlquantify / outra, estimador de Hodges-Lehmann) no
recorte de 110 conjuntos, para cada fase. Por biblioteca, devolve a mediana
dessas razoes entre os metodos, que e o valor que a tabela ja mostrava, e o
1o e o 3o quartis, que mostram o quanto os metodos variam em torno dela.

No grafico, o ponto cheio marca o par sem diferenca estatistica naquela
fase, com o intervalo da razao contendo 1, e o vazado marca o par que
difere. Nao ha margem de equivalencia declarada para custo, entao a decisao
e a do teste de diferenca e nao a do TOST.
"""
import csv
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm

PASTA = Path(__file__).resolve().parent
RAIZ = Path("C:/Users/luiz_/OneDrive/Codigos/exp-mlquantify")
sys.path.insert(0, str(RAIZ))
import recorte_analise  # noqa: E402  precisa do RAIZ no sys.path
SUMMARY = Path("C:/Users/luiz_/exp-mlquantify-run/results/final-pareado/summary.csv")
REFERENCIA = PASTA / "q2-fases.csv"

MEDIDAS = [("tempo_treino_s", "Tempo de treinamento"),
           ("tempo_estimacao_total_s", "Tempo de estimação"),
           ("memoria_treino_mb", "Memória no treinamento"),
           ("memoria_estimacao_mb", "Memória na estimação")]
LIBS = ["quantificationlib", "quapy", "qunfold"]


def ler():
    fora = recorte_analise.excluidos()
    linhas = []
    with io.open(SUMMARY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and r["dataset_id"] not in fora:
                linhas.append(r)
    existentes = {(r["biblioteca"], r["metodo"]) for r in linhas}
    for r in list(linhas):
        for alias in filter(None, (a.strip() for a in r.get("aliases", "")
                                   .replace(";", ",").replace("|", ",")
                                   .split(","))):
            if (r["biblioteca"], alias) not in existentes:
                linhas.append(dict(r, metodo=alias))
    return linhas


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v


def hodges_lehmann(d):
    """Estimador de Hodges-Lehmann e o intervalo de confianca de 95%."""
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


def razoes(linhas, coluna):
    """Razao de Hodges-Lehmann, intervalo e veredito por (metodo, biblioteca)."""
    # mlquantify: mediana das repeticoes por (conjunto, metodo, tamanho)
    minha = defaultdict(list)
    for r in linhas:
        if r["biblioteca"] == "mlquantify":
            minha[(r["dataset_id"], r["metodo"], r["tamanho_amostra"])].append(
                num(r[coluna]))
    minha = {k: float(np.nanmedian(v)) for k, v in minha.items()
             if np.isfinite(v).any()}
    # pares (minha, dela) por (metodo, biblioteca, conjunto)
    pares = defaultdict(list)
    for r in linhas:
        if r["biblioteca"] == "mlquantify":
            continue
        k = (r["dataset_id"], r["metodo"], r["tamanho_amostra"])
        if k not in minha:
            continue
        a, b = minha[k], num(r[coluna])
        if a > 0 and b > 0:
            pares[(r["metodo"], r["biblioteca"], r["dataset_id"])].append((a, b))
    por_par = defaultdict(list)
    for (met, lib, _), ab in pares.items():
        ab = np.array(ab)
        # uma observacao por conjunto: mediana de cada lado, depois a razao
        por_par[(met, lib)].append(np.log2(np.median(ab[:, 0])
                                           / np.median(ab[:, 1])))
    out = {}
    for par, d in por_par.items():
        d = np.asarray(d)
        d = d[np.isfinite(d)]
        if len(d) >= 6:
            hl, baixo, alto = hodges_lehmann(d)
            out[par] = dict(razao=2.0 ** hl, baixo=2.0 ** baixo,
                            alto=2.0 ** alto,
                            igual=bool(baixo < 0 < alto))
    return out


linhas = ler()
# os 39 pares da comparacao, os mesmos das figuras do Q2
with io.open(PASTA / "q2-tempo.csv", encoding="utf-8") as f:
    OFICIAIS = {(r["metodo"], r["biblioteca"]) for r in csv.DictReader(f)}
with io.open(REFERENCIA, encoding="utf-8") as f:
    ref = {r["medida"]: r for r in csv.DictReader(f)}

saida = []
for coluna, rotulo in MEDIDAS:
    rz = razoes(linhas, coluna)
    for lib in LIBS:
        v = np.array([t["razao"] for (m, l), t in rz.items()
                      if l == lib and (m, l) in OFICIAIS])
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iguais = sum(t["igual"] for (m, l), t in rz.items()
                     if l == lib and (m, l) in OFICIAIS)
        saida.append(dict(medida=rotulo, biblioteca=lib, n=len(v),
                          mediana=med, q1=q1, q3=q3, sem_diferenca=iguais))
        print(f"{rotulo:<24}{lib:<19}n={len(v):>3}  mediana {med:.3f}  "
              f"IQR [{q1:.3f}, {q3:.3f}]  sem diferenca {iguais}")

with io.open(PASTA / "q2-fases-iqr.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(saida[0]))
    w.writeheader()
    w.writerows(saida)
print("\nreferencia (q2-fases.csv):")
for k, r in ref.items():
    print(" ", k, {l: r.get(l) for l in LIBS})


# ------------------------------------------------------------------ boxplot
# Um grafico por biblioteca, com uma caixa por medida e fase e os metodos
# por cima, todos na mesma escala log2, para entrar como subfiguras no LaTeX.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CORES = {"quapy": "#2b6cb0", "quantificationlib": "#b7791f", "qunfold": "#1d5c46"}
MARCAS = {"quapy": "o", "quantificationlib": "s", "qunfold": "^"}
TINTA, FRACO, GRADE = "#1a1a1a", "#6b6b6b", "#d9d9d9"
VERDE_CLARO = "#e8f2ec"
ROTULOS = ["Tempo\ntreino", "Tempo\nestimação", "Memória\ntreino", "Memória\nestimação"]

por_lib = {lib: [] for lib in LIBS}
for coluna, _ in MEDIDAS:
    rz = razoes(linhas, coluna)
    for lib in LIBS:
        da_lib = [t for (m, l), t in rz.items()
                  if l == lib and (m, l) in OFICIAIS]
        por_lib[lib].append((np.log2([t["razao"] for t in da_lib]),
                             np.array([t["igual"] for t in da_lib])))

tudo = np.concatenate([np.concatenate([x for x, _ in v])
                       for v in por_lib.values()])
y_min, y_max = np.floor(tudo.min()) - 0.4, np.ceil(tudo.max()) + 0.4
yt = np.arange(np.ceil(y_min), np.floor(y_max) + 1)


def fracao(e):
    e = int(round(e))
    return str(2 ** e) if e >= 0 else f"1/{2 ** -e}"


rng = np.random.default_rng(0)
for lib in LIBS:
    fig, ax = plt.subplots(figsize=(3.6, 3.25), dpi=200)
    ax.axhspan(y_min, 0, color=VERDE_CLARO, zorder=0)
    ax.axhline(0, color=FRACO, lw=0.9, zorder=1)
    cor = CORES[lib]
    caixas = ax.boxplot([x for x, _ in por_lib[lib]],
                        positions=range(4), widths=0.55,
                        patch_artist=True, showfliers=False, zorder=2,
                        medianprops=dict(color=TINTA, lw=1.4),
                        whiskerprops=dict(color=cor, lw=1.0),
                        capprops=dict(color=cor, lw=1.0),
                        boxprops=dict(edgecolor=cor, lw=1.0))
    for c in caixas["boxes"]:
        c.set_facecolor(matplotlib.colors.to_rgba(cor, 0.22))
    for i, (v, igual) in enumerate(por_lib[lib]):
        x = i + rng.uniform(-0.16, 0.16, len(v))
        # cheio: o intervalo da razao contem 1, entao o par nao difere.
        # Vai por ultimo e por cima, porque e a excecao que se procura
        ax.scatter(x[~igual], v[~igual], s=11, marker=MARCAS[lib],
                   facecolor="white", edgecolor=cor, linewidth=0.7, zorder=3)
        ax.scatter(x[igual], v[igual], s=13, marker=MARCAS[lib], color=cor,
                   edgecolor=cor, linewidth=0.4, zorder=4)
    ax.set_xticks(range(4), ROTULOS)
    ax.set_yticks(yt, [fracao(e) for e in yt])
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", color=GRADE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA, labelsize=8, length=0)
    ax.set_ylabel("Razão mlquantify / referência", fontsize=9, color=TINTA,
                  labelpad=4)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(PASTA / f"q2-fases-{lib}.{ext}", bbox_inches="tight")
    plt.close(fig)
print("boxplots gravados")
