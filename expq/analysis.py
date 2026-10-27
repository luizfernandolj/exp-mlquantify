"""Respostas de Q1, Q2 e Q3 a partir dos resultados agregados.

A agregacao (``expq/aggregate.py``) produz uma linha por metodo, biblioteca e
tamanho de amostra. Este modulo vai um passo adiante: sintetiza cada
comparacao em uma tabela pequena, do tamanho que entra no texto, escreve as
figuras e monta o ``respostas.md`` com os numeros no lugar.

Cada tabela sai em tres formatos -- `.csv` para conferir, `.md` para ler e
`.tex` para colar no texto -- e a unidade de comparacao e sempre o **caso
pareado**: um conjunto de dados em que as duas bibliotecas comparadas
produziram estimativa. Media sobre casos pareados nao depende de quais
conjuntos cada biblioteca conseguiu rodar, que e o vies que atrapalharia a
leitura de Q1 e de Q2.

Roda com a execucao em andamento. Nesse caso o topo de cada saida diz quantas
combinacoes das planejadas entraram na conta, para que um numero provisorio
nao seja confundido com o final.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import REFERENCE, expand_aliases

#: Medidas de erro de Q1. (coluna, rotulo)
ERROS = [("mae", "AE"), ("mrae", "RAE"), ("mse", "SE"), ("nkld_medio", "NKLD")]

#: Medidas de custo de Q2. (coluna, rotulo, unidade)
CUSTOS = [
    ("tempo_treino_s", "tempo de treinamento", "s"),
    ("tempo_estimacao_total_s", "tempo de estimacao", "s"),
    ("memoria_treino_mb", "memoria no treinamento", "MB"),
    ("memoria_estimacao_mb", "memoria na estimacao", "MB"),
]

#: Limiares de equivalencia pratica para Q1. Duas implementacoes do mesmo
#: metodo sobre os mesmos dados deveriam dar o mesmo numero; a diferenca que
#: sobra vem de detalhe de implementacao (ordem de operacoes, tolerancia do
#: otimizador, desempate). 1e-3 e a casa em que uma prevalencia e reportada,
#: 1e-2 e a casa em que uma diferenca passaria a mudar uma decisao.
LIMIARES = [1e-3, 1e-2]

#: Cores por biblioteca. Os tres primeiros slots da paleta categorica, que sao
#: os que se separam em todos os pares, inclusive para quem nao distingue
#: vermelho e verde -- o grafico de dispersao compara todos contra todos.
CORES = {
    "quantificationlib": "#2a78d6",
    "quapy": "#eb6834",
    "qunfold": "#1baf7a",
    REFERENCE: "#6250d6",
}
TINTA = "#1f2328"
TINTA_FRACA = "#6b7280"
GRADE = "#e5e7eb"


# --------------------------------------------------------------------------
# Quanto da execucao entrou na conta
# --------------------------------------------------------------------------

def cobertura(cfg) -> dict:
    """Combinacoes medidas sobre as planejadas, para marcar o provisorio."""
    from .catalog import load_selected
    from .ledger import load_done
    from .registry import specs_for

    done = load_done(cfg)
    planejadas = 0
    for row in load_selected(cfg).to_dict("records"):
        for lib in cfg.libraries:
            for s in specs_for(lib, groups=cfg.groups, methods=cfg.methods,
                               skip=cfg.skip_methods):
                planejadas += 1
    feitas = sum(1 for k in done if k.split("|")[0])
    return {"feitas": feitas, "planejadas": planejadas,
            "fracao": feitas / planejadas if planejadas else np.nan,
            "completa": feitas >= planejadas}


# --------------------------------------------------------------------------
# Casos pareados
# --------------------------------------------------------------------------

def _pares(ok: pd.DataFrame, coluna: str, chave: list[str]) -> pd.DataFrame:
    """Um caso por linha, com o valor da proposta e o da outra biblioteca.

    Linhas sem valor em uma das duas saem fora: a comparacao e par a par, e
    um metodo que uma biblioteca nao implementa nao entra na media da outra.
    """
    if coluna not in ok.columns:
        return pd.DataFrame()
    pivot = ok.pivot_table(index=chave, columns="biblioteca", values=coluna,
                           aggfunc="mean")
    if REFERENCE not in pivot.columns:
        return pd.DataFrame()
    partes = []
    for lib in sorted(c for c in pivot.columns if c != REFERENCE):
        par = pivot[[REFERENCE, lib]].dropna().reset_index()
        par.columns = chave + ["proposta", "outra"]
        par["biblioteca"] = lib
        partes.append(par)
    if not partes:
        return pd.DataFrame()
    casos = pd.concat(partes, ignore_index=True)
    casos["diferenca"] = casos["proposta"] - casos["outra"]
    return casos


def casos_erro(summary: pd.DataFrame, coluna: str = "mae") -> pd.DataFrame:
    """Casos pareados de erro: um por conjunto, metodo e tamanho de amostra."""
    ok = summary[summary["status"] == "ok"]
    return _pares(ok, coluna, ["dataset_id", "metodo", "tamanho_amostra"])


def casos_custo(summary: pd.DataFrame, coluna: str) -> pd.DataFrame:
    """Casos pareados de custo: um por conjunto e metodo.

    O custo e medido uma vez por combinacao, e nao por tamanho de amostra,
    entao as quatro linhas de uma combinacao repetem a mesma medida. Medida
    marcada como suspeita (relogio e CPU divergindo, sinal de que a maquina
    esteve suspensa ou sob disputa) fica fora: o erro dela continua valendo
    para Q1, o tempo nao.
    """
    ok = summary[summary["status"] == "ok"]
    if "medida_suspeita" in ok.columns:
        suspeita = ok["medida_suspeita"].astype(str).str.lower() == "true"
        ok = ok[~suspeita]
    ok = ok.drop_duplicates(subset=["dataset_id", "biblioteca", "metodo"])
    return _pares(ok, coluna, ["dataset_id", "metodo"])


# --------------------------------------------------------------------------
# Q1 - as implementacoes concordam?
# --------------------------------------------------------------------------

def _resumo_erro(casos: pd.DataFrame) -> dict:
    """Estatisticas de um conjunto de casos pareados de erro."""
    d = casos["diferenca"]
    linha = {
        "n_casos": len(casos),
        "n_conjuntos": casos["dataset_id"].nunique(),
        "n_metodos": casos["metodo"].nunique(),
        "media_proposta": casos["proposta"].mean(),
        "media_outra": casos["outra"].mean(),
        "diferenca_media": d.mean(),
        "diferenca_abs_media": d.abs().mean(),
        "diferenca_abs_mediana": d.abs().median(),
        "diferenca_abs_p95": d.abs().quantile(0.95),
        "diferenca_abs_maxima": d.abs().max(),
    }
    for t in LIMIARES:
        linha[f"iguais_ate_{t:g}"] = float((d.abs() < t).mean())
    linha["proposta_melhor"] = float((d < 0).mean())
    linha["correlacao"] = casos["proposta"].corr(casos["outra"])
    linha["correlacao_posto"] = casos["proposta"].corr(casos["outra"],
                                                       method="spearman")
    return linha


def q1_equivalencia(summary: pd.DataFrame, coluna: str = "mae") -> pd.DataFrame:
    """Uma linha por biblioteca comparada, e uma por tamanho de amostra.

    Responde Q1 em um numero: a fracao dos casos pareados em que as duas
    implementacoes do mesmo metodo dao o mesmo erro, dentro de cada limiar.
    """
    casos = casos_erro(summary, coluna)
    if casos.empty:
        return pd.DataFrame()
    linhas = []
    for lib, sub in casos.groupby("biblioteca"):
        linhas.append({"biblioteca": lib, "tamanho_amostra": "todos",
                       **_resumo_erro(sub)})
        for m, por_tamanho in sub.groupby("tamanho_amostra"):
            linhas.append({"biblioteca": lib, "tamanho_amostra": str(int(m)),
                           **_resumo_erro(por_tamanho)})
    ordem = {"todos": 0}
    return (pd.DataFrame(linhas)
            .sort_values(["biblioteca", "tamanho_amostra"],
                         key=lambda c: c.map(lambda v: ordem.get(v, 1))
                         if c.name == "tamanho_amostra" else c)
            .reset_index(drop=True))


def q1_discordancias(summary: pd.DataFrame, coluna: str = "mae",
                     quantos: int = 15) -> pd.DataFrame:
    """Os metodos que mais discordam entre implementacoes.

    Onde Q1 nao se confirma, a lista de excecoes e o resultado: um metodo com
    diferenca grande indica formulacao diferente sob o mesmo nome, e nao ruido
    de implementacao.
    """
    casos = casos_erro(summary, coluna)
    if casos.empty:
        return pd.DataFrame()
    linhas = []
    for (metodo, lib), sub in casos.groupby(["metodo", "biblioteca"]):
        d = sub["diferenca"]
        linhas.append({
            "metodo": metodo, "biblioteca": lib, "n_casos": len(sub),
            "media_proposta": sub["proposta"].mean(),
            "media_outra": sub["outra"].mean(),
            "diferenca_media": d.mean(),
            "diferenca_abs_media": d.abs().mean(),
            "diferenca_abs_maxima": d.abs().max(),
            f"iguais_ate_{LIMIARES[1]:g}": float((d.abs() < LIMIARES[1]).mean()),
            "proposta_melhor": float((d < 0).mean()),
        })
    tab = pd.DataFrame(linhas).sort_values("diferenca_abs_media", ascending=False)
    return tab.head(quantos).reset_index(drop=True)


def q1_testes(tests: pd.DataFrame) -> pd.DataFrame:
    """Resumo dos testes de hipotese, por medida.

    O teste responde se ha diferenca; os postos medios dizem a favor de quem.
    Postos proximos de (k+1)/2 em todas as bibliotecas e o retrato da
    equivalencia.

    As linhas sao separadas por conjunto de bibliotecas comparadas, e nao so
    por medida: um metodo que existe em tres bibliotecas tem posto de 1 a 3, e
    um que existe em duas tem posto de 1 a 2. Misturar os dois na mesma media
    compara numeros de escalas diferentes.
    """
    if tests.empty:
        return pd.DataFrame()
    cols = [c for c in tests.columns if c.startswith("rank_")]
    linhas = []
    for (medida, rotulo, libs), sub in tests.groupby(
            ["medida", "rotulo", "bibliotecas"]):
        linha = {
            "medida": medida, "rotulo": rotulo, "bibliotecas": libs,
            "metodos_testados": len(sub),
            "com_diferenca": int(sub["diferenca_significativa"].astype(str)
                                 .str.lower().eq("true").sum()),
            "friedman": int((sub["teste"] == "friedman").sum()),
            "wilcoxon": int((sub["teste"] == "wilcoxon").sum()),
        }
        linha["fracao_com_diferenca"] = round(
            linha["com_diferenca"] / linha["metodos_testados"], 4)
        for c in cols:
            linha[c] = sub[c].mean()
        linhas.append(linha)
    return (pd.DataFrame(linhas)
            .sort_values(["medida", "metodos_testados"], ascending=[True, False])
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# Q2 - quanto custa
# --------------------------------------------------------------------------

def q2_resumo(summary: pd.DataFrame) -> pd.DataFrame:
    """Razao de custo entre a proposta e cada biblioteca, por medida.

    A razao e calculada caso a caso e depois resumida pela mediana: medias de
    tempo sao dominadas pelo maior conjunto, e a pergunta aqui e como as duas
    se comparam no caso tipico. Razao 1 e empate, abaixo de 1 a proposta e
    mais barata.
    """
    linhas = []
    for coluna, rotulo, unidade in CUSTOS:
        casos = casos_custo(summary, coluna)
        if casos.empty:
            continue
        for lib, sub in casos.groupby("biblioteca"):
            razao = sub["proposta"] / sub["outra"].replace(0, np.nan)
            razao = razao.replace([np.inf, -np.inf], np.nan).dropna()
            if razao.empty:
                continue
            linhas.append({
                "medida": coluna, "rotulo": rotulo, "unidade": unidade,
                "biblioteca": lib, "n_casos": len(sub),
                "mediana_proposta": sub["proposta"].median(),
                "mediana_outra": sub["outra"].median(),
                "razao_mediana": razao.median(),
                "razao_p25": razao.quantile(0.25),
                "razao_p75": razao.quantile(0.75),
                "proposta_mais_barata": float((sub["proposta"]
                                               < sub["outra"]).mean()),
            })
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# Q3 - esforco de uso
# --------------------------------------------------------------------------

def q3_resumo(q3: pd.DataFrame) -> pd.DataFrame:
    """Reordena a tabela de Q3 nas colunas que entram no texto."""
    if q3.empty:
        return q3
    cols = ["biblioteca", "metodos_distintos", "combinacoes",
            "taxa_conclusao", "taxa_sem_configuracao_manual",
            "configuracoes_manuais_total", "configuracoes_manuais_media",
            "linhas_de_codigo", "configuracoes_manuais_script"]
    return q3[[c for c in cols if c in q3.columns]].copy()


# --------------------------------------------------------------------------
# Escrita das tabelas
# --------------------------------------------------------------------------

def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    if isinstance(v, (float, np.floating)):
        a = abs(v)
        if a != 0 and (a < 1e-3 or a >= 1e5):
            return f"{v:.2e}"
        return f"{v:.4f}".rstrip("0").rstrip(".") if a < 1 else f"{v:.3f}"
    return str(v)


def _markdown(df: pd.DataFrame) -> str:
    cab = list(df.columns)
    linhas = ["| " + " | ".join(cab) + " |",
              "|" + "|".join("---" for _ in cab) + "|"]
    for _, r in df.iterrows():
        linhas.append("| " + " | ".join(_fmt(r[c]) for c in cab) + " |")
    return "\n".join(linhas) + "\n"


def _latex(df: pd.DataFrame, legenda: str, rotulo: str) -> str:
    """Tabela em LaTeX com booktabs, pronta para colar no texto."""
    def esc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%")

    cab = " & ".join(rf"\textbf{{{esc(c)}}}" for c in df.columns)
    corpo = "\n".join("    " + " & ".join(esc(_fmt(r[c])) for c in df.columns)
                      + r" \\" for _, r in df.iterrows())
    return (
        "\\begin{table}[htb]\n"
        "  \\centering\n"
        f"  \\caption{{{esc(legenda)}}}\n"
        f"  \\label{{tab:{rotulo}}}\n"
        f"  \\begin{{tabular}}{{{'l' + 'r' * (len(df.columns) - 1)}}}\n"
        "    \\toprule\n"
        f"    {cab} \\\\\n"
        "    \\midrule\n"
        f"{corpo}\n"
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def escrever(df: pd.DataFrame, pasta: Path, nome: str, legenda: str) -> None:
    """Grava a mesma tabela em csv, md e tex."""
    if df.empty:
        print(f"  {nome}: vazia, nao gravada")
        return
    df.to_csv(pasta / f"{nome}.csv", index=False, encoding="utf-8")
    (pasta / f"{nome}.md").write_text(f"### {legenda}\n\n" + _markdown(df),
                                      encoding="utf-8")
    (pasta / f"{nome}.tex").write_text(_latex(df, legenda, nome),
                                       encoding="utf-8")
    print(f"  {nome:<22} {len(df):>4} linhas  (csv, md, tex)")


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------

def _estilo(ax) -> None:
    """Grade e eixos recessivos, do jeito que nao disputa com os dados."""
    ax.set_axisbelow(True)
    ax.grid(True, color=GRADE, linewidth=0.6, linestyle="-")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TINTA_FRACA, labelsize=9, length=3)


def figura_q1(summary: pd.DataFrame, destino: Path, coluna: str = "mae",
              rotulo: str = "AE") -> Path | None:
    """Dispersao do erro, uma faceta por biblioteca comparada.

    Uma faceta por biblioteca, e nao tres cores no mesmo painel: com milhares
    de casos as nuvens se cobrem, e a pergunta de Q1 e sobre a distancia de
    cada ponto ate a diagonal, nao sobre a mistura.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    casos = casos_erro(summary, coluna)
    if casos.empty:
        return None
    libs = sorted(casos["biblioteca"].unique())
    fig, axes = plt.subplots(1, len(libs), figsize=(3.4 * len(libs), 3.7),
                             squeeze=False)
    alto = max(casos["proposta"].max(), casos["outra"].max()) * 1.05
    for ax, lib in zip(axes[0], libs):
        sub = casos[casos["biblioteca"] == lib]
        ax.plot([0, alto], [0, alto], color=TINTA_FRACA, linewidth=1.0,
                zorder=1)
        ax.scatter(sub["proposta"], sub["outra"], s=7, alpha=0.25,
                   color=CORES.get(lib, CORES[REFERENCE]), linewidths=0,
                   zorder=2)
        iguais = float((sub["diferenca"].abs() < LIMIARES[1]).mean())
        ax.set_title(f"{lib}\n{len(sub)} casos, {iguais:.0%} iguais ate "
                     f"{LIMIARES[1]:g}", fontsize=10, color=TINTA)
        ax.set_xlabel(f"{rotulo} {REFERENCE}", fontsize=9, color=TINTA_FRACA)
        ax.set_xlim(0, alto)
        ax.set_ylim(0, alto)
        ax.set_aspect("equal")
        _estilo(ax)
    axes[0][0].set_ylabel(f"{rotulo} da outra biblioteca", fontsize=9,
                          color=TINTA_FRACA)
    fig.suptitle(f"Q1: erro do mesmo metodo nas duas implementacoes "
                 f"({rotulo}; a linha e a igualdade)", fontsize=11, color=TINTA)
    fig.tight_layout()
    caminho = destino / f"q1_dispersao_{coluna}.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def figura_q2(summary: pd.DataFrame, destino: Path) -> Path | None:
    """Razao de custo por medida, em escala log, com o empate marcado.

    Uma faceta por medida: segundos e megabytes no mesmo eixo seria o eixo
    duplo disfarcado. A caixa vai do primeiro ao terceiro quartil e o bigode
    ate o percentil 5 e 95.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, NullLocator

    dados = {}
    for coluna, rotulo, unidade in CUSTOS:
        casos = casos_custo(summary, coluna)
        if casos.empty:
            continue
        por_lib = {}
        for lib, sub in casos.groupby("biblioteca"):
            razao = (sub["proposta"] / sub["outra"].replace(0, np.nan))
            razao = razao.replace([np.inf, -np.inf], np.nan).dropna()
            if len(razao):
                por_lib[lib] = razao.to_numpy()
        if por_lib:
            dados[(coluna, rotulo)] = por_lib
    if not dados:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), squeeze=False)
    for ax, ((coluna, rotulo), por_lib) in zip(axes.ravel(), dados.items()):
        libs = sorted(por_lib)
        # bigode no percentil 5--95: com razao em escala log o caso extremo
        # estica o painel e esmaga justamente a faixa onde estao os dados
        caixas = ax.boxplot([por_lib[l] for l in libs], patch_artist=True,
                            widths=0.5, showfliers=False, whis=(5, 95),
                            medianprops=dict(color=TINTA, linewidth=1.4),
                            whiskerprops=dict(color=TINTA_FRACA, linewidth=1.0),
                            capprops=dict(color=TINTA_FRACA, linewidth=1.0),
                            tick_labels=[l.replace("quantificationlib", "qlib")
                                         for l in libs])
        for caixa, lib in zip(caixas["boxes"], libs):
            caixa.set(facecolor=CORES.get(lib, CORES[REFERENCE]), alpha=0.35,
                      edgecolor=CORES.get(lib, CORES[REFERENCE]), linewidth=1.2)
        ax.axhline(1.0, color=TINTA_FRACA, linewidth=1.0, zorder=1)
        ax.set_yscale("log")
        # limites simetricos em torno de 1: metade acima e metade abaixo, para
        # que "mais caro" e "mais barato" ocupem o mesmo espaco no olho
        juntos = np.concatenate([por_lib[l] for l in libs])
        extremo = np.nanpercentile(np.abs(np.log10(juntos[juntos > 0])), 97)
        extremo = float(np.clip(extremo, 0.35, 3.0))
        ax.set_ylim(10 ** -extremo, 10 ** extremo)
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_title(rotulo, fontsize=10, color=TINTA)
        ax.set_ylabel(f"razao {REFERENCE} / outra", fontsize=9,
                      color=TINTA_FRACA)
        _estilo(ax)
    for ax in axes.ravel()[len(dados):]:
        ax.set_visible(False)
    fig.suptitle("Q2: custo da proposta sobre o da outra biblioteca "
                 "(1 e empate; abaixo de 1 a proposta e mais barata)",
                 fontsize=11, color=TINTA)
    fig.tight_layout()
    caminho = destino / "q2_razao_custo.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


def figura_q3(q3: pd.DataFrame, destino: Path) -> Path | None:
    """Esforco de uso em barras, uma faceta por medida.

    Uma faceta por medida porque as escalas nao se comparam -- linhas de
    codigo, contagem de configuracoes e taxa entre 0 e 1 no mesmo eixo seria
    o eixo duplo disfarcado. A proposta aparece destacada; as outras ficam na
    mesma cor, porque aqui a pergunta e de magnitude, nao de identidade.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    if q3.empty:
        return None
    # a taxa de conclusao fica so na tabela: as quatro ficam entre 97% e 99%,
    # e quatro barras do mesmo tamanho nao dizem nada
    medidas = [("metodos_distintos", "metodos cobertos", "{:.0f}"),
               ("taxa_sem_configuracao_manual",
                "combinacoes sem configuracao manual", "{:.1%}"),
               ("configuracoes_manuais_script", "configuracoes manuais "
                "no script de uso", "{:.0f}"),
               ("linhas_de_codigo", "linhas de codigo no script de uso",
                "{:.0f}")]
    medidas = [m for m in medidas if m[0] in q3.columns]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), squeeze=False)
    for ax, (coluna, rotulo, forma) in zip(axes.ravel(), medidas):
        sub = q3.sort_values(coluna)
        nomes = [b.replace("quantificationlib", "qlib") for b in sub["biblioteca"]]
        # a transparencia entra na propria cor: o barh aceita uma cor por
        # barra, mas um unico alpha para todas
        cores = [to_rgba(CORES[REFERENCE], 1.0) if b == REFERENCE
                 else to_rgba(TINTA_FRACA, 0.45) for b in sub["biblioteca"]]
        barras = ax.barh(nomes, sub[coluna].fillna(0), height=0.6, color=cores)
        for barra, v in zip(barras, sub[coluna]):
            if pd.isna(v):
                continue
            ax.annotate(forma.format(v),
                        (barra.get_width(), barra.get_y() + barra.get_height() / 2),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TINTA)
        ax.set_title(rotulo, fontsize=10, color=TINTA)
        ax.set_xlim(0, float(sub[coluna].max()) * 1.18 if sub[coluna].notna().any() else 1)
        _estilo(ax)
        ax.grid(axis="y", visible=False)
    for ax in axes.ravel()[len(medidas):]:
        ax.set_visible(False)
    fig.suptitle(f"Q3: esforco de uso ({REFERENCE} destacada)", fontsize=11,
                 color=TINTA)
    fig.tight_layout()
    caminho = destino / "q3_esforco.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return caminho


# --------------------------------------------------------------------------
# O texto com as respostas
# --------------------------------------------------------------------------

def _linha_q1(tab: pd.DataFrame) -> str:
    todos = tab[tab["tamanho_amostra"] == "todos"]
    partes = []
    for _, r in todos.iterrows():
        partes.append(
            f"* **{r['biblioteca']}**: {int(r['n_casos'])} casos pareados "
            f"({int(r['n_metodos'])} metodos, {int(r['n_conjuntos'])} conjuntos). "
            f"Diferenca absoluta media de {r['diferenca_abs_media']:.4f} em AE, "
            f"mediana {r['diferenca_abs_mediana']:.4f}, maxima "
            f"{r['diferenca_abs_maxima']:.4f}. "
            f"Iguais ate {LIMIARES[0]:g} em {r[f'iguais_ate_{LIMIARES[0]:g}']:.1%} "
            f"dos casos e ate {LIMIARES[1]:g} em "
            f"{r[f'iguais_ate_{LIMIARES[1]:g}']:.1%}. "
            f"A proposta tem erro menor em {r['proposta_melhor']:.1%} dos casos; "
            f"correlacao de {r['correlacao']:.4f}.")
    return "\n".join(partes)


def _linha_q2(tab: pd.DataFrame) -> str:
    partes = []
    for rotulo, sub in tab.groupby("rotulo", sort=False):
        itens = []
        for _, r in sub.iterrows():
            itens.append(f"{r['biblioteca']} {r['razao_mediana']:.2f}x "
                         f"(IQR {r['razao_p25']:.2f}--{r['razao_p75']:.2f}, "
                         f"mais barata em {r['proposta_mais_barata']:.0%} dos "
                         f"{int(r['n_casos'])} casos)")
        partes.append(f"* **{rotulo}**: " + "; ".join(itens) + ".")
    return "\n".join(partes)


def respostas(cfg, tabelas: dict, cob: dict, figuras: list) -> str:
    """Monta o ``respostas.md`` com os numeros no lugar."""
    cabecalho = [f"# Respostas de Q1, Q2 e Q3 -- execucao `{cfg.run_id}`", ""]
    if not cob["completa"]:
        cabecalho += [
            f"> **PROVISORIO.** {cob['feitas']} das {cob['planejadas']} "
            f"combinacoes planejadas foram medidas ({cob['fracao']:.1%}). "
            "Os conjuntos que faltam sao os maiores do catalogo, entao as "
            "medias de custo devem subir quando eles entrarem. Rode de novo "
            "ao final para atualizar.", ""]
    else:
        cabecalho += [f"Execucao completa: {cob['feitas']} combinacoes.", ""]

    partes = cabecalho
    q1 = tabelas.get("q1_equivalencia", pd.DataFrame())
    if not q1.empty:
        partes += [
            "## Q1 -- as implementacoes dao o mesmo erro?", "",
            "Unidade de comparacao: um caso e um conjunto de dados, um metodo "
            "e um tamanho de amostra em que as duas bibliotecas produziram "
            "estimativa.", "",
            _linha_q1(q1), ""]
        disc = tabelas.get("q1_discordancias", pd.DataFrame())
        if not disc.empty:
            piores = disc.head(5)
            partes += [
                "Onde a equivalencia nao se sustenta (maior diferenca "
                "absoluta media):", "",
                "\n".join(f"* `{r['metodo']}` contra {r['biblioteca']}: "
                          f"{r['diferenca_abs_media']:.4f} de diferenca media, "
                          f"{r['media_proposta']:.4f} contra "
                          f"{r['media_outra']:.4f}"
                          for _, r in piores.iterrows()), ""]
        testes = tabelas.get("q1_testes", pd.DataFrame())
        if not testes.empty:
            ae = testes[testes["medida"] == "mae"]
            if len(ae):
                total = int(ae["metodos_testados"].sum())
                com = int(ae["com_diferenca"].sum())
                partes += [
                    f"Nos testes de hipotese sobre AE, {com} de {total} "
                    f"metodos acusam diferenca ({com / total:.1%}) a 5%. "
                    "Os postos medios so se comparam dentro do mesmo conjunto "
                    "de bibliotecas, porque o posto vai de 1 ate o numero "
                    "delas:", ""]
                for _, r in ae.iterrows():
                    postos = ", ".join(
                        f"{c.replace('rank_', '')} {r[c]:.2f}"
                        for c in testes.columns
                        if c.startswith("rank_") and pd.notna(r[c]))
                    partes.append(
                        f"* {r['bibliotecas'].replace('|', ' / ')} "
                        f"({int(r['metodos_testados'])} metodos, "
                        f"{int(r['com_diferenca'])} com diferenca): {postos}.")
                partes.append("")

    q2 = tabelas.get("q2_resumo", pd.DataFrame())
    if not q2.empty:
        partes += [
            "## Q2 -- quanto custa", "",
            "Razao entre o custo da proposta e o da outra biblioteca, caso a "
            "caso, resumida pela mediana. Abaixo de 1 a proposta e mais "
            "barata.", "",
            _linha_q2(q2), ""]

    q3 = tabelas.get("q3_resumo", pd.DataFrame())
    if not q3.empty:
        partes += ["## Q3 -- esforco de uso", "", _markdown(q3), ""]

    if figuras:
        partes += ["## Figuras", "",
                   "\n".join(f"* `figuras/{p.name}`" for p in figuras if p), ""]
    return "\n".join(partes)


# --------------------------------------------------------------------------
# Entrada
# --------------------------------------------------------------------------

def run(cfg, com_figuras: bool = True, erro: str = "mae") -> dict:
    """Le as tabelas agregadas, escreve a analise e devolve os caminhos."""
    summary_path = cfg.run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"{summary_path} nao existe; rode aggregate_results.py primeiro")

    summary = expand_aliases(pd.read_csv(summary_path, low_memory=False))
    testes_path = cfg.run_dir / "testes_estatisticos.csv"
    testes = pd.read_csv(testes_path) if testes_path.exists() else pd.DataFrame()
    q3_path = cfg.run_dir / "q3_uso.csv"
    q3 = pd.read_csv(q3_path) if q3_path.exists() else pd.DataFrame()

    cob = cobertura(cfg)
    destino = cfg.run_dir / "analise"
    destino.mkdir(parents=True, exist_ok=True)
    figuras_dir = destino / "figuras"
    figuras_dir.mkdir(exist_ok=True)

    print(f"Analisando {cfg.run_dir} ...")
    if not cob["completa"]:
        print(f"  PROVISORIO: {cob['feitas']} de {cob['planejadas']} "
              f"combinacoes ({cob['fracao']:.1%})")

    tabelas = {
        "q1_equivalencia": q1_equivalencia(summary, erro),
        "q1_discordancias": q1_discordancias(summary, erro),
        "q1_testes": q1_testes(testes),
        "q2_resumo": q2_resumo(summary),
        "q3_resumo": q3_resumo(q3),
    }
    legendas = {
        "q1_equivalencia": "Q1: concordancia entre implementacoes do mesmo "
                           "metodo, por biblioteca e tamanho de amostra",
        "q1_discordancias": "Q1: metodos com maior discordancia entre "
                            "implementacoes",
        "q1_testes": "Q1: resumo dos testes de hipotese por medida",
        "q2_resumo": "Q2: razao de custo entre a proposta e cada biblioteca",
        "q3_resumo": "Q3: esforco de uso por biblioteca",
    }
    for nome, tab in tabelas.items():
        escrever(tab, destino, nome, legendas[nome])

    figuras = []
    if com_figuras:
        figuras = [figura_q1(summary, figuras_dir, erro),
                   figura_q2(summary, figuras_dir),
                   figura_q3(q3, figuras_dir)]
        for p in figuras:
            if p:
                print(f"  {p.name}")

    texto = respostas(cfg, tabelas, cob, figuras)
    (destino / "respostas.md").write_text(texto, encoding="utf-8")
    print(f"\n{destino / 'respostas.md'}")
    return {"pasta": destino, "tabelas": tabelas, "figuras": figuras}
