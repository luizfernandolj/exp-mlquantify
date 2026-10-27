"""Reuniao dos resultados e analise das questoes de pesquisa.

Junta os arquivos parciais de todas as maquinas e produz, em
``results/<run_id>/``:

``summary.csv``
    uma linha por (conjunto, biblioteca, metodo, tamanho de amostra), com
    erro, tempo, memoria e situacao. E o arquivo principal.
``per_sample.csv``
    uma linha por amostra, com a prevalencia real e a estimada.
``q1_erro.csv``, ``q2_custo.csv``, ``q3_uso.csv``
    as tabelas que respondem diretamente a cada questao.
``testes_estatisticos.csv``, ``nemenyi_<medida>.csv``
    Friedman e Nemenyi com alfa = 0,05, como recomendado por Demsar (2006).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

#: Medidas comparadas entre bibliotecas. (coluna, rotulo, menor_e_melhor)
MEASURES = [
    ("mae", "erro absoluto (AE)", True),
    ("mrae", "erro absoluto relativo (RAE)", True),
    ("tempo_treino_s", "tempo de treinamento", True),
    ("tempo_estimacao_total_s", "tempo de estimacao", True),
    ("memoria_treino_mb", "memoria no treinamento", True),
    ("memoria_estimacao_mb", "memoria na estimacao", True),
]

REFERENCE = "mlquantify"


# --------------------------------------------------------------------------
# Leitura dos parciais
# --------------------------------------------------------------------------

def _read_parts(run_dir: Path, pattern: str) -> pd.DataFrame:
    paths = sorted((run_dir / "parts").rglob(pattern))
    if not paths:
        return pd.DataFrame()
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            print(f"  aviso: nao foi possivel ler {path.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def collect(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Le os parciais e remove repeticoes deixadas por execucoes reiniciadas."""
    summary = _read_parts(cfg.run_dir, "*.summary.csv")
    samples = _read_parts(cfg.run_dir, "*.samples.csv")

    if not summary.empty:
        keys = ["dataset_id", "biblioteca", "metodo", "tamanho_amostra"]
        summary = summary.drop_duplicates(subset=keys, keep="last")
        summary = summary.sort_values(keys).reset_index(drop=True)
    if not samples.empty:
        keys = ["dataset_id", "biblioteca", "metodo", "tamanho_amostra", "indice_amostra"]
        samples = samples.drop_duplicates(subset=keys, keep="last")
    return summary, samples


def expand_aliases(summary: pd.DataFrame) -> pd.DataFrame:
    """Repete cada linha sob os nomes equivalentes do metodo.

    Bibliotecas diferentes dao nomes diferentes a mesma formulacao: o GACC
    da mlquantify corresponde ao ACC da QuaPy, da QuantificationLib e da
    qunfold, que ja sao multiclasse nativos. Sem essa equivalencia, a
    comparacao por nome compararia formulacoes distintas. A coluna
    ``metodo_original`` preserva o nome usado na execucao.
    """
    from .registry import all_specs

    if summary.empty:
        return summary

    out = summary.assign(metodo_original=summary["metodo"], via_alias=False)
    extra = []
    existing = set(zip(summary["biblioteca"], summary["metodo"]))
    for spec in all_specs():
        if not spec.aliases:
            continue
        sub = summary[(summary["biblioteca"] == spec.library)
                      & (summary["metodo"] == spec.canonical)]
        if sub.empty:
            continue
        for alias in spec.aliases:
            if (spec.library, alias) in existing:
                continue  # a biblioteca ja tem um metodo proprio com esse nome
            extra.append(sub.assign(metodo=alias, metodo_original=spec.canonical,
                                    via_alias=True))
    if extra:
        out = pd.concat([out] + extra, ignore_index=True)
    return out


# --------------------------------------------------------------------------
# Testes estatisticos
# --------------------------------------------------------------------------

def _friedman_nemenyi(block: pd.DataFrame, alpha: float = 0.05):
    """Testa a diferenca entre bibliotecas na matriz conjuntos x bibliotecas.

    ``block`` tem uma linha por conjunto de dados e uma coluna por
    biblioteca, com valores ja completos (blocos incompletos sao descartados
    antes de chamar esta funcao).

    Com tres ou mais bibliotecas, Friedman e, havendo diferenca, Nemenyi,
    como em Demsar (2006). Com exatamente duas, Friedman nao se aplica -- o
    scipy exige tres grupos -- e o teste indicado para o par e o de Wilcoxon
    por postos sinalizados. Varios metodos existem em apenas duas das
    bibliotecas comparadas, entao este caso e comum, nao excecao.

    Devolve ``((nome_do_teste, estatistica, p), nemenyi)``.
    """
    from scipy.stats import friedmanchisquare, wilcoxon

    if block.shape[1] < 2 or len(block) < 3:
        return None, None

    colunas = [block[c].to_numpy() for c in block.columns]

    # Implementacoes equivalentes dao erro identico, que e justamente o que
    # Q1 investiga. Sem diferenca nenhuma nao ha postos a ordenar: o
    # Wilcoxon recusa a amostra e o Friedman divide por zero.
    if all(np.allclose(colunas[0], outra) for outra in colunas[1:]):
        nome = "friedman" if block.shape[1] > 2 else "wilcoxon"
        return (nome, 0.0, 1.0), None

    if block.shape[1] == 2:
        try:
            stat, p = wilcoxon(colunas[0], colunas[1])
        except ValueError as exc:  # amostra que o teste nao aceita
            print(f"  aviso: Wilcoxon indisponivel ({exc})")
            return None, None
        return ("wilcoxon", float(stat), float(p)), None

    stat, p = friedmanchisquare(*colunas)
    nemenyi = None
    if p < alpha:
        try:
            import scikit_posthocs as sp

            nemenyi = sp.posthoc_nemenyi_friedman(block.to_numpy())
            nemenyi.index = block.columns
            nemenyi.columns = block.columns
        except Exception as exc:
            print(f"  aviso: Nemenyi indisponivel ({exc})")
    return ("friedman", float(stat), float(p)), nemenyi


def _blocks(df: pd.DataFrame, column: str, by_method: bool = True):
    """Gera (rotulo, matriz conjuntos x bibliotecas) com blocos completos."""
    ok = df[(df["status"] == "ok") & df[column].notna()]
    groups = ok.groupby("metodo") if by_method else [("todos", ok)]
    for method, sub in groups:
        block = sub.pivot_table(index="dataset_id", columns="biblioteca",
                                values=column, aggfunc="mean")
        block = block.dropna(axis=0, how="any")
        if block.shape[1] >= 2 and len(block) >= 3:
            yield method, block


def statistical_tests(summary: pd.DataFrame, alpha: float = 0.05):
    """Friedman e Nemenyi por metodo e por medida."""
    rows, nemenyi_tables = [], {}
    for column, label, lower_better in MEASURES:
        if column not in summary.columns:
            continue
        for method, block in _blocks(summary, column):
            result, nemenyi = _friedman_nemenyi(block, alpha)
            if result is None:
                continue
            teste, stat, p = result
            ranks = block.rank(axis=1, ascending=lower_better).mean()
            rows.append({
                "medida": column, "rotulo": label, "metodo": method,
                "n_conjuntos": len(block),
                "bibliotecas": "|".join(block.columns),
                "teste": teste,
                "estatistica": round(stat, 4), "p_valor": p,
                "diferenca_significativa": bool(p < alpha),
                **{f"rank_{c}": round(ranks[c], 3) for c in block.columns},
            })
            if nemenyi is not None:
                nemenyi_tables[f"{column}__{method}"] = nemenyi
    return pd.DataFrame(rows), nemenyi_tables


# --------------------------------------------------------------------------
# Q1 - erro
# --------------------------------------------------------------------------

def q1_error(summary: pd.DataFrame) -> pd.DataFrame:
    """Erro por metodo e biblioteca, e distancia ate a biblioteca proposta.

    Para cada metodo implementado em mais de uma biblioteca, compara o erro
    medio sobre os conjuntos em que *todas* as bibliotecas do par produziram
    estimativa, de modo que a media nao dependa de quais conjuntos cada uma
    conseguiu rodar.
    """
    ok = summary[summary["status"] == "ok"]
    rows = []
    for (method, m), sub in ok.groupby(["metodo", "tamanho_amostra"]):
        libs = sorted(sub["biblioteca"].unique())
        if REFERENCE not in libs:
            continue
        pivot_ae = sub.pivot_table(index="dataset_id", columns="biblioteca",
                                   values="mae", aggfunc="mean")
        pivot_rae = sub.pivot_table(index="dataset_id", columns="biblioteca",
                                    values="mrae", aggfunc="mean")
        for lib in libs:
            if lib == REFERENCE:
                continue
            pair_ae = pivot_ae[[REFERENCE, lib]].dropna()
            pair_rae = pivot_rae[[REFERENCE, lib]].dropna()
            if pair_ae.empty:
                continue
            diff = pair_ae[REFERENCE] - pair_ae[lib]
            rows.append({
                "metodo": method, "tamanho_amostra": m, "biblioteca": lib,
                "n_conjuntos_pareados": len(pair_ae),
                "mae_mlquantify": pair_ae[REFERENCE].mean(),
                "mae_referencia": pair_ae[lib].mean(),
                "diferenca_mae_media": diff.mean(),
                "diferenca_mae_absoluta_media": diff.abs().mean(),
                "diferenca_mae_maxima": diff.abs().max(),
                "mrae_mlquantify": pair_rae[REFERENCE].mean() if len(pair_rae) else np.nan,
                "mrae_referencia": pair_rae[lib].mean() if len(pair_rae) else np.nan,
                "correlacao_mae": pair_ae[REFERENCE].corr(pair_ae[lib]),
                "conjuntos_equivalentes_1e3": float((diff.abs() < 1e-3).mean()),
                "conjuntos_equivalentes_1e2": float((diff.abs() < 1e-2).mean()),
                "mlquantify_melhor": float((diff < 0).mean()),
            })
    return pd.DataFrame(rows).sort_values(["metodo", "tamanho_amostra", "biblioteca"])


# --------------------------------------------------------------------------
# Q2 - custo
# --------------------------------------------------------------------------

def q2_cost(summary: pd.DataFrame) -> pd.DataFrame:
    """Tempo e memoria por metodo e biblioteca, com razao sobre a proposta."""
    ok = summary[summary["status"] == "ok"]
    # Medicoes feitas enquanto a maquina esteve suspensa, ou sob disputa
    # pesada, nao entram na comparacao de custo (o erro delas continua
    # valido e e usado no Q1).
    if "medida_suspeita" in ok.columns:
        suspeitas = ok["medida_suspeita"].astype(str).str.lower() == "true"
        if suspeitas.any():
            print(f"  q2: {int(suspeitas.sum())} linhas com medida suspeita "
                  "foram descartadas do custo")
        ok = ok[~suspeitas]
    # o custo e por metodo, nao por tamanho de amostra: uma linha basta
    cost = ok.drop_duplicates(subset=["dataset_id", "biblioteca", "metodo"])
    rows = []
    for method, sub in cost.groupby("metodo"):
        libs = sorted(sub["biblioteca"].unique())
        if REFERENCE not in libs:
            continue
        for column in ("tempo_treino_s", "tempo_estimacao_total_s",
                       "memoria_treino_mb", "memoria_estimacao_mb"):
            pivot = sub.pivot_table(index="dataset_id", columns="biblioteca",
                                    values=column, aggfunc="median")
            for lib in libs:
                if lib == REFERENCE or lib not in pivot.columns:
                    continue
                pair = pivot[[REFERENCE, lib]].dropna()
                if pair.empty:
                    continue
                ratio = pair[REFERENCE] / pair[lib].replace(0, np.nan)
                rows.append({
                    "metodo": method, "medida": column, "biblioteca": lib,
                    "n_conjuntos_pareados": len(pair),
                    "mlquantify_mediana": pair[REFERENCE].median(),
                    "referencia_mediana": pair[lib].median(),
                    "mlquantify_media": pair[REFERENCE].mean(),
                    "referencia_media": pair[lib].mean(),
                    "razao_mediana": ratio.median(),
                    "mlquantify_mais_rapida": float((pair[REFERENCE] < pair[lib]).mean()),
                })
    return pd.DataFrame(rows).sort_values(["medida", "metodo", "biblioteca"])


# --------------------------------------------------------------------------
# Q3 - esforco de uso
# --------------------------------------------------------------------------

def q3_usage(summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Configuracoes manuais, taxa de sucesso sem ajustes e linhas de codigo."""
    rows = []
    loc = _load_loc(cfg)
    for lib, sub in summary.groupby("biblioteca"):
        per_method = sub.drop_duplicates(subset=["dataset_id", "metodo"])
        total = len(per_method)
        ok = int((per_method["status"] == "ok").sum())

        # A forma direta so e sondada quando difere da documentada; onde nao
        # ha sondagem, a propria execucao ja foi sem configuracao manual.
        direct = per_method["forma_direta_ok"]
        probed = direct.notna() & (direct.astype(str) != "")
        direct_ok = int(((direct.astype(str).str.lower() == "true") & probed).sum())
        not_probed_ok = int((~probed & (per_method["status"] == "ok")).sum())

        rows.append({
            "biblioteca": lib,
            "combinacoes": total,
            "concluidas_sem_erro": ok,
            "taxa_conclusao": round(ok / total, 4) if total else np.nan,
            "combinacoes_que_exigem_ajuste": int(probed.sum()),
            "concluidas_sem_configuracao_manual": direct_ok + not_probed_ok,
            "taxa_sem_configuracao_manual": (
                round((direct_ok + not_probed_ok) / total, 4) if total else np.nan),
            "configuracoes_manuais_total": int(
                per_method["configuracoes_manuais"].fillna(0).sum()),
            "configuracoes_manuais_media": round(
                float(per_method["configuracoes_manuais"].fillna(0).mean()), 3),
            "metodos_distintos": per_method["metodo"].nunique(),
            "linhas_de_codigo": loc.get(lib, {}).get("linhas_de_codigo", np.nan),
            "configuracoes_manuais_script": loc.get(lib, {}).get("configuracoes_manuais",
                                                                 np.nan),
        })
    return pd.DataFrame(rows).sort_values("biblioteca")


def _load_loc(cfg: Config) -> dict:
    path = cfg.catalog_dir / "q3_linhas_de_codigo.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {r["biblioteca"]: r for _, r in df.iterrows()}


# --------------------------------------------------------------------------
# Visao geral
# --------------------------------------------------------------------------

def overview(summary: pd.DataFrame) -> pd.DataFrame:
    """Contagem por biblioteca e situacao, util para conferir a execucao."""
    per_method = summary.drop_duplicates(subset=["dataset_id", "biblioteca", "metodo"])
    tab = (per_method.groupby(["biblioteca", "status"]).size()
           .unstack(fill_value=0).reset_index())
    tab["total"] = tab.select_dtypes("number").sum(axis=1)

    # Combinacoes concluidas em que alguma amostra foi descartada: contam
    # como concluidas, mas com menos de 400 amostras medidas.
    if "amostras_descartadas_total" in per_method.columns:
        desc = pd.to_numeric(per_method["amostras_descartadas_total"],
                             errors="coerce").fillna(0)
        parciais = (per_method.assign(_d=desc)
                    .query("status == 'ok' and _d > 0")
                    .groupby("biblioteca").size())
        tab["ok_com_descarte"] = tab["biblioteca"].map(parciais).fillna(0).astype(int)
    return tab


def discarded_samples(summary: pd.DataFrame) -> pd.DataFrame:
    """Amostras descartadas por combinacao, com o motivo.

    Uma estimativa invalida em uma amostra isolada nao reprova a combinacao
    (ver ``MAX_INVALID_FRACTION`` em ``expq/runner.py``), entao o descarte
    tem de ficar visivel em algum lugar: e aqui.
    """
    if "amostras_descartadas_total" not in summary.columns:
        return pd.DataFrame()
    df = summary.copy()
    df["_total"] = pd.to_numeric(df["amostras_descartadas_total"],
                                 errors="coerce").fillna(0)
    df["_tamanho"] = pd.to_numeric(df["n_amostras_descartadas"],
                                   errors="coerce").fillna(0)
    com_descarte = df[df["_total"] > 0]
    if com_descarte.empty:
        return pd.DataFrame()

    rows = []
    for (ds, lib, met), sub in com_descarte.groupby(
            ["dataset_id", "biblioteca", "metodo"]):
        por_tamanho = {int(r["tamanho_amostra"]): int(r["_tamanho"])
                       for _, r in sub.iterrows() if r["_tamanho"] > 0}
        rows.append({
            "dataset_id": ds, "biblioteca": lib, "metodo": met,
            "status": sub["status"].iloc[0],
            "n_classes": sub["n_classes"].iloc[0],
            "amostras_descartadas": int(sub["_total"].max()),
            "amostras_medidas": int(pd.to_numeric(sub["n_amostras"],
                                                  errors="coerce").fillna(0).sum()),
            "tamanhos_afetados": "|".join(f"{m}:{n}" for m, n
                                          in sorted(por_tamanho.items())),
            "motivo": sub["motivo_descarte"].iloc[0],
        })
    return pd.DataFrame(rows).sort_values(
        ["amostras_descartadas", "biblioteca", "metodo"], ascending=[False, True, True])


def run(cfg: Config, write_samples: bool = True) -> dict:
    """Reune tudo e grava os arquivos finais."""
    print(f"Reunindo resultados de {cfg.run_dir} ...")
    summary, samples = collect(cfg)
    if summary.empty:
        print("Nenhum resultado encontrado. A execucao ja terminou?")
        return {}

    out = {}
    cfg.run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = cfg.run_dir / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    out["summary"] = summary_path
    print(f"  summary.csv          {len(summary):>8} linhas")

    # As comparacoes entre bibliotecas usam os nomes equivalentes de metodo.
    comparable = expand_aliases(summary)

    if write_samples and not samples.empty:
        samples_path = cfg.run_dir / "per_sample.csv"
        samples.to_csv(samples_path, index=False, encoding="utf-8")
        out["per_sample"] = samples_path
        print(f"  per_sample.csv       {len(samples):>8} linhas")

    for name, frame in (("q1_erro", q1_error(comparable)),
                        ("q2_custo", q2_cost(comparable)),
                        ("q3_uso", q3_usage(summary, cfg)),
                        ("visao_geral", overview(summary)),
                        ("amostras_descartadas", discarded_samples(summary))):
        if frame.empty and name == "amostras_descartadas":
            continue
        path = cfg.run_dir / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        out[name] = path
        print(f"  {name + '.csv':<20} {len(frame):>8} linhas")

    tests, nemenyi = statistical_tests(comparable)
    tests_path = cfg.run_dir / "testes_estatisticos.csv"
    tests.to_csv(tests_path, index=False, encoding="utf-8")
    out["testes"] = tests_path
    print(f"  {'testes_estatisticos.csv':<20} {len(tests):>8} linhas")

    if nemenyi:
        nem_dir = cfg.run_dir / "nemenyi"
        nem_dir.mkdir(exist_ok=True)
        for key, table in nemenyi.items():
            table.to_csv(nem_dir / f"{key.replace('(', '').replace(')', '')}.csv",
                         encoding="utf-8")
        print(f"  nemenyi/             {len(nemenyi):>8} tabelas")

    print("\nvisao geral por biblioteca:")
    print(overview(summary).to_string(index=False))
    return out
