"""Execucao de um job: um conjunto de dados em uma biblioteca.

Um job percorre todos os metodos que a biblioteca implementa e, para cada
metodo, segue os passos da Secao 4.3.4:

1. o quantificador e treinado sobre a particao de treinamento;
2. estima a prevalencia de todas as amostras geradas pelo protocolo UPP;
3. sao registrados erro (AE, RAE, SE, NKLD), tempo e pico de memoria.

As medidas de tempo sao repetidas ``cost_repeats`` vezes e resumidas pela
mediana. O pico de memoria e medido em uma repeticao separada, porque o
``tracemalloc`` altera o tempo de execucao.

Os resultados de cada metodo sao gravados assim que ficam prontos, e uma
linha e acrescentada ao ledger. Uma execucao interrompida retoma exatamente
de onde parou, sem precisar ler os arquivos de resultado.
"""

from __future__ import annotations

import csv
import time
import tracemalloc
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from .config import Config
from .metrics import (PrevalenceError, PrevalenceValueError, all_metrics,
                      normalize_prevalence, prevalence_from_labels)
from .registry import Ctx, descricao_protocolo, specs_for


class MethodTimeout(RuntimeError):
    pass


class TooManyInvalidSamples(RuntimeError):
    pass


#: Diferenca, em segundos, entre relogio e tempo de CPU acima da qual a
#: medicao de custo de um metodo e marcada como suspeita.
SUSPECT_GAP = 300.0

#: Fracao das amostras do protocolo que pode ter estimativa invalida sem que
#: a combinacao seja reprovada.
#:
#: Alguns metodos devolvem, em uma amostra isolada, um vetor sem massa
#: nenhuma -- os de limiar da mlquantify fazem isso em amostras de 10
#: instancias de problemas multiclasse, quando a decomposicao
#: um-contra-resto estima zero para todas as classes. Reprovar a combinacao
#: inteira por causa disso descartaria as outras 399 medidas, que sao
#: validas, e subestimaria a cobertura da biblioteca. Acima deste limite,
#: porem, nao se trata mais de amostra isolada: o metodo nao esta estimando
#: nada, e a combinacao e reprovada.
MAX_INVALID_FRACTION = 0.05


# --------------------------------------------------------------------------
# Classificador base
# --------------------------------------------------------------------------

def make_classifier_factory(cfg: Config):
    """Devolve uma funcao que cria um classificador base novo a cada chamada.

    A regressao logistica do scikit-learn e usada em todas as bibliotecas,
    sempre com a mesma semente, de modo que a comparacao isole o
    quantificador e nao o classificador (Secao 4.3.4).
    """
    from sklearn.linear_model import LogisticRegression

    def make():
        clf = LogisticRegression(max_iter=cfg.classifier_max_iter,
                                 random_state=cfg.random_state)
        if cfg.classifier_pipeline == "scaled":
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), clf)
        return clf

    return make


def _semear(cfg: Config) -> None:
    """Fixa as fontes globais de aleatoriedade antes de cada celula.

    Nem toda aleatoriedade tem parametro no construtor. A quapy le a semente
    de ``quapy.environ['_R_SEED']`` no Monte Carlo da KDEy e na selecao de
    modelo, e sobra o estado global do numpy, que varias bibliotecas usam
    sem dizer. Sem isto, repetir a execucao nao repete os numeros.
    """
    np.random.seed(cfg.random_state)
    try:
        import quapy

        quapy.environ["_R_SEED"] = cfg.random_state
    except Exception:  # quapy ausente nesta celula: nada a semear
        pass


# --------------------------------------------------------------------------
# Colunas dos arquivos de saida
# --------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    # identificacao
    "run_id", "node", "timestamp", "dataset_id", "nome_arquivo", "origem",
    "tipo_dataset", "n_instancias", "n_atributos", "n_classes",
    "razao_desbalanceamento", "n_treino", "n_teste",
    "biblioteca", "versao_biblioteca", "metodo", "familia", "grupo", "classe",
    "aliases",
    # protocolo
    "protocolo", "tamanho_amostra", "n_amostras", "n_amostras_descartadas",
    "amostras_descartadas_total", "motivo_descarte", "n_prevalencias",
    "semente", "proporcao_teste", "protocolo_escores",
    # situacao e esforco de uso (Q3)
    "status", "tipo_erro", "mensagem_erro", "forma_direta_ok",
    "erro_forma_direta", "usou_decomposicao_manual", "configuracoes_manuais",
    # erro de quantificacao (Q1)
    "mae", "ae_dp", "ae_mediana", "ae_p25", "ae_p75", "ae_max",
    "mrae", "rae_dp", "rae_mediana", "rae_p25", "rae_p75", "rae_max",
    "mse", "nkld_medio", "rae_eps",
    "prev_real_min", "prev_real_max",
    # custo (Q2)
    "tempo_treino_s", "tempo_treino_dp", "tempo_estimacao_s",
    "tempo_estimacao_por_amostra_s", "tempo_estimacao_total_s", "tempo_total_s",
    "memoria_treino_mb", "memoria_estimacao_mb", "repeticoes_custo",
    "cpu_total_s", "medida_suspeita",
    # ambiente
    "python", "sklearn", "threads",
]

SAMPLE_COLUMNS = [
    "run_id", "node", "dataset_id", "tipo_dataset", "biblioteca", "metodo",
    "tamanho_amostra", "indice_amostra", "prev_real", "prev_estimada",
    "ae", "rae", "se", "nkld", "tempo_estimacao_s",
]


#: Grupos de colunas acrescentados ao ``summary.csv`` com a execucao final ja
#: em andamento, do mais recente para o mais antigo. O numero de campos de uma
#: linha diz sob qual esquema ela foi gravada, e e assim que ``_realign`` a le
#: sem deslocar nada. Ao acrescentar colunas novas, registre-as aqui.
SUMMARY_ANTERIORES = (
    ("protocolo_escores",),
    ("n_amostras_descartadas", "amostras_descartadas_total", "motivo_descarte"),
    ("cpu_total_s", "medida_suspeita"),
)


def _esquemas(columns: list[str], anteriores: tuple) -> dict[int, list[str]]:
    """Mapa ``{numero de campos: nomes}`` do esquema atual e dos anteriores."""
    esquemas = {len(columns): list(columns)}
    atual = list(columns)
    for grupo in anteriores:
        atual = [c for c in atual if c not in grupo]
        esquemas.setdefault(len(atual), list(atual))
    return esquemas


def _append_rows(path: Path, columns: list[str], rows: list[dict],
                 anteriores: tuple = ()) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if not new:
        _realign(path, columns, anteriores)
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()


def _realign(path: Path, columns: list[str], anteriores: tuple = ()) -> None:
    """Reescreve o arquivo com as colunas atuais, se o cabecalho envelheceu.

    Um arquivo parcial gravado por uma versao anterior tem o cabecalho de
    entao. Acrescentar linhas novas a ele escreveria os valores na ordem
    nova sob o cabecalho velho, deslocando as colunas sem erro nenhum. Como
    os parciais sao pequenos, o arquivo e simplesmente reescrito.

    As linhas sao lidas pela posicao, e nao pelo cabecalho: um parcial pode
    ter linhas de esquemas diferentes, se recebeu linhas novas antes de haver
    o realinhamento. O numero de campos da linha escolhe o esquema em
    ``anteriores``; linha de tamanho desconhecido interrompe a gravacao, para
    nao transformar um desalinhamento visivel em valores trocados em silencio.
    """
    esquemas = _esquemas(columns, anteriores)
    with path.open(newline="", encoding="utf-8") as fh:
        linhas = list(csv.reader(fh))
    if not linhas:
        return
    if linhas[0] == list(columns) and all(len(l) == len(columns) for l in linhas[1:]):
        return

    antigas = []
    for n, linha in enumerate(linhas[1:], 2):
        nomes = esquemas.get(len(linha))
        if nomes is None:
            raise ValueError(
                f"{path.name}: linha {n} tem {len(linha)} campos, que nao "
                f"correspondem a nenhum esquema conhecido {sorted(esquemas)}")
        antigas.append(dict(zip(nomes, linha)))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(antigas)


# --------------------------------------------------------------------------
# Medicao de um metodo
# --------------------------------------------------------------------------

def _predict_all(quantifier, X_test, samples, n_classes, deadline):
    """Estima a prevalencia de todas as amostras, cronometrando cada uma.

    O limite e verificado sobre o tempo de CPU do processo, e nao sobre o
    relogio: se a maquina for suspensa no meio da execucao, o relogio salta
    horas e um metodo barato pareceria ter estourado o limite.

    Devolve ``(preds, tempos, invalidas)``. A amostra cuja estimativa nao e
    um vetor de prevalencias valido entra como ``None`` em ``preds``, para
    nao desalinhar os indices, e e contada em ``invalidas``; o tempo dela
    fica, porque a estimativa foi calculada, so nao serve. Acima de
    ``MAX_INVALID_FRACTION`` a combinacao inteira e reprovada.

    Nao importa quem recusou a estimativa. Algumas bibliotecas checam o
    vetor por conta propria e levantam a excecao dentro do ``predict`` -- a
    QuaPy faz isso quando o ``SMM`` multiclasse produz prevalencia negativa
    --, e para a comparacao isso e a mesma coisa que o nosso proprio
    ``normalize_prevalence`` recusar a saida. Ja falta de memoria e tempo
    limite nao sao estimativa invalida: dizem respeito a medicao inteira e
    sobem.
    """
    preds, times, invalidas = [], [], []
    for i in range(len(samples)):
        if deadline is not None and time.process_time() > deadline:
            raise MethodTimeout(f"tempo limite excedido apos {i} amostras")
        idx = samples.get(i)
        t0 = time.perf_counter()
        try:
            estimativa = normalize_prevalence(
                quantifier.predict(X_test[idx]), n_classes)
        except (MemoryError, MethodTimeout):
            raise
        except PrevalenceError as exc:
            estimativa = None
            invalidas.append((i, str(exc)))
        except Exception as exc:
            estimativa = None
            invalidas.append((i, f"{type(exc).__name__}: {exc}"[:200]))
        times.append(time.perf_counter() - t0)
        preds.append(estimativa)

    limite = MAX_INVALID_FRACTION * len(samples)
    if len(invalidas) > limite:
        raise TooManyInvalidSamples(
            f"{len(invalidas)} de {len(samples)} amostras com estimativa "
            f"invalida (limite {limite:.0f}): {_motivo(invalidas)}")
    return preds, np.asarray(times), invalidas


def _motivo(invalidas) -> str:
    """A mensagem mais frequente entre as amostras descartadas."""
    if not invalidas:
        return ""
    return Counter(msg for _, msg in invalidas).most_common(1)[0][0]


def _measure_memory(build, X_train, y_train, X_test, samples):
    """Pico de memoria no treinamento e na estimacao, via ``tracemalloc``."""
    tracemalloc.start()
    try:
        quantifier = build()
        quantifier.fit(X_train, y_train)
        fit_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
        for i in range(len(samples)):
            quantifier.predict(X_test[samples.get(i)])
        predict_peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return fit_peak / 1e6, predict_peak / 1e6


def _probe_naive(spec, ctx, X_train, y_train, X_test, samples, n_classes):
    """Tenta a forma direta do metodo, sem nenhuma configuracao manual.

    Alimenta a proporcao de combinacoes concluidas sem configuracao manual
    (Q3). So faz o treinamento e uma estimacao, porque interessa apenas se a
    biblioteca aceita o uso direto.
    """
    try:
        quantifier = spec.naive(ctx)
        quantifier.fit(X_train, y_train)
        raw = quantifier.predict(X_test[samples.get(0)])
        normalize_prevalence(raw, n_classes)
        return True, ""
    except PrevalenceValueError:
        # A chamada direta funcionou: o formato esta certo e so a estimativa
        # desta amostra veio degenerada. Para Q3 isso conta como aceito, que
        # e a pergunta aqui -- a qualidade da estimativa e assunto de Q1.
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:300]


def run_method(spec, ctx, cfg, dataset_info, X_train, y_train, X_test, y_test,
               samples, n_classes, report=None):
    """Executa um metodo e devolve (linhas_resumo, linhas_por_amostra)."""
    import platform

    import sklearn

    base = {
        "run_id": cfg.run_id, "node": cfg.node,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **dataset_info,
        "n_treino": len(y_train), "n_teste": X_test.shape[0],
        "biblioteca": spec.library, "versao_biblioteca": _version(spec.library),
        "metodo": spec.canonical, "familia": spec.family, "grupo": spec.group,
        "classe": spec.class_name, "aliases": "|".join(spec.aliases),
        "protocolo": cfg.protocol, "n_prevalencias": cfg.n_prevalences,
        "semente": cfg.random_state, "proporcao_teste": cfg.test_size,
        "protocolo_escores": descricao_protocolo(cfg.random_state),
        "usou_decomposicao_manual": bool(spec.binary_only and ctx.is_multiclass),
        "configuracoes_manuais": spec.manual_count(ctx),
        "repeticoes_custo": cfg.cost_repeats,
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "threads": cfg.threads_per_worker,
    }

    if not spec.available:
        base.update({"status": "indisponivel", "tipo_erro": "NotImplemented",
                     "mensagem_erro": spec.unavailable_reason,
                     "forma_direta_ok": "", "erro_forma_direta": ""})
        return [{**base, "tamanho_amostra": -1, "n_amostras": 0}], []

    _semear(cfg)

    # --- Q3: a biblioteca aceita o uso direto, sem configuracao manual? ---
    if spec.naive_differs(ctx):
        naive_ok, naive_err = _probe_naive(spec, ctx, X_train, y_train, X_test,
                                           samples, n_classes)
    else:
        naive_ok, naive_err = None, ""  # forma direta e documentada coincidem

    # O prazo cobre a estimativa das amostras, nao o ajuste: quem o verifica
    # e o ``_predict_all``, entre uma amostra e a proxima -- e a primeira
    # verificacao ja acontece com o orcamento gasto no ``fit``, que com
    # ``cost_repeats`` e repetido. Um ajuste lento estoura o limite e sai
    # como ``timeout`` acima dele; um ajuste que nao devolve nunca e
    # alcancado, e so o ``--job-timeout`` encerra. O relogio de CPU e proposital
    # (ver ``_predict_all``), mas tem o avesso: ele tambem para enquanto o
    # processo espera disco, entao sob paginacao pesada a contagem congela.
    deadline = time.process_time() + cfg.method_timeout if cfg.method_timeout else None
    t_total = time.perf_counter()
    cpu_total = time.process_time()

    try:
        fit_times, sample_times, preds, invalidas = [], [], None, []
        for rep in range(max(1, cfg.cost_repeats)):
            quantifier = spec.adapted(ctx)
            t0 = time.perf_counter()
            quantifier.fit(X_train, y_train)
            fit_times.append(time.perf_counter() - t0)

            rep_preds, rep_times, rep_invalidas = _predict_all(
                quantifier, X_test, samples, n_classes, deadline)
            sample_times.append(rep_times)
            if preds is None:
                preds, invalidas = rep_preds, rep_invalidas

        fit_median = float(np.median(fit_times))
        fit_std = float(np.std(fit_times))
        per_sample = np.median(np.vstack(sample_times), axis=0)

        mem_fit = mem_predict = float("nan")
        if cfg.measure_memory:
            mem_fit, mem_predict = _measure_memory(
                lambda: spec.adapted(ctx), X_train, y_train, X_test, samples)

    except Exception as exc:
        status = "timeout" if isinstance(exc, MethodTimeout) else "erro"
        base.update({
            "status": status, "tipo_erro": type(exc).__name__,
            "mensagem_erro": str(exc)[:300].replace("\n", " "),
            "forma_direta_ok": "" if naive_ok is None else naive_ok,
            "erro_forma_direta": naive_err,
            "tempo_total_s": round(time.perf_counter() - t_total, 6),
            "cpu_total_s": round(time.process_time() - cpu_total, 6),
            "medida_suspeita": True,
        })
        if report:
            report(f"{spec.canonical}: {status} ({type(exc).__name__})")
        return [{**base, "tamanho_amostra": -1, "n_amostras": 0}], []

    total_time = time.perf_counter() - t_total
    total_cpu = time.process_time() - cpu_total
    # Se o relogio andou muito mais que a CPU, a maquina foi suspensa (ou
    # houve disputa pesada) durante a medicao: os tempos deste metodo nao
    # servem para a comparacao de custo.
    suspect = (total_time - total_cpu) > SUSPECT_GAP

    true_prevs = [prevalence_from_labels(y_test[samples.get(i)], n_classes)
                  for i in range(len(samples))]

    motivo_descarte = _motivo(invalidas)

    summary_rows, sample_rows = [], []
    for m in sorted(set(int(s) for s in samples.sizes)):
        do_tamanho = samples.by_size(m)
        idxs = [i for i in do_tamanho if preds[i] is not None]
        if not idxs:
            continue  # nada medido neste tamanho
        mets = [all_metrics(true_prevs[i], preds[i], m) for i in idxs]
        aes = np.array([x["ae"] for x in mets])
        raes = np.array([x["rae"] for x in mets])
        ses = np.array([x["se"] for x in mets])
        nklds = np.array([x["nkld"] for x in mets])
        times_m = per_sample[idxs]
        prev_min = min(float(true_prevs[i].min()) for i in idxs)
        prev_max = max(float(true_prevs[i].max()) for i in idxs)

        summary_rows.append({
            **base,
            "status": "ok", "tipo_erro": "", "mensagem_erro": "",
            "forma_direta_ok": "" if naive_ok is None else naive_ok,
            "erro_forma_direta": naive_err,
            "tamanho_amostra": m, "n_amostras": len(idxs),
            "n_amostras_descartadas": len(do_tamanho) - len(idxs),
            "amostras_descartadas_total": len(invalidas),
            "motivo_descarte": motivo_descarte,
            "mae": aes.mean(), "ae_dp": aes.std(), "ae_mediana": np.median(aes),
            "ae_p25": np.percentile(aes, 25), "ae_p75": np.percentile(aes, 75),
            "ae_max": aes.max(),
            "mrae": raes.mean(), "rae_dp": raes.std(), "rae_mediana": np.median(raes),
            "rae_p25": np.percentile(raes, 25), "rae_p75": np.percentile(raes, 75),
            "rae_max": raes.max(),
            "mse": ses.mean(), "nkld_medio": nklds.mean(), "rae_eps": 1.0 / (2 * m),
            "prev_real_min": prev_min, "prev_real_max": prev_max,
            "tempo_treino_s": fit_median, "tempo_treino_dp": fit_std,
            "tempo_estimacao_s": float(times_m.sum()),
            "tempo_estimacao_por_amostra_s": float(times_m.mean()),
            "tempo_estimacao_total_s": float(per_sample.sum()),
            "tempo_total_s": total_time,
            "memoria_treino_mb": mem_fit, "memoria_estimacao_mb": mem_predict,
            "cpu_total_s": total_cpu, "medida_suspeita": suspect,
        })

        if cfg.save_per_sample:
            for i in idxs:
                met = all_metrics(true_prevs[i], preds[i], m)
                sample_rows.append({
                    "run_id": cfg.run_id, "node": cfg.node,
                    "dataset_id": dataset_info["dataset_id"],
                    "tipo_dataset": dataset_info["tipo_dataset"],
                    "biblioteca": spec.library, "metodo": spec.canonical,
                    "tamanho_amostra": m, "indice_amostra": i,
                    "prev_real": _fmt(true_prevs[i]), "prev_estimada": _fmt(preds[i]),
                    "ae": met["ae"], "rae": met["rae"], "se": met["se"],
                    "nkld": met["nkld"], "tempo_estimacao_s": float(per_sample[i]),
                })

    return summary_rows, sample_rows


def _fmt(p) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in np.asarray(p)) + "]"


def _version(library: str) -> str:
    import importlib.metadata as md

    try:
        return md.version(library)
    except Exception:
        return "?"


# --------------------------------------------------------------------------
# Job completo
# --------------------------------------------------------------------------

def run_job(cfg: Config, dataset_row: dict, library: str, done: set,
            progress=None) -> dict:
    """Executa todos os metodos de uma biblioteca sobre um conjunto de dados."""
    from .data import load_dataset
    from .ledger import ledger_entry
    from .protocol import load_or_make, split

    dataset_id = dataset_row["dataset_id"]
    path = cfg.root / dataset_id

    ds = load_dataset(path, dataset_id, cache_dir=cfg.cache_dir / "prepared",
                      max_rows=cfg.max_rows, random_state=cfg.random_state)
    tr, te = split(ds.X, ds.y, cfg.test_size, cfg.random_state)
    X_train, y_train = ds.X[tr], ds.y[tr]
    X_test, y_test = ds.X[te], ds.y[te]

    samples = load_or_make(cfg.cache_dir / "samples", dataset_id, y_test,
                           cfg.sample_sizes, cfg.n_prevalences, cfg.random_state)

    dataset_info = {
        "dataset_id": dataset_id,
        "nome_arquivo": dataset_row.get("nome_arquivo", Path(dataset_id).name),
        "origem": dataset_row.get("origem", ""),
        "tipo_dataset": ds.dataset_type, "n_instancias": ds.n_instances,
        "n_atributos": ds.n_features, "n_classes": ds.n_classes,
        "razao_desbalanceamento": round(ds.imbalance_ratio, 4),
    }

    ctx = Ctx(make_clf=make_classifier_factory(cfg), n_classes=ds.n_classes,
              random_state=cfg.random_state)
    specs = specs_for(library, groups=cfg.groups, methods=cfg.methods,
                      skip=cfg.skip_methods)

    safe = dataset_id.replace("/", "__").replace("\\", "__").replace(".csv", "")
    summary_path = cfg.parts_dir / f"{safe}__{library}.summary.csv"
    samples_path = cfg.parts_dir / f"{safe}__{library}.samples.csv"

    import multiprocessing as mp

    parent = mp.parent_process()

    counts = {"ok": 0, "erro": 0, "timeout": 0, "indisponivel": 0, "pulado": 0}
    for n, spec in enumerate(specs, start=1):
        # Se o processo principal morreu (execucao interrompida de fora), este
        # trabalhador continuaria rodando sozinho, consumindo nucleos e
        # segurando os arquivos do ledger. Encerra junto com ele.
        if parent is not None and not parent.is_alive():
            break

        key = f"{dataset_id}|{library}|{spec.canonical}"
        if cfg.resume and key in done:
            counts["pulado"] += 1
            if progress:
                progress({"type": "method", "dataset": dataset_id, "library": library,
                          "method": spec.canonical, "i": n, "n": len(specs),
                          "status": "pulado"})
            continue

        if progress:
            progress({"type": "method_start", "dataset": dataset_id,
                      "library": library, "method": spec.canonical,
                      "i": n, "n": len(specs)})

        t0 = time.perf_counter()
        try:
            summary_rows, sample_rows = run_method(
                spec, ctx, cfg, dataset_info, X_train, y_train, X_test, y_test,
                samples, ds.n_classes)
            status = summary_rows[0]["status"] if summary_rows else "erro"
            error = summary_rows[0].get("mensagem_erro", "") if summary_rows else ""
        except Exception as exc:  # falha fora da medicao do metodo
            summary_rows, sample_rows = [], []
            status, error = "erro", f"{type(exc).__name__}: {exc}"[:300]
            traceback.print_exc()

        _append_rows(summary_path, SUMMARY_COLUMNS, summary_rows, SUMMARY_ANTERIORES)
        _append_rows(samples_path, SAMPLE_COLUMNS, sample_rows)
        ledger_entry(cfg, dataset_id=dataset_id, library=library,
                     method=spec.canonical, status=status,
                     seconds=round(time.perf_counter() - t0, 4), error=error)

        counts[status] = counts.get(status, 0) + 1
        if progress:
            progress({"type": "method", "dataset": dataset_id, "library": library,
                      "method": spec.canonical, "i": n, "n": len(specs),
                      "status": status, "seconds": time.perf_counter() - t0})

    return {"dataset_id": dataset_id, "library": library, "counts": counts,
            "n_methods": len(specs)}
