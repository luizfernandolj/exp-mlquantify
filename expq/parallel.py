"""Escalonador dos jobs em varios nucleos.

Um job e o par (conjunto de dados, biblioteca). Cada job roda em um processo
proprio, por tres razoes: o tempo limite pode ser aplicado de fato (basta
encerrar o processo), uma falha grave em uma biblioteca nao derruba a
execucao inteira, e o paralelismo fica entre processos, com uma thread por
processo, de modo que as medidas de tempo nao dependam do BLAS.

As barras de progresso mostram o total de combinacoes e, em cada
trabalhador, o conjunto de dados, a biblioteca e o metodo em andamento.
"""

from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import time
from dataclasses import fields as dataclass_fields

from tqdm import tqdm

from .config import Config
from .ledger import close_ledger, ledger_entry, load_done, silence_warnings
from .registry import specs_for

#: Salto de relogio, em segundos, acima do qual se considera que a maquina
#: foi suspensa. O laco do escalonador roda a cada 0,05 s, entao qualquer
#: coisa acima de um minuto so pode ser suspensao.
SUSPEND_GAP = 60.0

#: Quantas falhas de sistema seguidas o escalonador tolera antes de desistir.
#: Quando a maquina fica sem memoria, chamadas banais do laco -- inclusive
#: ``time.sleep`` -- passam a levantar ``OSError`` (no Windows, o
#: ``WinError 1450``, "recursos insuficientes do sistema"). Isso costuma
#: passar sozinho, porque os trabalhadores que terminam devolvem memoria,
#: entao vale insistir em vez de derrubar a execucao inteira: sem isso o
#: escalonador morre e os trabalhadores ficam orfaos, trabalhando sem que
#: ninguem lance os jobs restantes.
MAX_FALHAS_DE_SISTEMA = 600


def config_from_dict(d: dict) -> Config:
    known = {f.name for f in dataclass_fields(Config)}
    kwargs = {}
    for k, v in d.items():
        if k not in known:
            continue
        kwargs[k] = tuple(v) if isinstance(v, list) else v
    return Config(**kwargs)


def build_jobs(cfg: Config, selected) -> list[dict]:
    """Lista de jobs, intercalando as bibliotecas.

    Se os jobs forem emitidos agrupados por biblioteca, todos os
    trabalhadores acabam na mesma biblioteca ao mesmo tempo. Com a qunfold
    isso e ruim duas vezes: e a mais lenta, entao ninguem termina por horas,
    e cada processo dela carrega o JAX, entao a memoria estoura. Intercalar
    mantem a mistura de bibliotecas em qualquer momento da execucao.
    """
    por_biblioteca: dict[str, list[dict]] = {}
    for row in selected.to_dict("records"):
        for library in cfg.libraries:
            specs = specs_for(library, groups=cfg.groups, methods=cfg.methods,
                              skip=cfg.skip_methods)
            if not specs:
                continue
            por_biblioteca.setdefault(library, []).append(
                {"dataset_row": row, "library": library,
                 "methods": [s.canonical for s in specs]})

    return interleave([j for fila in por_biblioteca.values() for j in fila])


def interleave(jobs: list[dict]) -> list[dict]:
    """Reordena os jobs em rodizio entre as bibliotecas."""
    filas: dict[str, list[dict]] = {}
    for job in jobs:
        filas.setdefault(job["library"], []).append(job)
    ordenados, listas = [], list(filas.values())
    for i in range(max((len(f) for f in listas), default=0)):
        for fila in listas:
            if i < len(fila):
                ordenados.append(fila[i])
    return ordenados


def pending_units(cfg: Config, jobs: list[dict], done: set[str]) -> int:
    total = 0
    for job in jobs:
        ds = job["dataset_row"]["dataset_id"]
        for method in job["methods"]:
            if f"{ds}|{job['library']}|{method}" not in done:
                total += 1
    return total


# --------------------------------------------------------------------------
# Processo trabalhador
# --------------------------------------------------------------------------

def _worker(cfg_dict: dict, dataset_row: dict, library: str, done: set,
            progress_queue, slot: int = 0) -> None:
    silence_warnings()
    cfg = config_from_dict(cfg_dict)
    if cfg.pin_cpu:
        from .config import pin_to_core

        pin_to_core(slot)
    try:
        from .runner import run_job

        result = run_job(cfg, dataset_row, library, done,
                         progress=progress_queue.put)
        progress_queue.put({"type": "job_done", "dataset": dataset_row["dataset_id"],
                            "library": library, "counts": result["counts"]})
    except Exception as exc:
        import traceback

        progress_queue.put({
            "type": "job_error", "dataset": dataset_row["dataset_id"],
            "library": library, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        })
    finally:
        close_ledger()


# --------------------------------------------------------------------------
# Execucao
# --------------------------------------------------------------------------

def run_all(cfg: Config, jobs: list[dict], logger) -> dict:
    """Roda todos os jobs, mantendo ``cfg.n_jobs`` processos ativos."""
    done = load_done(cfg) if cfg.resume else set()
    total = pending_units(cfg, jobs, done)
    if total == 0:
        print("Nada a fazer: todas as combinacoes ja constam no ledger.")
        return {"total": 0}

    # Os jobs ja concluidos saem antes do rodizio: sem isso, uma execucao
    # retomada pode deixar o que sobrou concentrado em uma unica biblioteca
    # e colocar todos os trabalhadores nela ao mesmo tempo.
    jobs = interleave([
        job for job in jobs
        if not all(f"{job['dataset_row']['dataset_id']}|{job['library']}|{m}" in done
                   for m in job["methods"])])

    ctx = mp.get_context("spawn")
    progress_queue = ctx.Queue()
    cfg_dict = cfg.to_dict()

    n_slots = max(1, min(cfg.n_jobs, len(jobs)))
    overall = tqdm(total=total, desc="Combinacoes", unit="comb", position=0,
                   dynamic_ncols=True, smoothing=0.05,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    slot_bars = [tqdm(total=1, position=i + 1, dynamic_ncols=True, leave=False,
                      bar_format="  {desc}") for i in range(n_slots)]
    for bar in slot_bars:
        bar.set_description_str("ocioso")

    running: dict[int, dict] = {}  # slot -> {proc, job, started}
    next_job = 0
    tally = {"ok": 0, "erro": 0, "timeout": 0, "indisponivel": 0, "pulado": 0}
    job_failures = []

    def launch(slot: int) -> bool:
        nonlocal next_job
        while next_job < len(jobs):
            job = jobs[next_job]
            next_job += 1
            ds = job["dataset_row"]["dataset_id"]
            if all(f"{ds}|{job['library']}|{m}" in done for m in job["methods"]):
                continue  # job inteiro ja registrado
            proc = ctx.Process(target=_worker,
                               args=(cfg_dict, job["dataset_row"], job["library"],
                                     done, progress_queue, slot), daemon=False)
            proc.start()
            running[slot] = {"proc": proc, "job": job, "started": time.perf_counter()}
            logger.info("inicio job %s | %s (slot %d)", ds, job["library"], slot)
            return True
        return False

    last_tick = time.perf_counter()

    def passo() -> None:
        """Uma volta do escalonador: progresso, tempo limite, novos jobs."""
        nonlocal last_tick

        # Se a maquina suspendeu, o relogio salta e todos os jobs em
        # andamento pareceriam ter estourado o tempo limite de uma vez.
        # O salto e descontado do inicio de cada job.
        now = time.perf_counter()
        gap = now - last_tick
        if gap > SUSPEND_GAP:
            logger.warning("relogio saltou %.0f s (maquina suspensa); "
                           "o tempo limite dos %d jobs em andamento foi "
                           "ajustado", gap, len(running))
            for info in running.values():
                info["started"] += gap
        last_tick = now

        # 1) consome as mensagens de progresso disponiveis
        drained = 0
        while drained < 500:
            try:
                msg = progress_queue.get_nowait()
            except queue_mod.Empty:
                break
            drained += 1
            _handle_message(msg, slot_bars, running, overall, tally, logger,
                            job_failures)

        # 2) recolhe os processos encerrados e aplica o tempo limite
        for slot, info in list(running.items()):
            proc, job = info["proc"], info["job"]
            elapsed = time.perf_counter() - info["started"]
            timed_out = cfg.job_timeout and elapsed > cfg.job_timeout

            if timed_out and proc.is_alive():
                ds = job["dataset_row"]["dataset_id"]
                logger.error("tempo limite do job %s | %s apos %.0fs",
                             ds, job["library"], elapsed)
                proc.terminate()
                proc.join(10)
                ledger_entry(cfg, dataset_id=ds, library=job["library"],
                             method="__job__", status="timeout_job",
                             seconds=round(elapsed, 1),
                             error=f"job encerrado apos {elapsed:.0f}s")
                job_failures.append((ds, job["library"], "tempo limite do job"))

            if not proc.is_alive():
                proc.join()
                _finish_slot(slot, info, slot_bars, overall, tally, logger, done)
                del running[slot]
                if not launch(slot):
                    slot_bars[slot].set_description_str("concluido")

        time.sleep(0.05)

    for slot in range(n_slots):
        launch(slot)

    falhas = 0
    try:
        while running:
            try:
                passo()
                falhas = 0
            except OSError as exc:
                # A maquina ficou sem algum recurso do sistema. Espera um
                # pouco, porque os trabalhadores que terminam devolvem
                # memoria, e tenta de novo; o relogio e reancorado para que a
                # espera nao seja confundida com suspensao.
                falhas += 1
                if falhas >= MAX_FALHAS_DE_SISTEMA:
                    raise
                if falhas == 1 or falhas % 60 == 0:
                    logger.error("falha de sistema no escalonador (%dx): %s: %s",
                                 falhas, type(exc).__name__, exc)
                try:
                    time.sleep(1.0)
                except OSError:
                    pass
                last_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nInterrompido. O progresso esta no ledger; rode de novo para retomar.")
        for info in running.values():
            info["proc"].terminate()
    finally:
        for bar in slot_bars:
            bar.close()
        overall.close()

    return {"total": total, "tally": tally, "falhas_de_job": job_failures}


def _handle_message(msg, slot_bars, running, overall, tally, logger, job_failures):
    kind = msg.get("type")
    if kind in ("method_start", "method"):
        slot = _slot_of(running, msg.get("dataset"), msg.get("library"))
        if slot is not None:
            name = msg["dataset"].split("/")[-1]
            slot_bars[slot].set_description_str(
                f"[{msg['i']:>3}/{msg['n']:<3}] {msg['library']:<17} "
                f"{msg['method']:<10} {name[:42]}")
    if kind == "method":
        status = msg.get("status", "?")
        tally[status] = tally.get(status, 0) + 1
        if status != "pulado":
            overall.update(1)
        overall.set_postfix_str(
            "ok={ok} erro={erro} timeout={timeout}".format(
                ok=tally.get("ok", 0), erro=tally.get("erro", 0),
                timeout=tally.get("timeout", 0)))
        if status in ("erro", "timeout"):
            logger.warning("%s | %s | %s -> %s", msg["dataset"], msg["library"],
                           msg["method"], status)
    elif kind == "job_error":
        logger.error("job %s | %s falhou: %s\n%s", msg["dataset"], msg["library"],
                     msg["error"], msg.get("traceback", ""))
        job_failures.append((msg["dataset"], msg["library"], msg["error"]))


def _slot_of(running, dataset, library):
    for slot, info in running.items():
        job = info["job"]
        if job["dataset_row"]["dataset_id"] == dataset and job["library"] == library:
            return slot
    return None


def _finish_slot(slot, info, slot_bars, overall, tally, logger, done):
    job = info["job"]
    ds = job["dataset_row"]["dataset_id"]
    elapsed = time.perf_counter() - info["started"]
    code = info["proc"].exitcode
    logger.info("fim job %s | %s em %.1fs (exit=%s)", ds, job["library"], elapsed, code)
    if code not in (0, None):
        # O processo morreu antes de registrar os metodos restantes; eles
        # continuam pendentes no ledger e serao refeitos na proxima execucao.
        logger.error("job %s | %s terminou com codigo %s", ds, job["library"], code)
    slot_bars[slot].set_description_str("ocioso")
