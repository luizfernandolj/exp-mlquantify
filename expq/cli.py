"""Linha de comando dos experimentos.

Os parametros que importam para dividir o trabalho entre maquinas sao
``--range``, ``--shard``, ``--datasets`` e ``--libraries``. Cada maquina usa
o mesmo ``--run-id`` e um ``--node`` diferente; os resultados ficam em
pastas separadas e sao reunidos no final por ``aggregate_results.py``.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import ALL_GROUPS, ALL_LIBRARIES, DEFAULT_GROUPS, Config, limit_threads

EPILOG = """\
exemplos:

  # catalogo e selecao dos conjuntos (uma vez, antes de tudo)
  python build_catalog.py

  # execucao completa nesta maquina, usando todos os nucleos
  python run_experiments.py --run-id final

  # tres maquinas, dividindo por faixa de indice do catalogo
  python run_experiments.py --run-id final --node pc1 --range 0:45
  python run_experiments.py --run-id final --node pc2 --range 45:90
  python run_experiments.py --run-id final --node pc3 --range 90:999

  # tres maquinas, dividindo em rodizio (fatias de tamanho parecido)
  python run_experiments.py --run-id final --node pc1 --shard 1/3
  python run_experiments.py --run-id final --node pc2 --shard 2/3
  python run_experiments.py --run-id final --node pc3 --shard 3/3

  # andamento, lido apenas do ledger
  python run_experiments.py --run-id final --status

  # metodos deixados para depois (ELM e neurais), quando o branch estiver pronto
  python run_experiments.py --run-id elm --groups elm --libraries mlquantify,quapy
  python run_experiments.py --run-id neural --groups neural --libraries mlquantify,quapy
"""


def _csv_list(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


def _int_list(text: str) -> tuple[int, ...]:
    return tuple(int(s) for s in text.replace(" ", "").split(",") if s)


def _range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition(":")
    return (int(lo or 0), int(hi) if hi else 10**9)


def _shard(text: str) -> tuple[int, int]:
    i, _, n = text.partition("/")
    return (int(i), int(n))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiments.py",
        description="Comparacao entre bibliotecas de quantificacao (Secao 4.3.4).",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("identificacao")
    g.add_argument("--run-id", default="run01",
                   help="nome da execucao; o mesmo em todas as maquinas")
    g.add_argument("--node", default=None,
                   help="nome desta maquina (padrao: hostname)")
    g.add_argument("--out-dir", default=None, metavar="PASTA",
                   help="grava resultados, logs e cache nesta pasta em vez de "
                        "dentro do repositorio; use quando o repositorio estiver "
                        "em pasta sincronizada (OneDrive, Dropbox)")

    g = p.add_argument_group("divisao do trabalho")
    g.add_argument("--range", type=_range, default=None, metavar="INI:FIM",
                   help="faixa de indices do catalogo de selecionados, ex. 0:45")
    g.add_argument("--shard", type=_shard, default=None, metavar="I/N",
                   help="fatia em rodizio, ex. 2/3")
    g.add_argument("--datasets", type=_csv_list, default=(),
                   help="roda apenas os conjuntos cujo caminho contem estes textos")
    g.add_argument("--skip-datasets", type=_csv_list, default=(),
                   help="ignora os conjuntos cujo caminho contem estes textos")
    g.add_argument("--types", type=_csv_list, default=("binario", "multiclasse"),
                   help="binario, multiclasse ou ambos")

    g = p.add_argument_group("metodos e bibliotecas")
    g.add_argument("--libraries", type=_csv_list, default=ALL_LIBRARIES,
                   help=f"padrao: {','.join(ALL_LIBRARIES)}")
    g.add_argument("--groups", type=_csv_list, default=DEFAULT_GROUPS,
                   help=f"grupos de metodos entre {','.join(ALL_GROUPS)} "
                        "(elm e neural ficam para depois)")
    g.add_argument("--methods", type=_csv_list, default=(),
                   help="roda apenas estes metodos canonicos")
    g.add_argument("--skip-methods", type=_csv_list, default=(),
                   help="ignora estes metodos")

    g = p.add_argument_group("protocolo")
    g.add_argument("--sample-sizes", type=_int_list, default=(10, 50, 100, 500),
                   help="tamanhos de amostra m (padrao: 10,50,100,500)")
    g.add_argument("--n-prevalences", type=int, default=100,
                   help="vetores de prevalencia por tamanho (padrao: 100)")
    g.add_argument("--test-size", type=float, default=0.5)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max-rows", type=int, default=None,
                   help="subamostra estratificada de conjuntos muito grandes")
    g.add_argument("--scaled-classifier", action="store_true",
                   help="usa Pipeline(StandardScaler, LogisticRegression)")

    g = p.add_argument_group("medidas de custo")
    g.add_argument("--cost-repeats", type=int, default=3,
                   help="repeticoes de tempo, resumidas pela mediana (padrao: 3)")
    g.add_argument("--no-memory", action="store_true",
                   help="nao mede pico de memoria (execucao mais rapida)")
    g.add_argument("--no-per-sample", action="store_true",
                   help="grava apenas o resumo, sem o detalhe por amostra")

    g = p.add_argument_group("execucao")
    g.add_argument("-j", "--jobs", type=int, default=0,
                   help="processos simultaneos (padrao: numero de nucleos)")
    g.add_argument("--job-timeout", type=float, default=7200.0,
                   help="limite por job, em segundos (0 desliga)")
    g.add_argument("--method-timeout", type=float, default=1800.0,
                   help="limite por metodo, em segundos (0 desliga)")
    g.add_argument("--fresh", action="store_true",
                   help="ignora o ledger e refaz tudo")
    g.add_argument("--dry-run", action="store_true",
                   help="mostra o plano e sai")
    g.add_argument("--status", action="store_true",
                   help="mostra o andamento lido do ledger e sai")
    g.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args) -> Config:
    cfg = Config(
        run_id=args.run_id,
        sample_sizes=tuple(args.sample_sizes),
        n_prevalences=args.n_prevalences,
        test_size=args.test_size,
        random_state=args.seed,
        max_rows=args.max_rows,
        classifier_pipeline="scaled" if args.scaled_classifier else "plain",
        cost_repeats=args.cost_repeats,
        measure_memory=not args.no_memory,
        save_per_sample=not args.no_per_sample,
        libraries=tuple(args.libraries),
        groups=tuple(args.groups),
        methods=tuple(args.methods),
        skip_methods=tuple(args.skip_methods),
        dataset_range=args.range,
        shard=args.shard,
        only_datasets=tuple(args.datasets),
        skip_datasets=tuple(args.skip_datasets),
        dataset_types=tuple(args.types),
        n_jobs=args.jobs,
        job_timeout=args.job_timeout,
        method_timeout=args.method_timeout,
        resume=not args.fresh,
    )
    if args.node:
        cfg.node = args.node
    if args.out_dir:
        cfg.set_out_dir(args.out_dir)
    return cfg


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    from .catalog import load_selected
    from .ledger import setup_logging, silence_warnings, status_report
    from .parallel import build_jobs, pending_units, run_all

    if args.status:
        print(json.dumps(status_report(cfg), indent=2, ensure_ascii=False))
        return 0

    # Uma thread por processo: o paralelismo fica entre jobs e as medidas de
    # tempo nao dependem do numero de nucleos que o BLAS resolve usar.
    limit_threads(cfg.threads_per_worker)
    silence_warnings()

    # O escalonador mede o tempo limite de cada job: ele tambem nao pode ser
    # rebaixado, e enquanto vive a maquina nao dorme.
    from .config import keep_full_speed

    keep_full_speed(stay_awake=True)

    selected = load_selected(cfg)
    if selected.empty:
        print("Nenhum conjunto de dados selecionado com esses filtros.")
        return 1

    jobs = build_jobs(cfg, selected)
    from .ledger import load_done

    done = load_done(cfg) if cfg.resume else set()
    pending = pending_units(cfg, jobs, done)

    print(f"execucao   : {cfg.run_id}   maquina: {cfg.node}")
    print(f"conjuntos  : {len(selected)} "
          f"({', '.join(f'{k}={v}' for k, v in selected['tipo_dataset'].value_counts().items())})")
    print(f"bibliotecas: {', '.join(cfg.libraries)}")
    print(f"grupos     : {', '.join(cfg.groups)}")
    print(f"protocolo  : {cfg.protocol} m={list(cfg.sample_sizes)} "
          f"x {cfg.n_prevalences} prevalencias = "
          f"{len(cfg.sample_sizes) * cfg.n_prevalences} amostras por conjunto")
    print(f"jobs       : {len(jobs)} (conjunto x biblioteca) em {cfg.n_jobs} processos")
    print(f"combinacoes: {pending} pendentes, {len(done)} ja no ledger")
    print(f"resultados : {cfg.parts_dir}")
    print(f"ledger     : {cfg.ledger_dir}")
    print(f"log        : {cfg.log_path}\n")

    if args.dry_run:
        for job in jobs[:20]:
            print(f"  {job['dataset_row']['dataset_id']:<60} {job['library']:<18} "
                  f"{len(job['methods'])} metodos")
        if len(jobs) > 20:
            print(f"  ... e mais {len(jobs) - 20} jobs")
        return 0

    # So cria as pastas da execucao quando ela vai mesmo acontecer.
    cfg.make_dirs()
    logger = setup_logging(cfg, verbose=args.verbose)
    cfg.save()
    logger.info("inicio da execucao %s em %s: %d jobs, %d combinacoes pendentes",
                cfg.run_id, cfg.node, len(jobs), pending)

    result = run_all(cfg, jobs, logger)

    print("\nresumo:", json.dumps(result.get("tally", {}), ensure_ascii=False))
    for ds, lib, err in result.get("falhas_de_job", [])[:10]:
        print(f"  job com falha: {ds} | {lib} | {err[:80]}")
    print(f"\nreuna os resultados com:\n  python aggregate_results.py --run-id {cfg.run_id}")
    logger.info("fim da execucao %s em %s", cfg.run_id, cfg.node)
    return 0


if __name__ == "__main__":
    sys.exit(main())
