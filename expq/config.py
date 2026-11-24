"""Configuracao central dos experimentos.

Todos os parametros do protocolo da Tabela 12 ficam aqui. A linha de comando
sobrescreve o que for necessario, e o objeto resultante e serializado junto
com os resultados para que cada execucao seja reproduzivel.
"""

from __future__ import annotations

import json
import os
import platform
import socket
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Bibliotecas comparadas. A primeira e a biblioteca proposta.
ALL_LIBRARIES = ("mlquantify", "quapy", "quantificationlib", "qunfold")

#: Grupos de metodos. ``core`` e o conjunto executado por padrao; ``elm`` e
#: ``neural`` dependem de codigo que ainda nao esta na versao publicada da
#: mlquantify (outro branch) e ficam para uma execucao posterior.
ALL_GROUPS = ("core", "meta", "elm", "neural", "text", "sens")
DEFAULT_GROUPS = ("core",)


@dataclass
class Config:
    # --- caminhos ---
    root: Path = ROOT
    datasets_dir: Path = ROOT / "datasets"
    catalog_dir: Path = ROOT / "descricao"
    results_dir: Path = ROOT / "results"
    logs_dir: Path = ROOT / "logs"
    cache_dir: Path = ROOT / ".cache"

    # --- identificacao da execucao ---
    run_id: str = "run01"
    node: str = field(default_factory=lambda: socket.gethostname())

    # --- protocolo (Tabela 12) ---
    test_size: float = 0.5
    random_state: int = 42
    sample_sizes: tuple[int, ...] = (10, 50, 100, 500)
    n_prevalences: int = 100
    protocol: str = "UPP"

    # --- classificador base ---
    classifier: str = "logistic_regression"
    classifier_max_iter: int = 1000
    #: ``plain`` = LogisticRegression pura (como na monografia);
    #: ``scaled`` = Pipeline(StandardScaler, LogisticRegression).
    classifier_pipeline: str = "plain"

    # --- medidas de custo ---
    cost_repeats: int = 3
    measure_memory: bool = True

    # --- selecao de metodos e bibliotecas ---
    libraries: tuple[str, ...] = ALL_LIBRARIES
    groups: tuple[str, ...] = DEFAULT_GROUPS
    methods: tuple[str, ...] = ()  # vazio = todos os metodos dos grupos
    skip_methods: tuple[str, ...] = ()

    # --- selecao de conjuntos de dados ---
    dataset_range: tuple[int, int] | None = None  # [inicio, fim) no catalogo
    shard: tuple[int, int] | None = None  # (indice, total), 1-based
    only_datasets: tuple[str, ...] = ()
    skip_datasets: tuple[str, ...] = ()
    dataset_types: tuple[str, ...] = ("binario", "multiclasse")
    max_rows: int | None = None  # subamostra conjuntos gigantes (None = sem limite)

    # --- criterios de selecao (Secao 4.3.3) ---
    # piso absoluto da classe rara; o criterio relativo do capitulo de
    # resultados (minoria com 10% das linhas) esta em recorte_analise.py
    min_minority: int = 30
    duplicate_key: str = "content"  # "content" | "shape_dist"

    # --- execucao ---
    n_jobs: int = 0  # 0 = os.cpu_count()
    threads_per_worker: int = 1
    job_timeout: float = 7200.0  # segundos por job (dataset x biblioteca)
    method_timeout: float = 1800.0  # segundos por metodo
    resume: bool = True
    save_per_sample: bool = True
    #: Prende cada trabalhador a um nucleo. E o unico controle que vale para
    #: o JAX, que ignora as variaveis de ambiente de threads.
    pin_cpu: bool = True

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        for name in ("root", "datasets_dir", "catalog_dir", "results_dir", "logs_dir", "cache_dir"):
            setattr(self, name, Path(getattr(self, name)))
        if self.n_jobs <= 0:
            self.n_jobs = os.cpu_count() or 1

    # --- diretorios derivados ---
    @property
    def run_dir(self) -> Path:
        return self.results_dir / self.run_id

    @property
    def parts_dir(self) -> Path:
        return self.run_dir / "parts" / self.node

    @property
    def ledger_dir(self) -> Path:
        """Um arquivo por processo, para que escritas simultaneas nao se misturem."""
        return self.run_dir / "ledger" / self.node

    @property
    def log_path(self) -> Path:
        return self.logs_dir / f"{self.run_id}__{self.node}.log"

    @property
    def selected_catalog(self) -> Path:
        return self.catalog_dir / "datasets_selecionados.csv"

    @property
    def full_catalog(self) -> Path:
        return self.catalog_dir / "catalogo_experimentos.csv"

    def set_out_dir(self, root) -> None:
        """Move resultados, logs e cache para fora do repositorio.

        Util quando o repositorio fica dentro de uma pasta sincronizada
        (OneDrive, Dropbox): o sincronizador mantem os arquivos abertos
        enquanto envia, o que trava a gravacao dos parciais e deixa a
        execucao mais lenta.
        """
        root = Path(root)
        self.results_dir = root / "results"
        self.logs_dir = root / "logs"
        self.cache_dir = root / "cache"

    def make_dirs(self) -> None:
        for d in (self.run_dir, self.parts_dir, self.ledger_dir, self.logs_dir,
                  self.cache_dir / "prepared", self.cache_dir / "samples", self.catalog_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- serializacao ---
    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        d["environment"] = environment_info()
        return d

    def save(self) -> Path:
        path = self.run_dir / f"config__{self.node}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def environment_info() -> dict:
    """Versoes e plataforma, registradas junto aos resultados."""
    import importlib.metadata as md

    versions = {}
    for pkg in ("mlquantify", "quapy", "quantificationlib", "qunfold",
                "scikit-learn", "numpy", "scipy", "pandas"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "nao instalado"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "hostname": socket.gethostname(),
        "versions": versions,
    }


def limit_threads(n: int = 1) -> None:
    """Fixa o numero de threads das bibliotecas numericas.

    Necessario para que as medidas de tempo nao dependam do paralelismo
    interno do BLAS: o paralelismo do experimento acontece entre processos.

    O JAX, usado pela qunfold, **ignora** ``OMP_NUM_THREADS`` e abre seu
    proprio pool, de cerca de 70 threads por processo. Com um trabalhador
    por nucleo isso gera centenas de threads disputando a maquina, o que
    multiplica por dezenas o tempo medido da qunfold e contamina tambem o
    tempo das outras bibliotecas. O XLA precisa ser limitado pelas suas
    proprias variaveis, antes de ``import jax``.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)

    os.environ["XLA_FLAGS"] = (
        f"--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads={n}")
    os.environ["JAX_PLATFORMS"] = "cpu"  # evita sondar GPU a cada processo


def keep_full_speed(stay_awake: bool = False) -> bool:
    """Pede ao Windows que nao rebaixe este processo, e opcionalmente que nao durma.

    Com o plano "Equilibrado", o Windows 11 aplica o EcoQoS a processos em
    segundo plano quando a sessao fica ociosa: o clock cai e, nos
    processadores hibridos, o trabalho vai para os nucleos de eficiencia. Na
    custo-final os controles CC e PCC, que nao mudaram, ficaram de 1,4 a 2
    vezes mais lentos no passo noturno dos gigantes, o que torna as medidas
    de custo incomparaveis com as de dia. ``SetProcessInformation`` desliga
    o EcoQoS so para este processo; ``SetThreadExecutionState`` impede que o
    computador durma enquanto o processo vive, como faz um reprodutor de
    video. Nenhum dos dois muda configuracao do sistema, e ambos acabam com
    o processo.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class _ThrottlingState(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        process_power_throttling = 4          # PROCESS_INFORMATION_CLASS
        execution_speed = 0x1                 # PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        state = _ThrottlingState(1, execution_speed, 0)  # controlado, desligado
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
        ok = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), process_power_throttling,
            ctypes.byref(state), ctypes.sizeof(state))
        if stay_awake:
            es_continuous, es_system_required = 0x80000000, 0x00000001
            kernel32.SetThreadExecutionState(es_continuous | es_system_required)
        return bool(ok)
    except Exception:
        return False  # versao antiga do Windows; segue como antes


def pin_to_core(core: int) -> int | None:
    """Prende o processo atual a um nucleo logico.

    As variaveis de ambiente nao seguram o pool de threads do JAX, que abre
    cerca de 70 threads por processo. A afinidade de CPU segura, e vale para
    qualquer biblioteca, entao e ela que garante uma thread util por
    trabalhador e medidas de tempo comparaveis entre as bibliotecas.
    """
    try:
        import psutil

        total = os.cpu_count() or 1
        chosen = core % total
        psutil.Process().cpu_affinity([chosen])
        return chosen
    except Exception:
        return None  # nem todo sistema permite; segue sem fixar
