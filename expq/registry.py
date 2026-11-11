"""Catalogo de metodos por biblioteca (Tabela 10 da monografia).

Cada entrada liga um *nome canonico* de metodo a classe que o implementa em
uma das bibliotecas comparadas. Duas construcoes sao registradas:

``build_naive``
    a forma direta, como aparece no fluxo comum de um script: instanciar o
    metodo com o classificador base e chamar ``fit``. E a construcao usada
    para medir a proporcao de combinacoes que uma biblioteca conclui *sem
    configuracao manual* (questao Q3).

``build_adapted``
    a forma documentada pela biblioteca, com os ajustes que ela exige
    (decomposicao binaria, estimador validado cruzadamente, parametros sem
    valor padrao). E a construcao usada para medir erro e custo (Q1 e Q2),
    para que nenhuma biblioteca seja penalizada por um uso incorreto.

``manual_configs`` conta quantas instrucoes existem em ``build_adapted``
apenas para adaptar o metodo ao formato da biblioteca, e nao ao fluxo comum
de leitura, divisao, treinamento, geracao de amostras e estimacao.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

from .config import ALL_LIBRARIES

CV_FOLDS = 5
DEFAULT_BINS = 8


# ---------------------------------------------------------------------------
# Protocolo comum de geracao dos escores de treinamento
#
# As quatro bibliotecas resolvem a mesma etapa -- obter predicoes fora da
# amostra para treinar a funcao de agregacao -- com parametros diferentes e,
# em duas delas, sem semente. Medido nas versoes instaladas:
#
#   biblioteca         dobras  estratifica  embaralha  semente
#   mlquantify              5      sim         nao       nao
#   quapy                   5      sim         nao       nao
#   quantificationlib       5      sim         sim       sim
#   qunfold                 5      sim         sim       sim
#
# O numero de dobras ja coincidia (na mlquantify ``cv=None`` cai em cinco,
# na quapy ``val_split=5`` e o padrao). O que nao coincidia era o
# embaralhamento -- e portanto a dependencia da ordem das linhas do arquivo
# -- e a semente. Daqui em diante toda construcao documentada
# (``build_adapted``) recebe os mesmos valores, nos nomes de cada
# biblioteca:
#
#   cv / val_split      = CV_FOLDS            cinco dobras
#   stratified          = True                dobras estratificadas por classe
#   shuffle             = True                embaralha antes de dividir
#   random_state / seed = ctx.random_state     a mesma semente em tudo
#
# Com isso a mlquantify, a quantificationlib e a qunfold passam a montar o
# mesmo ``StratifiedKFold(n_splits=5, shuffle=True, random_state=42)``, ou
# seja, dobras identicas. A quapy e a excecao: ela nao expoe o
# embaralhamento, porque ``val_split`` inteiro cai em
# ``cross_val_predict(cv=5)``, que usa ``StratifiedKFold(shuffle=False)``.
# Por isso a particao de treinamento ja chega embaralhada do ``split`` em
# expq/protocol.py -- sobre linhas em ordem aleatoria, dobras sem
# embaralhamento sao dobras sorteadas, e nenhuma biblioteca depende mais da
# ordem do arquivo.
#
# Isto e protocolo do experimento, nao configuracao exigida pela
# biblioteca: vale igual para as quatro e por isso NAO entra em
# ``manual_configs``, que segue medindo o que cada uma cobra para funcionar
# (Q3). A forma direta (``build_naive``), que e a sonda de Q3, continua sem
# nenhum destes valores: ela existe justamente para medir o padrao da
# biblioteca.
# ---------------------------------------------------------------------------


def _parametros(cls) -> dict:
    """Parametros do construtor de ``cls``, ou vazio se nao der para olhar."""
    import inspect

    try:
        return dict(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        return {}


def _protocolo(cls, ctx: Ctx) -> dict:
    """Valores do protocolo comum, nos nomes que ``cls`` entende.

    Filtrar pela assinatura evita uma lista de excecoes por metodo: o CC nao
    gera escores fora da amostra, a HDx nao tem classificador, a EMQ da
    quapy nao tem ``stratified``. Cada classe recebe o que sabe usar.

    O numero de dobras e o unico valor com ressalva. Na mlquantify
    ``cv=None`` significa *use cinco*, e escrever cinco nao muda nada. Na
    quapy ``val_split=None`` significa *sem predicoes fora da amostra*: a
    PCC e a EMQ treinam na particao inteira de proposito, e sobrescrever
    isso trocaria o metodo em vez de alinha-lo. Por isso ``val_split`` so e
    escrito onde a classe ja o traz como inteiro.
    """
    ps = _parametros(cls)
    valores = {"stratified": True, "shuffle": True,
               "random_state": ctx.random_state, "seed": ctx.random_state}
    if "cv" in ps:
        valores["cv"] = CV_FOLDS
    if "val_split" in ps and ps["val_split"].default is not None:
        valores["val_split"] = CV_FOLDS
    return {k: v for k, v in valores.items() if k in ps}


def descricao_protocolo(random_state: int) -> str:
    """Rotulo do protocolo de escores, gravado junto de cada medida."""
    return "cv{}-estratificado-embaralhado-semente{}".format(CV_FOLDS, random_state)


@dataclass
class Ctx:
    """Contexto entregue a cada construtor de metodo."""

    make_clf: Callable[[], object]
    n_classes: int
    random_state: int = 42

    @property
    def is_multiclass(self) -> bool:
        return self.n_classes > 2


@dataclass(frozen=True)
class MethodSpec:
    canonical: str
    family: str
    library: str
    class_name: str
    build_adapted: Callable[[Ctx], object]
    build_naive: Callable[[Ctx], object] | None = None
    group: str = "core"
    binary_only: bool = False
    needs_classifier: bool = True
    manual_configs: int = 0
    manual_configs_multiclass: int = 0
    aliases: tuple[str, ...] = ()
    available: bool = True
    unavailable_reason: str = ""
    notes: str = ""

    def naive(self, ctx: Ctx):
        return (self.build_naive or self.build_adapted)(ctx)

    def adapted(self, ctx: Ctx):
        return self.build_adapted(ctx)

    def naive_differs(self, ctx: Ctx) -> bool:
        """Se a forma direta e a documentada coincidem, nao ha o que sondar."""
        if self.build_naive is None:
            return False
        return bool(self.manual_configs) or (self.binary_only and ctx.is_multiclass)

    def manual_count(self, ctx: Ctx) -> int:
        extra = self.manual_configs_multiclass if ctx.is_multiclass else 0
        return self.manual_configs + extra


_REGISTRY: dict[str, list[MethodSpec]] = {lib: [] for lib in ALL_LIBRARIES}


def register(spec: MethodSpec) -> MethodSpec:
    _REGISTRY.setdefault(spec.library, []).append(spec)
    return spec


# ===========================================================================
# mlquantify - biblioteca proposta
#
# A decomposicao multiclasse e aplicada internamente quando o metodo e
# binario e o problema nao e, sem configuracao do usuario (Secao 4.3.2), de
# modo que forma direta e forma documentada coincidem em todos os metodos.
# ===========================================================================


def _mlq(canonical, family, module, cls_name, *, binary_only=False,
         needs_classifier=True, group="core", aliases=(), notes="", kwargs=None):
    kwargs = kwargs or {}

    def build(ctx: Ctx):
        import importlib

        cls = getattr(importlib.import_module("mlquantify." + module), cls_name)
        prot = dict(kwargs)
        prot.update(_protocolo(cls, ctx))
        return cls(ctx.make_clf(), **prot) if needs_classifier else cls(**prot)

    return register(MethodSpec(
        canonical=canonical, family=family, library="mlquantify",
        class_name="mlquantify.{}.{}".format(module, cls_name), build_adapted=build,
        group=group, binary_only=binary_only, needs_classifier=needs_classifier,
        manual_configs=0, aliases=aliases, notes=notes,
    ))


_mlq("CC", "contagem", "counting", "CC")
_mlq("PCC", "contagem", "counting", "PCC")
_mlq("ACC", "contagem ajustada", "counting", "ACC", binary_only=True,
     notes="decomposicao um-contra-todos aplicada internamente")
_mlq("GACC", "contagem ajustada", "counting", "GACC")
# O apelido PACC <-> GPACC e declarado SO deste lado, e isso e deliberado.
#
# O expand_aliases acrescenta um apelido apenas quando a biblioteca nao tem
# metodo proprio com aquele nome. Declarando dos dois lados -- como estava --,
# a mlquantify ganhava PACC e as outras ganhavam GPACC, e o mesmo par se
# materializava sob os dois nomes: 564 casos contados duas vezes por
# biblioteca, 1128 no total, inflando o total de Q1 em 7,9% contra a
# quantificationlib e 4,6% contra a quapy. Conferido por hash das medicoes.
#
# Entre os dois lados, este. Tirando o apelido daqui e deixando o das outras,
# a duplicacao tambem acabava, mas o bloco de teste do PACC passava a comparar
# quantificationlib contra quapy, sem a mlquantify -- uma comparacao que nao e
# a pergunta de Q1 e que ainda assim entrava na contagem de metodos com
# diferenca. Declarado deste lado, os 27 blocos seguem todos ancorados na
# proposta: o PACC compara mlquantify, quantificationlib e quapy, e o GPACC
# compara mlquantify e qunfold.
#
# O ACC -> GACC das outras duas continua, e nao duplica: a mlquantify tem ACC
# e GACC como metodos distintos, entao os apelidos produzem duas comparacoes
# diferentes contra o mesmo metodo da outra biblioteca, e nao duas copias da
# mesma.
_mlq("GPACC", "contagem ajustada", "counting", "GPACC", aliases=("PACC",))
_mlq("FM", "contagem ajustada", "counting", "FM")
_mlq("TAC", "selecao de limiar", "counting", "TAC", binary_only=True)
_mlq("T50", "selecao de limiar", "counting", "T50", binary_only=True)
_mlq("X", "selecao de limiar", "counting", "TX", binary_only=True)
_mlq("MAX", "selecao de limiar", "counting", "TMAX", binary_only=True)
_mlq("MS", "selecao de limiar", "counting", "MS", binary_only=True)
_mlq("MS2", "selecao de limiar", "counting", "MS2", binary_only=True)
# Sem classificador: o MLPE da mlquantify deixou de ser agregativo, como o
# da quapy sempre foi. O `fit` le as proporcoes de `y` e nao treina nada, e a
# classe nao aceita mais estimador -- passar um levanta TypeError.
_mlq("MLPE", "verossimilhanca", "likelihood", "MLPE", needs_classifier=False,
     notes="linha de base trivial de Hassan, Maletzke e Batista (2021); "
           "nao agregativo, nao recebe classificador")
# MANUAL: max_iter=1000 para casar com a quapy (MAX_ITER=1000) e com a
# quantificationlib (max_iter=1000); a tolerancia ja e 1e-4 nas tres. O
# padrao da biblioteca, 100, e o menor dos quatro e esta mordendo: no
# conjunto banana a estimativa muda 0,144 entre 100 e 1000 iteracoes, ou
# seja, o EM estava sendo truncado antes de convergir.
_mlq("EMQ", "verossimilhanca", "likelihood", "EMQ", kwargs={"max_iter": 1000},
     notes="max_iter alinhado em 1000 com a quapy e a quantificationlib; o "
           "padrao da biblioteca e 100 e trunca o EM antes da convergencia")
_mlq("CDE", "verossimilhanca", "likelihood", "CDE")
_mlq("HDy", "casamento de distribuicoes", "matching", "HDy", binary_only=True,
     notes="varre [10, 20, ..., 110] e toma a mediana das estimativas, que e "
           "a definicao do metodo e o mesmo que a quapy faz; nao mexido")
# O mesmo HDy com os 8 intervalos fixos da quantificationlib, para o par com
# ela ficar alinhado. Fica a parte do ``HDy`` varrido, que continua sendo o
# par da quapy: a quapy fixa a varredura [10..110] no codigo, sem parametro,
# entao nao da para alinhar os tres num valor so. Dois pares, cada um
# internamente alinhado, e melhor do que um par torto e outro certo.
#
# O caminho inverso -- fazer a qlib varrer -- foi medido na sens-hdy e nao
# serve: o ``solve_hd`` dela indexa ``problem.solution.primal_vars`` sem
# conferir o status do cvxpy, e com muitos intervalos sobre amostra pequena
# o problema fica degenerado e o acesso estoura. Deu 4 erros e 7 timeouts
# contra 1 timeout do HDy de 8 intervalos, e as perdas caem justamente nos
# conjuntos de muitas classes.
_mlq("HDy8", "casamento de distribuicoes", "matching", "HDy", binary_only=True,
     kwargs={"bins_size": DEFAULT_BINS},
     notes="o HDy com os 8 intervalos fixos da quantificationlib, para o par "
           "com ela medir implementacao e nao parametrizacao")
_mlq("GHDy", "casamento de distribuicoes", "matching", "GHDy")
_mlq("DyS", "casamento de distribuicoes", "matching", "DyS", binary_only=True,
     kwargs={"bins_size": DEFAULT_BINS},
     notes="intervalos fixos em DEFAULT_BINS para casar com a quapy; o "
           "padrao da biblioteca varre [2, 4, ..., 20, 30]")
_mlq("SORD", "casamento de distribuicoes", "matching", "SORD", binary_only=True)
_mlq("SMM", "casamento de distribuicoes", "matching", "SMM", binary_only=True)
_mlq("KDEyML", "casamento de distribuicoes", "matching", "KDEyML")
_mlq("KDEyHD", "casamento de distribuicoes", "matching", "KDEyHD")
_mlq("KDEyCS", "casamento de distribuicoes", "matching", "KDEyCS")
_mlq("GKDEyML", "casamento de distribuicoes", "matching", "GKDEyML")
_mlq("EDy", "casamento de distribuicoes", "matching", "EDy")
_mlq("MMD", "casamento de distribuicoes", "matching", "MMD_RKHS", needs_classifier=False)
_mlq("HDx", "nao agregativo", "matching", "HDx", needs_classifier=False,
     kwargs={"bins_size": [DEFAULT_BINS]},  # valida array-like, nao int
     notes="intervalos fixos em DEFAULT_BINS, que e o valor das outras "
           "tres; o padrao da biblioteca varre [2, 4, ..., 20, 30]")
_mlq("GHDx", "nao agregativo", "matching", "GHDx", needs_classifier=False)
_mlq("EDx", "nao agregativo", "matching", "EDx", needs_classifier=False)
_mlq("PWK", "vizinhos e arvores", "neighbors", "PWK", needs_classifier=False)


class _PWKSklearn:
    """Invólucro do ``PWKCLF`` com a interface que o scikit-learn exige.

    MANUAL: o ``PWKCLF`` da mlquantify nao implementa ``get_params``, entao o
    ``clone`` do scikit-learn o recusa -- e com isso ele nao pode ser passado
    a nenhum quantificador da propria biblioteca que valide cruzado, que e o
    caso do ACC. O invólucro so repassa ``fit`` e ``predict`` e declara os
    parametros; nao muda o metodo.
    """

    def __init__(self, alpha=1, n_neighbors=10):
        self.alpha = alpha
        self.n_neighbors = n_neighbors

    def get_params(self, deep=True):
        return {"alpha": self.alpha, "n_neighbors": self.n_neighbors}

    def set_params(self, **params):
        for nome, valor in params.items():
            setattr(self, nome, valor)
        return self

    def fit(self, X, y):
        import numpy as np
        from mlquantify.neighbors._classification import PWKCLF

        self._clf = PWKCLF(alpha=self.alpha, n_neighbors=self.n_neighbors)
        self._clf.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self._clf.predict(X)


def _build_pwk_ajustado(ctx: Ctx):
    """O PWK com ajuste de contagem, que e o par do PWKQuantifier da qlib.

    O ``PWK`` da mlquantify e ``class PWK(CC)`` e nao aceita estimador
    externo -- o kNN ponderado e intrinseco ao metodo. Para obter o
    equivalente ajustado monta-se o ``TAC`` sobre o ``PWKCLF``, que e o mesmo
    classificador que o ``PWK`` usa por dentro. Os padroes batem com os da
    qlib: ``n_neighbors=10`` nos dois, e ``alpha=1``, que e o valor do estudo
    original.
    """
    from mlquantify.counting import ACC

    # ACC e nao TAC: o PWKCLF so tem ``predict``, e o TAC e de predicao suave.
    # O ACC e ``CrispPredictionMixin``, que e o que a qlib faz -- a matriz de
    # confusao dela sai de ``knn_estimator_.predict(X)``, predicao dura. (ACC
    # e TAC sao o mesmo estimador: medidos, batem bit a bit nos 564 casos.)
    base = _PWKSklearn(alpha=1, n_neighbors=10)
    return ACC(estimator=base, **_protocolo(ACC, ctx))


register(MethodSpec(
    canonical="PWKAC", family="vizinhos e arvores", library="mlquantify",
    class_name="mlquantify.counting.TAC(mlquantify.neighbors.PWKCLF)",
    build_adapted=_build_pwk_ajustado, binary_only=True, manual_configs=1,
    notes="o PWK ajustado: TAC sobre o PWKCLF, que e o par do PWKQuantifier "
          "da quantificationlib. O PWK sem ajuste continua registrado a parte"))


def _mlq_ensemble(ctx: Ctx):
    from mlquantify.counting import CC
    from mlquantify.meta import EnsembleQ

    return EnsembleQ(CC(ctx.make_clf()))


def _mlq_bootstrap(ctx: Ctx):
    from mlquantify.counting import CC
    from mlquantify.meta import AggregativeBootstrap

    return AggregativeBootstrap(CC(ctx.make_clf()))


register(MethodSpec("Ensembles", "metaquantificacao", "mlquantify",
                    "mlquantify.meta.EnsembleQ", _mlq_ensemble, group="meta",
                    manual_configs=1, notes="exige um quantificador base (CC)"))
register(MethodSpec("Bootstrap", "metaquantificacao", "mlquantify",
                    "mlquantify.meta.AggregativeBootstrap", _mlq_bootstrap,
                    group="meta", manual_configs=1,
                    notes="exige um quantificador base (CC)"))


def _unavailable(ctx: Ctx):
    raise NotImplementedError("metodo ainda nao disponivel nesta versao da biblioteca")


for _name in ("SVM(Q)", "SVM(KLD)", "SVM(NKLD)", "SVM(AE)", "SVM(RAE)"):
    register(MethodSpec(_name, "minimizacao explicita da perda (ELM)", "mlquantify",
                        "-", _unavailable, group="elm", available=False,
                        unavailable_reason="nao publicado na mlquantify 0.5.1 (outro branch)"))
for _name in ("QuaNet", "HistNetQ", "GMNet"):
    register(MethodSpec(_name, "neurais", "mlquantify", "-", _unavailable,
                        group="neural", available=False,
                        unavailable_reason="nao publicado na mlquantify 0.5.1 (outro branch)"))


# ===========================================================================
# QuaPy
#
# Metodos binarios interrompem a execucao em problemas multiclasse e exigem
# o encapsulamento manual em ``newOneVsAll`` (Secao 4.2).
# ===========================================================================


def _quapy(canonical, family, module, cls_name, *, binary_only=False,
           needs_classifier=True, group="core", aliases=(), notes="", kwargs=None):
    kwargs = kwargs or {}

    def _make(ctx: Ctx, protocolo: bool):
        import importlib

        cls = getattr(importlib.import_module("quapy.method." + module), cls_name)
        extra = dict(kwargs)
        if protocolo:
            extra.update(_protocolo(cls, ctx))
        return cls(ctx.make_clf(), **extra) if needs_classifier else cls(**extra)

    def build_naive(ctx: Ctx):
        return _make(ctx, protocolo=False)

    def build_adapted(ctx: Ctx):
        q = _make(ctx, protocolo=True)
        if binary_only and ctx.is_multiclass:
            from quapy.method.base import newOneVsAll

            return newOneVsAll(q)  # MANUAL: decomposicao um-contra-todos
        return q

    return register(MethodSpec(
        canonical=canonical, family=family, library="quapy",
        class_name="quapy.method.{}.{}".format(module, cls_name),
        build_adapted=build_adapted, build_naive=build_naive if binary_only else None,
        group=group, binary_only=binary_only, needs_classifier=needs_classifier,
        manual_configs=0, manual_configs_multiclass=1 if binary_only else 0,
        aliases=aliases, notes=notes,
    ))


_quapy("CC", "contagem", "aggregative", "CC")
_quapy("PCC", "contagem", "aggregative", "PCC")
# Nome canonico GACC, sem apelido para ACC. A classe e multiclasse nativa, ou
# seja, a formulacao generalizada -- e por isso o par dela e o GACC da
# mlquantify, nao o ACC, que e binario com decomposicao um-contra-todos. Com o
# apelido, o mesmo metodo era comparado com os dois, e uma das duas comparacoes
# era entre formulacoes diferentes.
_quapy("GACC", "contagem ajustada", "aggregative", "ACC",
       notes="multiclasse nativa, ou seja, a formulacao generalizada; por isso "
             "o nome canonico leva o G e o par e o GACC da mlquantify")
_quapy("PACC", "contagem ajustada", "aggregative", "PACC")
_quapy("T50", "selecao de limiar", "aggregative", "T50", binary_only=True)
_quapy("X", "selecao de limiar", "aggregative", "X", binary_only=True)
_quapy("MAX", "selecao de limiar", "aggregative", "MAX", binary_only=True)
_quapy("MS", "selecao de limiar", "aggregative", "MS", binary_only=True)
_quapy("MS2", "selecao de limiar", "aggregative", "MS2", binary_only=True)
_quapy("MLPE", "verossimilhanca", "non_aggregative",
       "MaximumLikelihoodPrevalenceEstimation", needs_classifier=False)
_quapy("EMQ", "verossimilhanca", "aggregative", "EMQ")
_quapy("HDy", "casamento de distribuicoes", "aggregative", "HDy", binary_only=True)
_quapy("DyS", "casamento de distribuicoes", "aggregative", "DyS",
       binary_only=True,
       kwargs={"n_bins": DEFAULT_BINS, "divergence": "topsoe"},
       notes="divergencia de Topsoe, a da definicao do metodo e a que a "
             "mlquantify usa; o padrao da quapy e Hellinger")
_quapy("SMM", "casamento de distribuicoes", "aggregative", "SMM", binary_only=True)
_quapy("KDEyML", "casamento de distribuicoes", "aggregative", "KDEyML")
_quapy("KDEyHD", "casamento de distribuicoes", "aggregative", "KDEyHD")
_quapy("KDEyCS", "casamento de distribuicoes", "aggregative", "KDEyCS")
_quapy("EDy", "casamento de distribuicoes", "aggregative", "EDy",
       kwargs={"distance": "euclidean"},
       notes="distancia euclidiana, como nas outras tres; o padrao da "
             "quapy e manhattan")
_quapy("HDx", "nao agregativo", "non_aggregative", "DMx", needs_classifier=False,
       notes="DMx com divergencia de Hellinger e o HDx")
_quapy("EDx", "nao agregativo", "non_aggregative", "EDx",
       needs_classifier=False, kwargs={"distance": "euclidean"},
       notes="distancia euclidiana, como nas outras tres; o padrao da "
             "quapy e manhattan")
_quapy("README", "nao agregativo", "non_aggregative", "ReadMe",
       needs_classifier=False, group="text",
       notes="exige matriz binaria de atributos; nao se aplica a dados tabulares")


def _quapy_ensemble(ctx: Ctx):
    from quapy.method.aggregative import CC
    from quapy.method.meta import Ensemble

    return Ensemble(CC(ctx.make_clf()), size=10)


register(MethodSpec("Ensembles", "metaquantificacao", "quapy",
                    "quapy.method.meta.Ensemble", _quapy_ensemble, group="meta",
                    manual_configs=2,
                    notes="exige quantificador base e tamanho do comite"))


def _svmperf(loss):
    def build(ctx: Ctx):
        import os

        from quapy.method.aggregative import SVMperf

        home = os.environ.get("SVMPERF_HOME")
        if not home:
            raise RuntimeError("SVMPERF_HOME nao definido; binario do SVM-perf ausente")
        return SVMperf(home, loss=loss)

    return build


for _name, _loss in (("SVM(Q)", "q"), ("SVM(KLD)", "kld"), ("SVM(NKLD)", "nkld"),
                     ("SVM(AE)", "mae"), ("SVM(RAE)", "mrae")):
    register(MethodSpec(_name, "minimizacao explicita da perda (ELM)", "quapy",
                        "quapy.method.aggregative.SVMperf", _svmperf(_loss),
                        group="elm", binary_only=True, manual_configs=2,
                        notes="exige o binario externo do SVM-perf (SVMPERF_HOME)"))


def _quapy_neural(cls_name):
    def build(ctx: Ctx):
        import importlib

        cls = getattr(importlib.import_module("quapy.method.meta"), cls_name)
        return cls(ctx.make_clf())

    return build


for _name in ("QuaNet", "HistNetQ", "GMNet"):
    register(MethodSpec(_name, "neurais", "quapy", "quapy.method.meta." + _name,
                        _quapy_neural(_name), group="neural", manual_configs=1,
                        notes="depende do PyTorch"))


# ===========================================================================
# QuantificationLib
#
# Os metodos que estimam a distribuicao de treinamento exigem um estimador
# validado cruzadamente (``CV_estimator``) e a separacao explicita entre
# ``estimator_train`` e ``estimator_test``. Metodos binarios exigem
# ``OneVsRestQuantifier``.
# ===========================================================================


def _euclidean(X, Y):
    """Distancia euclidiana para a QuantificationLib, que exige um callable."""
    from sklearn.metrics.pairwise import euclidean_distances

    return euclidean_distances(X, Y)


def _qlib_cv(ctx: Ctx):
    from quantificationlib.estimators.cross_validation import CV_estimator
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=ctx.random_state)
    return CV_estimator(estimator=ctx.make_clf(), cv=cv)


class _VarreduraIntervalos:
    """Varre numeros de intervalos e devolve a mediana das estimativas.

    A ``HDy`` da quantificationlib recebe ``n_bins`` escalar e o usa como
    dimensao de array, entao a varredura que a definicao do metodo pede nao
    cabe dentro dela. Este envelope faz por fora o que a mlquantify faz por
    dentro, com os mesmos intervalos e a mesma agregacao.

    Custa uma varredura, nao onze execucoes: as instancias dos estimadores
    sao compartilhadas entre os ajustes, e a ``fit`` da biblioteca detecta
    que elas ja estao ajustadas e nao as reajusta. As predicoes de treino e
    de teste vao prontas do primeiro para os demais, de modo que o
    classificador roda uma vez e o que se repete e so o histograma e a
    otimizacao -- que e justamente o que a varredura multiplica.
    """

    def __init__(self, fabrica, bins):
        self._fabrica = fabrica
        self._bins = list(bins)
        self._qs = []

    def fit(self, X, y):
        self._qs = []
        pred_treino = None
        for b in self._bins:
            q = self._fabrica(b)
            if pred_treino is None:
                q.fit(X, y)
                pred_treino = q.predictions_train_
            else:
                q.fit(X, y, predictions_train=pred_treino)
            self._qs.append(q)
        return self

    def predict(self, X):
        import numpy as np

        pred_teste = None
        estimativas = []
        for q in self._qs:
            if pred_teste is None:
                p = q.predict(X)
                pred_teste = q.predictions_test_
            else:
                p = q.predict(X, predictions_test=pred_teste)
            estimativas.append(np.asarray(p, dtype=float).ravel())
        return np.median(np.vstack(estimativas), axis=0)


def _qlib(canonical, family, module, cls_name, *, needs_train=True,
          train_cv=True, binary_only=False, needs_classifier=True,
          group="core", aliases=(), notes="", kwargs=None):
    kwargs = kwargs or {}

    def _cls():
        import importlib

        return getattr(importlib.import_module("quantificationlib." + module), cls_name)

    def build_naive(ctx: Ctx):
        cls = _cls()
        if not needs_classifier:
            return cls(**kwargs)
        clf = ctx.make_clf()
        if needs_train:
            return cls(estimator_train=clf, estimator_test=clf, **kwargs)
        return cls(estimator_test=clf, **kwargs)

    def build_adapted(ctx: Ctx):
        cls = _cls()
        if not needs_classifier:
            q = cls(**kwargs)
        elif needs_train:
            # MANUAL: a classe exige os dois estimadores separados. O de treino
            # e validado cruzadamente, que e a forma documentada; o EMQ e a
            # excecao (``train_cv=False``), porque nas outras tres bibliotecas
            # ele le os escores do classificador treinado na particao inteira.
            treino = _qlib_cv(ctx) if train_cv else ctx.make_clf()
            q = cls(estimator_train=treino, estimator_test=ctx.make_clf(), **kwargs)
        else:
            q = cls(estimator_test=ctx.make_clf(), **kwargs)
        if binary_only and ctx.is_multiclass:
            from quantificationlib.decomposition.multiclass import OneVsRestQuantifier
            from sklearn.multiclass import OneVsRestClassifier

            # MANUAL: decomposicao um-contra-resto, que ainda exige que os dois
            # estimadores sejam instancias de OneVsRestClassifier
            return OneVsRestQuantifier(
                base_quantifier=q,
                estimator_train=OneVsRestClassifier(_qlib_cv(ctx)),
                estimator_test=OneVsRestClassifier(ctx.make_clf()))
        return q

    manual = 1 if (needs_train and needs_classifier) else 0
    return register(MethodSpec(
        canonical=canonical, family=family, library="quantificationlib",
        class_name="quantificationlib.{}.{}".format(module, cls_name),
        build_adapted=build_adapted,
        build_naive=build_naive if (manual or binary_only) else None,
        group=group, binary_only=binary_only, needs_classifier=needs_classifier,
        manual_configs=manual, manual_configs_multiclass=2 if binary_only else 0,
        aliases=aliases, notes=notes,
    ))


_qlib("CC", "contagem", "baselines.cc", "CC", needs_train=False)
_qlib("PCC", "contagem", "baselines.cc", "PCC", needs_train=False)
# Mesmo caso da quapy: a classe resolve a multiclasse por minimizacao de
# distancia, que e a formulacao generalizada, entao o par e o GACC e nao o ACC.
# Par do TAC, nao do GACC, por decisao de quem escreve o estudo. No binario
# -- 76,1% dos conjuntos -- esta classe resolve a forma fechada de Forman,
# ``p = (p0 - fpr) / (tpr - fpr)``, com tpr e fpr do classificador validado
# cruzado no limiar padrao: e exatamente o que o TAC da mlquantify faz. O
# GACC resolve regressao restrita no simplex sempre, que e outro
# solucionador para a mesma grandeza.
#
# RESSALVA, que o texto precisa fazer: na multiclasse esta classe troca de
# metodo. Ela inverte a matriz de confusao transposta --
# ``linalg.inv(self.cm_.T).dot(prevalences_0)``, corta os negativos e
# renormaliza --, e so cai em minimizacao de distancia no ``except
# LinAlgError``, quando a inversao falha. O TAC, binario, e decomposto
# um-contra-resto pelo harness. Medido: no binario os dois batem a 1e-6 em
# 93,8% dos casos; na multiclasse, em 0,8%.
#
# E ate no binario ha uma troca escondida: se |cm_[1,1] - cm_[0,1]| <= 0.001
# a classe devolve a contagem sem ajuste, isto e, vira CC. A mlquantify so
# faz isso quando tpr - fpr e exatamente zero.
# O par fica certo no binario e aproximado na multiclasse -- o inverso do
# que valia com o GACC.
_qlib("TAC", "selecao de limiar", "baselines.ac", "AC",
      notes="no binario e a forma fechada de Forman no limiar padrao, que e o "
            "TAC; na multiclasse a biblioteca troca para minimizacao de "
            "distancia, que e a formulacao generalizada")
_qlib("PACC", "contagem ajustada", "baselines.ac", "PAC")
_qlib("FM", "contagem ajustada", "multiclass.friedman", "FriedmanME")
_qlib("EMQ", "verossimilhanca", "multiclass.em", "EM", train_cv=False,
      notes="escores do classificador treinado na particao inteira, como "
            "nas outras tres; a classe exige o estimador de treino assim "
            "mesmo")
# Pareado com o ``HDy`` da mlquantify, por decisao de quem escreve o estudo.
#
# A ressalva fica registrada, porque ela e real: esta classe recebe um unico
# numero de intervalos (``n_bins=8``, ``bin_strategy=equal_width``, os padroes
# dela), enquanto a definicao do metodo varre intervalos e toma a mediana das
# estimativas, que e o que a mlquantify e a quapy fazem. Medido: 0,0434 de
# diferenca absoluta media contra a mlquantify, com a mlquantify melhor em so
# 15,2% dos 560 casos e p = 1,4e-29 -- discordancia enviesada numa direcao. O
# texto deve dizer que parte dela vem da variante de intervalo fixo, e nao
# so da implementacao.
_qlib("HDy8", "casamento de distribuicoes", "multiclass.df", "HDy",
      notes="n_bins escalar (8, largura igual), que e o padrao da classe. O "
            "par e o HDy8 da mlquantify, o mesmo metodo com o mesmo numero "
            "de intervalos, para a comparacao medir implementacao e nao "
            "parametrizacao")
# Sensibilidade: a familia de limiar varrendo os escores observados ----------
#
# A ``evaluate_thresholds`` da mlquantify tem dois modos. O padrao,
# ``score_edges="fixed"``, varre ``np.linspace(0, 1, 101)`` -- grade fixa de
# 101 limiares equiespacados. O outro varre ``np.unique(probabilities)``, os
# escores observados, que e o que a quapy faz (``np.unique(decision_scores)``
# em ``_threshold_optim.py``) e o que Forman define. O modo nao esta exposto
# no quantificador: o ``_adjust`` chama a funcao sem o argumento, entao o
# usuario nao tem como pedir a varredura exata.
#
# Medido com a grade, contra a quapy: MS 0,0384, MS2 0,0203, T50 0,0172,
# MAX 0,0085, X 0,0080 -- ordenados por quanto o criterio depende do conjunto
# de candidatos, que e a assinatura da causa. Estas variantes medem quanto
# disso a varredura exata devolve.
@contextlib.contextmanager
def _limiar_exato():
    """Troca a ``evaluate_thresholds`` do modulo pelo modo exato.

    E um swap no modulo, nao na instancia, porque o ``_adjust`` chama a funcao
    pelo nome global. Vale so dentro do ``with``, e so faz sentido numa
    execucao que rode apenas os metodos de limiar.
    """
    from mlquantify.counting import _adjustment as _adj
    from mlquantify.counting._utils import evaluate_thresholds as _ev

    original = _adj.evaluate_thresholds
    _adj.evaluate_thresholds = (
        lambda y, probs, score_edges="unique": _ev(y, probs, score_edges))
    try:
        yield
    finally:
        _adj.evaluate_thresholds = original


def _com_limiar_exato(cls):
    """Subclasse que varre os escores observados em vez da grade de 101.

    Nao duplica o corpo do ``_adjust`` -- nem o da base nem o do MS, que tem
    o proprio --, so o executa com a funcao trocada.
    """

    class _Exato(cls):
        def _adjust(self, *args, **kwargs):
            with _limiar_exato():
                return super()._adjust(*args, **kwargs)

    _Exato.__name__ = cls.__name__ + "Exato"
    _Exato.__qualname__ = _Exato.__name__
    return _Exato


def _mlq_limiar_exato(canonical, cls_name):
    def build_adapted(ctx: Ctx):
        import importlib

        base = getattr(importlib.import_module("mlquantify.counting"), cls_name)
        cls = _com_limiar_exato(base)
        return cls(ctx.make_clf(), **_protocolo(base, ctx))

    return register(MethodSpec(
        canonical=canonical, family="selecao de limiar", library="mlquantify",
        class_name=f"mlquantify.counting.{cls_name} (varredura exata)",
        build_adapted=build_adapted, group="sens", binary_only=True,
        manual_configs=1,
        notes="o mesmo metodo com score_edges=\"unique\": varre os escores "
              "observados, como a quapy e como Forman define, em vez da grade "
              "fixa de 101 limiares que e o padrao da biblioteca"))


_mlq_limiar_exato("T50exato", "T50")
_mlq_limiar_exato("Xexato", "TX")
_mlq_limiar_exato("MAXexato", "TMAX")
_mlq_limiar_exato("MSexato", "MS")
_mlq_limiar_exato("MS2exato", "MS2")


# Sensibilidade, fora das execucoes oficiais (grupo ``sens``, que o
# ``DEFAULT_GROUPS`` nao inclui): o mesmo ``HDy`` da quantificationlib, mas
# com a varredura de intervalos em vez do ``n_bins=8`` fixo. Serve para
# separar quanto da discordancia do par ``HDy`` vem da parametrizacao e
# quanto vem da implementacao. Os intervalos sao os mesmos que a mlquantify
# usa por padrao -- ``linspace(10, 110, 11)``, agregados pela mediana --,
# para que o envelope comparado seja o mesmo dos dois lados.
_HDY_BINS = tuple(range(10, 111, 10))


def _build_hdy_varrido(ctx: Ctx):
    from quantificationlib.multiclass.df import HDy as _HDyQlib

    # MANUAL: as duas instancias sao criadas uma vez e compartilhadas por
    # todos os ajustes da varredura, para o classificador rodar uma vez so.
    treino = _qlib_cv(ctx)
    teste = ctx.make_clf()

    def fabrica(n_bins):
        return _HDyQlib(estimator_train=treino, estimator_test=teste,
                        n_bins=n_bins, bin_strategy="equal_width")

    return _VarreduraIntervalos(fabrica, _HDY_BINS)


register(MethodSpec(
    canonical="HDySweep", family="casamento de distribuicoes",
    library="quantificationlib",
    class_name="quantificationlib.multiclass.df.HDy (varrido)",
    build_adapted=_build_hdy_varrido, group="sens",
    manual_configs=2,
    notes="o HDy da quantificationlib com a varredura de intervalos de 10 a "
          "110 e mediana das estimativas, em vez do n_bins=8 fixo. Nao e um "
          "metodo da biblioteca: e o envelope da definicao do metodo aplicado "
          "por fora, para medir quanto da discordancia vem da parametrizacao"))

_qlib("SORD", "casamento de distribuicoes", "binary.emd", "SORDy", binary_only=True)
_qlib("EDy", "casamento de distribuicoes", "multiclass.energy", "EDy",
      kwargs={"distance": _euclidean},
      notes="distancia euclidiana, como nas outras tres; o padrao da "
            "biblioteca e manhattan")
_qlib("HDx", "nao agregativo", "multiclass.df", "HDX", needs_train=False,
      needs_classifier=False)
_qlib("EDx", "nao agregativo", "multiclass.energy", "EDX", needs_train=False,
      needs_classifier=False)
# Nome proprio: esta classe e um metodo AC sobre o classificador PWK -- o
# ``fit`` dela roda o kNN no treino inteiro para montar a matriz de confusao
# e ajusta a contagem. O ``PWK`` da mlquantify e ``class PWK(CC)``, contagem
# pura. Medido: o erro da mlquantify fica plano com o tamanho da amostra
# (0,1218 a 0,1068), o da qlib cai pela metade (0,1437 a 0,0744) -- enviesado
# contra consistente. O par dela e o ``PWKAC`` da mlquantify, abaixo.
_qlib("PWKAC", "vizinhos e arvores", "multiclass.knn", "PWKQuantifier",
      needs_train=False, needs_classifier=False,
      notes="AC sobre o classificador PWK: monta a matriz de confusao no fit "
            "e ajusta a contagem; nao e contagem pura como o PWK da mlquantify")


def _qlib_ensemble(ctx: Ctx):
    from quantificationlib.bag_generator import PriorShift_BagGenerator
    from quantificationlib.baselines.cc import CC
    from quantificationlib.ensembles.eoq import EnsembleOfClassifiers, EoQ

    return EoQ(
        base_quantifier=CC(estimator_test=ctx.make_clf()), n_quantifiers=10,
        bag_generator=PriorShift_BagGenerator(n_bags=10,
                                              random_state=ctx.random_state),
        combination_strategy="mean",
        # MANUAL: o EoQ ainda exige os estimadores do comite, mesmo quando o
        # quantificador base ja recebeu um classificador
        ensemble_estimator_train=EnsembleOfClassifiers(ctx.make_clf()),
        ensemble_estimator_test=EnsembleOfClassifiers(ctx.make_clf()))


# O EoQ da versao 0.1.2 nao se comporta como documentado: com o quantificador
# base configurado da forma indicada, EoQ.fit chama fit(X=None) e quebra; sem
# o estimador no quantificador base ele executa, mas devolve sempre o mesmo
# vetor de prevalencias, independentemente da amostra de teste. Fica
# registrado como indisponivel em vez de gerar numeros sem sentido. Para
# reavaliar, mude available para True.
register(MethodSpec("Ensembles", "metaquantificacao", "quantificationlib",
                    "quantificationlib.ensembles.eoq.EoQ", _qlib_ensemble,
                    group="meta", manual_configs=5, available=False,
                    unavailable_reason="EoQ 0.1.2 devolve estimativa constante, "
                                       "independente da amostra de teste",
                    notes="exige quantificador base, gerador de sacolas, estrategia "
                          "de combinacao e os dois estimadores do comite"))


# ===========================================================================
# qunfold
#
# Os metodos agregativos exigem ``CVClassifier``; os baseados em histograma
# exigem o numero de intervalos, que nao tem valor padrao.
#
# Atencao aos nomes: as classes da qunfold materializam o *framework
# unificado* da Secao 3.4.4, ou seja, sao as formulacoes **generalizadas**.
# O que a qunfold chama de ``HDy`` e o GHDy, nao o HDy classico; o mesmo
# vale para ``ACC``, ``PACC`` e ``HDx``. Por isso o nome canonico registrado
# aqui leva o G, e esses metodos nao recebem alias para a versao sem G: sao
# formulacoes distintas e nao devem ser comparadas entre si.
# ===========================================================================


def _qf_cv(ctx: Ctx):
    from qunfold.sklearn import CVClassifier

    return CVClassifier(ctx.make_clf(), n_estimators=CV_FOLDS,
                        random_state=ctx.random_state)


def _qunfold(canonical, family, cls_name, *, needs_cv=True, needs_classifier=True,
             needs_bins=False, aliases=(), notes="", group="core", kwargs=None):
    kwargs = kwargs or {}
    def _cls():
        import qunfold

        return getattr(qunfold, cls_name)

    def build_naive(ctx: Ctx):
        cls = _cls()
        args = [ctx.make_clf()] if needs_classifier else []
        if needs_bins:
            args.append(DEFAULT_BINS)
        return cls(*args)

    def build_adapted(ctx: Ctx):
        cls = _cls()
        args = []
        if needs_classifier:
            # MANUAL: classificador encapsulado para predicoes fora da amostra
            args.append(_qf_cv(ctx) if needs_cv else ctx.make_clf())
        if needs_bins:
            args.append(DEFAULT_BINS)  # MANUAL: n_bins nao tem valor padrao
        # Todo metodo da qunfold sorteia o ponto de partida do otimizador em
        # ``minimize``; sem ``seed`` cada uma das 400 amostras parte de um
        # ponto diferente e a execucao nao se repete.
        return cls(*args, **{**kwargs, **_protocolo(cls, ctx)})

    manual = (1 if (needs_cv and needs_classifier) else 0) + (1 if needs_bins else 0)
    return register(MethodSpec(
        canonical=canonical, family=family, library="qunfold",
        class_name="qunfold." + cls_name, build_adapted=build_adapted,
        build_naive=build_naive if manual else None, group=group,
        needs_classifier=needs_classifier, manual_configs=manual,
        aliases=aliases, notes=notes,
    ))


_qunfold("GACC", "contagem ajustada", "ACC",
         notes="a classe qunfold.ACC e a formulacao generalizada (GACC)")
_qunfold("GPACC", "contagem ajustada", "PACC",
         notes="a classe qunfold.PACC e a formulacao generalizada (GPACC)")
# O par do EMQ e o ``ExpectationMaximizer``, nao o ``LikelihoodMaximizer``.
# Os dois existem na qunfold e a propria biblioteca os distingue pela
# referencia: o ``ExpectationMaximizer`` e Saerens et al. (2002), o mesmo
# algoritmo EM do EMQ da mlquantify, da quapy e da quantificationlib; o
# ``LikelihoodMaximizer`` e Alexandari et al. (2020), que minimiza a
# log-verossimilhanca negativa regularizada com o otimizador generico. A
# qunfold diz que sao "asymptotically equivalent", o que e uma afirmacao
# sobre o limite, nao sobre o que cada um calcula numa amostra de 10 itens.
# Pareado com o ``LikelihoodMaximizer``, o EMQ media EM contra otimizacao
# numerica, que e a mesma confusao que o ACC/GACC e o HDy produziam.
_qunfold("EMQ", "verossimilhanca", "ExpectationMaximizer", needs_cv=False,
         # MANUAL: 1e-6/100 sao os padroes da qunfold; 1e-4/1000 sao os das
         # outras tres, e e neles que o criterio de parada deixa de ser fonte
         # de diferenca.
         kwargs={"tol": 1e-4, "max_iter": 1000},
         notes="EM de Saerens et al. (2002), o mesmo algoritmo do EMQ das "
               "outras tres: laco iterativo com max_iter=100 e tol=1e-6, sem "
               "otimizador generico e sem ponto de partida aleatorio")
# Nome canonico proprio, e por isso sem par na mlquantify: e um metodo
# diferente, nao outra implementacao do EMQ. A mlquantify tem o arcabouco
# correspondente -- o ``LikelihoodComposeQuantifier`` com a
# ``RegularizedMixtureNLLLoss``, com os mesmos ``tau_0`` e ``tau_1`` --, mas
# nao expoe um quantificador nomeado que o use como maximizador de
# verossimilhanca; o ``MLPE`` o sobrescreve para devolver a prevalencia de
# treino. Se um dia expuser, este e o par dele.
_qunfold("MLQ", "verossimilhanca", "LikelihoodMaximizer", needs_cv=False,
         notes="maxima verossimilhanca de Alexandari et al. (2020): minimiza "
               "a log-verossimilhanca negativa regularizada (tau_0, tau_1) "
               "com trust-ncg e Hessiana por autodiferenciacao. Medido como "
               "EMQ ate a execucao final-emq, o que pareava EM contra "
               "otimizacao numerica")
_qunfold("GHDy", "casamento de distribuicoes", "HDy", needs_bins=True,
         notes="a classe qunfold.HDy e a formulacao generalizada (GHDy)")
_qunfold("EDy", "casamento de distribuicoes", "EDy")
# Tambem com nome proprio, pelo mesmo motivo: ``qunfold.KMM`` e kernel mean
# matching com nucleo de energia, e o ``MMD`` da mlquantify e o
# ``MMD_RKHS``, com nucleo RBF. Mesmo arcabouco, nucleo diferente, logo
# estimadores diferentes.
#
# Medido enquanto estava pareado como ``MMD``: 0,0428 de diferenca absoluta
# media, mas com a mlquantify melhor em 51,2% dos 508 casos e p = 0,133 --
# magnitude sem vies e sem sistematicidade, que e o que se ve comparando
# dois metodos distintos de qualidade parecida, e nao duas implementacoes
# do mesmo metodo.
_qunfold("KMM", "casamento de distribuicoes", "KMM", needs_classifier=False,
         notes="kernel mean matching com nucleo de energia. Nao e pareado "
               "com o MMD da mlquantify, que e o MMD_RKHS com nucleo RBF")
_qunfold("GHDx", "nao agregativo", "HDx", needs_classifier=False, needs_bins=True,
         notes="a classe qunfold.HDx e a formulacao generalizada (GHDx)")
_qunfold("EDx", "nao agregativo", "EDx", needs_classifier=False)


# ===========================================================================
# Consulta
# ===========================================================================


def specs_for(library: str, groups=("core",), methods=(), skip=()) -> list[MethodSpec]:
    """Metodos de uma biblioteca, filtrados por grupo e nome."""
    out = []
    for spec in _REGISTRY.get(library, []):
        if spec.group not in groups:
            continue
        if methods and spec.canonical not in methods:
            continue
        if spec.canonical in skip:
            continue
        out.append(spec)
    return out


def all_specs() -> list[MethodSpec]:
    return [s for specs in _REGISTRY.values() for s in specs]


def coverage_table():
    """Tabela de cobertura: metodo canonico x biblioteca."""
    import pandas as pd

    rows = []
    for spec in all_specs():
        for name in (spec.canonical,) + tuple(spec.aliases):
            rows.append({
                "metodo": name,
                "familia": spec.family,
                "biblioteca": spec.library,
                "grupo": spec.group,
                "classe": spec.class_name,
                "alias_de": "" if name == spec.canonical else spec.canonical,
                "binario_apenas": spec.binary_only,
                "configuracoes_manuais": spec.manual_configs,
                "configuracoes_manuais_multiclasse": (spec.manual_configs
                                                      + spec.manual_configs_multiclass),
                "disponivel": spec.available,
                "motivo_indisponivel": spec.unavailable_reason,
                "observacoes": spec.notes,
            })
    return pd.DataFrame(rows).sort_values(["familia", "metodo", "biblioteca"])
