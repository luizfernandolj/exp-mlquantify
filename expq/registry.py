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

from dataclasses import dataclass
from typing import Callable

from .config import ALL_LIBRARIES

CV_FOLDS = 5
DEFAULT_BINS = 8


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
         needs_classifier=True, group="core", aliases=(), notes=""):
    def build(ctx: Ctx):
        import importlib

        cls = getattr(importlib.import_module("mlquantify." + module), cls_name)
        return cls(ctx.make_clf()) if needs_classifier else cls()

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
_mlq("GPACC", "contagem ajustada", "counting", "GPACC", aliases=("PACC",))
_mlq("FM", "contagem ajustada", "counting", "FM")
_mlq("TAC", "selecao de limiar", "counting", "TAC", binary_only=True)
_mlq("T50", "selecao de limiar", "counting", "T50", binary_only=True)
_mlq("X", "selecao de limiar", "counting", "TX", binary_only=True)
_mlq("MAX", "selecao de limiar", "counting", "TMAX", binary_only=True)
_mlq("MS", "selecao de limiar", "counting", "MS", binary_only=True)
_mlq("MS2", "selecao de limiar", "counting", "MS2", binary_only=True)
_mlq("MLPE", "verossimilhanca", "likelihood", "MLPE",
     notes="linha de base trivial de Hassan, Maletzke e Batista (2021)")
_mlq("EMQ", "verossimilhanca", "likelihood", "EMQ")
_mlq("CDE", "verossimilhanca", "likelihood", "CDE")
_mlq("HDy", "casamento de distribuicoes", "matching", "HDy", binary_only=True)
_mlq("GHDy", "casamento de distribuicoes", "matching", "GHDy")
_mlq("DyS", "casamento de distribuicoes", "matching", "DyS", binary_only=True)
_mlq("SORD", "casamento de distribuicoes", "matching", "SORD", binary_only=True)
_mlq("SMM", "casamento de distribuicoes", "matching", "SMM", binary_only=True)
_mlq("KDEyML", "casamento de distribuicoes", "matching", "KDEyML")
_mlq("KDEyHD", "casamento de distribuicoes", "matching", "KDEyHD")
_mlq("KDEyCS", "casamento de distribuicoes", "matching", "KDEyCS")
_mlq("GKDEyML", "casamento de distribuicoes", "matching", "GKDEyML")
_mlq("EDy", "casamento de distribuicoes", "matching", "EDy")
_mlq("MMD", "casamento de distribuicoes", "matching", "MMD_RKHS", needs_classifier=False)
_mlq("HDx", "nao agregativo", "matching", "HDx", needs_classifier=False)
_mlq("GHDx", "nao agregativo", "matching", "GHDx", needs_classifier=False)
_mlq("EDx", "nao agregativo", "matching", "EDx", needs_classifier=False)
_mlq("PWK", "vizinhos e arvores", "neighbors", "PWK", needs_classifier=False)


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

    def _make(ctx: Ctx):
        import importlib

        cls = getattr(importlib.import_module("quapy.method." + module), cls_name)
        return cls(ctx.make_clf(), **kwargs) if needs_classifier else cls(**kwargs)

    def build_naive(ctx: Ctx):
        return _make(ctx)

    def build_adapted(ctx: Ctx):
        q = _make(ctx)
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
_quapy("ACC", "contagem ajustada", "aggregative", "ACC", aliases=("GACC",),
       notes="multiclasse nativa; equivale a GACC em problemas multiclasse")
_quapy("PACC", "contagem ajustada", "aggregative", "PACC", aliases=("GPACC",))
_quapy("T50", "selecao de limiar", "aggregative", "T50", binary_only=True)
_quapy("X", "selecao de limiar", "aggregative", "X", binary_only=True)
_quapy("MAX", "selecao de limiar", "aggregative", "MAX", binary_only=True)
_quapy("MS", "selecao de limiar", "aggregative", "MS", binary_only=True)
_quapy("MS2", "selecao de limiar", "aggregative", "MS2", binary_only=True)
_quapy("MLPE", "verossimilhanca", "non_aggregative",
       "MaximumLikelihoodPrevalenceEstimation", needs_classifier=False)
_quapy("EMQ", "verossimilhanca", "aggregative", "EMQ")
_quapy("HDy", "casamento de distribuicoes", "aggregative", "HDy", binary_only=True)
_quapy("DyS", "casamento de distribuicoes", "aggregative", "DyS", binary_only=True)
_quapy("SMM", "casamento de distribuicoes", "aggregative", "SMM", binary_only=True)
_quapy("KDEyML", "casamento de distribuicoes", "aggregative", "KDEyML")
_quapy("KDEyHD", "casamento de distribuicoes", "aggregative", "KDEyHD")
_quapy("KDEyCS", "casamento de distribuicoes", "aggregative", "KDEyCS")
_quapy("EDy", "casamento de distribuicoes", "aggregative", "EDy")
_quapy("HDx", "nao agregativo", "non_aggregative", "DMx", needs_classifier=False,
       notes="DMx com divergencia de Hellinger e o HDx")
_quapy("EDx", "nao agregativo", "non_aggregative", "EDx", needs_classifier=False)
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


def _qlib_cv(ctx: Ctx):
    from quantificationlib.estimators.cross_validation import CV_estimator
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=ctx.random_state)
    return CV_estimator(estimator=ctx.make_clf(), cv=cv)


def _qlib(canonical, family, module, cls_name, *, needs_train=True,
          binary_only=False, needs_classifier=True, group="core", aliases=(),
          notes="", kwargs=None):
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
            # MANUAL: estimador validado cruzadamente para a distribuicao de treino
            q = cls(estimator_train=_qlib_cv(ctx), estimator_test=ctx.make_clf(), **kwargs)
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
_qlib("ACC", "contagem ajustada", "baselines.ac", "AC", aliases=("GACC",),
      notes="na multiclasse a biblioteca resolve por minimizacao de distancia (GACC)")
_qlib("PACC", "contagem ajustada", "baselines.ac", "PAC", aliases=("GPACC",))
_qlib("FM", "contagem ajustada", "multiclass.friedman", "FriedmanME")
_qlib("EMQ", "verossimilhanca", "multiclass.em", "EM")
_qlib("HDy", "casamento de distribuicoes", "multiclass.df", "HDy")
_qlib("SORD", "casamento de distribuicoes", "binary.emd", "SORDy", binary_only=True)
_qlib("EDy", "casamento de distribuicoes", "multiclass.energy", "EDy")
_qlib("HDx", "nao agregativo", "multiclass.df", "HDX", needs_train=False,
      needs_classifier=False)
_qlib("EDx", "nao agregativo", "multiclass.energy", "EDX", needs_train=False,
      needs_classifier=False)
_qlib("PWK", "vizinhos e arvores", "multiclass.knn", "PWKQuantifier",
      needs_train=False, needs_classifier=False)


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
             needs_bins=False, aliases=(), notes="", group="core"):
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
        return cls(*args)

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
_qunfold("EMQ", "verossimilhanca", "LikelihoodMaximizer", needs_cv=False)
_qunfold("GHDy", "casamento de distribuicoes", "HDy", needs_bins=True,
         notes="a classe qunfold.HDy e a formulacao generalizada (GHDy)")
_qunfold("EDy", "casamento de distribuicoes", "EDy")
_qunfold("MMD", "casamento de distribuicoes", "KMM", needs_classifier=False,
         notes="KMM com nucleo de energia")
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
