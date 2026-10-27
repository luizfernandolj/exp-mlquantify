"""Experimento de um conjunto de dados com a qunfold.

Script de referencia para a questao Q3. A biblioteca nao oferece protocolos
de avaliacao nem metricas de erro, entao a geracao das amostras e o calculo
das medidas precisam ser escritos no proprio script.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from qunfold import ACC, EDx, EDy, HDx, HDy, KMM, LikelihoodMaximizer, PACC
from qunfold.sklearn import CVClassifier

NEEDS_CV = {"ACC", "PACC", "HDy", "EDy"}
NEEDS_BINS = {"HDy", "HDx"}
NEEDS_CLASSIFIER = {"ACC", "PACC", "EMQ", "HDy", "EDy"}
METHODS = {
    "ACC": ACC,
    "PACC": PACC,
    "EMQ": LikelihoodMaximizer,
    "HDy": HDy,
    "EDy": EDy,
    "HDx": HDx,
    "EDx": EDx,
    "KMM": KMM,
}


def absolute_error(real, estimated):
    return float(np.mean(np.abs(estimated - real)))


def relative_absolute_error(real, estimated, eps):
    real = (real + eps) / (1 + eps * len(real))
    estimated = (estimated + eps) / (1 + eps * len(estimated))
    return float(np.mean(np.abs(estimated - real) / real))


def uniform_prevalence_samples(y, size, repeats, n_classes, rng):
    """Protocolo UPP escrito a mao: a biblioteca nao oferece protocolos."""
    by_class = [np.flatnonzero(y == c) for c in range(n_classes)]
    samples = []
    for _ in range(repeats):
        prevalence = rng.dirichlet(np.ones(n_classes))
        counts = np.random.multinomial(size, prevalence)
        index = np.concatenate(
            [rng.choice(by_class[c], counts[c], replace=True) for c in range(n_classes)]
        )
        samples.append(index)
    return samples


def main(path):
    data = pd.read_csv(path)
    target = "target" if "target" in data.columns else data.columns[-1]
    X = data.drop(columns=[target]).to_numpy(dtype=float)
    y = pd.factorize(data[target])[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )
    n_classes = len(np.unique(y))
    rng = np.random.default_rng(42)

    results = []
    for name, method in METHODS.items():
        arguments = []
        if name in NEEDS_CLASSIFIER:
            estimator = LogisticRegression(max_iter=1000, random_state=42)
            if name in NEEDS_CV:
                # MANUAL: classificador encapsulado para predicoes fora da amostra.
                estimator = CVClassifier(estimator, n_estimators=5, random_state=42)
            arguments.append(estimator)
        if name in NEEDS_BINS:
            arguments.append(8)  # MANUAL: n_bins nao tem valor padrao
        quantifier = method(*arguments)
        quantifier.fit(X_train, y_train)

        for size in (10, 50, 100, 500):
            for index in uniform_prevalence_samples(y_test, size, 100, n_classes, rng):
                real = np.bincount(y_test[index], minlength=n_classes) / len(index)
                estimated = quantifier.predict(X_test[index])
                results.append(
                    {
                        "metodo": name,
                        "tamanho": size,
                        "ae": absolute_error(real, estimated),
                        "rae": relative_absolute_error(real, estimated, 1 / (2 * size)),
                    }
                )

    table = pd.DataFrame(results)
    print(table.groupby(["metodo", "tamanho"])[["ae", "rae"]].mean())


if __name__ == "__main__":
    main(sys.argv[1])
