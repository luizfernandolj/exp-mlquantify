"""Experimento de um conjunto de dados com a QuantificationLib.

Script de referencia para a questao Q3. As linhas marcadas com ``# MANUAL:``
existem apenas para adaptar o metodo ou os dados ao formato da biblioteca.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from quantificationlib.bag_generator import PriorShift_BagGenerator
from quantificationlib.baselines.ac import AC, PAC
from quantificationlib.baselines.cc import CC, PCC
from quantificationlib.estimators.cross_validation import CV_estimator
from quantificationlib.metrics.multiclass import mean_absolute_error
from quantificationlib.multiclass.df import HDy
from quantificationlib.multiclass.em import EM
from quantificationlib.multiclass.energy import EDy

NEEDS_TRAIN = {"AC", "PAC", "EM", "HDy", "EDy"}
METHODS = {
    "CC": CC,
    "PCC": PCC,
    "AC": AC,
    "PAC": PAC,
    "EM": EM,
    "HDy": HDy,
    "EDy": EDy,
}


def relative_absolute_error(real, estimated, eps):
    real = (real + eps) / (1 + eps * len(real))
    estimated = (estimated + eps) / (1 + eps * len(estimated))
    return float(np.mean(np.abs(estimated - real) / real))


def main(path):
    data = pd.read_csv(path)
    target = "target" if "target" in data.columns else data.columns[-1]
    X = data.drop(columns=[target]).to_numpy(dtype=float)
    y = pd.factorize(data[target])[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )
    n_classes = len(np.unique(y))

    # MANUAL: os metodos que estimam a distribuicao de treinamento exigem um
    # estimador validado cruzadamente para gerar predicoes fora da amostra.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_estimator = CV_estimator(
        estimator=LogisticRegression(max_iter=1000, random_state=42), cv=folds
    )

    results = []
    for name, method in METHODS.items():
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        if name in NEEDS_TRAIN:
            # MANUAL: separacao explicita entre estimador de treino e de teste.
            quantifier = method(estimator_train=cv_estimator, estimator_test=estimator)
        else:
            quantifier = method(estimator_test=estimator)
        quantifier.fit(X_train, y_train)

        for size in (10, 50, 100, 500):
            generator = PriorShift_BagGenerator(
                n_bags=100, bag_size=size, method="Uniform", random_state=42
            )
            prevalences, indexes = generator.generate_bags(X_test, y_test)
            for bag in range(indexes.shape[1]):
                index = indexes[:, bag]
                real = prevalences[:, bag]
                estimated = quantifier.predict(X_test[index])
                results.append(
                    {
                        "metodo": name,
                        "tamanho": size,
                        "ae": mean_absolute_error(real, estimated),
                        "rae": relative_absolute_error(real, estimated, 1 / (2 * size)),
                    }
                )

    table = pd.DataFrame(results)
    print(table.groupby(["metodo", "tamanho"])[["ae", "rae"]].mean())


if __name__ == "__main__":
    main(sys.argv[1])
