"""Experimento de um conjunto de dados com a mlquantify.

Script de referencia para a questao Q3: conduz o experimento completo, da
leitura dos dados ao calculo das medidas de erro, seguindo apenas a
documentacao oficial da biblioteca. Formatado pelo Black (88 colunas).
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlquantify import set_config
from mlquantify.counting import CC, GACC, GPACC, PCC
from mlquantify.likelihood import EMQ
from mlquantify.matching import DyS, EDy, HDy, KDEyML
from mlquantify.metrics import AE, RAE
from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels

set_config(prevalence_return_type="array")

METHODS = {
    "CC": CC,
    "PCC": PCC,
    "GACC": GACC,
    "GPACC": GPACC,
    "EMQ": EMQ,
    "HDy": HDy,
    "DyS": DyS,
    "KDEyML": KDEyML,
    "EDy": EDy,
}


def main(path):
    data = pd.read_csv(path)
    target = "target" if "target" in data.columns else data.columns[-1]
    X = data.drop(columns=[target]).to_numpy(dtype=float)
    y = pd.factorize(data[target])[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )

    protocol = UPP(batch_size=[10, 50, 100, 500], n_prevalences=100, random_state=42)
    samples = list(protocol.split(X_test, y_test))

    results = []
    for name, method in METHODS.items():
        quantifier = method(LogisticRegression(max_iter=1000, random_state=42))
        quantifier.fit(X_train, y_train)
        for index in samples:
            real = get_prev_from_labels(y_test[index])
            estimated = quantifier.predict(X_test[index])
            results.append(
                {
                    "metodo": name,
                    "tamanho": len(index),
                    "ae": AE(real, estimated),
                    "rae": RAE(real, estimated),
                }
            )

    table = pd.DataFrame(results)
    print(table.groupby(["metodo", "tamanho"])[["ae", "rae"]].mean())


if __name__ == "__main__":
    main(sys.argv[1])
