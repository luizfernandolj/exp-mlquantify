"""Experimento de um conjunto de dados com a QuaPy.

Script de referencia para a questao Q3. As linhas marcadas com ``# MANUAL:``
existem apenas para adaptar o metodo ou os dados ao formato da biblioteca,
e nao fazem parte do fluxo comum do experimento.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import ACC, CC, PACC, PCC, EMQ, DyS, EDy, HDy, KDEyML
from quapy.method.base import newOneVsAll
from quapy.protocol import UPP

METHODS = {
    "CC": CC,
    "PCC": PCC,
    "ACC": ACC,
    "PACC": PACC,
    "EMQ": EMQ,
    "HDy": HDy,
    "DyS": DyS,
    "KDEyML": KDEyML,
    "EDy": EDy,
}

# MANUAL: metodos binarios param em problemas multiclasse e precisam ser
# encapsulados em uma classe de decomposicao um contra todos.
BINARY_ONLY = {"HDy", "DyS"}


def main(path):
    data = pd.read_csv(path)
    target = "target" if "target" in data.columns else data.columns[-1]
    X = data.drop(columns=[target]).to_numpy(dtype=float)
    y = pd.factorize(data[target])[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=42
    )
    n_classes = len(np.unique(y))

    # MANUAL: os protocolos operam sobre LabelledCollection, nao sobre arrays.
    test_collection = LabelledCollection(X_test, y_test)

    results = []
    for name, method in METHODS.items():
        quantifier = method(LogisticRegression(max_iter=1000, random_state=42))
        if name in BINARY_ONLY and n_classes > 2:
            quantifier = newOneVsAll(quantifier)  # MANUAL: decomposicao binaria
        quantifier.fit(X_train, y_train)

        for size in (10, 50, 100, 500):
            protocol = UPP(
                test_collection, sample_size=size, repeats=100, random_state=42
            )
            for sample, real in protocol():
                estimated = quantifier.predict(sample)
                results.append(
                    {
                        "metodo": name,
                        "tamanho": size,
                        "ae": qp.error.ae(real, estimated),
                        "rae": qp.error.rae(real, estimated),
                    }
                )

    table = pd.DataFrame(results)
    print(table.groupby(["metodo", "tamanho"])[["ae", "rae"]].mean())


if __name__ == "__main__":
    main(sys.argv[1])
