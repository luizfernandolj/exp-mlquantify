import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm

from mlquantify.adjust_counting import AC, PAC
from mlquantify.utils import get_prev_from_labels
from mlquantify.model_selection import UPP
from mlquantify.metrics import MAE, AE
from mlquantify import set_config
from quapy.method.aggregative import ACC, PACC

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

set_config(prevalence_return_type="array")

binary_dataset = "datasets/binarios/datasets/occupancy.csv"
multiclass_dataset = "datasets/multiclasse/quapy_data/dry-bean.csv"

def load_data(path):
    data = pd.read_csv(path)

    data = data.dropna()

    if "target" not in data.columns:
        column = "class"
    else:
        column = "target"

    X = data.drop(columns=[column])
    y = data[column]
    return X, y


def test_ac_pac_quapy(path):
    X, y = load_data(path)

    X = X.to_numpy()
    y = y.to_numpy()

    clf = RandomForestClassifier(random_state=42)
    upp = UPP(batch_size=1000, n_prevalences=20, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ac_quapy = ACC(clf)
    pac_quapy = PACC(clf)

    ac_mlquantify = AC(clf)
    pac_mlquantify = PAC(clf)

    ac_quapy.fit(X_train, y_train)
    pac_quapy.fit(X_train, y_train)

    ac_mlquantify.fit(X_train, y_train)
    pac_mlquantify.fit(X_train, y_train)

    results = {"AC Quapy": [], "PAC Quapy": [], "AC MLQuantify": [], "PAC MLQuantify": []}

    for idx in tqdm(upp.split(X_test, y_test), total=20):
        X_batch = X_test[idx]
        y_batch = y_test[idx]

        # Test on the same data
        y_pred_quapy_ac = ac_quapy.predict(X_batch)
        y_pred_quapy_pac = pac_quapy.predict(X_batch)

        y_pred_mlquantify_ac = ac_mlquantify.predict(X_batch)
        y_pred_mlquantify_pac = pac_mlquantify.predict(X_batch)

        real = get_prev_from_labels(y_batch)

        mae_ac_quapy = MAE(real, y_pred_quapy_ac)
        mae_pac_quapy = MAE(real, y_pred_quapy_pac)

        mae_ac_mlquantify = MAE(real, y_pred_mlquantify_ac)
        mae_pac_mlquantify = MAE(real, y_pred_mlquantify_pac)

        results["AC Quapy"].append(mae_ac_quapy)
        results["PAC Quapy"].append(mae_pac_quapy)
        results["AC MLQuantify"].append(mae_ac_mlquantify)
        results["PAC MLQuantify"].append(mae_pac_mlquantify)

        assert np.isclose(mae_ac_quapy, mae_ac_mlquantify, atol=1e-2), f"AC Quapy: {mae_ac_quapy}, AC MLQuantify: {mae_ac_mlquantify}"
        assert np.isclose(mae_pac_quapy, mae_pac_mlquantify, atol=1e-2), f"PAC Quapy: {mae_pac_quapy}, PAC MLQuantify: {mae_pac_mlquantify}"


    df = pd.DataFrame(results)
    df.to_csv(f"results_ac_pac_quapy_{path.split('/')[-1].split('.')[0]}.csv", index=False)        
        


if __name__ == "__main__":
    test_ac_pac_quapy(binary_dataset)
    test_ac_pac_quapy(multiclass_dataset)

