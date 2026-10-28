"""Model deployment basics: save and load models with pickle and joblib.

Part A: a scikit-learn model (joblib is the recommended choice for models
        holding large NumPy arrays).
Part B: load the from-scratch NumPy network saved by train.py and predict.

Run:  python deploy.py     (run train.py first for Part B)
"""
import os
import pickle
import time

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def part_a_sklearn():
    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
    print(f"Trained RandomForest, test accuracy: {clf.score(X_te, y_te):.3f}")

    # --- pickle ---
    t = time.perf_counter()
    with open("iris_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("iris_model.pkl", "rb") as f:
        loaded_pkl = pickle.load(f)
    print(f"pickle : {os.path.getsize('iris_model.pkl') / 1024:.1f} KB, "
          f"save+load {time.perf_counter() - t:.3f}s")

    # --- joblib ---
    t = time.perf_counter()
    joblib.dump(clf, "iris_model.joblib")
    loaded_job = joblib.load("iris_model.joblib")
    print(f"joblib : {os.path.getsize('iris_model.joblib') / 1024:.1f} KB, "
          f"save+load {time.perf_counter() - t:.3f}s")

    # Loaded models must predict identically to the original
    assert np.array_equal(clf.predict(X_te), loaded_pkl.predict(X_te))
    assert np.array_equal(clf.predict(X_te), loaded_job.predict(X_te))
    print("Loaded models give identical predictions.\n")


def part_b_scratch_model():
    if not os.path.exists("model.joblib"):
        print("model.joblib not found - run `python train.py` first.")
        return
    # The class must be importable when unpickling (nn.network.NeuralNetwork)
    model = joblib.load("model.joblib")
    sample = np.array([[0.0, 0.5], [1.0, -0.5]])  # points from the two moons
    print("From-scratch NumPy model loaded.")
    print("Probabilities:", model.predict_proba(sample).ravel().round(3))
    print("Predicted classes:", model.predict(sample).ravel())


if __name__ == "__main__":
    part_a_sklearn()
    part_b_scratch_model()
