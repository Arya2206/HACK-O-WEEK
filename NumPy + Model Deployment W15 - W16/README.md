# Neural Networks From Scratch + Model Deployment Basics

Hack-o-Week task: model deployment basics, perceptrons, activation functions,
forward/backward propagation, loss functions, optimizers (SGD, Adam), and a
neural network built from scratch with NumPy.

## Project structure

```
.
├── nn/
│   ├── activations.py   # sigmoid, ReLU, tanh, step, softmax (+ derivatives)
│   ├── losses.py        # MSE, binary cross-entropy (+ gradients)
│   ├── optimizers.py    # SGD (with momentum) and Adam
│   └── network.py       # NeuralNetwork: forward, backward, fit, predict
├── perceptron.py        # single perceptron: learns AND/OR, fails on XOR
├── train.py             # gradient check, SGD vs Adam, saves the model
├── deploy.py            # pickle vs joblib, loading the saved model
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt
python perceptron.py   # perceptron demo
python train.py        # train, gradient-check, save model.pkl / model.joblib
python deploy.py       # save/load with pickle and joblib
```

## 1. Model deployment basics (pickle / joblib)

Training is expensive; prediction should be cheap. So we train once, **serialize**
the fitted model to disk, and **load** it in a separate process (an API, an app).

```python
import pickle, joblib

pickle.dump(model, open("model.pkl", "wb"))      # standard library
model = pickle.load(open("model.pkl", "rb"))

joblib.dump(model, "model.joblib")               # better for big NumPy arrays
model = joblib.load("model.joblib")
```

- `joblib` is the usual choice for scikit-learn models with large arrays.
  On the tiny model in `deploy.py` pickle was actually a bit faster/smaller; the
  difference only matters for large models.
- The class definition (e.g. `nn.network.NeuralNetwork`) must be importable when loading.
- **Security:** never unpickle files from untrusted sources, since loading can execute code.
- Use the same library versions for saving and loading.

## 2. Perceptrons, activation functions, forward/backward propagation

A **perceptron** computes `step(w·x + b)` and learns with the rule
`w ← w + lr · (y − ŷ) · x`. It can learn AND/OR but not XOR (not linearly
separable), so we stack layers and use differentiable activations.

| Activation | Formula | Notes |
|---|---|---|
| Sigmoid | 1 / (1 + e^−z) | outputs in (0, 1); used for binary output |
| Tanh | (e^z − e^−z) / (e^z + e^−z) | zero-centered |
| ReLU | max(0, z) | cheap, avoids vanishing gradients |

**Forward propagation:** for each layer, `z = a·W + b`, `a = activation(z)`.

**Backward propagation (chain rule):** starting from `dL/dŷ`, move backwards:
`δ = (δ·Wᵀ) ⊙ activation'(z)`, then `dW = aᵀ·δ`, `db = Σδ`.
`train.py` verifies this against numerical gradients (max error ≈ 1e-10).

## 3. Loss functions and optimizers

- **MSE:** `mean((ŷ − y)²)`: regression.
- **Binary cross-entropy:** `−mean(y·log ŷ + (1−y)·log(1−ŷ))`: classification.

Optimizers turn gradients into weight updates:

- **SGD:** `w ← w − lr · g` (optionally with momentum).
- **Adam:** keeps running averages of the gradient (`m`) and squared gradient (`v`),
  bias-corrects them, and updates `w ← w − lr · m̂ / (√v̂ + ε)`. It adapts the step
  size per parameter.

## 4. Neural network from scratch (NumPy)

```python
from nn.network import NeuralNetwork

net = NeuralNetwork([2, 16, 16, 1], hidden_activation="tanh", loss="bce")
net.fit(X_train, y_train, epochs=300, optimizer="adam", lr=0.02, batch_size=32)
net.predict(X_test)
```

Each training step: **forward → loss → backward → optimizer update**.

## Results (make_moons, 600 samples, noise 0.2)

| Optimizer | Final train loss | Test accuracy |
|---|---|---|
| SGD (lr 0.5) | ≈ 0.029 | 0.947 |
| Adam (lr 0.02) | ≈ 0.024 | 0.933 |

Adam reached a lower training loss and smoother progress, while SGD scored
slightly higher on this small 150-sample test set. That gap is within noise, so
don't read it as one optimizer being better. Running `train.py` generates
`loss_curves.png` for comparison.

| Perceptron | AND | OR | XOR |
|---|---|---|---|
| Accuracy | 1.00 | 1.00 | 0.50 |

## Ideas to extend

- Add ReLU/softmax multi-class output and cross-entropy loss
- Add learning-rate decay or L2 regularization
- Serve the saved model behind a Flask/FastAPI endpoint


## Interactive web app (VS Code)

This project includes `app.py`, a small browser-based playground for the trained
two-moons neural network. It lets you change two input values, view the predicted
class and class-1 probability, and explore the decision boundary.

Predictions are also printed in the VS Code terminal each time you move either input slider.

### Run it on Windows

Open the **inner** `hack-o-week-nn` folder in VS Code (the folder containing
`train.py`, `deploy.py`, `app.py`, and the `nn` folder), then open Terminal:

```powershell
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open that URL in
your browser. Keep the terminal running while you use the app; press `Ctrl+C` in
the terminal to stop it.

If the browser does not open automatically, copy the `Local URL` printed in the
terminal into your browser. Run `python train.py` first to generate a fresh
`model.joblib` using this project's `nn` module.

**Security:** only load model files you trust. Pickle/joblib model files can
execute code during loading.
