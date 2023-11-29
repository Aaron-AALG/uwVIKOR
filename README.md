# Unweighted VIKOR method

![python-version](https://img.shields.io/badge/python->=3.8-orange.svg)
[![pypi-version](https://img.shields.io/pypi/v/uwvikor.svg)](https://pypi.python.org/pypi/uwvikor/)
![license](https://img.shields.io/pypi/l/uwvikor.svg)
[![Downloads](https://static.pepy.tech/personalized-badge/uwvikor?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads)](https://pepy.tech/project/uwvikor)

The Unweighted VIKOR method (UW-VIKOR) is a multiple-criteria decision-making (MCDM) technique for ranking alternatives based on the classical VIKOR (_VIseKriterijumska Optimizacija I Kompromisno Resenje_) approach, however this method does not require the introduction of a priori weighting scheme. Instead, the decision-maker only has to determine the bounds $\{(l_j,u_j)\}_{j=1}^{M}$ between which the weights vary, thus reducing the demands on the algorithm.

As a consequence, the weights are considered decision variables in a non-linear mathematical optimization problem that considers the VIKOR $Q$-score as the objective function. The optimization yields two scores subjected to maximize ($Q_{i}^{L}$) and minimize ($Q_{i}^{U}$) the $Q$-score per each alternative as well as the optimal weights attached to them ($W_{i}^{L}$ and $W_{i}^{U}$). Then, per each alternative, we get the score intervals $[S_{i}^{L}$, $S_{i}^{U}]$ and $[R_{i}^{L}, R_{i}^{U}]$ by evaluating them with the optimal weights. Finally, we can rank alternatives using an aggregation function of the $[Q_{i}^{L}, Q_{i}^{U}]$ scores.

## Installation

You can install the uwVIKOR library from GitHub:

```terminal
    git clone https://github.com/Aaron-AALG/uwVIKOR.git
    python3 -m pip install -e uwVIKOR
```

You can also install it directly from PyPI:

```terminal
    pip install uwVIKOR
```

## Input parameters and Output

### Input

* **data**: dataframe which contains the alternatives and the criteria.
* **directions**: array with the optimal direction of the criteria.
* **L**: array with the lower bounds of the weights.
* **U**: array with the upper bounds of the weights.
* **v**: value of the utility parameter (By default v = 0.5).
* **w0**: array with the initial guess of the weights (By default w0 = []).
* **display**: logical argument to indicate whether to show print convergence messages or not (By default display = False).

### Output

Dictionary which contains three keys:

* **Ranking**: List with $S$, $R$ and $Q$ scores in regard to the optimal weights.
* **Weights_min**: List with the weights that minimize the $Q$ score.
* **Weights_max**: List with the weights that maximize the $Q$ score.

## Example

The `uwVIKOR` function is implemented in order to manage decision matrices as input data which will be converted to **NumPy** arrays. Here is an example in which three alternatives and four criteria are used::

```python
    import pandas as pd
    import numpy as np
    from uwVIKOR.uwVIKOR import *

    data = pd.DataFrame({"c1":[173, 176, 142],
                        "c2":[10, 11, 5],
                        "c3":[11.4, 12.3, 8.2],
                        "c4":[10.01, 10.48, 7.3]})
    directions = ["max", "max", "min", "min"]
    L = np.array([0.1 for _ in range(data.shape[1])])
    U = np.array([0.4 for _ in range(data.shape[1])])
    v = 0.75

    x = uwVIKOR(data, directions, L, U, v)
```

### Generalization to classic VIKOR

Given that uwVIKOR generalizes VIKOR, we can also compute it by limiting the amplitude of the boundaries. For this function, it is recommended to use the Numpy numerical epsilon as the difference between $L$ and $U$. Here is an example:

```python
weights = np.array([0.25, 0.2, 0.2, 0.35])
epsilon = np.finfo(float).eps

try:
  x = uwVIKOR(data,
               directions, 
               weights, 
               weights + epsilon, 
               )
except:
  x = uwVIKOR(data,
               directions, 
               weights - epsilon, 
               weights, 
               )
```

## Optimization in Python

This library uses the [minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) function of the `scipy.optimize` module to carry out the optimization problems. In particular, $Q_{i}^{L}$ and $Q_{i}^{U}$ are obtained one by one, thus we can apply the **SLSQP** method.
