Unweighted VIKOR method
=======================

The Un-Weighted Technique for Order Preference by Similarity to Ideal Solution (uwVIKOR) ranks decision alternatives based on the classical VIKOR approach, however this method does not require the introduction of a priori weights.

As a consequence of working with unknown weights, the method does not take into account the relative importance of criteria. Then, the positive ideal solution (PIS) and a negative ideal solution (NIS) varies depending on the conditions of problem. Hence, the functions of relative proximity (Q) is an operator which are optimized as two mathematical programming problems of maximize (Q_L) and minimize (Q_U), considering weights as variables. Finally, per each alternative, we get the intervals [Q_L, Q_U], and so [S_L, S_U] and [R_L, R_U], hence we can rank them in accordance with a determined comparison method.

Installation
======================

You can install the uwVIKOR library from GitHub::

    git clone https://github.com/Aaron-AALG/uwVIKOR.git
    python3 -m pip install -e uwVIKOR


You can also install it directly from PyPI::

    pip install uwVIKOR


Input-Output
======================

Input
------

* **data**: dataframe which contains the alternatives and the criteria.
* **directions**: array with the optimal direction of the criteria.
* **L**: array with the lower bounds of the weigths.
* **U**: array with the upper bounds of the weigths.
* **v**: value of the utility parameter. (By default v = 0.5)
* **w0**: array with the initial guess of the weights. (By default w0 = [])
* **display**: logical argument to indicate whether to show print convergence messages or not. (By default display = False)

Output
------

Dictionary which contains three keys:

* **Ranking**: List with S, R and Q scores in regard of the optimal weights.
* **Weights_min**: List with the weights that minimizes the Q score.
* **Weights_max**: List with the weights that maximizes the Q score.

Example
======================

uwVIKOR is implemented in order to manage **Pandas** DataFrames as input data which will be converted to **NumPy** arrays. Here is an example in which we only use three alternatives and four criteria::

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


Optimization in Python
======================

This library uses the `minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`_ function of the scipy.optimize module to carry out the optimization problems. In particular, Q_L and Q_U are obtained one by one, thus we can apply the **SLSQP** method.
