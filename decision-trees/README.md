# Decision Trees in Classification & Regression

Decision trees are a type of **supervised learning algorithm** used for both classification and regression tasks. They work by recursively splitting the dataset into smaller subsets based on feature values, creating a tree-like model that resembles a flowchart with decision nodes and leaf nodes.

## Mathematical Formulation of Decision Trees

Let's assume that we have training vectors $x_i \in \mathbb{R}^n$ where $i = 1, 2, \ldots, l$ and a label vector $y \in \mathbb{R}^l$. A decision tree recursively partitions the feature space such that samples with the same labels or similar target values are grouped together.

The algorithm seeks to minimize following expression:

$$
J(Q_m, \theta) = \frac{n_m^{left}}{n_m} H(Q_m^{left}(\theta)) + \frac{n_m^{right}}{n_m} H(Q_m^{right}(\theta))
$$

where $H(\cdot)$ is a cost function (e.g., Gini impurity or entropy for classification, MSE for regression), $\theta = (k, t_k)$ is a candidate split consisting of a feature $k$ and a threshold $t_k$. The set $Q_m$ represents the samples at node $m$ with $n_m$ total samples. Each split divides the data into left and right subsets, $Q_m^{left}(\theta)$ and $Q_m^{right}(\theta)$, respectively.


