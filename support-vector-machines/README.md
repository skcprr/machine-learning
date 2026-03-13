# Support Vector Machines (SVM)

**Support Vector Machines** are a set of supervised learning methods used for classification and regression tasks.

## Linear SVM

Let's assume that the training dataset consists of $n$ points of the form:

$$
(\mathbf{x_1}, y_1), (\mathbf{x_2}, y_2) \ldots, (\mathbf{x_n}, y_n),
$$

where $y_i$ can be either $1$ or $-1$ depending on the class of the $\mathbf{x_i}$. Task is to find a hyperplane which separates the group of points $\mathbf{x_i}$ with $y_i = 1$ from group of points with $y_i = -1$. Distance between nearest points from each group to the hyperplane is set to maximum.

The hyperplane can be formulated as:

$$
\mathbf{w}^T\mathbf{x} + b = 0,
$$

where $\mathbf{w}$ is a vector normal to the hyperplane, $b$ is an offset parameter describing distance between the hyperplane and the origin of the coordinate system.

### Hard-Margin

If the training data is linearly separable, two hyperplanes can be drawn, which separate the data points in such a way that the distance between them is maximum. The area  constrained by these hyperplanes are called the margin. Those hyperplanes can be written as:

$$
\begin{align*}
\mathbf{w}^T\mathbf{x} + b &= 1 \\
\mathbf{w}^T\mathbf{x} + b &= -1
\end{align*}
$$

The distance between them is $\frac{2}{\left\lVert \mathbf{x} \right\rVert^2}$ so, in order to maximize the distance $\left\lVert \mathbf{x} \right\rVert^2$ has to be minimized. To ensure that no data points lie within the margin, the following constraint must be satisfied:

$$
y_i\left(\mathbf{w}^T\mathbf{x_i} + b \right)\geq 1, \forall i\in \{1,\ldots, n\}.
$$

So the optimization task can be formulated as:

$$
\underset{\textbf{w},b}{\min} 
\{ \frac{1}{2}\left\lVert \mathbf{x} \right\rVert^2\} \\
\text{subject to:} \\ y_i\left(\mathbf{w}^T\mathbf{x_i} + b \right)\geq 1, \forall i\in \{1,\ldots, n\}.
$$

### Soft-Margin

In the cases where the traning data is not linearly separable the parameter $\zeta$ is added. This parameter defines degree of acceptable violation of the margin by $\mathbf{x_i}$ point. Then optimization problem is:

$$
\underset{\textbf{w},b, \zeta}{\min} 
\{ \frac{1}{2}\left\lVert \mathbf{x} \right\rVert^2 + C\sum_i^n\zeta_i \} \\
\text{subject to:} \\ y_i\left(\mathbf{w}^T\mathbf{x_i} + b \right)\geq 1 - \zeta_i,  \zeta_i \geq 0,  \forall i\in \{1,\ldots, n\}.
$$

Hyperparameter $C$ defines trade-off between increasing the margin width and ensuring that $\mathbf{x_i}$ lies on the correct side of the margin.
