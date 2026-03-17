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
\{ \frac{1}{2}\left\lVert \mathbf{w} \right\rVert^2\} \\
\text{subject to:} \\ y_i\left(\mathbf{w}^T\mathbf{x_i} + b \right)\geq 1, \forall i\in \{1,\ldots, n\}.
$$

### Soft-Margin

In the cases where the traning data is not linearly separable the parameter $\zeta$ is added. This parameter defines degree of acceptable violation of the margin by $\mathbf{x_i}$ point. Then optimization problem is:

$$
\underset{\textbf{w},b, \zeta}{\min} 
\{ \frac{1}{2}\left\lVert \mathbf{w} \right\rVert^2 + C\sum_i^n\zeta_i \} \\
\text{subject to:} \\ y_i\left(\mathbf{w}^T\mathbf{x_i} + b \right)\geq 1 - \zeta_i,  \zeta_i \geq 0,  \forall i\in \{1,\ldots, n\}.
$$

Hyperparameter $C$ defines trade-off between increasing the margin width and ensuring that $\mathbf{x_i}$ lies on the correct side of the margin.

As shown in the figure, bigger value of hyperparameter $C$ decreases the margin, while for small values of $C$, the margin is wider.

<img width="969" height="448" alt="Image" src="https://github.com/user-attachments/assets/86271a93-0ebe-48b5-8259-bc3fa21fec58" />

## Nonlinear SVM

It is often the case that the data is not linearly separable in such a way that a linear hyperplane fails regardless of the margin settings.

The main idea is to map original data from lower-dimensional space into a higher-dimensional space where a linear separation is possible.
Let's define a mapping function $\phi(\mathbf{x})$, the optimization problem involves the dot product of the transformed vectors:  

$$
\phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j),
$$

but calculating these transformations explicitly could be computationally expensive.


### Kernel Trick

The **Kernel Trick** allows the calculation of the dot product in the high-dimensional space without performing the transformation $\phi$.
The dot product is replaced with a **Kernel Function $K(x_i, x_j)$**:

$$
K(\mathbf{x_i},\mathbf{x_j}) = \phi(\mathbf{x}_i)^T\phi(\mathbf{x}_j).
$$

Some common kernels include:

- ___Polynomial___: $K(\mathbf{x_i},\mathbf{x_j}) = (\mathbf{x_i}\cdot\mathbf{x_j} + r)^d$, when $d=1$, this become the linear kernel,
- ___Gaussian Radial Basis Function___: $K(\mathbf{x_i},\mathbf{x_j}) = \exp{(-\gamma\left\lVert \mathbf{x_i} - \mathbf{x_j} \right\rVert^2)}$, for $\gamma >0$,
- ___Sigmoid Function___: $K(\mathbf{x_i},\mathbf{x_j}) =\tanh{\left(\kappa\mathbf{x}_i\cdot\mathbf{x_j} + c\right)}$, for $\kappa > 0$ and $c <0$.

As shown in the figure, these kernel function produce different decision boundaries:

<img width="892" height="800" alt="Image" src="https://github.com/user-attachments/assets/a7e4dbc3-acee-4686-8463-cf7de170c06f" />

## Regression

Support Vector Machines can also be applied in regression problems. In this context, the goal is to ensure that as many data points as possible lie within the margin. The width of the margin is controlled by the hyperparameter $\varepsilon$. The  optimization problem is formulated as:

$$
\min\{ \frac{1}{2}\left\lVert \mathbf{w} \right\rVert^2\} \\
\text{subject to:} \|\mathbf{y}_i - \mathbf{w}_i\mathbf{x}_i - b \| \leq \varepsilon, \text{ }\forall i \in \{1, \ldots, n\}.
$$

The figure below shows  linear regression using SVM:

<img width="535" height="432" alt="Image" src="https://github.com/user-attachments/assets/e19770da-c6c8-4755-b756-83910bddac93" />

And the figure below illustrates nonlinear regression:

<img width="535" height="432" alt="Image" src="https://github.com/user-attachments/assets/dab45d24-b720-42ec-a3c0-2bac7af5c88e" />
