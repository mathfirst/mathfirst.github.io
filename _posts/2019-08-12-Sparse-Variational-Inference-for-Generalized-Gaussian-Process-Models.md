---
layout: post
title: Sparse Variational Inference for Generalized Gaussian Process Models - Tutorial 4
description: This is a tutorial for this ICML 2015 paper 'Sparse Variational Inference for Generalized Gaussian Process Models'. It covers how to calculate the predictive distributions.
date: 2019-08-12
---
<p>
In this continued blog, we will talk about how to calculate the predictive distributions. Also, this part is not available in the original paper. Therefore, if we would like to reproduce the experiments, we have to derive the prediction density by ourselves.
</p>

### Calculating the predictive distributions

<p>
Consider a data vector $\mathbf{y}$, where each entry $y_i$ is a noisy observation of the function $f(\mathbf{x}_i)$, for all the points $\mathbf{X}=\left\{\mathbf{x}_{i}\right\}_{i=1}^{n}$. We consider the noise to be independent Gaussian with variance $\sigma^2$. Introducing a Gaussian process prior over $f(\cdot)$, let the vector $f$ contain values of the function at the points $\mathbf{X}$. We shall also introduce a set of inducing variables: let the vector $\boldsymbol{f_u}$ contain values of the function $f$ at the points $\mathbf{Z}=\left\{\mathbf{z}_{i}\right\}_{i=1}^{m}$ which live in the same space as $\mathbf{X}$. Using the standard Gaussian process methodologies, we can write
</p>

$$
p(\mathbf{y} | \boldsymbol{f})=\mathcal{N}\left(\mathbf{y} | \boldsymbol{f}, \sigma^{2} \mathbf{I}\right)
\tag{1}\label{noise}
$$

$$
p(\boldsymbol{f} | \boldsymbol{f_u})=\mathcal{N}\left(\boldsymbol{f} | \mathbf{K}_{n m} \mathbf{K}_{m m}^{-1} \boldsymbol{f_u}, \widetilde{\mathbf{K}}\right)
$$

$$
p(\boldsymbol{f_u})=\mathcal{N}\left(\boldsymbol{f_u} | \mathbf{0}, \mathbf{K}_{m m}\right)
$$

<p>
where $\mathbf{K}_{nm}$ is the covariance function between training points and inducing points and we have defined $\widetilde{\mathbf{K}}=\mathbf{K}_{n n}-\mathbf{K}_{nm} \mathbf{K}_{mm}^{-1} \mathbf{K}_{mn}$.
</p>

<p>
In our case, after obtaining the optimal variational parameters $\boldsymbol{m}$ and $\boldsymbol{V}$ via optimizing VLB, we get $q(\boldsymbol{f_u})$ which denotes the marginal distribution of inducing function wrt approximate posterior $q(\boldsymbol{f},\boldsymbol{f_u})$
</p>

$$ 
q(\boldsymbol{f_u})=\mathcal{N}(\boldsymbol{f_u} | \boldsymbol{m}, \boldsymbol{V})
$$

<p>
To make a prediction, we need to integrate
</p>

$$ 
p\left(\boldsymbol{f}_{\star}\right)=\int p\left(\boldsymbol{f}_{\star} | \boldsymbol{f_u}\right) q(\boldsymbol{f_u}) \mathrm{d} \boldsymbol{f_u}
$$

where

$$ 
p\left(\boldsymbol{f}_{\star} | \boldsymbol{f_u}\right)=\mathcal{N}\left(\boldsymbol{f}_{\star} | \mathbf{K}_{\star m} \mathbf{K}_{mm}^{-1} \boldsymbol{f_u}, \mathbf{K}_{\star \star}-\mathbf{K}_{\star m} \mathbf{K}_{mm}^{-1} \mathbf{K}_{m \star}\right)
$$

After integrating out $\boldsymbol{f_u}$, we get

$$
\begin{aligned}
p\left(\boldsymbol{f}_{\star}\right)&=\mathcal{N}\left(\boldsymbol{f}_{\star} | \mathbf{K}_{\star m} \mathbf{K}_{mm}^{-1} \boldsymbol{m}, \mathbf{K}_{\star \star}+\mathbf{K}_{\star m} \mathbf{K}_{mm}^{-1} (\mathbf{V}-\mathbf{K}_{mm})\mathbf{K}_{mm}^{-1} \mathbf{K}_{m\star}\right)\\
&=\mathcal{N}\left(\boldsymbol{f}_{\star} | \mathbf{A} \boldsymbol{m}, \mathbf{K}_{\star \star}+\mathbf{A} (\boldsymbol{V}-\mathbf{K}_{mm})\mathbf{A}^{\textbf{T}}\right)=\mathcal{N}\left(\boldsymbol{f}_{\star} | \boldsymbol{\mu_*}, \boldsymbol{V_*}\right)
\end{aligned}\tag{2}\label{marginal-f_star}
$$

<p>
where $\mathbf{A}$ denotes $\mathbf{K}_{\star m} \mathbf{K}_{mm}^{-1}$.
</p>

To get the predictive distribution, we need to calculate the following integral

$$
    p(y|x_*)=\int p(f_{*} |x_{*},\boldsymbol{m},\boldsymbol{V})p(y|x_*,f_*)df_*
    \tag{3}\label{pred-y}
$$

#### Standard GP regression

<p>
For the standard GP regression, we use the Gaussian likelihood, i.e. \eqref{noise}, combining with Eq. \eqref{marginal-f_star} and Eq. \eqref{pred-y}.
</p>

$$ \label{pred-y-Gauss}
p(\boldsymbol{y})=\mathcal{N}\left(\boldsymbol{y} ; \mathbf{A} \boldsymbol{m}, \mathbf{K}_{\star \star}+\mathbf{A} (\boldsymbol{V}-\mathbf{K}_{mm})\mathbf{A}^{\textbf{T}}+\sigma^{2} \mathbf{I}\right)=\mathcal{N}(\boldsymbol{y}; \boldsymbol{\mu_*}, \boldsymbol{V_*}+\sigma^{2} \mathbf{I})
$$

#### Count Regression

<p>
For the count regression case, we use Poisson likelihood. Here we consider one test instance
</p>

$$\label{Poisson-lik}
    p(y|x_*,f_*)=\frac{1}{y!} e^{-e^{f_*}} e^{f_* y}
$$

$$
p(y|x_*)=\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2 \pi v_*}} \exp \left(-\frac{(f_*-\mu_*)^{2}}{2 v_*}\right) \frac{1}{y!} e^{-e^{f_*}} e^{f_* y} df_*
$$

<p>
We need to integrate $f_*$ out, but the analytical form is not available. So we turn to Gauss–Hermite quadrature. Firstly, we change variables as follows
</p>

$$
z_*=\frac{f_*-\mu_*}{\sqrt{v_*}} \Leftrightarrow f_*=\sqrt{v_*} z_*+\mu_*
$$

<p>
we get
</p>

$$
p(y|x_*)=\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi}} \exp \left(-\frac{z_*^{2}}{2}\right) \frac{1}{y!} e^{-e^{\sqrt{v_*} z_*+\mu_*}} e^{(\sqrt{v_*} z_*+\mu_*) y} dz_*
$$

<p>
leading to
</p>

$$
p(y|x_*) \approx \frac{1}{\sqrt{2\pi}} \sum_{j=1}^{n} w_{j}\frac{1}{y!}  \exp\left(-e^{\sqrt{v_*} z_j+\mu_*}+(\sqrt{v_*} z_j+\mu_*) y\right)
$$

#### Binary Classification

<p>
For the binary classification case, we use sigmoid function $\sigma(f)$ as our likelihood function. Here we consider one test instance
</p>

$$
\begin{aligned}
p\left(y_{*}=1 | \mathbf{x}_{*}, \boldsymbol{m}, \boldsymbol{V}\right)&= \int \sigma\left(f_{*}\right) p\left(f_{*} | \mathbf{x}_{*}, \boldsymbol{m}, \boldsymbol{V}\right) d f_{*}\\
&\approx\frac{1}{\sqrt{2\pi}} \sum_{j=1}^{n} w_{j}\sigma(\sqrt{v_*} z_j+\mu_*)
\end{aligned}
$$

<p>
As the probability of the two classes must sum to $1$, we have $p(y=-1 | \mathbf{x}, \mathbf{w})=1-p(y=+1 | \mathbf{x}, \mathbf{w})$. Besides, $\sigma(-z)=1-\sigma(z)$. Hence, the above fomula can be written more consicely as
</p>

$$
\begin{aligned}
p\left(y_{*} | \mathbf{x}_{*}, \boldsymbol{m}, \boldsymbol{V}\right) &= \int \sigma\left(y_{*}f_{*}\right) p\left(f_{*} | \mathbf{x}_{*}, \boldsymbol{m}, \boldsymbol{V}\right) d f_{*}\\
&\approx\frac{1}{\sqrt{2\pi}} \sum_{j=1}^{n} w_{j}\sigma\left(y_{*}(\sqrt{v_*} z_j+\mu_*)\right)
\end{aligned}
$$

<p>
In the next blog, we'll talk about the details of the implementation and experiments.
</p>