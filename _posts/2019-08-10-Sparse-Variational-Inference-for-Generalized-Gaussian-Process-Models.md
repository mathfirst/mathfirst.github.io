---
layout: post
title: Sparse Variational Inference for Generalized Gaussian Process Models - Tutorial 2
description: This is a tutorial for this ICML 2015 paper 'Sparse Variational Inference for Generalized Gaussian Process Models'. It covers fixed point method, stochastic variational inference and some experiments.
date: 2019-08-10
---
<p>
In this continued blog, we will talk about VLB optimization, including calculating the gradients of the VLB and the fixed point method. Aditionally, we'll talk a little about stochastic variational inference.
</p>

### Calculating the gradients of the VLB
<p>
Now we have the VLB handy. Our goal is to optimize variational parameters, so we need to calculate the gradients of the VLB w.r.t $\boldsymbol{m}$ and $\boldsymbol{V}$. In the first paper, the chain rule is employed when the gradients of the log likelihood expectation term are calculated. Specifically, 
</p>

$$
\frac{\partial \mathrm{VLB}}{\partial \boldsymbol{m}}=\frac{\partial VLB}{\partial m_{q_{i}}}\frac{\partial m_{q_{i}}}{\partial \boldsymbol{m}}
$$

$$
\frac{\partial \mathrm{VLB}}{\partial \boldsymbol{V}}=\frac{\partial VLB}{\partial \sqrt{v_{q_{i}}}}\frac{\partial(\sqrt{v_{q_{i}}})}{\partial \boldsymbol{V}}
$$

<p>
The paper has put the first step of the chain rule in detail. One $\frac{1}{\sqrt{2\pi}}$ is missing in Eq. (7) of the original paper, but it does not affect the final outcome as the missing term is absorbed into the subscript of the expectation in the last step of Eq. (8). The second step can be obtained easily via Eq. (2a) and (2b) in <a href="https://kaikaizhao.github.io/notes/2019/08/09/Sparse-Variational-Inference-for-Generalized-Gaussian-Process-Models" target="_blank">Tutorial 1</a>.
</p>

<p>
After calculating the derivatives of KL term, we put together the derivatives of the two parts of VLB. Then we get the derivatives of the VLB,
</p>

$$
    \frac{\partial \mathrm{VLB}}{\partial \boldsymbol{m}}=\sum_{i}\left(\rho_{i} K_{M}^{-1} K_{M i}\right)-K_{M}^{-1}\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)\tag{1}\label{de-m}
$$

$$
\frac{\partial \mathrm{VLB}}{\partial V}=\frac{1}{2} \sum_{i}\left(\lambda_{i} K_{M}^{-1} K_{M i} K_{i M} K_{M}^{-1}\right)+\frac{1}{2}\left(V^{-1}-K_{M}^{-1}\right)\tag{2}\label{de-V}
$$

<p>where $\rho_i$ and $\lambda_i$ are derived from the first step of chain rule when we calculate the derivatives of log likelihood expectation. They are expectations of the first derivatives and second derivatives of log likelihood. These are available in the Table 1 of the original paper. In the next article we will continue to talk about this topic.</p>

### VLB optimization
<p>
By first-order optimality, the optimal variational parameters can be found via the conditions $\left.\frac{\partial \mathrm{VLB}}{\partial \boldsymbol{m}}\right|_{\boldsymbol{m}=\boldsymbol{m}^{*}}=0$ and $\frac{\partial \mathrm{VLB}}{\partial V} |_{V=V^{\star}}=0$.
</p>

$$
\boldsymbol{m}^{\star}=K_{M N} \boldsymbol{\rho}^{\star}+\boldsymbol{m}_{\mathcal{U}}
\tag{3a}\label{sol-m}
$$

$$
V^{\star}=\left(K_{M}^{-1}-K_{M}^{-1} K_{M N} \operatorname{diag}\left(\lambda^{\star}\right) K_{N M} K_{M}^{-1}\right)^{-1}\tag{3b}\label{sol-V}
$$

<p>
In general, \eqref{sol-m} and \eqref{sol-V} are a set of nonlinear equations coupled through their dependencies on $\boldsymbol{\rho}$ and $\lambda$, except Gaussian likelihood. For the case of count regression, when we calculate $\boldsymbol{m}$, we need to know $\boldsymbol{\rho}$ that is dependent on $m$ and $v$ from $\rho=-e^{m+\frac{1}{2} v}+y$. From (2a) and (2b) in <a href="https://kaikaizhao.github.io/notes/2019/08/09/Sparse-Variational-Inference-for-Generalized-Gaussian-Process-Models" target="_blank">Tutorial 1</a>, $m$ and $v$ are dependent on $\boldsymbol{m}$ and $\boldsymbol{V}$. Therefore, they are coupled, that is to say, if you want to compute $\boldsymbol{m}$, you have to know $\boldsymbol{m}$ and $\boldsymbol{V}$. So gradient ascent is a standard approach to solving this problem.
</p>
<p>
For the covariance, we optimize the Cholesky factor $L$ of $V=LL^T$ instead of $V$ directly, which guarantees that $V$ is positive-definite. In our case the gradient is
</p>

$$
\frac{\partial \mathrm{VLB}}{\partial L}=\sum_{i}\left(\lambda_{i} K_{M}^{-1} K_{M i} K_{i M} K_{M}^{-1} L\right)+\left(L^{-1^{T}}-K_{M}^{-1} L\right)\tag{4}\label{L}
$$

### Fixed point method

<p>
First of all, we introduce the description of <b>fixed-point theorem</b> from <a href="https://en.wikipedia.org/wiki/Fixed-point_iteration" target="_blank">WIKIPEDIA</a>.
</p>

> If a function $f$ defined on the real line with real values is Lipschitz continuous with Lipschitz constant $L<1$, then this function has precisely one fixed point, and the fixed-point iteration converges towards that fixed point for any initial guess $x_{0}$. This theorem can be generalized to any metric space.

<p>
The first paper proposed to optimize the covariance via a fixed-point operator, $T$:
</p>

$$
    T(V)=\left(K_{M}^{-1}-K_{M}^{-1} K_{M N} \operatorname{diag}(\boldsymbol{\lambda}) K_{N M} K_{M}^{-1}\right)^{-1}\tag{5}\label{fp-V}
$$

<p>
The above formula exactly applies fixed-point iteration to \eqref{sol-V}. The critical condition is Lipschitz constant $ L<1 $, i.e. for all $U$, $V$
</p>

$$
\|T(V)-T(U)\| \leq L\|V-U\|
$$

<p>
In the second paper, the authors try to build a connection between fixed-point update between natural gradients.
</p>

<p>
Recall that for the Gaussian distribution in the form of exponential family, the natural parameters
</p>

$$
\theta=\left[\begin{array}{c}{\frac{\mu}{\sigma_{1}^{2}}} \\ {\frac{1}{2 \sigma^{2}}}\end{array}\right]=\left[\begin{array}{c}{V^{-1} \boldsymbol{m}} \\ {\frac{1}{2} V^{-1}}\end{array}\right]
$$

<p>
Here we omit some derivations which can be found in the appendix of the second paper. And we directly present the fixed-point update for the mean and the covariance. Fundamentally, the following two equations use natural gradients to update natural parameters.
</p>

$$
V^{-1} \boldsymbol{m} \leftarrow \Sigma^{-1} \mu+\sum_{i}\left(\rho_{i}+\left(\boldsymbol{m}^{T} d_{i}\right) \gamma_{i}\right) d_{i}\tag{6a}\label{nat-Vm}
$$

$$
\frac{1}{2} V^{-1} \leftarrow \frac{1}{2} \Sigma^{-1}+\sum_{i} \frac{1}{2} \gamma_{i} d_{i} d_{i}^{T}
\tag{6b}\label{nat-V}
$$

<p>
where $\gamma_i=-\lambda_i$, $d_i=K_M^{-1}K_{Mi}$, so we can vectorize the update formula for $\boldsymbol{m}$ as
</p>

$$
    \boldsymbol{m}\leftarrow \boldsymbol{V}\boldsymbol{d}(\mathbf{\rho}+\boldsymbol{d}^T\boldsymbol{m}\odot\mathbf{\gamma})\tag{6c}\label{nat-m}
$$

<p>
where $\boldsymbol{d}=K_M^{-1}K_{Mx}$.
</p>

### Optimization Strategies

<p>
In this section, we talk about four methods to optimize variational parameters, i.e. $\boldsymbol{m}$ and $\boldsymbol{V}$.
</p>

#### Gradient Descent

<p>
The first strategy is to optimize $(\boldsymbol{m},\boldsymbol{V})$ by coordinate ascent across parameters. Specifically, we alternately update $\boldsymbol{m}$ and $\boldsymbol{V}$ via the corresponding gradients, i.e. \eqref{de-m} and \eqref{de-V}, with an appropriate learning rate. Although our problem is to maximize VLB, we minimize negative VLB in our implementation and hence we call it gradient descent.
</p>

#### FPbatch(FPb)

<p>
FPb takes gradient steps for $\boldsymbol{m}$ until there is no change in VLB or it takes FP update for $\boldsymbol{V}$ until no change in VLB as well. FP update means updating $\boldsymbol{V}$ via \eqref{fp-V}.
</p>

#### FPincremental(FPi)

<p>
FPi alternates between taking one gradient step for $\boldsymbol{m}$ and one fixed point step for $\boldsymbol{V}$ until no change in VLB.
</p>

#### SVI

<p>
The coordinate ascent algorithm is inefficinet for large data sets. The fourth paper borrows the idea of stochastic optimization from <a href="http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf" target="_blank">Stochastic Variational Inference</a>. The update formulae are almost the same to \eqref{nat-Vm} and \eqref{nat-V}. More specifically, it subsamples the data to form noisy estimates of the natural gradient of the ELBO, and it follows these estimates with a decreasing step-size. Besides, another difference is that a step size $\rho_t$ is added.
</p>

To make the formulae consistent with the preceding FP formulae, we changed the Eq. (24) and (25) in the fourth paper slightly.

$$
V^{-1} \leftarrow(1-\rho_t) V^{-1}+\rho_t\left(\Sigma^{-1}+ \frac{N}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \hat{\gamma}_{i} d_{i} d_{i}^{T}\right)\tag{7a}\label{SVI-V}
$$

$$
V^{-1} m \leftarrow(1-\rho_t) V^{-1} m+\rho_t\left(\Sigma^{-1} \mu+\frac{N}{|\mathcal{M}|} \sum_{i \in \mathcal{M}}\left(\hat{\rho}_{i}+\left(m^{T} d_{i}\right) \hat{\gamma}_{i}\right) d_{i}\right)\tag{7b}\label{SVI-m}
$$

<p>
where the last terms of both \eqref{SVI-V} and \eqref{SVI-m} represent a stochastic gradient estimated by sampling a mini-batch $\mathcal{M}$ uniformly at random from a dataset of size $N$.
</p>

<p>
In the next blog, we'll talk about how to optimize hyperparameters, i.e. kernel parameters, including the length scale and the vertical scale.
</p>