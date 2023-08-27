---
layout: post
title: Sparse Variational Inference for Generalized Gaussian Process Models - Tutorial 1
description: This is a tutorial for this ICML 2015 paper 'Sparse Variational Inference for Generalized Gaussian Process Models'. It covers fixed point method, stochastic variational inference and some experiments.
date: 2019-08-09
---

<p>In Summer 2019 semester I was honored to work with Professor Roni Khardon for my independent study. Thank Lijiang Guo, Weizhe Chen and Yadi Wei for helpful discussions.</p>

<p>
During this summer I worked on the following papers. I spent some time understanding the algorithms proposed in the papers and implementing some of them. However, due to page limit the derivations in these papers are brief, hence we had to complete the unpublished derivation by ourselves. Therefore, we will try to present a full picture of the first paper.
</p>

<ol>
    <li>Rishit Sheth, Yuyang Wang, and Roni Khardon. <a href="http://homes.sice.indiana.edu/rkhardon/PUB/icml15sparseFPGP.pdf" target="_blank">Sparse variational inference for generalized gaussian process</a>. ICML 2015.</li>
    <li>Rishit Sheth and Roni Khardon. <a href="http://proceedings.mlr.press/v51/sheth16.pdf" target="_blank">A Fixed-Point Operator for Inference in Variational Bayesian Latent Gaussian Models</a>. AISTATS 2016.</li>
    <li>Matthew D. Hoffman, David M. Blei, Chong Wang and John Paisley. <a href="http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf" target="_blank">Stochastic Variational Inference</a>. Journal of Machine Learning Research 14(2013).</li>
    <li>Rishit Sheth and Roni Khardon. <a href="https://arxiv.org/abs/1612.03957" target="_blank">Monte Carlo Structured SVI for Two-Level Non-Conjugate Models</a>. arXiv:1612.03957.</li>
</ol>
<!-- , but we will exactly explain the ideas in the fist paper, fixed point method, and the connection between FP method and SVI -->
<p>
In this tutorial we cannot cover all the ideas and details in the above papers. The main topic is sparse variational inference for generalized gaussian process models. To explain this main topic better, we will also include some ideas from other papers where appropriate, such as the fixed point method, SVI and their connection as well.
</p>

<p>
We will assume prior knowledge of (sparse) Gaussian process and variational inference.
</p>

## Variational Lower Bound for Sparse Gaussian Processes

<p>
For sparse variational Gaussian Processes, we follow <a href="http://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf" target="_blank">Titsias 2009</a>. We first need to select the inducing inputs $X_m$, and $f$ and $f_m$ denote the training latent function values and the inducing variables, respectively. Then, the initial joint model is
</p>

$$
    p\left(\mathbf{y}, \mathbf{f}, \mathbf{f}_{m}\right)=p(\mathbf{y} | \mathbf{f}) p\left(\mathbf{f} | \mathbf{f}_{m}\right) p\left(\mathbf{f}_{m}\right)
$$

<p>
where the conditional GP prior is $p\left(\mathbf{f} | \mathbf{f}_{m}\right)=N\left(\mathbf{f} | K_{n m} K_{m m}^{-1} \mathbf{f}_{m}, K_{n n}-K_{n m} K_{m m}^{-1} K_{m n}\right)$. Then the true marginal likelihood can be written as
</p>

$$
    \log p(\mathbf{y})=\log \int p(\mathbf{y} | \mathbf{f}) p\left(\mathbf{f} | \mathbf{f}_{m}\right) p\left(\mathbf{f}_{m}\right) d \mathbf{f} d \mathbf{f}_{m}
$$

<p>
The posterior $p(\mathbf{f},\mathbf{f}_{m}|\mathbf{y})$ is approximated by the variational distribution
</p>

$$
q\left(\mathbf{f}, \mathbf{f}_{m}\right)=p\left(\mathbf{f} | \mathbf{f}_{m}\right) \phi\left(\mathbf{f}_{m}\right)
$$

<p>
where $\phi$ is a multivariate Gaussian distribution with mean $\boldsymbol{m}$ and covaraince $\boldsymbol{V}$ and these two parameters are exactly what we are going to optimize.
</p>

<p>
Applying Jensen's inequality we obtain
</p>

$$
\begin{aligned}
    \log p(\mathbf{y})&=\log \int p\left(\mathbf{f} | \mathbf{f}_{m}\right)\phi\left(\mathbf{f}_{m}\right)\cdot \frac{p(\mathbf{y} | \mathbf{f}) p\left(\mathbf{f} | \mathbf{f}_{m}\right) p\left(\mathbf{f}_{m}\right)}{p\left(\mathbf{f} | \mathbf{f}_{m}\right)\phi\left(\mathbf{f}_{m}\right)}  d \mathbf{f} d \mathbf{f}_{m}\\
    &\ge \int p\left(\mathbf{f} | \mathbf{f}_{m}\right) \phi\left(\mathbf{f}_{m}\right) \log \frac{p(\mathbf{y} | \mathbf{f}) p\left(\mathbf{f} | \mathbf{f}_{m}\right) p\left(\mathbf{f}_{m}\right)}{p\left(\mathbf{f} | \mathbf{f}_{m}\right) \phi\left(\mathbf{f}_{m}\right)} d \mathbf{f} d \mathbf{f}_{m}\\
    &= \int p\left(\mathbf{f} | \mathbf{f}_{m}\right) \phi\left(\mathbf{f}_{m}\right) \log \frac{p(\mathbf{y} | \mathbf{f})  p\left(\mathbf{f}_{m}\right)}{ \phi\left(\mathbf{f}_{m}\right)} d \mathbf{f} d \mathbf{f}_{m}\\
    &= \int p\left(\mathbf{f} | \mathbf{f}_{m}\right) \phi\left(\mathbf{f}_{m}\right) \log p(\mathbf{y} | \mathbf{f}) d \mathbf{f} d \mathbf{f}_{m} + \int p\left(\mathbf{f} | \mathbf{f}_{m}\right) \phi\left(\mathbf{f}_{m}\right) \log \frac{p\left(\mathbf{f}_{m}\right)}{ \phi\left(\mathbf{f}_{m}\right)} d \mathbf{f} d \mathbf{f}_{m}\\
    &= \int p\left(\mathbf{f} | \mathbf{f}_{m}\right) \phi\left(\mathbf{f}_{m}\right) \log p(\mathbf{y} | \mathbf{f}) d \mathbf{f} d \mathbf{f}_{m} - \int  \phi\left(\mathbf{f}_{m}\right) \log \frac{\phi\left(\mathbf{f}_{m}\right)}{ p\left(\mathbf{f}_{m}\right)} d \mathbf{f}_{m}\\
    =& \int q\left(\mathbf{f}, \mathbf{f}_{m}\right) \log p(\mathbf{y} | \mathbf{f}) d \mathbf{f} d \mathbf{f}_{m} - \int  \phi\left(\mathbf{f}_{m}\right) \log \frac{\phi\left(\mathbf{f}_{m}\right)}{ p\left(\mathbf{f}_{m}\right)} d \mathbf{f}_{m}\\
    =&\int q\left(\mathbf{f}\right) \log p(\mathbf{y} | \mathbf{f}) d \mathbf{f} - \int  \phi\left(\mathbf{f}_{m}\right) \log \frac{\phi\left(\mathbf{f}_{m}\right)}{ p\left(\mathbf{f}_{m}\right)} d \mathbf{f}_{m}\\
    =&\int q\left(\mathbf{f}\right) \log p(\mathbf{y} | \mathbf{f}) d \mathbf{f} - \mathrm{KL}(\phi(\mathbf{f_m})\|p(\mathbf{f_m}))\\
\end{aligned}
$$

<p>
After we substitute the integral sign and $m$ with a sum sign and $\mathcal{U}$ respectively, we obtain the Eq. (2) in the first paper, which is
</p>

$$
    \log p(\boldsymbol{y}) \geq \sum_{i=1}^{N} \mathbb{E}_{q\left(f\left(\boldsymbol{x}_{i}\right)\right)}\left[\log p\left(y_{i} | f\left(\boldsymbol{x}_{i}\right)\right)\right] -\mathrm{KL}\left(\phi\left(\boldsymbol{f}_{\mathcal{U}}\right) \| p\left(\boldsymbol{f}_{\mathcal{U}}\right)\right)
    \tag{1}\label{VLB}
$$

<p>
where $q(f(x_i))$ is the marginal distribution of latent function values $f(x_i)$ w.r.t the approximate posterior $q(f_{\mathcal{X}},f_{\mathcal{U}})$. The right hand side of Eq.(\ref{VLB}) is the so-called <b>variational lower bound(VLB)</b>.
</p>
<p>
As $q(f_{\mathcal{X}},f_{\mathcal{U}})$ is a joint Gaussian distribution, the marginal distribution of $f(x_i)$ is specified by a univariate Gaussian with mean $m_q(x_i)$ and variance $v_q(x_i)$ where
</p>

$$
    m_{q}(\boldsymbol{x})=m(\boldsymbol{x})+K_{x M} K_{M}^{-1}\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)\tag{2a}\label{VLB-m}
$$

$$
    v_{q}(\boldsymbol{x})=k(\boldsymbol{x}, \boldsymbol{x})+K_{x M} K_{M}^{-1}\left(V-K_{M}\right) K_{M}^{-1} K_{M x}\tag{2b}\label{VLB-V}
$$

## Calculating VLB
In this section we first calculate the VLB and then we derive the gradients for variational parameters in next section. 
<p>
The VLB consists of two parts: the expectation of log likelihood and the KL divergence. Actually, we don't need to compute the VLB, because we can stop iterations when there are no changes for the variational parameters $\boldsymbol{m}$ and $\boldsymbol{V}$. However, we would like to observe the changes for the VLB, hence we will calculate it explicitly.
</p>

### Expectation of log likelihood
<p>
We use a change of variable to simply the calculation, that is, $f_i = z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}}$. Then the expectation of log likelihood has the following form,
</p>

$$\label{VLB-1}
    \mathbb{E}_{q_{i}\left(f_{i}\right)}\left[\log p\left(y_{i} | f_{i}\right)\right]=\frac{1}{\sqrt{2 \pi}} \int \log p\left(y_{i} | z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}}\right) e^{-\frac{1}{2} z_{i}^{2}} d z_{i}
$$

<h4>
Regression
</h4>
<p>
For regression tasks, we employ <b>Gaussian likelihood</b>, 
</p>

$$
\begin{aligned}
\mathbb{E}_{q_{i}\left(f_{i}\right)}\left[\log p\left(y_{i} | f_{i}\right)\right]
&=\frac{1}{\sqrt{2 \pi}} \int \log (\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(y_i-f_i)^2}{2\sigma^2}}) e^{-\frac{1}{2} z_{i}^{2}} d z_{i}\\
&=\frac{1}{\sqrt{2 \pi}} \int \left(-\log (\sqrt{2\pi}\sigma) -\frac{(y_i-f_i)^2}{2\sigma^2}\right) e^{-\frac{1}{2} z_{i}^{2}} d z_{i}\\
&=-\log (\sqrt{2\pi}\sigma) - \frac{1}{\sqrt{2 \pi}} \int \frac{y_i^2-2y_if_i+f_i^2}{2\sigma^2}e^{-\frac{1}{2} z_{i}^{2}}d z_{i}\\
&=-\log (\sqrt{2\pi}\sigma) - \frac{1}{\sqrt{2\pi}}\int \frac{y_i^2}{2\sigma^2}e^{-\frac{1}{2} z_{i}^{2}}d z_{i} + \frac{1}{\sqrt{2 \pi}\sigma^2} \int y_i(z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}}) e^{-\frac{1}{2} z_{i}^{2}} d z_{i}- \frac{1}{\sqrt{2\pi}}\int \frac{f_i^2}{2\sigma^2}e^{-\frac{1}{2} z_{i}^{2}}d z_{i}\\
&=-\log (\sqrt{2\pi}\sigma) - \frac{y_i^2}{2\sigma^2} + \frac{y_i m_{q_{i}}}{\sigma^2} - \frac{1}{\sqrt{2\pi}}\int \frac{(z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}})^2}{2\sigma^2}e^{-\frac{1}{2} z_{i}^{2}}d z_{i}\\
&=-\log (\sqrt{2\pi}\sigma) - \frac{y_i^2}{2\sigma^2} + \frac{y_i m_{q_{i}}}{\sigma^2} - \frac{m_{q_{i}}^2+v_{q_{i}}}{2\sigma^2}\\
&=-\log (\sqrt{2\pi}\sigma) - \frac{(y_i-m_{q_{i}})^2+v_{q_{i}}}{2\sigma^2}
\end{aligned}\label{cal-exp-logGausslik}
$$

<h4>
Count Regression
</h4>
<p>
For count regression tasks, we employ <b>Poisson likelihood</b>, 
</p>

$$
    p(y|\lambda)=\frac{1}{y!}e^{-\lambda}\lambda^y
    \label{Poisson-lik-lambda}
$$

<p>
The Poisson only has support for non-negative integers, whereas a Gaussian has support over all real numbers (including negatives). Therefore, we use a link function to connect the Poisson process with the Gaussian process. We have tried two kinds of link functions: $\lambda=e^{f}$ and $\lambda=\ln(1+e^{f})$.
</p>

<p>For the link function $\lambda=e^{f}$,</p>

$$\label{cal-exp-loglik}
\begin{aligned}
\mathbb{E}_{q_{i}\left(f_{i}\right)}\left[\log p\left(y_{i} | f_{i}\right)\right]
&=\frac{1}{\sqrt{2 \pi}} \int \log (\frac{1}{y_i!} e^{-e^{f_i}} e^{f_iy_i}) e^{-\frac{1}{2} z_{i}^{2}} d z_{i}\\
&=\frac{1}{\sqrt{2 \pi}} \int \left(\log (\frac{1}{y_i!}) - e^{z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}}} + (z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}})y_i\right) e^{-\frac{1}{2} z_{i}^{2}} d z_{i}\\
&=\frac{1}{\sqrt{2 \pi}}\log (\frac{1}{y_i!})\int e^{-\frac{1}{2} z_{i}^{2}} d z_{i} - \frac{1}{\sqrt{2 \pi}} \int \left(e^{z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}}}e^{-\frac{1}{2} z_{i}^{2}}\right)d z_{i} + \frac{1}{\sqrt{2 \pi}} \int (z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}})y_i e^{-\frac{1}{2} z_{i}^{2}} d z_{i}\\
&=\frac{1}{\sqrt{2\pi}}\log (\frac{1}{y_i!})\sqrt{2\pi}-\frac{1}{\sqrt{2 \pi}}e^{m_{q_{i}}} \int \left( e^{z_{i} \sqrt{v_{q_{i}}}-\frac{1}{2} z_{i}^{2}}\right)d z_{i} + \frac{y_i\sqrt{v_{q_{i}}}}{\sqrt{2 \pi}} \int z_{i}e^{-\frac{1}{2} z_{i}^{2}} d z_{i} +\frac{m_{q_i}y_i}{\sqrt{2\pi}}\int e^{-\frac{1}{2} z_{i}^{2}} d z_{i}\\
&=\log (\frac{1}{y_i!})-\frac{1}{\sqrt{2 \pi}}e^{m_{q_{i}}} \int \left( e^{-\frac{1}{2} z_{i}^{2}+z_{i} \sqrt{v_{q_{i}}}-\frac{1}{2} v_{q_{i}}+\frac{1}{2} v_{q_{i}}}\right)d z_{i} +\frac{y_im_{q_i}}{\sqrt{2\pi}}\sqrt{2\pi}\\
&=\log (\frac{1}{y_i!})-\frac{1}{\sqrt{2 \pi}}e^{\frac{1}{2}v_{q_{i}}}e^{m_{q_{i}}} \int e^{-\frac{1}{2} (z_{i}-\sqrt{v_{q_{i}}})^2}d z_{i} + y_i m_{q_i}\\
&=-\log y_i!-e^{m_{q_{i}} + \frac{1}{2}v_{q_{i}}} + y_i m_{q_i}\\
&=-\log\Gamma(y_i+1)-e^{m_{q_{i}} + \frac{1}{2}v_{q_{i}}} + y_i m_{q_i}
\end{aligned}
$$

<p>
where we use the fact that $\Gamma(x+1)=x!$ and $\int e^{-\frac{1}{2\sigma^2}x^2}dx=(2\pi \sigma^2)^{\frac{1}{2}}$.
</p>

<p>
However, the closed form for the expectation of log Poisson likelihood with the link function of $\lambda=\ln(1+e^{f})$ is not available. In this case, Gaussian-Hermite quadrature is employed to calculate the approximate expectation. If you are not familiar with Gaussian-Hermite quadrature, check out my blog on <a href="https://kaikaizhao.github.io/notes/2019/08/02/Numerical-Integration-Gaussian-Hermite-Quadrature" target="_blank">Numerical Integration - Gaussian-Hermite Quadrature</a>.
</p>

<h4>
Binary Classification
</h4>
<p>
For binary classification tasks, we use <b>sigmoid function</b> as the likelihood function. Unfortunately, the closed form for expectation of log likelihood is not available either. Here we can still use Gaussian-Hermite quadrature to get an approximate value.
</p>

$$
    \begin{aligned}
    \mathbb{E}_{q_{i}\left(f_{i}\right)}\left[\log p\left(y_{i}|f_{i}\right)\right]
    &=\frac{1}{\sqrt{2\pi}}\int e^{-\frac{1}{2} z_{i}^{2}} \log \frac{1}{1+\exp(-y_i f_i)}dz_{i}\\
    &=\frac{1}{\sqrt{2\pi}}\int e^{-\frac{1}{2} z_{i}^{2}} \log \frac{1}{1+\exp(-y_i (z_{i} \sqrt{v_{q_{i}}}+m_{q_{i}}))}dz_{i}\\
    &\approx\frac{1}{\sqrt{2\pi}}\sum_j^n w_j \log \frac{1}{1+\exp(-y_i (z_{j} \sqrt{v_{q_{i}}}+m_{q_{i}}))}\\
    \end{aligned}
$$

<p>
where $z$ denotes sample points and $w$ represents the corresponding weights, so their subscripts are different from the subscripts of $y$, $m$ or $v$.
</p>

### KL divergence

<p>
Recall for two Gaussian distributions $\mathcal{N}\left(\boldsymbol{\mu}_{0}, \Sigma_{0}\right)$ and $\mathcal{N}\left(\boldsymbol{\mu}_{1}, \Sigma_{1}\right)$, we have 
</p>

$$
\begin{aligned} \mathrm{KL}\left(\mathcal{N}_{0} \| \mathcal{N}_{1}\right)=& \frac{1}{2} \log \left|\Sigma_{1} \Sigma_{0}^{-1}\right|+\\ & \frac{1}{2} \operatorname{tr} \Sigma_{1}^{-1}\left(\left(\boldsymbol{\mu}_{0}-\boldsymbol{\mu}_{1}\right)\left(\boldsymbol{\mu}_{0}-\boldsymbol{\mu}_{1}\right)^{\top}+\Sigma_{0}-\Sigma_{1}\right) 
\end{aligned}
$$

<p>
In the case of the second term of VLB, i.e. $\mathrm{KL}\left(\phi\left(\boldsymbol{f}_{\mathcal{U}}\right) \| p\left(\boldsymbol{f}_{\mathcal{U}}\right)\right)$, we have
</p>

$$
    \begin{aligned} \mathrm{KL}\left(\phi\left(\boldsymbol{f}_{\mathcal{U}}\right) \| p\left(\boldsymbol{f}_{\mathcal{U}}\right)\right)&= \frac{1}{2} \log \left|K_MV^{-1}\right|+ \frac{1}{2} \operatorname{tr} K_M^{-1}\left(\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)^{\top}+V-K_M\right)\\
    &=\frac{1}{2} \log \left|K_M\right|+ \frac{1}{2} \log \left|V^{-1}\right|+ \frac{1}{2} \operatorname{tr} K_M^{-1}\left(\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)^{\top}+V-K_M\right)
    \end{aligned}\tag{3}\label{de-KL}
$$

### Deriving the formulae related to $\lambda=\ln(1+e^{f})$
<p>
Since the canonical link function $\lambda=e^{f}$ may bring numerical issues, we introduce a good alternative link function, that is, $\lambda=\ln(1+e^{f})$ which could be stabler. In this section, we will talk about calculating the first derivatives and the second derivatives for this link, including the corresponding expectations. We still use the chain rule. Specifically, we first calculate derivatives of log Poisson likelihood w.r.t $\lambda$ and then calculate derivatives of link function $\lambda=\ln(1+e^{f})$ w.r.t $f$.
</p>

$$
\begin{aligned}
\log p(y|f)&=\ln(\frac{1}{y!}\frac{1}{1+e^f}[\ln(1+e^f)]^y)\\
&=-\ln\Gamma(y+1) -\ln(1+e^f) + y\ln\ln(1+e^f)
\end{aligned}
$$

$$
\begin{aligned}
    \frac{\partial}{\partial f} \log p(y|f)&=\frac{\partial \log p(y|f)}{\partial \lambda} \frac{\partial \lambda}{\partial f}\\
    &\overset{\lambda=\ln(1+e^{f})}{=}(-1 + \frac{y}{\lambda})\frac{e^{f}}{1+e^{f}}\\
    &=\left(\frac{y}{\ln(1+e^{f})}-1\right)\frac{1}{1+e^{-f}}\\
    &=\frac{y}{(1+e^{-f})\ln(1+e^{f})}-\frac{1}{1+e^{-f}}\\
    &=\left(\frac{y}{\ln(1+e^{f})}-1\right)\frac{1}{1+e^{-f}}\\
\end{aligned}
$$

$$
\begin{aligned}
    \frac{\partial^2}{\partial f^2} \log p(y|f)&=\frac{\partial^2 \log p(y|f)}{\partial \lambda^2} \frac{\partial \lambda}{\partial f}\frac{\partial \lambda}{\partial f}+\frac{\partial \log p(y|f)}{\partial \lambda} \frac{\partial^2 \lambda}{\partial f^2}\\
    &\overset{\lambda=\ln(1+e^{f})}{=}(-\frac{y}{\lambda^2})\frac{e^{f}}{1+e^{f}}\frac{e^{f}}{1+e^{f}}+(-1 + \frac{y}{\lambda})\frac{e^{-f}}{(1+e^{-f})^2}\\
    &=(-\frac{y}{\lambda^2})\frac{1}{(1+e^{-f})^2}+(-1 + \frac{y}{\lambda})\frac{e^{-f}}{(1+e^{-f})^2}\\
    &=\left( (\frac{y}{\ln(1+e^f)}-1)e^{-f}-\frac{y}{\ln^2(1+e^f)} \right)\frac{1}{(1+e^{-f})^2}\\
\end{aligned}
$$

<p>
Once we obtain the above formulae, we can get expectations of the deravatives w.r.t $\mathcal{N}(f | m, v)$  through Gaussian-Hermite quadrature since the closed form expressions are not available.
</p>
