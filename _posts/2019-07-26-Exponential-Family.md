---
layout: post
title: Exponential family
tags: SVI
description: When the probability distributions can be written in the form of exponential family, it will facilitate the derivations of the expectations of variational distributions.
date: 2019-07-26
---

<p> When I was delving into the classical <a href="http://www.columbia.edu/~jwp2128/Papers/HoffmanBleiWangPaisley2013.pdf" target="_blank">Stochastic Variational Inference</a>(SVI) paper, I found SVI is built upon natural gradients which are based on the form of exponential family. In particular, exponential family significantly facilitates the derivations of the expectations of likelihood with respect to variational distributions.</p>

Most of the distributions we have seen are from exponential family, except the Gaussian mixture. In this blog, I will summarize what I have explored and learned about exponential family. Also, I will present some beautiful properties from exponential family.

### Definition
The exponential family of distributions over $$x$$, given parameters $$\eta$$, is defined to be the set of distributions of the form

\begin{equation}
    p(x|\eta)=h(x) \exp \\{\eta^{T}T(x) -A(\eta)\\}
\end{equation}

where:

* $$\eta$$ is a vector of natural parameters
* $$T(x)$$ is sufficent statistics
* $$A(\eta)$$ is log normalizer

Let me explain the log normalizer a little bit more by calculating the integral of the above density estimation function over $$x$$
\begin{equation}
    \int p(x|\eta)dx=\frac{\int h(x) \exp \\{\eta^{T}T(x) \\}dx}{\exp\\{ A(\eta)\\} }=1
    \label{eq:integral}
\end{equation}
\begin{equation}
    A(\eta)=\log \left(\int h(x) \exp \\{\eta^{T}T(x) \\}dx\right)
    \label{eq:log-normalizer}
\end{equation}
From Eq. \eqref{eq:integral} and Eq. \eqref{eq:log-normalizer}, we can get why $$A(\eta)$$ is dubbed log normalizer.

### Beautiful properties

If we take the first derivative of the log normalizer, i.e. Eq. \eqref{eq:log-normalizer}, with respect to $\eta$, we have

$$
    \begin{aligned}
    \frac{\partial A(\eta)}{\partial \eta}
    &=\frac{\partial \log \left(\int h(x) \exp \{\eta^{T}T(x)\}dx\right) }{\partial \eta}\\
    &=\frac{\int h(x) \exp \{\eta^{T}T(x)\}T(x)dx }{\int h(x) \exp \{\eta^{T}T(x) \}dx}\\
    &\overset{\eqref{eq:integral}}{=}\frac{\int h(x)\exp\{\eta^{T}T(x)\}T(x)dx}{\exp\{A(\eta)\}}\\
    &=\int h(x)\exp\{\eta^{T}T(x)-A(\eta)\}T(x)dx\\
    &=\int p(x|\eta)T(x)dx\\
    &=E[T(x)]
    \end{aligned}
$$

Let's see what will happen if we take second derivative,

$$
    \begin{aligned} 
    \frac{\partial A(\eta)}{\partial \eta \partial \eta^{T}} &=\frac{\partial}{\partial \eta^{T}} \int h(x) \exp \left\{\eta^{T} T(x)-A(\eta)\right\} T(x) dx \\ 
    &=\int h(x) \exp \left\{\eta^{T} T(x)-A(\eta)\right\} T(x)\left(T(x)-A^{\prime}(\eta)\right) dx \\ 
    &=\int p(x | \eta) T^{2}(x) d x-A^{\prime}(\eta) \int p(x | \eta) T(x) d x \\ &=E\left[T^{2}(x)\right]-E[T(x)] E[T(x)] \\ 
    &=Var[T(x)]
    \end{aligned}
$$

The above derivation tells us another property from exponential family, that is, the second derivative of log normalizer equals the variance of sufficient statistics.

### Some examples

Let's look at some common probability distributions which can be written in the form of exponential family.

#### Gaussian distribution

$$
    \begin{aligned}
        p\left(x | \mu, \sigma^{2}\right) &=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{1}{2 \sigma^{2}}(x-\mu)^{2}\right\} \\ &=\frac{1}{\left(2 \pi \sigma^{2}\right)^{1 / 2}} \exp \left\{-\frac{1}{2 \sigma^{2}} x^{2}+\frac{\mu}{\sigma^{2}} x-\frac{1}{2 \sigma^{2}} \mu^{2}\right\}\\
        &=\frac{1}{\left(2 \pi\right)^{1 / 2}} \exp \left\{\underbrace{\left[\begin{array}{c}{x} \\ {x^{2}}\end{array}\right]^{T}}_{T(X)}\underbrace{\left[\begin{array}{c}{\frac{\mu}{\sigma_{1}^{2}}} \\ {-\frac{1}{2 \sigma^{2}}}\end{array}\right]}_{\eta}-\underbrace{\left(\frac{\mu^{2}}{2 \sigma^{2}}+\frac{1}{2} \ln (\sigma^{2})\right)}_{A(\eta)}\right\}
    \end{aligned}
$$

which, after some simple rearrangement, can be cast in the standard exponential family form with

$$
\eta=\left[\begin{array}{l}{\eta_{1}} \\ {\eta_{2}}\end{array}\right]=\left[\begin{array}{c}{\frac{\mu}{\sigma_{1}^{2}}} \\ {-\frac{1}{2 \sigma^{2}}}\end{array}\right], \quad T(x)=\left[\begin{array}{c}{x} \\ {x^{2}}\end{array}\right], \quad h(x)=(2 \pi)^{-1/2}
$$

$$
\eta_{2}=-\frac{1}{2 \sigma^{2}} \Longrightarrow \sigma^{2}=-\frac{1}{2 \eta_{2}} \quad \mu=\eta_{1} \sigma^{2}=\eta_{1} \frac{-1}{2 \eta_{2}}=\frac{-\eta_{1}}{2 \eta_{2}}
$$

$$
A(\eta)=\frac{\mu^{2}}{2 \sigma^{2}}+\frac{1}{2} \ln (\sigma^{2})=\frac{\left(\frac{-\eta_{1}}{2 \eta_{2}}\right)^{2}}{2\left(\frac{-1}{2 \eta_{2}}\right)}+\frac{1}{2} \ln (-\frac{1}{2 \eta_{2}})=\frac{-\eta_{1}^{2}}{4 \eta_{2}}-\frac{1}{2} \ln \left(-2 \eta_{2}\right)
$$

#### Dirichlet distribution

Suppose the parameters $\{\mu_k\}$ of multinomial distribution are drawn from a Dirichlet distribution parameterized by $\alpha$, we have

$$
    \begin{aligned}
    p(\mu | \alpha) &=\frac{\Gamma\left(\sum_{k} \alpha_{k}\right)}{\prod_{k} \Gamma\left(\alpha_{k}\right)} \prod_{k} \mu_{k}^{\alpha_{k}-1} \\ &=\exp \left\{\sum_{k}\left(\alpha_{k}-1\right) \log \mu_{k}-\left[\sum_{k} \log \Gamma\left(\alpha_{k}\right)-\log \Gamma\left(\sum_{k} \alpha_{k}\right)\right]\right\}
    \end{aligned}
$$

which can be cast in the standard exponential family form with

$$
\eta=\alpha-1
$$

$$
A(\eta)=\sum_{k} \log \Gamma\left(\alpha_{k}\right)-\log \Gamma\left(\sum_{k} \alpha_{k}\right)
$$

$$
T(\mu)=\log \mu
$$

Then the expectation of sufficient statistics equals

$$
    \begin{aligned}
    E\left[T_{k}(\mu)\right]&=E\left[\log \mu_{k}\right] =\frac{\partial}{\partial \eta_{k}} A(\eta) \\ &=\Psi\left(\alpha_{k}\right)-\Psi\left(\sum_{j} \alpha_{j}\right)
    \end{aligned}
$$

where $\Psi(\cdot)$ is the first derivative of log gamma function and it is called the digamma function. The above form is helpful for the derivations in latent Dirichlet allocation with variational inference. Here we already have a glimpse of the application of exponential family. It is powerful in variational models.

### Maximum likelihood and sufficient statistics
<p>For this part, we closely follow the Section 2.4.1 of Bishop's PRML book.</p>

Consider a set of independent identically distributed data denoted by $$\mathbf{X}=\left\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\right\}$$, for which the likelihood function is given by

$$
p(\mathbf{X} | \boldsymbol{\eta})=\left(\prod_{n=1}^{N} h\left(\mathbf{x}_{n}\right)\right)  \exp \left\{\boldsymbol{\eta}^{\mathrm{T}} \sum_{n=1}^{N} T\left(\mathbf{x}_{n}\right)-N\cdot A(\boldsymbol{\eta})\right\}
$$

$$
\ln{p(\mathbf{X} | \boldsymbol{\eta})} =\sum_{n=1}^{N} h\left(\mathbf{x}_{n}\right) + \boldsymbol{\eta}^{\mathrm{T}} \sum_{n=1}^{N} T\left(\mathbf{x}_{n}\right)-N\cdot A(\boldsymbol{\eta})
$$


\begin{equation}
\frac{\partial \ln{p(\mathbf{X} | \boldsymbol{\eta})}}{\partial \boldsymbol{\eta}} = \sum_{n=1}^{N} T\left(\mathbf{x}_{n}\right)-N\cdot A^{\prime}(\boldsymbol{\eta})=0
\end{equation}

<p>After setting the gradient of $\ln{p(\mathbf{X} | \boldsymbol{\eta})}$ w.r.t $\boldsymbol{\eta}$ to zero, we get the following condition to be satisfied by the maximum likelihood estimator $\boldsymbol{\eta_{\text{ML}}}$</p>

\begin{equation}
    A^{\prime}(\boldsymbol{\eta_{\text{ML}}})=\frac{1}{N}\sum_{n=1}^{N} T\left(\mathbf{x}_{n}\right)
    \label{eq:max-lik}
\end{equation}

which can in principle be solved to obtain $$\boldsymbol{\eta_{\text{ML}}}$$. We see that the solution for the maximum likelihood estimator depends on the data only through $\sum_{n} T\left(\mathbf{x}_{n}\right)$, which is therefore called the **sufficient statistic** of the exponential family distribution. We do not need to store the entire data set itself but only the value of the sufficient statistic. For the Gaussian distribution, for instance, $$T(x)=\left(x, x^{2}\right)^{\mathrm{T}}$$, and so we should keep both the sum of $$\left\{x_{n}\right\}$$ and the sum of $$\left\{x_{n}^2\right\}$$.

If we consider the limit $N \rightarrow \infty$, then the right-hand side of Eq. \eqref{eq:max-lik} becomes $$\mathbb{E}[T(\mathbf{x})]$$, and so by comparing with the first property we have got, we see in this limit $$\boldsymbol{\eta_{\text{ML}}}$$ will equal the true value $$\boldsymbol{\eta}$$.