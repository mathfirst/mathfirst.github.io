---
layout: post
title: Numerical Integration - Gaussian-Hermite Quadrature
description: How to calculate the expectation of a function w.r.t a normal distribution when its closed form is not available
date: 2019-08-02
---

### Why do we need Gaussian-Hermite Quadrature?
<p>
Speaking of Bayesian machine learning, you may have to deal with non-Gaussian likelihoods, say Poisson likelihood, which require some approximation of the posterior as the prior is non-conjugate. 
</p>
<p>
In the case of variational inference, when we compute the variational evidence lower bound(ELBO), the first term of ELBO usually represents the goodness of fit for a certain model, that is to say, we need to calculate the expectation of log likelihood w.r.t the variational distribution. Generally, our variational distribution is easy to handle, like Gaussian distribution. 
</p>
<p>
What I'm trying to say is how to calculate
</p>

\begin{equation}
\mathrm{E}[h(y)] \text{ with }  y \sim N\left(\mu, \sigma^{2}\right)
\end{equation}

The above is equivalent to calculate

\begin{equation}
\int_{-\infty}^{\infty} \frac{1}{\sigma \sqrt{2\pi}} h(y) \exp \left(-\frac{(y-\mu)^{2}}{2 \sigma^{2}}\right) dy
\label{eq:expect}
\end{equation}

### The mathematical formulations

<p>
In many cases, the closed form of data fit term, e.g. Eq. \eqref{eq:expect}, is not available because of the non-conjugacy between our likelihood and prior. Then we need approximation for calculating the expectation and Gaussian-Hermite Quadrature is a good tool for this.
</p>

<p>
In numerical analysis, <a href="https://en.wikipedia.org/wiki/Gauss%E2%80%93Hermite_quadrature" target="_blank">Gaussian-Hermite Quadrature</a> is used to approximate the value of integrals of the following kind:
</p>

\begin{equation}
\int_{-\infty}^{+\infty} e^{-x^{2}} f(x) dx
\label{eq:Hermite}
\end{equation}

The basic idea is

\begin{equation}
\int_{-\infty}^{+\infty} e^{-x^{2}} f(x) d x \approx \sum_{i=1}^{n} w_{i} f\left(x_{i}\right)
\end{equation}

where $$n$$ is the number of sample points used. The $$x_i$$ are the roots of the physicists' version of the Hermite polynomial $$H_n(x) (i = 1,2,\ldots,n)$$, and the associated weights $$w_i$$ are given by
\begin{equation}
w_{i}=\frac{2^{n-1} n ! \sqrt{\pi}}{n^{2}\left[H_{n-1}\left(x_{i}\right)\right]^{2}}
\end{equation}

As we can see that Eq. \eqref{eq:expect} does not exactly correspond to the Hermite polynomial, i.e. Eq. \eqref{eq:Hermite}, we need to change variables:
\begin{equation}
x=\frac{y-\mu}{\sqrt{2} \sigma} \Leftrightarrow y=\sqrt{2} \sigma x+\mu
\end{equation}

Then we obtain
\begin{equation}
E[h(y)]=\int_{-\infty}^{+\infty} \frac{1}{\sqrt{\pi}} \exp \left(-x^{2}\right) h(\sqrt{2} \sigma x+\mu) d x
\end{equation}

leading to:
\begin{equation}
E[h(y)] \approx \frac{1}{\sqrt{\pi}} \sum_{i=1}^{n} w_{i} h\left(\sqrt{2} \sigma x_{i}+\mu\right)
\label{eq:Exp-Hermite}
\end{equation}

### Python implementation

In this section, we introduce how to get approximate value of the aformentioned integral through Gaussian-Hermite Quadrature. In this blog, we use the scipy function <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_hermitenorm.html" target="_blank">roots_hermitenorm</a> to get the sample points $$x_i$$ and weights $$w_i$$ which employs $$\int_{-\infty}^{+\infty} e^{-\frac{x^{2}}{2}}f(x)dx$$ not the above $$\int_{-\infty}^{+\infty} e^{-x^{2}} f(x) dx$$, so we change variables as follows
\begin{equation}
x=\frac{z}{\sqrt{2}}
\end{equation}

After $$x$$ is substituted with $$z$$ in Eq. \eqref{eq:Exp-Hermite}, we obtain
\begin{equation}
E[h(y)]=\int_{-\infty}^{+\infty} \frac{1}{\sqrt{2\pi}} \exp \left(-z^{2}\right) h(\sigma z+\mu) d z
\end{equation}

leading to:
\begin{equation}
E[h(y)] \approx \frac{1}{\sqrt{2\pi}} \sum_{i=1}^{n} w_{i} h\left(\sigma z_{i}+\mu\right)
\label{eq:Exp-Hermite-z}
\end{equation}


Now I present an example to demonstrate how to use this scipy function. 

We can consider a binary classification task and assume that the latent distribution of the prediction for a test point we've got is $$p(f_*)=N(u_*,v_*)$$. Then if we use sigmoid function $$\sigma(f_*)=\frac{1}{1+e^{-f_*}}$$, the prediction is supposed to be

$$
    \begin{aligned}
    p\left(y_{*}=1 | \mathbf{x}_{*}, u_*, v_*\right) 
    &\approx \int \sigma\left(f_{*}\right) p\left(f_{*} | \mathbf{x}_{*}, u_*, v_*\right) d f_{*}\\
    &\approx\frac{1}{\sqrt{2\pi}} \sum_{j=1}^{n} w_{j}\sigma(\sqrt{v_*} z_j+u_*)
    \end{aligned}\tag{12}
$$

Its corresponding python code is as follows,

```python
import numpy as np
from scipy.special import roots_hermitenorm, expit # sigmoid function

z,w = roots_hermitenorm(n=50);
expectation = 1/np.sqrt(2*np.pi)*np.dot(w, expit(np.sqrt(v)*z+u))
```
