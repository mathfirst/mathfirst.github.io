---
layout: post
title: Sparse Variational Inference for Generalized Gaussian Process Models - Tutorial 3
description: This is a tutorial for this ICML 2015 paper 'Sparse Variational Inference for Generalized Gaussian Process Models'. It covers hyperparameters optimization.
date: 2019-08-11
---
<p>
In this continued blog, we will talk about hyperparameters optimization, including calculating the gradients of the VLB w.r.t kernel parameters. Since the derivations w.r.t hyperparameters optimization are not given in the original paper, we'll derive them by ourselves.
</p>

### Hyperparameters optimization

<p>
In most cases, the kernel function is given by the following formula:
</p>

$$
k\left(x, x^{\prime}\right)=\sigma_{f}^{2} \exp \left(-\frac{1}{2l^2}\left(x-x^{\prime}\right)^{2}\right)
$$

<p>
we can write our hyperparameters together as $\boldsymbol{\theta}=\{l, \sigma_f^2\}$.
</p>

<p>
Now what we need to do is deriving the gradients of VLB w.r.t $\boldsymbol{\theta}$. The VLB is as follows,
</p>

$$\label{VLB-explicit-hpy}
\begin{aligned}
    \log p(\boldsymbol{y}) &\geq \sum_{i=1}^{N} \mathbb{E}_{q\left(f\left(\boldsymbol{x}_{i}\right)\right)}\left[\log p\left(y_{i} | f\left(\boldsymbol{x}_{i}\right)\right)\right] -\mathrm{KL}\left(\phi\left(\boldsymbol{f}_{\mathcal{U}}\right) \| p\left(\boldsymbol{f}_{\mathcal{U}}\right)\right)
\end{aligned}
$$

<p>
We already have the marginal distribution of $q(f(\boldsymbol{x}_{i}))$ which is given by a univariate Gaussian with mean $m_q(\boldsymbol{x}_i)$ and variance $v_q(\boldsymbol{x}_i)$ where
</p>

$$\label{m-q}
    m_{q}(\boldsymbol{x})=m(\boldsymbol{x})+K_{x M} K_{M}^{-1}(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}})
$$

$$
v_{q}(\boldsymbol{x})=k(\boldsymbol{x}, \boldsymbol{x})+K_{x M} K_{M}^{-1}\left(\boldsymbol{V}-K_{M}\right) K_{M}^{-1} K_{M x}
$$

<p>
The VLB contains two terms: the expectation of log likelihood w.r.t the approximate marginal $q_i$ and the KL divergence.
</p>

#### The gradients of the first term of VLB w.r.t hyperparameters

$$
\begin{aligned}
\frac{\partial\mathbb{E}_{q\left(f\left(\boldsymbol{x}_{i}\right)\right)}\left[\log p\left(y_{i} | f\left(\boldsymbol{x}_{i}\right)\right)\right]}{\partial \boldsymbol{\theta}}&=\frac{\partial\mathbb{E}_{q\left(f\left(\boldsymbol{x}_{i}\right)\right)}\left[\log p(y_{i} | f(\boldsymbol{x}_{i}))\right] }{\partial m_q(\boldsymbol{x}_i)}\frac{\partial m_q(\boldsymbol{x}_i)}{\partial\boldsymbol{\theta}}+\frac{\partial\mathbb{E}_{q\left(f\left(\boldsymbol{x}_{i}\right)\right)}\left[\log p(y_{i} | f(\boldsymbol{x}_{i}))\right] }{\partial v_q(\boldsymbol{x}_i)}\frac{\partial v_q(\boldsymbol{x}_i)}{\partial\boldsymbol{\theta}}\\
&=\rho_i\cdot\frac{\partial[m(\boldsymbol{x})+K_{iM} K_{M}^{-1}(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}})]}{\partial\boldsymbol{\theta}}+\frac{\lambda_i}{2}\cdot\frac{\partial[K_{ii}+K_{iM} K_{M}^{-1}\left(\boldsymbol{V}-K_{M}\right) K_{M}^{-1} K_{Mi}]}{\partial\boldsymbol{\theta}}\\
&=\rho_i\cdot\frac{\partial K_{iM} K_{M}^{-1}\boldsymbol{m}}{\partial\boldsymbol{\theta}}+\frac{\lambda_i}{2}\cdot\frac{\partial[K_{ii}+K_{iM} K_{M}^{-1}\left(\boldsymbol{V}-K_{M}\right) K_{M}^{-1} K_{Mi}]}{\partial\boldsymbol{\theta}}\\
&=\rho_i\cdot\left(\frac{\partial K_{iM} }{\partial\boldsymbol{\theta}}K_{M}^{-1}\boldsymbol{m}+K_{iM}\frac{\partial  K_{M}^{-1}}{\partial\boldsymbol{\theta}}\boldsymbol{m}\right)+\frac{\lambda_i}{2}\cdot(\frac{\partial K_{ii}}{\partial\boldsymbol{\theta}}+\frac{\partial K_{iM}}{\partial\boldsymbol{\theta}}K_{M}^{-1}(\boldsymbol{V}-K_{M}) K_{M}^{-1} K_{Mi}\\& +K_{iM}\frac{\partial  K_{M}^{-1}}{\partial\boldsymbol{\theta}}(\boldsymbol{V}-K_{M}) K_{M}^{-1} K_{Mi}+K_{iM}K_{M}^{-1}\frac{\partial  (\boldsymbol{V}-K_{M})}{\partial\boldsymbol{\theta}} K_{M}^{-1} K_{Mi}\\&
+K_{iM}K_{M}^{-1}(\boldsymbol{V}-K_{M})\frac{\partial K_{M}^{-1}}{\partial\boldsymbol{\theta}} K_{Mi})+K_{iM}K_{M}^{-1}(\boldsymbol{V}-K_{M})K_{M}^{-1}\frac{\partial K_{Mi}}{\partial\boldsymbol{\theta}} )\\
&=\rho_i\cdot\left(\frac{\partial K_{iM} }{\partial\boldsymbol{\theta}}K_{M}^{-1}\boldsymbol{m}+K_{iM}K_{M}^{-1}\frac{\partial  K_{M}}{\partial\boldsymbol{\theta}}K_{M}^{-1}\boldsymbol{m}\right)+\frac{\lambda_i}{2}\cdot(\frac{\partial K_{ii}}{\partial\boldsymbol{\theta}}+\frac{\partial K_{iM}}{\partial\boldsymbol{\theta}}K_{M}^{-1}(\boldsymbol{V}-K_{M}) K_{M}^{-1} K_{Mi}\\& +K_{iM}K_{M}^{-1}\frac{\partial  K_{M}}{\partial\boldsymbol{\theta}}K_{M}^{-1}(\boldsymbol{V}-K_{M}) K_{M}^{-1} K_{Mi}-K_{iM}K_{M}^{-1}\frac{\partial  K_{M}}{\partial\boldsymbol{\theta}} K_{M}^{-1} K_{Mi}\\&
+K_{iM}K_{M}^{-1}(\boldsymbol{V}-K_{M})K_{M}^{-1}\frac{\partial  K_{M}}{\partial\boldsymbol{\theta}}K_{M}^{-1} K_{Mi})+K_{iM}K_{M}^{-1}(\boldsymbol{V}-K_{M})K_{M}^{-1}\frac{\partial K_{Mi}}{\partial\boldsymbol{\theta}} )
\end{aligned}
$$

<p>
where the third equality follows from $m(\boldsymbol{x})=\boldsymbol{0}$ and $\boldsymbol{m}_{\mathcal{U}}=\boldsymbol{0}$, and the last equality follows from $\frac{\partial K^{-1}}{\partial \theta}=-K^{-1} \frac{\partial K}{\partial \theta} K^{-1}$. These kernel matrices $K_{iM}$, $K_M$, $K_M^{-1}$, $K_{iM}$ and $K_{ii}$ depend on $\boldsymbol{\theta}$ while $\boldsymbol{m}$ and $\boldsymbol{V}$ do not depend on $\boldsymbol{\theta}$. Hence, we can take $\boldsymbol{m}$ and $\boldsymbol{V}$ as constants when we calculate the gradients of VLB w.r.t $\boldsymbol{\theta}$.
</p>

#### The gradients of KL term w.r.t hyperparameters

<p>
The second term of the VLB is the KL divergence,
</p>

$$\label{de-KL-hyp}
    \begin{aligned} \mathrm{KL}\left(\phi\left(\boldsymbol{f}_{\mathcal{U}}\right) \| p\left(\boldsymbol{f}_{\mathcal{U}}\right)\right)&= \frac{1}{2} \log \left|K_MV^{-1}\right|+ \frac{1}{2} \operatorname{tr} \left(K_M^{-1}\left((\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}})\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)^{\top}+V-K_M\right)\right)\\
    &=\frac{1}{2} \log \left|K_M\right|+ \frac{1}{2} \log \left|V^{-1}\right|+ \frac{1}{2} \operatorname{tr} \left(K_M^{-1}\left((\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}})\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)^{\top}+V-K_M\right)\right)
    \end{aligned}
$$

$$
    \begin{aligned}
    \frac{\partial\mathrm{KL}\left(\phi\left(\boldsymbol{f}_{\mathcal{U}}\right) \| p\left(\boldsymbol{f}_{\mathcal{U}}\right)\right)}{\partial \boldsymbol{\theta}}
    &=\frac{1}{2} \operatorname{tr}\left( K_M^{-1}\frac{\partial K_M}{\partial \boldsymbol{\theta}} \right) + \frac{1}{2} \operatorname{tr} \left(\frac{\partial K_M^{-1}}{\partial \boldsymbol{\theta}} \left((\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}})\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)^{\top}+V-K_M\right)-\frac{\partial K_M}{\partial \boldsymbol{\theta}}\right)\\
    &=\frac{1}{2} \operatorname{tr}\left( K_M^{-1}\frac{\partial K_M}{\partial \boldsymbol{\theta}} \right) + \frac{1}{2} \operatorname{tr} \left(K_M^{-1}\frac{\partial K_M}{\partial \boldsymbol{\theta}}K_M^{-1} \left((\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}})\left(\boldsymbol{m}-\boldsymbol{m}_{\mathcal{U}}\right)^{\top}+V-K_M\right)-\frac{\partial K_M}{\partial \boldsymbol{\theta}}\right)
    \end{aligned}
$$

<p>
where we use the formula $\frac{\partial \log |K|}{\partial \theta}=\operatorname{tr}\left(K^{-1} \frac{\partial K}{\partial \theta}\right)$.
</p>


#### Partial derivatives w.r.t the length scale $l$

$$
    (\frac{\partial K_M}{\partial l})_{mm^\prime}=\frac{1}{l^3}\cdot k(Z_m,Z_m')(Z_m-Z_m')^2 \qquad \frac{\partial K_M}{\partial l}=\frac{1}{l^3}\cdot S_M\odot K_M
$$

<p>
where $k(,)$ denotes the squared exponential kernel function, $S$ is the squared distance matrix and $\odot$ is Hadamard product.
</p>

$$
    (\frac{\partial K_{iM}}{\partial l})_{im}=\frac{1}{l^3}\cdot k(X_i,Z_m)(X_i-Z_m)^2 \qquad \frac{\partial K_{iM}}{\partial l}=\frac{1}{l^3}\cdot S_{iM}\odot K_{iM}
$$

$$
    \frac{\partial K_{ii}}{\partial l}=0
$$

#### Partial derivatives w.r.t the vertical scale $\sigma_f^2$

$$
    \frac{\partial K_M}{\partial \sigma_f^2}=\frac{K_M}{\sigma_f^2}
$$

<p>
since we have the constraint of non-negativeness on $\sigma_f^2$, we can define $\gamma=\log(\sigma_f^2)$, and then use the chain rule.
</p>

$$
    \frac{\partial K_M}{\partial \gamma}=\frac{\partial K_M}{\partial \sigma_f^2}\frac{\partial \sigma_f^2}{\partial \gamma}=K_M
$$

$$
    \frac{\partial K_{iM}}{\partial \sigma_f^2}=\frac{K_{iM}}{\sigma_f^2} \qquad \frac{\partial K_{iM}}{\partial \gamma}=\frac{\partial K_{iM}}{\partial \sigma_f^2}\frac{\partial \sigma_f^2}{\partial \gamma}=K_{iM}
$$

$$
\frac{\partial K_{ii}}{\partial \sigma_f^2}=1 \qquad \frac{\partial K_{ii}}{\partial \gamma}=\frac{\partial K_{ii}}{\partial \sigma_f^2}\frac{\partial \sigma_f^2}{\partial \gamma}=e^{\gamma}
$$

<p>
where we optimize $\gamma$ instead of $\sigma_f^2$. After $\gamma$ is obtained, we get $\sigma_f^2$ via $\gamma=\log(\sigma_f^2)$.
</p>

<p>
We'll explore how to optimize the locations of inducing points in the future. Then if we managed it, we'll complete this blog.
</p>
<br>
<p>
In the next blog, we'll talk about how to calculate the predictive distribution.
</p>