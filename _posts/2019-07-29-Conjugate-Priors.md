---
layout: post
title: Conjugate priors
tags: SVI
description: If a conjugate prior is available for a likelihood function, the posterior distribution has the same form as the prior.
date: 2019-07-29
---

<p>When I was studying variational inference, conjugacy ever made me confused. I spent some time figuring it out and found it is important for us to read some top-conference papers. So I would like to write down what I have learned and my thoughts as well. I hope it could be helpful for the friends who are struggling with this concept.</p>

### Definition

<p>First of all, let's see what's the definition of conjugate priors in the Section 2.4.2 of Bishop's PRML book</p>
<blockquote>
    <p>
        In general, for a given probability distribution $p(x|\eta)$, we can seek a prior $p(\eta)$ that is conjugate to the likelihood function, so that the posterior distribution has the same functional form as the prior.
    </p>
</blockquote>

<p>The critical point in the above definition is the final posterior distribution and the prior share the same form. Note that the exponential family will play an important role in the following explanation for the conjugacy concept. If you are not comfortable with it, welcome to check out my <a href="https://kaikaizhao.github.io/notes/2019/07/26/Exponential-Family" target="_blank">exponential family</a> blog.</p>

### Mathematical formulation

Recall the Bayes formula

$$
p(\beta | x)=\frac{p(x | \beta) p(\beta)}{\int p(x | \beta) p(\beta) d \beta}
$$

<p><b>Note:</b> the denominator is not a function w.r.t $\beta$ and it can be considered as a normalization term.</p>

$$
\underbrace{p(\beta | x)}_{\text{posterior}} \propto \underbrace{p(x | \beta)}_{\text {likelihood}}\underbrace{p(\beta) }_{\text{prior}}
$$

#### Given a certain likelihood function, how can we find its conjugate prior?

<p>The following derivation will end up a very useful and interesting result. I learned the following result from the Yida Xu's <a href="https://github.com/roboticcam/machine-learning-notes/blob/master/variational.pdf" target="_blank">variational inference</a> tutorial. Here we employ the exponential family form of probability distributions because it simplifies the derivation process significantly.</p>

* **Likelihood density:**

$$
p\left(x | \beta\right)=h\left(x\right) \exp \left\{t\left(x\right) \beta -\color{red}{A_l(\beta)}\right\}
$$

* **Prior:**

$$
p(\beta)=h(\beta) \exp \left\{\alpha^{T} t(\beta)-A_{g}(\alpha)\right\}=\underbrace{\exp \left(-A_{g}(\alpha)\right)}_{\text { normalization }} h(\beta) \exp \left\{\alpha^{T} t(\beta)\right\}
$$

<p>Let sufficient statistics $t(\beta)=\left[\beta\quad \color{red}{A_l(\beta)}\right]^{T}$ and $\alpha=\left[\alpha_{1}\quad \alpha_{2}\right]^{T}$</p>

$$
p(\beta)=h(\beta) \exp \left\{\left[\alpha_{1}\quad \alpha_{2}\right]\left[\beta\quad \color{red}{A_l(\beta)}\right]^{T}-A_{g}(\alpha)\right\}
$$

* **Posterior:**

$$
    \begin{aligned} 
    p\left(\beta | x \alpha\right) & \propto \underbrace{h(\beta) \exp \left\{\alpha t(\beta)^{T}\right\}}_{\text{prior}} \underbrace{\exp \left\{t(x) \beta-A_{l}(\beta)\right\}}_{\text{likelihood}} \\ &=h(\beta) \exp \left\{\left(\alpha_{1}+t(x)\right) \beta+\alpha_{2} A_{l}(\beta)-A_{l}(\beta)\right\} \\ &=h(\beta) \exp \left\{\left(\alpha_{1}+t(x)\right) \beta+\left(\alpha_{2}-1\right) A_{l}(\beta)\right\} \\ &=h(\beta) \exp \left\{\left[\alpha_{1}+t(x) \quad \alpha_{2}-1\right] t(\beta)^{T}\right\}
    \end{aligned}
$$

<p>Finally, we find the posterior distribution has the same form as the prior distribution if <b>the second term of the sufficient statistics for the prior distribution is equal to the log normalizer of the likelihood function</b>.</p>

### Examples

<p>If we have a binomial distribution with the parameter $\mu$ and $N$, we have</p>

$$
\operatorname{Bin}(m | N, \mu)=\left(\begin{array}{c}{N} \\ {m}\end{array}\right) \mu^{m}(1-\mu)^{N-m}
$$

where

$$
\left(\begin{array}{c}{N} \\ {m}\end{array}\right) \equiv \frac{N !}{(N-m) ! m !}\tag{1}\label{binomial}
$$

<p>Suppose the prior of its parameter $\mu$ is subject to the following beta distribution</p>

$$
\operatorname{Beta}(\mu | a, b)=\frac{\Gamma(a+b)}{\Gamma(a) \Gamma(b)} \mu^{a-1}(1-\mu)^{b-1}\tag{2}\label{beta}
$$

<p>The posterior distribution of $\mu$ can be obtained by multiplying the beta prior \eqref{beta} by the binomial likelihood function \eqref{binomial} and normalizing as follows.</p>
 
$$
p(\mu | m, l, a, b)=\frac{\Gamma(m+a+l+b)}{\Gamma(m+a) \Gamma(l+b)} \mu^{m+a-1}(1-\mu)^{l+b-1}
$$

<p>where $l=N-m$.</p>

#### Conjugate pairs

<table style="width:100%">
    <tr>
        <th>Prior</th>
        <th>Likelihood</th> 
    </tr>
    <tr>
        <td>Gaussian $e^{-\left\|\mu-\mu_{0}\right\|^{2} /\left(2 \sigma^{2}\right)}$</td>
        <td>Gaussian $e^{-\|x-\mu\|^{2} /\left(2 \sigma^{2}\right)}$</td> 
    </tr>
    <tr>
        <td>Beta $\frac{\Gamma(r+s)}{\Gamma(r) \Gamma(s)} \alpha^{r-1}(2-\alpha)^{s-1}$</td>
        <td>Bernoulli $\quad \alpha^{x}(1-\alpha)^{1-x}$</td> 
    </tr>
    <tr>
        <td>Dirichlet $\frac{\Gamma\left(\sum \alpha_{i}\right)}{\prod \Gamma\left(\alpha_{i}\right)} \prod \theta_{i}^{\alpha_{i}-1}$</td>
        <td>Multinomial $\frac{\left(\sum x_{i}\right) !}{\prod x_{i} !} \prod \theta_{i}^{x_{i}}$</td> 
    </tr>
    <tr>
        <td>Wishart</td>
        <td>Gaussian(variance)</td> 
    </tr>
</table>
<br>
<p>As can be seen from the above table, when the likelihood function belongs to Gaussian distribution, its conjugate prior for the parameter $\mu$ is the Gaussian distribution while its conjugate prior for the variance is Wishart distribution. Hence, we should <b><font color="red">specify the parameter first when we talk about conjugate priors</font></b>.</p>