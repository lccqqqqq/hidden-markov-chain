# Domain Walls, Ising Model, and Effective Interactions

## Motivation

We want to proceed to an intermediate regime where we can derive an effective interaction between the switchings (domain walls).

The clean way to do it is to look at the **posterior over hidden paths given the observed sequence**. That posterior is exactly a 1D Ising chain in a random field, and the switchings are domain walls.

One important caveat: if you average over all observations, there is **no extra interaction** between switches beyond the Markov prior. The nontrivial interaction appears only **conditioned on a realized sequence** $x_{1:T}$.

---

## Mapping to the 1D Ising Model

Take hidden state

$$s_t \in \{+1, -1\}, \qquad +1 \equiv A, \quad -1 \equiv B.$$

The transition matrix is

$$P(s_{t+1} = s_t) = 1 - \varepsilon, \qquad P(s_{t+1} = -s_t) = \varepsilon.$$

Let the emission probabilities be

$$e_+(x) = e_A(x), \qquad e_-(x) = e_B(x).$$

Then for a fixed observation sequence $x_{1:T}$,

$$P(s_{1:T} \mid x_{1:T}) \propto \pi_{s_1} \prod_{t=1}^{T-1} Q_{s_t, s_{t+1}} \prod_{t=1}^{T} e_{s_t}(x_t).$$

Now write the transition weight as

$$Q_{s,s'} = \sqrt{\varepsilon(1-\varepsilon)} \, \exp(J \, s s'), \qquad J = \frac{1}{2} \log \frac{1-\varepsilon}{\varepsilon}.$$

And write the emission as

$$e_{s_t}(x_t) = \sqrt{e_A(x_t) e_B(x_t)} \, \exp(h_t s_t),$$

with

$$h_t = \frac{1}{2} \log \frac{e_A(x_t)}{e_B(x_t)}.$$

So the posterior becomes

$$P(s_{1:T} \mid x_{1:T}) \propto \exp\left[J \sum_{t=1}^{T-1} s_t s_{t+1} + \sum_{t=1}^{T} h_t s_t + h_0 s_1\right],$$

where

$$h_0 = \frac{1}{2} \log \frac{\pi_A}{\pi_B}.$$

So this is exactly a **1D random-field Ising model**.

---

## Domain-Wall Variables

Now define the domain-wall variables

$$n_t = \frac{1 - s_t s_{t+1}}{2} \in \{0, 1\}.$$

Then $n_t = 1$ means there is a switch between $t$ and $t+1$.

Since

$$s_t s_{t+1} = 1 - 2n_t,$$

the coupling term becomes

$$J \sum_{t=1}^{T-1} s_t s_{t+1} = J(T-1) - 2J \sum_{t=1}^{T-1} n_t.$$

So each wall has a bare cost

$$2J = \log \frac{1-\varepsilon}{\varepsilon}.$$

At the prior level, walls are just a dilute gas with fugacity

$$y = e^{-2J} = \frac{\varepsilon}{1-\varepsilon}.$$

---

## The Exact Wall Action

The interesting part comes from the field term. If you reconstruct the spin from the wall history,

$$s_t = s_1 \prod_{u < t} (1 - 2n_u),$$

then

$$\sum_{t=1}^{T} h_t s_t = s_1 \sum_{t=1}^{T} h_t \prod_{u < t} (1 - 2n_u).$$

That is the exact wall action. It is nonlocal in the wall variables.

---

## Effective Pair Potential

To see the effective interaction cleanly, order the walls:

$$\tau_1 < \tau_2 < \cdots < \tau_K.$$

Suppose the background phase is $A$ on the left. Then a $B$-domain is an interval bounded by two neighboring walls,

$$(\tau_{2j-1}, \tau_{2j}].$$

Take one such interval $(u, v]$. Compare:
- path with no walls on that interval, so it stays in $A$,
- path with two walls, so it flips to $B$ on $(u, v]$ and then back to $A$.

The exact Bayes factor is

$$\frac{P(B\text{-domain on } (u,v] \mid x)}{P(\text{no domain on } (u,v] \mid x)} = \left(\frac{\varepsilon}{1-\varepsilon}\right)^2 \prod_{t=u+1}^{v} \frac{e_B(x_t)}{e_A(x_t)}.$$

So the corresponding effective pair potential between the two walls is

$$U_A(u, v \mid x) = 2 \log \frac{1-\varepsilon}{\varepsilon} + \sum_{t=u+1}^{v} \log \frac{e_A(x_t)}{e_B(x_t)}.$$

Equivalently, in terms of $h_t$,

$$U_A(u, v \mid x) = 2\mu + 2\sum_{t=u+1}^{v} h_t, \qquad \mu := \log \frac{1-\varepsilon}{\varepsilon}.$$

Likewise, for an $A$-domain inside a $B$-background,

$$U_B(u, v \mid x) = 2\mu - 2\sum_{t=u+1}^{v} h_t.$$

That is the effective interaction.

---

## The Physical Picture

So the picture is:
- each wall costs $\mu$,
- two adjacent walls enclosing an opposite-phase domain interact through the cumulative log-likelihood ratio across the interval between them.

For multiple domains, the weight factorizes over disjoint intervals, plus the obvious hard-core ordering constraint:

$$u_1 < v_1 < u_2 < v_2 < \cdots$$

So in the interval gas language, the posterior is a gas of disjoint opposite-phase domains with weights

$$w_A(u, v \mid x) = e^{-U_A(u,v \mid x)}, \qquad w_B(u, v \mid x) = e^{-U_B(u,v \mid x)}.$$

---

## The Intermediate Regime: Linear Confinement

Under data actually generated from state $A$,

$$\mathbb{E}_A\left[\log \frac{e_A(X)}{e_B(X)}\right] = D_{\text{KL}}(e_A \| e_B).$$

Therefore

$$\mathbb{E}_A[U_A(u, v \mid x)] = 2\mu + (v - u) \, D_{\text{KL}}(e_A \| e_B).$$

Similarly under $B$,

$$\mathbb{E}_B[U_B(u, v \mid x)] = 2\mu + (v - u) \, D_{\text{KL}}(e_B \| e_A).$$

So on average the two walls feel a **linear confining potential**:

$$U(\ell) \sim 2\mu + \gamma \ell,$$

with string tension

$$\gamma_A = D_{\text{KL}}(e_A \| e_B), \qquad \gamma_B = D_{\text{KL}}(e_B \| e_A).$$

**This is the key result.**

The walls want to annihilate because separating them creates a domain whose emission law disagrees with the data, and the disagreement accumulates linearly with length.

---

## Disorder and Fluctuations

There is also disorder around that mean. Define

$$\Lambda_t := \log \frac{e_A(x_t)}{e_B(x_t)}.$$

Then for an $A$-background,

$$U_A(u, v \mid x) = 2\mu + \sum_{t=u+1}^{v} \Lambda_t.$$

For a long interval of length $\ell = v - u$,

$$\sum_{t=u+1}^{v} \Lambda_t \approx \ell \, D_{\text{KL}}(e_A \| e_B) + \sqrt{\ell \, \Delta_A} \, \eta,$$

where

$$\Delta_A = \text{Var}_{X \sim e_A}\left[\log \frac{e_A(X)}{e_B(X)}\right].$$

So in the intermediate regime the effective pair potential is

$$U_A(\ell) \approx 2\mu + \gamma_A \ell + \sqrt{\Delta_A \ell} \, \eta.$$

That is: linear confinement plus random fluctuations.

---

## Crossover Length

This gives a natural crossover length:

$$\ell_* \sim \frac{2\mu}{\gamma} = \frac{2 \log((1-\varepsilon)/\varepsilon)}{D_{\text{KL}}(e_\sigma \| e_{-\sigma})}.$$

Interpretation:
- for $\ell \ll \ell_*$, the bare wall cost dominates;
- for $\ell \gg \ell_*$, the emission evidence dominates and long wrong-state domains are strongly suppressed;
- around $\ell \sim \ell_*$, you are in the intermediate regime where the wall gas is dilute but definitely interacting.

---

## Summary

The clean answer is:
- the exact posterior over hidden states is a random-field Ising chain;
- switchings are domain walls;
- the effective interaction between two neighboring walls at $u < v$ is the interval potential

$$U_\sigma(u, v \mid x) = 2 \log \frac{1-\varepsilon}{\varepsilon} + \sum_{t=u+1}^{v} \log \frac{e_\sigma(x_t)}{e_{-\sigma}(x_t)};$$

- on average this becomes a linear confining interaction with slope given by a KL divergence.

That is the natural domain-wall theory for the intermediate regime.

---

## The Marginal Sequence Probability as a Partition Function

The clean separation is:
- the **domain-wall / Ising picture** is most natural for the **posterior over hidden states** $P(s_{1:T} \mid x_{1:T})$,
- the **marginal law of the emitted sequence** $P(x_{1:T})$ is the corresponding **partition function** after summing over all hidden states.

For the 2-state HMM with

$$Q = \begin{pmatrix} 1-\varepsilon & \varepsilon \\ \varepsilon & 1-\varepsilon \end{pmatrix},$$

and emissions $e_A(k), e_B(k)$ over $k = 1, \ldots, N$, define spins

$$s_t = +1 \text{ for } A, \qquad s_t = -1 \text{ for } B.$$

Then for a fixed observed sequence $x_{1:T}$,

$$P(x_{1:T}, s_{1:T}) = \pi_{s_1} \prod_{t=1}^{T-1} Q_{s_t, s_{t+1}} \prod_{t=1}^{T} e_{s_t}(x_t).$$

Now write

$$J = \frac{1}{2} \log \frac{1-\varepsilon}{\varepsilon}, \qquad h(k) = \frac{1}{2} \log \frac{e_A(k)}{e_B(k)}, \qquad h_0 = \frac{1}{2} \log \frac{\pi_A}{\pi_B}.$$

Also

$$Q_{s,s'} = \sqrt{\varepsilon(1-\varepsilon)} \, e^{Jss'},$$

$$e_s(k) = \sqrt{e_A(k) e_B(k)} \, e^{h(k)s},$$

$$\pi_s = \sqrt{\pi_A \pi_B} \, e^{h_0 s}.$$

So the **exact marginal sequence probability** is

$$P(x_{1:T}) = \left[\sqrt{\pi_A \pi_B} \, (\varepsilon(1-\varepsilon))^{\frac{T-1}{2}} \prod_{t=1}^{T} \sqrt{e_A(x_t) e_B(x_t)}\right] Z(x_{1:T}),$$

where

$$Z(x_{1:T}) = \sum_{s_{1:T} \in \{\pm 1\}^T} \exp\left[J \sum_{t=1}^{T-1} s_t s_{t+1} + \sum_{t=1}^{T} h(x_t) s_t + h_0 s_1\right].$$

Start from the joint:                                                               
                                                                                
$$P(x_{1:T}, s_{1:T}) = \pi_{s_1} \prod_{t=1}^{T-1} Q_{s_t, s_{t+1}} \prod_{t=1}^{T}
e_{s_t}(x_t).$$                                                                    
                                                                                    
Marginalize by summing out the hidden states:                                       
                                                                                    
$$P(x_{1:T}) = \sum_{s_{1:T}} \pi_{s_1} \prod_t Q_{s_t, s_{t+1}} \prod_t            
e_{s_t}(x_t).$$                                                                     
                                                                                    
Now rewrite each factor in exponential form (this is just algebra, taking logs and  
re-exponentiating):                    
                                                                                    
$$Q_{s,s'} = \sqrt{\varepsilon(1-\varepsilon)}; e^{Jss'}, \qquad e_{s_t}(x_t) =     
\sqrt{e_A(x_t)e_B(x_t)}; e^{h(x_t),s_t}, \qquad \pi_{s_1} = \sqrt{\pi_A\pi_B};
e^{h_0 s_1}.$$                                                                      
                                                    
You can verify these by plugging in $s = +1$ and $s = -1$ and checking they         
reproduce the original values (e.g., $Q_{+1,+1} = 1-\varepsilon$, $Q_{+1,-1} =
\varepsilon$).                                                                      
                                                    
Substitute back:                       

$$P(x_{1:T}) =                                                                      
\underbrace{\sqrt{\pi_A\pi_B};[\varepsilon(1-\varepsilon)]^{\frac{T-1}{2}} \prod_t
\sqrt{e_A(x_t)e_B(x_t)}}{C(x{1:T})} ;\cdot; \sum_{s_{1:T}} e^{J\sum s_t s_{t+1} +   
\sum h(x_t)s_t + h_0 s_1}.$$                          
                                        
The prefactor $C$ collects all the square-root pieces that don't depend on $s$. The 
sum is $Z(x_{1:T})$.
                                                                                    
So: $P(x_{1:T}) = C \cdot Z$. No approximation, just rewriting products as          
exponentials of sums

That is the exact marginal pdf over emitted sequences.

So yes: the emitted-sequence law is an Ising **partition function in a token-dependent field**.

---

## Token Correlations

For **low-order token marginals**, yes, you can compute them from the states easily.

If the hidden chain is in stationarity, here it is symmetric so

$$\pi_A = \pi_B = \frac{1}{2}.$$

Then the one-token marginal is just

$$P(x_t = k) = \frac{1}{2} e_A(k) + \frac{1}{2} e_B(k).$$

The two-token marginal at lag $\tau$ is

$$P(x_t = k, x_{t+\tau} = l) = \sum_{i,j \in \{A,B\}} \pi_i (Q^\tau)_{ij} e_i(k) e_j(l).$$

Since for this symmetric chain

$$Q^\tau = \frac{1}{2} \begin{pmatrix} 1 + \rho^\tau & 1 - \rho^\tau \\ 1 - \rho^\tau & 1 + \rho^\tau \end{pmatrix}, \qquad \rho = 1 - 2\varepsilon,$$

this becomes

$$P(x_t = k, x_{t+\tau} = l) = p(k) p(l) + \frac{\rho^\tau}{4} \Delta(k) \Delta(l),$$

with

$$p(k) = \frac{e_A(k) + e_B(k)}{2}, \qquad \Delta(k) = e_A(k) - e_B(k).$$

So token correlations are easy: they inherit the same decay factor $(1-2\varepsilon)^\tau$ as the hidden chain.

But for the **full sequence probability** $P(x_{1:T})$, low-order state statistics are not enough. You need the full transfer-matrix / partition-function expression above. In general the emitted process is itself **not Markov of order 1**.

---

## Does $h_t$ Only Take Two Values?

In general, **no**.

You have

$$h_t = h(x_t) = \frac{1}{2} \log \frac{e_A(x_t)}{e_B(x_t)}.$$

So each token $k$ carries its own field value

$$h(k) = \frac{1}{2} \log \frac{e_A(k)}{e_B(k)}.$$

Therefore:
- for an alphabet of size $N$, $h_t$ can take **up to $N$ distinct values**;
- if two tokens have the same likelihood ratio $e_A(k)/e_B(k)$, they give the same field;
- only in the **binary-token case** does $h_t$ take two values;
- if some emission probability is zero, then $h(k) = \pm\infty$, which is a hard constraint rather than a finite field.

So the random field is not "two-valued" unless your emission alphabet is binary.

The useful conceptual picture is:
- the hidden coupling $J$ comes from switching cost,
- the token $x_t$ injects a local field $h(x_t)$,
- the marginal emitted-sequence probability is the partition function of that 1D random-field Ising chain.
