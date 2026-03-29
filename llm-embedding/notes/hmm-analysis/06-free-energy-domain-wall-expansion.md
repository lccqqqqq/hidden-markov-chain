# Free Energy in the Domain-Wall Expansion

## Starting point

From the Ising mapping (Section 5), the marginal sequence probability is $P(x_{1:T}) = C \cdot Z$ where

$$Z(x_{1:T}) = \sum_{s_{1:T} \in \{\pm 1\}^T} \exp\!\left[J\sum_{t=1}^{T-1} s_t s_{t+1} + \sum_{t=1}^{T} h_t s_t + h_0 s_1\right],$$

with

$$J = \frac{1}{2}\log\frac{1-\varepsilon}{\varepsilon}, \qquad h_t = h(x_t) = \frac{1}{2}\log\frac{e_A(x_t)}{e_B(x_t)}, \qquad h_0 = \frac{1}{2}\log\frac{\pi_A}{\pi_B}.$$

The free energy (at temperature 1) is $-F = \log Z$.

---

## Domain-wall rewriting

Introduce wall variables $n_t = \frac{1 - s_t s_{t+1}}{2} \in \{0,1\}$. The coupling term becomes

$$J\sum_{t=1}^{T-1} s_t s_{t+1} = J(T-1) - 2J\sum_{t=1}^{T-1} n_t.$$

Each wall costs $2J = \log\frac{1-\varepsilon}{\varepsilon}$, so the wall fugacity is

$$y = e^{-2J} = \frac{\varepsilon}{1-\varepsilon}.$$

Reconstruct spins from the initial spin and wall positions:

$$s_t = s_1 \prod_{u < t}(1 - 2n_u).$$

Then $Z$ becomes

$$Z = e^{J(T-1)} \sum_{s_1 = \pm 1} e^{h_0 s_1} \sum_{\{n_t\}} e^{-2J\sum n_t} \exp\!\left[\sum_t h_t \, s_1 \prod_{u:\tau_u < t}(-1)\right].$$

Group by number of walls $K$ at ordered positions $\tau_1 < \cdots < \tau_K$:

$$Z = e^{J(T-1)} \sum_{s_1 = \pm 1} e^{h_0 s_1} \sum_{K=0}^{T-1} y^K \sum_{\tau_1 < \cdots < \tau_K} \exp\!\left[\sum_t h_t \, s_1 \prod_{u:\tau_u < t}(-1)\right].$$

---

## $K = 0$: No walls (frozen sector)

With no walls, all spins equal $s_1$:

$$Z_0 = e^{J(T-1)} \sum_{s_1 = \pm 1} \exp\!\left[s_1\!\left(h_0 + \sum_{t=1}^T h_t\right)\right] = e^{J(T-1)} \cdot 2\cosh\!\left(h_0 + H_T\right),$$

where

$$H_T := \sum_{t=1}^T h_t = \frac{1}{2}\sum_{t=1}^T \log\frac{e_A(x_t)}{e_B(x_t)}$$

is the total log-likelihood-ratio evidence.

The frozen free energy is

$$-F_0 = J(T-1) + \log 2 + \log\cosh(h_0 + H_T).$$

For large $|h_0 + H_T|$ (strong evidence for one state), this simplifies to $|h_0 + H_T|$ — the MAP decoder picks whichever state the evidence favors.

---

## $K = 1$: Single domain wall

A wall at position $\tau$ flips all spins after time $\tau$. With a wall at $\tau$, the spin configuration is:

$$s_t = \begin{cases} s_1 & t \leq \tau, \\ -s_1 & t > \tau. \end{cases}$$

So the field term becomes:

$$\sum_{t=1}^T h_t s_t = s_1 \sum_{t=1}^{\tau} h_t + (-s_1) \sum_{t=\tau+1}^{T} h_t = s_1 \!\left(\sum_{t=1}^{\tau} h_t - \sum_{t=\tau+1}^{T} h_t\right).$$

Writing $H_\tau = \sum_{t=1}^\tau h_t$ and noting $\sum_{t=\tau+1}^T h_t = H_T - H_\tau$:

$$\sum_t h_t s_t = s_1\big(H_\tau - (H_T - H_\tau)\big) = s_1(2H_\tau - H_T).$$

Including the boundary field and the coupling term (with one wall contributing $y = e^{-2J}$, and $T-2$ non-wall bonds contributing $e^{J(T-2)}$, so total coupling is $e^{J(T-1)} \cdot y$):

$$Z_1 = e^{J(T-1)} \cdot y \sum_{\tau=1}^{T-1} \sum_{s_1 = \pm 1} \exp\!\big[s_1(h_0 + 2H_\tau - H_T)\big].$$

The sum over $s_1 = \pm 1$ gives $e^{+(\cdots)} + e^{-(\cdots)} = 2\cosh(\cdots)$:

$$Z_1 = e^{J(T-1)} \cdot y \cdot 2\sum_{\tau=1}^{T-1} \cosh\!\big(h_0 + 2H_\tau - H_T\big).$$

---

## Free energy to first order in fugacity

Combining $Z = Z_0 + Z_1 + O(y^2)$ and dividing:

$$\frac{Z_1}{Z_0} = \frac{e^{J(T-1)} \cdot y \cdot 2\sum_\tau \cosh(h_0 + 2H_\tau - H_T)}{e^{J(T-1)} \cdot 2\cosh(h_0 + H_T)} = y\sum_{\tau=1}^{T-1} \frac{\cosh(h_0 + 2H_\tau - H_T)}{\cosh(h_0 + H_T)}.$$

Therefore

$$Z = Z_0\!\left[1 + y\sum_{\tau=1}^{T-1} \frac{\cosh(h_0 + 2H_\tau - H_T)}{\cosh(h_0 + H_T)} + O(y^2)\right].$$

Taking the log:

$$\boxed{-F = J(T-1) + \log 2 + \log\cosh(h_0 + H_T) + y\sum_{\tau=1}^{T-1} \frac{\cosh(h_0 + 2H_\tau - H_T)}{\cosh(h_0 + H_T)} + O(y^2).}$$

---

## Interpretation

### Frozen sector ($K = 0$)

The first three terms give the free energy when the hidden state never switches. The total evidence $H_T$ determines which state dominates:
- if $H_T \gg 1$, the data favors state $A$ throughout,
- if $H_T \ll -1$, the data favors state $B$ throughout,
- the $\cosh$ interpolates smoothly between these.

### Single-wall correction ($K = 1$)

The $O(y)$ term is a sum over all possible wall insertion points $\tau$. Each is weighted by the ratio

$$\frac{\cosh(h_0 + 2H_\tau - H_T)}{\cosh(h_0 + H_T)}.$$

The argument $2H_\tau - H_T = H_\tau - (H_T - H_\tau)$ compares the evidence before $\tau$ to the evidence after $\tau$. A wall at $\tau$ is favorable (ratio large) when the evidence before $\tau$ strongly favors a different state than the evidence after $\tau$ — i.e., there is a genuine change point in the data.

### Fugacity

The wall fugacity $y = \varepsilon/(1-\varepsilon) \approx \varepsilon$ for small $\varepsilon$ controls the density of walls. The expansion is valid when $Ty \ll 1$, i.e., the expected number of walls is small.

### Connection to the confining potential (Section 5)

The $K = 1$ term inserts a single wall. For $K = 2$ (two walls enclosing a domain), the cost includes the confining potential

$$U_\sigma(u, v \mid x) = 2\mu + \sum_{t=u+1}^{v} \log\frac{e_\sigma(x_t)}{e_{-\sigma}(x_t)},$$

which on average grows linearly with domain length $\ell = v - u$ with string tension $\gamma = D_{\mathrm{KL}}(e_\sigma \| e_{-\sigma})$. This suppresses large wrong-phase domains exponentially.

---

## Validity regimes

| Regime | Condition | Description |
|---|---|---|
| Frozen | $Ty \ll 1$ | Keep $K = 0$ only; mixture of two i.i.d. models |
| Dilute wall gas | $Ty \lesssim 1$ | Keep $K = 0, 1$; single change-point corrections |
| Interacting walls | $Ty \sim O(1)$ | Need multi-wall terms; confining potential matters |
| Dense walls / Gaussian | $Ty \gg 1$ | Wall expansion breaks down; CLT on counts applies |
