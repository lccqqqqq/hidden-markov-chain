# Small-$\varepsilon$ Limit: Rare-Switch Expansion

## Simplifications via Diagonalization

The nicest simplification is to diagonalize the transition part. Write

$$\alpha_t = \begin{pmatrix} \alpha_t(A) \\ \alpha_t(B) \end{pmatrix}, \qquad \alpha_t(i) = P(x_{1:t}, z_t = i),$$

so

$$\alpha_t = D_{x_t} Q \, \alpha_{t-1},$$

with

$$Q = \begin{pmatrix} 1-\varepsilon & \varepsilon \\ \varepsilon & 1-\varepsilon \end{pmatrix}, \qquad D_{x_t} = \begin{pmatrix} e_A(x_t) & 0 \\ 0 & e_B(x_t) \end{pmatrix}.$$

Now define the sum and difference variables

$$m_t = \alpha_t(A) + \alpha_t(B), \qquad d_t = \alpha_t(A) - \alpha_t(B).$$

Here $m_t$ is especially nice because

$$m_t = P(x_{1:t}),$$

the prefix probability you actually care about.

Also define

$$u_t = e_A(x_t) + e_B(x_t), \qquad v_t = e_A(x_t) - e_B(x_t), \qquad \rho = 1 - 2\varepsilon.$$

Then the exact recursion becomes

$$m_t = \frac{1}{2}\big(u_t \, m_{t-1} + \rho \, v_t \, d_{t-1}\big),$$

$$d_t = \frac{1}{2}\big(v_t \, m_{t-1} + \rho \, u_t \, d_{t-1}\big).$$

So the whole 2-state HMM reduces to a 2-scalar recursion. The parameter $\varepsilon$ enters only through

$$\rho = 1 - 2\varepsilon.$$

---

## Tracking Only the Imbalance

A further simplification is to divide out the total mass and track only the imbalance

$$r_t = \frac{d_t}{m_t}.$$

Then

$$m_t = \frac{1}{2}(u_t + \rho v_t r_{t-1}) \, m_{t-1},$$

$$r_t = \frac{v_t + \rho u_t r_{t-1}}{u_t + \rho v_t r_{t-1}}.$$

So the sequence probability is

$$P(x_{1:T}) = m_T = m_1 \prod_{t=2}^{T} \frac{u_t + \rho v_t r_{t-1}}{2},$$

with

$$m_1 = \pi_A e_A(x_1) + \pi_B e_B(x_1),$$

$$r_1 = \frac{\pi_A e_A(x_1) - \pi_B e_B(x_1)}{\pi_A e_A(x_1) + \pi_B e_B(x_1)}.$$

That is not a closed form in raw counts, but it is an exact one-dimensional nonlinear recursion.

---

## Special Limits

**For $\varepsilon = 0$**, the hidden state never changes, so

$$P(x_{1:T}) = \pi_A \prod_{t=1}^{T} e_A(x_t) + \pi_B \prod_{t=1}^{T} e_B(x_t).$$

This is just a mixture of two i.i.d. models.

**For $\varepsilon = \frac{1}{2}$**, $\rho = 0$, and the recursion collapses:

$$m_t = \frac{u_t}{2} m_{t-1}.$$

Hence

$$P(x_{1:T}) = \big[\pi_A e_A(x_1) + \pi_B e_B(x_1)\big] \prod_{t=2}^{T} \frac{e_A(x_t) + e_B(x_t)}{2}.$$

If you also start in stationarity, $\pi_A = \pi_B = \frac{1}{2}$, then the whole observed sequence is i.i.d. with token law

$$q(k) = \frac{e_A(k) + e_B(k)}{2},$$

so

$$P(x_{1:T}) = \prod_{t=1}^{T} q(x_t).$$

---

## The Most Useful Takeaway

$$\text{transition effect} \iff \rho = 1 - 2\varepsilon.$$

When $\varepsilon \approx \frac{1}{2}$, the imbalance mode is weak and the model is close to an i.i.d. mixture.
When $\varepsilon \approx 0$, the model is close to a mixture of long single-state runs.

---

## Rare-Switch Expansion

For $\varepsilon \ll 1$, the natural simplification is an expansion in the **number of hidden switches**.

Let

$$a_t := e_A(x_t), \qquad b_t := e_B(x_t),$$

for the observed sequence $x_{1:T}$.

Also define segment products

$$A_{u:v} := \prod_{t=u}^{v} a_t, \qquad B_{u:v} := \prod_{t=u}^{v} b_t.$$

The exact likelihood is

$$P(x_{1:T}) = \sum_{z_{1:T}} \pi_{z_1} \prod_{t=2}^{T} Q_{z_{t-1}, z_t} \prod_{t=1}^{T} e_{z_t}(x_t),$$

with

$$Q = \begin{pmatrix} 1-\varepsilon & \varepsilon \\ \varepsilon & 1-\varepsilon \end{pmatrix}.$$

Because every switch contributes a factor $\varepsilon$, and every non-switch contributes a factor $1-\varepsilon$, you can group terms by the number $K$ of switches:

$$P(x_{1:T}) = \sum_{K=0}^{T-1} \varepsilon^K (1-\varepsilon)^{T-1-K} \, S_K(x_{1:T}),$$

where $S_K$ is the total emission weight of all hidden paths with exactly $K$ switches.

That is the rare-transition expansion.

---

## First Few Terms

### No hidden switch: $K = 0$

There are only two possibilities: always in $A$, or always in $B$. So

$$S_0 = \pi_A A_{1:T} + \pi_B B_{1:T}.$$

Hence the leading term is

$$P(x_{1:T}) = (1-\varepsilon)^{T-1} \big[\pi_A A_{1:T} + \pi_B B_{1:T}\big] + O(\varepsilon).$$

So as $\varepsilon \to 0$, the HMM becomes just a **mixture of two i.i.d. models**.

### One hidden switch: $K = 1$

If the chain starts in $A$ and switches once after time $s$, the weight is

$$\pi_A \, A_{1:s} B_{s+1:T}.$$

If it starts in $B$ and switches once after time $s$, the weight is

$$\pi_B \, B_{1:s} A_{s+1:T}.$$

Summing over all switch positions $s = 1, \ldots, T-1$,

$$S_1 = \pi_A \sum_{s=1}^{T-1} A_{1:s} B_{s+1:T} + \pi_B \sum_{s=1}^{T-1} B_{1:s} A_{s+1:T}.$$

So keeping up to one switch gives

$$P(x_{1:T}) \approx (1-\varepsilon)^{T-1} S_0 + \varepsilon (1-\varepsilon)^{T-2} S_1.$$

This is usually the best "small-$\varepsilon$" approximation when $T\varepsilon \ll 1$.

### Two hidden switches: $K = 2$

For completeness,

$$S_2 = \pi_A \sum_{1 \leq r < s \leq T-1} A_{1:r} B_{r+1:s} A_{s+1:T} + \pi_B \sum_{1 \leq r < s \leq T-1} B_{1:r} A_{r+1:s} B_{s+1:T}.$$

So the exact expansion is

$$P(x_{1:T}) = (1-\varepsilon)^{T-1} S_0 + \varepsilon (1-\varepsilon)^{T-2} S_1 + \varepsilon^2 (1-\varepsilon)^{T-3} S_2 + \cdots$$

---

## First-Order Taylor Form

If you want a literal expansion in powers of $\varepsilon$, expand the prefactors too:

$$(1-\varepsilon)^{T-1} = 1 - (T-1)\varepsilon + O(\varepsilon^2),$$

so

$$P(x_{1:T}) = S_0 + \varepsilon \big[S_1 - (T-1) S_0\big] + O(\varepsilon^2).$$

That is the fixed-$T$, $\varepsilon \to 0$ asymptotic series.

---

## A Useful Computational Simplification

Define prefix products

$$A_s := A_{1:s} = \prod_{t=1}^{s} a_t, \qquad B_s := B_{1:s} = \prod_{t=1}^{s} b_t.$$

Then

$$A_{1:s} B_{s+1:T} = A_s \frac{B_T}{B_s}, \qquad B_{1:s} A_{s+1:T} = B_s \frac{A_T}{A_s},$$

so

$$S_1 = \pi_A B_T \sum_{s=1}^{T-1} \frac{A_s}{B_s} + \pi_B A_T \sum_{s=1}^{T-1} \frac{B_s}{A_s},$$

whenever those ratios are well-defined.

In log form, define

$$\ell_s := \sum_{t=1}^{s} \log \frac{a_t}{b_t}.$$

Then

$$\frac{A_s}{B_s} = e^{\ell_s},$$

and

$$S_1 = \pi_A B_T e^{\sum_{t=1}^{T} \log b_t} \sum_{s=1}^{T-1} e^{\ell_s} + \pi_B A_T e^{\sum_{t=1}^{T} \log a_t} \sum_{s=1}^{T-1} e^{-\ell_s}.$$

This makes the one-switch term very interpretable: it is controlled by the cumulative evidence favoring $A$ versus $B$.

---

## Interpretation

The dominant zero-switch term is

$$\pi_A \prod_{t=1}^{T} e_A(x_t) + \pi_B \prod_{t=1}^{T} e_B(x_t),$$

which says:
- either the whole sequence was generated in state $A$,
- or the whole sequence was generated in state $B$.

The one-switch correction says:
- maybe the sequence starts looking like $A$,
- then at some hidden time $s$, it starts looking like $B$,
- or vice versa.

So in the rare-transition limit the sequence is naturally described by:
- a starting state,
- a small number of switch points,
- long emission segments between switches.

---

## When This Approximation is Valid

The expected number of hidden switches in $T - 1$ steps is about

$$(T-1)\varepsilon.$$

So:
- if $(T-1)\varepsilon \ll 1$, keeping only $S_0$ or $S_0 + S_1$ is very good;
- if $T\varepsilon = O(1)$, you need several $S_K$ terms;
- if $T\varepsilon \gg 1$, this rare-switch expansion is no longer the right simplification.

That last case is where the Gaussian / long-window description starts to re-emerge.

---

## For $\varepsilon = 1$

The hidden state alternates deterministically, giving

$$P(x_{1:T}) = \pi_A \prod_{t \text{ odd}} e_A(x_t) \prod_{t \text{ even}} e_B(x_t) + \pi_B \prod_{t \text{ odd}} e_B(x_t) \prod_{t \text{ even}} e_A(x_t).$$

One more important point: for a completely arbitrary observed sequence and generic emissions, there is usually no further collapse to a formula depending only on token counts, because the matrices $D_{x_t} Q$ for different symbols do not commute. So the exact transfer-matrix recursion is already the canonical closed form. The real simplifications come from special $\varepsilon$ values, small-$\varepsilon$ expansions, or special emission structure.
