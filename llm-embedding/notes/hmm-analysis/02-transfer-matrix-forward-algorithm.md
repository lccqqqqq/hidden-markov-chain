# Transfer Matrix / Forward Algorithm for 2-State HMM

## Setup

Suppose I have a 2-state HMM, such that the state transition is $1-\varepsilon$ for self-transition and $\varepsilon$ for other-transition. Each state has its own categorical emission probabilities over $N$ tokens. Can I apply the transfer matrix/forward algorithm and write down analytical, closed form expression for sequence probabilities?

---

## Transfer-Matrix Form

For a 2-state HMM with hidden states $A, B$, transition matrix

$$Q = \begin{pmatrix} 1 - \varepsilon & \varepsilon \\ \varepsilon & 1 - \varepsilon \end{pmatrix},$$

and categorical emissions

$$e_A(k) = P(x_t = k \mid z_t = A), \qquad e_B(k) = P(x_t = k \mid z_t = B), \qquad k = 1, \ldots, N,$$

the probability of any observed sequence $x_{1:T} = (x_1, \ldots, x_T)$ is exactly given by a $2 \times 2$ transfer-matrix product.

Let

$$D_k = \text{diag}(e_A(k), e_B(k)).$$

If the initial hidden-state distribution is $\pi = (\pi_A, \pi_B)^T$, then

$$P(x_{1:T}) = \mathbf{1}^T D_{x_T} Q D_{x_{T-1}} Q \cdots D_{x_2} Q \, D_{x_1} \pi,$$

where $\mathbf{1} = (1, 1)^T$.

That is the forward algorithm in closed matrix form.

Equivalently, defining the forward vector

$$\alpha_1 = D_{x_1} \pi, \qquad \alpha_t = D_{x_t} Q \, \alpha_{t-1},$$

you get

$$P(x_{1:T}) = \mathbf{1}^T \alpha_T.$$

So the answer is:
- **yes**, the transfer-matrix / forward-algorithm form is exact and analytic;
- **but** for a completely arbitrary sequence $x_{1:T}$, this matrix product is usually the simplest closed form.

The reason you usually cannot simplify further is that the matrices

$$M_k := D_k Q$$

depend on the observed token $k$, and different $M_k$'s generally do **not commute**. So for a generic sequence there is no single scalar formula depending only on counts of tokens.

There are, however, useful special cases.

If the sequence is a run of the same token $k$ repeated $L$ times, then

$$P(k^L) = \mathbf{1}^T (D_k Q)^{L-1} D_k \pi,$$

and now you can diagonalize the single $2 \times 2$ matrix $D_k Q$. Its eigenvalues are

$$\lambda_{\pm}(k) = \frac{(1-\varepsilon)(e_A(k) + e_B(k)) \pm \sqrt{(1-\varepsilon)^2(e_A(k) - e_B(k))^2 + 4\varepsilon^2 e_A(k) e_B(k)}}{2}.$$

So in that case $P(k^L)$ is a linear combination of $\lambda_+(k)^{L-1}$ and $\lambda_-(k)^{L-1}$.

---

## Sum/Difference Basis

Because your transition matrix is symmetric, it helps to work in the sum/difference basis. If

$$\alpha_t = \begin{pmatrix} a_t \\ b_t \end{pmatrix}, \qquad m_t = a_t + b_t, \quad d_t = a_t - b_t,$$

and for the observed symbol $x_t$ you write

$$u_t = e_A(x_t) + e_B(x_t), \qquad v_t = e_A(x_t) - e_B(x_t),$$

then with $\rho = 1 - 2\varepsilon$,

$$\begin{pmatrix} m_t \\ d_t \end{pmatrix} = \frac{1}{2} \begin{pmatrix} u_t & \rho v_t \\ v_t & \rho u_t \end{pmatrix} \begin{pmatrix} m_{t-1} \\ d_{t-1} \end{pmatrix},$$

and the sequence probability is just

$$P(x_{1:T}) = m_T.$$

So there is a very explicit 2-dimensional recursion.

---

## Cleanest Summary

$$P(x_{1:T}) = \sum_{z_{1:T} \in \{A,B\}^T} \pi_{z_1} \, \varepsilon^{N_{\text{switch}}(z)} (1-\varepsilon)^{T-1-N_{\text{switch}}(z)} \prod_{t=1}^{T} e_{z_t}(x_t),$$

and the transfer-matrix formula is the efficient closed-form evaluation of that sum.
