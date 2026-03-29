# Unpacking the Forward Recursion

## The Expression

$$P(x_{1:T}) = \mathbf{1}^T D_{x_T} Q D_{x_{T-1}} Q \cdots D_{x_2} Q \, D_{x_1} \pi, \qquad \mathbf{1} = \begin{pmatrix} 1 \\ 1 \end{pmatrix}.$$

---

## What Each Piece Means

Let the hidden state at time $t$ be $z_t \in \{A, B\}$, and observed token be $x_t \in \{1, \ldots, N\}$.

You have

$$Q = \begin{pmatrix} 1-\varepsilon & \varepsilon \\ \varepsilon & 1-\varepsilon \end{pmatrix}, \qquad D_k = \begin{pmatrix} e_A(k) & 0 \\ 0 & e_B(k) \end{pmatrix},$$

where $e_A(k) = P(x_t = k \mid z_t = A)$ and $e_B(k) = P(x_t = k \mid z_t = B)$.

If the initial hidden-state distribution is

$$\pi = \begin{pmatrix} \pi_A \\ \pi_B \end{pmatrix},$$

then

$$P(x_{1:T}) = \mathbf{1}^T D_{x_T} Q D_{x_{T-1}} Q \cdots D_{x_2} Q \, D_{x_1} \pi.$$

**Start with $\pi$.** This is your probability of starting in $A$ or $B$.

**Then $D_{x_1} \pi$** means: weight each starting state by how likely it is to emit the first observed symbol $x_1$.

So

$$D_{x_1} \pi = \begin{pmatrix} e_A(x_1) \pi_A \\ e_B(x_1) \pi_B \end{pmatrix}.$$

That vector is not yet the total probability; it is the pair of contributions from paths that end at hidden state $A$ or $B$ after explaining $x_1$.

**Next multiply by $Q$.** That propagates those contributions forward one step through the hidden-state transition.

**Then multiply by $D_{x_2}$.** That weights by the probability of emitting $x_2$ from the new hidden state.

**Repeat until time $T$.** At the end, sum the two entries with $\mathbf{1}^T$, because the final hidden state could be either $A$ or $B$.

---

## Forward Vector Interpretation

Define

$$\alpha_t = \begin{pmatrix} \alpha_t(A) \\ \alpha_t(B) \end{pmatrix},$$

where

$$\alpha_t(i) = P(x_{1:t}, z_t = i).$$

Then the recursion is

$$\alpha_1 = D_{x_1} \pi,$$

and for $t \geq 2$,

$$\alpha_t = D_{x_t} Q \, \alpha_{t-1}.$$

Finally,

$$P(x_{1:T}) = \mathbf{1}^T \alpha_T = \alpha_T(A) + \alpha_T(B).$$

That is the forward algorithm in matrix form.

---

## Written Componentwise

For two states, this is

$$\alpha_t(A) = e_A(x_t) \big[(1-\varepsilon) \alpha_{t-1}(A) + \varepsilon \, \alpha_{t-1}(B)\big],$$

$$\alpha_t(B) = e_B(x_t) \big[\varepsilon \, \alpha_{t-1}(A) + (1-\varepsilon) \alpha_{t-1}(B)\big].$$

And at the start,

$$\alpha_1(A) = \pi_A e_A(x_1), \qquad \alpha_1(B) = \pi_B e_B(x_1).$$

So each step says:
- take previous mass in $A$, $B$,
- mix it according to the hidden-state transitions,
- then multiply by the emission probability of the observed symbol.

---

## For $T = 2$

The formula becomes

$$P(x_1, x_2) = \mathbf{1}^T D_{x_2} Q D_{x_1} \pi.$$

Expanding:

$$P(x_1, x_2) = e_A(x_2) \big[(1-\varepsilon) \pi_A e_A(x_1) + \varepsilon \, \pi_B e_B(x_1)\big] + e_B(x_2) \big[\varepsilon \, \pi_A e_A(x_1) + (1-\varepsilon) \pi_B e_B(x_1)\big].$$

This is exactly the sum over the four hidden paths:

$$(A \to A), \quad (A \to B), \quad (B \to A), \quad (B \to B).$$

For example, the $A \to B$ path contributes

$$\pi_A \cdot e_A(x_1) \cdot \varepsilon \cdot e_B(x_2).$$

---

## Why This Equals the Sum Over Hidden Paths

The brute-force definition is

$$P(x_{1:T}) = \sum_{z_1, \ldots, z_T \in \{A,B\}} P(z_1) \prod_{t=2}^{T} P(z_t \mid z_{t-1}) \prod_{t=1}^{T} P(x_t \mid z_t).$$

That has $2^T$ terms, one for each hidden-state path.

The matrix product is just a compact way to perform exactly that sum without writing all $2^T$ terms explicitly.

Each multiplication by $Q$ sums over the previous hidden state.
Each multiplication by $D_{x_t}$ inserts the emission factor for the observed symbol.

So the recursion compresses an exponential sum into $O(T)$ matrix-vector multiplies.

---

## One Subtle Convention

There are two common conventions:
1. emit first, then transition
2. transition first, then emit

Your formula

$$D_{x_T} Q \cdots D_{x_2} Q D_{x_1} \pi$$

corresponds to:
- start in $z_1 \sim \pi$,
- emit $x_1$,
- transition,
- emit $x_2$,
- transition,
- etc.

That is perfectly standard. Just be consistent.

---

## Intuition in Words

At each time $t$, the 2-vector $\alpha_t$ stores:
- the probability of having seen the observations so far and currently being in $A$,
- the probability of having seen the observations so far and currently being in $B$.

Then:
- $Q$ moves probability mass between hidden states,
- $D_{x_t}$ rewards or suppresses each hidden state according to how compatible it is with the actual observed token.

At the end, add the two possibilities.

---

## Compact Version Specialized to Your Symmetric Chain

Since

$$Q = \begin{pmatrix} 1-\varepsilon & \varepsilon \\ \varepsilon & 1-\varepsilon \end{pmatrix},$$

the update is just

$$\alpha_t = \begin{pmatrix} e_A(x_t) & 0 \\ 0 & e_B(x_t) \end{pmatrix} \begin{pmatrix} 1-\varepsilon & \varepsilon \\ \varepsilon & 1-\varepsilon \end{pmatrix} \alpha_{t-1}.$$

So you can think of each step as:
- first blur the previous belief by rare switching,
- then reweight by the evidence from the new symbol.
