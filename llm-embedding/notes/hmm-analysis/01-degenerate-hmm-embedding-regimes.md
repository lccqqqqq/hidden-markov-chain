# Two-State HMM: Embedding Regimes (Degenerate Emissions)

## Setup

Consider a two-state hidden Markov model. The states are A and B. The state transition probabilities are $P_{AA}$ for A goes to A, $P_{BB}$ for B goes to B, $P_{AB}$ for A goes to B, and $P_{BA}$ for B goes to A. When state A is unchanged, a token zero is emitted. If A changes to B, again a token zero is emitted. If state B goes to B, one is emitted, and B goes to A, one is emitted.

The question: analyze the limit of rapid transitions (based on prior notes), then extend to the limit where $P_{AA}$ and $P_{BB}$ are close to 1 so transitions are rare. In the latter limit, is there a natural embedding and network?

---

## Key Simplification

The emitted token depends only on whether the current hidden state is A or B, not on whether a transition occurred. So the observed binary process is itself just a two-state Markov chain. This "HMM" is degenerate: the hidden state is perfectly revealed by the token.

There are really **two distinct rare-transition regimes**, and they lead to different "natural" representations.

---

## Model in Clean Form

Let $z_t \in \{A, B\}$ be the hidden state, with transition matrix

$$T = \begin{pmatrix} P_{AA} & P_{AB} \\ P_{BA} & P_{BB} \end{pmatrix}, \qquad P_{AA} + P_{AB} = 1, \quad P_{BA} + P_{BB} = 1.$$

Emission rule:
- if the current state is $A$, emit 0,
- if the current state is $B$, emit 1.

So the emitted token does **not** actually depend on whether the state changed; it depends only on which state you were in. Therefore the observed process $x_t \in \{0, 1\}$ is itself just a two-state Markov chain:

$$x_t = 0 \iff z_t = A, \qquad x_t = 1 \iff z_t = B.$$

---

## 1. Basic structure of the observed process

Write

$$a := P_{AB}, \qquad b := P_{BA},$$

so that

$$P_{AA} = 1 - a, \qquad P_{BB} = 1 - b.$$

The stationary probabilities are

$$\pi_A = \frac{b}{a+b}, \qquad \pi_B = \frac{a}{a+b}.$$

Since token 1 means state $B$,

$$\mathbb{E}[x_t] = \pi_B.$$

The nontrivial eigenvalue of the chain is

$$\lambda = P_{AA} + P_{BB} - 1 = 1 - a - b.$$

This is the persistence parameter. The correlation time is of order

$$\tau_c \sim \frac{1}{1 - \lambda} = \frac{1}{a + b}.$$

So:
- **rapid transitions** means $a + b$ is not small, equivalently $\lambda$ not close to 1;
- **rare transitions** means $a, b \ll 1$, equivalently $\lambda \approx 1$.

For this two-state chain one has

$$\text{Cov}(x_t, x_{t+\tau}) = \pi_A \pi_B \, \lambda^{|\tau|}.$$

So the count of ones in a window of length $L$,

$$N_1 = \sum_{t=1}^{L} x_t,$$

has

$$\text{Var}(N_1) = L \sigma_{\text{eff}}^2 + O(1),$$

with long-run variance

$$\sigma_{\text{eff}}^2 = \pi_A \pi_B \left(1 + 2 \sum_{\tau \geq 1} \lambda^\tau\right) = \pi_A \pi_B \frac{1 + \lambda}{1 - \lambda} = \pi_A \pi_B \frac{2 - (a+b)}{a+b}.$$

This is exactly the kind of long-run covariance object your note identifies as the source of the canonical embedding in the Gaussian window limit. In that framework, one factorizes the long-run covariance $\Sigma = E E^\top$, and the corresponding count-model is realized by an embedding lookup, sum pooling, and a quadratic head. The note also emphasizes that low-rank HMM structure gives a low-rank dependence direction and that order corrections sit beyond the bag-of-words Gaussian term.

---

## 2. The Gaussian / rapid-mixing regime

If your window is long compared to the correlation time,

$$L \gg \tau_c \sim \frac{1}{a+b},$$

then the CLT picture applies well. In that case

$$\frac{N_1 - L\pi_B}{\sqrt{L}}$$

is approximately Gaussian.

Because the vocabulary has only two tokens, the count vector

$$N = (N_0, N_1), \qquad N_0 + N_1 = L,$$

has only **one nontrivial direction**. So the covariance embedding is automatically one-dimensional.

A convenient centered scalar feature is

$$\phi(0) = -\pi_B, \qquad \phi(1) = \pi_A,$$

since then

$$\sum_{t=1}^{L} \phi(x_t) = N_1 - L\pi_B.$$

The whitened feature is

$$\psi(0) = -\frac{\pi_B}{\sigma_{\text{eff}}}, \qquad \psi(1) = \frac{\pi_A}{\sigma_{\text{eff}}},$$

so that

$$q = \frac{1}{\sqrt{L}} \sum_{t=1}^{L} \psi(x_t)$$

has asymptotic variance 1.

So in the rapid-mixing / Gaussian regime, the natural embedding is just a **scalar phase embedding**:

$$0 \mapsto -c, \qquad 1 \mapsto +d,$$

with the centered-whitened choice above being the most canonical.

Then the natural network is exactly the one from your note, specialized to rank one:

$$\text{embedding lookup} \to \text{sum} \to \text{scalar quadratic head}.$$

Equivalently,

$$\log p(\text{window}) \approx c_0 - \frac{1}{2} q^2.$$

So in that regime the answer is very clean:
- **natural embedding**: one-dimensional centered/whitened token feature;
- **natural network**: Deep Sets with a scalar quadratic head.

---

## 3. What changes when transitions are rare

Now suppose

$$a, b \ll 1, \qquad \lambda = 1 - a - b \approx 1.$$

Then the correlation time

$$\tau_c \sim \frac{1}{a+b}$$

becomes very large. This creates a sharp distinction.

### Regime A: $L(a+b) \gg 1$

If the window is still much longer than the typical run length, then the Gaussian picture survives. The only difference is that the variance is large:

$$\sigma_{\text{eff}}^2 \sim \frac{2\pi_A \pi_B}{a+b}.$$

So the same one-dimensional embedding remains natural. The bag-of-words model still works.

### Regime B: $L(a+b) \lesssim 1$

This is the genuinely interesting rare-transition regime.

Now a typical window contains **zero or one switch**, not many. So the count distribution is no longer close to a single Gaussian.

In fact, if $L(a+b) \ll 1$, then with probability $1 - O(L(a+b))$, the whole window is all zeros or all ones:
- all zeros with probability approximately $\pi_A$,
- all ones with probability approximately $\pi_B$,

and the first correction comes from windows with one switch:

$$000\cdots 011\cdots 1 \quad \text{or} \quad 111\cdots 100\cdots 0.$$

So the count $N_1$ is approximately **bimodal**, with mass near 0 and $L$, plus a small bridge from one-switch windows.

That means:
- a **single Gaussian** on counts is not the right approximation;
- a single quadratic Deep Sets head is no longer the natural architecture.

This is exactly the kind of situation where the note's permutation-invariant Gaussian approximation stops being the right leading description, because order effects are no longer a small correction.

---

## 4. Exact sequence model in the rare-transition regime

Because the observation reveals the state, the exact sequence likelihood is especially simple.

Let

$$N_{ij} = \#\{t = 1, \ldots, L-1 : x_t = i, \, x_{t+1} = j\}, \qquad i, j \in \{0, 1\}.$$

Then

$$\log P(x_{1:L}) = \log \pi_{x_1} + N_{00} \log P_{AA} + N_{01} \log P_{AB} + N_{10} \log P_{BA} + N_{11} \log P_{BB}.$$

So the exact sufficient statistics are:
- the initial token $x_1$,
- the adjacent-pair counts $N_{00}, N_{01}, N_{10}, N_{11}$.

In the rare-transition limit, the most important ones are the switch counts

$$N_{01}, \qquad N_{10}.$$

So the most natural representation is no longer "how many zeros and ones are in the window," but rather:
- which phase you started in,
- how long you stayed there,
- how many times you switched.

That is the metastable description.

---

## 5. Natural embedding and network in the rare-transition regime

### If you only care about long-window bag-of-words statistics

Then the natural embedding is still **one scalar dimension**, because there are only two tokens and one nontrivial covariance direction. The network is still the Gaussian Deep Sets network from the notes:

$$\text{lookup} \to \text{sum} \to \text{quadratic head}.$$

But this is only natural if

$$L \gg \frac{1}{a+b}.$$

### If you care about the true rare-transition regime at fixed or moderate $L$

Then the natural latent variable is the **phase/run state**, not just the count.

The exact minimal model is a first-order Markov chain on the observed symbols:

$$\log P(x_{1:L}) = u_{x_1} + \sum_{t=1}^{L-1} J_{x_t, x_{t+1}},$$

where

$$u_0 = \log \pi_A, \quad u_1 = \log \pi_B,$$

and

$$J = \begin{pmatrix} \log P_{AA} & \log P_{AB} \\ \log P_{BA} & \log P_{BB} \end{pmatrix}.$$

In neural-network language, that is equivalent to a tiny recurrent or pairwise model:
- either a **2-state RNN** whose hidden state stores the current phase,
- or a **bigram / pair-potential network**,
- or a **1D convolution on adjacent pairs**.

That is the natural network in the sticky regime.

The natural embedding in that regime is therefore:
- token embedding for phase: $0 \mapsto A$, $1 \mapsto B$, or numerically $0 \mapsto -1$, $1 \mapsto +1$;
- plus a transition-sensitive mechanism that scores $(x_t, x_{t+1})$.

So the right answer is **not** just "an embedding," but "an embedding plus memory of the previous token."

---

## 6. The conceptual summary

There is a crossover controlled by

$$L(a+b).$$

If $L(a+b) \gg 1$, many switches occur inside a window. Then:
- counts are approximately Gaussian,
- the note's covariance embedding applies directly,
- the natural model is 1D embedding + sum pooling + quadratic head.

If $L(a+b) \ll 1$, switches are rare within a window. Then:
- counts are bimodal or mixture-like, not Gaussian,
- order is essential,
- the natural model is a sticky two-state Markov / RNN / pairwise model.

So in the rare-transition limit, there **is** a natural embedding and network, but it is different:
- **embedding**: a one-dimensional phase label is still natural;
- **network**: not Deep Sets, but a sequential model that remembers the current phase and penalizes switches.

The Gaussian embedding from your notes survives only after coarse-graining over many runs.
