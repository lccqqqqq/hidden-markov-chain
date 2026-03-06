## Introduction

A **Hidden Markov Model** (HMM) is a doubly stochastic process consisting of an underlying Markov chain whose states are not directly observable (the "hidden" states), coupled with a set of state-dependent observation distributions that produce the data we *do* observe. At each discrete time step $t$, the system occupies a hidden state $Z_{t}$ and emits an observable token $X_{t}$ drawn from a distribution conditioned on $Z_{t}$.

HMMs were introduced by Baum and Petrie (1966)[^bp66] in the context of statistical inference on probabilistic functions of Markov chains, with subsequent foundational contributions by Baum, Petrie, Soules, and Weiss (1970). The Viterbi algorithm for optimal decoding was developed independently by Viterbi (1967) for convolutional codes. The framework gained widespread engineering adoption through work by Baker (1975) and Jelinek (1976) on automatic speech recognition at IBM, and was made broadly accessible by Rabiner's landmark tutorial (1989)[^rab89], which organized HMM inference into three canonical problems and remains the standard pedagogical reference.

The central appeal of HMMs for studying neural sequence models is that they provide a **minimal, analytically tractable generative process** with nontrivial sequential structure: the hidden states are Markovian, but the marginal process over observations is *not*. This means that optimal next-token prediction — the task transformers are trained on — requires the predictor to perform nontrivial inference over latent variables. HMMs thus serve as a controlled testbed to ask: *what internal representations does a transformer learn when it learns to predict sequences with hidden structure?*

[^bp66]: L.E. Baum and T. Petrie, "Statistical Inference for Probabilistic Functions of Finite State Markov Chains," *Annals of Mathematical Statistics*, 37(6):1554–1563, 1966.
[^rab89]: L.R. Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition," *Proc. IEEE*, 77(2):257–286, 1989.

## Motivation

- Language models are trained over <u>next-token prediction</u> $\to$ pretrained model speaks fluent English, "emergent" skills like grammar, keeping consistent tones, etc.
- Predicting the next token means essentially the model understand the underlying reality that led to the creation of that token.
- A toy model that sharpens this:
	- Stochastic generative model with analytic handle

## Structure of HMM

- **State space** $Z_{t}$ from a discrete set $S$ with size $N$.
- **Observations** $X_{t}$ from another discrete set $\Omega$ with size $K$.
- **Markovian Assumption**: The state is Markovian, $P(Z_{t}|Z_{<t})=P(Z_{t}|Z_{t-1})$.
- **Observation Independence Assumption**: The emission probability depends only on the current state, $P(X_{t}|Z_{t}, Z_{<t}) = P(X_{t}|Z_{t})$.
- Note: The hidden states are Markovian, but the marginals on the observation sequences are not.
- Assign model parameters:
$$
A_{ij}=P(Z_{t+1}=j \ | \   Z_{t}=i), \quad B_{i\mu} \equiv B_{i}(X_{t}=\mu) = P(X_{t}=\mu \ | \   Z_{t}=i)
$$
Also the initial parameters $\pi$, a distribution over the initial hidden state $Z_{1}$. An HMM is characterized by the triple $\lambda \equiv(A, B, \pi)$.[^1]

The joint probability of a state sequence $Z_{1:T}$ and observation sequence $X_{1:T}$ factorizes as:
$$
P(X_{1:T}, Z_{1:T} \mid \lambda) = \pi_{Z_{1}} B_{Z_{1}}(X_{1}) \prod_{t=2}^{T} A_{Z_{t-1}, Z_{t}} \, B_{Z_{t}}(X_{t})
$$

## Background: The Three Fundamental Problems

Rabiner (1989) organized HMM computation into three canonical problems:

| Problem | Given | Compute | Algorithm |
|---|---|---|---|
| **Evaluation** | $\lambda$, $X_{1:T}$ | $P(X_{1:T} \mid \lambda)$ | Forward algorithm |
| **Decoding** | $\lambda$, $X_{1:T}$ | $\arg\max_{Z_{1:T}} P(Z_{1:T} \mid X_{1:T}, \lambda)$ | Viterbi algorithm |
| **Learning** | $X_{1:T}$ | $\arg\max_{\lambda} P(X_{1:T} \mid \lambda)$ | Baum-Welch (EM) |

### The Forward Algorithm

The evaluation problem asks: given model parameters $\lambda = (A, B, \pi)$ and an observation sequence $X_{1:T}$, what is $P(X_{1:T} \mid \lambda)$? A naive marginalization over all hidden state sequences requires summing $N^{T}$ terms — computationally intractable. The forward algorithm solves this in $O(N^{2}T)$ via dynamic programming.

**Forward variable.** Define $\alpha_{t}(j) := P(X_{1:t}, Z_{t} = j \mid \lambda)$, the joint probability of observing the partial sequence $X_{1:t}$ and being in state $j$ at time $t$.

**Initialization** ($t=1$):
$$
\alpha_{1}(j) = \pi_{j} \, B_{j}(X_{1}), \quad j = 1, \ldots, N
$$

**Recursion** ($t = 1, \ldots, T-1$):
$$
\alpha_{t+1}(j) = \left[ \sum_{i=1}^{N} \alpha_{t}(i) \, A_{ij} \right] B_{j}(X_{t+1})
$$
*Interpretation:* to be in state $j$ at $t+1$ having seen $X_{1:t+1}$, we sum over all predecessor states $i$ at time $t$ (each weighted by their forward probability and the transition $A_{ij}$), then multiply by the emission $B_{j}(X_{t+1})$.

**Termination:**
$$
P(X_{1:T} \mid \lambda) = \sum_{j=1}^{N} \alpha_{T}(j)
$$

In matrix-vector form, writing $\boldsymbol{\alpha}_{t}$ as a row vector and $D_{t} = \mathrm{diag}(B_{1}(X_{t}), \ldots, B_{N}(X_{t}))$:
$$
\boldsymbol{\alpha}_{T} = \boldsymbol{\pi}^{\top} D_{1} \, A \, D_{2} \, A \, D_{3} \cdots A \, D_{T}
$$
This matrix product representation is important for the connection to transformer computations.

**Complexity:** $O(N^{2}T)$ time, $O(NT)$ space if storing all $\alpha$ values (needed for Baum-Welch), or $O(N)$ if only the final likelihood is needed. For long sequences, $\alpha_{t}(j)$ underflows exponentially; standard remedies include log-space computation (log-sum-exp trick) or Rabiner's scaling coefficients.

### The Backward Algorithm

Define the backward variable $\beta_{t}(i) := P(X_{t+1:T} \mid Z_{t} = i, \lambda)$, the probability of future observations given current state $i$.

**Initialization** ($t = T$): $\beta_{T}(i) = 1$ for all $i$.

**Recursion** ($t = T-1, \ldots, 1$):
$$
\beta_{t}(i) = \sum_{j=1}^{N} A_{ij} \, B_{j}(X_{t+1}) \, \beta_{t+1}(j)
$$

**Key identity** (holds for any $t$):
$$
P(X_{1:T} \mid \lambda) = \sum_{j=1}^{N} \alpha_{t}(j) \, \beta_{t}(j)
$$

Together, forward and backward variables yield the **posterior state occupancy** (smoothed estimate using the *full* observation sequence):
$$
\gamma_{t}(j) := P(Z_{t} = j \mid X_{1:T}, \lambda) = \frac{\alpha_{t}(j) \, \beta_{t}(j)}{\sum_{l} \alpha_{t}(l) \, \beta_{t}(l)}
$$
and the **posterior transition probability**:
$$
\xi_{t}(i, j) := P(Z_{t}=i, Z_{t+1}=j \mid X_{1:T}, \lambda) = \frac{\alpha_{t}(i) \, A_{ij} \, B_{j}(X_{t+1}) \, \beta_{t+1}(j)}{P(X_{1:T} \mid \lambda)}
$$

### The Viterbi Algorithm

The Viterbi algorithm finds the single most probable hidden state sequence $Z^{*}_{1:T} = \arg\max_{Z_{1:T}} P(Z_{1:T} \mid X_{1:T}, \lambda)$. It mirrors the forward algorithm but replaces **sum** with **max**.

**Viterbi variable:** $\delta_{t}(j) := \max_{Z_{1}, \ldots, Z_{t-1}} P(Z_{1:t-1}, Z_{t}=j, X_{1:t} \mid \lambda)$.

**Initialization:** $\delta_{1}(j) = \pi_{j} B_{j}(X_{1})$, $\psi_{1}(j) = 0$.

**Recursion** ($t = 2, \ldots, T$):
$$
\delta_{t}(j) = \max_{i} \left[ \delta_{t-1}(i) \, A_{ij} \right] B_{j}(X_{t}), \qquad \psi_{t}(j) = \arg\max_{i} \left[ \delta_{t-1}(i) \, A_{ij} \right]
$$
where $\psi_{t}(j)$ records the backpointer to the best predecessor state.

**Termination:** $Z^{*}_{T} = \arg\max_{j} \delta_{T}(j)$.

**Backtracking** ($t = T-1, \ldots, 1$): $Z^{*}_{t} = \psi_{t+1}(Z^{*}_{t+1})$.

**Complexity:** $O(N^{2}T)$ time, $O(NT)$ space. In practice, Viterbi is implemented in log-space where products become sums and max is numerically stable.

*Note:* Viterbi decoding finds the *jointly* most probable sequence, which differs from **posterior (MPM) decoding** $\hat{Z}_{t} = \arg\max_{j} \gamma_{t}(j)$ that minimizes expected per-symbol errors but may produce impossible transitions.

### The Baum-Welch Algorithm (EM)

The Baum-Welch algorithm is the Expectation-Maximization (EM) algorithm specialized to HMMs. It iteratively re-estimates $\lambda = (A, B, \pi)$ to find a local maximum of $P(X_{1:T} \mid \lambda)$.

**E-step:** Run forward-backward to compute $\gamma_{t}(j)$ and $\xi_{t}(i,j)$ using the current $\lambda$.

**M-step:** Re-estimate parameters:
$$
\pi_{j}^{\,\mathrm{new}} = \gamma_{1}(j)
$$
$$
A_{ij}^{\,\mathrm{new}} = \frac{\sum_{t=1}^{T-1} \xi_{t}(i,j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)}
$$
$$
B_{j}^{\,\mathrm{new}}(\mu) = \frac{\sum_{t=1}^{T} \gamma_{t}(j) \, \mathbf{1}[X_{t}=\mu]}{\sum_{t=1}^{T} \gamma_{t}(j)}
$$

Each iteration is guaranteed to increase (or leave unchanged) the log-likelihood. Convergence is to a **local** maximum; multiple random restarts are standard practice. Complexity is $O(N^{2}T)$ per iteration. The algorithm was developed by Baum et al. (1970) and later recognized as a special case of the general EM framework of Dempster, Laird, and Rubin (1977).

### Algorithm, belief states

- Task of transformer: predict $P(X_{t}|X_{<t})$.
- <span style='color:red'>What computation is required in this prediction?</span> <u>The Forward-backward algorithm</u> is at the foundation of HMM theory, that *predicts $P(X_{1:T})$ given any HMM parameters $\lambda$* (and the next-token probability is deduced from marginalization)
	- Helper "forward function" $\alpha_{t}(j):= P(X_{1:t},Z_{t}=j)$ - then $P(X_{1:T}) = \sum_{j=1}^{N} \alpha_{t}(j)$.
	- The forward function obeys the following recursive update rule: $\alpha_{t+1}(j) = \sum_{i=1}^{N} \alpha_{t}(i) A_{ij}B_{j}(X_{t+1})$.
	- Define <u>belief states</u>: $\eta_{t}(j) := P(Z_{t}=j|X_{1:t}) = \frac{\alpha_{t}(j)}{\sum_{l=1}^{N}\alpha_{t}(l)}$.
	- Writing in vector form, there's a recursive update rule for the belief states as well.
$$
\boldsymbol{\eta}_{t+1} = \frac{\boldsymbol{\eta}_{t} T^{(X_{t+1})}}{\lVert \boldsymbol{\eta}_{t} T^{(X_{t+1})} \rVert }
$$
where $T^{(x)}_{ij} = A_{ij}B_{j}(x)$ is the *symbol-labeled transition matrix* (combining transition and emission into a single matrix for each observable symbol $x$), and $\lVert \cdot \rVert$ denotes the $L^{1}$ norm (i.e., sum of entries), which ensures $\boldsymbol{\eta}_{t+1}$ remains a valid probability distribution. The denominator $\langle \boldsymbol{\eta}_{t} \mid T^{(x)} \mid \mathbf{1} \rangle$ is precisely the predictive probability $P(X_{t+1}=x \mid X_{1:t})$.

**Belief states are sufficient statistics:** $P(X_{t+1} \mid X_{1:t})$ depends on $X_{1:t}$ only through $\boldsymbol{\eta}_{t}$. Any system that predicts optimally must implicitly compute a representation equivalent to the belief state.

#### Geometry: probability simplex, IFS, and fractals

The belief state $\boldsymbol{\eta}_{t}$ is a probability distribution over $N$ hidden states and therefore lives on the $(N{-}1)$-dimensional **probability simplex** $\Delta^{N-1}$. The belief update defines a family of maps $\{f^{(x)}\}_{x \in \Omega}$ on the simplex — one for each observable token — where
$$
f^{(x)}(\boldsymbol{\eta}) = \frac{\boldsymbol{\eta} \, T^{(x)}}{\lVert \boldsymbol{\eta} \, T^{(x)} \rVert}
$$
Each map is a composition of a linear map (multiplication by $T^{(x)}$) followed by a projective normalization. For positive/irreducible $T^{(x)}$, these are **contraction mappings** under the Hilbert projective metric on the simplex. The collection $\{f^{(x)}\}$ constitutes a *place-dependent* **Iterated Function System** (IFS), and by the contraction mapping theorem, the IFS possesses a unique compact invariant set — the **attractor** $\Lambda \subset \Delta^{N-1}$, which is exactly the closure of all reachable belief states.

For *nonunifilar* HMMs (where a given state can emit the same symbol while transitioning to multiple distinct states), the attractor is generically an uncountably infinite set with nontrivial **fractal** structure (Cantor sets, Sierpinski-like triangles, etc.). The fractal dimension of $\Lambda$ is an intrinsic property of the HMM process.

>[!note|Paper TLDR] Conjecture
> For the model to predict $P(X_{t}|X_{<t})$, as it digest the input sequence the model should form the belief states inside the model, or "an understanding of the underlying state of the world model."


[^1]: Note that in literature this is sometimes represented by one emission matrix $T_{\mu ij} = P(X_{t},Z_{t+1}|Z_{t})$. But applying Bayes rule (along with the assumptions of HMM) directly gives $T_{\mu ji}=A_{ij}B_{i\mu}$.
