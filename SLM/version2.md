# Version 2: Beyond Z₂ — Extending the Quadratic Autoregressive Model

> **Prerequisite**: This document builds on the Z₂ quadratic tutorial. See [z2_quadratic_tutorial.md](z2_quadratic_tutorial.md) and its companion scripts ([01_linear_fails.py](01_linear_fails.py), [02_train_full.py](02_train_full.py), [03_train_banded.py](03_train_banded.py), [04_exact_training.py](04_exact_training.py)) for the foundational material.

---

## 1. From Absolute to Relative Indexing

### 1.1 What the original model does

In the v1 tutorial, the quadratic weights C[t, i, j] use **absolute position indices**. The conditional for position t is:

```
Δₜ = bₜ + Σ_{i<t} W[t,i] · xᵢ + Σ_{i<j<t} C[t,i,j] · xᵢ · xⱼ
```

Every position t has its own independent set of weights. For the Z₂ data, the XOR interaction "product of the two positions immediately before t" is represented by three separate parameters:

- C[2, 0, 1] for block 0
- C[5, 3, 4] for block 1
- C[8, 6, 7] for block 2
- ...

The model must independently discover the same XOR pattern in each block. This works — the model has enough parameters — but it treats structurally identical interactions as unrelated.

### 1.2 Relative indexing

Instead, we can parameterize by **relative offsets** (lags) measured backward from the target position:

```
δ = t − i    (how many steps back)
```

The conditional becomes:

```
Δₜ = b + Σ_{δ=1}^{K} W[δ] · x_{t−δ} + Σ_{δ₁>δ₂≥1} C[δ₁,δ₂] · x_{t−δ₁} · x_{t−δ₂}
```

Now the weights are **shared across all positions** — the model is translation-invariant. The XOR interaction "2-back times 1-back" is a single parameter C[2, 1], reused everywhere.

### 1.3 The weight-sharing conflict

> **Script**: [05_relative_indexing.py](05_relative_indexing.py)

With fully shared weights (including bias and linear terms), the model faces a conflict on the Z₂ data. Consider what W[δ=1] must do:

- At **random positions** (0, 1 mod 3): the previous token carries no information about the current one. W[1] should be 0.
- At **XOR positions** (2 mod 3): the previous token is one of the XOR inputs. W[1] should be large (~2L).

A single shared W[1] can't satisfy both. The model compromises:

```
--- Learned parameters ---
bias       = -0.51
W[δ=1]    =  1.19
W[δ=2]    =  1.32
C[δ1=2,δ2=1] = -2.91

--- XOR predictions (position 2) ---
  (0, 0) → P(x₂=1) = 0.37  (target: 0)
  (0, 1) → P(x₂=1) = 0.66  (target: 1)
  (1, 0) → P(x₂=1) = 0.69  (target: 1)
  (1, 1) → P(x₂=1) = 0.29  (target: 0)
```

The predictions are in the right direction but far from sharp. Final loss ≈ 0.63, well above the theoretical minimum of 0.46.

The core issue: the Z₂ data has **position-dependent structure** (positions play different roles within each block), which clashes with translation invariance. The model needs some way to distinguish "which role am I playing in this block?" — either through position-dependent biases, positional encodings, or a different data structure.

---

## 2. The Nonlinearity and Its Limits

(Working with absolute indexing again)
### 2.1 How the sigmoid works

Each conditional in the model is a Bernoulli distribution:

```
P(xₜ = 1 | x_{<t}) = σ(Δₜ)
```

where σ(z) = 1/(1 + e⁻ᶻ) is the **sigmoid** function. It maps the logit Δₜ ∈ ℝ to a probability in (0, 1). For non-binary variables (xₜ ∈ {0, 1, ..., q−1}), the sigmoid would be replaced by **softmax**, which generalizes to multiple classes.

The sigmoid is **monotonic**: σ(Δ) > 0.5 iff Δ > 0. This means the model can only classify xₜ = 1 vs 0 correctly if the polynomial Δₜ itself separates the two cases. The sigmoid converts the polynomial into a probability, but it can't fix a polynomial that gets the sign wrong.

### 2.2 Why pairwise interactions fail for 3-input XOR

> **Script**: [06_three_body_xor.py](06_three_body_xor.py)

Now consider blocks of 4: three random bits and their sum mod 2 (3-input XOR). The logit for the XOR position has 7 free parameters:

```
Δ = b + W₀x₀ + W₁x₁ + W₂x₂ + C₀₁x₀x₁ + C₀₂x₀x₂ + C₁₂x₁x₂
```

Try the pattern that works for 2-input XOR: b = −L, all W = 2L, all C = −4L. Six of eight cases work, but (1,1,1) breaks:

```
Δ(1,1,1) = −L + 2L + 2L + 2L − 4L − 4L − 4L = −7L
```

We need +L (since 1⊕1⊕1 = 1), but get −7L. No choice of the 7 parameters can fix all 8 cases — the function x₀⊕x₁⊕x₂ requires a **cubic** term x₀x₁x₂ that the quadratic model simply doesn't have.

Over the reals, 3-input XOR is:

```
x₀ ⊕ x₁ ⊕ x₂ = x₀ + x₁ + x₂ − 2x₀x₁ − 2x₀x₂ − 2x₁x₂ + 4x₀x₁x₂
```

The 4x₀x₁x₂ term is essential and has no quadratic substitute.

### 2.3 Training confirms complete failure

Training the quadratic model (with absolute indexing) on this data:

```
T = 12 (3 blocks of 4)
Theoretical minimum loss: 0.5199

Step    0 | Loss: 0.6931 | Gap: 0.1733
Step 5000 | Loss: 0.6958 | Gap: 0.1759

--- Predictions at position 3 (XOR of 0,1,2) ---
  (0,0,0) → P(x₃=1) = 0.50  (target: 0)
  (0,0,1) → P(x₃=1) = 0.50  (target: 1)
  (0,1,0) → P(x₃=1) = 0.50  (target: 1)
  (0,1,1) → P(x₃=1) = 0.49  (target: 0)
  (1,0,0) → P(x₃=1) = 0.50  (target: 1)
  (1,0,1) → P(x₃=1) = 0.50  (target: 0)
  (1,1,0) → P(x₃=1) = 0.47  (target: 0)
  (1,1,1) → P(x₃=1) = 0.47  (target: 1)
```

The model learns **nothing** — all predictions are ~0.5, all weights stay near zero, and the loss remains at ln 2 ≈ 0.693 (pure coin-flip). The optimizer finds that any attempt to fit some cases makes others worse, so it gives up entirely.

This is a fundamental expressiveness limitation, not an optimization failure. To handle n-input XOR, the model needs n-body interaction terms.

---

## 3. Improving Expressivity

> **Script**: [07_improving_expressivity.py](07_improving_expressivity.py)

Two approaches to go beyond the quadratic barrier, both tested on the same 3-input XOR data (blocks of 4).

### 3.1 Approach 1: Cubic terms

The most direct fix — add 3-body interactions to the logit:

```
Δₜ = bₜ + Σ_{i<t} W[t,i] xᵢ + Σ_{i<j<t} C[t,i,j] xᵢxⱼ + Σ_{i<j<k<t} D[t,i,j,k] xᵢxⱼxₖ
```

The new D tensor has shape (T, T, T, T) with mask i < j < k < t. For T = 12 this adds C(12,4) = 495 active parameters. The 3-input XOR solution uses D[3,0,1,2] (and analogous entries for later blocks) to provide the missing x₀x₁x₂ term.

**Result**: converges to the theoretical minimum.

```
Step     0 | Loss: 0.6931 | Gap: 0.1733
Step  8000 | Loss: 0.5231 | Gap: 0.0032

  (0,0,0) → P(x₃=1) = 0.001  (target: 0)
  (0,0,1) → P(x₃=1) = 0.999  (target: 1)
  ...
  (1,1,1) → P(x₃=1) = 0.999  (target: 1)   ← the case that broke quadratic
```

**Downside**: the parameter count grows as O(T⁴) for cubic, O(T⁵) for quartic, etc. Each new order of XOR requires a new tensor of one higher rank.

### 3.2 Approach 2: Stacking layers

Instead of adding explicit higher-order terms, **compose** two quadratic layers with a sigmoid nonlinearity between them:

```
Layer 1:  h_t = σ(b¹_t + Σ W¹[t,i] xᵢ + Σ C¹[t,i,j] xᵢxⱼ)
Layer 2:  Δ_t = b²_t + linear(x, h) + quad(x,x) + quad(h,h) + cross(x,h)
```

The key new ingredient is the **cross-term** x_i · h_j. Here's why it works:

1. Layer 1 at position 2 can learn h₂ ≈ x₀ ⊕ x₁ (2-input XOR — which quadratic *can* do).
2. Layer 2 at position 3 sees both x₂ and h₂. The cross-term x₂ · h₂ effectively computes x₂ · (x₀ ⊕ x₁).
3. Then layer 2 just solves (x₀ ⊕ x₁) ⊕ x₂ — another 2-input XOR.

Stacking decomposes 3-input XOR into two stages of 2-input XOR. This is analogous to how stacking transformer blocks lets each layer build on features from previous layers.

**Result**: also converges to the theoretical minimum, with even sharper predictions.

```
Step     0 | Loss: 0.6931 | Gap: 0.1733
Step  8000 | Loss: 0.5217 | Gap: 0.0018

  (0,0,0) → P(x₃=1) = 0.001  (target: 0)
  (0,0,1) → P(x₃=1) = 0.999  (target: 1)
  ...
  (1,1,1) → P(x₃=1) = 0.999  (target: 1)
```

### 3.3 Trade-offs

| | Cubic model | Stacked model |
|---|---|---|
| How it works | Explicit 3-body term D[t,i,j,k] xᵢxⱼxₖ | Composes two 2-body stages via sigmoid |
| Parameter scaling | O(T⁴) for 3-body, O(T^{n+1}) for n-body | O(T³) per layer (quadratic), stacking adds layers not tensor rank |
| For n-input XOR | Needs order-n tensor | Needs ~log₂(n) layers (each layer doubles the effective order) |
| Analogy | Higher-order Boltzmann machine | Deep network / stacked transformer blocks |

The stacked approach scales better: adding a layer roughly doubles the effective interaction order, while the explicit approach requires a new tensor dimension for each order. This is essentially why deep networks are more parameter-efficient than shallow wide ones for capturing complex interactions.

---

## 4. Extrapolation: 4- and 5-Input XOR

> **Script**: [08_extrapolation.py](08_extrapolation.py)

How far do our two architectures reach?

### 4.1 Results

| n-input XOR | Theoretical min | Cubic (3-body) | Stacked 2-layer |
|---|---|---|---|
| 3 (§3) | 0.520 | **0.523** (works) | **0.522** (works) |
| 4 | 0.555 | 0.697 (coin-flip) | **0.556** (works) |
| 5 | 0.578 | 0.695 (coin-flip) | 0.627 (partial) |

### 4.2 Cubic model: fails at 4-input XOR

Just as the quadratic model failed at 3-input XOR, the cubic model fails at 4-input XOR. The pattern is strict: an order-k polynomial in the logit can represent at most k-input XOR. 4-input XOR requires the quartic term x₀x₁x₂x₃, which isn't in the model. The loss stays at ln 2 (coin-flip), and all predictions are ~0.5.

### 4.3 Stacked 2-layer: succeeds at 4-input, partially at 5-input

**4-input XOR (100% accuracy)**: The stacked model handles this by using the h · h terms in layer 2. Each h value encodes a 2-body feature via the sigmoid, and their product effectively gives order-4 interactions:

1. Layer 1 learns h₂ ≈ x₀ ⊕ x₁ and h₃ ≈ some other useful 2-body feature of {x₀, x₁, x₂}
2. Layer 2 combines these via h · h and x · h cross-terms to reconstruct 4-input XOR

The key insight: the sigmoid is a **smooth** nonlinearity, not a step function. The products h_i · h_j carry richer information than Boolean products would — the continuous intermediate values allow the second layer to extract higher-order interactions than a purely combinatorial analysis would suggest.

**5-input XOR (87.5% accuracy)**: The model gets most cases right but fails on some. With scalar hidden features (one number per position), two layers can build up to effective order ~4 interactions. 5-input XOR needs order 5. The model would need either:
- **A third layer** (stacking three quadratic layers, each doubling the effective order: 2 → 4 → 8 ≥ 5)
- **Wider hidden states** (multiple features per position per layer, giving more capacity to encode complex interactions)

### 4.4 Effective order doubles with each layer

Why does the stacked model succeed at 4-input but not 5-input XOR? Each layer computes degree-2 interactions among its inputs, then applies sigmoid. The highest-order feature available to layer L is h^(L−1), which has effective order 2^(L−1). The h · h product in layer L gives:

```
order(h^(L−1) · h^(L−1)) = 2^(L−1) + 2^(L−1) = 2^L
```

So the maximum effective interaction order **doubles** with each layer:

| Layers (N) | Max effective order | Handles n-input XOR up to |
|---|---|---|
| 1 | 2¹ = 2 | n = 2 |
| 2 | 2² = 4 | n = 4 |
| 3 | 2³ = 8 | n = 8 |
| N | 2^N | n = 2^N |

This matches the experiments: 2 layers handles 4-input (2² = 4 ≥ 4) but not 5-input (2² = 4 < 5; would need 3 layers since 2³ = 8 ≥ 5).

The argument is that n-input XOR decomposes as a balanced binary tree of 2-input XORs. Each level of the tree maps to one layer, and a tree of depth N has 2^N leaves — so N = ⌈log₂(n)⌉ layers suffice.

**Caveat**: 2^N is an upper bound on what the architecture *can* represent. Whether it's achieved depends on the hidden states having enough **capacity** to faithfully encode the intermediate results. With scalar features (one number per position), this works for n-input XOR because each intermediate value is a single bit (the partial XOR). For other functions with richer intermediate structure, you'd need **wider** hidden states (multiple features per position per layer) — analogous to how transformers use high-dimensional residual streams rather than scalar hidden states.

### 4.5 The full scaling picture

| Architecture | Max XOR it can solve | Scaling |
|---|---|---|
| Linear (order 1) | — (can't do XOR at all) | — |
| Quadratic (order 2) | 2-input | Needs order-n tensor for n-input |
| Cubic (order 3) | 3-input | O(T⁴) parameters |
| N stacked quadratic layers | 2^N-input | O(N · T³) parameters |

Stacking is exponentially more efficient: 3 layers with O(T³) parameters each gives effective order 8, while the explicit single-layer approach would need an order-8 tensor with O(T⁹) parameters. This is the core argument for depth over width in neural network design.

---

## 5. Connection to Transformers

The models in this tutorial are toy versions of mechanisms that appear in transformers. Here's how the pieces map.

### 5.1 Attention as a 3-site coupling

A single attention head computes, for each position t:

```
Attention(t) = Σ_{i<t} softmax(Q_t · K_i / √d) · V_i
```

The score Q_t · K_i is a **dot product between two positions' representations** — this is a bilinear (2-body) interaction, analogous to our W[t,i] · x_i. But the output multiplies this score by V_i, making the full attention term a **3-site coupling**:

```
(representation at t) × (representation at i) × (representation at i again, via V)
```

More precisely: the query Q_t comes from position t, the key K_i comes from position i, and the value V_i also comes from position i. The softmax over i makes it a weighted sum, but each term involves the interaction of t's representation with i's representation — a pairwise interaction that produces a new feature at position t.

This is directly analogous to our linear term Σ W[t,i] · x_i — attention is a **data-dependent** version of it, where the weights W[t,i] are themselves computed from the data (via Q · K) rather than being fixed parameters.

### 5.2 Where are the quadratic terms?

In our model, the quadratic term C[t,i,j] · x_i · x_j couples **two** past positions when predicting position t — a 3-site interaction.

In a transformer, attention couples position t with one past position at a time (a 2-site interaction per term). But **multi-head attention** runs several heads in parallel, each producing its own weighted sum. After concatenation and projection, the combined output at position t is a function of multiple (head-selected) past positions. When this feeds into the next layer, the effective interaction order increases.

So transformers don't have an explicit C[t,i,j] · x_i · x_j term in a single layer. Instead, they build up multi-position interactions **across layers** — which is exactly the stacking mechanism from §3.2.

### 5.3 The MLP: per-position nonlinearity

After attention, each transformer layer applies a **feedforward MLP** independently at each position:

```
MLP(z) = W₂ · ReLU(W₁ · z + b₁) + b₂
```

This is a per-position nonlinear transformation — it doesn't mix information across positions (that's attention's job). Its role is to **compute nonlinear functions of the features** that attention has gathered. To see why this matters, consider what happens without it.

#### What attention does to values is linear

Despite the softmax nonlinearity in computing attention weights, the operation on the **values** is a weighted sum — linear in V:

```
output_t = Σ_i  α_ti · V_i       where α_ti = softmax(Q_t · K_i / √d)
```

The weights α_ti are nonlinear functions of Q and K, but once computed, the combination of values is just a linear mix. This means stacking attention-only layers (no MLP) produces increasingly sophisticated *attention patterns*, but the transformation of the actual feature vectors remains linear. Multiple linear transformations compose to a single linear transformation — so depth without nonlinearity buys you nothing.

#### The MLP breaks linearity

The MLP applies ReLU (or GELU, etc.) to create genuinely nonlinear features. Each hidden neuron in the MLP computes:

```
ReLU(w · z + b)
```

This is a **feature detector**: it checks whether a particular linear combination of the gathered information exceeds a threshold, outputting zero below and passing the value through above. The hidden layer of the MLP computes many such detectors in parallel, and W₂ linearly combines them into the output.

The result: after the MLP, the residual stream at position t contains nonlinear functions of whatever attention gathered from past positions. When the *next* layer's attention forms weighted sums of these nonlinear features, the products and combinations can express higher-order interactions that no single attention layer could.

#### Direct analogy to our model

This is exactly what happens in our stacked quadratic model. Without the sigmoid between layers:

```
h_t = quadratic(x_{<t})     ← no sigmoid
logit_t = quadratic(x, h)   ← layer 2
```

Layer 2's logit is quadratic in h, which is itself quadratic in x, giving a degree-4 polynomial in x. But this is *exactly equivalent* to a single layer with a quartic tensor — no efficiency gain from stacking. The composition of polynomials is just a higher-degree polynomial.

With the sigmoid:

```
h_t = σ(quadratic(x_{<t}))  ← sigmoid applied
logit_t = quadratic(x, h)
```

Now h_t is a nonlinear function that can't be written as any fixed-degree polynomial. Products like h_i · h_j in layer 2 create functions with effectively unbounded polynomial degree — this is what enables the exponential scaling 2^N from §4.4. The sigmoid is doing the same job as the MLP's ReLU: it creates a class of intermediate features that, when combined by the next layer, are fundamentally more expressive than any polynomial of matching parameter count.

#### Two jobs, two components

A transformer layer thus has a clean division of labor:

| Component | What it does | Mixes positions? | Nonlinear? |
|---|---|---|---|
| Attention | Gathers information from past positions into each position's representation | Yes | Only in the weights (softmax), linear in values |
| MLP | Transforms the gathered information into nonlinear features | No (per-position) | Yes (ReLU/GELU) |

Both are necessary. Attention without MLP can only linearly remix features. MLP without attention can only transform each position in isolation (no inter-position interaction). Together, alternating across layers, they build up the exponentially growing interaction order that makes deep transformers powerful.

### 5.4 Summary of analogies

| Our model | Transformer |
|---|---|
| W[t,i] · x_i (linear term) | Attention: data-dependent weighted sum of past positions |
| C[t,i,j] · x_i · x_j (quadratic term) | No single-layer analogue; emerges from stacking |
| Sigmoid between layers | MLP nonlinearity (ReLU) between attention layers |
| Stacking layers to increase interaction order | Stacking transformer blocks to capture longer-range dependencies |
| Fixed W[t,i] parameters | Attention weights computed dynamically from Q · K |
| Explicit higher-order tensors (D, etc.) | Not used; transformers rely entirely on depth for higher-order interactions |

The fundamental architectural choice is the same: **depth (stacking) vs. width (higher-order explicit terms)**. Transformers chose depth — they use only pairwise interactions (attention) per layer, but stack many layers with nonlinearities between them. This is why a 1-layer transformer can't do everything a 12-layer transformer can, even with the same hidden dimension.

---

## 6. Embeddings: From Scalars to One-Hot to Learned

> **Script**: [09_one_hot_embedding.py](09_one_hot_embedding.py)

So far every token has been a single number: x_t ∈ {0, 1}. This works for a binary vocabulary, but breaks for larger vocab sizes. Understanding why — and how to fix it — leads directly to the embedding layers used in real transformers.

### 6.1 Why scalar encoding breaks

Suppose we had a vocab of size q = 4, with tokens {0, 1, 2, 3}. Using the raw numbers as the representation means the quadratic interaction C[t,i,j] · x_i · x_j computes products like 2 · 3 = 6. But there's no reason the interaction between tokens 2 and 3 should be "6 times" the interaction between tokens 1 and 1 (which gives 1 · 1 = 1).

The scalar encoding imposes two false assumptions:

1. **Ordering**: token 2 is "between" tokens 1 and 3. For categorical data (words, subwords), this is meaningless.
2. **One parameter per position pair**: C[t,i,j] is a single scalar. The model has one number to describe the interaction between positions i and j, regardless of *which tokens* appear there. But the effect of seeing "cat" at i and "dog" at j should be completely independent from seeing "the" at i and "runs" at j.

For binary vocab (q = 2), this issue is hidden: x ∈ {0, 1} has only two states, and a single weight effectively captures both cases (x = 0 contributes nothing, x = 1 contributes the weight).

### 6.2 One-hot encoding

Represent each token as a one-hot vector: token k → e_k (the k-th standard basis vector in ℝ^q). Now:

**Linear term**: W[t, i] becomes a q × q matrix (output class × input token):

```
Σ_c W[t, i, a, c] · x_i[c]
```

Since x_i is one-hot with value k, this picks out column k of W — a separate q-dimensional weight for each possible input token. Each output class gets its own response to each input token.

**Quadratic term**: C[t, i, j] becomes a q × q × q tensor (output class × input token at i × input token at j):

```
Σ_{c,d} C[t, i, j, a, c, d] · x_i[c] · x_j[d]
```

This picks out C[t, i, j, a, k_i, k_j] — a completely independent parameter for each combination of (output class, token at i, token at j). This is the right level of expressiveness: every pair of token values interacts independently.

### 6.3 A subtlety: softmax is shift-invariant

The output layer uses **softmax** (generalizing sigmoid to q classes):

```
P(x_t = a | x_{<t}) = softmax(Δ_t)[a] = exp(Δ_t[a]) / Σ_b exp(Δ_t[b])
```

A critical property: adding the same constant to all logits doesn't change the distribution. So the quadratic term *must produce different values for different output classes*. If C gives a single scalar shared across output classes (as in a naive first attempt), it has **zero effect** under softmax — the information cancels entirely.

This is why C needs the output dimension a: C[t, i, j, **a**, c, d]. Each output class must get its own contribution from the pairwise context.

### 6.4 Verification on Z₂ data

Training the one-hot model (q = 2) on the original 2-input XOR data:

```
T = 12, vocab size q = 2
Theoretical minimum loss: 0.4621

Step    0 | Loss: 0.6931 | Gap: 0.2310
Step 1000 | Loss: 0.4668 | Gap: 0.0047
Step 5000 | Loss: 0.4706 | Gap: 0.0085

--- XOR predictions (position 2) ---
  (0,0) → P(x₂=0)=1.00, P(x₂=1)=0.00  (target: 0)
  (0,1) → P(x₂=0)=0.00, P(x₂=1)=1.00  (target: 1)
  (1,0) → P(x₂=0)=0.00, P(x₂=1)=1.00  (target: 1)
  (1,1) → P(x₂=0)=1.00, P(x₂=1)=0.00  (target: 0)
```

Converges to the theoretical minimum with near-perfect predictions — confirming that the one-hot model is equivalent to the scalar model for q = 2.

### 6.5 Why one-hot is wasteful at scale

One-hot vectors are q-dimensional and maximally sparse. For a real vocabulary (q = 50,000 for a small LLM tokenizer):

- **Parameter explosion**: W[t, i] is q × q = 2.5 billion entries per position pair. The quadratic C with its q³ output-input-input structure is even worse.
- **No similarity structure**: every token is orthogonal to every other. "cat" and "kitten" are as different as "cat" and "quantum" — the model must learn their similarity from scratch at every position.
- **Sparse gradients**: on each training step, only the weights for the tokens that actually appeared get updated. Most parameters are idle.

### 6.6 The fix: learned embeddings

Map each token to a **dense** vector in ℝ^d where d ≪ q:

```
embed(token_k) = E[k, :]    where E is a (q, d) learnable matrix
```

This is mathematically one-hot followed by a linear projection: embed(k) = E^T · e_k = row k of E. The embedding matrix compresses the sparse q-dimensional representation into a dense d-dimensional one.

Why this helps:

1. **Parameter efficiency**: all weight matrices operate on d-dimensional vectors instead of q-dimensional. The quadratic term needs d² parameters per position pair instead of q². For d = 768 vs q = 50,000, that's a ~4000× reduction.
2. **Similarity structure**: tokens with similar roles end up with nearby embedding vectors. "cat" and "kitten" automatically produce similar effects in all interactions, because their embeddings are close in ℝ^d.
3. **Generalization**: anything the model learns about "cat" partially transfers to "kitten" for free (nearby embeddings → similar logits).

The embedding dimension d controls the expressiveness/efficiency tradeoff. For q = 2, d = 1 suffices (which is what our scalar model was doing all along). For q = 50,000, typical d is 768–1024.

---

## Outline

*(Further sections to be developed)*

---
