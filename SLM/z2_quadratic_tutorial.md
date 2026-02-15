# Learning Z₂ Addition with a Quadratic Autoregressive Model

A PyTorch tutorial on training a simple probabilistic model to learn XOR structure in binary sequences.

---

## 1. Introduction & the Data

### Z₂ addition

The group Z₂ = {0, 1} under addition mod 2 gives us XOR:

| a | b | a ⊕ b |
|---|---|-------|
| 0 | 0 |   0   |
| 0 | 1 |   1   |
| 1 | 0 |   1   |
| 1 | 1 |   0   |

### Sequence structure

Our sequences have length T (a multiple of 3). Positions come in **blocks of 3**: two random bits followed by their XOR.

```
positions:  0  1  2 | 3  4  5 | 6  7  8 | ...
role:       a  b  c | a  b  c | a  b  c | ...
constraint: c=a⊕b   | c=a⊕b   | c=a⊕b   | ...
```

Example with T = 12:

```
0, 0, 0 | 1, 1, 0 | 0, 1, 1 | 1, 0, 1
```

### True distribution

Each block is independent. Within a block, the two "random" positions are uniform in {0, 1}, and the third is deterministic. So there are exactly **4 valid triples** per block:

```
{(0,0,0), (0,1,1), (1,0,1), (1,1,0)}
```

each with probability 1/4. The full sequence has probability (1/4)^{T/3}.

### Entropy

Per position, the entropy is:

- Positions 0, 1 (mod 3): each contributes ln 2 nats (= 1 bit) (fully random)
- Position 2 (mod 3): contributes 0 nats (deterministic given the first two)

Average per-position entropy: **(2/3) · ln 2 ≈ 0.462 nats**.

This is the theoretical minimum loss any correct model can achieve.

---

## 2. Why a Linear Model Fails (XOR)

> **Script**: [01_linear_fails.py](01_linear_fails.py)

Suppose we try to predict x₂ from x₀, x₁ using a single logistic unit:

```
P(x₂ = 1 | x₀, x₁) = σ(a + b₁x₀ + b₂x₁)
```

where σ(z) = 1/(1 + e^{−z}) is the sigmoid function. We need this to match XOR:

| x₀ | x₁ | target P(x₂=1) |
|----|-----|-----------------|
| 0  |  0  |       0         |
| 0  |  1  |       1         |
| 1  |  0  |       1         |
| 1  |  1  |       0         |

From rows 1 and 2: increasing x₁ (with x₀=0) must **increase** the output, so b₂ > 0.

From rows 3 and 4: increasing x₁ (with x₀=1) must **decrease** the output, so b₂ < 0.

**Contradiction.** No choice of (a, b₁, b₂) works. XOR is not linearly separable — this is the classic result that motivates going beyond linear features.

---

## 3. The Subtlety: Energy-Based vs. Autoregressive

A natural idea: use **pairwise (quadratic) interactions** to capture XOR. But *where* you put the quadratic terms matters enormously.

### 3.1 Energy-based (Boltzmann) interpretation — doesn't work

Define a joint distribution via an energy function with pairwise terms:

```
log P(x) = a + Σᵢ bᵢxᵢ + Σᵢ<ⱼ cᵢⱼ xᵢxⱼ  −  log Z
```

This looks like it has quadratic interactions. But watch what happens when we compute a conditional. Since xₜ ∈ {0, 1}, the energy splits into terms that involve xₜ and terms that don't:

```
E(x) = a + Σᵢ bᵢxᵢ + Σᵢ<ⱼ cᵢⱼ xᵢxⱼ

     = E₀(x_{-t})  +  xₜ · (bₜ + Σ_{i≠t} cᵢₜ xᵢ)
       ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       doesn't              call this Δ
       involve xₜ
```

where E₀ collects every term not involving xₜ. So P(x) = exp(E₀ + xₜ·Δ) / Z. Now compute the conditional:

```
                     exp(E₀ + Δ) / Z
P(xₜ=1 | x_{-t}) = ─────────────────────────────
                     exp(E₀) / Z  +  exp(E₀ + Δ) / Z
```

Both Z and exp(E₀) cancel:

```
                   = exp(Δ) / (1 + exp(Δ))  =  σ(Δ)
```

So the sigmoid emerges naturally from the binary variable + exponential family setup. And crucially, Δ = bₜ + Σ_{i≠t} cᵢₜ xᵢ is **linear** in the conditioning variables — the quadratic terms cᵢⱼ xᵢxⱼ where neither index is t ended up in E₀ and cancelled out. The conditional is:

```
P(xₜ = 1 | rest) = σ(bₜ + Σ_{i≠t} cᵢₜ xᵢ)
```

This is exactly the form we just proved can't do XOR!

**Key insight:** A pairwise Boltzmann machine has quadratic interactions in the *joint*, but only *linear* conditionals. To capture XOR in the conditionals, you'd need a **3-body** term x₀x₁x₂ in the energy — making it a higher-order Boltzmann machine.

### 3.2 Autoregressive interpretation — works

Instead of an energy-based joint, build the distribution autoregressively via the chain rule:

```
P(x₀, ..., x_{T-1}) = Π_t P(xₜ | x_{<t})
```

and put the quadratic features **directly inside each conditional**:

```
P(xₜ = 1 | x_{<t}) = σ(bₜ + Σ_{i<t} Wₜᵢ xᵢ + Σ_{i<j<t} Cₜᵢⱼ xᵢ xⱼ)
```

Now the product xᵢxⱼ **appears explicitly** in the argument of the sigmoid — it doesn't cancel. This is a fundamentally different model family from the Boltzmann machine.

### 3.3 Concrete solution for position 2

We need P(x₂=1 | x₀, x₁) = x₀ ⊕ x₁. Using the quadratic conditional, the logit (the input to σ — equivalently, the log-odds log(p/(1−p))) is:

```
logit = b₂ + W₂₀·x₀ + W₂₁·x₁ + C₂₀₁·x₀·x₁
```

Set b₂ = −L, W₂₀ = W₂₁ = 2L, C₂₀₁ = −4L for large L > 0. Check:

| x₀ | x₁ | logit              | σ(logit)    | target |
|----|-----|--------------------|-------------|--------|
| 0  |  0  | −L                 | ≈ 0         |   0    |
| 0  |  1  | −L + 2L = L        | ≈ 1         |   1    |
| 1  |  0  | −L + 2L = L        | ≈ 1         |   1    |
| 1  |  1  | −L + 2L + 2L − 4L = −L | ≈ 0    |   0    |

As L → ∞, this converges to exact XOR. The quadratic term C₂₀₁ is essential — it's what "bends" the decision boundary.

---

## 4. The Full Model — Math

### The conditional

Each conditional is parameterized as:

```
P(xₜ = 1 | x_{<t}) = σ(Δₜ)

where  Δₜ = bₜ + Σ_{i<t} Wₜᵢ xᵢ + Σ_{i<j<t} Cₜᵢⱼ xᵢ xⱼ
```

Since xₜ is binary, the complementary probability is P(xₜ = 0 | x_{<t}) = 1 − σ(Δₜ) = σ(−Δₜ). We can write both cases compactly:

```
P(xₜ | x_{<t}) = σ((2xₜ − 1) · Δₜ)
```

since (2xₜ − 1) flips the sign of Δₜ when xₜ = 0.

### Chain rule decomposition

The joint distribution is the product of all conditionals:

```
P(x₀, ..., x_{T-1}) = Π_{t=0}^{T-1} P(xₜ | x_{<t})

                     = Π_{t=0}^{T-1} σ((2xₜ − 1) · Δₜ)
```

Taking the log:

```
log P(x₀, ..., x_{T-1}) = Σ_{t=0}^{T-1} log σ((2xₜ − 1) · Δₜ)
```

There is **no partition function** — each factor is already normalized (σ(Δ) + σ(−Δ) = 1), so the product is automatically a valid distribution over {0,1}^T. This is the core advantage of the autoregressive factorization over the energy-based approach.

### Computational cost

The joint involves all T conditionals, each referencing entries of C — but this is less redundant than it looks. Each entry C[t,i,j] appears in exactly **one** conditional (the one for position t, since the mask enforces i < j < t). Computing the full log P(x) touches each entry of C once across all terms — total work O(T³), the same as a single pass through C.

We only ever need log P(x) in two places:

- **Training**: compute log P(x) for a batch, take gradient, update. This is what `binary_cross_entropy_with_logits` does — it computes −log σ((2xₜ−1)·Δₜ) per position and sums.
- **Sampling**: we don't need the full joint at all. Go left to right — compute Δ₀, sample x₀ ∼ Bernoulli(σ(Δ₀)), compute Δ₁ (which depends on x₀), sample x₁, and so on. One conditional at a time, O(t²) per position.

We never need to enumerate P over all 2^T configurations (except in the exact training trick in §7, which only works for small T anyway).

### Parameters

| Parameter | Shape    | Constraint                  | Count          |
|-----------|----------|-----------------------------|----------------|
| b         | (T,)     | none                        | T              |
| W         | (T, T)   | lower-triangular (i < t)    | T(T−1)/2       |
| C         | (T, T, T)| i < j < t                  | O(T³/6)        |

For T = 12 (4 blocks), C has on the order of ~300 free parameters. This is fine for a tutorial.

### Loss

Negative log-likelihood per position, averaged over the batch:

```
L = −(1/T) Σ_t E_data[ xₜ log σ(Δₜ) + (1−xₜ) log(1 − σ(Δₜ)) ]
```

This is just the mean of `F.binary_cross_entropy_with_logits` across positions and batch.

**Theoretical minimum**: (2/3) ln 2 ≈ 0.462 nats per position.

---

## 5. PyTorch Implementation — Full Model

> **Script**: [02_train_full.py](02_train_full.py)

### 5.1 Data generation

```python
import torch

def sample_z2_data(batch_size, T=12):
    """Sample sequences where every 3rd element is XOR of the previous two."""
    assert T % 3 == 0
    x = torch.zeros(batch_size, T)
    for block in range(T // 3):
        i = block * 3
        x[:, i]     = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 1] = torch.randint(0, 2, (batch_size,)).float()
        x[:, i + 2] = (x[:, i] + x[:, i + 1]) % 2  # XOR
    return x
```

A few sample sequences:

```python
>>> sample_z2_data(4, T=9)
tensor([[1., 0., 1., 0., 0., 0., 1., 1., 0.],
        [0., 1., 1., 1., 0., 1., 0., 0., 0.],
        [1., 1., 0., 0., 1., 1., 1., 0., 1.],
        [0., 0., 0., 1., 1., 0., 0., 1., 1.]])
```

Note: we generate **fresh data each step** — this is an infinite-data regime, so there's no overfitting concern.

**Block alignment is fixed.** Every training sequence has its XOR triples at the same positions (0,1,2), (3,4,5), (6,7,8), .... The model doesn't need to discover *where* the triples are. However, the autoregressive conditionals do cross block boundaries — when computing P(x₅ | x_{<5}), the model sees positions 0,1,2,3,4, not just the within-block context 3,4. It must learn on its own that the cross-block weights (e.g. W[5,0], C[5,0,1]) should be zero.

### 5.2 Model class

```python
import torch.nn as nn

class QuadraticAutoregressive(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.bias = nn.Parameter(torch.zeros(T))
        self.W = nn.Parameter(torch.zeros(T, T))       # linear weights
        self.C = nn.Parameter(torch.zeros(T, T, T))     # quadratic weights

        # Precompute masks (not parameters, just buffers)
        # Linear mask: W[t, i] is active only if i < t
        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        # Quadratic mask: C[t, i, j] is active only if i < j < t
        idx = torch.arange(T)
        # C[t, i, j]: need i < j < t
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &   # i < j
            (idx[None, None, :] < idx[:, None, None])      # j < t
        ).float()
        self.register_buffer('quad_mask', quad_mask)

    def forward(self, x):
        """
        x: (B, T) binary tensor
        returns: logits (B, T) — one per position
        """
        B, T = x.shape

        # Linear contribution: for each t, sum_{i<t} W[t,i] * x[i]
        W_masked = self.W * self.linear_mask          # (T, T)
        linear = x @ W_masked.T                       # (B, T)

        # Quadratic contribution: for each t, sum_{i<j<t} C[t,i,j] * x[i] * x[j]
        # Outer product of x with itself: (B, T, T)
        xx = x.unsqueeze(-1) * x.unsqueeze(-2)        # (B, T, T)

        # Contract with masked C: for each t, sum over i,j
        C_masked = self.C * self.quad_mask             # (T, T, T)
        quadratic = torch.einsum('bij, tij -> bt', xx, C_masked)  # (B, T)

        logits = self.bias + linear + quadratic        # (B, T)
        return logits
```

**What's happening in the forward pass:**

1. **Linear term**: We mask W to be strictly lower-triangular (ensuring each position only sees past positions), then compute Wx via a matrix multiply.

2. **Quadratic term**: We form the outer product x ⊗ x of shape (B, T, T) — entry (b, i, j) is xᵢxⱼ for sample b. Then we contract this against C[t, i, j] (masked so i < j < t) using `einsum`, producing one scalar per (batch, position).

3. Add bias and return logits (pre-sigmoid values).

### 5.3 Training loop

```python
import torch.nn.functional as F

T = 12
model = QuadraticAutoregressive(T)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

theoretical_min = (2/3) * torch.log(torch.tensor(2.0)).item()
print(f"Theoretical minimum loss: {theoretical_min:.4f}")

for step in range(5001):
    x = sample_z2_data(batch_size=512, T=T)
    logits = model(x)

    loss = F.binary_cross_entropy_with_logits(logits, x, reduction='mean')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
              f"Gap: {loss.item() - theoretical_min:.4f}")
```

Expected output (approximate):

```
Theoretical minimum loss: 0.4621
Step    0 | Loss: 0.6931 | Gap: 0.2310
Step 1000 | Loss: 0.4739 | Gap: 0.0118
Step 2000 | Loss: 0.4660 | Gap: 0.0039
Step 3000 | Loss: 0.4635 | Gap: 0.0014
Step 4000 | Loss: 0.4649 | Gap: 0.0028
Step 5000 | Loss: 0.4643 | Gap: 0.0022
```

The loss converges close to the theoretical minimum. Note that SGD loss is noisy (each step uses a fresh mini-batch), so the reported loss fluctuates around the true expected value. The model has enough capacity to perfectly learn the data distribution.

### 5.4 Inspecting learned parameters

After training, we can verify the model has learned the right structure:

```python
with torch.no_grad():
    # Biases at "random" positions should be near 0 (predicting 50/50)
    for t in range(T):
        if t % 3 != 2:  # positions 0,1, 3,4, 6,7, ...
            print(f"bias[{t}] = {model.bias[t].item():.3f}")  # expect ~0

    # For XOR positions (2, 5, 8, 11), check the key weights
    for block in range(T // 3):
        t = block * 3 + 2  # XOR position
        i, j = block * 3, block * 3 + 1  # the two source positions
        print(f"\nPosition {t} (XOR of {i},{j}):")
        print(f"  bias[{t}]    = {model.bias[t].item():.2f}")    # expect large negative
        print(f"  W[{t},{i}]   = {model.W[t,i].item():.2f}")     # expect large positive
        print(f"  W[{t},{j}]   = {model.W[t,j].item():.2f}")     # expect large positive
        print(f"  C[{t},{i},{j}]= {model.C[t,i,j].item():.2f}")  # expect large negative
```

You should see something like:

```
Position 2 (XOR of 0,1):
  bias[2]    = -6.56
  W[2,0]     = 13.25
  W[2,1]     = 13.25
  C[2,0,1]   = -26.68

Position 5 (XOR of 3,4):
  bias[5]    = -6.21
  W[5,3]     = 12.89
  W[5,4]     = 12.89
  C[5,3,4]   = -26.65

Position 8 (XOR of 6,7):
  bias[8]    = -5.73
  W[8,6]     = 12.43
  W[8,7]     = 12.50
  C[8,6,7]   = -26.61
...
```

The pattern matches our theoretical solution: b ≈ −L, W ≈ 2L, C ≈ −4L, with L ≈ 6.5. The model has independently discovered the XOR solution for each block. (The exact values of L vary by run and don't converge to a fixed point — they grow slowly as the sigmoid sharpens toward a step function.)

For a quick visual, you could also plot these with `plt.bar` (e.g., bar-chart the `|C[t,i,j]|` magnitudes at each XOR position).

---

## 6. Sparse / Banded Versions

### 6.1 The full model's parameter scaling

Before introducing banding, let's be explicit about how the full model scales. For an order-n interaction term, the source indices must satisfy i₁ < i₂ < ... < iₙ < t (all strictly before the target position). The number of such tuples drawn from {0, ..., t−1} is C(t, n), so summing over all positions t:

| Parameter | Indices | Constraint | Total count | Growth |
|-----------|---------|-----------|-------------|--------|
| bₜ | (t) | — | T | O(T) |
| Wₜᵢ | (t, i) | i < t | T(T−1)/2 | O(T²) |
| Cₜᵢⱼ | (t, i, j) | i < j < t | T(T−1)(T−2)/6 | O(T³) |
| Dₜᵢⱼₖ | (t, i, j, k) | i < j < k < t | T(T−1)(T−2)(T−3)/24 | O(T⁴) |
| order-n | (t, i₁...iₙ) | i₁ < ... < iₙ < t | C(T, n+1) | O(T^(n+1)) |

For T = 100, C alone has ~160k parameters and D has ~4 million. This motivates restricting which entries are allowed.

### 6.2 Fully banded model — all indices close

The simplest restriction: position t can only interact with positions within a window of size k. That is, for C[t, i, j], we require:

```
t − i ≤ k   AND   t − j ≤ k
```

Since i < j < t, the constraint t − i ≤ k is the binding one (i is furthest from t). This forces **all** indices into a window of size k, so i, j, and t are all within k of each other.

For a general order-n term, each source index must satisfy t − iₘ ≤ k. The number of valid source tuples per position is at most C(k, n) (choosing n indices from a window of k), giving:

| Parameter | Valid sources per t | Total | Growth |
|-----------|---------------------|-------|--------|
| bₜ | 1 | T | O(T) |
| Wₜᵢ | k | kT | O(kT) |
| Cₜᵢⱼ | C(k,2) = k(k−1)/2 | k²T/2 | O(k²T) |
| Dₜᵢⱼₖ | C(k,3) | k³T/6 | O(k³T) |
| order-n | C(k,n) | C(k,n)·T | O(kⁿT) |

The key feature: **linear in T** (times a constant depending on k and the order). For the Z₂ data with k = 3, C has just ~300 parameters regardless of sequence length.

This is the right model when interactions are purely local — which they are for the Z₂ data, where each XOR depends only on the 2 immediately preceding positions.

### 6.3 Implementation — band mask on the dense tensor

> **Script**: [03_train_banded.py](03_train_banded.py)

In code, we keep C as a dense (T, T, T) tensor but multiply by a mask that zeroes out entries outside the band.

```python
class BandedQuadraticAutoregressive(nn.Module):
    def __init__(self, T, bandwidth=3):
        super().__init__()
        self.T = T
        self.bandwidth = bandwidth
        self.bias = nn.Parameter(torch.zeros(T))
        self.W = nn.Parameter(torch.zeros(T, T))
        self.C = nn.Parameter(torch.zeros(T, T, T))

        # Linear mask: i < t
        linear_mask = torch.tril(torch.ones(T, T), diagonal=-1)
        self.register_buffer('linear_mask', linear_mask)

        # Quadratic mask: i < j < t (same as before)
        idx = torch.arange(T)
        quad_mask = (
            (idx[None, :, None] < idx[None, None, :]) &
            (idx[None, None, :] < idx[:, None, None])
        ).float()

        # Band mask: only keep entries where t-i <= bandwidth and t-j <= bandwidth
        band = (
            ((idx[:, None, None] - idx[None, :, None]) <= bandwidth) &
            ((idx[:, None, None] - idx[None, None, :]) <= bandwidth)
        ).float()

        self.register_buffer('mask', quad_mask * band)
        self.register_buffer('linear_band',
            linear_mask * ((idx[:, None] - idx[None, :]) <= bandwidth).float()
        )

    def forward(self, x):
        B, T = x.shape
        W_masked = self.W * self.linear_band
        linear = x @ W_masked.T

        xx = x.unsqueeze(-1) * x.unsqueeze(-2)
        C_masked = self.C * self.mask
        quadratic = torch.einsum('bij, tij -> bt', xx, C_masked)

        return self.bias + linear + quadratic
```

With `bandwidth=3`, position t can only look at positions t−1, t−2, t−3. Since XOR at position t only needs positions t−1 and t−2, this is sufficient.

**Result**: Training the banded model with bandwidth=3 achieves the **same loss** (≈ 0.462) as the full model — confirming that the data has purely local structure.

### 6.4 Sidebar: `torch.sparse` for large T

For very large T, even storing a (T, T, T) tensor of zeros is wasteful. PyTorch's sparse COO tensors let you materialize only the nonzero entries of C. For a banded model with bandwidth k, that's O(k² · T) entries instead of O(T³) — avoiding the cubic memory cost entirely.

Here's how you'd set up the sparse index structure:

```python
def make_sparse_C_parameter(T, bandwidth=3):
    """Create a sparse C tensor with only band-local entries."""
    indices = []
    for t in range(T):
        for i in range(max(0, t - bandwidth), t):
            for j in range(i + 1, t):
                indices.append([t, i, j])

    if len(indices) == 0:
        # No quadratic interactions possible (T too small)
        return torch.sparse_coo_tensor(
            torch.empty(3, 0, dtype=torch.long),
            torch.empty(0),
            size=(T, T, T)
        )

    indices = torch.tensor(indices).T  # (3, nnz)
    values = torch.zeros(indices.shape[1])  # learnable values
    return torch.sparse_coo_tensor(indices, values, size=(T, T, T))
```

At tutorial scale (T ≤ 18), the dense-with-mask approach from §6.3 is simpler and fast enough. The sparse approach pays off when T is large and the bandwidth is small relative to T.

**Trade-offs:**

| Approach          | Memory       | Code complexity | Speed (small T) | Speed (large T) |
|-------------------|-------------|-----------------|------------------|------------------|
| Dense + mask      | O(T³)       | Simple          | Fast             | Slow/OOM         |
| Sparse            | O(k²T)      | More complex    | Overhead         | Efficient        |

### 6.5 Long-range with local pairs

The fully banded model assumes all interactions are local: if position t depends on xᵢxⱼ, then both i and j must be near t. But some data might have **long-range** dependencies where distant positions interact — just not positions that are far from *each other*.

For example, in a longer sequence, position 100 might need to know about the product x₄₉·x₅₀ (two adjacent positions, but far from t = 100). The fully banded model with k = 3 would zero this out since 100 − 49 > 3.

The alternative: constrain the **source indices to be near each other**, but let them be arbitrarily far from t. For C[t, i, j]:

```
|i − j| ≤ k     (sources are local to each other)
i < j < t        (causality, as before)
no constraint on t − i or t − j
```

For a general order-n term with source indices i₁ < ... < iₙ, the constraint is that all pairwise distances are ≤ k, i.e. iₙ − i₁ ≤ k (the sources fit in a window of size k). Per position t, the number of valid source tuples is roughly C(k, n) choices of indices within the window, times ~t choices for where to place the window. This gives:

| Parameter | Valid sources per t | Total | Growth |
|-----------|---------------------|-------|--------|
| bₜ | 1 | T | O(T) |
| Wₜᵢ | t (unconstrained) | T²/2 | O(T²) |
| Cₜᵢⱼ | ~kt | kT²/2 | O(kT²) |
| Dₜᵢⱼₖ | ~k²t | k²T²/2 | O(k²T²) |
| order-n | ~k^(n−1) · t | O(k^(n−1) T²) | O(k^(n−1) T²) |

Note that W is unconstrained — O(T²) — because a single source index has no pair to be "local" with. The banding only kicks in at order 2 and above.

### 6.6 Comparison of scaling

| Scheme | W | C | D | order-n |
|--------|---|---|---|---------|
| Full | O(T²) | O(T³) | O(T⁴) | O(T^(n+1)) |
| Fully banded | O(kT) | O(k²T) | O(k³T) | O(kⁿT) |
| Long-range, local pairs | O(T²) | O(kT²) | O(k²T²) | O(k^(n−1) T²) |

The tradeoff is clear:

- **Fully banded** is linear in T — the cheapest option, but blind to long-range interactions.
- **Long-range, local pairs** is quadratic in T but with one fewer power of k. It lets any position attend to distant correlated clusters, as long as the clusters are internally compact.
- **Full** is the most expressive but the most expensive, with O(T^(n+1)) growth.

For the Z₂ data in this tutorial, fully banded with k = 3 is sufficient. But for sequences with longer-range structure — where distant groups of nearby positions interact — the long-range local-pairs scheme offers a middle ground between expressiveness and parameter efficiency.

---

## 7. Exact Training (No Corpus)

> **Script**: [04_exact_training.py](04_exact_training.py)

### 7.1 Motivation

So far we've trained on sampled data. But the Z₂ distribution is simple enough that we can **enumerate** all possible sequences and compute the loss exactly. This eliminates sampling noise entirely.

For T positions, there are 2^T possible binary sequences. This is feasible for T ≤ ~18 (2^18 = 262144 sequences).

### 7.2 True distribution

Each sequence has a simple probability under the true distribution:

```python
def true_log_probs(T):
    """Compute log P_true(x) for all 2^T binary sequences."""
    N = 2 ** T
    # Generate all binary vectors of length T
    all_x = torch.zeros(N, T)
    for i in range(T):
        # Bit i of the sequence index
        all_x[:, i] = (torch.arange(N) >> i) % 2

    # Check which sequences are valid (XOR constraint satisfied in every block)
    valid = torch.ones(N, dtype=torch.bool)
    for block in range(T // 3):
        i = block * 3
        xor_ok = ((all_x[:, i] + all_x[:, i+1]) % 2 == all_x[:, i+2])
        valid &= xor_ok

    # Valid sequences have probability (1/4)^{T/3}, invalid have probability 0
    log_p = torch.full((N,), float('-inf'))
    log_p[valid] = -(T / 3) * torch.log(torch.tensor(4.0))
    return all_x, log_p
```

### 7.3 Exact expected loss

The expected negative log-likelihood under the true distribution is:

```
L_exact = Σ_x P_true(x) · [−log P_model(x)]
```

where the sum is over all 2^T sequences. In code:

```python
def exact_loss(model, T):
    """Compute the exact expected NLL under the true distribution."""
    all_x, true_log_p = true_log_probs(T)
    true_p = torch.exp(true_log_p)         # (2^T,) — zero for invalid sequences

    logits = model(all_x)                   # (2^T, T)

    # Per-position log P_model(x_t | x_{<t})
    log_p_model = -F.binary_cross_entropy_with_logits(
        logits, all_x, reduction='none'
    )  # (2^T, T)

    # Total log P_model(x) = sum over positions
    log_p_model_total = log_p_model.sum(dim=1)   # (2^T,)

    # Expected NLL: E_true[-log P_model(x)] per position
    nll = -(true_p * log_p_model_total).sum() / T
    return nll
```

### 7.4 Training with exact loss

```python
T = 9  # 3 blocks, 2^9 = 512 sequences — very fast
model = QuadraticAutoregressive(T)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

for step in range(2001):
    loss = exact_loss(model, T)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step:4d} | Exact loss: {loss.item():.6f}")
```

Expected output:

```
Step    0 | Exact loss: 0.693147
Step  500 | Exact loss: 0.491853
Step 1000 | Exact loss: 0.472786
Step 1500 | Exact loss: 0.467549
Step 2000 | Exact loss: 0.465349
```

The loss converges smoothly to the theoretical minimum — no sampling noise, just clean gradient descent.

### 7.5 Comparison: exact vs. SGD

Both methods converge to the same solution. The differences:

| Property           | SGD on samples        | Exact training          |
|--------------------|-----------------------|-------------------------|
| Loss per step      | Noisy (mini-batch)    | Exact                   |
| Gradient           | Stochastic estimate   | True gradient           |
| Cost per step      | O(B · T²)            | O(2^T · T²)            |
| Scalability        | Any T                 | T ≤ ~18                 |
| Final solution     | Same                  | Same                    |

For small T, exact training is a useful **debugging tool**: if your model can represent the distribution, exact training will find it without any sampling artifacts.

---

## 8. Results & Interpretation

### 8.1 What the trained model looks like

After training (either method), the parameters have a clean structure:

**Biases (b):**
- Positions 0, 1, 3, 4, 6, 7, ... (the "random" positions): b ≈ 0. The model predicts 50/50, which is correct — these bits are uniformly random.
- Positions 2, 5, 8, 11, ... (the XOR positions): b ≈ −L for some large L.

**Linear weights (W):**
- XOR position t depends on positions t−2 and t−1 with large positive weights: W[t, t−2] ≈ W[t, t−1] ≈ 2L.
- All other entries are near zero.

**Quadratic weights (C):**
- The only large entries are C[t, t−2, t−1] ≈ −4L at XOR positions.
- Everything else is near zero.

The model has discovered the **block-diagonal** structure of the problem entirely from data.

### 8.2 Verification checklist

To confirm everything works:

1. **Loss convergence**: Final loss ≈ 0.462 nats/position (= (2/3) ln 2). ✓

2. **XOR prediction accuracy**: Sample sequences and check that the model's prediction at XOR positions matches the true XOR:
   ```python
   x = sample_z2_data(1000, T)
   logits = model(x)
   preds = (logits[:, 2::3] > 0).float()        # predicted XOR bits
   targets = x[:, 2::3]                           # true XOR bits
   accuracy = (preds == targets).float().mean()
   print(f"XOR accuracy: {accuracy.item():.4f}")  # expect ~1.0
   ```

3. **Banded model**: Training with bandwidth=3 gives the same final loss. ✓

4. **Exact vs. SGD**: Both converge to the same parameter values (up to optimization noise). ✓

### 8.3 Takeaway

The tutorial demonstrates a key distinction in probabilistic modeling:

- **Where** you put interaction terms (in the energy vs. in the conditionals) determines the expressiveness of your model.
- A pairwise energy model has linear conditionals — insufficient for XOR.
- Putting quadratic features directly in the autoregressive conditionals gives the model the capacity to learn XOR.
