# Motivation

Modern language models (GPT, Claude, etc.) are trained on a deceptively simple task: given a sequence of tokens (words, sub-words), predict the next one. Despite the simplicity of this objective, the trained models develop rich internal representations — grammar, factual knowledge, reasoning patterns — none of which were explicitly programmed. A natural question, both for machine learning and for physics, is: **what computational structure does the model build internally in order to make good predictions?**

To make this question precise and tractable, we replace natural language with a **synthetic stochastic process** whose statistics we control exactly: a Hidden Markov Model (HMM).

## Hidden Markov Models in one paragraph

An HMM is a two-layer stochastic process. A hidden (unobserved) variable $Z_t$ evolves as a Markov chain on a finite state space of size $N$, with transition matrix $A_{ij} = P(Z_{t+1}=j \mid Z_t=i)$. At each time step, the current hidden state emits an observable token $X_t$ from a vocabulary of size $K$, with probabilities $B_{i\mu} = P(X_t = \mu \mid Z_t = i)$. The observer sees only the token sequence $X_1, X_2, \ldots$ — not the hidden states. Crucially, although the hidden states are Markovian, the observed token sequence is *not*: the marginal process $\{X_t\}$ has long-range correlations whose structure encodes the latent dynamics.

## Transformers in one paragraph

A transformer is a neural network architecture that processes sequences in parallel using a mechanism called *attention*: at each position, the model computes a weighted combination of information from all preceding positions, where the weights are learned functions of the data. A stack of such attention layers, optionally interleaved with nonlinear feedforward layers (MLPs), maps the input sequence to a predicted probability distribution over the next token. The model is trained by minimizing the cross-entropy loss between its predictions and the actual next tokens in a large training corpus.

## Why this combination?

For an HMM, the optimal prediction strategy is known exactly: it is the forward algorithm, which recursively updates a *belief state* — a probability distribution over the $N$ hidden states given all past observations. This belief state is a sufficient statistic for next-token prediction. The entropy rate of the process sets an absolute lower bound on the prediction loss achievable by any model, and the Bayes-optimal loss at finite context length provides a tighter bound for models (like transformers) that can only look back a fixed number of steps.

This setup lets us ask sharp questions:

1. **How close can a transformer get to the information-theoretic optimum?** By comparing the trained model's loss against the entropy rate and the Bayes-optimal loss, we can cleanly decompose the gap into a *finite-context cost* (inherent to the architecture's limited memory) and a *capacity gap* (how far the model's learned strategy is from optimal given its context window).

2. **What internal representations does the model learn?** If the forward algorithm is the optimal computation, does the transformer discover something analogous to belief states? We can inspect the model's learned embeddings and intermediate activations to look for signatures of the latent HMM structure.

3. **How do architecture choices matter?** By sweeping over model size, depth, attention heads, and the presence/absence of MLP layers, we map out which architectural features are necessary for good performance on this inference task.

## Key findings

In this note, we study these questions on a specific HMM whose hidden states are arranged on a *cylinder graph* — a structured topology with three depth levels and six angular positions per level, giving 18 hidden states and a 48-token vocabulary. The key findings are:

- The best transformer (206k parameters) reaches within ~0.004 nats of the Bayes-optimal predictor at its context length, meaning it has essentially learned to perform near-optimal Bayesian filtering.
- Transformers consistently outperform $n$-gram models (which memorize token co-occurrence statistics), suggesting they learn something qualitatively beyond counting.
- The learned token embeddings spontaneously discover the depth-level structure of the HMM — tokens from the same hidden-state cluster are mapped to similar directions in embedding space — without any supervision about this structure.
