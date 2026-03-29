#%% Imports and setup
import sys, os
import numpy as np
from scipy.special import logsumexp, gammaln
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from hmm import Mess3Proc

hmm = Mess3Proc()
d_vocab = hmm.d_vocab  # 3
n_states = hmm.num_hidden_states  # 3
mu = np.ones(d_vocab) / d_vocab  # (1/3, 1/3, 1/3)

print(f"Mess3: {d_vocab} tokens, {n_states} hidden states")
print(f"Stationary token distribution: {mu}")

#%% Forward algorithm: compute log P(x) exactly
def log_forward(sequence, hmm):
    """
    Forward algorithm with log-sum-exp stability.
    Returns log P(x_1, ..., x_L) marginalizing over hidden states.
    """
    E = hmm.emission_matrices  # (d_vocab, n_states, n_states)
    # E[j, i, k] = P(observe j AND transition to k | currently in state i)
    pi = hmm.get_stationary_distribution()
    log_pi = np.log(pi)

    x0 = sequence[0]
    log_E = np.log(E[x0] + 1e-300)  # (n_states, n_states)
    # alpha_0(k) = sum_i pi(i) * E[x0, i, k]
    log_alpha = logsumexp(log_pi[:, None] + log_E, axis=0)

    for t in range(1, len(sequence)):
        log_E = np.log(E[sequence[t]] + 1e-300)
        # alpha_t(k) = sum_i alpha_{t-1}(i) * E[x_t, i, k]
        log_alpha = logsumexp(log_alpha[:, None] + log_E, axis=0)

    return logsumexp(log_alpha)

# Sanity check
_, obs = hmm.generate_sequence(100)
print(f"log P(x) for a test sequence of length 100: {log_forward(obs, hmm):.4f}")

#%% Generate data: N sequences of length L
L = 100
N = 50000
np.random.seed(42)
from tqdm import tqdm
log_probs = np.zeros(N)
freqs = np.zeros((N, d_vocab))

for i in tqdm(range(N)):
    _, obs = hmm.generate_sequence(L)
    log_probs[i] = log_forward(obs, hmm)
    for j in range(d_vocab):
        freqs[i, j] = np.sum(obs == j) / L

delta_f = freqs - mu  # (N, d_vocab)
counts = np.round(freqs * L).astype(int)  # integer
count vectors

print(f"Generated {N} sequences of length {L}")
print(f"Mean frequency: {freqs.mean(axis=0)}")
print(f"Std of frequency: {freqs.std(axis=0)}")
print(f"Mean log P / L = {log_probs.mean() / L:.6f}")

#%% Gaussianity of log P(x)
# By CLT for HMMs: log P(x) / L -> -h_obs, with Gaussian fluctuations:
#   sqrt(L) * (log P(x)/L + h_obs) -> N(0, sigma^2)
# So log P(x) ~ N(-L*h_obs, L*sigma^2)
from scipy import stats

h_obs_est = -log_probs.mean() / L  # estimate of observation entropy rate
sigma2_est = log_probs.var() / L    # estimate of asymptotic variance

# Standardize: z_i = (log P(x_i) - mean) / std
z_scores = (log_probs - log_probs.mean()) / log_probs.std()

skewness = stats.skew(z_scores)
kurtosis = stats.kurtosis(z_scores)  # excess kurtosis (0 for Gaussian)
_, shapiro_p = stats.shapiro(z_scores[:5000])  # Shapiro-Wilk (max 5000 samples)

print("=== Gaussianity of log P(x) ===")
print(f"  h_obs (estimated): {h_obs_est:.6f} nats/token")
print(f"  sigma^2 (estimated): {sigma2_est:.6f}")
print(f"  Skewness: {skewness:.4f} (0 for Gaussian)")
print(f"  Excess kurtosis: {kurtosis:.4f} (0 for Gaussian)")
print(f"  Shapiro-Wilk p-value: {shapiro_p:.4e} (>0.05 = consistent with Gaussian)")

#%% Estimate 1: empirical covariance -> Sigma_true
# CLT: Var(f) = Sigma_true / L, so Sigma_true = L * Sigma_emp
delta_f_2d = delta_f[:, :2]  # project to 2D (drop last, since sum = 0)

Sigma_emp = np.cov(delta_f_2d, rowvar=False)
Sigma_true_from_emp = L * Sigma_emp
Sigma_true_inv_from_emp = np.linalg.inv(Sigma_true_from_emp)

print("Estimate 1: from empirical covariance")
print(f"  Sigma_emp (should scale as 1/L):\n    {Sigma_emp}")
print(f"  Sigma_true = L * Sigma_emp (L-independent):\n    {Sigma_true_from_emp}")
print(f"  Sigma_true^{{-1}}:\n    {Sigma_true_inv_from_emp}")

#%% Group sequences by type (frequency vector)
type_groups = defaultdict(list)
for i in range(N):
    key = tuple(counts[i])
    type_groups[key].append(log_probs[i])

n_types = len(type_groups)
group_sizes = np.array([len(v) for v in type_groups.values()])

print(f"Number of distinct types: {n_types}")
print(f"  (expected ~ C(L+2,2) = {(L+1)*(L+2)//2})")
print(f"Samples per type: min={group_sizes.min()}, "
      f"median={np.median(group_sizes):.0f}, max={group_sizes.max()}")

#%% Estimate 2: quadratic fit to histogram in CLT-scaled coordinates
# P(type f) ~ count_f / N  (histogram estimator)
# Fit: log P(type f) ~ const - 1/2 z^T Sigma_true^{-1} z
# where z = sqrt(L) * delta_f is the CLT-scaled variable

type_z_2d = []     # CLT-scaled deviations for each type
log_p_type = []    # log(count / N) for each type
type_weights = []  # count (for weighted regression)

for key, group_log_probs in type_groups.items():
    f_vec = np.array(key) / L
    delta = f_vec[:2] - mu[:2]
    z = np.sqrt(L) * delta
    type_z_2d.append(z)
    log_p_type.append(np.log(len(group_log_probs) / N))
    type_weights.append(len(group_log_probs))

type_z_2d = np.array(type_z_2d)
log_p_type = np.array(log_p_type)
type_weights = np.array(type_weights)

# Weighted quadratic regression: features [1, z0, z1, z0^2, z0*z1, z1^2]
poly = PolynomialFeatures(degree=2, include_bias=True)
X = poly.fit_transform(type_z_2d)

reg = LinearRegression(fit_intercept=False)
reg.fit(X, log_p_type, sample_weight=type_weights)
y_pred = reg.predict(X)

# R^2 (weighted)
ss_res = np.sum(type_weights * (log_p_type - y_pred) ** 2)
ss_tot = np.sum(type_weights * (log_p_type - np.average(log_p_type, weights=type_weights)) ** 2)
r2 = 1 - ss_res / ss_tot

# Extract Hessian J_z and hence Sigma_true^{-1}_fit = -J_z
fn = poly.get_feature_names_out()
ct = reg.coef_
im = {name: i for i, name in enumerate(fn)}

J_z = np.array([
    [2 * ct[im['x0^2']], ct[im['x0 x1']]],
    [ct[im['x0 x1']], 2 * ct[im['x1^2']]]
])
Sigma_true_inv_from_fit = -J_z

print("Estimate 2: from quadratic fit to log P(type)")
print(f"  R^2 = {r2:.4f}")
print(f"  Hessian J_z:\n    {J_z}")
print(f"  Sigma_true^{{-1}} = -J_z:\n    {Sigma_true_inv_from_fit}")

#%% Compare the two estimates
ratio = Sigma_true_inv_from_fit / Sigma_true_inv_from_emp
rel_frob = np.linalg.norm(Sigma_true_inv_from_fit - Sigma_true_inv_from_emp) / \
           np.linalg.norm(Sigma_true_inv_from_emp)

print("=== Comparison ===")
print(f"Sigma_true^{{-1}} from fit:\n  {Sigma_true_inv_from_fit}")
print(f"Sigma_true^{{-1}} from L*Sigma_emp:\n  {Sigma_true_inv_from_emp}")
print(f"Elementwise ratio (should -> 1):\n  {ratio}")
print(f"Relative Frobenius error: {rel_frob:.4f}")
