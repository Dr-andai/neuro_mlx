"""
Bayesian MLP — Multi-Class Classification
==========================================

CORE IDEA:
  A standard MLP learns one fixed weight per connection.
  A Bayesian MLP learns a *distribution* over each weight: N(mu, sigma).

  At inference time we sample from that distribution. The spread (sigma) is
  the model telling us "I'm unsure about this weight." Wide sigma = high
  uncertainty. We train it to be as certain as the data justifies.

OBJECTIVE — ELBO (Evidence Lower Bound):
  We can't compute the exact posterior p(w|data), so we approximate it with
  a tractable Gaussian q(w|mu, sigma) and maximise:

    ELBO = E_q[log p(y|x,w)]  -  KL(q(w|mu,sigma) || p(w))
           ^^^ data fit term        ^^^ complexity penalty

  Minimising -ELBO gives:
    loss = cross_entropy(logits, y)  +  (1/M) * KL

  The 1/M scaling is crucial — without it the KL term (which grows with
  parameter count) drowns the data term and nothing is learned.
  M = number of mini-batches per epoch.
"""

import math
import mlx.core as mx
import mlx.nn as nn


# ===========================================================================
# Bayesian Linear Layer
# ===========================================================================

class BayesianLinear(nn.Module):
    """
    Linear layer where weights and biases are distributions, not scalars.

    Each weight w has:
      w_mu  — the centre of its bell curve (the "best guess")
      w_rho — an unconstrained raw value controlling the spread

    We convert rho → sigma (std dev) via Softplus:
      sigma = log(1 + exp(rho))

    WHY Softplus?  Sigma must be positive. If we trained sigma directly its
    gradient could push it negative. Rho can be any real number, and
    softplus maps it to (0, +inf), so gradients always keep sigma valid.

    WHY rho_init = -3?  softplus(-3) ≈ 0.05 — a narrow, confident starting
    distribution. The network widens weights only if the data warrants it.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        # Kaiming-style uniform bounds — keeps activations in a healthy range
        # at initialisation for networks of arbitrary depth.
        bound = 1.0 / math.sqrt(input_dim)

        # mx.random.uniform already returns mx.array — no extra wrapping needed
        self.w_mu = mx.random.uniform(-bound, bound, (input_dim, output_dim))
        self.b_mu = mx.random.uniform(-bound, bound, (output_dim,))

        # Note the trailing comma in (output_dim,) — without it Python reads
        # it as a plain integer, not a 1-element shape tuple.
        self.w_rho = mx.full((input_dim, output_dim), -3.0)
        self.b_rho = mx.full((output_dim,), -3.0)

        # Populated during __call__; read by BayesianClassificationMLP.total_kl()
        self.current_kl = mx.array(0.0)

    def __call__(self, x):
        # --- rho → sigma (always positive) ---
        w_sigma = mx.log(1.0 + mx.exp(self.w_rho))
        b_sigma = mx.log(1.0 + mx.exp(self.b_rho))

        # --- Reparameterization Trick ---
        # Problem: sampling w ~ N(mu, sigma) is a stochastic node — no gradient flows through it.
        # Solution: rewrite the sample as   w = mu + sigma * eps,  eps ~ N(0,1)
        #   Now mu and sigma are deterministic nodes that gradients CAN flow through.
        #   eps is just a constant noise vector drawn before the forward pass.
        # This trick is what makes end-to-end training of Bayesian nets possible.
        epsilon_w = mx.random.normal(self.w_mu.shape)
        epsilon_b = mx.random.normal(self.b_mu.shape)
        sampled_weight = self.w_mu + w_sigma * epsilon_w
        sampled_bias   = self.b_mu + b_sigma * epsilon_b

        # --- Cache this layer's KL divergence ---
        # Using the analytical (closed-form) formula rather than sampling
        # gives an exact, low-variance KL estimate — much more stable training.
        self.current_kl = self._kl_loss(self.w_mu, w_sigma) + self._kl_loss(self.b_mu, b_sigma)

        return mx.matmul(x, sampled_weight) + sampled_bias

    def _kl_loss(self, mu, sigma):
        """
        Analytical KL divergence:  KL( N(mu, sigma) || N(0, 1) )

        Derivation collapses to:
          0.5 * sum( sigma^2 + mu^2 - 1 - log(sigma^2) )
                     ^^^^^^^^   ^^^^   ^   ^^^^^^^^^^^^^
                     variance  bias  const  entropy term

        Properties:
          - Always >= 0  (KL is non-negative by definition)
          - Equals 0 only when mu=0 AND sigma=1 (learned dist == prior)
          - mx.log(mx.square(sigma)) == 2 * mx.log(sigma)  — same thing, avoids
            potential domain issues if sigma were ever exactly 0
        """
        return 0.5 * mx.sum(
            mx.square(sigma) + mx.square(mu) - 1.0 - mx.log(mx.square(sigma))
        )


# ===========================================================================
# Full Bayesian MLP
# ===========================================================================

class BayesianClassificationMLP(nn.Module):
    """
    Two-layer Bayesian MLP.

    KEY BEHAVIOUR: every forward pass samples a fresh set of weights.
    The same input will produce slightly different logits on each call.
    This variance IS the model uncertainty — use it during inference.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.layer1 = BayesianLinear(in_dim, hidden_dim)
        self.layer2 = BayesianLinear(hidden_dim, num_classes)

    def __call__(self, x):
        x = mx.maximum(self.layer1(x), 0.0)   # ReLU activation
        return self.layer2(x)                  # raw logits — no softmax here

    def total_kl(self):
        """
        Sum KL from every Bayesian layer.
        MUST be called after a forward pass so current_kl is populated.
        """
        return self.layer1.current_kl + self.layer2.current_kl


# ===========================================================================
# ELBO Loss  (imported and called from training.py)
# ===========================================================================

def elbo_loss(model, x_batch, y_batch, num_batches: int):
    """
    Negative ELBO for one mini-batch.

    loss = NLL  +  KL / M

    NLL  — cross-entropy between predicted logits and true labels.
           Pushes the sampled weights to make correct predictions.

    KL/M — regularisation that pulls weight distributions back to the prior.
           Divided by M (batches per epoch) so one epoch sums to KL/1,
           matching the standard ELBO derivation over the full dataset.

    Without the 1/M scaling: KL (O(params)) >> NLL (O(batch)) → model ignores data.
    """
    logits = model(x_batch)

    # cross_entropy expects raw logits (applies log-softmax internally)
    nll = mx.mean(nn.losses.cross_entropy(logits, y_batch))

    # KL is populated inside __call__ above — call total_kl() after forward pass
    kl = model.total_kl()

    return nll + kl / num_batches
