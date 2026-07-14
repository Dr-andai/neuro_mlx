"""
Single Bayesian layer

w_mu: where the weight's bell curve centres
w_rho: control knob for how wide the bell curve is.
We use rho because std isn't a single number.
allows us to use a mathematical trick called Softplus 
to ensure the width stays positive
"""

import mlx.core as mx
import mlx.nn as nn
# nn.Module needs its own initialization to set up internal structures 
# (parameter tracking, training/eval modes, etc.)

# define a single bayesian layer
class BayesianLayer(nn.Module):
    def __init__(self): 
        super().__init__() # initialize weights

        # # In MLX, trainable parameters MUST be wrapped in mx.array 
        # inside the constructor so the framework tracks them as state. 
        self.w_mu = mx.array([0.0])
        self.w_rho = mx.array([-3.0])

        self.b_mu = mx.array([0.0])
        self.b_rho = mx.array([-3.0])

    def __call__(self, x):
        w_sigma = mx.log(1.0 + mx.exp(self.w_rho))
        b_sigma = mx.log(1.0 + mx.exp(self.b_rho))

        epsilon_w = mx.random.normal(self.w_mu.shape)
        epsilon_b = mx.random.normal(self.b_mu.shape)
        
        # reparameterization trick
        # Shift and stretch the standard bell curve
        # to match our current weight's mean and width
        sampled_weight = self.w_mu + w_sigma * epsilon_w
        sampled_bias = self.b_mu + b_sigma * epsilon_b

        return (x * sampled_weight) + sampled_bias

# implementing ELBO
def log_gaussian_prior(x, mu=0.0, sigma=1.0):
    return -0.5 * mx.log(2 * mx.pi) - mx.log(sigma) - ((x - mu)**2) / (2 * sigma **2)

# ERROR: def beginner_elbo_loss(model_params, X_batch, Y_batch, model) was wrong.
# nn.value_and_grad calls fn(*args) — it does NOT prepend model_params.
# It calls model.update(params) internally before fn runs, so model.w_mu
# etc. are already differentiable when fn accesses them. Adding model_params
# shifted all args by one, leaving 'model' missing at the call site.
def beginner_elbo_loss(X_batch, Y_batch, model):
    # sample the current layer characteristics
    w_sigma = mx.log(1.0 + mx.exp(model.w_rho))
    b_sigma = mx.log(1.0 + mx.exp(model.b_rho))

    eps_w = mx.random.normal(model.w_mu.shape)
    eps_b = mx.random.normal(model.b_mu.shape)
    sampled_w = model.w_mu + w_sigma * eps_w
    sampled_b = model.b_mu + b_sigma * eps_b

    # complexity penalty (kl approximation)
    log_q_w = log_gaussian_prior(sampled_w, model.w_mu, w_sigma)
    log_p_w = log_gaussian_prior(sampled_w, 0.0, 1.0)

    log_q_b = log_gaussian_prior(sampled_b, model.b_mu, b_sigma)
    log_p_b = log_gaussian_prior(sampled_b, 0.0, 1.0) # FIX 2: Fixed duplicate log_q_b bug here

    complexity_penalty = mx.sum(log_q_w - log_p_w) + mx.sum(log_q_b - log_p_b)

    # accuracy goal: calculate prediction using our current sample
    predictions = (X_batch * sampled_w) + sampled_b
    # how far are we from the real scores
    prediction_error = mx.mean((predictions - Y_batch) ** 2)

    return (complexity_penalty / 100) + prediction_error
